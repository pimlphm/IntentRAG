"""Build new source-derived questions; retain every accepted/rejected attempt.

Labels are exact normalized source quotes, not maintenance authorization labels.
All selected source pages are absent from the complete legacy gold-page set.
No retrieval outcome is used to select, reject, or regenerate a question.
"""
from pathlib import Path
import argparse, hashlib, json, re, time, urllib.request
import numpy as np
import pandas as pd

ROOT = Path(r'${INTENTRAG_PROJECT_ROOT}')
OUT = Path(__file__).resolve().parents[1]
SEED = 20261014

def norm(x):
    return re.sub(r'\s+', ' ', str(x)).strip()

def digest(x):
    return hashlib.sha256(norm(x).encode()).hexdigest()

def ask(prompt, model, max_tokens=600):
    schema={'type':'object','properties':{'items':{'type':'array','items':{'type':'object',
        'properties':{'question':{'type':'string'},'answers':{'type':'array','items':{'type':'object',
        'properties':{'source_label':{'type':'string','enum':['S1','S2']},'quote':{'type':'string'}},
        'required':['source_label','quote']}}},'required':['question','answers']}}},'required':['items']}
    payload = dict(model=model, prompt=prompt, stream=False, format=schema,
                   keep_alive='30m', options=dict(temperature=0, seed=SEED,
                   num_ctx=4096, num_predict=max_tokens))
    request = urllib.request.Request('http://localhost:11434/api/generate',
                                    data=json.dumps(payload).encode(),
                                    headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(request, timeout=180) as f:
        data = json.load(f)
    return data

def source_records(n):
    ev = pd.read_parquet(ROOT/'01_processed_corpus/v5/evidence_objects.parquet').fillna('')
    legacy = pd.read_csv(ROOT/'03_benchmark/benchmark_v21_evidence_sufficient_locked/qa_warehouse_v21_full.csv').fillna('')
    legacy_pages = set(legacy.gold_page_id.astype(str))
    ev['text_hash'] = ev.text.map(digest)
    legacy_hashes = set(ev.loc[ev.evidence_id.isin(set(legacy.gold_evidence_id)), 'text_hash'])
    quality = ev.text.str.len().between(160,1800) & ~ev.page_id.isin(legacy_pages) & ~ev.text_hash.isin(legacy_hashes)
    quality &= ev.is_boilerplate.eq(0) & ev.is_index_or_toc.eq(0)
    quality &= ~ev.text.str.contains(r'TABLE OF CONTENTS|LIST OF EFFECTIVE|\.{3,}\s*\.{3,}|INDEX TO',case=False,regex=True)
    quality &= ev.components.astype(str).str.len().gt(0)
    quality &= ev.text.str.contains(r'if|when|check|inspect|replace|warning|caution|limit|pressure|voltage|条件|故障|检查|原因|警告|建议',case=False)
    candidates = ev[quality].drop_duplicates('text_hash').copy()
    candidates['rng'] = np.random.default_rng(SEED).random(len(candidates))
    candidates['prefer'] = candidates.granularity.isin(['paragraph','table_row']).astype(int)
    candidates = candidates.sort_values(['prefer','rng'],ascending=[False,True]).drop_duplicates('page_id')
    grouped=[]
    n_boeing=min(32,n//4,int(candidates.aircraft_model.eq('Boeing 737').sum()))
    allocation = {'Boeing 737': n_boeing, 'Cessna 172': n-n_boeing}
    for aircraft, count in allocation.items():
        sub=candidates[candidates.aircraft_model.eq(aircraft)].head(count)
        for _, a in sub.iterrows():
            # Second source has a distinct page/text identity within the same section.
            others=ev[ev.aircraft_model.eq(aircraft) & ev.document_id.eq(a.document_id)
                      & ev.detected_section_id.eq(a.detected_section_id)
                      & ~ev.page_id.isin(legacy_pages) & ~ev.text_hash.isin(legacy_hashes)
                      & ~ev.text_hash.eq(a.text_hash) & ev.text.str.len().between(140,1600)
                      & ev.is_boilerplate.eq(0) & ev.is_index_or_toc.eq(0)].copy()
            others['dist']=(pd.to_numeric(others.page_number,errors='coerce')-float(a.page_number)).abs()
            others=others[others.dist.le(2)].sort_values(['dist','page_number'])
            b=others.iloc[0] if len(others) else None
            grouped.append({'source_index':len(grouped),'aircraft':aircraft,
                'sources':[{'label':f'S{i+1}','evidence_id':r.evidence_id,'page_id':r.page_id,
                            'document_id':r.document_id,'section_id':r.detected_section_id,
                            'page_number':r.page_number,'text':norm(r.text),'text_hash':r.text_hash}
                           for i,r in enumerate([a]+([b] if b is not None else []))]})
    return grouped, dict(legacy_pages=len(legacy_pages), candidates=int(len(candidates)),
                        aircraft_candidate_counts=candidates.groupby('aircraft_model').size().to_dict(),
                        selected_source_groups=len(grouped), requested=n)

def validate(item, sources):
    q=norm(item.get('question',''))
    if not 16 <= len(q) <= 380:
        return None,'question_length'
    if re.search(r'EVID_|S[12]\]|quote|excerpt|provided text|source [12]',q,re.I):
        return None,'source_label_or_prompt_reference'
    answers=item.get('answers',[])
    if not answers:
        return None,'empty_answer'
    available={s['label']:s for s in sources}
    support=[]
    for a in answers:
        s=available.get(a.get('source_label'))
        quote=norm(a.get('quote','')).strip('"“”')
        if s is None or len(quote)<16 or len(quote)>650 or quote not in s['text']:
            return None,'non_verbatim_answer'
        # Reject a question that contains a long copied answer string.
        if quote.lower() in q.lower():
            return None,'answer_in_question'
        support.append(dict(evidence_id=s['evidence_id'],page_id=s['page_id'],
                            document_id=s['document_id'],section_id=s['section_id'],quote=quote))
    if len(set(s['evidence_id'] for s in support)) < len(support):
        return None,'duplicate_support'
    return dict(question=q,answers=support,kind='multi_source' if len(support)>1 else 'single_source'),'accepted'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--sources',type=int,default=128)
    ap.add_argument('--model',default='qwen2.5:3b');ap.add_argument('--limit',type=int,default=0)
    ap.add_argument('--extend-to',type=int,default=0)
    args=ap.parse_args()
    d=OUT/'data';d.mkdir(parents=True,exist_ok=True)
    source_file=d/'fresh_source_selection.json'
    if source_file.exists() and json.loads(source_file.read_text(encoding='utf-8')):
        groups=json.loads(source_file.read_text(encoding='utf-8'))
    else:
        groups,stats=source_records(args.sources)
        source_file.write_text(json.dumps(groups,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
        (d/'fresh_source_selection_stats.json').write_text(json.dumps(stats,indent=2),encoding='utf-8')
    if args.extend_to>len(groups):
        more,stats=source_records(args.extend_to)
        present={g['sources'][0]['evidence_id'] for g in groups}
        for g in more:
            if g['sources'][0]['evidence_id'] in present:continue
            g['source_index']=len(groups);groups.append(g);present.add(g['sources'][0]['evidence_id'])
        source_file.write_text(json.dumps(groups,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
        stats['selected_source_groups']=len(groups)
        (d/'fresh_source_selection_stats.json').write_text(json.dumps(stats,indent=2),encoding='utf-8')
    attempts_file=d/'fresh_generation_attempts.jsonl'
    previous=[json.loads(s) for s in attempts_file.read_text(encoding='utf-8').splitlines()] if attempts_file.exists() else []
    done={r['source_index'] for r in previous}
    target=groups[:args.limit] if args.limit else groups
    for g in target:
        if g['source_index'] in done:continue
        sources=g['sources']
        text='\n\n'.join('['+s['label']+'] '+s['text'] for s in sources)
        prompt='''You create an evaluation question bank from real aviation source text.
Return only JSON with key items, a list of two objects. Each object has question
and answers (a list of objects with source_label and quote).
Write one concise specific question about the component, condition, or procedure
in S1. If S2 exists, make the second question require complementary facts from
both S1 and S2; otherwise make a different question about S1. Name the relevant
component and operational condition in the question; avoid generic questions.
Do not copy a long answer phrase into the question. Do not mention source labels,
passage, supplied text, question bank, or the task instructions in questions.
Each answer quote must be a short EXACT VERBATIM CONTIGUOUS substring from its
source, including punctuation and spacing. Preserve the source language.
Do not invent procedures or infer authorizations. If the text is unusable,
return {"items":[]}.
Aircraft: '''+g['aircraft']+'\nSources:\n'+text
        started=time.perf_counter()
        record=dict(source_index=g['source_index'],model=args.model,sources=[s['evidence_id'] for s in sources])
        try:
            raw=ask(prompt,args.model)
            content=json.loads(raw['response'])
            raw.pop('context',None)
            validations=[]
            for item in content.get('items',[])[:2]:
                valid,reason=validate(item,sources)
                validations.append(dict(raw_item=item,accepted=valid,reason=reason))
            record.update(raw_response=raw,validation=validations,elapsed_s=time.perf_counter()-started)
        except Exception as exc:
            record.update(error=repr(exc),validation=[],elapsed_s=time.perf_counter()-started)
        with attempts_file.open('a',encoding='utf-8') as f:
            f.write(json.dumps(record,ensure_ascii=False,default=str)+'\n')
        print(f"SOURCE {g['source_index']+1}/{len(groups)} accepted={sum(v['accepted'] is not None for v in record['validation'])} seconds={record['elapsed_s']:.1f}",flush=True)
    attempts=[json.loads(s) for s in attempts_file.read_text(encoding='utf-8').splitlines()] if attempts_file.exists() else []
    accepted=[]; seen=set()
    for r in attempts:
        g=groups[r['source_index']]
        for v in r.get('validation',[]):
            item=v.get('accepted')
            if item is None:continue
            key=g['aircraft']+'|'+norm(item['question']).lower()
            if key in seen:continue
            seen.add(key)
            accepted.append(dict(question_id=f'FRESH_{len(accepted):04d}',aircraft_model=g['aircraft'],
                question=item['question'],split='fresh_locked_test',synthetic_generated=True,
                query_kind=item['kind'],source_group=g['source_index'],
                gold_evidence_ids=';'.join(a['evidence_id'] for a in item['answers']),
                gold_page_ids=';'.join(sorted(set(a['page_id'] for a in item['answers']))),
                gold_section_ids=';'.join(sorted(set(a['section_id'] for a in item['answers']))),
                answer_quotes_json=json.dumps(item['answers'],ensure_ascii=False),
                gold_answer_brief='\n'.join(a['quote'] for a in item['answers']),
                generation_model=r['model'],source_page_disjoint_from_legacy=True))
    pd.DataFrame(accepted).to_csv(d/'fresh_questions.csv',index=False,encoding='utf-8-sig')
    print(f'FRESH ACCEPTED={len(accepted)} ATTEMPTED_GROUPS={len(attempts)}',flush=True)

if __name__=='__main__':main()
