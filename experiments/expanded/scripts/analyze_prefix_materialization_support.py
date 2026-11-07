"""Post-hoc source materialization audit; no retrieval or generation changes."""
from pathlib import Path
import hashlib
import json
import re
import unicodedata
import numpy as np
import pandas as pd
from tokenizers import Tokenizer
from build_kv_prompts import ROOT

BASE = Path(__file__).resolve().parents[1]
OUT = BASE/'results/real_kv'
LAYOUTS = ['query_first','canonical_state_first','hierarchical_state_first']
HEADER = re.compile(r'\[S(\d+)\] id=([^;\n]+); document=([^;\n]+); page=([^;\n]+); role=([^\n]+)\n')
UNITS = re.compile(r'[A-Za-z0-9]+|[\u4e00-\u9fff]')


def sha(s): return hashlib.sha256(s.encode('utf-8')).hexdigest()
def norm(s): return ''.join(c for c in unicodedata.normalize('NFKC',s).casefold() if c.isalnum())


def anchors(gold):
    units = UNITS.findall(unicodedata.normalize('NFKC',gold))
    chinese = sum(bool(re.fullmatch(r'[\u4e00-\u9fff]',u)) for u in units) >= 12
    size = 24 if chinese else 6
    groups = [units[n:n+size] for n in range(0,len(units),size)]
    eligible = [group for group in groups if len(group)==size and
                (not chinese or sum(bool(re.fullmatch(r'[\u4e00-\u9fff]',u)) for u in group)>=12)]
    return [norm(''.join(group)) for group in eligible], ('24 units, >=12 CJK' if chinese else '6 English/alphanumeric words'), len(units)


def source_region(prompt):
    body = prompt.split('Source excerpts:\n',1)[1]
    body = body.split('\nQuestion: ',1)[0]
    return body.split('<|im_end|>',1)[0]


def hit(ids, qrels):
    return float(bool(set(ids)&set(qrels))) if qrels else np.nan


def anchor_coverage(reference, bodies):
    return sum(any(a in b for b in bodies) for a in reference)/len(reference) if reference else np.nan


def main():
    protocol_path=BASE/'reports/prefix_materialization_support_protocol.json'
    protocol=json.loads(protocol_path.read_text(encoding='utf-8'))
    assert protocol['protocol_frozen_before_metric_computation']
    canonical={q['question_id']:q for q in json.loads((BASE/'data/canonical_questions.json').read_text(encoding='utf-8'))}
    prompts=[json.loads(l) for l in (BASE/'data/real_kv_prompts.jsonl').read_text(encoding='utf-8').splitlines()]
    runs=pd.read_csv(OUT/'kv_pairs.csv').set_index(['query_id','method','namespace'])
    retrieval=[json.loads(l) for l in (BASE/'results/all_retrieval_runs.jsonl').read_text(encoding='utf-8').splitlines()]
    top10={(r['question_id'],r['method']):r['top_evidence_ids'].split(';') for r in retrieval}
    corpus=pd.read_parquet(ROOT/'01_processed_corpus/v5/evidence_objects.parquet').fillna('').set_index('evidence_id').to_dict('index')
    tokenizer=Tokenizer.from_file(str(BASE/'models/qwen2_5_3b_tokenizer/tokenizer.json'))
    tokcache={}
    def toklen(s):
        key=sha(s)
        if key not in tokcache:tokcache[key]=len(tokenizer.encode(s,add_special_tokens=False).ids)
        return tokcache[key]
    rows, audits=[],[]
    for p in prompts:
        qid,method,layout=p['query_id'],p['method'],p['namespace']
        q=canonical[qid];run=runs.loc[qid,method,layout]
        assert sha(p['prompt'])==run.prompt_sha256
        body=source_region(p['prompt']);matches=list(HEADER.finditer(body));blocks=[]
        # A budget-truncated next header is a delimiter, not preceding source text.
        # Original source text is whitespace-flattened by the builder; a new-line
        # [S... (or the one-character final '[') can only be an added header.
        header_starts=[m.start() for m in re.finditer(r'(?m)^\[(?:S|$)',body)]
        ends=[next((start for start in header_starts if start>=m.end()),len(body)) for m in matches]
        for i,m in enumerate(matches):
            end=ends[i]
            text=body[m.end():end].strip()
            eid=m.group(2);r=corpus[eid]
            full=re.sub(r'\s+',' ',r['text']).strip()
            nt,nf=norm(text),norm(full)
            blocks.append({'source_label':'S'+m.group(1),'evidence_id':eid,'document_id':m.group(3),
                'page':m.group(4),'role':m.group(5),'granularity':r['granularity'],
                'component_family':r['component_family'],'source_text_sha256':sha(text),
                'original_text_sha256':sha(full),'body_characters':len(text),'original_body_characters':len(full),
                'body_tokens_budget_tokenizer':toklen(text),'original_body_tokens_budget_tokenizer':toklen(full),
                'has_source_body':bool(text),'normalized_is_source_prefix':nf.startswith(nt) if nt else True,
                'normalized_complete_source':nt==nf,'source_truncated':nf.startswith(nt) and nt!=nf,
                'normalized_character_retention':len(nt)/len(nf) if nf else np.nan})
        ids=[b['evidence_id'] for b in blocks];bodies=[norm(body[m.end():ends[i]].strip()) for i,m in enumerate(matches)]
        retrieved=top10[qid,method]
        full_bodies=[norm(re.sub(r'\s+',' ',corpus[e]['text']).strip()) for e in retrieved]
        gold=q.get('gold_answer_brief','') or '';ref,anchor_kind,nunits=anchors(gold)
        whole=norm(gold)
        whole_defined=bool(ref and whole)
        token_mass=sum(b['body_tokens_budget_tokenizer'] for b in blocks)
        coarse={'page','section','component_profile'}
        row={'query_id':qid,'seed_query_id':p['seed_query_id'],'pair_index':p['pair_index'],'aircraft_model':p['aircraft_model'],
            'topic':p['topic'],'method':method,'namespace':layout,'cluster_id':q['cluster_id'],'source_group':q['source_group'],
            'target_prompt_sha256':run.prompt_sha256,'materialized_source_region_sha256':sha(body),
            'retrieved_source_ids':';'.join(retrieved),'emitted_source_ids':';'.join(ids),'emitted_body_nonempty_ids':';'.join(b['evidence_id'] for b in blocks if b['has_source_body']),
            'accepted_ids':';'.join(q['accepted_ids']),'exact_ids':';'.join(q['exact_ids']),
            'top10_accepted_hit_proxy':hit(retrieved,q['accepted_ids']),'top10_exact_hit_proxy':hit(retrieved,q['exact_ids']),
            'emitted_id_accepted_hit_proxy':hit(ids,q['accepted_ids']),'emitted_id_exact_hit_proxy':hit(ids,q['exact_ids']),
            'emitted_body_nonempty_accepted_hit_proxy':hit([b['evidence_id'] for b in blocks if b['has_source_body']],q['accepted_ids']),
            'n_retrieved_sources':len(retrieved),'n_emitted_blocks':len(blocks),'n_nonempty_source_bodies':sum(b['has_source_body'] for b in blocks),
            'n_truncated_blocks':sum(b['source_truncated'] for b in blocks),'n_complete_blocks':sum(b['normalized_complete_source'] for b in blocks),
            'n_source_prefix_mismatches':sum(not b['normalized_is_source_prefix'] for b in blocks),
            'partial_source_headers':max(0,len(header_starts)-len(matches)),
            'source_body_tokens_budget_tokenizer':token_mass,
            'profile_body_token_fraction':sum(b['body_tokens_budget_tokenizer'] for b in blocks if b['granularity']=='component_profile')/token_mass if token_mass else np.nan,
            'coarse_body_token_fraction':sum(b['body_tokens_budget_tokenizer'] for b in blocks if b['granularity'] in coarse)/token_mass if token_mass else np.nan,
            'role_body_token_composition':json.dumps({role:sum(b['body_tokens_budget_tokenizer'] for b in blocks if b['role']==role)/token_mass for role in sorted({b['role'] for b in blocks})},ensure_ascii=False) if token_mass else '{}',
            'gold_brief_sha256':sha(gold),'gold_brief_missing':not bool(gold),'gold_reference_units':nunits,
            'anchor_rule':anchor_kind,'eligible_gold_anchors':len(ref),
            'top10_full_source_reference_anchor_coverage_proxy':anchor_coverage(ref,full_bodies),
            'emitted_reference_anchor_coverage_proxy':anchor_coverage(ref,bodies),
            'whole_gold_brief_visible_proxy':float(any(whole in b for b in bodies)) if whole_defined else np.nan,
            'native_prompt_tokens':int(run.prompt_token_count_native),'actual_cached_input_tokens':int(run.warm_timing_cache_n),
            'actual_cache_fraction':float(run.warm_timing_cache_n/run.prompt_token_count_native)}
        row['accepted_id_hit_lost_at_materialization_proxy']=row['top10_accepted_hit_proxy']-row['emitted_id_accepted_hit_proxy']
        row['reference_anchor_coverage_lost_at_materialization_proxy']=row['top10_full_source_reference_anchor_coverage_proxy']-row['emitted_reference_anchor_coverage_proxy']
        rows.append(row)
        audits.append({'profile':{k:row[k] for k in ['query_id','seed_query_id','pair_index','method','namespace','target_prompt_sha256']},
                       'blocks':blocks,'gold_anchor_sha256':[sha(a) for a in ref],'source_region_sha256':sha(body)})
    d=pd.DataFrame(rows)
    assert len(d)==594 and not d.duplicated(['query_id','method','namespace']).any()
    assert d.pair_index.nunique()==6 and d.method.nunique()==33
    assert d.actual_cache_fraction.between(0,1).all()
    if not d.n_source_prefix_mismatches.eq(0).all():
        print('PREFIX_MISMATCH_PROFILES',d.loc[d.n_source_prefix_mismatches.gt(0),['query_id','method','namespace','n_source_prefix_mismatches']].head(10).to_string(),flush=True)
        for audit in audits:
            for block in audit['blocks']:
                if not block['normalized_is_source_prefix']:
                    print('MISMATCH_BLOCK',audit['profile'],block,flush=True)
                    break
    assert d.n_source_prefix_mismatches.eq(0).all(), 'Unexpected source body mismatches require review'
    # Identical source materialization for canonical and hierarchy is checked, not assumed.
    same=d.pivot(index=['query_id','method'],columns='namespace',values='materialized_source_region_sha256')
    body_same=same.canonical_state_first.eq(same.hierarchical_state_first)
    d.to_csv(OUT/'prefix_materialization_support.csv',index=False,encoding='utf-8-sig')
    (OUT/'prefix_materialization_source_blocks.jsonl').write_text('\n'.join(json.dumps(a,ensure_ascii=False) for a in audits)+'\n',encoding='utf-8')
    metrics=['actual_cache_fraction','top10_accepted_hit_proxy','top10_exact_hit_proxy','emitted_id_accepted_hit_proxy',
        'emitted_id_exact_hit_proxy','emitted_reference_anchor_coverage_proxy','whole_gold_brief_visible_proxy',
        'top10_full_source_reference_anchor_coverage_proxy','profile_body_token_fraction','coarse_body_token_fraction',
        'accepted_id_hit_lost_at_materialization_proxy','reference_anchor_coverage_lost_at_materialization_proxy','n_emitted_blocks']
    pair=d.groupby(['pair_index','query_id','aircraft_model','namespace'],as_index=False)[metrics].mean()
    pair.to_csv(OUT/'prefix_materialization_pair_means.csv',index=False,encoding='utf-8-sig')
    summary=d.groupby(['namespace','aircraft_model']).agg(n_profiles=('query_id','size'),n_pairs=('pair_index','nunique'),
        n_defined_anchor_profiles=('emitted_reference_anchor_coverage_proxy','count'),**{m:(m,'mean') for m in metrics}).reset_index()
    summary.to_csv(OUT/'prefix_materialization_support_summary.csv',index=False,encoding='utf-8-sig')
    contrasts=[]
    for baseline,candidate in [('query_first','canonical_state_first'),('canonical_state_first','hierarchical_state_first')]:
        a=d[d.namespace.eq(baseline)].set_index(['query_id','method']);b=d[d.namespace.eq(candidate)].set_index(['query_id','method'])
        assert a.index.equals(b.index)
        x=pd.DataFrame({'query_id':a.index.get_level_values('query_id'),'method':a.index.get_level_values('method'),
            'pair_index':a.pair_index.to_numpy(),'aircraft_model':a.aircraft_model.to_numpy(),
            'baseline':baseline,'candidate':candidate,'cache_fraction_gain':(b.actual_cache_fraction-a.actual_cache_fraction).to_numpy(),
            'reference_anchor_support_loss_proxy':(a.emitted_reference_anchor_coverage_proxy-b.emitted_reference_anchor_coverage_proxy).to_numpy(),
            'accepted_id_support_loss_proxy':(a.emitted_id_accepted_hit_proxy-b.emitted_id_accepted_hit_proxy).to_numpy(),
            'source_body_identical':(a.materialized_source_region_sha256==b.materialized_source_region_sha256).to_numpy()})
        contrasts.append(x)
    ct=pd.concat(contrasts,ignore_index=True)
    ct.to_csv(OUT/'prefix_materialization_paired_contrasts.csv',index=False,encoding='utf-8-sig')
    cp=ct.groupby(['pair_index','query_id','aircraft_model','baseline','candidate'],as_index=False)[
        ['cache_fraction_gain','reference_anchor_support_loss_proxy','accepted_id_support_loss_proxy']].mean()
    cp.to_csv(OUT/'prefix_materialization_paired_pair_means.csv',index=False,encoding='utf-8-sig')
    report={'n_profiles':len(d),'n_pairs':6,'n_methods':33,'zero_exclusions':True,'protocol_sha256':sha(protocol_path.read_text(encoding='utf-8')),
        'gold_missing_profiles':int(d.gold_brief_missing.sum()),'anchor_undefined_profiles':int(d.emitted_reference_anchor_coverage_proxy.isna().sum()),
        'partial_source_headers':int(d.partial_source_headers.sum()),'source_prefix_mismatches':int(d.n_source_prefix_mismatches.sum()),
        'canonical_hierarchy_source_region_identical_pairs_methods':int(body_same.sum()),'n_pairs_methods':len(body_same),
        'overall_layout_means':d.groupby('namespace')[metrics].mean().reset_index().to_dict('records'),
        'paired_contrast_pair_mean_values':cp.to_dict('records'),
        'boundary':'Post-hoc EID presence and long-reference-anchor visibility; not QA correctness, actionable support or admissibility. All statistical units are the six query pairs, not 594 independent tasks.'}
    (OUT/'prefix_materialization_support_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=True),encoding='utf-8')
    print(d.groupby('namespace')[['actual_cache_fraction','top10_accepted_hit_proxy','emitted_id_accepted_hit_proxy','emitted_reference_anchor_coverage_proxy','profile_body_token_fraction','n_emitted_blocks']].mean().to_string())
    print('SOURCE_BODY_IDENTICAL_CANONICAL_HIERARCHY',int(body_same.sum()),'/',len(body_same),flush=True)


if __name__=='__main__':main()
