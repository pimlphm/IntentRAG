"""Common query pairs and prefix layouts for every evaluated retriever."""
from pathlib import Path
import argparse,hashlib,json,re
import numpy as np
import pandas as pd
from tokenizers import Tokenizer

ROOT=Path(r'${INTENTRAG_PROJECT_ROOT}')
OUT=Path(__file__).resolve().parents[1]
TOPICS=[('engine',r'engine|发动机'),('hydraulic',r'hydraulic|液压'),('electrical',r'battery|electrical|电池|电气'),
        ('controls',r'flap|rudder|aileron|elevator|襟翼|方向舵|副翼'),('gear',r'brake|gear|wheel|刹车|起落架|机轮'),
        ('fuel',r'fuel|燃油'),('air_conditioning',r'pack|air condition|空调')]
SYSTEM='Use only supplied source excerpts. Preserve scope and conditions. State uncertainty when support is missing. Do not authorize maintenance. Reply briefly with source labels.'

def topic(q):
    return next((name for name,rx in TOPICS if re.search(rx,q,re.I)),'other')

def context(eids,ev,tokenizer,budget,canonical):
    if canonical:eids=sorted(eids)
    blocks=[];used=0
    for n,eid in enumerate(eids,1):
        r=ev[eid];header=f'[S{n}] id={eid}; document={r["document_id"]}; page={r["page_number"]}; role={r["role"]}\n'
        text=header+re.sub(r'\s+',' ',r['text']).strip()+'\n'
        if len(tokenizer.encode(text,add_special_tokens=False).ids)>budget-used:
            lo,hi=0,len(text)
            while lo<hi:
                mid=(lo+hi+1)//2
                if len(tokenizer.encode(text[:mid],add_special_tokens=False).ids)<=budget-used:lo=mid
                else:hi=mid-1
            text=text[:lo]
        if not text:break
        blocks.append(text);used+=len(tokenizer.encode(text,add_special_tokens=False).ids)
        if used>=budget:break
    return '\n'.join(blocks),used

def prompt(q,eids,ev,tokenizer,budget,layout):
    body,n=context(eids,ev,tokenizer,budget,layout!='query_first')
    scope='Aircraft='+q['aircraft_model']+'; revision=unverified; authority=unverified.\n'
    if layout=='query_first':user=scope+'Question: '+q['question']+'\nSource excerpts:\n'+body
    elif layout=='canonical_state_first':user=scope+'Source excerpts:\n'+body+'\nQuestion: '+q['question']
    else:
        # Ancestor identity is routing metadata only, never counted as leaf support.
        ancestor=sorted({(ev[e]['document_id'],ev[e]['detected_section_id'],str(ev[e]['component_family'])) for e in eids})
        index='Persistent routing index (metadata, not answer evidence):\n'+'\n'.join('|'.join(t) for t in ancestor[:4])+'\n'
        user=scope+index+'Source excerpts:\n'+body+'\nQuestion: '+q['question']
    return '<|im_start|>system\n'+SYSTEM+'<|im_end|>\n<|im_start|>user\n'+user+'<|im_end|>\n<|im_start|>assistant\n',n

def main():
    p=argparse.ArgumentParser();p.add_argument('--pairs-per-aircraft',type=int,default=3);p.add_argument('--budget',type=int,default=1600)
    p.add_argument('--retrieval',type=Path,default=OUT/'results/all_retrieval_runs.jsonl');a=p.parse_args()
    q=[r for r in json.loads((OUT/'data/canonical_questions.json').read_text(encoding='utf-8')) if r['split']=='v21_locked_blind_test']
    frame=pd.DataFrame(q).drop_duplicates('cluster_id');frame['topic']=frame.question.map(topic)
    rng=np.random.default_rng(20261014);pairs=[]
    for aircraft,sub in frame.groupby('aircraft_model',sort=True):
        eligible=[(name,g) for name,g in sub.groupby('topic',sort=True) if len(g)>=2]
        rng.shuffle(eligible)
        for name,g in eligible[:a.pairs_per_aircraft]:
            x=g.iloc[rng.choice(len(g),2,replace=False)].to_dict('records')
            pairs.append(dict(aircraft=aircraft,topic=name,seed=x[0],target=x[1]))
    runs=[json.loads(l) for l in a.retrieval.read_text(encoding='utf-8').splitlines()]
    methods=sorted({r['method'] for r in runs});lookup={(r['question_id'],r['method']):r for r in runs}
    ev=pd.read_parquet(ROOT/'01_processed_corpus/v5/evidence_objects.parquet').fillna('').set_index('evidence_id').to_dict('index')
    tokenizer=Tokenizer.from_file(str(OUT/'models/qwen2_5_3b_tokenizer/tokenizer.json'))
    records=[]
    for pi,pair in enumerate(pairs):
        for m in methods:
            rows=[lookup[(pair[k]['question_id'],m)] for k in ['seed','target']]
            e=[r['top_evidence_ids'].split(';') for r in rows]
            for layout in ['query_first','canonical_state_first','hierarchical_state_first']:
                prompts=[prompt(pair[k],el,ev,tokenizer,a.budget,layout) for k,el in zip(['seed','target'],e)]
                records.append(dict(method=m,query_id=pair['target']['question_id'],seed_query_id=pair['seed']['question_id'],
                    namespace=layout,pair_index=pi,prompt=prompts[1][0],seed_prompt=prompts[0][0],
                    target_source_tokens=prompts[1][1],seed_source_tokens=prompts[0][1],
                    shared_evidence_ids=sorted(set(e[0])&set(e[1])),aircraft_model=pair['aircraft'],topic=pair['topic']))
    dest=OUT/'data/real_kv_prompts.jsonl'
    dest.write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in records)+'\n',encoding='utf-8')
    (OUT/'data/real_kv_pair_manifest.json').write_text(json.dumps(dict(seed=20261014,n_pairs=len(pairs),n_methods=len(methods),
        n_profiles=len(records),budget=a.budget,pairs=[{k:v for k,v in pair.items() if k not in ['seed','target']}|
        {'seed_query':pair['seed']['question_id'],'target_query':pair['target']['question_id']} for pair in pairs],
        selection='same visible aircraft and lexical topic; no qrels or observed reuse used',
        inherited_revision='unknown',action_authority='unverified',hierarchical_index='metadata; not claim evidence'),indent=2),encoding='utf-8')
    print('KV PAIRS',len(pairs),'METHODS',len(methods),'PROFILES',len(records),flush=True)

if __name__=='__main__':main()
