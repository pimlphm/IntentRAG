"""Merge executed methods, check coverage, and compare paired test outcomes."""
from pathlib import Path
import argparse,json,hashlib,math
import numpy as np
import pandas as pd
from scipy.stats import binomtest

OUT=Path(__file__).resolve().parents[1]
ROOT=Path(r'${INTENTRAG_PROJECT_ROOT}')
SEED=20261014

def read_runs(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()] if path.exists() else []

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--allow-incomplete',action='store_true');args=ap.parse_args()
    questions=json.loads((OUT/'data/canonical_questions.json').read_text(encoding='utf-8'))
    qmap={q['question_id']:q for q in questions}
    rows=read_runs(OUT/'results/retrieval_runs.jsonl')+read_runs(OUT/'results/neural_retrieval_runs.jsonl')
    frame=pd.DataFrame(rows)
    if frame.duplicated(['question_id','method']).any():
        raise ValueError('Duplicate query-method rows; do not select favorable records.')
    assert set(frame.question_id)<=set(qmap)
    methods=sorted(frame.method.unique())
    actual=set(zip(frame.question_id,frame.method))
    missing=[(qid,m) for qid in qmap for m in methods if (qid,m) not in actual]
    pd.DataFrame(missing,columns=['question_id','method']).to_csv(OUT/'reports/missing_retrieval_rows.csv',index=False,encoding='utf-8-sig')
    if missing and not args.allow_incomplete:raise RuntimeError(f'{len(missing)} missing query-method executions')
    corpus=pd.read_parquet(ROOT/'01_processed_corpus/v5/evidence_objects.parquet').fillna('')
    corpus_map=corpus.set_index('evidence_id').to_dict('index')
    precision=[]
    for r in frame.to_dict('records'):
        top=r['top_evidence_ids'].split(';')
        q=qmap[r['question_id']]
        assert len(top)<=10 and len(set(top))==len(top)
        assert all(corpus_map[e]['aircraft_model']==q['aircraft_model'] for e in top)
        precision.append(len(set(top)&set(q['accepted_ids']))/max(1,len(top)))
    frame['accepted_precision_at_10']=precision
    frame['n_required_source_groups']=[len(qmap[qid].get('support_groups',[])) for qid in frame.question_id]
    frame['query_kind']=[qmap[qid].get('query_kind','unspecified') for qid in frame.question_id]
    frame.to_csv(OUT/'results/all_retrieval_query_results.csv',index=False,encoding='utf-8-sig')
    with (OUT/'results/all_retrieval_runs.jsonl').open('w',encoding='utf-8') as f:
        for r in frame.to_dict('records'):f.write(json.dumps(r,ensure_ascii=False)+'\n')
    metrics=['accepted_hit_at_10','exact_hit_at_10','accepted_precision_at_10','accepted_set_recall_at_10',
        'exact_set_recall_at_10','required_quote_support_fraction','all_quote_support','page_hit_at_10']
    summary=[];rng=np.random.default_rng(SEED)
    for (split,method),g in frame.groupby(['split','method']):
        # Repeated source-derived question texts are averaged before resampling.
        clustered=g.groupby('cluster_id')[metrics].mean()
        row=dict(split=split,method=method,n_records=len(g),n_question_clusters=len(clustered),
                 n_source_groups=g.source_group.nunique())
        for metric in metrics:
            values=clustered[metric].dropna().to_numpy()
            row[metric]=float(g[metric].mean()) if g[metric].notna().any() else math.nan
            row[metric+'_cluster_mean']=float(values.mean()) if len(values) else math.nan
            if len(values):
                boot=values[rng.integers(0,len(values),(2000,len(values)))].mean(axis=1)
                row[metric+'_lo'],row[metric+'_hi']=map(float,np.quantile(boot,[.025,.975]))
            source_values=g.groupby('source_group')[metric].mean().dropna().to_numpy()
            row[metric+'_source_group_mean']=float(source_values.mean()) if len(source_values) else math.nan
            if len(source_values):
                source_boot=source_values[rng.integers(0,len(source_values),(2000,len(source_values)))].mean(axis=1)
                row[metric+'_source_lo'],row[metric+'_source_hi']=map(float,np.quantile(source_boot,[.025,.975]))
        summary.append(row)
    pd.DataFrame(summary).to_csv(OUT/'results/all_retrieval_summary.csv',index=False,encoding='utf-8-sig')
    frame.groupby(['split','aircraft_model','method'])[metrics].agg(['mean','count']).to_csv(OUT/'results/all_retrieval_aircraft_summary.csv',encoding='utf-8-sig')
    frame[frame.split.eq('fresh_locked_test')].groupby(['n_required_source_groups','query_kind','method'])[metrics].agg(['mean','count']).to_csv(OUT/'results/fresh_required_sources_summary.csv',encoding='utf-8-sig')
    comparisons=[]
    targets=['IntentRAG balanced fair','Multi-granularity entropy+constraints','IntentRAG calibrated multi-granularity','IntentRAG multi + mMARCO CE@50']
    for target,split in [(target,split) for target in targets for split in ['v21_locked_blind_test','fresh_locked_test']]:
        part=frame[frame.split.eq(split)]
        for reference in methods:
            if reference==target:continue
            for metric in ['accepted_hit_at_10','required_quote_support_fraction']:
                wide=part[part.method.isin([target,reference])].pivot(index='question_id',columns='method',values=metric).dropna()
                if target not in wide or reference not in wide or len(wide)==0:continue
                wide['cluster']=[qmap[q]['cluster_id'] for q in wide.index]
                d=(wide[target]-wide[reference]).groupby(wide.cluster).mean().to_numpy()
                boot=d[rng.integers(0,len(d),(2000,len(d)))].mean(axis=1)
                wins=int((d>0).sum());losses=int((d<0).sum())
                comparisons.append(dict(split=split,method=target,reference=reference,metric=metric,
                    n_paired=len(wide),n_question_clusters=len(d),mean_paired_difference=float(d.mean()),
                    ci_low=float(np.quantile(boot,.025)),ci_high=float(np.quantile(boot,.975)),
                    wins=wins,losses=losses,ties=int(len(d)-wins-losses),
                    sign_test_p=float(binomtest(wins,wins+losses,.5).pvalue) if wins+losses else 1.0))
    comp=pd.DataFrame(comparisons)
    if len(comp):
        for _,group in comp.groupby(['method','split','metric']):
            order=group.sign_test_p.sort_values().index;previous=0
            for rank,i in enumerate(order):
                previous=max(previous,min(1,comp.at[i,'sign_test_p']*(len(order)-rank)))
                comp.loc[i,'holm_p']=previous
        comp.to_csv(OUT/'results/paired_method_comparisons.csv',index=False,encoding='utf-8-sig')
    oldgold=pd.read_csv(ROOT/'04_results/v21/v21_gold_sets.csv').fillna('')
    accepted_union=set(x for field in oldgold.accepted_gold_evidence_ids for x in str(field).split(';') if x)
    old_primary_pages=set(pd.read_csv(ROOT/'03_benchmark/benchmark_v21_evidence_sufficient_locked/qa_warehouse_v21_full.csv').gold_page_id.astype(str))
    fresh=[q for q in questions if q['split']=='fresh_locked_test']
    audit=[]
    for q in fresh:
        pages={a['page_id'] for a in q['answer_quotes']}
        audit.append(dict(question_id=q['question_id'],primary_pages_disjoint=not bool(pages&old_primary_pages),
            target_seen_in_legacy_accepted_equivalence=bool(set(q['exact_ids'])&accepted_union),
            actual_n_required_sources=len(q['support_groups']),source_group=q['source_group']))
    pd.DataFrame(audit).to_csv(OUT/'reports/fresh_source_overlap_audit.csv',index=False,encoding='utf-8-sig')
    manifest=dict(n_queries=len(questions),n_distinct_question_clusters=len(set(q['cluster_id'] for q in questions)),
        n_methods=len(methods),n_executed_rows=len(frame),n_missing=len(missing),
        split_counts=pd.Series([q['split'] for q in questions]).value_counts().to_dict(),
        fresh_aircraft_counts=pd.Series([q['aircraft_model'] for q in fresh]).value_counts().to_dict(),
        methods=methods,all_primary_fresh_pages_disjoint=all(r['primary_pages_disjoint'] for r in audit),
        fresh_targets_in_legacy_accepted_union=sum(r['target_seen_in_legacy_accepted_equivalence'] for r in audit),
        source_derived=True,expert_review=False,bootstrap='2000 query-text-cluster and source-group draws separately; sign tests on cluster differences',
        display_name_overrides={'Calibrated lexical control':'Calibrated three-view control'},
        high_precision_endpoint='Actual measured rates; no success threshold imposed after observing test')
    (OUT/'reports/expanded_execution_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(manifest,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
