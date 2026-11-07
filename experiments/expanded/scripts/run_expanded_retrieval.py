"""Matched-budget expanded retrieval; labels enter after ranking only.

All legacy calibration/validation/test rows are reported separately. The learned
rankers use calibration labels only and freeze before validation/test scoring.
Fresh questions are source-derived, with verbatim quote-validated support groups.
"""
from pathlib import Path
import argparse, ast, hashlib, importlib.util, json, math, re, sys, time
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import joblib
from baseline_adapter import build_index, method_registry, METHODS

ROOT=Path(r'${INTENTRAG_PROJECT_ROOT}')
OUT=Path(__file__).resolve().parents[1]
SEED=20261014
K=10

def norm(x):return re.sub(r'\s+',' ',str(x)).strip()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ids(x):return [x for x in str(x).split(';') if x]
def clean(q):return re.split(r'\s+(?:Evidence cue|Include evidence cues):',norm(q),flags=re.I)[0]

def load_module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod)
    return mod

def canonical_questions(ev,include_fresh):
    base=pd.read_csv(ROOT/'03_benchmark/benchmark_v21_evidence_sufficient_locked/qa_warehouse_v21_full.csv').fillna('')
    qrels=pd.read_csv(ROOT/'04_results/v21/v21_gold_sets.csv').fillna('').set_index('question_id')
    rows=[]
    for _,q in base.iterrows():
        l=qrels.loc[q.question_id]
        rows.append(dict(question_id=q.question_id,question=clean(q.question),aircraft_model=q.aircraft_model,
            split=q.split,query_kind=q.query_type,source_group=q.gold_page_id,
            exact_ids=ids(l.exact_gold_evidence_ids),accepted_ids=ids(l.accepted_gold_evidence_ids),
            answer_quotes=[],gold_answer_brief=q.gold_answer_brief,
            old_track=q.track,synthetic_generated=True,primary_page_disjoint=False))
    fresh_file=OUT/'data/fresh_questions_verified.csv'
    if include_fresh and fresh_file.exists():
        fresh=pd.read_csv(fresh_file).fillna('')
        for _,q in fresh.iterrows():
            quotes=json.loads(q.answer_quotes_json)
            rows.append(dict(question_id=q.question_id,question=q.question,aircraft_model=q.aircraft_model,
                split='fresh_locked_test',query_kind=q.query_kind,source_group='fresh_'+str(q.source_group),
                exact_ids=ids(q.gold_evidence_ids),accepted_ids=ids(q.gold_evidence_ids),
                answer_quotes=quotes,gold_answer_brief=q.gold_answer_brief,old_track='',
                synthetic_generated=True,primary_page_disjoint=True))
    known=set(ev.evidence_id)
    by_page={p:df for p,df in ev.groupby('page_id',sort=False)}
    for q in rows:
        assert not re.search('Evidence cue|Include evidence cues',q['question'],re.I)
        assert set(q['exact_ids']) <= known
        q['support_groups']=[]
        for a in q['answer_quotes']:
            sub=by_page[a['page_id']]
            found=sub[sub.text.map(norm).str.contains(re.escape(norm(a['quote'])),regex=True)].evidence_id.tolist()
            assert a['evidence_id'] in found
            q['support_groups'].append(found)
            q['accepted_ids']=sorted(set(q['accepted_ids'])|set(found))
        q['cluster_id']=q['aircraft_model']+'|'+q['question'].lower()
    return rows

def evaluate(q,selected,evmap):
    returned=set(selected);exact=set(q['exact_ids']);accepted=set(q['accepted_ids'])
    actual=[evmap[x] for x in selected]
    gold=[evmap[x] for x in exact]
    roles={str(x['role']) for x in actual};goldroles={str(x['role']) for x in gold}
    pages={x['page_id'] for x in actual};goldpages={x['page_id'] for x in gold}
    groups=q['support_groups']
    coverage=float(np.mean([bool(returned.intersection(g)) for g in groups])) if groups else math.nan
    return dict(accepted_hit_at_10=float(bool(returned&accepted)),exact_hit_at_10=float(bool(returned&exact)),
        exact_set_recall_at_10=len(returned&exact)/max(1,len(exact)),
        accepted_set_recall_at_10=len(returned&accepted)/max(1,len(accepted)),
        required_quote_support_fraction=coverage,all_quote_support=float(coverage==1) if groups else math.nan,
        page_hit_at_10=float(bool(pages&goldpages)),role_match_fraction=len(roles&goldroles)/max(1,len(goldroles)),
        n_evidence=len(selected),n_gold_exact=len(exact),n_gold_accepted=len(accepted))

class BM25:
    """Okapi BM25 with Robertson IDF, k1=1.2, b=.75, no target features."""
    def __init__(self,ev,tokenizer):
        self.ev=ev;self.tok=tokenizer
        from sklearn.feature_extraction.text import CountVectorizer
        self.vec=CountVectorizer(tokenizer=tokenizer,token_pattern=None)
        count=self.vec.fit_transform(ev.text.map(norm)).astype(np.float32)
        lengths=np.asarray(count.sum(axis=1)).ravel();N=len(ev)
        df=np.asarray((count>0).sum(axis=0)).ravel()
        self.idf=np.log(1+(N-df+.5)/(df+.5))
        adj=1.2*(.25+.75*lengths/max(1,lengths.mean()))
        coo=count.tocoo();coo.data=(coo.data*2.2)/(coo.data+adj[coo.row])*self.idf[coo.col]
        self.matrix=coo.tocsr()
    def score(self,q):
        a=self.vec.transform([q]);a.data[:]=1
        return (self.matrix@a.T).toarray().ravel()

class GranularRetriever:
    def __init__(self):
        self.mod=load_module(ROOT/'_pipeline_multigranularity_memory_v1/run_multigranularity_evidence_memory.py','expanded_mg_original')
        self.config=json.loads((ROOT/'configs/multigranularity_memory_v1/multigranularity_memory_v1.json').read_text())
        self.detail,self.views=self.mod.make_views(self.config)
        self.detail_ids=self.detail.evidence_id.to_numpy()
        self.flags=self.mod.evidence_role_flags(self.detail)
        self.lr_multi=None;self.lr_local=None
    def scores(self,q,aircraft):
        ranks,h,w=self.mod.view_scores_for_query(self.config,self.detail,self.views,q,aircraft)
        eligible=self.detail.aircraft_model.eq(aircraft).to_numpy()
        self._ranks,self._h,self._w,self._eligible=ranks,h,w,eligible
        return ranks,h,w,eligible
    def select(self,q,aircraft):
        r,h,w,eligible=self.scores(q,aircraft)
        def f(ranks,weights):
            return self.mod.combine_scores(ranks,{name:weights.get(name,0.) for name in ranks})
        families={
            'Detail-only':r['detail'],
            'Page-resolved':f(r,{'detail':.2,'page':.8}),
            'Section-resolved':f(r,{'detail':.2,'section':.8}),
            'Profile-resolved':f(r,{'detail':.2,'profile':.8}),
            'Short-term fixed':f(r,{'detail':.65,'page':.35}),
            'Long-term fixed':f(r,{'section':.7,'profile':.3}),
            'Multi-granularity fixed':f(r,self.config['fixed_view_weights']),
            'Multi-granularity entropy':f(r,w)}
        result={name:self.mod.top_indices(score,eligible,K) for name,score in families.items()}
        result['Multi-granularity entropy+constraints']=self.mod.admissibility_assemble(
            self.detail,families['Multi-granularity entropy'],eligible,q,K,4,self.flags)
        feature=self.features(q,r,h,w)
        if self.lr_multi is not None:
            candidate=self.candidates(r,eligible)
            p=self.lr_multi.predict_proba(feature[candidate])[:,1]
            order=candidate[np.argsort(-p,kind='stable')]
            result['IntentRAG calibrated multi-granularity']=order[:K].tolist()
            p=self.lr_local.predict_proba(feature[candidate,:3])[:,1]
            order=candidate[np.argsort(-p,kind='stable')]
            result['Calibrated lexical control']=order[:K].tolist()
        return {name:self.detail_ids[index].tolist() for name,index in result.items()},dict(
            **{f'entropy_{name}':v for name,v in h.items()},**{f'weight_{name}':v for name,v in w.items()})
    def candidates(self,r,eligible):
        union=set()
        for arr in r.values():union.update(self.mod.top_indices(arr,eligible,120))
        return np.asarray(sorted(union),dtype=int)
    def features(self,q,r,h,w):
        # All variables derive from query text or corpus; no qid, split, track, or labels.
        scores=[]
        for name in ['detail','page','section','profile']:scores.append(r[name]*61)
        scores.extend([scores[0]*scores[1],scores[0]*scores[2],scores[0]*scores[3]])
        obligations=self.mod.visible_obligations(q)
        for role in ['action','cause','condition','warning','threshold']:
            scores.append(self.flags[role].astype(float)*float(obligations[role]))
        scores.append(np.array([h['detail']]*len(self.detail)))
        scores.append(np.array([w['section']+w['profile']]*len(self.detail)))
        return np.column_stack(scores).astype(np.float32)
    def train(self,calibration):
        X=[];Y=[]
        for j,q in enumerate(calibration):
            r,h,w,elig=self.scores(q['question'],q['aircraft_model'])
            candidate=self.candidates(r,elig)
            x=self.features(q['question'],r,h,w)[candidate]
            y=np.isin(self.detail_ids[candidate],q['accepted_ids']).astype(int)
            if y.any():X.append(x);Y.append(y)
            if j%30==0:print('CALIBRATION',j,flush=True)
        x=np.concatenate(X);y=np.concatenate(Y)
        self.lr_multi=make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=500,class_weight='balanced',random_state=SEED))
        self.lr_local=make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=500,class_weight='balanced',random_state=SEED))
        self.lr_multi.fit(x,y);self.lr_local.fit(x[:,:3],y)
        model_dir=OUT/'models';model_dir.mkdir(exist_ok=True)
        joblib.dump(self.lr_multi,model_dir/'calibrated_multigranularity.joblib')
        joblib.dump(self.lr_local,model_dir/'calibrated_lexical_control.joblib')
        (model_dir/'training_manifest.json').write_text(json.dumps(dict(seed=SEED,
            query_ids=[q['question_id'] for q in calibration],n_candidate_rows=len(y),n_positive=int(y.sum()),
            labels='legacy accepted evidence; calibration split only',
            features='detail/page/section/profile ranks, products, visible role matches, entropy weights',
            fixed_C=.1,control_features='detail/page/section ranks; three-view control'),indent=2),encoding='utf-8')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--include-fresh',action='store_true')
    parser.add_argument('--limit',type=int,default=0);args=parser.parse_args()
    for name in ['data','results','reports','models']:(OUT/name).mkdir(exist_ok=True)
    ev=pd.read_parquet(ROOT/'01_processed_corpus/v5/evidence_objects.parquet').fillna('')
    evmap=ev.set_index('evidence_id').to_dict('index')
    questions=canonical_questions(ev,args.include_fresh)
    pd.DataFrame(questions).to_csv(OUT/'data/canonical_questions.csv',index=False,encoding='utf-8-sig')
    (OUT/'data/canonical_questions.json').write_text(json.dumps(questions,ensure_ascii=False),encoding='utf-8')
    print('BUILD BASELINE INDEX',flush=True); baseline=build_index(ROOT,ev)
    print('BUILD GRANULAR INDEX',flush=True);gran=GranularRetriever()
    model=OUT/'models/calibrated_multigranularity.joblib'
    if model.exists():
        gran.lr_multi=joblib.load(model);gran.lr_local=joblib.load(OUT/'models/calibrated_lexical_control.joblib')
    else:gran.train([q for q in questions if q['split']=='v21_calibration'])
    print('BUILD BM25',flush=True);bm25=BM25(ev,baseline.v21_module.tokenize)
    registry=method_registry()+[dict(method=x,display_name=('Calibrated three-view control' if x=='Calibrated lexical control' else x),official_reproduction=False,
        implementation_kind='original_multigranularity_or_calibration_control') for x in [
        'True BM25','IntentRAG balanced fair','Detail-only','Page-resolved','Section-resolved','Profile-resolved',
        'Short-term fixed','Long-term fixed','Multi-granularity fixed','Multi-granularity entropy',
        'Multi-granularity entropy+constraints','IntentRAG calibrated multi-granularity','Calibrated lexical control']]
    pd.DataFrame(registry).to_csv(OUT/'data/method_registry.csv',index=False,encoding='utf-8-sig')
    dest=OUT/'results/retrieval_runs.jsonl'
    completed={(r['question_id'],r['method']) for r in map(json.loads,dest.read_text(encoding='utf-8').splitlines())} if dest.exists() else set()
    selected_questions=questions[:args.limit] if args.limit else questions
    for j,q in enumerate(selected_questions):
        all_methods=METHODS+('True BM25','IntentRAG balanced fair','Detail-only','Page-resolved','Section-resolved','Profile-resolved',
            'Short-term fixed','Long-term fixed','Multi-granularity fixed','Multi-granularity entropy',
            'Multi-granularity entropy+constraints','IntentRAG calibrated multi-granularity','Calibrated lexical control')
        if all((q['question_id'],m) in completed for m in all_methods):continue
        t=time.perf_counter();original=baseline.retrieve_all(q['question'],q['aircraft_model'],K)
        elapsed=time.perf_counter()-t
        chosen={name:[r['evidence_id'] for r in rows] for name,rows in original.items()}
        raw=bm25.score(q['question']);raw[~ev.aircraft_model.eq(q['aircraft_model']).to_numpy()]=-1e9
        ix=baseline.v21_module.top_indices(raw,K);chosen['True BM25']=ev.iloc[ix].evidence_id.tolist()
        local_ev,local_index=baseline._v21(q['aircraft_model'])
        raw_scores=baseline._query_cache[(q['question'],q['aircraft_model'],'v21_scores')]
        order=baseline.v21_module.route_union_indices(raw_scores,
            ['Hybrid lexical','TFIDF-char','BM25/word','FD-GRAG-RC-V21-package','FD-GRAG-RC-V21-cue-route'],
            K,[('BM25/word',2),('TFIDF-char',2),('Hybrid lexical',2),('FD-GRAG-RC-V21-package',2),('FD-GRAG-RC-V21-cue-route',2)])
        chosen['IntentRAG balanced fair']=local_ev.iloc[order].evidence_id.tolist()
        t=time.perf_counter();extras,router=gran.select(q['question'],q['aircraft_model']);mg_elapsed=time.perf_counter()-t
        chosen.update(extras)
        with dest.open('a',encoding='utf-8') as handle:
            for method,top in chosen.items():
                if (q['question_id'],method) in completed:continue
                assert all(evmap[e]['aircraft_model']==q['aircraft_model'] for e in top)
                assert len(top)<=K and len(set(top))==len(top)
                row=dict(question_id=q['question_id'],question=q['question'],aircraft_model=q['aircraft_model'],
                    split=q['split'],cluster_id=q['cluster_id'],source_group=q['source_group'],method=method,
                    top_evidence_ids=';'.join(top),synthetic_query=True,gold_in_online_features=False,
                    **evaluate(q,top,evmap),**router,grouped_baseline_runtime_s=elapsed,multigranularity_runtime_s=mg_elapsed)
                handle.write(json.dumps(row,ensure_ascii=False,allow_nan=True)+'\n')
        print(f'QUERY {j+1}/{len(selected_questions)} methods={len(chosen)} runtime={elapsed+mg_elapsed:.2f}',flush=True)
    frame=pd.DataFrame(map(json.loads,dest.read_text(encoding='utf-8').splitlines()))
    frame.to_csv(OUT/'results/retrieval_query_results.csv',index=False,encoding='utf-8-sig')
    metrics=['accepted_hit_at_10','exact_hit_at_10','exact_set_recall_at_10','accepted_set_recall_at_10','required_quote_support_fraction','all_quote_support','page_hit_at_10','role_match_fraction']
    for group in [['split','method'],['split','aircraft_model','method']]:
        summary=frame.groupby(group)[metrics].agg(['mean','count']).reset_index()
        summary.columns=['_'.join(c).rstrip('_') if isinstance(c,tuple) else c for c in summary.columns]
        summary.to_csv(OUT/'results'/('retrieval_summary_'+('aircraft' if 'aircraft_model' in group else 'split')+'.csv'),index=False,encoding='utf-8-sig')
    manifest=dict(n_queries=len(questions),n_distinct_questions=len(set(q['cluster_id'] for q in questions)),
        n_methods=len(registry),n_query_method_rows=len(frame),seed=SEED,k=K,
        split_counts=pd.Series([q['split'] for q in questions]).value_counts().to_dict(),
        source_hash=sha(ROOT/'01_processed_corpus/v5/evidence_objects.parquet'),
        benchmark_hash=sha(ROOT/'03_benchmark/benchmark_v21_evidence_sufficient_locked/qa_warehouse_v21_full.csv'),
        source_derived_queries=True,calibration_labels_only=True,
        model_sha256=sha(model),real_gpu_cache_measured_by_separate_script=True,
        metric_warning='Historical R@10 is binary Hit@10; true set recall is reported separately. Role matches do not validate actions.')
    (OUT/'reports/retrieval_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest),flush=True)

if __name__=='__main__':main()
