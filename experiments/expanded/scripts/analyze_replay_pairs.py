"""Compare all methods under paired request orders; no independence inflation."""
import argparse
import json
from pathlib import Path
import pandas as pd


def main():
    ap=argparse.ArgumentParser()
    exp=Path(__file__).resolve().parents[1]
    ap.add_argument('--memory-dir',type=Path,default=exp/'results/memory_replay')
    ap.add_argument('--target',default='IntentRAG balanced fair')
    args=ap.parse_args()
    path=args.memory_dir/'replay_order_summary.csv'
    if path.exists():
        s=pd.read_csv(path)
    else:
        d=pd.read_csv(args.memory_dir/'replay_query_results.csv')
        rows=[]
        for key,g in d.groupby(['split','method','capacity','order','policy','seed'],dropna=False):
            mass=g.requested_token_mass.sum()
            rows.append({**dict(zip(['split','method','capacity','order','policy','seed'],key)),
                'n_queries':g.question_id.nunique(),
                'token_weighted_reused_fraction':g.reused_token_mass.sum()/mass if mass else float('nan'),
                'module_hit_fraction':g.n_cached_identity_hits.sum()/g.n_known_requested_ids.sum() if g.n_known_requested_ids.sum() else float('nan')})
        s=pd.DataFrame(rows)
        s.to_csv(path,index=False,encoding='utf-8-sig')
    rows=[]
    for key,g in s.groupby(['split','capacity','order','policy']):
        for metric in ['token_weighted_reused_fraction','module_hit_fraction']:
            w=g.pivot(index='seed',columns='method',values=metric)
            if args.target not in w:continue
            for other in w.columns:
                if other==args.target:continue
                pair=w[[args.target,other]].dropna()
                delta=pair[args.target]-pair[other]
                if len(delta)==0:continue
                rows.append({**dict(zip(['split','capacity','order','policy'],key)),
                    'target':args.target,'reference':other,'metric':metric,
                    'n_paired_request_orders':len(delta),'mean_paired_delta':delta.mean(),
                    'paired_delta_min':delta.min(),'paired_delta_max':delta.max(),
                    'paired_delta_sd':delta.std(ddof=1) if len(delta)>1 else float('nan'),
                    'wins':int((delta>0).sum()),'losses':int((delta<0).sum()),'ties':int((delta==0).sum()),
                    'uncertainty_scope':'Variation across the same fixed-query workload orders; not a population CI'})
    pd.DataFrame(rows).to_csv(args.memory_dir/'paired_reuse_method_comparisons.csv',index=False,encoding='utf-8-sig')
    print(json.dumps({'rows':len(rows),'request_order_summary':str(path),'target':args.target}))


if __name__=='__main__':main()
