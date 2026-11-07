"""Standalone source-traceable QA and real GPU KV plots, without image titles."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from make_expanded_figures import Exporter, axes_arrows, forest, make_summary, FAMILY_COLORS, family, display

EXP=Path(__file__).resolve().parents[1]

def qa_forest(ex,summary,origin,metric,stem,xlabel,word=False):
    before=len(ex.catalog)
    forest(ex,summary,origin,metric,stem+('_word' if word else ''),xlabel,word=word)
    if len(ex.catalog)>before:
        ex.catalog[-1]['claim']='Actual generated answers are audited under a common source-context and generation budget.'
        endpoint=(
            'For each answer, quote support is the fraction of parser-recognized quotations that are substrings of the actual supplied source context, with zero assigned when no quotation is recognized. '+
            'The plotted mean therefore combines quotation emission/parser recognition and quotation support; it is not precision conditional on emitting a quotation. '+
            'The generation instruction permits at most one quotation, so an empty quotation field is allowed and does not by itself imply invalid or ungrounded output. '+
            'A supported quotation can still fail to answer the question or provide sufficient maintenance evidence. '
            if metric=='quote_supported_fraction' else
            'JSON-and-answer parsing is a binary endpoint requiring a JSON object with an answer string. It does not require complete citation or quotation fields and is not full schema compliance. '+
            'It does not test whether the answer is correct, sufficient or applicable. '
            if metric=='structured_parse_ok' else
            'Only valid blinded judgments define this endpoint; invalid or absent decisions remain missing rather than correct. ')
        ex.catalog[-1]['caption_en']=(
            'Each point is the normalized-question-cluster mean on the frozen 32-question '+origin+' subset; bars are '+
            '95% percentile bootstrap intervals from 2,000 cluster draws. Every method uses the same local Qwen2.5 3B '+
            'generator and 3,000 source-context token ceiling. The full question denominator is retained; mechanical endpoints score failed executions zero. '+endpoint+
            'All 33 methods are retained. Mechanical JSON/answer parsing and context-substring checks are not semantic QA accuracy, '+
            'source authorization or revision validation. Semantic scoring has no valid decisions in this run and is not plotted. '+
            'Machine-derived questions do not replace an independent expert test.')

def main():
    mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
        'font.size':14,'axes.labelsize':16,'xtick.labelsize':14,'ytick.labelsize':13,
        'legend.fontsize':13,'svg.fonttype':'none','pdf.fonttype':42,'legend.frameon':False})
    ex=Exporter(EXP/'figures_qa_kv',Path('${NATURE_FIGURE_SKILL}'))
    qa=EXP/'results/qa_expanded_main'
    if (qa/'qa_answer_results.csv').exists():
        q=pd.read_csv(qa/'qa_answer_results.csv')
        q['split']=q.origin
        met=['structured_parse_ok','quote_supported_fraction','cited_accepted_evidence_fraction_proxy']
        summary=make_summary(q,met,2000)
        # Preserve denominators and the different conditional-quotation endpoint
        # without replacing the all-answer plotted score or adding exclusions.
        endpoint_meta=pd.read_csv(qa/'qa_summary_by_origin_method.csv').rename(columns={'origin':'split'})
        columns=['split','method','n_answers','n_question_clusters','n_valid_json',
            'quote_verbatim_precision_given_quote','n_answers_with_quotes',
            'citation_valid_precision_given_citation','n_answers_with_citations']
        summary=summary.merge(endpoint_meta[columns],on=['split','method'],how='left',validate='one_to_one')
        calls=q.groupby(['split','method']).agg(
            n_actual_generation_calls=('execution_kind',lambda x:int(x.eq('actual_call').sum())),
            n_identical_prompt_aliases=('execution_kind',lambda x:int(x.eq('duplicate_alias').sum())),
            n_failed_executions=('status',lambda x:int(x.ne('ok').sum()))).reset_index()
        summary=summary.merge(calls,on=['split','method'],how='left',validate='one_to_one')
        for origin in ['legacy','fresh']:
            for word in [False,True]:
                qa_forest(ex,summary,origin,'quote_supported_fraction',origin+'_answer_quote_support','Mean context-supported quote score\n(no recognized quotation = 0)',word)
                qa_forest(ex,summary,origin,'structured_parse_ok',origin+'_answer_schema_compliance','JSON and answer parse fraction',word)
        judge=qa/'blind_semantic_audit/answer_audit_results.csv'
        if judge.exists():
            j=pd.read_csv(judge);j['split']=j.origin
            sm=make_summary(j,['answer_correct_supported_judge_proxy'],2000)
            for origin in ['legacy','fresh']:
                for word in [False,True]:
                    qa_forest(ex,sm,origin,'answer_correct_supported_judge_proxy',origin+'_blinded_answer_audit','Answer correctness and support (model-judge proxy)',word)
    kvpath=EXP/'results/real_kv/kv_pairs.csv'
    if kvpath.exists():
        k=pd.read_csv(kvpath)
        k['actual_cache_fraction']=k.warm_actual_reused_prompt_tokens/k.prompt_token_count_native
        k['token_conservation_error']=k.warm_timing_cache_n+k.warm_timing_prompt_n-k.prompt_token_count_native
        assert k.token_conservation_error.eq(0).all(), 'Cache counter conservation failed'
        layouts=['query_first','canonical_state_first','hierarchical_state_first']
        labels=['Query first','Canonical state first','Hierarchy metadata first']
        colors=['#555555','#356EA5','#007F78']
        fig,ax=plt.subplots(figsize=(9,7.3));fig.subplots_adjust(left=.14,right=.96,bottom=.22,top=.94)
        arrays=[k.loc[k.namespace.eq(l),'actual_cache_fraction'].to_numpy() for l in layouts]
        v=ax.violinplot(arrays,positions=np.arange(3),showextrema=False,showmedians=False,widths=.7)
        rng=np.random.default_rng(20261014)
        for i,(a,c,body) in enumerate(zip(arrays,colors,v['bodies'])):
            body.set_facecolor(c);body.set_alpha(.15);body.set_edgecolor(c)
            ax.scatter(i+rng.uniform(-.17,.17,len(a)),a,s=11,alpha=.35,color=c,edgecolors='none',rasterized=True)
            lo,median,hi=np.quantile(a,[.25,.5,.75]);ax.plot([i,i],[lo,hi],color=c,lw=5)
            ax.scatter([i],[median],s=62,color='white',edgecolor=c,lw=1.5,zorder=5)
        ax.set_xticks(range(3),['Query\nfirst','Canonical\nstate first','Hierarchy\nmetadata first'])
        ax.set_xlim(-.55,2.55);ax.set_ylim(0,1.02);ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_ylabel('Actual reused prompt-token fraction',labelpad=14);ax.grid(axis='y',alpha=.15);axes_arrows(ax)
        ex.save(fig,'actual_kv_reuse_by_prefix_layout',k,
            'Prefix layout affects actual engine KV reuse for fixed visible-query pairs.',
            'All 594 correlated execution profiles are shown: six visible-query pairs, 33 methods and three layouts, or 198 profiles per layout. '+
            'Points are raw observations with seeded horizontal display jitter only; violin densities use Scott bandwidth, '+
            'white dots mark medians and thick strokes interquartile ranges. The six pairs, not the method-by-layout profiles, are the independent sampling units. '+
            'The denominator is the target prompt native token count; reuse is timings.cache_n from the GPU engine. '+
            'Counts include the common system and scope headers. Hierarchy metadata is not answer evidence. '+
            'No inferential test or semantic admissibility is claimed from the marginal distributions; paired contrasts are reported separately.')
        fig,ax=plt.subplots(figsize=(8.5,7.5));fig.subplots_adjust(left=.14,right=.96,bottom=.13,top=.84)
        for l,label,c in zip(layouts,labels,colors):
            g=k[k.namespace.eq(l)];ax.scatter(g.seed_target_lcp_native,g.warm_actual_reused_prompt_tokens,s=20,c=c,alpha=.55,label=label,rasterized=True)
        upper=max(k.seed_target_lcp_native.max(),k.warm_actual_reused_prompt_tokens.max())*1.05
        ax.plot([0,upper],[0,upper],ls='--',color='#999999',lw=1)
        ax.set_xlim(0,upper);ax.set_ylim(0,upper);ax.set_xlabel('Seed-to-target native prefix overlap (tokens)',labelpad=12)
        ax.set_ylabel('Actual reused input KV tokens',labelpad=12);ax.grid(alpha=.15);axes_arrows(ax)
        fig.legend(loc='upper center',bbox_to_anchor=(.55,.985),fontsize=12,ncol=1,frameon=False)
        ex.save(fig,'native_prefix_overlap_vs_actual_kv',k,
            'Native shared-prefix overlap is compared with real cached input tokens.',
            'All 594 correlated profiles (six visible-query pairs, 33 methods and three layouts) are shown without exclusions. '+
            'Each profile was executed as cold target, distinct cold seed and warm target on one fixed GPU slot, with RAM prompt caching disabled '+
            'and identical 32-token greedy decoding settings. '+
            'The dashed diagonal is equality, not a fitted causal relationship. Actual reuse counters are not final slot occupancy. '+
            'Counts include common headers; overlapping marks need not be independent observations. '+
            'Counters validate computational reuse only; they do not certify scope, revision, authority or answer correctness. '+
            'Concurrent device workload limits throughput inference.')
        k.groupby('namespace').agg(n_profiles=('method','size'),n_methods=('method','nunique'),
            mean_cache_fraction=('actual_cache_fraction','mean'),median_cache_fraction=('actual_cache_fraction','median'),
            mean_cached_tokens=('warm_actual_reused_prompt_tokens','mean'),
            mean_cold_prefill_ms=('cold_timing_prompt_ms','mean'),mean_warm_prefill_ms=('warm_timing_prompt_ms','mean'),
            exact_output_identical_fraction=('outputs_identical','mean')).to_csv(EXP/'results/real_kv/layout_summary.csv',encoding='utf-8-sig')
    (ex.out/'figure_manifest.json').write_text(json.dumps({'backend':'Python','figures':ex.catalog,'title_free':True,
        'qa_boundary':'JSON-object/answer-string parsing (not full schema) and context-substring endpoints; no recognized quotation scores zero, though empty quotation is allowed. No valid semantic judgments or semantic plot.',
        'kv_boundary':'Native slot counters, not semantic admissibility; concurrent device workload limits timing inference'},ensure_ascii=False,indent=2),encoding='utf-8')
    print('QA/KV FIGURES',len(ex.catalog),flush=True)

if __name__=='__main__':main()
