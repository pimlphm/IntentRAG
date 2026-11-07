"""Standalone, title-free figures for executed expanded retrieval and LRU replay.

All methods and negative results are retained. Figures show measured retrieval
and source-module reuse separately. No QA improvement is inferred from reuse.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import textwrap

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

SEED=20261014
TARGET='IntentRAG balanced fair'
DISPLAY={
    'BM25/word':'Word TF-IDF (legacy label)',
    'TFIDF-char':'Character TF-IDF',
    'FD-GRAG-RC-V21-final':'IntentRAG visible-router adaptation',
    'Calibrated lexical control':'Calibrated three-view control',
}
WORD_ALIASES={
    'BM25/word':'Word TF-IDF','TFIDF-char':'Char TF-IDF',
    'SVD semantic surrogate':'SVD surrogate','Hybrid lexical':'Lexical hybrid',
    'Parent-section retrieval':'Parent-section','Component-profile retrieval':'Component profile',
    'Controlled GraphRAG':'Controlled GraphRAG','Causal-graph informed RAG':'Causal graph control',
    'Component-graph informed RAG':'Component graph control','Agentic RAG':'Agentic route control',
    'Evidence-flow baseline':'Evidence-flow control','GaRAGe-style relevance RAG':'GaRAGe-style relevance',
    'GaRAGe-style attribution RAG':'GaRAGe-style attribution','GaRAGe-style deflective RAG':'GaRAGe-style deflection',
    'FD-GRAG-RC-V21-final':'IntentRAG visible route','True BM25':'Okapi BM25',
    'IntentRAG balanced fair':'IntentRAG balanced','Detail-only':'Local detail only',
    'Page-resolved':'Page route','Section-resolved':'Section route','Profile-resolved':'Profile route',
    'Short-term fixed':'Short-term fixed','Long-term fixed':'Long-term fixed',
    'Multi-granularity fixed':'Multi-view fixed','Multi-granularity entropy':'Multi-view entropy',
    'Multi-granularity entropy+constraints':'Entropy + constraints',
    'IntentRAG calibrated multi-granularity':'Calibrated multi-view',
    'Calibrated lexical control':'Calibrated three-view',
    'Neural dense (multilingual MiniLM)':'MiniLM dense',
    'Neural dense + mMARCO CE@30':'Dense + mMARCO@30',
    'Neural dense + mMARCO CE@50':'Dense + mMARCO@50',
    'Neural lexical+dense RRF':'Lexical-dense RRF',
    'IntentRAG multi + mMARCO CE@50':'IntentRAG multi + CE@50',
}
assert len(set(WORD_ALIASES.values()))==len(WORD_ALIASES), 'Display aliases must remain unique'
FAMILY_COLORS={'Lexical/local':'#555555','Structural controls':'#7A6F9B',
               'Memory/granularity':'#007F78','Neural retrieval':'#356EA5'}
FAMILY_MARKERS={'Lexical/local':'o','Structural controls':'s',
                'Memory/granularity':'^','Neural retrieval':'D'}
PALETTE=['#0072B2','#D55E00','#009E73','#7A6F9B','#555555','#C38D27',
         '#7B8F69','#556B7B','#7B4F61','#92745A','#4E729C','#4B8A88']
GRANS=['facetpoint','paragraph','table_row','page','section','component_profile','other']
GRAN_LABELS=['Facet span','Paragraph','Table row','Page','Section','Profile','Other']
GRAN_COLORS=['#356EA5','#76A5B8','#6C9690','#B89965','#8E82A5','#BC8E9B','#888888']


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def family(m):
    if re.search(r'MiniLM|mMARCO|BGE|E5|dense|cross.encoder|RRF',m,re.I):return 'Neural retrieval'
    if m in {'BM25/word','TFIDF-char','SVD semantic surrogate','Hybrid lexical','True BM25','Detail-only','Calibrated lexical control'}:return 'Lexical/local'
    if re.search(r'granularity|Short-term|Long-term|resolved',m,re.I):return 'Memory/granularity'
    return 'Structural controls'


def display(m):return DISPLAY.get(m,m)


def word_display(m):
    if m not in WORD_ALIASES:raise ValueError('Define a unique display alias for '+m)
    return WORD_ALIASES[m]


def axes_arrows(ax):
    ax.spines[['top','right','left','bottom']].set_visible(False)
    props=dict(arrowstyle='-|>',lw=1.1,color='#222222',mutation_scale=14)
    ax.annotate('',xy=(1.018,0),xytext=(0,0),xycoords='axes fraction',arrowprops=props,annotation_clip=False)
    ax.annotate('',xy=(0,1.018),xytext=(0,0),xycoords='axes fraction',arrowprops=props,annotation_clip=False)
    ax.tick_params(axis='both',length=4,width=1,pad=7)


def make_summary(d,metrics,bootstraps):
    rng=np.random.default_rng(SEED); rows=[]
    for (split,method),g in d.groupby(['split','method'],sort=True):
        part=g.groupby('cluster_id')[metrics].mean()
        row={'split':split,'method':method,'n_records':len(g),'n_clusters':len(part)}
        for metric in metrics:
            defined=part[metric].notna()
            row[metric+'_n_defined_clusters']=int(defined.sum())
            row[metric+'_n_undefined_clusters']=int((~defined).sum())
            a=part.loc[defined,metric].to_numpy(float)
            row[metric+'_record_mean']=float(g[metric].mean()) if len(a) else np.nan
            row[metric]=float(a.mean()) if len(a) else np.nan
            if len(a):
                means=a[rng.integers(0,len(a),size=(bootstraps,len(a)))].mean(axis=1)
                row[metric+'_lo'],row[metric+'_hi']=np.quantile(means,[.025,.975])
            else:row[metric+'_lo']=row[metric+'_hi']=np.nan
        rows.append(row)
    return pd.DataFrame(rows)


class Exporter:
    def __init__(self,out,skill_scripts):
        self.out=out; self.out.mkdir(parents=True,exist_ok=True)
        self.source=self.out/'source_data';self.source.mkdir(exist_ok=True)
        self.qa=self.out/'qa';self.qa.mkdir(exist_ok=True)
        sys.path.insert(0,str(skill_scripts))
        from audit_panel_alignment import require_matplotlib_panel_alignment
        self.alignment=require_matplotlib_panel_alignment
        self.catalog=[]
    def save(self,fig,stem,data,claim,caption):
        base=self.out/stem
        fig.canvas.draw()
        self.alignment(fig,json_out=str(self.qa/(stem+'.alignment.json')),
            overlay_svg=str(self.qa/(stem+'.alignment.svg')),strict=True)
        fig.savefig(str(base)+'.svg',facecolor='white',bbox_inches=None)
        fig.savefig(str(base)+'.pdf',facecolor='white',bbox_inches=None)
        fig.savefig(str(base)+'.png',dpi=600,facecolor='white',bbox_inches=None)
        data.to_csv(self.source/(stem+'.csv'),index=False,encoding='utf-8-sig')
        self.catalog.append({'figure':stem,'claim':claim,'caption_en':caption,
            'source_data':'source_data/'+stem+'.csv','n_axes':len(fig.axes),
            'size_inches':fig.get_size_inches().tolist(),'title_in_image':False})
        plt.close(fig)


def forest(ex,summary,split,metric,stem,xlabel,word=False):
    s=summary[summary.split.eq(split)].copy()
    if s.empty:return
    if s[metric].isna().all():return
    # Ranking is a display order only; no baseline is dropped.
    s=s.sort_values([metric,'method'],ascending=[False,True],na_position='last').reset_index(drop=True)
    height=max(6,1.4+.32*len(s)) if word else max(6,1.55+.37*len(s))
    fig,ax=plt.subplots(figsize=(9 if word else 12.8,height))
    fig.subplots_adjust(left=.43 if word else .46,right=.89 if word else .87,bottom=.09,top=.97)
    s['display_alias']=s.method.map(word_display if word else display)
    y=np.arange(len(s)); ax.set_yticks(y,s.display_alias if word else [textwrap.fill(v,width=39) for v in s.display_alias],fontsize=12 if word else 11.5)
    for i,r in s.iterrows():
        if pd.isna(r[metric]):
            ax.text(.03,i,'Not defined',ha='left',va='center',fontsize=11,color='#777777');continue
        color='#D55E00' if r.method==TARGET else FAMILY_COLORS[family(r.method)]
        ax.errorbar(r[metric],i,xerr=[[r[metric]-r[metric+'_lo']],[r[metric+'_hi']-r[metric]]],
            fmt=FAMILY_MARKERS[family(r.method)],color=color,elinewidth=1.5,
            markersize=7.5 if r.method==TARGET else 6,capsize=3)
        ax.text(1.055 if word else 1.08,i,f'{r[metric]:.3f}',transform=ax.get_yaxis_transform(),va='center',ha='left',fontsize=12)
    ax.set_ylim(len(s)-.4,-.8);ax.set_xlim(0,1.04)
    ax.set_xticks([0,.2,.4,.6,.8,1]);ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.set_xlabel(xlabel,labelpad=12);ax.grid(axis='x',alpha=.15);axes_arrows(ax)
    ex.save(fig,stem,s,
        'All executed methods are compared under the same query and Top-10 budgets.',
        'Each point is the mean over normalized question-text clusters; bars are 95% percentile bootstrap intervals from '+
        '2,000 cluster draws. All methods are retained, including controls and negative results. Missing endpoints are shown explicitly. '+
        f'Split: {split}. This is source-derived retrieval evaluation, not maintenance correctness or QA accuracy.')


def granularity_composition(ex,d,canonical,corpus,tokenizer_path,split,accepted_only,stem,word=False):
    from tokenizers import Tokenizer
    tok=Tokenizer.from_file(str(tokenizer_path))
    emap={str(r['evidence_id']):r for r in corpus.to_dict('records')}
    tokens={}
    accepted={str(q['question_id']):set(q['accepted_ids']) for q in canonical}
    source=[]
    for method,g in d[d.split.eq(split)].groupby('method',sort=True):
        amounts=dict.fromkeys(GRANS,0)
        n=0
        for row in g.itertuples():
            for eid in str(row.top_evidence_ids).split(';'):
                if accepted_only and eid not in accepted[row.question_id]:continue
                r=emap[eid];gr=r['granularity'] if r['granularity'] in GRANS else 'other'
                if eid not in tokens:tokens[eid]=len(tok.encode(str(r['text']),add_special_tokens=False).ids)
                amounts[gr]+=tokens[eid];n+=1
        total=sum(amounts.values())
        source.append({'method':method,'n_source_units':n,'source_token_mass':total,
                       **{gr:amounts[gr]/total if total else np.nan for gr in GRANS}})
    s=pd.DataFrame(source).sort_values('method').reset_index(drop=True)
    s['display_alias']=s.method.map(word_display if word else display)
    fig,ax=plt.subplots(figsize=(9 if word else 12.8,max(6,1.4+.32*len(s)) if word else max(6,1.7+.37*len(s))))
    fig.subplots_adjust(left=.43 if word else .46,right=.88 if word else .96,bottom=.18,top=.95)
    left=np.zeros(len(s));ys=np.arange(len(s))
    for gr,label,c in zip(GRANS,GRAN_LABELS,GRAN_COLORS):
        ax.barh(ys,s[gr].fillna(0),left=left,height=.65,label=label,color=c,edgecolor='white',linewidth=.4)
        left+=s[gr].fillna(0).to_numpy()
    ax.set_yticks(ys,s.display_alias if word else [textwrap.fill(v,39) for v in s.display_alias],fontsize=12 if word else 11.5);ax.set_ylim(len(s)-.3,-.7)
    if word:
        for i,r in s.iterrows():ax.text(1.05,i,str(int(r.n_source_units)),transform=ax.get_yaxis_transform(),va='center',fontsize=12)
        ax.text(1.05,1.01,'Units',transform=ax.transAxes,va='bottom',fontsize=11)
    ax.set_xlim(0,1.01);ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.set_xlabel('Share of retained source-text token mass',labelpad=12)
    handles,labels=ax.get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.68 if word else .71,.022),
               ncol=2 if word else 3,fontsize=11,columnspacing=1.1,handlelength=1.1,frameon=False)
    axes_arrows(ax)
    scope='Post-hoc accepted support units' if accepted_only else 'All returned source units'
    ex.save(fig,stem,s,
        'The actual granularity of retained source text differs across methods.',
        f'{scope} are pooled over {split}. Fractions are weighted by official Qwen2.5 source-text token counts. '+
        'The granularity label belongs to the returned source unit; a parent routing score does not change it. '+
        'Accepted support labels are used after retrieval only. This describes output composition, not causal information gain or summary sufficiency. '+
        ('Right-hand counts are pooled source-unit occurrences, not independent facts or successful questions.' if word else ''))


def read_order_summary(memory_dir):
    path=memory_dir/'replay_order_summary.csv'
    if path.exists():return pd.read_csv(path)
    # Older core replay may lack this derived table. Read only necessary columns.
    cols=['split','method','capacity','order','policy','seed','requested_token_mass','reused_token_mass','n_cached_identity_hits','n_known_requested_ids','question_id']
    d=pd.read_csv(memory_dir/'replay_query_results.csv',usecols=cols)
    out=[]
    for key,g in d.groupby(['split','method','capacity','order','policy','seed'],dropna=False):
        mass=g.requested_token_mass.sum()
        out.append({**dict(zip(['split','method','capacity','order','policy','seed'],key)),
            'token_weighted_reused_fraction':g.reused_token_mass.sum()/mass if mass else np.nan,
            'module_hit_fraction':g.n_cached_identity_hits.sum()/g.n_known_requested_ids.sum() if g.n_known_requested_ids.sum() else np.nan,
            'n_queries':g.question_id.nunique()})
    return pd.DataFrame(out)


def reuse_curves(ex,orders,split):
    s=orders[orders.split.eq(split)&orders.policy.eq('flat_identity_lru')&orders.order.eq('random')].copy()
    assert s.capacity.gt(0).all(), 'Log capacity axis requires positive module capacity'
    s['family']=s.method.map(family)
    for fam,part in s.groupby('family',sort=True):
        methods=sorted(part.method.unique());fig,ax=plt.subplots(figsize=(13,7.5))
        fig.subplots_adjust(left=.09,right=.54,bottom=.12,top=.95)
        for i,m in enumerate(methods):
            g=part[part.method.eq(m)].groupby('capacity').token_weighted_reused_fraction.agg(['mean','min','max']).reset_index()
            ax.errorbar(g.capacity,g['mean'],yerr=np.vstack([g['mean']-g['min'],g['max']-g['mean']]),
                label=textwrap.fill(display(m),36),color=PALETTE[i%len(PALETTE)],
                marker=['o','s','^','D','v','P','X'][i%7],markersize=6,capsize=3,lw=1.6)
        ax.set_xscale('log',base=2);ax.set_xticks([16,64,256],['16','64','256']);ax.set_xlim(12,330)
        ax.set_ylim(0,1.02);ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_xlabel('Capacity (exact source modules)',labelpad=12);ax.set_ylabel('Source-token-mass reuse fraction',labelpad=12)
        ax.grid(alpha=.15);axes_arrows(ax)
        ax.legend(loc='center left',bbox_to_anchor=(1.07,.5),fontsize=11,frameon=False,
                  handlelength=2,columnspacing=1.2,labelspacing=.75)
        stem='reuse_capacity_'+split+'_'+re.sub('[^a-z0-9]+','_',fam.lower())
        ex.save(fig,stem,part,
            'Reuse opportunity changes with finite source-module capacity across the same workload orders.',
            'Each method uses the same exact-source identity LRU and the same five random request orders. '+
            'Points are mean workload reuse ratios; bars span the observed minimum and maximum across orders, not a population CI. '+
            f'Method family: {fam}; split: {split}. Families partition the method registry for legibility; every actual method is retained. '+
            'The result measures source-module token mass, not GPU KV hits, prefill speed, or QA improvement.')


def quality_reuse(ex,summary,orders,split):
    reuse=orders[orders.split.eq(split)&orders.order.eq('random')&orders.policy.eq('flat_identity_lru')&orders.capacity.eq(64)].groupby('method').token_weighted_reused_fraction.mean()
    s=summary[summary.split.eq(split)].merge(reuse.rename('reuse_fraction'),on='method',how='inner')
    fig,ax=plt.subplots(figsize=(9,7.5));fig.subplots_adjust(left=.12,right=.96,bottom=.13,top=.82)
    handles=[]
    for fam,g in s.groupby(s.method.map(family)):
        ax.scatter(g.reuse_fraction,g.accepted_hit_at_10,s=60,c=FAMILY_COLORS[fam],marker=FAMILY_MARKERS[fam],alpha=.8,edgecolor='white',linewidth=.7)
        handles.append(Line2D([],[],marker=FAMILY_MARKERS[fam],color='none',markerfacecolor=FAMILY_COLORS[fam],markersize=7,label=fam))
    ours=s[s.method.eq(TARGET)]
    if len(ours):ax.scatter(ours.reuse_fraction,ours.accepted_hit_at_10,s=145,marker='*',c='#D55E00',edgecolor='#333333',linewidth=.6)
    handles.append(Line2D([],[],marker='*',color='none',markerfacecolor='#D55E00',markersize=11,label='IntentRAG balanced fair'))
    ax.set_xlim(0,1.02);ax.set_ylim(0,1.02);ax.xaxis.set_major_formatter(PercentFormatter(1));ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_xlabel('Source-token-mass reuse (64 modules)',labelpad=12);ax.set_ylabel('Accepted evidence Hit@10',labelpad=12)
    ax.grid(alpha=.15);axes_arrows(ax)
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.54,.985),ncol=2,fontsize=11,frameon=False,labelspacing=.7)
    ex.save(fig,'quality_reuse_'+split,s,
        'Retrieval support and source-module reuse are distinct measured axes.',
        'Each point is one executed method. Retrieval is the normalized-question-cluster accepted Hit@10; reuse is the mean across five request orders with 64 exact source modules. '+
        'All methods are retained; coincident marks may represent multiple methods, which remain individually listed in Source Data. '+
        'A high reuse fraction can accompany weak support; no QA quality or causality is inferred. Split: '+split+'.')


def reuse_mechanism_curve(ex,orders,split,word=False):
    comparators=[TARGET,'True BM25','Detail-only','Long-term fixed','Profile-resolved',
                 'Multi-granularity fixed','Multi-granularity entropy','Multi-granularity entropy+constraints']
    s=orders[orders.split.eq(split)&orders.policy.eq('flat_identity_lru')&orders.order.eq('random')&orders.method.isin(comparators)].copy()
    if s.empty:return
    assert s.capacity.gt(0).all()
    fig,ax=plt.subplots(figsize=(9 if word else 12,6.5 if word else 7.2));fig.subplots_adjust(left=.12 if word else .1,right=.55,bottom=.15 if word else .13,top=.95)
    for i,m in enumerate(comparators):
        part=s[s.method.eq(m)]
        if part.empty:continue
        g=part.groupby('capacity').token_weighted_reused_fraction.agg(['mean','min','max']).reset_index()
        color='#D55E00' if m==TARGET else PALETTE[(i+1)%len(PALETTE)]
        ax.errorbar(g.capacity,g['mean'],yerr=np.vstack([g['mean']-g['min'],g['max']-g['mean']]),
            marker=['o','s','^','D','v','P','X','h'][i],color=color,capsize=3,lw=1.6,
            label=word_display(m) if word else textwrap.fill(display(m),36),markersize=6)
    ax.set_xscale('log',base=2);ax.set_xticks([16,64,256],['16','64','256']);ax.set_xlim(12,330)
    ax.set_ylim(0,1.02);ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_xlabel('Capacity (exact source modules)',labelpad=12);ax.set_ylabel('Source-token-mass reuse fraction',labelpad=12)
    ax.grid(alpha=.15);axes_arrows(ax)
    ax.legend(loc='center left',bbox_to_anchor=(1.055 if word else 1.07,.5),fontsize=12 if word else 11,labelspacing=.8,frameon=False)
    ex.save(fig,'reuse_capacity_mechanistic_'+split+('_word' if word else ''),s,
        'Local, coarse, and multi-granularity states expose different finite-capacity reuse opportunities.',
        'This main-view comparison uses a declared set of local, long-only, multi-granularity and primary retrieval controls. '+
        'All remaining methods are retained in the full family capacity figures and forests. Mean and min–max range are over '+
        'the same five request orders. Modules retain exact source identity; neither this graph nor a higher reuse ratio establishes QA improvement. Split: '+split+'.')


def paired_granularity_mechanism(ex,d,split,bootstraps,word=False):
    baseline='Detail-only';metric='required_quote_support_fraction'
    methods=['Profile-resolved','Long-term fixed','Section-resolved','Short-term fixed',
             'Multi-granularity fixed','Multi-granularity entropy','Multi-granularity entropy+constraints',
             'Calibrated lexical control','IntentRAG calibrated multi-granularity',TARGET,'True BM25']
    part=d[d.split.eq(split)].copy()
    qcluster=part.drop_duplicates('question_id').set_index('question_id').cluster_id
    raw=[];summary=[];rng=np.random.default_rng(SEED+31)
    for method in methods:
        a=part[part.method.isin([baseline,method])].pivot(index='question_id',columns='method',values=metric)
        if baseline not in a or method not in a:continue
        defined=a[[baseline,method]].notna().all(axis=1)
        paired=a.loc[defined,[baseline,method]].copy()
        if paired.empty:continue
        paired['delta']=paired[method]-paired[baseline]
        paired['cluster_id']=qcluster.reindex(paired.index)
        values=paired.groupby('cluster_id').delta.mean().to_numpy()
        boot=values[rng.integers(0,len(values),size=(bootstraps,len(values)))].mean(axis=1)
        lo,hi=np.quantile(boot,[.025,.975])
        summary.append({'method':method,'baseline':baseline,'mean_paired_shift':values.mean(),
            'ci_low':lo,'ci_high':hi,'n_paired_ids':len(paired),'n_clusters':len(values),
            'n_undefined_pairs':int((~defined).sum()),'wins':int((paired.delta>0).sum()),
            'losses':int((paired.delta<0).sum()),'ties':int((paired.delta==0).sum())})
        for qid,r in paired.iterrows():raw.append({'question_id':qid,'cluster_id':r.cluster_id,
            'method':method,'baseline':baseline,'delta':r.delta})
    s=pd.DataFrame(summary)
    if s.empty:return
    fig,ax=plt.subplots(figsize=(9 if word else 11,6.5 if word else 7.2));fig.subplots_adjust(left=.43,right=.88 if word else .86,bottom=.16 if word else .14,top=.95)
    for i,r in s.iterrows():
        c='#D55E00' if r.method==TARGET else FAMILY_COLORS[family(r.method)]
        ax.errorbar(r.mean_paired_shift,i,xerr=[[r.mean_paired_shift-r.ci_low],[r.ci_high-r.mean_paired_shift]],
            fmt=FAMILY_MARKERS[family(r.method)],color=c,markersize=7,capsize=4,elinewidth=1.6)
        ax.text(1.055 if word else 1.06,i,f'{r.mean_paired_shift:+.3f}',transform=ax.get_yaxis_transform(),va='center',fontsize=12)
    ax.axvline(0,color='#666666',lw=1,ls='--')
    ax.set_yticks(range(len(s)),[word_display(m) if word else textwrap.fill(display(m),36) for m in s.method],fontsize=12 if word else 11.5)
    ax.set_ylim(len(s)-.4,-.6);ax.set_xlim(-1,1)
    ax.set_xticks([-1,-.5,0,.5,1]);ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.set_xlabel('Paired required-quote support shift\nrelative to detail-only',labelpad=12)
    ax.grid(axis='x',alpha=.15);axes_arrows(ax)
    pd.DataFrame(raw).to_csv(ex.source/'fresh_granularity_paired_raw_differences.csv',index=False,encoding='utf-8-sig')
    ex.save(fig,'fresh_granularity_paired_support_shift'+('_word' if word else ''),s,
        'Long and short evidence routes must be tested against the same local-source support requirements.',
        'Differences are paired on the same fresh queries and averaged within normalized-question clusters. '+
        'Bars are 95% percentile intervals from 2,000 paired cluster-bootstrap draws. Detail-only is the common reference. '+
        'Negative differences and the calibrated controls remain visible. This tests whether each route retains required source quotes; '+
        'it does not assume that entropy routing or multi-granularity always improves support.')


def entropy_support_diagnostic(ex,d,split,bootstraps):
    g=d[d.split.eq(split)&d.method.eq(TARGET)].copy()
    if 'entropy_detail' not in g or g.entropy_detail.notna().sum()==0:return
    # Real observed query clusters are bootstrapped; no synthetic values added.
    valid=g.entropy_detail.notna()&g.accepted_hit_at_10.notna()
    grouped=g.loc[valid].groupby('cluster_id')[['entropy_detail','accepted_hit_at_10']].mean()
    edges=np.linspace(0,1,11);rng=np.random.default_rng(SEED+53);rows=[]
    for i,(lo,hi) in enumerate(zip(edges[:-1],edges[1:])):
        mask=grouped.entropy_detail.ge(lo)&(grouped.entropy_detail.le(hi) if i==9 else grouped.entropy_detail.lt(hi))
        part=grouped.loc[mask];a=part.accepted_hit_at_10.to_numpy()
        if len(a):
            draws=a[rng.integers(0,len(a),size=(bootstraps,len(a)))].mean(axis=1)
            low,high=np.quantile(draws,[.025,.975]);mean=a.mean();center=part.entropy_detail.mean()
        else:low=high=mean=center=np.nan
        rows.append({'bin_low':lo,'bin_high':hi,'bin_center':(lo+hi)/2,'observed_entropy_mean':center,
            'accepted_hit_mean':mean,'ci_low':low,'ci_high':high,'n_question_clusters':len(a),
            'n_missing_query_rows':int((~valid).sum())})
    s=pd.DataFrame(rows);fig,ax=plt.subplots(figsize=(8.2,6.5))
    fig.subplots_adjust(left=.14,right=.94,bottom=.16,top=.94)
    # NaN bins remain gaps: lines never bridge unobserved entropy ranges.
    ax.plot(s.bin_center,s.accepted_hit_mean,color='#007F78',lw=1.5)
    populated=s.n_question_clusters.gt(0)
    z=s.loc[populated]
    ax.errorbar(z.bin_center,z.accepted_hit_mean,yerr=np.vstack([z.accepted_hit_mean-z.ci_low,z.ci_high-z.accepted_hit_mean]),
                fmt='none',color='#007F78',capsize=4,elinewidth=1.5)
    ax.scatter(z.bin_center,z.accepted_hit_mean,s=24+z.n_question_clusters*3,
               color='#007F78',edgecolor='white',linewidth=.8)
    ax.set_xlim(0,1);ax.set_ylim(0,1.05);ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_xlabel('Normalized detail-score entropy',labelpad=12)
    ax.set_ylabel('Observed accepted evidence Hit@10',labelpad=12)
    ax.grid(alpha=.15);axes_arrows(ax)
    ex.save(fig,'entropy_support_'+split,s,
        'Score concentration and actual evidence support are separate empirical quantities.',
        'Queries are grouped into fixed 0.1-wide normalized entropy bins after retrieval. Points are mean support across question-text clusters; '+
        'bars are 95% cluster-bootstrap intervals and marker area increases with cluster count. Empty bins remain gaps. '+
        'Entropy is not a calibrated correctness probability, safety confidence, or mutual information. Primary method: IntentRAG balanced fair; split: '+split+'.')


def main():
    ap=argparse.ArgumentParser();exp=Path(__file__).resolve().parents[1]
    ap.add_argument('--input-root',type=Path,required=True)
    ap.add_argument('--query-results',type=Path)
    ap.add_argument('--memory-dir',type=Path,required=True)
    ap.add_argument('--corpus',type=Path,required=True)
    ap.add_argument('--tokenizer-json',type=Path,required=True)
    ap.add_argument('--out',type=Path,default=exp/'figures_core')
    ap.add_argument('--bootstrap-samples',type=int,default=2000)
    ap.add_argument('--word-only',action='store_true',help='Add large-text Word exports while keeping complete original exports')
    ap.add_argument('--skill-scripts',type=Path,default=Path('${NATURE_FIGURE_SKILL}'))
    args=ap.parse_args()
    mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
        'font.size':12,'axes.labelsize':13,'xtick.labelsize':12,'ytick.labelsize':11.5,
        'legend.fontsize':11,'svg.fonttype':'none','pdf.fonttype':42,'ps.fonttype':42,
        'axes.linewidth':1.1,'legend.frameon':False})
    query=args.query_results or args.input_root/'results/all_retrieval_query_results.csv'
    if not query.exists():query=args.input_root/'results/retrieval_query_results.csv'
    d=pd.read_csv(query)
    if d.duplicated(['question_id','method']).any():raise ValueError('Duplicate query-method observations')
    canonical=json.loads((args.input_root/'data/canonical_questions.json').read_text(encoding='utf-8'))
    expected=pd.Series([q['split'] for q in canonical]).value_counts().to_dict()
    counts=d.groupby(['split','method']).size()
    missing=[(split,method,int(n),expected[split]) for (split,method),n in counts.items() if n!=expected[split]]
    if missing:raise ValueError('Incomplete methods must not be plotted as matched-budget final results: '+str(missing))
    ex=Exporter(args.out,args.skill_scripts)
    metrics=['accepted_hit_at_10','exact_hit_at_10','accepted_set_recall_at_10','required_quote_support_fraction']
    summary=make_summary(d,metrics,args.bootstrap_samples)
    summary.to_csv(ex.source/'all_cluster_bootstrap_summary.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame([{'method':m,'display_alias':word_display(m),'method_family':family(m)} for m in sorted(d.method.unique())]).to_csv(ex.source/'method_display_aliases.csv',index=False,encoding='utf-8-sig')
    old='v21_locked_blind_test';fresh='fresh_locked_test'
    for split,short in [(old,'legacy_test'),(fresh,'fresh_test')]:
        suffix='_word' if args.word_only else ''
        forest(ex,summary,split,'accepted_hit_at_10',short+'_accepted_hit_forest'+suffix,'Accepted evidence Hit@10',args.word_only)
        forest(ex,summary,split,'exact_hit_at_10',short+'_exact_hit_forest'+suffix,'Exact evidence Hit@10',args.word_only)
    forest(ex,summary,fresh,'required_quote_support_fraction','fresh_required_quote_support_forest'+suffix,'Required-quote support fraction',args.word_only)
    corpus=pd.read_parquet(args.corpus).fillna('')
    for split,short in [(old,'legacy_test'),(fresh,'fresh_test')]:
        granularity_composition(ex,d,canonical,corpus,args.tokenizer_json,split,False,short+'_returned_granularity'+suffix,args.word_only)
        granularity_composition(ex,d,canonical,corpus,args.tokenizer_json,split,True,short+'_accepted_support_granularity'+suffix,args.word_only)
    orders=read_order_summary(args.memory_dir)
    orders.to_csv(ex.source/'replay_order_summary_for_figures.csv',index=False,encoding='utf-8-sig')
    for split in [old,fresh]:
        if not args.word_only:reuse_curves(ex,orders,split);quality_reuse(ex,summary,orders,split)
        reuse_mechanism_curve(ex,orders,split,args.word_only)
    paired_granularity_mechanism(ex,d,fresh,args.bootstrap_samples,args.word_only)
    if not args.word_only:
        for split in [old,fresh]:entropy_support_diagnostic(ex,d,split,args.bootstrap_samples)
    catalog=ex.catalog
    previous=args.out/'figure_manifest.json'
    if args.word_only and previous.exists():
        stored=json.loads(previous.read_text(encoding='utf-8')).get('figures',[])
        current={f['figure']:f for f in stored}
        current.update({f['figure']:f for f in catalog});catalog=list(current.values())
    manifest={'status':'rendered_pending_visual_qa','backend':'Python','figures':catalog,
        'inputs':[{'path':str(p),'sha256':sha(p)} for p in [query,args.input_root/'data/canonical_questions.json',args.corpus,args.tokenizer_json]],
        'n_methods':d.method.nunique(),'n_query_ids':d.question_id.nunique(),
        'all_methods_in_forest':True,'bootstrap_samples':args.bootstrap_samples,'seed':SEED,
        'journal_contract':'Provisional general scientific figures; no target-journal compliance claimed',
        'all_single_axes':all(f['n_axes']==1 for f in catalog),'all_no_image_titles':True,
        'source_granularity':'Pooled selected source-text token mass; accepted IDs used post hoc only',
        'reuse_boundary':'Exact source identity modules. No GPU KV or downstream QA gain measurement'}
    (args.out/'figure_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    notes='\n\n'.join(f"{f['figure']}\n{f['claim']}\n{f['caption_en']}" for f in catalog)
    (args.out/'figure_explanations_en.txt').write_text(notes,encoding='utf-8')
    print(json.dumps({'new_exports':len(ex.catalog),'total_figures':len(catalog),'methods':manifest['n_methods'],'out':str(args.out)}))


if __name__=='__main__':main()
