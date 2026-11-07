"""Independent title-free computational-reuse / source-visibility figures.

All six query-pair means are shown. Observed values are never downsampled or
simulated. EID presence and reference anchors are explicitly support proxies.
"""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
from make_expanded_figures import axes_arrows

BASE=Path(__file__).resolve().parents[1]
OUT=BASE/'figures_qa_kv'
QA=OUT/'qa'
SKILL=Path(r'${NATURE_FIGURE_SKILL}')
sys.path.insert(0,str(SKILL))
from audit_panel_alignment import require_matplotlib_panel_alignment


def save(fig,stem,data,claim,caption,catalog):
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,json_out=str(QA/(stem+'.alignment.json')),
        overlay_svg=str(QA/(stem+'.alignment.svg')),strict=True)
    fig.savefig(OUT/(stem+'.svg'),bbox_inches=None,facecolor='white')
    fig.savefig(OUT/(stem+'.pdf'),bbox_inches=None,facecolor='white')
    fig.savefig(OUT/(stem+'.png'),dpi=600,bbox_inches=None,facecolor='white')
    data.to_csv(OUT/'source_data'/(stem+'.csv'),index=False,encoding='utf-8-sig')
    catalog.append({'figure':stem,'claim':claim,'caption_en':caption,'n_axes':1,
                    'title_in_image':False,'source_data':'source_data/'+stem+'.csv'})
    plt.close(fig)


def main():
    mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
        'font.size':14,'axes.labelsize':15,'xtick.labelsize':13,'ytick.labelsize':13,
        'legend.fontsize':12,'svg.fonttype':'none','pdf.fonttype':42})
    OUT.mkdir(exist_ok=True);QA.mkdir(exist_ok=True);(OUT/'source_data').mkdir(exist_ok=True)
    d=pd.read_csv(BASE/'results/real_kv/prefix_materialization_pair_means.csv')
    c=pd.read_csv(BASE/'results/real_kv/prefix_materialization_paired_pair_means.csv')
    assert len(d)==18 and len(c)==12 and d.pair_index.nunique()==6
    assert d[['actual_cache_fraction','emitted_reference_anchor_coverage_proxy']].notna().all().all()
    layouts=['query_first','canonical_state_first','hierarchical_state_first']
    names=['Query first','Canonical state first','Hierarchy metadata first']
    colors=['#555555','#356EA5','#007F78'];markers=['o','s','^']
    aircraft_handles=[Line2D([],[],marker='o',ls='',color='#444444',markerfacecolor='#444444',label='Boeing 737'),
                      Line2D([],[],marker='o',ls='',color='#444444',markerfacecolor='white',label='Cessna 172')]
    catalog=[]
    fig,ax=plt.subplots(figsize=(9,7.6));fig.subplots_adjust(left=.16,right=.95,bottom=.16,top=.76)
    handles=[]
    for layout,label,color,marker in zip(layouts,names,colors,markers):
        for aircraft,g in d[d.namespace.eq(layout)].groupby('aircraft_model'):
            ax.scatter(g.actual_cache_fraction,g.emitted_reference_anchor_coverage_proxy,s=85,marker=marker,
                facecolors=color if aircraft=='Boeing 737' else 'white',edgecolors=color,lw=1.5,zorder=3)
        handles.append(Line2D([],[],marker=marker,ls='',color=color,label=label,markersize=8))
    fig.legend(handles=handles+aircraft_handles,loc='upper center',bbox_to_anchor=(.55,.985),ncol=2,
        frameon=False,columnspacing=1.8,handletextpad=.65)
    upper=max(.30,float(d.actual_cache_fraction.max())*1.12)
    ax.set_xlim(0,upper);ax.set_ylim(-.035,1.025)
    ax.set_xlabel('Actual reused input-token fraction',labelpad=12)
    ax.set_ylabel('Visible reference-anchor fraction (proxy)',labelpad=12)
    ax.xaxis.set_major_formatter(PercentFormatter(1));ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(alpha=.15);axes_arrows(ax)
    save(fig,'actual_kv_vs_materialized_reference_support',d,
        'More computational reuse need not preserve the same materialized query-reference text.',
        'Each mark is the mean of all 33 methods within one of six fixed visible-query pairs; all 18 layout-by-pair means are shown. '+
        'Filled marks denote Boeing 737 and hollow marks Cessna 172; marker shape and colour identify layout. '+
        'The horizontal axis uses actual native-engine cached-input tokens divided by the full target prompt count, including common headers. '+
        'The vertical axis is post-hoc coverage of eligible long reference anchors in actual emitted source bodies, not in the upstream Top-10 list. '+
        'English anchors contain six lexical words; Chinese-dominant anchors contain 24 lexical units including at least 12 CJK characters. '+
        'NFKC/case/spacing/punctuation-normalized matching is performed inside individual source blocks; no cross-block concatenation matches are allowed. '+
        'These six correlated source-derived pairs do not establish QA correctness, action authority or statistical superiority. No fit or causal relation is inferred.',catalog)
    fig,ax=plt.subplots(figsize=(9,7.6));fig.subplots_adjust(left=.17,right=.95,bottom=.16,top=.78)
    contrast_specs=[('query_first','canonical_state_first','Canonical minus query first','#356EA5','s'),
                    ('canonical_state_first','hierarchical_state_first','Hierarchy minus canonical','#007F78','^')]
    handles=[]
    for baseline,candidate,label,color,marker in contrast_specs:
        for aircraft,g in c[c.baseline.eq(baseline)&c.candidate.eq(candidate)].groupby('aircraft_model'):
            ax.scatter(g.cache_fraction_gain,g.reference_anchor_support_loss_proxy,s=90,marker=marker,
                facecolors=color if aircraft=='Boeing 737' else 'white',edgecolors=color,lw=1.5,zorder=4)
        handles.append(Line2D([],[],marker=marker,ls='',color=color,label=label,markersize=8))
    fig.legend(handles=handles+aircraft_handles,loc='upper center',bbox_to_anchor=(.55,.98),ncol=2,
        frameon=False,columnspacing=1.5,handletextpad=.6)
    xmin=min(-.03,float(c.cache_fraction_gain.min())-.025);xmax=max(.03,float(c.cache_fraction_gain.max())+.025)
    ymin=min(-.03,float(c.reference_anchor_support_loss_proxy.min())-.025)
    ymax=max(.06,float(c.reference_anchor_support_loss_proxy.max())+.045)
    ax.set_xlim(xmin,xmax);ax.set_ylim(ymin,ymax)
    ax.axvline(0,ls='--',lw=1,color='#999999');ax.axhline(0,ls='--',lw=1,color='#999999')
    ax.set_xlabel('Change in reused input-token fraction',labelpad=12)
    ax.set_ylabel('Loss of reference-anchor coverage (proxy)',labelpad=12)
    ax.xaxis.set_major_formatter(PercentFormatter(1));ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(alpha=.15);axes_arrows(ax)
    save(fig,'paired_cache_gain_vs_materialized_support_loss',c,
        'Canonical ordering can trade visible reference support for cached-prefix length; early metadata can reduce cache without changing source bodies.',
        'Each mark is a paired layout contrast averaged over the same 33 methods within one of the six visible-query pairs; all 12 contrast-by-pair means are shown. '+
        'Positive horizontal values denote an increase in actual cached-input fraction, while positive vertical values denote a decrease in emitted reference-anchor coverage. '+
        'Filled marks denote Boeing 737 and hollow marks Cessna 172. '+
        'All 198 method-by-pair canonical and hierarchical source-region SHA values are identical; thus their zero vertical differences are audited equality, not assumed evidence quality. '+
        'The contrast changes canonical ordering and placement together, so it does not identify an isolated causal effect of query position. '+
        'Source-budget truncation and the position of query-dependent routing metadata explain the illustrated materialization mechanisms; neither proxy measures semantic admissibility or full-answer correctness. '+
        'No regression, significance test, purity estimate or independent 594-question claim is made.',catalog)
    (OUT/'figure_manifest_materialization.json').write_text(json.dumps({'backend':'Python','title_free':True,
        'n_profiles_in_source_audit':594,'n_pair_units':6,'n_method_profiles_per_layout':198,
        'figures':catalog,'boundary':'Post-hoc emitted support proxies, not QA or admissibility'},ensure_ascii=False,indent=2),encoding='utf-8')
    print('MATERIALIZATION FIGURES',len(catalog),flush=True)


if __name__=='__main__':main()
