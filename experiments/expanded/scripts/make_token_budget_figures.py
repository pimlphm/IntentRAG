"""Two standalone empirical figures for equal source-body token-budget replay."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from make_expanded_figures import (Exporter, FAMILY_COLORS, FAMILY_MARKERS,
    PALETTE, TARGET, axes_arrows, family, sha, word_display)


class TokenExporter(Exporter):
    """Keep export formats explicit for this standalone script's source audit."""
    def save(self, fig, stem, data, claim, caption):
        base = self.out/stem
        fig.canvas.draw()
        self.alignment(fig, json_out=str(self.qa/(stem+".alignment.json")),
            overlay_svg=str(self.qa/(stem+".alignment.svg")), strict=True)
        fig.savefig(str(base)+".svg", facecolor="white", bbox_inches=None)
        fig.savefig(str(base)+".pdf", facecolor="white", bbox_inches=None)
        fig.savefig(str(base)+".png", dpi=600, facecolor="white", bbox_inches=None)
        data.to_csv(self.source/(stem+".csv"), index=False, encoding="utf-8-sig")
        self.catalog.append({"figure": stem, "claim": claim, "caption_en": caption,
            "source_data": "source_data/"+stem+".csv", "n_axes": len(fig.axes),
            "size_inches": fig.get_size_inches().tolist(), "title_in_image": False})
        plt.close(fig)


def main():
    exp = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=exp/"figures_token_budget")
    ap.add_argument("--orders", type=Path, default=exp/"results/memory_replay_token_budget/replay_order_summary.csv")
    ap.add_argument("--retrieval-summary", type=Path, default=exp/"figures/source_data/all_cluster_bootstrap_summary.csv")
    ap.add_argument("--skill-scripts", type=Path, default=Path("${NATURE_FIGURE_SKILL}"))
    args = ap.parse_args()
    mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 12, "axes.labelsize": 13, "xtick.labelsize": 12, "ytick.labelsize": 12,
        "legend.fontsize": 12, "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.linewidth": 1.1, "legend.frameon": False})
    orders = pd.read_csv(args.orders)
    q = pd.read_csv(args.retrieval_summary)
    split = "fresh_locked_test"
    observed = orders[orders.split.eq(split) & orders.order.eq("random")].copy()
    assert observed.capacity_source_tokens.gt(0).all()
    assert observed.max_cache_tokens_after.le(observed.capacity_source_tokens).all()
    counts = observed.groupby(["method", "capacity_source_tokens"]).size()
    assert counts.eq(5).all(), "Every method and budget must have the same five workload orders"
    ex = TokenExporter(args.out, args.skill_scripts)

    reuse = observed[observed.capacity_source_tokens.eq(4096)].groupby("method").agg(
        reuse_fraction=("token_weighted_reused_fraction", "mean"),
        reuse_order_min=("token_weighted_reused_fraction", "min"),
        reuse_order_max=("token_weighted_reused_fraction", "max"),
        oversize_module_fraction=("oversize_module_request_fraction", "mean"),
        oversize_token_fraction=("oversize_token_mass_fraction", "mean"),
        mean_cache_tokens=("mean_cache_tokens_after", "mean"),
        n_random_orders=("seed", "count")).reset_index()
    s = q[q.split.eq(split)].merge(reuse, on="method", how="outer", validate="one_to_one")
    assert len(s) == 33 and s.accepted_hit_at_10.notna().all() and s.reuse_fraction.notna().all()
    s["method_family"] = s.method.map(family)
    s["display_alias"] = s.method.map(word_display)
    fig, ax = plt.subplots(figsize=(9, 7.2))
    fig.subplots_adjust(left=.13, right=.96, bottom=.14, top=.81)
    handles = []
    for fam, part in s.groupby("method_family"):
        ax.scatter(part.reuse_fraction, part.accepted_hit_at_10, s=70,
            color=FAMILY_COLORS[fam], marker=FAMILY_MARKERS[fam], alpha=.78,
            edgecolor="white", linewidth=.7)
        handles.append(Line2D([], [], marker=FAMILY_MARKERS[fam], color="none",
            markerfacecolor=FAMILY_COLORS[fam], markersize=8, label=fam))
    primary = s[s.method.eq(TARGET)]
    ax.scatter(primary.reuse_fraction, primary.accepted_hit_at_10, s=165,
        marker="*", color="#D55E00", edgecolor="#333333", linewidth=.6, zorder=5)
    handles.append(Line2D([], [], marker="*", color="none", markerfacecolor="#D55E00",
        markersize=12, label="IntentRAG balanced"))
    xmax = max(.12, min(1.02, s.reuse_fraction.max()+.05))
    ax.set_xlim(-.025*xmax, xmax)
    ax.set_ylim(-.025, 1.025)
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_xlabel("Source-body token reuse (4,096-token budget)", labelpad=12)
    ax.set_ylabel("Accepted evidence Hit@10", labelpad=12)
    ax.grid(alpha=.15)
    axes_arrows(ax)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.54, .99),
        ncol=2, fontsize=12, frameon=False, labelspacing=.65)
    ex.save(fig, "fresh_quality_reuse_4096_source_tokens", s,
        "Support and reusable token mass remain distinct under equal source-body token capacity.",
        "Each point is one of all 33 executed methods on the fresh source-derived test queries. "
        "The vertical axis is accepted evidence Hit@10 averaged over normalized question-text clusters; "
        "the horizontal axis is mean token-mass reuse across the same five random past-only LRU request orders. "
        "All methods use a 4,096-token full-source-body capacity; items larger than the capacity are not cached. "
        "Coincident methods remain separately listed in Source Data. Identity keys and management overhead are excluded. "
        "This is not RAM fairness, actual GPU KV reuse, authorization validity or downstream QA accuracy.")

    comparators = [TARGET, "True BM25", "Detail-only", "Long-term fixed", "Profile-resolved",
        "Multi-granularity fixed", "Multi-granularity entropy", "Multi-granularity entropy+constraints"]
    s = observed[observed.method.isin(comparators)].copy()
    fig, ax = plt.subplots(figsize=(9, 6.5))
    fig.subplots_adjust(left=.12, right=.55, bottom=.17, top=.94)
    for i, m in enumerate(comparators):
        part = s[s.method.eq(m)]
        g = part.groupby("capacity_source_tokens").token_weighted_reused_fraction.agg(["mean", "min", "max"]).reset_index()
        ax.errorbar(g.capacity_source_tokens, g["mean"],
            yerr=np.vstack([g["mean"]-g["min"], g["max"]-g["mean"]]),
            marker=["o", "s", "^", "D", "v", "P", "X", "h"][i],
            color="#D55E00" if m == TARGET else PALETTE[(i+1) % len(PALETTE)],
            markersize=6, capsize=3, linewidth=1.6, label=word_display(m))
    ax.set_xscale("log", base=2)
    ax.set_xticks([1024, 4096, 16384], ["1,024", "4,096", "16,384"])
    ax.set_xlim(800, 21500)
    ax.set_ylim(-.02, 1.02)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_xlabel("Capacity (full-source-body tokens)", labelpad=12)
    ax.set_ylabel("Source-token-mass reuse fraction", labelpad=12)
    ax.grid(alpha=.15)
    axes_arrows(ax)
    ax.legend(loc="center left", bbox_to_anchor=(1.06, .5), fontsize=12,
        labelspacing=.8, frameon=False)
    ex.save(fig, "fresh_reuse_capacity_source_tokens", s,
        "Finite-capacity reuse depends on full source-body storage cost as well as granularity.",
        "The declared local, coarse and multi-granularity controls use identical hard capacities "
        "of 1,024, 4,096 and 16,384 full-source-body tokenizer tokens. Points are mean workload reuse "
        "fractions; bars span observed minimum to maximum over the same five random orders, not population "
        "confidence intervals. Every query looks up only past cached exact identities before updating. "
        "Oversize items are excluded from cache but retained in the frozen retrieval output. "
        "The displayed mechanism subset complements the all-33-method scatter and complete result tables. "
        "Keys, management and index overhead are not counted; no GPU KV-byte or prefill result is claimed.")
    manifest = {"status": "rendered_pending_visual_qa", "backend": "Python",
        "figures": ex.catalog, "n_methods_in_all_method_scatter": len(reuse), "split": split,
        "inputs": [{"path": str(p), "sha256": sha(p)} for p in [args.orders, args.retrieval_summary]],
        "all_no_image_titles": True, "all_single_axes": True,
        "journal_contract": "Provisional standalone empirical scientific plots; no exact journal compliance claimed",
        "budget_boundary": "Same full-source-body tokenizer-token cap, not total RAM or GPU KV bytes"}
    (args.out/"figure_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (args.out/"figure_explanations_en.txt").write_text("\n\n".join(
        f"{f['figure']}\n{f['claim']}\n{f['caption_en']}" for f in ex.catalog), encoding="utf-8")
    print(json.dumps({"figures": len(ex.catalog), "out": str(args.out)}))


if __name__ == "__main__":
    main()
