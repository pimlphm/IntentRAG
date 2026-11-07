from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


THIS = Path(__file__).resolve()
WORK_ROOT = THIS.parents[1]
IN_RESULTS = WORK_ROOT / "results" / "long_short_reuse_v2_strict" / "long_short_reuse_v2_query_seed_results.csv"
OUT_TABLES = WORK_ROOT / "tables" / "long_short_reuse_v2_strict"
OUT_FIGURES = WORK_ROOT / "figures" / "long_short_reuse_v2_strict"
OUT_REPORTS = WORK_ROOT / "reports"

POLICY_LABELS = {
    "fresh_only": "Fresh only",
    "naive_semantic_reuse": "Naive reuse",
    "fixed_50_50": "Fixed 50/50",
    "entropy_adaptive": "Entropy adaptive",
    "admissibility_signature_gated_entropy": "Gated entropy",
    "component_document_signature_gated_entropy": "Strict comp-doc gate",
}

POLICY_ORDER = [
    "fresh_only",
    "naive_semantic_reuse",
    "fixed_50_50",
    "entropy_adaptive",
    "admissibility_signature_gated_entropy",
    "component_document_signature_gated_entropy",
]

COLORS = {
    "fresh_only": "#3b6fb6",
    "naive_semantic_reuse": "#b34d4d",
    "fixed_50_50": "#8a6fbd",
    "entropy_adaptive": "#d08b32",
    "admissibility_signature_gated_entropy": "#3d8f63",
    "component_document_signature_gated_entropy": "#2b6f6f",
}

MARKERS = {
    "fresh_only": "o",
    "naive_semantic_reuse": "X",
    "fixed_50_50": "s",
    "entropy_adaptive": "^",
    "admissibility_signature_gated_entropy": "D",
    "component_document_signature_gated_entropy": "o",
}


def ensure_dirs() -> None:
    for path in [OUT_TABLES, OUT_FIGURES, OUT_REPORTS]:
        path.mkdir(parents=True, exist_ok=True)


def bootstrap_ci(values: np.ndarray, seed: int = 20260816, n_boot: int = 5000) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    boots = np.array([rng.choice(values, size=len(values), replace=True).mean() for _ in range(n_boot)])
    return float(values.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def build_aircraft_summary(records: pd.DataFrame) -> pd.DataFrame:
    seed_means = (
        records.groupby(["aircraft_model", "seed", "policy"], sort=True)[
            ["accepted_hit_10", "actual_reuse_slots", "fresh_slots"]
        ]
        .mean()
        .reset_index()
    )
    rows = []
    for (aircraft, policy), group in seed_means.groupby(["aircraft_model", "policy"], sort=True):
        row = {
            "aircraft_model": aircraft,
            "policy": policy,
            "policy_label": POLICY_LABELS.get(policy, policy),
            "n_seeds": int(group["seed"].nunique()),
            "n_query_seed_rows": int(len(records[(records["aircraft_model"].eq(aircraft)) & (records["policy"].eq(policy))])),
        }
        for metric in ["accepted_hit_10", "actual_reuse_slots", "fresh_slots"]:
            mean, low, high = bootstrap_ci(group[metric].to_numpy(), seed=20260816 + len(rows) + len(metric))
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci95_low"] = low
            row[f"{metric}_ci95_high"] = high
        rows.append(row)
    out = pd.DataFrame(rows)
    out["policy_order"] = out["policy"].map({p: i for i, p in enumerate(POLICY_ORDER)})
    return out.sort_values(["aircraft_model", "policy_order"]).drop(columns=["policy_order"])


def build_overall_source(records: pd.DataFrame) -> pd.DataFrame:
    seed_means = (
        records.groupby(["seed", "policy"], sort=True)[
            ["accepted_hit_10", "actual_reuse_slots", "fresh_slots", "page_hit_10", "section_hit_10"]
        ]
        .mean()
        .reset_index()
    )
    rows = []
    for policy, group in seed_means.groupby("policy", sort=True):
        row = {"policy": policy, "policy_label": POLICY_LABELS.get(policy, policy)}
        for metric in ["accepted_hit_10", "actual_reuse_slots", "fresh_slots", "page_hit_10", "section_hit_10"]:
            mean, low, high = bootstrap_ci(group[metric].to_numpy(), seed=20260901 + len(rows) + len(metric))
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci95_low"] = low
            row[f"{metric}_ci95_high"] = high
        rows.append(row)
    out = pd.DataFrame(rows)
    out["policy_order"] = out["policy"].map({p: i for i, p in enumerate(POLICY_ORDER)})
    return out.sort_values("policy_order").drop(columns=["policy_order"])


def style_axes(ax: plt.Axes) -> None:
    ax.tick_params(axis="both", labelsize=16)
    ax.grid(True, axis="both", color="#d9d9d9", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(1.1)


def add_axis_arrowheads(ax: plt.Axes) -> None:
    arrowprops = dict(arrowstyle="-|>", color="#4a4a4a", lw=1.25, mutation_scale=14, shrinkA=0, shrinkB=0)
    ax.annotate(
        "",
        xy=(1.018, 0.0),
        xytext=(0.985, 0.0),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops=arrowprops,
        annotation_clip=False,
    )
    ax.annotate(
        "",
        xy=(0.0, 1.018),
        xytext=(0.0, 0.985),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops=arrowprops,
        annotation_clip=False,
    )


def plot_tradeoff(source: pd.DataFrame) -> dict[str, str]:
    fig, ax = plt.subplots(figsize=(12.4, 7.0))
    for _, row in source.iterrows():
        policy = row["policy"]
        x = row["actual_reuse_slots_mean"]
        y = row["accepted_hit_10_mean"]
        # Plot every policy at its measured coordinates.  The two gated
        # policies are intentionally almost coincident; a larger hollow
        # marker distinguishes the strict policy without visually moving the
        # observation or its confidence interval.
        display_x = x
        display_y = y
        markerfacecolor = "white" if policy == "component_document_signature_gated_entropy" else COLORS[policy]
        ax.errorbar(
            display_x,
            display_y,
            xerr=np.array([[x - row["actual_reuse_slots_ci95_low"]], [row["actual_reuse_slots_ci95_high"] - x]]),
            yerr=np.array([[y - row["accepted_hit_10_ci95_low"]], [row["accepted_hit_10_ci95_high"] - y]]),
            fmt=MARKERS[policy],
            color=COLORS[policy],
            markerfacecolor=markerfacecolor,
            markeredgecolor=COLORS[policy],
            markeredgewidth=2.0 if policy == "component_document_signature_gated_entropy" else 1.4,
            markersize=17 if policy == "component_document_signature_gated_entropy" else 12,
            capsize=4,
            linewidth=1.8,
            label=row["policy_label"],
            zorder=4 if policy == "component_document_signature_gated_entropy" else 3,
        )
    label_offsets = {
        "fresh_only": (0.22, 0.016),
        "naive_semantic_reuse": (-2.55, 0.013),
        "fixed_50_50": (0.22, -0.010),
        "entropy_adaptive": (0.60, -0.004),
        "admissibility_signature_gated_entropy": (0.20, 0.028),
        "component_document_signature_gated_entropy": (0.0, 0.0),
    }
    for _, row in source.iterrows():
        dx, dy = label_offsets[row["policy"]]
        if row["policy"] == "component_document_signature_gated_entropy":
            # The strict and broader gates have the same rounded support
            # value and nearly identical reuse.  One shared numeric label
            # avoids overprinting while the concentric markers retain both
            # observations at their true coordinates.
            continue
        base_x = row["actual_reuse_slots_mean"]
        base_y = row["accepted_hit_10_mean"]
        label = f"{row['accepted_hit_10_mean']:.3f}"
        if row["policy"] == "admissibility_signature_gated_entropy":
            label += " (both gates)"
        ax.text(base_x + dx, base_y + dy, label, fontsize=15)
    ax.set_xlabel("Mean reused long-state slots out of 10", fontsize=18)
    ax.set_ylabel("Accepted hit@10", fontsize=18)
    ax.set_xlim(-0.55, 10.55)
    ax.set_ylim(0.27, 0.845)
    style_axes(ax)
    add_axis_arrowheads(ax)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=13)
    fig.subplots_adjust(left=0.105, right=0.70, bottom=0.17, top=0.955)
    png = OUT_FIGURES / "figure_a_v2_reuse_slots_accepted_tradeoff.png"
    pdf = OUT_FIGURES / "figure_a_v2_reuse_slots_accepted_tradeoff.pdf"
    fig.savefig(png, dpi=350)
    fig.savefig(pdf)
    plt.close(fig)
    return {"png": str(png), "pdf": str(pdf)}


def plot_aircraft_gated(aircraft_summary: pd.DataFrame) -> dict[str, str]:
    keep = aircraft_summary[
        aircraft_summary["policy"].isin([
            "admissibility_signature_gated_entropy",
            "component_document_signature_gated_entropy",
        ])
    ].copy()
    keep["reuse_retention_rate"] = keep["actual_reuse_slots_mean"] / 10.0
    keep["reuse_retention_ci95_low"] = keep["actual_reuse_slots_ci95_low"] / 10.0
    keep["reuse_retention_ci95_high"] = keep["actual_reuse_slots_ci95_high"] / 10.0
    source_path = OUT_TABLES / "figure_b_v2_aircraft_gated_reuse_accepted_source.csv"
    keep.to_csv(source_path, index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(10.8, 6.5))
    aircraft = ["Boeing 737", "Cessna 172"]
    x = np.arange(len(aircraft))
    width = 0.18
    series = [
        ("admissibility_signature_gated_entropy", "reuse_retention_rate", "Gated reuse", "#6aa376", -1.5),
        ("component_document_signature_gated_entropy", "reuse_retention_rate", "Strict reuse", "#2b6f6f", -0.5),
        ("admissibility_signature_gated_entropy", "accepted_hit_10_mean", "Gated accepted", "#4f79b8", 0.5),
        ("component_document_signature_gated_entropy", "accepted_hit_10_mean", "Strict accepted", "#8a6fbd", 1.5),
    ]
    for policy, metric, label, color, offset in series:
        sub = keep[keep["policy"].eq(policy)].set_index("aircraft_model").loc[aircraft]
        vals = sub[metric].to_numpy(dtype=float)
        if metric == "reuse_retention_rate":
            low = sub["reuse_retention_ci95_low"].to_numpy(dtype=float)
            high = sub["reuse_retention_ci95_high"].to_numpy(dtype=float)
        else:
            low = sub["accepted_hit_10_ci95_low"].to_numpy(dtype=float)
            high = sub["accepted_hit_10_ci95_high"].to_numpy(dtype=float)
        bars = ax.bar(
            x + offset * width,
            vals,
            width,
            yerr=np.vstack([vals - low, high - vals]),
            capsize=4,
            color=color,
            label=label,
            linewidth=0,
        )
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.018,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=13,
                rotation=0,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(aircraft, fontsize=17)
    ax.set_ylabel("Rate", fontsize=18)
    ax.set_ylim(0, 0.94)
    style_axes(ax)
    add_axis_arrowheads(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=2, frameon=False, fontsize=14)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.14, top=0.82)
    png = OUT_FIGURES / "figure_b_v2_aircraft_gated_reuse_accepted.png"
    pdf = OUT_FIGURES / "figure_b_v2_aircraft_gated_reuse_accepted.pdf"
    fig.savefig(png, dpi=350)
    fig.savefig(pdf)
    plt.close(fig)
    return {"png": str(png), "pdf": str(pdf), "source_csv": str(source_path)}


def main() -> None:
    ensure_dirs()
    records = pd.read_csv(IN_RESULTS).fillna("")
    aircraft_summary = build_aircraft_summary(records)
    overall_source = build_overall_source(records)
    aircraft_summary_path = OUT_TABLES / "long_short_reuse_v2_aircraft_policy_summary_ci.csv"
    tradeoff_source_path = OUT_TABLES / "figure_a_v2_reuse_slots_accepted_tradeoff_source.csv"
    aircraft_summary.to_csv(aircraft_summary_path, index=False, encoding="utf-8-sig")
    overall_source.to_csv(tradeoff_source_path, index=False, encoding="utf-8-sig")
    fig_a = plot_tradeoff(overall_source)
    fig_b = plot_aircraft_gated(aircraft_summary)

    def rel(path: str | Path) -> str:
        return Path(path).resolve().relative_to(WORK_ROOT.resolve()).as_posix()

    manifest = {
        "created_by": "make_long_short_reuse_figures_v2_strict.py",
        "work_root": "work/multigranularity_temporal_v1",
        "input": rel(IN_RESULTS),
        "boundary": "Offline long/short-state reuse simulator over existing cue-free IntentRAG packages; not a real KV-cache hit measurement, not live retrieval, not aircraft maintenance correctness.",
        "new_policy": "component_document_signature_gated_entropy",
        "new_policy_rule": "Existing aircraft/authority/condition/warning gate plus non-empty component_family compatibility with current fresh package; for Cessna source-document compatibility is required. No gold/track/query_type/required_slots used.",
        "aircraft_policy_summary": rel(aircraft_summary_path),
        "figures": {
            "figure_a": {
                "description": "Overall reused long-state slots versus accepted hit@10 trade-off across six policies.",
                "display_note": "All policies are drawn at measured coordinates. The strict comp-doc gate uses a larger hollow marker around the nearly coincident broader-gate point; the shared label reports their identical rounded accepted hit@10.",
                "png": rel(fig_a["png"]),
                "pdf": rel(fig_a["pdf"]),
                "source_csv": rel(tradeoff_source_path),
            },
            "figure_b": {
                "description": "Boeing vs Cessna reuse retention and accepted hit@10 for gated entropy and strict component-document gated entropy.",
                "png": rel(fig_b["png"]),
                "pdf": rel(fig_b["pdf"]),
                "source_csv": rel(fig_b["source_csv"]),
            },
        },
        "qa_constraints": [
            "single-panel independent figures",
            "no title inside figure",
            "large fonts",
            "legend outside or in blank space",
            "numeric labels offset from marks",
        ],
    }
    manifest_path = OUT_REPORTS / "long_short_reuse_v2_figure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "ok", "manifest": str(manifest_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
