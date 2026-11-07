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
IN_RESULTS = WORK_ROOT / "results" / "long_short_reuse" / "long_short_reuse_query_seed_results.csv"
OUT_TABLES = WORK_ROOT / "tables" / "long_short_reuse"
OUT_FIGURES = WORK_ROOT / "figures" / "long_short_reuse"
OUT_REPORTS = WORK_ROOT / "reports"

POLICY_LABELS = {
    "fresh_only": "Fresh only",
    "naive_semantic_reuse": "Naive reuse",
    "fixed_50_50": "Fixed 50/50",
    "entropy_adaptive": "Entropy adaptive",
    "admissibility_signature_gated_entropy": "Gated entropy",
}

POLICY_ORDER = [
    "fresh_only",
    "naive_semantic_reuse",
    "fixed_50_50",
    "entropy_adaptive",
    "admissibility_signature_gated_entropy",
]

COLORS = {
    "fresh_only": "#3b6fb6",
    "naive_semantic_reuse": "#b34d4d",
    "fixed_50_50": "#8a6fbd",
    "entropy_adaptive": "#d08b32",
    "admissibility_signature_gated_entropy": "#3d8f63",
}

MARKERS = {
    "fresh_only": "o",
    "naive_semantic_reuse": "X",
    "fixed_50_50": "s",
    "entropy_adaptive": "^",
    "admissibility_signature_gated_entropy": "D",
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


def plot_tradeoff(source: pd.DataFrame) -> dict[str, str]:
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    for _, row in source.iterrows():
        policy = row["policy"]
        x = row["actual_reuse_slots_mean"]
        y = row["accepted_hit_10_mean"]
        xerr = np.array([[x - row["actual_reuse_slots_ci95_low"]], [row["actual_reuse_slots_ci95_high"] - x]])
        yerr = np.array([[y - row["accepted_hit_10_ci95_low"]], [row["accepted_hit_10_ci95_high"] - y]])
        ax.errorbar(
            x,
            y,
            xerr=xerr,
            yerr=yerr,
            fmt=MARKERS[policy],
            color=COLORS[policy],
            markersize=12,
            capsize=4,
            linewidth=1.8,
            label=row["policy_label"],
        )
    label_offsets = {
        "fresh_only": (0.18, 0.004),
        "naive_semantic_reuse": (-2.4, 0.012),
        "fixed_50_50": (0.22, -0.010),
        "entropy_adaptive": (0.18, -0.018),
        "admissibility_signature_gated_entropy": (0.18, 0.010),
    }
    for _, row in source.iterrows():
        dx, dy = label_offsets[row["policy"]]
        ax.text(
            row["actual_reuse_slots_mean"] + dx,
            row["accepted_hit_10_mean"] + dy,
            f"{row['accepted_hit_10_mean']:.3f}",
            fontsize=15,
            ha="left",
            va="center",
        )
    ax.set_xlabel("Mean reused long-state slots out of 10", fontsize=18)
    ax.set_ylabel("Accepted hit@10", fontsize=18)
    ax.set_xlim(-0.4, 10.5)
    ax.set_ylim(0.28, 0.825)
    style_axes(ax)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=15)
    fig.tight_layout(rect=[0, 0, 0.77, 1])
    png = OUT_FIGURES / "figure_a_reuse_slots_accepted_tradeoff.png"
    pdf = OUT_FIGURES / "figure_a_reuse_slots_accepted_tradeoff.pdf"
    fig.savefig(png, dpi=350)
    fig.savefig(pdf)
    plt.close(fig)
    return {"png": str(png), "pdf": str(pdf)}


def plot_aircraft_gated(aircraft_summary: pd.DataFrame) -> dict[str, str]:
    gated = aircraft_summary[aircraft_summary["policy"].eq("admissibility_signature_gated_entropy")].copy()
    gated["reuse_retention_rate"] = gated["actual_reuse_slots_mean"] / 10.0
    gated["reuse_retention_ci95_low"] = gated["actual_reuse_slots_ci95_low"] / 10.0
    gated["reuse_retention_ci95_high"] = gated["actual_reuse_slots_ci95_high"] / 10.0
    gated = gated.sort_values("aircraft_model")
    source_path = OUT_TABLES / "figure_b_aircraft_gated_reuse_accepted_source.csv"
    gated[
        [
            "aircraft_model",
            "reuse_retention_rate",
            "reuse_retention_ci95_low",
            "reuse_retention_ci95_high",
            "accepted_hit_10_mean",
            "accepted_hit_10_ci95_low",
            "accepted_hit_10_ci95_high",
            "actual_reuse_slots_mean",
            "fresh_slots_mean",
        ]
    ].to_csv(source_path, index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(7.8, 6.2))
    x = np.arange(len(gated))
    width = 0.34
    reuse = gated["reuse_retention_rate"].to_numpy()
    acc = gated["accepted_hit_10_mean"].to_numpy()
    reuse_err = np.vstack(
        [
            reuse - gated["reuse_retention_ci95_low"].to_numpy(),
            gated["reuse_retention_ci95_high"].to_numpy() - reuse,
        ]
    )
    acc_err = np.vstack(
        [
            acc - gated["accepted_hit_10_ci95_low"].to_numpy(),
            gated["accepted_hit_10_ci95_high"].to_numpy() - acc,
        ]
    )
    bars1 = ax.bar(
        x - width / 2,
        reuse,
        width,
        yerr=reuse_err,
        capsize=4,
        color="#6aa376",
        label="Reuse retention",
    )
    bars2 = ax.bar(
        x + width / 2,
        acc,
        width,
        yerr=acc_err,
        capsize=4,
        color="#4f79b8",
        label="Accepted hit@10",
    )
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.025,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=15,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(gated["aircraft_model"], fontsize=17)
    ax.set_ylabel("Rate", fontsize=18)
    ax.set_ylim(0, 0.92)
    style_axes(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=2, frameon=False, fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    png = OUT_FIGURES / "figure_b_aircraft_gated_reuse_accepted.png"
    pdf = OUT_FIGURES / "figure_b_aircraft_gated_reuse_accepted.pdf"
    fig.savefig(png, dpi=350)
    fig.savefig(pdf)
    plt.close(fig)
    return {"png": str(png), "pdf": str(pdf), "source_csv": str(source_path)}


def main() -> None:
    ensure_dirs()
    records = pd.read_csv(IN_RESULTS).fillna("")
    aircraft_summary = build_aircraft_summary(records)
    overall_source = build_overall_source(records)

    aircraft_summary_path = OUT_TABLES / "long_short_reuse_aircraft_policy_summary_ci.csv"
    tradeoff_source_path = OUT_TABLES / "figure_a_reuse_slots_accepted_tradeoff_source.csv"
    aircraft_summary.to_csv(aircraft_summary_path, index=False, encoding="utf-8-sig")
    overall_source.to_csv(tradeoff_source_path, index=False, encoding="utf-8-sig")

    fig_a = plot_tradeoff(overall_source)
    fig_b = plot_aircraft_gated(aircraft_summary)

    def rel(path: str | Path) -> str:
        return Path(path).resolve().relative_to(WORK_ROOT.resolve()).as_posix()

    manifest = {
        "created_by": "make_long_short_reuse_figures.py",
        "work_root": "work/multigranularity_temporal_v1",
        "input": rel(IN_RESULTS),
        "boundary": "Offline long/short-state reuse simulator over existing cue-free IntentRAG packages; not a real KV-cache hit measurement, not live retrieval, not aircraft maintenance correctness.",
        "aircraft_policy_summary": rel(aircraft_summary_path),
        "figures": {
            "figure_a": {
                "description": "Overall reused long-state slots versus accepted hit@10 trade-off across five policies.",
                "png": rel(fig_a["png"]),
                "pdf": rel(fig_a["pdf"]),
                "source_csv": rel(tradeoff_source_path),
            },
            "figure_b": {
                "description": "Boeing vs Cessna reuse retention and accepted hit@10 for admissibility-signature-gated entropy.",
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
    manifest_path = OUT_REPORTS / "long_short_reuse_figure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "ok", "manifest": str(manifest_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
