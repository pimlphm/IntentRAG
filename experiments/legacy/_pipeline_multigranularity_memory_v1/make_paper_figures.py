"""
Build single-panel paper figures for multigranularity_memory_v1.

The script reads only real CSV outputs produced by
run_multigranularity_evidence_memory.py. It does not fabricate placeholder
data. If required CSVs are not present yet, it exits with a clear pending
message so it can be rerun after the pipeline finishes.

Generated figures:
  1. locked-test method comparison
  2. short/long memory weight allocation
  3. per-query entropy-only vs admissibility-assembly delta / negative result

Each figure is a standalone single plot with no in-figure title.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "04_results" / "multigranularity_memory_v1"
TABLES = ROOT / "06_tables" / "multigranularity_memory_v1"
FIGURES = ROOT / "05_figures" / "multigranularity_memory_v1"
REPORTS = ROOT / "07_reports" / "multigranularity_memory_v1"
CONFIG = ROOT / "configs" / "multigranularity_memory_v1" / "multigranularity_memory_v1.json"
PIPELINE_SCRIPT = ROOT / "_pipeline_multigranularity_memory_v1" / "run_multigranularity_evidence_memory.py"

QUERY_RESULTS = RESULTS / "query_results.csv"
ROUTER_WEIGHTS = RESULTS / "router_weights.csv"
SUMMARY = RESULTS / "summary.csv"
PAIRED_BOOTSTRAP = RESULTS / "paired_bootstrap.csv"


METHOD_LABELS = {
    "flat_detail_hybrid": "Flat detail",
    "fixed_multigranularity": "Fixed multi-grain",
    "entropy_multigranularity": "Entropy-only",
    "entropy_admissibility_assembly": "Documentary assembly",
    "existing_aesr_v2_balanced_route": "AESR-v2 balanced",
    "existing_cue_free_intentrag_reference": "Existing reference",
    "page_only": "Page",
    "section_only": "Section",
    "profile_only": "Profile",
    "short_memory_fixed": "Short pair",
    "long_memory_fixed": "Long pair",
}

METHOD_ORDER = [
    "flat_detail_hybrid",
    "fixed_multigranularity",
    "entropy_multigranularity",
    "entropy_admissibility_assembly",
    "existing_cue_free_intentrag_reference",
    "existing_aesr_v2_balanced_route",
]

GRANULARITY_METHODS = [
    ("flat_detail_hybrid", "Detail"),
    ("page_only", "Page"),
    ("section_only", "Section"),
    ("profile_only", "Profile"),
    ("short_memory_fixed", "Short pair"),
    ("long_memory_fixed", "Long pair"),
]


@dataclass
class FigureRecord:
    figure_id: str
    path_png: Path
    path_pdf: Path
    source_files: list[Path]
    source_rows: int
    description: str


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require_csv(path: Path, columns: set[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Pending pipeline output: {path}")
    frame = pd.read_csv(path)
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    if frame.empty:
        raise ValueError(f"{path} exists but has no rows")
    return frame


def pretty_method(method: str) -> str:
    return METHOD_LABELS.get(method, method.replace("_", " "))


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 16,
            "axes.labelsize": 18,
            "xtick.labelsize": 14,
            "ytick.labelsize": 15,
            "legend.fontsize": 13,
            "figure.dpi": 140,
            "savefig.dpi": 320,
            "axes.linewidth": 1.2,
            "lines.linewidth": 2.5,
            "patch.linewidth": 1.0,
        }
    )


def add_axis_arrows(ax: plt.Axes) -> None:
    """Add compact axis arrowheads outside the data rectangle."""
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)
    ax.annotate(
        "",
        xy=(1.012, 0),
        xytext=(0, 0),
        xycoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", color="black", lw=1.8, mutation_scale=14),
        annotation_clip=False,
    )
    ax.annotate(
        "",
        xy=(0, 1.018),
        xytext=(0, 0),
        xycoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", color="black", lw=1.8, mutation_scale=14),
        annotation_clip=False,
    )


def save_single_figure(fig: plt.Figure, stem: str) -> tuple[Path, Path]:
    FIGURES.mkdir(parents=True, exist_ok=True)
    png = FIGURES / f"{stem}.png"
    pdf = FIGURES / f"{stem}.pdf"
    fig.savefig(png, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png, pdf


def locked_test_method_comparison(summary: pd.DataFrame) -> FigureRecord:
    required = {
        "evaluation_split",
        "aircraft_model",
        "method",
        "offline_accepted_hit_10_mean",
        "offline_exact_hit_10_mean",
        "offline_page_hit_10_mean",
        "offline_section_hit_10_mean",
    }
    missing = sorted(required - set(summary.columns))
    if missing:
        raise ValueError(f"summary.csv missing columns for method comparison: {missing}")

    data = summary[
        summary["evaluation_split"].eq("locked_blind_test")
        & summary["aircraft_model"].eq("All")
    ].copy()
    data = data[data["method"].isin(METHOD_ORDER)].copy()
    if data.empty:
        raise ValueError("summary.csv has no locked_blind_test / All rows")
    data["method_rank"] = data["method"].map({m: i for i, m in enumerate(METHOD_ORDER)}).fillna(999)
    data = data.sort_values(["method_rank", "method"])

    metrics = [
        ("offline_accepted_hit_10_mean", "Accepted"),
        ("offline_page_hit_10_mean", "Page"),
        ("offline_section_hit_10_mean", "Section"),
    ]
    exact_available = data["offline_exact_hit_10_mean"].notna().any()
    if exact_available:
        metrics.insert(1, ("offline_exact_hit_10_mean", "Exact"))

    x = np.arange(len(data))
    width = min(0.18, 0.72 / len(metrics))
    fig, ax = plt.subplots(figsize=(max(9.5, 1.25 * len(data) + 4), 6.2))
    colors = ["#4C78A8", "#F58518", "#54A24B", "#B279A2"]
    for idx, (column, label) in enumerate(metrics):
        offset = (idx - (len(metrics) - 1) / 2) * width
        ax.bar(x + offset, data[column].astype(float), width=width, label=f"{label} hit@10", color=colors[idx])

    ax.set_ylabel("Mean locked post-hoc hit@10")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels([pretty_method(m) for m in data["method"]], rotation=18, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    add_axis_arrows(ax)
    fig.subplots_adjust(left=0.08, right=0.78, bottom=0.24, top=0.96)
    png, pdf = save_single_figure(fig, "fig_mgm_locked_test_method_comparison")
    return FigureRecord(
        "fig_mgm_locked_test_method_comparison",
        png,
        pdf,
        [SUMMARY],
        int(len(data)),
        "Locked-blind method comparison using real summary.csv rows. Caption note: locked labels are used post hoc only on synthetic source-derived queries.",
    )


def short_long_weight_allocation(router: pd.DataFrame) -> FigureRecord:
    required = {
        "evaluation_split",
        "aircraft_model",
        "detail_weight",
        "page_weight",
        "section_weight",
        "profile_weight",
        "short_term_weight",
        "long_term_weight",
    }
    missing = sorted(required - set(router.columns))
    if missing:
        raise ValueError(f"router_weights.csv missing columns for weight allocation: {missing}")

    data = router[router["evaluation_split"].eq("locked_blind_test")].copy()
    if data.empty:
        raise ValueError("router_weights.csv has no locked_blind_test rows")
    group = data.groupby("aircraft_model", dropna=False)[
        ["detail_weight", "page_weight", "section_weight", "profile_weight", "short_term_weight", "long_term_weight"]
    ].mean()
    order = sorted(group.index.astype(str))
    group = group.loc[order]

    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    y = np.arange(len(group))
    left = np.zeros(len(group))
    components = [
        ("detail_weight", "Detail", "#4C78A8"),
        ("page_weight", "Page", "#72B7B2"),
        ("section_weight", "Section", "#F58518"),
        ("profile_weight", "Profile", "#E45756"),
    ]
    for column, label, color in components:
        values = group[column].astype(float).to_numpy()
        ax.barh(y, values, left=left, height=0.58, label=label, color=color)
        left += values

    ax.set_xlabel("Mean entropy-router weight")
    ax.set_yticks(y)
    ax.set_yticklabels(order)
    ax.set_xlim(0, 1.03)
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    for idx, (_, row) in enumerate(group.iterrows()):
        ax.text(
            1.01,
            idx,
            f"short {row['short_term_weight']:.2f} / long {row['long_term_weight']:.2f}",
            va="center",
            ha="left",
            fontsize=12,
            clip_on=False,
        )
    add_axis_arrows(ax)
    png, pdf = save_single_figure(fig, "fig_mgm_short_long_weight_allocation")
    return FigureRecord(
        "fig_mgm_short_long_weight_allocation",
        png,
        pdf,
        [ROUTER_WEIGHTS],
        int(len(data)),
        "Short/detail-page and long/section-profile entropy-router weight allocation.",
    )


def entropy_vs_admissibility_delta(query_results: pd.DataFrame) -> FigureRecord:
    required = {
        "question_id",
        "evaluation_split",
        "aircraft_model",
        "method",
        "offline_accepted_hit_10",
        "package_authority_present",
        "package_condition_present",
        "package_action_present",
        "package_warning_present",
    }
    missing = sorted(required - set(query_results.columns))
    if missing:
        raise ValueError(f"query_results.csv missing columns for per-query delta: {missing}")

    data = query_results[
        query_results["evaluation_split"].eq("locked_blind_test")
        & query_results["method"].isin(["entropy_multigranularity", "entropy_admissibility_assembly"])
    ].copy()
    if data.empty:
        raise ValueError("query_results.csv has no entropy/admissibility locked_blind_test rows")
    pivot = data.pivot_table(
        index=["question_id", "aircraft_model"],
        columns="method",
        values=[
            "offline_accepted_hit_10",
            "package_authority_present",
            "package_condition_present",
            "package_action_present",
            "package_warning_present",
        ],
        aggfunc="first",
    )
    needed = [
        ("offline_accepted_hit_10", "entropy_multigranularity"),
        ("offline_accepted_hit_10", "entropy_admissibility_assembly"),
    ]
    if any(col not in pivot.columns for col in needed):
        raise ValueError("query_results.csv lacks paired entropy/admissibility rows for some queries")
    diff = pd.DataFrame(index=pivot.index).reset_index()
    for metric in [
        "offline_accepted_hit_10",
        "package_authority_present",
        "package_condition_present",
        "package_action_present",
        "package_warning_present",
    ]:
        diff[f"{metric}_entropy"] = pivot[(metric, "entropy_multigranularity")].to_numpy()
        diff[f"{metric}_admissibility"] = pivot[(metric, "entropy_admissibility_assembly")].to_numpy()
        diff[f"{metric}_delta"] = diff[f"{metric}_admissibility"] - diff[f"{metric}_entropy"]
    TABLES.mkdir(parents=True, exist_ok=True)
    delta_csv = TABLES / "entropy_vs_admissibility_query_delta.csv"
    diff.to_csv(delta_csv, index=False)

    diff = diff.sort_values(["offline_accepted_hit_10_delta", "aircraft_model", "question_id"]).reset_index(drop=True)
    x = np.arange(len(diff))
    y = diff["offline_accepted_hit_10_delta"].astype(float).to_numpy()
    colors = np.where(y < 0, "#D62728", np.where(y > 0, "#2CA02C", "#9E9E9E"))

    fig, ax = plt.subplots(figsize=(12.5, 6.2))
    ax.axhline(0, color="#333333", linewidth=1.2, zorder=1)
    ax.scatter(x, y, s=36, c=colors, alpha=0.88, edgecolor="white", linewidth=0.4, zorder=3)
    # Use a light rolling mean only as visual guide; it is not a fitted claim.
    if len(y) >= 9:
        window = max(9, int(round(len(y) * 0.06)))
        rolling = pd.Series(y).rolling(window=window, min_periods=1, center=True).mean().to_numpy()
        ax.plot(x, rolling, color="#4C78A8", linewidth=2.0, alpha=0.8, label="Rolling mean")
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    ax.set_xlabel("Locked-blind queries sorted by accepted-hit delta", labelpad=10)
    ax.set_ylabel("Accepted-hit@10 Δ", labelpad=12)
    ax.set_ylim(-1.15, 1.15)
    ax.set_yticks([-1, 0, 1])
    ax.grid(axis="y", alpha=0.25)

    neg = int((y < 0).sum())
    zero = int((y == 0).sum())
    pos = int((y > 0).sum())
    ax.text(
        0.02,
        0.96,
        f"accepted-hit@10 delta: +{pos} / 0 {zero} / −{neg}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=13,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.92, pad=4),
    )
    ax.margins(x=0.025)
    add_axis_arrows(ax)
    fig.subplots_adjust(left=0.10, right=0.82, bottom=0.18, top=0.96)
    png, pdf = save_single_figure(fig, "fig_mgm_entropy_vs_admissibility_query_delta")
    return FigureRecord(
        "fig_mgm_entropy_vs_admissibility_query_delta",
        png,
        pdf,
        [QUERY_RESULTS, delta_csv],
        int(len(diff)),
        "Per-query locked-blind accepted-hit difference (admissibility assembly minus entropy-only); negative values are preserved as negative results.",
    )


def conceptual_mechanism_figure() -> FigureRecord:
    """Draw a single conceptual mechanism figure with no title.

    This figure is intentionally not data-measured. It explains the intended
    evidence-before-language flow and explicitly marks KV reuse as not measured.
    """

    setup_style()
    # A compact aspect ratio keeps the text legible after insertion at one
    # manuscript-column width.
    fig, ax = plt.subplots(figsize=(12.4, 7.2))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    def box(
        xy: tuple[float, float],
        wh: tuple[float, float],
        text: str,
        face: str,
        edge: str = "#2F3B45",
        fontsize: int = 17,
        dashed: bool = False,
    ) -> tuple[float, float, float, float]:
        x, y = xy
        w, h = wh
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.018,rounding_size=0.025",
            facecolor=face,
            edgecolor=edge,
            linewidth=1.7,
            linestyle="--" if dashed else "-",
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, color="#1F2933")
        return x, y, w, h

    def arrow(start: tuple[float, float], end: tuple[float, float], color: str = "#455A64", curved: float = 0.0) -> None:
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=18,
                linewidth=2.1,
                color=color,
                connectionstyle=f"arc3,rad={curved}",
                shrinkA=6,
                shrinkB=6,
            )
        )

    q = box((0.035, 0.43), (0.12, 0.16), "visible\nquery", "#E8F1FF")
    router = box((0.205, 0.43), (0.145, 0.16), "entropy\nrouter", "#EAF7EA")
    short = box((0.405, 0.62), (0.19, 0.14), "short-lived\nDetail / Page", "#FFF4D6")
    long = box((0.405, 0.28), (0.19, 0.14), "persistent\nSection / Profile", "#FCE7E7")
    sig = box(
        (0.655, 0.39),
        (0.19, 0.24),
        "proposed\ncompatibility gate\nscope · condition\nauthority · provenance\nrevision proxy (future)",
        "#EFE8FF",
        fontsize=15,
    )
    state = box((0.875, 0.55), (0.105, 0.14), "bounded\nevidence\nstate", "#E6FFFA", fontsize=15)
    llm = box((0.875, 0.25), (0.105, 0.14), "LLM\nverbalizer", "#F3F4F6", fontsize=15)
    kv = box(
        (0.64, 0.075),
        (0.23, 0.11),
        "conceptual interface;\nlayerwise KV reuse not measured",
        "#FFFFFF",
        edge="#9AA5B1",
        fontsize=14,
        dashed=True,
    )

    arrow((q[0] + q[2], q[1] + q[3] / 2), (router[0], router[1] + router[3] / 2))
    arrow((router[0] + router[2], router[1] + router[3] * 0.70), (short[0], short[1] + short[3] / 2))
    arrow((router[0] + router[2], router[1] + router[3] * 0.30), (long[0], long[1] + long[3] / 2))
    arrow((short[0] + short[2], short[1] + short[3] / 2), (sig[0], sig[1] + sig[3] * 0.72), curved=-0.08)
    arrow((long[0] + long[2], long[1] + long[3] / 2), (sig[0], sig[1] + sig[3] * 0.28), curved=0.08)
    arrow((sig[0] + sig[2], sig[1] + sig[3] * 0.67), (state[0], state[1] + state[3] / 2))
    arrow((state[0] + state[2] / 2, state[1]), (llm[0] + llm[2] / 2, llm[1] + llm[3]))
    arrow((sig[0] + sig[2] * 0.55, sig[1]), (kv[0] + kv[2] * 0.52, kv[1] + kv[3]), color="#9AA5B1", curved=0.18)

    fig.subplots_adjust(left=0.02, right=0.985, bottom=0.04, top=0.98)
    png, pdf = save_single_figure(fig, "fig_mgm_conceptual_mechanism")
    return FigureRecord(
        "fig_mgm_conceptual_mechanism",
        png,
        pdf,
        [CONFIG, PIPELINE_SCRIPT],
        1,
        "Conceptual mechanism figure; not a measured result and not a KV-cache experiment.",
    )


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int = 5000) -> tuple[float, float, float]:
    clean = values[np.isfinite(values)].astype(float)
    if len(clean) == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(clean.mean())
    if len(clean) == 1:
        return mean, mean, mean
    samples = rng.choice(clean, size=(n_boot, len(clean)), replace=True).mean(axis=1)
    low, high = np.percentile(samples, [2.5, 97.5])
    return mean, float(low), float(high)


def granularity_ablation(query_results: pd.DataFrame) -> FigureRecord:
    required = {
        "question_id",
        "evaluation_split",
        "aircraft_model",
        "method",
        "offline_accepted_hit_10",
    }
    missing = sorted(required - set(query_results.columns))
    if missing:
        raise ValueError(f"query_results.csv missing columns for granularity ablation: {missing}")

    locked = query_results[
        query_results["evaluation_split"].eq("locked_blind_test")
        & query_results["method"].isin([method for method, _ in GRANULARITY_METHODS])
    ].copy()
    if locked.empty:
        raise ValueError("query_results.csv has no locked_blind_test rows for granularity ablations")

    rng = np.random.default_rng(20260816)
    rows: list[dict] = []
    groups = [("All", locked)] + [
        (aircraft, frame)
        for aircraft, frame in locked.groupby("aircraft_model", dropna=False)
    ]
    for group_name, frame in groups:
        for method, label in GRANULARITY_METHODS:
            vals = frame.loc[frame["method"].eq(method), "offline_accepted_hit_10"].astype(float).to_numpy()
            if len(vals) == 0:
                continue
            mean, low, high = bootstrap_mean_ci(vals, rng)
            rows.append(
                {
                    "aircraft_model": str(group_name),
                    "method": method,
                    "label": label,
                    "n_queries": int(len(vals)),
                    "accepted_hit_10_mean": mean,
                    "ci_low": low,
                    "ci_high": high,
                }
            )
    table = pd.DataFrame(rows)
    TABLES.mkdir(parents=True, exist_ok=True)
    ablation_csv = TABLES / "granularity_ablation_accepted_ci.csv"
    table.to_csv(ablation_csv, index=False)

    desired_groups = ["All", "Boeing 737", "Cessna 172"]
    present_groups = [g for g in desired_groups if g in set(table["aircraft_model"])]
    present_groups += [g for g in sorted(set(table["aircraft_model"])) if g not in present_groups]
    x = np.arange(len(GRANULARITY_METHODS))
    fig, ax = plt.subplots(figsize=(10.8, 6.4))
    colors = {
        "All": "#1F77B4",
        "Boeing 737": "#FF7F0E",
        "Cessna 172": "#2CA02C",
    }
    markers = {"All": "o", "Boeing 737": "s", "Cessna 172": "^"}
    offsets = np.linspace(-0.12, 0.12, max(1, len(present_groups)))
    for offset, group_name in zip(offsets, present_groups):
        group = table[table["aircraft_model"].eq(group_name)].set_index("method")
        means, lows, highs = [], [], []
        for method, _label in GRANULARITY_METHODS:
            row = group.loc[method]
            means.append(float(row["accepted_hit_10_mean"]))
            lows.append(float(row["accepted_hit_10_mean"] - row["ci_low"]))
            highs.append(float(row["ci_high"] - row["accepted_hit_10_mean"]))
        xpos = x + offset
        ax.errorbar(
            xpos,
            means,
            yerr=np.vstack([lows, highs]),
            marker=markers.get(group_name, "o"),
            markersize=8,
            linewidth=2.3,
            capsize=4,
            label=group_name,
            color=colors.get(group_name, None),
            zorder=3,
        )
    ax.set_ylabel("Accepted hit@10")
    ax.set_xlabel("Granularity / memory view")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _method, label in GRANULARITY_METHODS], rotation=16, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    ax.margins(x=0.04)
    add_axis_arrows(ax)
    fig.subplots_adjust(left=0.10, right=0.80, bottom=0.20, top=0.96)
    png, pdf = save_single_figure(fig, "fig_mgm_granularity_ablation")
    return FigureRecord(
        "fig_mgm_granularity_ablation",
        png,
        pdf,
        [QUERY_RESULTS, ablation_csv],
        int(len(locked)),
        "Locked-blind granularity ablation: accepted@10 means with 95% bootstrap confidence intervals for All/Boeing/Cessna.",
    )


def write_manifest(records: list[FigureRecord]) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    manifest = {
        "figure_set": "multigranularity_memory_v1_paper_figures",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve().relative_to(ROOT)),
        "rule_summary": {
            "single_panel": True,
            "no_in_figure_titles": True,
            "real_csv_only": True,
            "docx_modified": False,
        },
        "figures": [],
    }
    rows = []
    for rec in records:
        sources = [
            {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in rec.source_files
            if path.exists()
        ]
        outputs = [
            {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in [rec.path_png, rec.path_pdf]
        ]
        item = {
            "figure_id": rec.figure_id,
            "description": rec.description,
            "source_rows": rec.source_rows,
            "sources": sources,
            "outputs": outputs,
        }
        manifest["figures"].append(item)
        rows.append(
            {
                "figure_id": rec.figure_id,
                "description": rec.description,
                "source_rows": rec.source_rows,
                "png": str(rec.path_png.relative_to(ROOT)),
                "pdf": str(rec.path_pdf.relative_to(ROOT)),
                "source_files": ";".join(str(p.relative_to(ROOT)) for p in rec.source_files),
            }
        )
    (REPORTS / "paper_figure_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    pd.DataFrame(rows).to_csv(TABLES / "paper_figure_manifest.csv", index=False)


def build_all(conceptual_only: bool = False) -> list[FigureRecord]:
    setup_style()
    records = [conceptual_mechanism_figure()]
    if conceptual_only:
        write_manifest(records)
        return records
    summary = require_csv(
        SUMMARY,
        {
            "evaluation_split",
            "aircraft_model",
            "method",
            "offline_accepted_hit_10_mean",
        },
    )
    router = require_csv(
        ROUTER_WEIGHTS,
        {
            "evaluation_split",
            "aircraft_model",
            "detail_weight",
            "page_weight",
            "section_weight",
            "profile_weight",
            "short_term_weight",
            "long_term_weight",
        },
    )
    query_results = require_csv(
        QUERY_RESULTS,
        {
            "question_id",
            "evaluation_split",
            "aircraft_model",
            "method",
            "offline_accepted_hit_10",
        },
    )
    records.extend(
        [
        locked_test_method_comparison(summary),
        short_long_weight_allocation(router),
        entropy_vs_admissibility_delta(query_results),
        granularity_ablation(query_results),
        ]
    )
    write_manifest(records)
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="print machine-readable status")
    parser.add_argument("--conceptual-only", action="store_true", help="build only the conceptual mechanism figure")
    args = parser.parse_args()
    try:
        records = build_all(conceptual_only=args.conceptual_only)
    except FileNotFoundError as exc:
        status = {"status": "pending", "message": str(exc)}
        if args.json:
            print(json.dumps(status, ensure_ascii=False, indent=2))
        else:
            print(f"PENDING: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        status = {"status": "failed", "message": str(exc)}
        if args.json:
            print(json.dumps(status, ensure_ascii=False, indent=2))
        else:
            print(f"FAILED: {exc}", file=sys.stderr)
        return 1
    status = {
        "status": "ok",
        "figures": [str(rec.path_png.relative_to(ROOT)) for rec in records],
        "manifest": str((REPORTS / "paper_figure_manifest.json").relative_to(ROOT)),
    }
    if args.json:
        print(json.dumps(status, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(status, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
