from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from textwrap import fill


METHODS = [
    "BM25/word",
    "TFIDF-char",
    "SVD semantic surrogate",
    "Hybrid lexical",
    "Parent-section retrieval",
    "Component-profile retrieval",
    "Controlled GraphRAG",
    "Causal-graph informed RAG",
    "Component-graph informed RAG",
    "Agentic RAG",
    "Evidence-flow baseline",
    "GaRAGe-style relevance RAG",
    "GaRAGe-style attribution RAG",
    "GaRAGe-style deflective RAG",
    "FD-GRAG-RC-V21-final",
]

DISPLAY = {
    "BM25/word": "BM25 lexical",
    "TFIDF-char": "TF-IDF character",
    "SVD semantic surrogate": "SVD semantic surrogate",
    "Hybrid lexical": "Hybrid lexical-semantic",
    "Parent-section retrieval": "Parent-section retrieval",
    "Component-profile retrieval": "Component-profile retrieval",
    "Controlled GraphRAG": "Controlled GraphRAG",
    "Causal-graph informed RAG": "Causal-graph informed RAG",
    "Component-graph informed RAG": "Component-graph informed RAG",
    "Agentic RAG": "Agentic RAG",
    "Evidence-flow baseline": "Evidence-flow baseline",
    "GaRAGe-style relevance RAG": "GaRAGe-style relevance RAG",
    "GaRAGe-style attribution RAG": "GaRAGe-style attribution RAG",
    "GaRAGe-style deflective RAG": "GaRAGe-style deflective RAG",
    "FD-GRAG-RC-V21-final": "FD-GRAG-RC-V21",
}

METHOD_ID = {m: f"M{i + 1}" for i, m in enumerate(METHODS)}

SHORT_NAME = {
    "BM25/word": "BM25",
    "TFIDF-char": "Char TF-IDF",
    "SVD semantic surrogate": "SVD surrogate",
    "Hybrid lexical": "Hybrid",
    "Parent-section retrieval": "Parent-section",
    "Component-profile retrieval": "Component-profile",
    "Controlled GraphRAG": "GraphRAG",
    "Causal-graph informed RAG": "Causal-graph",
    "Component-graph informed RAG": "Component-graph",
    "Agentic RAG": "Agentic RAG",
    "Evidence-flow baseline": "Evidence-flow",
    "GaRAGe-style relevance RAG": "GaRAGe-rel",
    "GaRAGe-style attribution RAG": "GaRAGe-attr",
    "GaRAGe-style deflective RAG": "GaRAGe-deflect",
    "FD-GRAG-RC-V21-final": "FD-GRAG-RC",
}

METRICS = [
    "exact_recall_10",
    "accepted_recall_10",
    "evidence_package_recall_10",
    "grounded_score",
    "hallucination_risk_proxy",
]

METRIC_LABEL = {
    "exact_recall_10": "Exact R@10",
    "accepted_recall_10": "Accepted R@10",
    "evidence_package_recall_10": "Package R@10",
    "accepted_page_recall_10": "Page R@10",
    "accepted_section_recall_10": "Section R@10",
    "grounded_score": "Grounded",
    "hallucination_risk_proxy": "Risk",
}

PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "baseline_dark": "#484878",
    "baseline_mid": "#7884B4",
    "baseline_soft": "#B4C0E4",
    "neutral_light": "#D8D8D8",
    "neutral_mid": "#8F8F8F",
    "neutral_dark": "#4D4D4D",
    "delta_up": "#2E9E44",
    "delta_down": "#E53935",
    "teal": "#42949E",
    "violet": "#9A4D8E",
    "gold": "#C99700",
    "red_strong": "#B64342",
}


def find_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "Cases").exists() and (parent / "04_results" / "v21").exists():
            return parent
    raise RuntimeError("Cannot locate V21 package root.")


ROOT = find_root()


def apply_publication_style() -> None:
    # nature-skills / nature-figure mandatory typography rules.
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["font.size"] = 14.6
    plt.rcParams["axes.labelsize"] = 16.4
    plt.rcParams["xtick.labelsize"] = 14.6
    plt.rcParams["ytick.labelsize"] = 14.6
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["legend.frameon"] = False
    plt.rcParams["figure.dpi"] = 150


def case_test_dir(slug: str) -> Path:
    case_dir = ROOT / "Cases" / slug
    return next(p for p in case_dir.iterdir() if p.is_dir() and p.name.startswith("7 "))


def case_code_dir(slug: str) -> Path:
    d = ROOT / "Cases" / slug / "sota_evidence_chain_code"
    d.mkdir(parents=True, exist_ok=True)
    return d


def clean_figure_dir(fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    for p in fig_dir.glob("*"):
        if p.is_file():
            p.unlink()


def method_color(method: str) -> str:
    if method == "FD-GRAG-RC-V21-final":
        return PALETTE["blue_main"]
    if "GaRAGe" in method:
        return PALETTE["violet"]
    if "Graph" in method or method in {"Evidence-flow baseline", "Agentic RAG"}:
        return PALETTE["teal"]
    if "Component" in method or "Parent" in method:
        return PALETTE["baseline_mid"]
    return PALETTE["neutral_mid"]


def load_tables(slug: str) -> Dict[str, pd.DataFrame]:
    table_dir = case_test_dir(slug) / "sota_evidence_chain_tables"
    return {
        "metrics": pd.read_csv(table_dir / f"{slug}_sota_metrics_by_method.csv"),
        "track": pd.read_csv(table_dir / f"{slug}_sota_metrics_by_track.csv"),
        "query": pd.read_csv(table_dir / f"{slug}_sota_metrics_by_query_type.csv"),
        "pred": pd.read_csv(table_dir / f"{slug}_complete_sota_evidence_chains.csv"),
        "ablation": pd.read_csv(table_dir / f"{slug}_ablation_metrics.csv"),
        "noev": pd.read_csv(table_dir / f"{slug}_no_evidence_probe_metrics.csv"),
    }


def enrich_methods(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["method_id"] = out["method"].map(METHOD_ID)
    out["method_short"] = out["method"].map(SHORT_NAME)
    out["method_display"] = out["method"].map(DISPLAY)
    out["method_color"] = out["method"].map(method_color)
    return out


def save(fig, out_dir: Path, stem: str, source: pd.DataFrame, caption: str) -> None:
    if not getattr(fig, "_manual_layout", False):
        fig.tight_layout()
    fig.savefig(out_dir / f"{stem}.png", dpi=600, bbox_inches="tight", facecolor="white", pad_inches=0.30)
    plt.close(fig)
    source.to_csv(out_dir / f"{stem}_source_data.csv", index=False, encoding="utf-8-sig")
    (out_dir / f"{stem}_caption.md").write_text(caption.strip() + "\n", encoding="utf-8")


def style_axis(ax, grid_axis: str = "x") -> None:
    ax.set_facecolor("white")
    ax.grid(False)
    ax.set_axisbelow(True)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color("#272727")
        ax.spines[s].set_linewidth(0.8)
    for s in ["right", "top"]:
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="both", colors="#272727", labelsize=14.6, width=1.05, length=5.0)
    ax.xaxis.label.set_color("#272727")
    ax.yaxis.label.set_color("#272727")
    ax.annotate(
        "",
        xy=(1.015, 0.0),
        xytext=(0.0, 0.0),
        xycoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", color="#272727", linewidth=1.65, mutation_scale=22),
        annotation_clip=False,
        zorder=20,
    )
    ax.annotate(
        "",
        xy=(0.0, 1.015),
        xytext=(0.0, 0.0),
        xycoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", color="#272727", linewidth=1.65, mutation_scale=22),
        annotation_clip=False,
        zorder=20,
    )


def readable_label(text: str, width: int = 19) -> str:
    return fill(str(text), width=width, break_long_words=False)


def spread_positions(values: List[float], lower: float, upper: float, min_gap: float) -> List[float]:
    """Return monotonic label positions with a minimum vertical gap."""
    if not values:
        return []
    order = np.argsort(values)
    sorted_vals = np.array(values, dtype=float)[order]
    placed = np.clip(sorted_vals, lower, upper)
    for i in range(1, len(placed)):
        placed[i] = max(placed[i], placed[i - 1] + min_gap)
    overflow = placed[-1] - upper
    if overflow > 0:
        placed -= overflow
        for i in range(len(placed) - 2, -1, -1):
            placed[i] = min(placed[i], placed[i + 1] - min_gap)
    placed = np.clip(placed, lower, upper)
    restored = np.empty_like(placed)
    restored[order] = placed
    return restored.tolist()


def method_legend(slug: str, out_dir: Path, metrics: pd.DataFrame) -> pd.DataFrame:
    df = enrich_methods(metrics[["method"]].drop_duplicates())
    df = df.set_index("method").reindex(METHODS).dropna(subset=["method_id"]).reset_index()
    df = df[["method_short", "method_display", "method"]]
    df.to_csv(out_dir / f"{slug}_method_name_legend.csv", index=False, encoding="utf-8-sig")
    (out_dir / f"{slug}_method_name_legend.md").write_text(
        "| Plot label | Full method |\n|---|---|\n"
        + "\n".join(f"| {r.method_short} | {r.method_display} |" for r in df.itertuples())
        + "\n",
        encoding="utf-8",
    )
    return df


def plot_primary_forest(slug: str, out_dir: Path, metrics: pd.DataFrame) -> None:
    df = enrich_methods(metrics).set_index("method").reindex(METHODS).dropna(subset=["method_id"]).reset_index()
    y = np.arange(len(df))[::-1]
    fig, ax = plt.subplots(figsize=(9.4, 6.4))
    metric_offsets = {
        "exact_recall_10": -0.24,
        "accepted_recall_10": -0.08,
        "evidence_package_recall_10": 0.08,
        "grounded_score": 0.24,
    }
    markers = {
        "exact_recall_10": "o",
        "accepted_recall_10": "s",
        "evidence_package_recall_10": "D",
        "grounded_score": "^",
    }
    for metric, off in metric_offsets.items():
        ax.scatter(
            df[metric],
            y + off,
            s=44,
            marker=markers[metric],
            color=df["method_color"],
            edgecolor="#272727",
            linewidth=0.45,
            label=METRIC_LABEL[metric],
            zorder=3,
        )
    plotted_vals = df[list(metric_offsets)].to_numpy(dtype=float)
    x_left = max(0.0, float(np.nanmin(plotted_vals)) - 0.025)
    x_right = min(1.02, float(np.nanmax(plotted_vals)) + 0.018)
    if x_left > 0.9:
        x_left = 0.9 - 0.02
    if x_right < 0.9:
        x_right = 0.9 + 0.02
    ax.axvline(0.9, color=PALETTE["delta_up"], linestyle="--", linewidth=0.9)
    xr = max(x_right - x_left, 1e-6)
    for metric, off in metric_offsets.items():
        if metric != "exact_recall_10":
            continue
        for yi, val in zip(y, df[metric].astype(float)):
            if not np.isfinite(val):
                continue
            if val > x_right - 0.075 * xr:
                tx, ha = val - 0.018 * xr, "right"
            else:
                tx, ha = val + 0.018 * xr, "left"
            ax.text(
                tx,
                yi + off,
                f"{val:.2f}",
                va="center",
                ha=ha,
                fontsize=9.6,
                fontweight="bold" if metric == "exact_recall_10" else "normal",
                color="#111111",
                bbox=dict(boxstyle="round,pad=0.10", facecolor="white", edgecolor="none", alpha=0.86),
                zorder=6,
            )
    ax.set_yticks(y)
    ax.set_yticklabels(df["method_short"])
    ax.set_xlim(x_left, x_right)
    ax.set_xlabel("Evidence retrieval / grounding score")
    ax.set_ylabel("")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=2, fontsize=13.2, handletextpad=0.3)
    style_axis(ax)
    save(fig, out_dir, f"{slug}_nature_primary_metric_forest", df, "Primary Nature-style metric forest with concise methodology labels.")


def plot_metric_matrix(slug: str, out_dir: Path, metrics: pd.DataFrame) -> None:
    df = enrich_methods(metrics).set_index("method").reindex(METHODS).dropna(subset=["method_id"]).reset_index()
    mat = df[METRICS].rename(columns=METRIC_LABEL)
    display = mat.copy()
    display["Risk"] = 1 - display["Risk"]
    cmap = LinearSegmentedColormap.from_list("nature_blue", ["#F7F8FB", "#B4C0E4", "#0F4D92"])
    fig, ax = plt.subplots(figsize=(10.8, 7.6))
    im = ax.imshow(display.values, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(np.arange(display.shape[1]))
    ax.set_xticklabels(["Exact", "Accepted", "Package", "Grounded", "1-Risk"], rotation=24, ha="right", fontsize=14.3)
    ax.set_yticks(np.arange(len(df)))
    ax.set_yticklabels(df["method_short"], fontsize=13.8)
    for i in range(display.shape[0]):
        for j in range(display.shape[1]):
            v = display.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=16.0, weight="bold", color="white" if v > 0.75 else "#272727")
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.ax.tick_params(labelsize=13.4)
    ax.set_xlabel("")
    ax.set_ylabel("")
    save(fig, out_dir, f"{slug}_nature_metric_matrix", pd.concat([df[["method_short", "method_display"]], mat], axis=1), "Compact metric matrix with risk inverted as 1-Risk for common colour direction.")


def plot_distribution(slug: str, out_dir: Path, pred: pd.DataFrame) -> None:
    df = enrich_methods(pred)
    order = [m for m in METHODS if m in set(df["method"])]
    values = [df.loc[df["method"] == m, "grounded_score"].values for m in order]
    fig, ax = plt.subplots(figsize=(9.8, 5.8))
    vp = ax.violinplot(values, vert=False, showmeans=True, showextrema=False, widths=0.72)
    for i, body in enumerate(vp["bodies"]):
        body.set_facecolor(method_color(order[i]))
        body.set_edgecolor("#272727")
        body.set_alpha(0.75)
        body.set_linewidth(0.5)
    vp["cmeans"].set_color("#272727")
    vp["cmeans"].set_linewidth(0.8)
    ax.axvline(0.9, color=PALETTE["delta_up"], linestyle="--", linewidth=0.9)
    ax.set_yticks(np.arange(1, len(order) + 1))
    ax.set_yticklabels([SHORT_NAME[m] for m in order])
    all_vals = np.concatenate([v for v in values if len(v)])
    ax.set_xlim(max(0.0, float(np.nanmin(all_vals)) - 0.025), min(1.02, float(np.nanmax(all_vals)) + 0.018))
    ax.set_xlabel("Grounded score per question")
    ax.set_ylabel("")
    style_axis(ax)
    save(fig, out_dir, f"{slug}_nature_grounded_distribution", df[["method_id", "method_display", "query_id", "grounded_score"]], "Question-level grounded-score distribution.")


def plot_volcano(slug: str, out_dir: Path, pred: pd.DataFrame) -> None:
    ours = pred[pred["method"] == "FD-GRAG-RC-V21-final"][["query_id", "grounded_score"]].rename(columns={"grounded_score": "ours"})
    rows = []
    for method, sub in pred.groupby("method"):
        if method == "FD-GRAG-RC-V21-final":
            continue
        merged = sub[["query_id", "grounded_score"]].merge(ours, on="query_id", how="inner")
        delta = merged["ours"].to_numpy() - merged["grounded_score"].to_numpy()
        delta = delta[np.isfinite(delta)]
        wins = int((delta > 0).sum())
        n = len(delta)
        if n:
            z = abs(wins - n / 2) / math.sqrt(max(n / 4, 1e-9))
            p = max(math.erfc(z / math.sqrt(2)), 1e-12)
        else:
            p = 1.0
        rows.append(
            {
                "method": method,
                "method_short": SHORT_NAME.get(method, method),
                "method_display": DISPLAY.get(method, method),
                "delta_grounded": float(delta.mean()) if n else np.nan,
                "neg_log10_p": -math.log10(p),
            }
        )
    df = pd.DataFrame(rows).sort_values("delta_grounded").reset_index(drop=True)
    df["neg_log10_p_plot"] = df["neg_log10_p"].clip(upper=12)
    if len(df) > 1:
        df["neg_log10_p_plot"] += np.linspace(-0.08, 0.08, len(df))
    x_min = float(df["delta_grounded"].min())
    x_max = float(df["delta_grounded"].max())
    y_min = float(df["neg_log10_p_plot"].min())
    y_max = float(df["neg_log10_p_plot"].max())
    x_span = max(x_max - x_min, 0.02)
    y_span = max(y_max - y_min, 0.50)
    x_axis_left = x_min - 0.22 * x_span
    x_axis_right = x_max + 0.30 * x_span
    y_lower = y_min - 0.22 * y_span
    y_upper = y_max + 0.24 * y_span
    fig, ax = plt.subplots(figsize=(9.8, 6.4))

    selected = set(df["method"])
    df["is_labeled"] = df["method"].isin(selected)
    bg = df[~df["is_labeled"]]
    if not bg.empty:
        ax.scatter(
            bg["delta_grounded"],
            bg["neg_log10_p_plot"],
            s=92,
            color="#A9ADB3",
            alpha=0.45,
            edgecolor="#272727",
            linewidth=0.60,
            zorder=3,
        )
    fg = df[df["is_labeled"]]
    ax.scatter(
        fg["delta_grounded"],
        fg["neg_log10_p_plot"],
        s=88,
        color=[method_color(m) for m in fg["method"]],
        edgecolor="#272727",
        linewidth=0.75,
        zorder=4,
    )
    ax.set_xlim(x_axis_left, x_axis_right)
    ax.set_ylim(y_lower, y_upper)

    xr = max(x_axis_right - x_axis_left, 1e-6)
    yr = max(y_upper - y_lower, 1e-6)
    point_boxes = [
        (
            float(r.delta_grounded) - 0.011 * xr,
            float(r.delta_grounded) + 0.011 * xr,
            float(r.neg_log10_p_plot) - 0.022 * yr,
            float(r.neg_log10_p_plot) + 0.022 * yr,
        )
        for r in df.itertuples()
    ]
    label_offsets = [
        (0.005, 0.00, "left"),
        (-0.005, 0.00, "right"),
        (0.000, 0.055, "center"),
        (0.000, -0.055, "center"),
        (0.007, 0.052, "left"),
        (-0.007, 0.052, "right"),
        (0.007, -0.052, "left"),
        (-0.007, -0.052, "right"),
        (0.010, 0.090, "left"),
        (-0.010, 0.090, "right"),
        (0.010, -0.090, "left"),
        (-0.010, -0.090, "right"),
        (0.014, 0.125, "left"),
        (-0.014, 0.125, "right"),
        (0.014, -0.125, "left"),
        (-0.014, -0.125, "right"),
        (0.018, 0.160, "left"),
        (-0.018, 0.160, "right"),
        (0.018, -0.160, "left"),
        (-0.018, -0.160, "right"),
    ]

    def volcano_label(text: str) -> str:
        return {
            "Component-graph": "CompG",
            "Component-profile": "CompP",
            "Parent-section": "Parent",
            "Causal-graph": "Causal",
            "GaRAGe-deflect": "Def",
            "GaRAGe-rel": "Rel",
            "GaRAGe-attr": "Attr",
            "GraphRAG": "Graph",
            "Char TF-IDF": "TF-IDF",
            "Evidence-flow": "E-flow",
            "SVD surrogate": "SVD",
            "Agentic RAG": "Agentic",
        }.get(str(text), str(text))

    def label_box(text: str, x: float, y: float, ha: str) -> tuple[float, float, float, float]:
        lines = str(text).split("\n")
        width = 0.0035 + 0.00120 * max(len(line) for line in lines)
        height = 0.038 + 0.034 * len(lines)
        if ha == "right":
            return (x - width, x, y - height / 2, y + height / 2)
        if ha == "center":
            return (x - width / 2, x + width / 2, y - height / 2, y + height / 2)
        return (x, x + width, y - height / 2, y + height / 2)

    def overlaps(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
        return not (a[1] < b[0] or a[0] > b[1] or a[3] < b[2] or a[2] > b[3])

    placed: list[tuple[float, float, float, float]] = []
    label_specs: list[tuple[str, float, float, float, float, str, str]] = []
    for _, row in fg.sort_values(["neg_log10_p_plot", "delta_grounded"], ascending=[True, True]).iterrows():
        px = float(row["delta_grounded"])
        py = float(row["neg_log10_p_plot"])
        label = volcano_label(row["method_short"])
        best = None
        best_score = float("inf")
        for ox, oy, ha in label_offsets:
            tx = min(max(px + ox, x_axis_left + 0.015 * xr), x_axis_right - 0.015 * xr)
            ty = min(max(py + oy, y_lower + 0.045 * yr), y_upper - 0.045 * yr)
            box = label_box(label, tx, ty, ha)
            left_guard = x_axis_left + 0.020 * xr
            right_guard = x_axis_right - 0.020 * xr
            lower_guard = y_lower + 0.035 * yr
            upper_guard = y_upper - 0.035 * yr
            if box[0] < left_guard:
                tx += left_guard - box[0]
            if box[1] > right_guard:
                tx -= box[1] - right_guard
            if box[2] < lower_guard:
                ty += lower_guard - box[2]
            if box[3] > upper_guard:
                ty -= box[3] - upper_guard
            box = label_box(label, tx, ty, ha)
            penalty = sum(10.0 for other in placed if overlaps(box, other))
            penalty += sum(7.0 for other in point_boxes if overlaps(box, other))
            if box[0] < left_guard or box[1] > right_guard or box[2] < lower_guard or box[3] > upper_guard:
                penalty += 6.0
            penalty += 5.0 * math.hypot((tx - px) / xr, (ty - py) / yr)
            if penalty < best_score:
                best_score = penalty
                best = (tx, ty, ha, box)
        assert best is not None
        tx, ty, ha, box = best
        placed.append(box)
        label_specs.append((label, px, py, tx, ty, ha, str(row["method"])))

    for label, px, py, tx, ty, ha, method in label_specs:
        ax.plot(
            [px, tx],
            [py, ty],
            color="#68707C",
            linewidth=0.9,
            linestyle=(0, (2.2, 2.0)),
            zorder=5,
        )
    for label, px, py, tx, ty, ha, method in label_specs:
        ax.text(
            tx,
            ty,
            label,
            ha=ha,
            va="center",
            fontsize=10.0,
            weight="bold",
            color="#111111",
            bbox=dict(boxstyle="round,pad=0.14", facecolor="white", edgecolor=method_color(method), linewidth=0.75, alpha=0.98),
            zorder=9,
            clip_on=False,
        )
    ax.set_xlabel("Grounded-score advantage of FD-GRAG-RC-V21", fontsize=15.6, labelpad=10)
    ax.set_ylabel("-log10 paired sign-test p", fontsize=15.6, labelpad=10)
    style_axis(ax)
    save(fig, out_dir, f"{slug}_nature_volcano", df, "Volcano-style paired comparison: x-axis shows mean grounded-score advantage of FD-GRAG-RC-V21 over each baseline; y-axis shows paired sign-test strength. All circular markers are labeled with short in-panel dashed leader lines.")


def plot_ablation(slug: str, out_dir: Path, ablation: pd.DataFrame) -> None:
    full = ablation[ablation["ablation_view"].str.contains("Full", case=False, na=False)]
    full_score = float(full["grounded_score"].iloc[0]) if not full.empty else float(ablation["grounded_score"].max())
    df = ablation[~ablation["ablation_view"].str.contains("Full", case=False, na=False)].copy()
    df = df.sort_values("grounded_score", ascending=True).reset_index(drop=True)
    df["label"] = [
        readable_label(x.replace("No ", "Without ").replace("full package optimizer", "package optimizer"), 27)
        for x in df["ablation_view"]
    ]
    df["full_grounded_score"] = full_score
    df["grounded_drop"] = full_score - df["grounded_score"].astype(float)

    y = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(10.8, max(5.4, 0.68 * len(df) + 1.4)))
    ax.hlines(y, df["grounded_score"], df["full_grounded_score"], color="#B7C0CC", linewidth=2.0, zorder=1)
    ax.scatter(
        df["grounded_score"],
        y,
        s=105,
        color=PALETTE["delta_down"],
        edgecolor="#272727",
        linewidth=0.8,
        zorder=3,
        label="Module removed",
    )
    ax.scatter(
        df["full_grounded_score"],
        y,
        s=118,
        color=PALETTE["blue_main"],
        edgecolor="#272727",
        linewidth=0.8,
        zorder=4,
        label="Full FD-GRAG-RC",
    )
    ax.axvline(full_score, color=PALETTE["blue_main"], linestyle=(0, (4, 3)), linewidth=1.1, zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels(df["label"], fontsize=13.4)
    x_min = max(0.0, float(df["grounded_score"].min()) - 0.018)
    x_max = min(1.006, full_score + 0.006)
    ax.set_xlim(x_min, x_max)
    for yi, (_, row) in zip(y, df.iterrows()):
        ax.text(
            float(row["grounded_score"]) - (x_max - x_min) * 0.012,
            yi,
            f"-{row['grounded_drop']:.3f}",
            va="center",
            ha="right",
            fontsize=13.8,
            weight="bold",
            color="#111111",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.92),
            zorder=5,
        )
        ax.text(
            min(full_score - (x_max - x_min) * 0.010, x_max - (x_max - x_min) * 0.012),
            yi - 0.18,
            f"{full_score:.3f}",
            va="center",
            ha="right",
            fontsize=10.6,
            color=PALETTE["blue_main"],
            bbox=dict(boxstyle="round,pad=0.10", facecolor="white", edgecolor="none", alpha=0.84),
            zorder=5,
        )
    ax.set_xlabel("Grounded score after module removal")
    ax.set_ylabel("")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=2, fontsize=12.8)
    style_axis(ax)
    save(fig, out_dir, f"{slug}_nature_ablation_waterfall", df, "Paired ablation plot: each line connects the ablated configuration to the full FD-GRAG-RC score.")


def plot_no_evidence(slug: str, out_dir: Path, pred: pd.DataFrame, noev: pd.DataFrame) -> None:
    ans = pred.groupby("method", as_index=False).agg(answerable_grounded=("grounded_score", "mean"))
    no = noev.groupby("method", as_index=False).agg(no_evidence_deflection=("correct_deflection", "mean"), no_evidence_risk=("unsupported_generation_risk_proxy", "mean"))
    df = enrich_methods(ans.merge(no, on="method", how="left")).set_index("method").reindex(METHODS).dropna(subset=["method_id"]).reset_index()
    y = np.arange(len(df))[::-1]
    fig, ax = plt.subplots(figsize=(9.4, 6.2))
    ax.hlines(y, df["no_evidence_deflection"], df["answerable_grounded"], color="#CFCECE", linewidth=1.2)
    ax.scatter(df["answerable_grounded"], y, marker="o", s=58, color=df["method_color"], edgecolor="#272727", linewidth=0.6, label="Evidence present")
    ax.scatter(df["no_evidence_deflection"], y, marker="s", s=52, color=PALETTE["gold"], edgecolor="#272727", linewidth=0.6, label="No-evidence deflection")
    vals = df[["no_evidence_deflection", "answerable_grounded"]].to_numpy(dtype=float)
    score_min = float(np.nanmin(vals))
    score_max = float(np.nanmax(vals))
    xr = max(score_max - score_min, 1e-6)
    for yi, (_, row) in zip(y, df.iterrows()):
        ev = float(row["answerable_grounded"])
        de = float(row["no_evidence_deflection"])
        ax.text(ev - 0.014 * xr, yi + 0.18, f"{ev:.2f}", ha="right", va="center", fontsize=9.8, fontweight="bold", bbox=dict(boxstyle="round,pad=0.08", fc="white", ec="none", alpha=0.86), zorder=6)
        ax.text(de + 0.014 * xr, yi - 0.18, f"{de:.2f}", ha="left", va="center", fontsize=9.8, bbox=dict(boxstyle="round,pad=0.08", fc="white", ec="none", alpha=0.86), zorder=6)
    ax.set_yticks(y)
    ax.set_yticklabels(df["method_short"])
    ax.set_xlim(max(0.0, score_min - 0.03), min(1.02, score_max + 0.03))
    ax.set_xlabel("Score")
    ax.set_ylabel("Method")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=2, fontsize=13.2)
    style_axis(ax)
    save(fig, out_dir, f"{slug}_nature_evidence_no_evidence", df, "Evidence-present grounding and no-evidence deflection in one scale.")


def plot_query_type(slug: str, out_dir: Path, query: pd.DataFrame) -> None:
    df = query.copy()
    df["method_short"] = df["method"].map(SHORT_NAME)
    piv = df.pivot_table(index="query_type", columns="method_short", values="grounded_score", aggfunc="mean").fillna(0)
    method_cols = [SHORT_NAME[m] for m in METHODS if SHORT_NAME[m] in piv.columns]
    piv = piv[method_cols]
    fig, ax = plt.subplots(figsize=(12.2, max(5.2, 0.58 * len(piv) + 1.6)))
    cmap = LinearSegmentedColormap.from_list("qt", ["#F7F8FB", "#B4C0E4", "#0F4D92"])
    im = ax.imshow(piv.values, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(method_cols)))
    ax.set_xticklabels(method_cols, rotation=24, ha="right", fontsize=12.8)
    ax.set_yticks(np.arange(len(piv.index)))
    ax.set_yticklabels([str(x).replace("_", " ") for x in piv.index], fontsize=13.2)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            val = float(piv.values[i, j])
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=12.8, weight="bold", color="white" if val > 0.75 else "#272727")
    cbar = fig.colorbar(im, ax=ax, fraction=0.026, pad=0.015)
    cbar.ax.tick_params(labelsize=13.0)
    ax.set_xlabel("Method")
    ax.set_ylabel("")
    save(fig, out_dir, f"{slug}_nature_query_family_matrix", piv.reset_index(), "Query-family grounded-score matrix.")


def plot_track(slug: str, out_dir: Path, track: pd.DataFrame) -> None:
    df = enrich_methods(track)
    piv = df.pivot_table(index="method", columns="track", values="grounded_score", aggfunc="mean")
    piv = piv.reindex(METHODS).dropna(how="all").reset_index()
    piv["method_short"] = piv["method"].map(SHORT_NAME)
    y = np.arange(len(piv))[::-1]
    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    for col, marker, color in [("unique_exact", "o", PALETTE["blue_main"]), ("broad_package", "s", PALETTE["teal"])]:
        if col in piv.columns:
            ax.scatter(piv[col], y, marker=marker, s=62, color=color, edgecolor="#272727", linewidth=0.6, label=col.replace("_", " "))
    ax.axvline(0.9, color=PALETTE["delta_up"], linestyle="--", linewidth=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(piv["method_short"])
    vals = piv[[c for c in ["unique_exact", "broad_package"] if c in piv.columns]].to_numpy(dtype=float)
    x_left = max(0.0, float(np.nanmin(vals)) - 0.025)
    x_right = min(1.02, float(np.nanmax(vals)) + 0.018)
    xr = max(x_right - x_left, 1e-6)
    for yi, (_, row) in zip(y, piv.iterrows()):
        if "unique_exact" in piv.columns and np.isfinite(row.get("unique_exact", np.nan)):
            v = float(row["unique_exact"])
            ax.text(v - 0.014 * xr, yi + 0.18, f"{v:.2f}", ha="right", va="center", fontsize=9.8, fontweight="bold", bbox=dict(boxstyle="round,pad=0.08", fc="white", ec="none", alpha=0.86), zorder=6)
        if "broad_package" in piv.columns and np.isfinite(row.get("broad_package", np.nan)):
            v = float(row["broad_package"])
            ax.text(v + 0.014 * xr, yi - 0.18, f"{v:.2f}", ha="left", va="center", fontsize=9.8, bbox=dict(boxstyle="round,pad=0.08", fc="white", ec="none", alpha=0.86), zorder=6)
    ax.set_xlim(x_left, x_right)
    ax.set_xlabel("Grounded score")
    ax.set_ylabel("")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=2, fontsize=13.2)
    style_axis(ax)
    save(fig, out_dir, f"{slug}_nature_track_dotplot", piv, "Track-specific grounded performance for unique-evidence and broad-package questions.")


def plot_risk_gate(slug: str, out_dir: Path, metrics: pd.DataFrame, noev: pd.DataFrame) -> None:
    m = metrics[["method", "hallucination_risk_proxy", "grounded_score"]].merge(
        noev[["method", "correct_deflection", "unsupported_generation_risk_proxy"]],
        on="method",
        how="left",
    )
    df = enrich_methods(m).set_index("method").reindex(METHODS).dropna(subset=["method_id"]).reset_index()
    fig, ax = plt.subplots(figsize=(8.4, 6.3))
    fig._manual_layout = True
    fig.subplots_adjust(left=0.15, right=0.96, bottom=0.17, top=0.92)

    xvals = df["hallucination_risk_proxy"].to_numpy(dtype=float)
    yvals = df["unsupported_generation_risk_proxy"].to_numpy(dtype=float)
    xmin = max(0.0, min(float(np.nanmin(xvals)), 0.1) - 0.010)
    xmax = min(1.02, max(float(np.nanmax(xvals)), 0.1) + 0.035)
    ymin = max(0.0, min(float(np.nanmin(yvals)), 0.1) - 0.055)
    ymax = min(1.28, max(float(np.nanmax(yvals)), 0.1) + 0.235)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    priority_methods = set(df["method"])
    df["is_labeled"] = df["method"].isin(priority_methods)

    bg = df[~df["is_labeled"]]
    if not bg.empty:
        ax.scatter(
            bg["hallucination_risk_proxy"],
            bg["unsupported_generation_risk_proxy"],
        s=72,
            color="#A9ADB3",
            alpha=0.48,
            edgecolor="#272727",
            linewidth=0.55,
            zorder=3,
        )
    fg = df[df["is_labeled"]]
    ax.scatter(
        fg["hallucination_risk_proxy"],
        fg["unsupported_generation_risk_proxy"],
        s=86,
        color=fg["method_color"],
        edgecolor="#272727",
        linewidth=0.7,
        zorder=4,
    )

    point_boxes = []
    xr = max(xmax - xmin, 1e-6)
    yr = max(ymax - ymin, 1e-6)
    for row in df.itertuples():
        x = float(row.hallucination_risk_proxy)
        y = float(row.unsupported_generation_risk_proxy)
        point_boxes.append((x - 0.012 * xr, x + 0.012 * xr, y - 0.022 * yr, y + 0.022 * yr))

    label_offsets = [
        (0.007, 0.000, "left"),
        (-0.007, 0.000, "right"),
        (0.000, -0.045, "center"),
        (0.000, 0.045, "center"),
        (0.010, -0.040, "left"),
        (-0.010, -0.040, "right"),
        (0.010, 0.040, "left"),
        (-0.010, 0.040, "right"),
        (0.015, -0.065, "left"),
        (-0.015, -0.065, "right"),
        (0.015, 0.065, "left"),
        (-0.015, 0.065, "right"),
        (0.022, -0.090, "left"),
        (-0.022, -0.090, "right"),
        (0.022, 0.090, "left"),
        (-0.022, 0.090, "right"),
    ]

    preferred_offsets = {
        "FD-GRAG-RC-V21-final": [(0.038, -0.105, "center"), (0.048, -0.080, "left")],
        "Agentic RAG": [(0.026, 0.000, "left"), (0.030, -0.080, "left")],
        "GaRAGe-style deflective RAG": [(0.030, 0.075, "left"), (0.038, 0.110, "left")],
        "Controlled GraphRAG": [(0.000, -0.085, "center"), (0.020, 0.025, "left")],
        "Evidence-flow baseline": [(0.038, -0.080, "left"), (0.045, -0.115, "left")],
        "BM25/word": [(0.020, -0.125, "center"), (-0.020, -0.125, "right")],
        "TFIDF-char": [(0.034, -0.100, "left"), (0.052, -0.135, "left")],
        "Component-graph informed RAG": [(0.000, -0.075, "center"), (0.020, -0.090, "left")],
        "SVD semantic surrogate": [(0.000, 0.060, "center"), (0.022, 0.040, "left")],
    }

    def short_label(text: str) -> str:
        return {
            "Component-graph": "CompG",
            "Component-profile": "CompP",
            "Parent-section": "Parent",
            "Causal-graph": "Causal",
            "GaRAGe-deflect": "Def",
            "GaRAGe-rel": "Rel",
            "GaRAGe-attr": "Attr",
            "GraphRAG": "Graph",
            "Char TF-IDF": "TF-IDF",
            "Evidence-flow": "E-flow",
            "SVD surrogate": "SVD",
            "Agentic RAG": "Agentic",
        }.get(str(text), str(text))

    def label_box(text: str, x: float, y: float, ha: str) -> tuple[float, float, float, float]:
        lines = str(text).split("\n")
        width = 0.008 + 0.00270 * max(len(line) for line in lines)
        height = 0.040 + 0.034 * len(lines)
        if ha == "right":
            return (x - width, x, y - height / 2, y + height / 2)
        if ha == "center":
            return (x - width / 2, x + width / 2, y - height / 2, y + height / 2)
        return (x, x + width, y - height / 2, y + height / 2)

    def overlaps(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
        return not (a[1] < b[0] or a[0] > b[1] or a[3] < b[2] or a[2] > b[3])

    placed: list[tuple[float, float, float, float]] = []
    label_specs: list[tuple[str, float, float, float, float, str, str, str]] = []
    manual_positions = {
        "Boeing_737": {
            "TFIDF-char": (0.004, 1.155, "center"),
            "BM25/word": (0.027, 1.155, "center"),
            "Parent-section retrieval": (0.051, 1.155, "center"),
            "Component-profile retrieval": (0.075, 1.155, "center"),
            "Hybrid lexical": (0.004, 1.075, "center"),
            "GaRAGe-style attribution RAG": (0.028, 1.075, "center"),
            "Causal-graph informed RAG": (0.052, 1.075, "center"),
            "Component-graph informed RAG": (0.096, 1.075, "center"),
            "SVD semantic surrogate": (0.007, 0.855, "center"),
            "Controlled GraphRAG": (0.010, 0.920, "center"),
            "GaRAGe-style relevance RAG": (0.034, 0.920, "center"),
            "Evidence-flow baseline": (0.058, 0.920, "center"),
            "FD-GRAG-RC-V21-final": (0.009, 0.168, "left"),
            "GaRAGe-style deflective RAG": (0.013, 0.285, "left"),
            "Agentic RAG": (0.044, 0.250, "left"),
        },
        "Cessna_172": {
            "Component-profile retrieval": (0.012, 1.155, "center"),
            "Parent-section retrieval": (0.038, 1.155, "center"),
            "GaRAGe-style attribution RAG": (0.064, 1.155, "center"),
            "SVD semantic surrogate": (0.104, 1.155, "center"),
            "BM25/word": (0.020, 1.075, "center"),
            "Hybrid lexical": (0.046, 1.075, "center"),
            "Causal-graph informed RAG": (0.072, 1.075, "center"),
            "Controlled GraphRAG": (0.012, 0.920, "center"),
            "GaRAGe-style relevance RAG": (0.038, 0.920, "center"),
            "Component-graph informed RAG": (0.064, 0.920, "center"),
            "TFIDF-char": (0.086, 0.920, "center"),
            "Evidence-flow baseline": (0.039, 0.835, "center"),
            "FD-GRAG-RC-V21-final": (0.012, 0.168, "left"),
            "GaRAGe-style deflective RAG": (0.025, 0.295, "left"),
            "Agentic RAG": (0.041, 0.250, "left"),
        },
    }

    def annotate_row(row: pd.Series) -> None:
        px = float(row["hallucination_risk_proxy"])
        py = float(row["unsupported_generation_risk_proxy"])
        text = short_label(row["method_short"])
        if row["method"] in manual_positions.get(slug, {}):
            tx, ty, ha = manual_positions[slug][row["method"]]
            tx = min(max(tx, xmin + 0.010), xmax - 0.010)
            ty = min(max(ty, ymin + 0.040), ymax - 0.040)
            best = (tx, ty, ha, label_box(text, tx, ty, ha))
        else:
            best = None
            best_score = float("inf")
            for ox, oy, ha in [*preferred_offsets.get(row["method"], []), *label_offsets]:
                tx = min(max(px + ox, xmin + 0.010), xmax - 0.010)
                ty = min(max(py + oy, ymin + 0.040), ymax - 0.040)
                box = label_box(text, tx, ty, ha)
                penalty = 0.0
                if box[0] < xmin or box[1] > xmax or box[2] < ymin or box[3] > ymax:
                    penalty += 10.0
                penalty += sum(2.0 for other in placed if overlaps(box, other))
                penalty += sum(1.0 for other in point_boxes if overlaps(box, other))
                penalty += 5.0 * math.hypot((tx - px) / xr, (ty - py) / yr)
                if penalty < best_score:
                    best_score = penalty
                    best = (tx, ty, ha, box)
            assert best is not None
        tx, ty, ha, box = best
        placed.append(box)
        label_specs.append((text, px, py, tx, ty, ha, str(row["method"]), str(row["method_color"])))

    label_df = fg.sort_values(
        ["unsupported_generation_risk_proxy", "hallucination_risk_proxy"],
        ascending=[True, True],
    )
    for _, row in label_df.iterrows():
        annotate_row(row)

    for text, px, py, tx, ty, ha, method, color in label_specs:
        ax.plot(
            [px, tx],
            [py, ty],
            color="#68707C",
            linewidth=0.82,
            linestyle=(0, (2.2, 2.0)),
            zorder=5,
        )
    for text, px, py, tx, ty, ha, method, color in label_specs:
        ax.text(
            tx,
            ty,
            text,
            ha=ha,
            va="center",
            fontsize=9.2,
            fontweight="bold" if method == "FD-GRAG-RC-V21-final" else "normal",
            color="#272727",
            bbox=dict(boxstyle="round,pad=0.13,rounding_size=0.05", fc="white", ec=color, lw=0.75, alpha=0.96),
            zorder=9,
            clip_on=False,
        )

    gate_ymax = min(ymax, ymin + 0.28 * yr)
    gate_xmax = min(xmax, xmin + 0.38 * xr)
    ax.plot([0.1, 0.1], [ymin, gate_ymax], color=PALETTE["delta_down"], linestyle=(0, (2.2, 2.0)), linewidth=0.9, zorder=1)
    ax.plot([xmin, gate_xmax], [0.1, 0.1], color=PALETTE["delta_down"], linestyle=(0, (2.2, 2.0)), linewidth=0.9, zorder=1)
    ax.set_xlabel("Evidence-present risk proxy")
    ax.set_ylabel("No-evidence risk proxy")
    style_axis(ax)
    save(fig, out_dir, f"{slug}_nature_risk_gate_scatter", df, "Risk split between evidence-present retrieval and no-evidence probe behavior. All circular markers are labeled with short dashed leader lines.")


def build_case(slug: str) -> dict:
    tables = load_tables(slug)
    fig_dir = case_test_dir(slug) / "sota_evidence_chain_figures"
    clean_figure_dir(fig_dir)
    legend = method_legend(slug, fig_dir, tables["metrics"])
    plot_primary_forest(slug, fig_dir, tables["metrics"])
    plot_metric_matrix(slug, fig_dir, tables["metrics"])
    plot_distribution(slug, fig_dir, tables["pred"])
    plot_volcano(slug, fig_dir, tables["pred"])
    plot_ablation(slug, fig_dir, tables["ablation"])
    plot_no_evidence(slug, fig_dir, tables["pred"], tables["noev"])
    plot_query_type(slug, fig_dir, tables["query"])
    plot_track(slug, fig_dir, tables["track"])
    plot_risk_gate(slug, fig_dir, tables["metrics"], tables["noev"])
    shutil.copy2(Path(__file__).resolve(), case_code_dir(slug) / "build_nature_style_case_figures.py")
    return {
        "case": slug,
        "figure_dir": str(fig_dir),
        "n_png": len(list(fig_dir.glob("*.png"))),
        "legend": str(fig_dir / f"{slug}_method_name_legend.csv"),
    }


def main() -> None:
    apply_publication_style()
    outputs = [build_case("Boeing_737"), build_case("Cessna_172")]
    report = ROOT / "07_reports" / "v21" / "v21_nature_style_figure_report.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# V21 Nature-style case figure report",
        "",
        "Figures were rebuilt from existing case-level CSV outputs using a nature-skills/nature-figure inspired contract: restrained palette, method-ID labels, source data per figure, and non-overlapping single-panel layouts.",
        "",
        "The user requested PNG-only outputs, so no SVG/PDF exports were produced in this pass.",
        "",
    ]
    for out in outputs:
        lines += [
            f"## {out['case']}",
            f"- PNG figures: {out['n_png']}",
            f"- Figure directory: `{out['figure_dir']}`",
            f"- Method-name legend: `{out['legend']}`",
            "",
        ]
    report.write_text("\n".join(lines), encoding="utf-8")
    manifest = ROOT / "04_results" / "v21" / "v21_nature_style_figure_manifest.json"
    manifest.write_text(json.dumps(outputs, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(outputs, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
