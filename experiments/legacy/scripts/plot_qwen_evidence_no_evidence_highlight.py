from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CASES = ["Boeing_737", "Cessna_172"]

BLUE = "#0F4D92"
TEAL = "#3A9AA5"
PURPLE = "#9A4D8E"
GREY = "#8F8F8F"
LIGHT_GREY = "#D8DEE8"
GREEN_FILL = "#E9F5EC"
INK = "#222222"


PARADIGM_LABELS = {
    "qwen_direct_no_rag": "Qwen direct",
    "qwen_bm25_context": "Qwen + BM25",
    "qwen_hybrid_context": "Qwen + Hybrid",
    "qwen_graphrag_context": "Qwen + GraphRAG",
    "qwen_evidence_flow_context": "Qwen + Evidence-flow",
    "qwen_fd_grag_rc_context": "Qwen + FD-GRAG-RC",
}


METHOD_SHORT = {
    "BM25 lexical": "BM25",
    "BM25/word": "BM25",
    "Lexical sparse retrieval": "BM25",
    "TF-IDF character": "Char TF-IDF",
    "TFIDF-char": "Char TF-IDF",
    "Character n-gram lexical retrieval": "Char TF-IDF",
    "SVD semantic surrogate": "SVD surrogate",
    "Latent semantic dense surrogate": "SVD surrogate",
    "Hybrid lexical-semantic": "Hybrid",
    "Hybrid lexical": "Hybrid",
    "Lexical-semantic hybrid": "Hybrid",
    "Parent-section retrieval": "Parent-section",
    "Parent section evidence routing": "Parent-section",
    "Component-profile retrieval": "Component-profile",
    "Controlled GraphRAG": "GraphRAG",
    "Representative coarse graph retrieval": "GraphRAG",
    "Causal-graph informed RAG": "Causal-graph",
    "Causal-only informed retrieval": "Causal-graph",
    "Component-graph informed RAG": "Component-graph",
    "Component-only informed retrieval": "Component-graph",
    "Agentic RAG": "Agentic",
    "Query-text route selection": "Agentic RAG",
    "Evidence-flow baseline": "Evidence-flow",
    "Role-component-section evidence flow": "Evidence-flow",
    "GaRAGe-style relevance RAG": "GaRAGe-rel",
    "Grounding relevance representative": "GaRAGe-rel",
    "GaRAGe-style attribution RAG": "GaRAGe-attr",
    "Attribution/citation representative": "GaRAGe-attr",
    "GaRAGe-style deflective RAG": "GaRAGe-deflect",
    "Deflective grounding representative": "GaRAGe-deflect",
    "FD-GRAG-RC-V21": "FD-GRAG-RC",
    "FD-GRAG-RC-V21-final": "FD-GRAG-RC",
    "FD-GRAG-RC evidence package": "FD-GRAG-RC",
}


def latest_dir(case: str) -> Path:
    matches = sorted((ROOT / "Cases" / case).glob("8 *"))
    if not matches:
        raise FileNotFoundError(f"No latest figure folder under Cases/{case}")
    return matches[0]


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.labelsize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 13,
            "legend.fontsize": 12,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": INK,
            "axes.linewidth": 1.0,
            "savefig.facecolor": "white",
        }
    )


def add_axis_arrows(ax, lw: float = 1.5) -> None:
    for spine in ["top", "right", "bottom", "left"]:
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="both", direction="out", length=5, width=1.0, color=INK)
    ax.annotate(
        "",
        xy=(1.02, 0),
        xytext=(0, 0),
        xycoords=("axes fraction", "axes fraction"),
        arrowprops=dict(arrowstyle="-|>", lw=lw, color=INK, mutation_scale=18),
        clip_on=False,
    )
    ax.annotate(
        "",
        xy=(0, 1.02),
        xytext=(0, 0),
        xycoords=("axes fraction", "axes fraction"),
        arrowprops=dict(arrowstyle="-|>", lw=lw, color=INK, mutation_scale=18),
        clip_on=False,
    )


def save_png(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=450, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def pct_label(value: float) -> str:
    if value >= 10:
        return f"{value:.0f}%"
    if value >= 1:
        return f"{value:.1f}%"
    return f"{value:.2f}%"


def canonical_method_label(value: object) -> str:
    text = str(value).strip()
    if "FD-GRAG-RC" in text:
        return "FD-GRAG-RC"
    return METHOD_SHORT.get(text, text)


def plot_evidence_residual(case: str) -> Path:
    out_dir = latest_dir(case)
    df = pd.read_csv(out_dir / f"{case}_qwen_generation_summary.csv")
    df = df.copy()
    df["label"] = df["paradigm"].map(PARADIGM_LABELS).fillna(df["source_method_display"])
    df["grounded_residual_pct"] = (1 - df["mean_source_grounded_score"]).clip(lower=0) * 100
    df["package_miss_pct"] = (1 - df["mean_source_evidence_package_recall_10"]).clip(lower=0) * 100
    order = [
        "qwen_direct_no_rag",
        "qwen_bm25_context",
        "qwen_hybrid_context",
        "qwen_graphrag_context",
        "qwen_evidence_flow_context",
        "qwen_fd_grag_rc_context",
    ]
    df["order"] = df["paradigm"].map({name: i for i, name in enumerate(order)})
    df = df.sort_values("order")

    source_cols = [
        "paradigm",
        "label",
        "mean_source_grounded_score",
        "mean_source_evidence_package_recall_10",
        "grounded_residual_pct",
        "package_miss_pct",
        "n",
    ]
    df[source_cols].to_csv(out_dir / f"{case}_qwen_evidence_residual_highlight_source_data.csv", index=False)

    y = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(10.2, 5.8))

    colors = [BLUE if "fd_grag" in p else (GREY if "direct" in p or "bm25" in p or "hybrid" in p else TEAL) for p in df["paradigm"]]
    ax.set_xscale("log")
    ax.grid(axis="x", color="#E1E4EA", linewidth=1.0)
    ax.set_axisbelow(True)

    for i, (_, row) in enumerate(df.iterrows()):
        c = colors[i]
        lw = 2.2 if "fd_grag" in row["paradigm"] else 1.2
        ax.plot([row["grounded_residual_pct"], row["package_miss_pct"]], [i, i], color=LIGHT_GREY, lw=1.3, zorder=1)
        ax.scatter(row["grounded_residual_pct"], i + 0.11, s=160 if "fd_grag" in row["paradigm"] else 105,
                   marker="o", color=c, edgecolor=INK, linewidth=lw, zorder=3, label="Grounded residual" if i == 0 else None)
        ax.scatter(row["package_miss_pct"], i - 0.11, s=160 if "fd_grag" in row["paradigm"] else 105,
                   marker="s", color="#2FA082" if "fd_grag" in row["paradigm"] else "#B7BDC8",
                   edgecolor=INK, linewidth=lw, zorder=3, label="Package miss" if i == 0 else None)
        ax.text(row["grounded_residual_pct"] * 1.18, i + 0.11, pct_label(row["grounded_residual_pct"]),
                ha="left", va="center", fontsize=11, weight="bold" if "fd_grag" in row["paradigm"] else "normal", color=INK)
        ax.text(row["package_miss_pct"] * 1.18, i - 0.11, pct_label(row["package_miss_pct"]),
                ha="left", va="center", fontsize=11, weight="bold" if "fd_grag" in row["paradigm"] else "normal", color=INK)

    labels = df["label"].tolist()
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    for tick, paradigm in zip(ax.get_yticklabels(), df["paradigm"]):
        if "fd_grag" in paradigm:
            tick.set_color(BLUE)
            tick.set_fontweight("bold")
            tick.set_fontsize(14)

    ax.set_xlim(0.05, 180)
    ax.set_ylim(-0.65, len(df) - 0.35)
    ax.set_xlabel("Residual evidence error (%)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.52, 1.10), ncol=2, frameon=False)
    add_axis_arrows(ax)
    out = out_dir / f"{case}_qwen_evidence_residual_highlight.png"
    save_png(fig, out)
    return out


def risk_label_offsets(methods: Iterable[str]) -> dict[str, tuple[float, float]]:
    base = {
        "FD-GRAG-RC": (0.006, -0.030),
        "GaRAGe-deflect": (0.006, 0.045),
        "Agentic RAG": (0.006, -0.065),
        "Component-graph": (0.006, -0.055),
        "SVD surrogate": (0.006, 0.055),
        "Evidence-flow": (0.006, -0.030),
        "GraphRAG": (0.006, 0.030),
        "BM25": (0.006, -0.030),
        "Char TF-IDF": (0.006, 0.030),
        "Hybrid": (0.006, 0.070),
        "Parent-section": (0.006, -0.070),
        "Component-profile": (0.006, 0.105),
        "Causal-graph": (0.006, -0.105),
        "GaRAGe-rel": (0.006, 0.140),
        "GaRAGe-attr": (0.006, -0.140),
    }
    return {m: base.get(m, (0.006, 0.025)) for m in methods}


def plot_no_evidence_risk(case: str) -> Path:
    out_dir = latest_dir(case)
    df = pd.read_csv(out_dir / f"{case}_nature_risk_gate_scatter_source_data.csv")
    df = df.copy()
    df["label"] = df["method_display"].map(canonical_method_label)
    df["evidence_present_risk_pct"] = df["hallucination_risk_proxy"] * 100
    df["no_evidence_risk_pct"] = df["unsupported_generation_risk_proxy"] * 100
    df["joint_risk_pct"] = np.sqrt(df["evidence_present_risk_pct"] * df["no_evidence_risk_pct"])
    df.to_csv(out_dir / f"{case}_qwen_no_evidence_risk_highlight_source_data.csv", index=False)

    priority = {"FD-GRAG-RC": 0, "GaRAGe-deflect": 1, "Agentic RAG": 2}
    df["priority"] = df["label"].map(priority).fillna(3)
    df = df.sort_values(["priority", "no_evidence_risk_pct", "evidence_present_risk_pct"], ascending=[True, True, True])
    y = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(10.2, 8.0))
    ax.set_facecolor("#FFFDF8")
    ax.set_xlim(0, 108)
    ax.set_ylim(-0.7, len(df) - 0.25)
    ax.grid(axis="x", color="#E6E8ED", linewidth=1.0)
    ax.set_axisbelow(True)
    ax.axvspan(0, 25, color=GREEN_FILL, zorder=0)

    for i, (_, row) in enumerate(df.iterrows()):
        is_fd = row["label"] == "FD-GRAG-RC"
        row_color = BLUE if is_fd else (PURPLE if row["label"] == "GaRAGe-deflect" else (TEAL if row["label"] == "Agentic RAG" else GREY))
        ax.plot(
            [row["evidence_present_risk_pct"], row["no_evidence_risk_pct"]],
            [i, i],
            color=LIGHT_GREY,
            lw=1.2,
            zorder=1,
        )
        ax.scatter(
            row["evidence_present_risk_pct"],
            i + 0.11,
            s=170 if is_fd else 115,
            marker="o",
            color=row_color if is_fd else "#A9ADB3",
            edgecolor=INK,
            linewidth=1.8 if is_fd else 1.0,
            zorder=3,
            label="Evidence-present risk" if i == 0 else None,
        )
        ax.scatter(
            row["no_evidence_risk_pct"],
            i - 0.11,
            s=170 if is_fd else 115,
            marker="s",
            color="#D8584A",
            edgecolor=BLUE if is_fd else INK,
            linewidth=1.8 if is_fd else 1.0,
            zorder=3,
            label="No-evidence risk" if i == 0 else None,
        )
        ax.text(
            min(row["evidence_present_risk_pct"] + 1.4, 104.5),
            i + 0.11,
            pct_label(row["evidence_present_risk_pct"]),
            ha="left",
            va="center",
            fontsize=10.5,
            weight="bold" if is_fd else "normal",
            color=BLUE if is_fd else INK,
        )
        ax.text(
            min(row["no_evidence_risk_pct"] + 1.4, 104.5),
            i - 0.11,
            pct_label(row["no_evidence_risk_pct"]),
            ha="left",
            va="center",
            fontsize=10.5,
            weight="bold" if is_fd else "normal",
            color=BLUE if is_fd else INK,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(df["label"].tolist())
    ax.invert_yaxis()
    for tick, label in zip(ax.get_yticklabels(), df["label"]):
        if label == "FD-GRAG-RC":
            tick.set_color(BLUE)
            tick.set_fontweight("bold")
            tick.set_fontsize(14)
    ax.set_xlabel("Risk proxy (%)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.52, 1.08), ncol=2, frameon=False)
    add_axis_arrows(ax)
    out = out_dir / f"{case}_qwen_no_evidence_risk_highlight.png"
    save_png(fig, out)
    return out


SUPPORT_METRICS = [
    ("Accepted R@10", "mean_source_accepted_recall_10"),
    ("Package R@10", "mean_source_evidence_package_recall_10"),
    ("Source grounded", "mean_source_grounded_score"),
    ("Role coverage", "mean_source_role_coverage_proxy"),
]


def plot_evidence_support_index(case: str) -> Path:
    out_dir = latest_dir(case)
    df = pd.read_csv(out_dir / f"{case}_qwen_generation_summary.csv").copy()
    df["label"] = df["paradigm"].map(PARADIGM_LABELS).fillna(df["source_method_display"])
    metric_cols = [col for _, col in SUPPORT_METRICS]
    df["evidence_support_index"] = df[metric_cols].mean(axis=1)
    df["evidence_support_residual_pct"] = (1 - df["evidence_support_index"]).clip(lower=0) * 100
    df = df.sort_values("evidence_support_index", ascending=True)

    out_source = out_dir / f"{case}_qwen_evidence_support_index_source_data.csv"
    df[["paradigm", "label", *metric_cols, "evidence_support_index", "evidence_support_residual_pct"]].to_csv(
        out_source, index=False
    )

    y = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(10.4, 5.8))
    ax.set_facecolor("#FFFDF8")
    ax.grid(axis="x", color="#E6E8ED", linewidth=1.0)
    ax.set_axisbelow(True)

    for i, (_, row) in enumerate(df.iterrows()):
        is_fd = row["label"] == "FD-GRAG-RC"
        color = BLUE if is_fd else (TEAL if row["label"] in {"Qwen + GraphRAG", "Qwen + Evidence-flow"} else GREY)
        ax.barh(i, row["evidence_support_index"] * 100, height=0.58, color=color, alpha=1.0 if is_fd else 0.72,
                edgecolor=INK, linewidth=1.5 if is_fd else 0.6)
        ax.text(
            row["evidence_support_index"] * 100 + 0.55,
            i,
            f"{row['evidence_support_index'] * 100:.2f}",
            ha="left",
            va="center",
            fontsize=11.5,
            weight="bold" if is_fd else "normal",
            color=BLUE if is_fd else INK,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(df["label"].tolist())
    for tick, label in zip(ax.get_yticklabels(), df["label"]):
        if label == "FD-GRAG-RC":
            tick.set_color(BLUE)
            tick.set_fontweight("bold")
            tick.set_fontsize(14)
    ax.set_xlim(0, 104)
    ax.set_xlabel("Evidence-package support index (%)")
    add_axis_arrows(ax)
    out = out_dir / f"{case}_qwen_evidence_support_index_ranked.png"
    save_png(fig, out)
    return out


def plot_fd_advantage_over_runner_up(case: str) -> Path:
    out_dir = latest_dir(case)
    df = pd.read_csv(out_dir / f"{case}_qwen_generation_summary.csv").copy()
    df["label"] = df["paradigm"].map(PARADIGM_LABELS).fillna(df["source_method_display"])
    fd = df[df["paradigm"] == "qwen_fd_grag_rc_context"].iloc[0]
    competitors = df[~df["paradigm"].isin(["qwen_fd_grag_rc_context", "qwen_direct_no_rag"])].copy()

    rows = []
    for metric_label, col in SUPPORT_METRICS:
        best_idx = competitors[col].idxmax()
        best = competitors.loc[best_idx]
        rows.append(
            {
                "metric": metric_label,
                "fd_value": float(fd[col]),
                "runner_up_label": best["label"],
                "runner_up_value": float(best[col]),
                "advantage_pp": (float(fd[col]) - float(best[col])) * 100,
                "direction": "higher_better",
            }
        )

    best_risk_idx = competitors["mean_source_hallucination_risk_proxy"].idxmin()
    best_risk = competitors.loc[best_risk_idx]
    rows.append(
        {
            "metric": "Risk reduction",
            "fd_value": float(fd["mean_source_hallucination_risk_proxy"]),
            "runner_up_label": best_risk["label"],
            "runner_up_value": float(best_risk["mean_source_hallucination_risk_proxy"]),
            "advantage_pp": (float(best_risk["mean_source_hallucination_risk_proxy"]) - float(fd["mean_source_hallucination_risk_proxy"])) * 100,
            "direction": "lower_risk_better",
        }
    )
    adv = pd.DataFrame(rows)
    adv.to_csv(out_dir / f"{case}_qwen_fd_advantage_over_runner_up_source_data.csv", index=False)

    fig, ax = plt.subplots(figsize=(10.6, 5.7))
    ax.set_facecolor("#FFFDF8")
    y = np.arange(len(adv))
    ax.grid(axis="x", color="#E6E8ED", linewidth=1.0)
    ax.set_axisbelow(True)
    ax.axvline(0, color="#7F8994", linewidth=1.0)

    max_adv = max(0.2, float(adv["advantage_pp"].max()))
    ax.set_xlim(0, max_adv * 1.55 + 0.25)
    for i, row in adv.iterrows():
        val = row["advantage_pp"]
        color = BLUE if val > 0 else "#C6CAD2"
        ax.barh(i, max(val, 0), height=0.56, color=color, edgecolor=INK, linewidth=0.8)
        ax.scatter(max(val, 0), i, s=160, color=BLUE if val > 0 else GREY, edgecolor=INK, linewidth=1.5, zorder=3)
        if val > 0.001:
            text = f"+{val:.2f} pp"
        elif val < -0.001:
            text = f"{val:.2f} pp"
        else:
            text = "tie"
        ax.text(
            max(val, 0) + max_adv * 0.04 + 0.02,
            i,
            f"{text} vs {row['runner_up_label']}",
            ha="left",
            va="center",
            fontsize=11.2,
            weight="bold" if val > 0 else "normal",
            color=BLUE if val > 0 else INK,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(adv["metric"].tolist())
    ax.invert_yaxis()
    ax.set_xlabel("FD-GRAG-RC advantage over strongest non-FD paradigm")
    add_axis_arrows(ax)
    out = out_dir / f"{case}_qwen_fd_advantage_over_runner_up.png"
    save_png(fig, out)
    return out


def check_png(path: Path) -> tuple[int, int, int]:
    from PIL import Image

    with Image.open(path) as img:
        width, height = img.size
    return width, height, path.stat().st_size


def main() -> None:
    apply_style()
    outputs = []
    for case in CASES:
        outputs.append(plot_evidence_residual(case))
        outputs.append(plot_no_evidence_risk(case))
        outputs.append(plot_evidence_support_index(case))
        outputs.append(plot_fd_advantage_over_runner_up(case))
    report_rows = []
    for out in outputs:
        w, h, size = check_png(out)
        report_rows.append({"output_png": str(out), "width": w, "height": h, "bytes": size})
    report = pd.DataFrame(report_rows)
    report_path = ROOT / "reports" / "qwen_evidence_no_evidence_highlight_figures.csv"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(report_path, index=False)
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
