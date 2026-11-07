from __future__ import annotations

import math
import re
import textwrap
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle, Rectangle
from matplotlib.colors import LinearSegmentedColormap


ROOT = Path.cwd()
FIG_ROOT = ROOT / "05_figures" / "fmeca_png"
REPORT_DIR = ROOT / "07_reports"
PAPER_DIR = ROOT / "paper_sections"
NOTEBOOK_DIR = ROOT / "09_notebook"

MAIN_DIR = FIG_ROOT / "main_figures"
RESULT_DIR = FIG_ROOT / "result_figures"
ABLATION_DIR = FIG_ROOT / "ablation_figures"
CASE_DIR = FIG_ROOT / "case_studies"
SUPPORT_DIR = FIG_ROOT / "writing_support"

for path in [MAIN_DIR, RESULT_DIR, ABLATION_DIR, CASE_DIR, SUPPORT_DIR, REPORT_DIR, PAPER_DIR, NOTEBOOK_DIR]:
    path.mkdir(parents=True, exist_ok=True)


FIGURES: list[dict[str, str]] = []
SKIPPED: list[str] = []


def install_fonts() -> None:
    candidates = [
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path(r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\calibri.ttf"),
    ]
    for p in candidates:
        if p.exists():
            try:
                font_manager.fontManager.addfont(str(p))
            except Exception:
                pass
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.edgecolor": "#263238",
            "axes.labelcolor": "#263238",
            "xtick.color": "#263238",
            "ytick.color": "#263238",
            "text.color": "#263238",
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
            "font.size": 9,
            "legend.frameon": False,
            "figure.dpi": 120,
            "savefig.dpi": 600,
        }
    )


install_fonts()

BLUE = "#0072B2"
SKY = "#56B4E9"
GREEN = "#009E73"
ORANGE = "#E69F00"
RED = "#D55E00"
PURPLE = "#CC79A7"
YELLOW = "#F0E442"
INK = "#263238"
MUTED = "#607D8B"
LIGHT = "#F5F7FA"
GRID = "#D7DEE8"
PALETTE = [BLUE, GREEN, ORANGE, PURPLE, SKY, RED, "#7A7A7A", "#4C78A8", "#72B7B2", "#B279A2"]

CMAP = LinearSegmentedColormap.from_list(
    "paper_teal_gold",
    ["#F7FBFF", "#D7EEF4", "#8DD3C7", "#3B9AB2", "#1B6D85"],
)


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="gb18030")


def shorten(value: object, width: int = 90) -> str:
    text = "" if pd.isna(value) else str(value)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= width:
        return text
    if width <= 3:
        return text[:width]
    return text[: width - 3] + "..."


def wrap(value: object, width: int = 40) -> str:
    text = "" if pd.isna(value) else str(value)
    text = re.sub(r"\s+", " ", text).strip()
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=False))


def savefig(fig: plt.Figure, path: Path, figure_id: str, figure_type: str, source_data: str, placement: str,
            key_message: str, claim_boundary: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=600)
    plt.close(fig)
    FIGURES.append(
        {
            "Figure ID": figure_id,
            "File Path": rel(path),
            "Type": figure_type,
            "Source Data": source_data,
            "Main/Supplementary": placement,
            "Key Message": key_message,
            "Claim Boundary": claim_boundary,
        }
    )


def add_key(ax, text: str, y: float = 0.018) -> None:
    ax.text(
        0.5,
        y,
        text,
        ha="center",
        va="bottom",
        transform=ax.transAxes,
        fontsize=9,
        color=INK,
        bbox=dict(boxstyle="round,pad=0.35", fc="#EEF4F7", ec="#C8D7DE", lw=0.8),
    )


def panel(ax, label: str, x: float = 0.02, y: float = 0.97) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=12, weight="bold", va="top", ha="left")


def card(ax, x: float, y: float, w: float, h: float, title: str, body: str = "", fc: str = "white",
         ec: str = "#B8C7D1", lw: float = 1.1, title_color: str = INK, fontsize: int = 8,
         rounding: float = 0.015) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.012,rounding_size={rounding}",
        fc=fc,
        ec=ec,
        lw=lw,
        mutation_aspect=1,
    )
    ax.add_patch(patch)
    ax.text(x + 0.02 * w, y + h - 0.08 * h, title, ha="left", va="top", fontsize=fontsize + 1,
            weight="bold", color=title_color)
    if body:
        ax.text(x + 0.02 * w, y + h - 0.30 * h, body, ha="left", va="top", fontsize=fontsize,
                color=INK, linespacing=1.25)
    return patch


def arrow(ax, start: tuple[float, float], end: tuple[float, float], color: str = MUTED, lw: float = 1.3,
          rad: float = 0.0) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            lw=lw,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=4,
            shrinkB=4,
        )
    )


def style_axis(ax) -> None:
    ax.grid(True, axis="x", color=GRID, lw=0.7, alpha=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#90A4AE")
    ax.spines["bottom"].set_color("#90A4AE")


def method_label(name: str) -> str:
    mapping = {
        "BM25/word": "BM25",
        "TFIDF-char": "TF-IDF",
        "SVD semantic surrogate": "Dense/SVD",
        "Hybrid lexical": "Hybrid",
        "Parent-section retrieval": "Parent",
        "Controlled GraphRAG": "GraphRAG",
        "Causal-graph informed RAG": "Causal",
        "Component-graph informed RAG": "CompGraph",
        "Evidence-flow baseline": "M-flow",
        "FD-GRAG-RC-V21-final": "V21 final",
        "FD-GRAG-RC-V21-package": "V21 package",
        "FD-GRAG-RC-V21-cue-route": "V21 cue",
        "FD-GRAG-RC-V21-exact": "V21 exact",
        "Hybrid lexical+dense": "Hybrid",
        "BM25": "BM25",
        "TF-IDF": "TF-IDF",
        "Dense/SVD": "Dense/SVD",
        "Parent-page RAG": "Parent",
        "Causal GraphRAG": "Causal",
        "Component-profile RAG": "CompProf",
        "M-flow evidence-flow": "M-flow",
        "Safety/authority RAG": "Safety",
        "FD-GRAG-RC-V5-LTR": "LTR",
        "FD-GRAG-RC-V5-package-optimizer": "Package optimizer",
        "FD-GRAG-RC-V5-final-router": "Final router",
    }
    return mapping.get(str(name), str(name).replace("FD-GRAG-RC-", "FD-"))


def combine_case_metric_files(pattern: str) -> tuple[pd.DataFrame | None, list[str]]:
    frames = []
    paths = []
    for p in sorted(ROOT.glob(pattern)):
        df = read_csv(p)
        if df is not None:
            frames.append(df)
            paths.append(rel(p))
    if not frames:
        return None, []
    return pd.concat(frames, ignore_index=True), paths


def weighted_pooled(df: pd.DataFrame, group_cols: list[str], metrics: list[str], weight_col: str) -> pd.DataFrame:
    rows = []
    for keys, sub in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        weights = pd.to_numeric(sub[weight_col], errors="coerce").fillna(0)
        row = dict(zip(group_cols, keys))
        row[weight_col] = weights.sum()
        for metric in metrics:
            vals = pd.to_numeric(sub[metric], errors="coerce")
            mask = vals.notna() & (weights > 0)
            row[metric] = np.average(vals[mask], weights=weights[mask]) if mask.any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def audit_inputs() -> dict[str, object]:
    data_patterns = {
        "current evaluation summaries": "04_results/v21/*.csv",
        "case SOTA metrics": "Cases/*/*/sota_evidence_chain_tables/*_sota_metrics_by_*.csv",
        "case top-10 evidence": "Cases/*/*/sota_evidence_chain_tables/*complete_sota_top10_evidence_long.csv",
        "case ablation metrics": "Cases/*/*/sota_evidence_chain_tables/*_ablation_metrics.csv",
        "FMECA plot data": "Cases/*/*/07_fmeca_analysis/*_plot_data.csv",
        "auxiliary proxy/rank metrics": "04_results/historical_weak_sota/*metrics*.csv",
        "existing PNG figures": "05_figures/**/*.png",
        "notebooks": "09_notebook/*.ipynb",
        "paper sections": "paper_v21/sections/*.tex",
    }
    summary = {}
    for label, pattern in data_patterns.items():
        matches = sorted(ROOT.glob(pattern))
        summary[label] = [rel(p) for p in matches[:12]]
        summary[label + " count"] = len(matches)
    return summary


def write_audit_report(summary: dict[str, object]) -> None:
    result_files = [
        "04_results/v21/v21_metrics_by_case.csv",
        "04_results/v21/v21_metrics_by_track.csv",
        "04_results/v21/v21_metrics_by_case_track.csv",
        "04_results/v21/v21_retrieval_predictions_full.csv",
        "04_results/v21/v21_target_attainment_matrix.csv",
        "04_results/v21/v21_safety_gate_metrics.csv",
        "04_results/v21/FD_GRAG_RC_V21_complete_QA_evidence_chain_rows.csv",
        "04_results/v21/FD_GRAG_RC_V21_complete_QA_evidence_chain_top10_long.csv",
        "Cases/*/7 */sota_evidence_chain_tables/*_sota_metrics_by_method.csv",
        "Cases/*/7 */sota_evidence_chain_tables/*_sota_metrics_by_query_type.csv",
        "Cases/*/7 */sota_evidence_chain_tables/*_ablation_metrics.csv",
        "04_results/historical_weak_sota/*retrieval_metrics_full.csv",
    ]
    lines = [
        "# FMECA PNG Plotting and Writing Audit",
        "",
        "Scope: this update is limited to plotting and writing. It reads existing result files and does not retrain, rerun retrieval, alter benchmark splits, or modify metric CSV values.",
        "",
        "## Data files found for plotting",
    ]
    for item in result_files:
        if "*" in item:
            matches = sorted(ROOT.glob(item))
            status = f"{len(matches)} match(es)"
            if "historical_weak_sota" in item:
                example = "auxiliary retrieval-metric tables found"
            else:
                example = "; ".join(rel(p) for p in matches[:4])
        else:
            p = ROOT / item
            status = "found" if p.exists() else "missing"
            example = rel(p) if p.exists() else item
        lines.append(f"- {item}: {status}. {example}")
    lines += [
        "",
        "## Existing figures found",
        f"- Existing PNG figures under `05_figures/` and case folders: {summary.get('existing PNG figures count', 0)}.",
        "- Existing figures include SOTA main-result plots, reference-style evidence panels, FMECA role/control plots, ablation plots, and tree visualizations.",
        "",
        "## Figures that can be redrawn as paper-style PNG",
        "- Method overview, evidence schema, reranking architecture, and claim-boundary figures can be generated from method structure rather than experimental metrics.",
        "- Main performance, query-type heatmap, evidence package completeness, ablation, and failure-analysis figures can be generated from existing v21/case result CSV files.",
        "- Case-study evidence-flow panels can be generated from the existing complete QA evidence-chain rows and top-10 evidence tables.",
        "",
        "## Data gaps that constrain interpretation",
        "- Current evaluation summary files do not contain MRR, nDCG@10, MAP@10, or Recall@20 fields; those appear only in auxiliary rank/proxy tables.",
        "- Current evaluation files do not expose every requested maintenance/FMECA proxy subfield such as `cause_action_joint_hit`; the available proxy fields are used without inventing missing metrics.",
        "- Requested ablations such as `no role tags`, `no FMECA alignment`, and `no query intent parser` are not all present as explicit ablation rows; available ablation views are plotted and missing module tests are reported.",
        "",
        "## Writing sections needing update",
        "- Model introduction should position FD-GRAG-RC-V21 as structured evidence-package retrieval rather than ordinary chunk retrieval.",
        "- Results and discussion should distinguish evidence retrieval, package completeness, proxy metrics, and real maintenance/safety claims.",
        "- Claim boundary needs explicit language that Recall@10 is not diagnostic accuracy and FMECA proxy is not certification.",
        "",
        "## Over-claiming risk",
        "- Evidence retrieval scores can be mistaken for real aircraft maintenance accuracy if not bounded.",
        "- FMECA-aligned proxy metrics can be mistaken for safety certification unless labelled as document-grounded proxies.",
        "- Gated/refusal behavior should be described as an evidence-support boundary, not a guarantee of correct diagnosis.",
        "",
        "## Experiment files intentionally not modified",
        "- `00_raw_documents/`, `01_processed_corpus/`, `03_benchmark/`, `04_results/`, `06_tables/`, case retrieval outputs, benchmark splits, and algorithm/runtime logic are left unchanged.",
    ]
    (REPORT_DIR / "fmeca_png_plotting_and_writing_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def figure_method_overview() -> None:
    fig, ax = plt.subplots(figsize=(13.5, 7.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.02, 0.975, "Figure V21-01. FD-GRAG-RC-V21 FMECA-aligned evidence-flow methodology",
            fontsize=15, weight="bold", ha="left", va="top")
    panel(ax, "(a)")

    # Layer bands
    bands = [
        (0.02, 0.74, 0.96, 0.18, "Document layer", "#F6FAFD"),
        (0.02, 0.46, 0.96, 0.22, "Structured evidence layer", "#F8FCF9"),
        (0.02, 0.18, 0.96, 0.22, "Retrieval and answer boundary", "#FFFDF5"),
    ]
    for x, y, w, h, label, fc in bands:
        ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec="#D6E0E7", lw=0.8))
        ax.text(x + 0.01, y + h - 0.018, label, fontsize=8, weight="bold", color=MUTED, ha="left", va="top")

    card(ax, 0.05, 0.78, 0.14, 0.10, "Raw maintenance docs", "manuals\nfault reports\nservice bulletins", "#FFFFFF", BLUE)
    card(ax, 0.25, 0.78, 0.14, 0.10, "Page / chunk", "layout-preserving\npage and section ids", "#FFFFFF", SKY)
    card(ax, 0.45, 0.78, 0.14, 0.10, "Episode / facet", "local diagnostic events\nfacet points", "#FFFFFF", GREEN)
    arrow(ax, (0.19, 0.83), (0.25, 0.83), BLUE)
    arrow(ax, (0.39, 0.83), (0.45, 0.83), BLUE)

    # Evidence cards
    for i, (x, role, color) in enumerate([(0.08, "role=action", GREEN), (0.24, "threshold", ORANGE), (0.40, "authority", BLUE), (0.56, "warning", RED)]):
        card(ax, x, 0.53, 0.13, 0.10, f"Evidence card {i+1}", f"{role}\ncomponent tagged\npage anchored", "#FFFFFF", color, fontsize=7)
    card(ax, 0.73, 0.51, 0.17, 0.14, "FMECA block", "failure mode\ncause / effect\naction / control", "#FFFFFF", PURPLE, fontsize=8)
    for sx in [0.21, 0.37, 0.53, 0.69]:
        arrow(ax, (sx, 0.58), (0.73, 0.58), MUTED, lw=0.8, rad=0.08)

    # Graph inset
    graph_nodes = {
        "symptom": (0.16, 0.31, GREEN),
        "component": (0.30, 0.34, BLUE),
        "cause": (0.42, 0.29, ORANGE),
        "action": (0.55, 0.33, GREEN),
        "warning": (0.42, 0.21, RED),
    }
    for a, b in [("symptom", "component"), ("component", "cause"), ("cause", "action"), ("cause", "warning")]:
        x1, y1, _ = graph_nodes[a]
        x2, y2, _ = graph_nodes[b]
        ax.plot([x1, x2], [y1, y2], color="#78909C", lw=1.3)
    for name, (x, y, c) in graph_nodes.items():
        ax.add_patch(Circle((x, y), 0.025, fc="#FFFFFF", ec=c, lw=2))
        ax.text(x, y - 0.048, name, ha="center", va="top", fontsize=7)
    ax.text(0.09, 0.39, "Causal + diagnostic graph", fontsize=9, weight="bold", ha="left")

    card(ax, 0.64, 0.27, 0.13, 0.095, "Multi-route retrieval", "BM25 + dense\nrole + component\ncausal + FMECA", "#FFFFFF", BLUE, fontsize=7)
    card(ax, 0.81, 0.27, 0.13, 0.095, "Evidence package", "ranked cards\ncoverage check\nsource trace", "#FFFFFF", GREEN, fontsize=7)
    card(ax, 0.72, 0.07, 0.18, 0.08, "Answer / refusal boundary", "answer only when evidence package is sufficient", "#FFFFFF", RED, fontsize=7)
    arrow(ax, (0.58, 0.32), (0.64, 0.32), BLUE)
    arrow(ax, (0.77, 0.32), (0.81, 0.32), BLUE)
    arrow(ax, (0.875, 0.27), (0.81, 0.15), RED)
    add_key(ax, "Key message: V21 retrieves structured diagnostic-maintenance evidence packages, not isolated chunks.")
    savefig(
        fig,
        MAIN_DIR / "figure_v21_01_method_overview.png",
        "V21-01",
        "method overview",
        "method design; no metric values",
        "Main",
        "V21 is organized around structured evidence packages rather than single text chunks.",
        "Conceptual method figure; it does not report maintenance effectiveness.",
    )


def figure_evidence_schema() -> None:
    fig, ax = plt.subplots(figsize=(12.6, 7.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.02, 0.975, "Figure V21-02. Evidence schema and FMECA alignment", fontsize=15, weight="bold", va="top")
    panel(ax, "(a)")
    card(ax, 0.39, 0.42, 0.22, 0.17, "Evidence object", "evidence_id\npage anchor\ntext span\nretrieval rank", "#FFFFFF", BLUE, fontsize=9)

    groups = [
        ("Source lineage", ["document", "page", "section", "chunk", "episode"], 0.07, 0.70, BLUE),
        ("Role typing", ["symptom", "cause", "action", "condition", "threshold"], 0.07, 0.23, GREEN),
        ("Component model", ["component", "subsystem", "aliases", "neighbors"], 0.38, 0.72, ORANGE),
        ("FMECA fields", ["failure mode", "effect", "cause", "control/action"], 0.68, 0.70, PURPLE),
        ("Authority and safety", ["authority", "warning", "limitation", "manual reference"], 0.70, 0.23, RED),
        ("Graph context", ["causal neighbors", "diagnostic path", "cross-page links"], 0.38, 0.12, SKY),
    ]
    for title, items, x, y, color in groups:
        body = "\n".join(f"- {item}" for item in items)
        card(ax, x, y, 0.22, 0.16, title, body, "#FFFFFF", color, fontsize=8)
        arrow(ax, (x + 0.11, y + (0.08 if y < 0.5 else 0.0)), (0.50, 0.50), color, lw=1.0, rad=0.07 if x < 0.4 else -0.07)

    ax.text(0.50, 0.37, "FMECA-aware retrieval index", ha="center", fontsize=10, weight="bold", color=INK)
    ax.text(0.50, 0.335, "document traceability + role/component constraints + safety controls", ha="center", fontsize=8, color=MUTED)
    add_key(ax, "Key message: each retrievable unit carries document lineage, diagnostic role, component context, and FMECA semantics.")
    savefig(
        fig,
        MAIN_DIR / "figure_v21_02_evidence_schema_fmeca.png",
        "V21-02",
        "schema",
        "method design; evidence object fields from existing corpus schema",
        "Main",
        "Structured evidence connects text spans to role, component, causal, and FMECA fields.",
        "Schema figure; it does not imply certified FMECA completeness.",
    )


def figure_candidate_pool() -> None:
    fig, ax = plt.subplots(figsize=(13.2, 7.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.02, 0.975, "Figure V21-03. Multi-route candidate pool and reranking architecture",
            fontsize=15, weight="bold", va="top")
    panel(ax, "(a)")
    routes = [
        ("BM25", "lexical match", BLUE),
        ("Dense", "semantic match", SKY),
        ("Role-aware", "required evidence role", GREEN),
        ("Component", "subsystem / alias", ORANGE),
        ("FMECA", "failure mode fields", PURPLE),
        ("Causal path", "symptom-cause-action", RED),
        ("Episode / facet", "local evidence hierarchy", "#4C78A8"),
        ("Page neighbor", "cross-page continuity", "#72B7B2"),
        ("Authority / threshold / warning", "safety-control evidence", "#B279A2"),
    ]
    yvals = np.linspace(0.78, 0.22, len(routes))
    for (title, body, color), y in zip(routes, yvals):
        card(ax, 0.05, y, 0.20, 0.055, title, body, "#FFFFFF", color, fontsize=7)
        arrow(ax, (0.25, y + 0.027), (0.39, 0.50), color, lw=0.9, rad=0.03)

    card(ax, 0.39, 0.38, 0.18, 0.23, "Candidate pool", "deduplicated evidence ids\npage/section neighbors\nrole and component metadata", "#FFFFFF", BLUE, fontsize=8)
    card(ax, 0.63, 0.55, 0.16, 0.11, "Rule score", "role fit\ncomponent fit\nsafety weighting", "#FFFFFF", GREEN, fontsize=8)
    card(ax, 0.63, 0.35, 0.16, 0.11, "Calibrated reranker", "rank evidence cards\npenalize unsupported routes", "#FFFFFF", ORANGE, fontsize=8)
    card(ax, 0.83, 0.40, 0.13, 0.18, "Diagnostic-\nmaintenance\npackage", "symptom\ncause\naction\nthreshold\nwarning\nauthority", "#FFFFFF", PURPLE, fontsize=8)
    arrow(ax, (0.57, 0.50), (0.63, 0.60), GREEN)
    arrow(ax, (0.57, 0.50), (0.63, 0.405), ORANGE)
    arrow(ax, (0.79, 0.60), (0.83, 0.50), GREEN)
    arrow(ax, (0.79, 0.405), (0.83, 0.50), ORANGE)
    add_key(ax, "Key message: multiple recall routes are merged into a package-level answer substrate, not a single top chunk.")
    savefig(
        fig,
        MAIN_DIR / "figure_v21_03_candidate_pool_reranking.png",
        "V21-03",
        "architecture",
        "method design; existing candidate-pool diagnostics support the route concept",
        "Main",
        "The architecture diversifies evidence before ranking and assembling the answer package.",
        "Architecture figure; does not add or tune retrieval results.",
    )


def load_sota_method_metrics() -> tuple[pd.DataFrame | None, list[str]]:
    return combine_case_metric_files("Cases/*/*/sota_evidence_chain_tables/*_sota_metrics_by_method.csv")


def load_sota_top10() -> tuple[pd.DataFrame | None, list[str]]:
    return combine_case_metric_files("Cases/*/*/sota_evidence_chain_tables/*complete_sota_top10_evidence_long.csv")


def selected_methods(existing: set[str]) -> list[str]:
    order = [
        "BM25/word",
        "TFIDF-char",
        "SVD semantic surrogate",
        "Hybrid lexical",
        "Parent-section retrieval",
        "Controlled GraphRAG",
        "Causal-graph informed RAG",
        "Component-graph informed RAG",
        "Evidence-flow baseline",
        "FD-GRAG-RC-V21-final",
    ]
    return [m for m in order if m in existing]


def figure_main_performance() -> None:
    df, paths = load_sota_method_metrics()
    if df is None:
        SKIPPED.append("Figure V21-04 skipped: no case SOTA metrics-by-method CSV files found.")
        return
    metrics = ["accepted_recall_10", "evidence_package_recall_10", "grounded_score", "hallucination_risk_proxy"]
    for c in metrics + ["n_queries"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    pooled = weighted_pooled(df, ["method"], metrics, "n_queries")
    pooled["aircraft_model"] = "Pooled"
    all_df = pd.concat([df, pooled], ignore_index=True)
    methods = selected_methods(set(all_df["method"].astype(str)))
    if not methods:
        SKIPPED.append("Figure V21-04 skipped: expected methods not present in metrics-by-method files.")
        return
    scopes = ["Boeing 737", "Cessna 172", "Pooled"]
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 7.2), sharey=True)
    y = np.arange(len(methods))
    metric_styles = [
        ("accepted_recall_10", "Accepted R@10", BLUE, "o"),
        ("evidence_package_recall_10", "Package R@10", GREEN, "s"),
        ("grounded_score", "Grounded score", ORANGE, "^"),
    ]
    for ax, scope in zip(axes, scopes):
        sub = all_df[all_df["aircraft_model"].eq(scope)].set_index("method").reindex(methods)
        for metric, label, color, marker in metric_styles:
            vals = pd.to_numeric(sub[metric], errors="coerce").values
            ax.scatter(vals, y, s=44, marker=marker, color=color, label=label, zorder=3)
        ax.set_title(scope)
        ax.set_xlim(0.74, 1.015)
        ax.set_xlabel("Document-grounded score")
        ax.set_yticks(y)
        ax.set_yticklabels([method_label(m) for m in methods])
        ax.invert_yaxis()
        style_axis(ax)
        if scope == "Boeing 737":
            panel(ax, "(a)", 0.0, 1.04)
        elif scope == "Cessna 172":
            panel(ax, "(b)", 0.0, 1.04)
        else:
            panel(ax, "(c)", 0.0, 1.04)
            ax.legend(loc="lower right", fontsize=8)
    fig.suptitle("Figure V21-04. Main retrieval and evidence-package performance", x=0.02, y=0.99,
                 ha="left", fontsize=15, weight="bold")
    fig.text(0.5, 0.035,
             "Key message: V21 is strongest when retrieval is evaluated as accepted and package-level document evidence.",
             ha="center", fontsize=9, bbox=dict(boxstyle="round,pad=0.35", fc="#EEF4F7", ec="#C8D7DE"))
    fig.text(0.5, 0.008,
             "Current evaluation summaries do not contain MRR/nDCG; this panel uses available R@10 and grounding metrics only.",
             ha="center", fontsize=8, color=MUTED)
    savefig(
        fig,
        RESULT_DIR / "figure_v21_04_main_performance_forest.png",
        "V21-04",
        "forest/dot plot",
        "; ".join(paths),
        "Main",
        "Accepted and package-level evidence metrics show the effect of structured retrieval.",
        "R@10 and grounding are evidence retrieval metrics, not real maintenance diagnostic accuracy; MRR/nDCG are unavailable in current evaluation summary files.",
    )


def compute_recall_at_k(top10: pd.DataFrame) -> pd.DataFrame:
    top10 = top10.copy()
    top10["rank"] = pd.to_numeric(top10["rank"], errors="coerce")
    top10["is_accepted_gold"] = top10["is_accepted_gold"].astype(str).str.lower().isin(["1", "true", "yes"])
    rows = []
    group_cols = ["aircraft_model", "method", "method_display", "query_id"]
    for keys, sub in top10.groupby(group_cols, dropna=False):
        aircraft, method, display, qid = keys
        row = {"aircraft_model": aircraft, "method": method, "method_display": display, "query_id": qid}
        for k in [1, 3, 5, 10]:
            row[f"recall_{k}"] = bool(sub.loc[sub["rank"] <= k, "is_accepted_gold"].any())
        rows.append(row)
    qdf = pd.DataFrame(rows)
    curves = []
    for keys, sub in qdf.groupby(["aircraft_model", "method", "method_display"], dropna=False):
        aircraft, method, display = keys
        row = {"aircraft_model": aircraft, "method": method, "method_display": display, "n_queries": len(sub)}
        for k in [1, 3, 5, 10]:
            row[f"recall_{k}"] = sub[f"recall_{k}"].mean()
        curves.append(row)
    per_aircraft = pd.DataFrame(curves)
    pooled_rows = []
    for keys, sub in per_aircraft.groupby(["method", "method_display"], dropna=False):
        method, display = keys
        weights = pd.to_numeric(sub["n_queries"], errors="coerce")
        row = {"aircraft_model": "Pooled", "method": method, "method_display": display, "n_queries": weights.sum()}
        for k in [1, 3, 5, 10]:
            row[f"recall_{k}"] = np.average(sub[f"recall_{k}"], weights=weights)
        pooled_rows.append(row)
    return pd.concat([per_aircraft, pd.DataFrame(pooled_rows)], ignore_index=True)


def figure_recall_curves() -> None:
    top10, paths = load_sota_top10()
    if top10 is None:
        SKIPPED.append("Figure V21-05 skipped: no complete_sota_top10_evidence_long CSV files found.")
        return
    curves = compute_recall_at_k(top10)
    methods = selected_methods(set(curves["method"].astype(str)))
    methods = [m for m in methods if m in ["BM25/word", "TFIDF-char", "SVD semantic surrogate", "Hybrid lexical", "Controlled GraphRAG", "Evidence-flow baseline", "FD-GRAG-RC-V21-final"]]
    k_values = [1, 3, 5, 10]
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 5.6), sharey=True)
    for ax, scope, lab in zip(axes, ["Boeing 737", "Cessna 172", "Pooled"], ["(a)", "(b)", "(c)"]):
        sub = curves[curves["aircraft_model"].eq(scope)].set_index("method")
        for idx, method in enumerate(methods):
            if method not in sub.index:
                continue
            vals = [sub.loc[method, f"recall_{k}"] for k in k_values]
            ax.plot(k_values, vals, marker="o", lw=1.8, color=PALETTE[idx % len(PALETTE)], label=method_label(method))
        ax.set_title(scope)
        ax.set_xlabel("k")
        ax.set_xticks(k_values)
        ax.set_ylim(0.55, 1.02)
        style_axis(ax)
        panel(ax, lab, 0.0, 1.04)
    axes[0].set_ylabel("Accepted evidence hit rate")
    axes[-1].legend(loc="lower right", fontsize=7, ncol=1)
    fig.suptitle("Figure V21-05. Recall@k curves from existing top-10 evidence outputs", x=0.02, y=0.99,
                 ha="left", fontsize=15, weight="bold")
    fig.text(0.5, 0.035, "Key message: early-rank evidence recovery matters for answer-ready packages; Recall@20 is not available in current top-10 files.",
             ha="center", fontsize=9, bbox=dict(boxstyle="round,pad=0.35", fc="#EEF4F7", ec="#C8D7DE"))
    savefig(
        fig,
        RESULT_DIR / "figure_v21_05_recall_at_k_curves.png",
        "V21-05",
        "recall curve",
        "; ".join(paths),
        "Main",
        "Top-10 evidence tables support Recall@1/3/5/10 curves from existing retrieved evidence.",
        "Curves are evidence hit rates from stored retrieval outputs; they are not real maintenance accuracy and do not include Recall@20.",
    )


def figure_query_heatmap() -> None:
    df, paths = combine_case_metric_files("Cases/*/*/sota_evidence_chain_tables/*_sota_metrics_by_query_type.csv")
    if df is None:
        SKIPPED.append("Figure V21-06 skipped: no SOTA metrics-by-query-type CSV files found.")
        return
    metric = "evidence_package_recall_10"
    df[metric] = pd.to_numeric(df[metric], errors="coerce")
    df["n_queries"] = pd.to_numeric(df["n_queries"], errors="coerce")
    pooled = weighted_pooled(df, ["method", "query_type"], [metric], "n_queries")
    methods = selected_methods(set(pooled["method"].astype(str)))
    methods = [m for m in methods if m in ["BM25/word", "TFIDF-char", "SVD semantic surrogate", "Hybrid lexical", "Controlled GraphRAG", "Evidence-flow baseline", "FD-GRAG-RC-V21-final"]]
    query_order = [
        "symptom_to_cause",
        "symptom_to_action",
        "condition_to_action",
        "threshold_query",
        "warning_or_limitation",
        "authority_or_manual_reference",
        "excluded_cause",
        "multi_hop_diagnosis",
        "cross_page_evidence_query",
        "component_to_failure",
    ]
    query_order = [q for q in query_order if q in set(pooled["query_type"])]
    mat = pooled.pivot(index="method", columns="query_type", values=metric).reindex(index=methods, columns=query_order)
    fig, ax = plt.subplots(figsize=(13.2, 6.8))
    im = ax.imshow(mat.values, aspect="auto", vmin=0.75, vmax=1.0, cmap=CMAP)
    ax.set_yticks(np.arange(len(methods)))
    ax.set_yticklabels([method_label(m) for m in methods])
    ax.set_xticks(np.arange(len(query_order)))
    ax.set_xticklabels([q.replace("_", "\n") for q in query_order], fontsize=7)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat.iloc[i, j]
            if not pd.isna(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color=INK if val < 0.92 else "white")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.015)
    cbar.set_label("Package R@10")
    ax.set_title("Figure V21-06. Query-type / FMECA-task performance heatmap", loc="left", fontsize=15, weight="bold")
    panel(ax, "(a)", 0.0, 1.09)
    fig.text(0.5, 0.025, "Key message: package-level gains should be read by task family, not only by an overall average.",
             ha="center", fontsize=9, bbox=dict(boxstyle="round,pad=0.35", fc="#EEF4F7", ec="#C8D7DE"))
    savefig(
        fig,
        RESULT_DIR / "figure_v21_06_query_type_fmeca_heatmap.png",
        "V21-06",
        "heatmap",
        "; ".join(paths),
        "Main",
        "Task-level heatmaps reveal where structured evidence packaging helps or remains weak.",
        "Task scores are document-grounded evidence metrics, not direct maintenance outcomes.",
    )


def slot_hit_table(top10: pd.DataFrame) -> pd.DataFrame:
    top10 = top10.copy()
    top10["role_l"] = top10["role"].fillna("").astype(str).str.lower() + " " + top10["role_group"].fillna("").astype(str).str.lower()
    top10["component_l"] = top10["component_family"].fillna("").astype(str).str.lower()
    slot_patterns = {
        "symptom": r"symptom|evidence|general",
        "cause": r"cause",
        "excluded cause": r"excluded",
        "action": r"action|inspection|replacement",
        "threshold": r"threshold",
        "condition": r"condition",
        "authority": r"authority|manual",
        "warning": r"warning|limitation|safety",
        "component": r"",
        "failure mode": r"failure|cause",
        "effect": r"effect|condition|symptom",
        "causal path": r"cause|condition|action|symptom",
    }
    rows = []
    group_cols = ["method", "method_display", "query_id"]
    for keys, sub in top10.groupby(group_cols, dropna=False):
        method, display, qid = keys
        row = {"method": method, "method_display": display, "query_id": qid}
        role_text = " ".join(sub["role_l"].tolist())
        comp_present = sub["component_l"].str.len().gt(0).any() and not sub["component_l"].eq("no_component").all()
        for slot, pattern in slot_patterns.items():
            if slot == "component":
                row[slot] = comp_present
            elif slot == "causal path":
                row[slot] = bool(re.search(r"cause", role_text)) and bool(re.search(r"action|condition", role_text))
            else:
                row[slot] = bool(re.search(pattern, role_text))
        rows.append(row)
    q = pd.DataFrame(rows)
    out = q.groupby(["method", "method_display"], dropna=False)[list(slot_patterns.keys())].mean().reset_index()
    return out


def figure_package_completeness() -> None:
    top10, paths = load_sota_top10()
    if top10 is None:
        SKIPPED.append("Figure V21-07 skipped: no top-10 evidence long table found.")
        return
    slots = slot_hit_table(top10)
    methods = selected_methods(set(slots["method"].astype(str)))
    methods = [m for m in methods if m in ["BM25/word", "TFIDF-char", "SVD semantic surrogate", "Hybrid lexical", "Controlled GraphRAG", "Evidence-flow baseline", "FD-GRAG-RC-V21-final"]]
    slot_cols = ["symptom", "cause", "excluded cause", "action", "threshold", "condition", "authority", "warning", "component", "failure mode", "effect", "causal path"]
    mat = slots.set_index("method").reindex(methods)[slot_cols]
    fig, ax = plt.subplots(figsize=(13.2, 6.2))
    ax.set_xlim(-0.5, len(slot_cols) - 0.5)
    ax.set_ylim(-0.5, len(methods) - 0.5)
    ax.invert_yaxis()
    for i, method in enumerate(methods):
        for j, slot in enumerate(slot_cols):
            val = mat.loc[method, slot]
            if pd.isna(val):
                continue
            ax.scatter(j, i, s=60 + 430 * val, color=CMAP(val), edgecolor="#455A64", linewidth=0.35)
            if val < 0.70:
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6, color=INK)
    ax.set_xticks(np.arange(len(slot_cols)))
    ax.set_xticklabels([s.replace(" ", "\n") for s in slot_cols], fontsize=7)
    ax.set_yticks(np.arange(len(methods)))
    ax.set_yticklabels([method_label(m) for m in methods])
    ax.grid(True, color="#E2E8EF", lw=0.7)
    ax.set_title("Figure V21-07. Evidence package completeness by retrieved evidence role", loc="left", fontsize=15, weight="bold")
    panel(ax, "(a)", 0.0, 1.08)
    fig.text(0.5, 0.025, "Key message: package completeness asks whether the retrieved set spans the diagnostic-maintenance evidence roles.",
             ha="center", fontsize=9, bbox=dict(boxstyle="round,pad=0.35", fc="#EEF4F7", ec="#C8D7DE"))
    savefig(
        fig,
        RESULT_DIR / "figure_v21_07_evidence_package_completeness.png",
        "V21-07",
        "matrix dot plot",
        "; ".join(paths),
        "Main",
        "Package completeness evaluates role coverage across retrieved evidence, not only a single hit.",
        "Role presence is derived from stored evidence tags and should not be interpreted as clinical or operational correctness.",
    )


def figure_proxy_metrics() -> None:
    p = ROOT / "04_results" / "historical_weak_sota" / "v5_retrieval_metrics_full.csv"
    df = read_csv(p)
    if df is None:
        SKIPPED.append("Figure V21-08 skipped: no historical proxy metric table found.")
        return
    df = df[df["scope"].eq("Pooled")].copy()
    metrics = [
        "maintenance_action_recall_10",
        "diagnosis_chain_coverage",
        "threshold_condition_joint_hit",
        "authority_citation_hit",
        "warning_limitation_hit",
        "evidence_package_completeness",
        "grounded_score",
        "hallucination_risk_proxy",
    ]
    methods = [
        "BM25",
        "Dense/SVD",
        "Hybrid",
        "Parent-page RAG",
        "Causal GraphRAG",
        "M-flow evidence-flow",
        "Safety/authority RAG",
        "FD-GRAG-RC-V5-package-optimizer",
    ]
    methods = [m for m in methods if m in set(df["method"])]
    mat = df.set_index("method").reindex(methods)[metrics].apply(pd.to_numeric, errors="coerce")
    fig, ax = plt.subplots(figsize=(13.4, 6.6))
    im = ax.imshow(mat.values, aspect="auto", cmap=CMAP, vmin=0, vmax=1)
    ax.set_yticks(np.arange(len(methods)))
    ax.set_yticklabels([method_label(m) for m in methods])
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels([m.replace("_", "\n") for m in metrics], fontsize=7)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat.iloc[i, j]
            if not pd.isna(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color="white" if val > 0.62 else INK)
    cbar = fig.colorbar(im, ax=ax, fraction=0.026, pad=0.015)
    cbar.set_label("Metric value")
    ax.set_title("Figure V21-08. Maintenance / FMECA document-grounded proxy metrics", loc="left", fontsize=15, weight="bold")
    panel(ax, "(a)", 0.0, 1.08)
    fig.text(0.5, 0.050,
             "Document-grounded proxy only, not real maintenance effectiveness or safety certification.",
             ha="center", fontsize=9, weight="bold", color=RED)
    fig.text(0.5, 0.020,
             "Key message: proxy metrics test whether retrieved documents contain action, threshold, authority, warning, and chain evidence.",
             ha="center", fontsize=8, bbox=dict(boxstyle="round,pad=0.3", fc="#EEF4F7", ec="#C8D7DE"))
    savefig(
        fig,
        RESULT_DIR / "figure_v21_08_maintenance_fmeca_proxy_metrics.png",
        "V21-08",
        "proxy metric heatmap",
        "auxiliary document-grounded proxy metrics table",
        "Supplementary",
        "Available proxy fields quantify document-grounded evidence completeness for maintenance-oriented answer support.",
        "Proxy metrics are not real maintenance effectiveness, aircraft safety certification, or expert validation.",
    )


def figure_ablation() -> None:
    df, paths = combine_case_metric_files("Cases/*/*/sota_evidence_chain_tables/*_ablation_metrics.csv")
    if df is None:
        SKIPPED.append("Figure V21-09 skipped: no ablation metrics CSV files found.")
        return
    for col in ["delta_grounded_vs_full", "delta_package_vs_full", "grounded_score", "evidence_package_recall_10"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[~df["ablation_view"].astype(str).str.contains("Full FD", na=False)].copy()
    pooled = df.groupby("ablation_view", dropna=False)[["delta_grounded_vs_full", "delta_package_vs_full"]].mean().reset_index()
    pooled = pooled.sort_values("delta_grounded_vs_full")
    labels = [shorten(v, 28) for v in pooled["ablation_view"]]
    y = np.arange(len(pooled))
    fig, ax = plt.subplots(figsize=(11.6, 6.4))
    vals = pooled["delta_grounded_vs_full"].values
    colors = [RED if v < 0 else GREEN for v in vals]
    ax.barh(y, vals, color=colors, alpha=0.82, edgecolor="#455A64", lw=0.4)
    ax.axvline(0, color=INK, lw=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Delta grounded score vs full model")
    style_axis(ax)
    for i, v in enumerate(vals):
        ax.text(v + (0.003 if v >= 0 else -0.003), i, f"{v:+.3f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=8)
        if v > 0:
            ax.text(v + 0.006, i, "non-monotonic effect;\nrequires further analysis", va="center", fontsize=7, color=GREEN)
    ax.set_title("Figure V21-09. Ablation delta waterfall from available module-removal views", loc="left", fontsize=15, weight="bold")
    panel(ax, "(a)", 0.0, 1.06)
    fig.text(0.5, 0.025,
             "Key message: ablations are interpreted as mechanism checks; unavailable module-specific removals are not inferred.",
             ha="center", fontsize=9, bbox=dict(boxstyle="round,pad=0.35", fc="#EEF4F7", ec="#C8D7DE"))
    savefig(
        fig,
        ABLATION_DIR / "figure_v21_09_ablation_delta_waterfall.png",
        "V21-09",
        "ablation delta plot",
        "; ".join(paths),
        "Supplementary",
        "Available ablation rows show which stored module-removal views reduce grounded evidence support.",
        "Only available ablation rows are plotted; missing requested module removals are not fabricated.",
    )


def infer_failure_attrs(row: pd.Series) -> dict[str, bool]:
    qtype = str(row.get("query_type", "")).lower()
    roles = str(row.get("required_roles", "")).lower()
    top_roles = str(row.get("top_evidence_roles", "")).lower()
    top_components = str(row.get("top_evidence_components", "")).lower()
    chain = str(row.get("evidence_chain_top10", "")).lower()
    accepted = float(row.get("accepted_recall_10", 0) or 0)
    package = float(row.get("package_recall_10", 0) or 0)
    page = float(row.get("page_recall_10", 0) or 0)
    section = float(row.get("section_recall_10", 0) or 0)
    grounded = float(row.get("grounded_score", 0) or 0)
    risk = float(row.get("hallucination_risk_proxy", 0) or 0)
    component_required = "component" in roles or "component" in qtype
    missing_role = any(r and r not in top_roles for r in re.split(r"[;,\s]+", roles) if r in ["action", "cause", "threshold", "condition", "warning", "authority"])
    boiler = any(tok in chain for tok in ["目录", "index", "intentionally left blank", "table of contents"])
    generic = top_components.count("no_component") >= max(2, len(top_components.split(";")) // 2)
    return {
        "component boundary miss": component_required and (grounded < 0.95 or generic),
        "role confusion": missing_role or (grounded < 0.7 and package < 1.0),
        "confirmed/excluded cause confusion": ("cause" in qtype or "cause" in roles or "excluded" in qtype) and grounded < 0.95,
        "cross-page miss": ("cross_page" in qtype or "page" in roles or "section" in roles) and (page < 1 or section < 1 or grounded < 0.7),
        "threshold miss": ("threshold" in qtype or "threshold" in roles) and (package < 1 or grounded < 0.95),
        "authority miss": ("authority" in qtype or "authority" in roles) and grounded < 0.95,
        "warning miss": ("warning" in qtype or "limitation" in qtype or "warning" in roles) and grounded < 0.95,
        "causal path miss": ("multi_hop" in qtype or "symptom_to_cause" in qtype or "component_to_failure" in qtype) and grounded < 0.75,
        "FMECA failure-mode mismatch": ("component_to_failure" in qtype or "failure" in qtype) and (accepted < 1 or grounded < 0.75),
        "boilerplate/index page retrieved": boiler,
        "OCR/image-only missing": (accepted < 1 and page == 0) or ("ocr" in chain),
        "too generic evidence": generic or risk > 0.3,
    }


def figure_failure_matrix() -> None:
    rows = read_csv(ROOT / "04_results" / "v21" / "FD_GRAG_RC_V21_complete_QA_evidence_chain_rows.csv")
    if rows is None:
        SKIPPED.append("Figure V21-10 skipped: complete QA evidence-chain rows file not found.")
        return
    for col in ["accepted_recall_10", "package_recall_10", "page_recall_10", "section_recall_10", "grounded_score", "hallucination_risk_proxy"]:
        rows[col] = pd.to_numeric(rows[col], errors="coerce")
    fail = rows[
        (rows["strict_pass"].astype(str).str.lower() != "true")
        | (rows["accepted_recall_10"] < 1)
        | (rows["package_recall_10"] < 1)
        | (rows["grounded_score"] < 0.95)
    ].copy()
    if fail.empty:
        SKIPPED.append("Figure V21-10 skipped: no failed or gated-review rows found.")
        return
    fail = fail.head(18)
    attrs = [
        "component boundary miss",
        "role confusion",
        "confirmed/excluded cause confusion",
        "cross-page miss",
        "threshold miss",
        "authority miss",
        "warning miss",
        "causal path miss",
        "FMECA failure-mode mismatch",
        "boilerplate/index page retrieved",
        "OCR/image-only missing",
        "too generic evidence",
    ]
    mat = []
    for _, row in fail.iterrows():
        inferred = infer_failure_attrs(row)
        mat.append([inferred[a] for a in attrs])
    mat = np.array(mat, dtype=int)
    fig, ax = plt.subplots(figsize=(14.0, 7.0))
    ax.imshow(mat, aspect="auto", cmap=LinearSegmentedColormap.from_list("fail", ["#FFFFFF", RED]), vmin=0, vmax=1)
    ax.set_yticks(np.arange(len(fail)))
    ax.set_yticklabels([f"{r.query_id} | {r.aircraft_model} | {r.query_type}" for r in fail.itertuples()], fontsize=7)
    ax.set_xticks(np.arange(len(attrs)))
    ax.set_xticklabels([a.replace(" ", "\n") for a in attrs], fontsize=7)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if mat[i, j]:
                ax.text(j, i, "●", ha="center", va="center", color="white", fontsize=9)
    ax.set_title("Figure V21-10. Failure analysis matrix from gated-review rows", loc="left", fontsize=15, weight="bold")
    panel(ax, "(a)", 0.0, 1.08)
    fig.text(0.5, 0.025,
             "Key message: failure rows expose structured evidence gaps rather than a single undifferentiated error category.",
             ha="center", fontsize=9, bbox=dict(boxstyle="round,pad=0.35", fc="#EEF4F7", ec="#C8D7DE"))
    savefig(
        fig,
        ABLATION_DIR / "figure_v21_10_failure_analysis_matrix.png",
        "V21-10",
        "failure matrix",
        "current complete QA evidence-chain rows",
        "Supplementary",
        "Gated-review cases can be categorized by interpretable evidence failure attributes.",
        "Attribute coding is derived from existing review/metric fields for analysis; it is not a new benchmark label or algorithm change.",
    )


def figure_claim_boundary() -> None:
    fig, ax = plt.subplots(figsize=(12.4, 7.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.02, 0.975, "Figure V21-11. Claim boundary and evidence hierarchy", fontsize=15, weight="bold", va="top")
    panel(ax, "(a)")
    levels = [
        ("1. Text span retrieval", "supported"),
        ("2. Role-aware evidence", "supported"),
        ("3. Component-aware evidence", "supported"),
        ("4. Causal-path evidence", "supported"),
        ("5. FMECA-aligned evidence package", "proxy only"),
        ("6. Generation-ready answer", "proxy only"),
        ("7. Expert-validated maintenance decision", "not claimed"),
        ("8. Real operational safety outcome", "future validation required"),
    ]
    y0 = 0.12
    h = 0.085
    for i, (label, status) in enumerate(levels):
        y = y0 + i * h
        if status == "supported":
            color, fc = GREEN, "#ECF8F3"
        elif status == "proxy only":
            color, fc = ORANGE, "#FFF8E7"
        else:
            color, fc = RED, "#FFF0EB"
        card(ax, 0.16, y, 0.54, 0.055, label, "", fc, color, fontsize=8, rounding=0.01)
        ax.text(0.75, y + 0.028, status, ha="left", va="center", fontsize=9, weight="bold", color=color)
        if i < len(levels) - 1:
            arrow(ax, (0.43, y + 0.057), (0.43, y + h - 0.005), MUTED, lw=1.0)
    ax.add_patch(Rectangle((0.12, 0.09), 0.78, 0.38, fill=False, ec=GREEN, lw=1.3, linestyle="-"))
    ax.text(0.13, 0.455, "Supported by this study", color=GREEN, weight="bold", fontsize=9)
    ax.add_patch(Rectangle((0.12, 0.47), 0.78, 0.18, fill=False, ec=ORANGE, lw=1.3, linestyle="--"))
    ax.text(0.13, 0.635, "Proxy only", color=ORANGE, weight="bold", fontsize=9)
    ax.add_patch(Rectangle((0.12, 0.65), 0.78, 0.18, fill=False, ec=RED, lw=1.3, linestyle=":"))
    ax.text(0.13, 0.815, "Not claimed / future validation", color=RED, weight="bold", fontsize=9)
    add_key(ax, "Key message: this study supports evidence-level traceability and package completeness, not certified maintenance decisions.")
    savefig(
        fig,
        SUPPORT_DIR / "figure_v21_11_claim_boundary_evidence_hierarchy.png",
        "V21-11",
        "claim boundary",
        "writing support; no metric values",
        "Main",
        "The manuscript should bound claims at evidence retrieval and document-grounded package support.",
        "Explicitly excludes operational safety certification, expert replacement, or guaranteed diagnosis.",
    )


def parse_top_evidence(top10: pd.DataFrame, query_id: str, n: int = 4) -> pd.DataFrame:
    sub = top10[top10["query_id"].astype(str).eq(str(query_id))].copy()
    if sub.empty:
        return sub
    sub["rank"] = pd.to_numeric(sub["rank"], errors="coerce")
    return sub.sort_values("rank").head(n)


def draw_case(row: pd.Series, evid: pd.DataFrame, path: Path, figure_id: str, title: str, status: str) -> None:
    fig, ax = plt.subplots(figsize=(14.0, 7.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.02, 0.975, title, fontsize=15, weight="bold", va="top")
    panel(ax, "(a)", 0.02, 0.915)
    question = shorten(row.get("question", ""), 190)
    intent = f"{row.get('query_type', '')} | {row.get('track', '')}"
    roles = str(row.get("required_roles", ""))
    comps = shorten(row.get("top_evidence_components", ""), 95)
    if not comps or comps == "...":
        comps = f"required roles: {roles}; query family: {row.get('query_type', '')}"
    card(ax, 0.03, 0.70, 0.27, 0.16, "Question", wrap(question, 42), "#FFFFFF", BLUE, fontsize=7)
    card(ax, 0.03, 0.49, 0.27, 0.14, "Parsed intent", f"{wrap(intent, 40)}\nroles: {wrap(roles, 35)}", "#FFFFFF", GREEN, fontsize=7)
    card(ax, 0.03, 0.30, 0.27, 0.12, "Component / failure-mode context", wrap(comps.replace(";", "; "), 38), "#FFFFFF", ORANGE, fontsize=7)
    arrow(ax, (0.30, 0.78), (0.36, 0.78), BLUE)
    arrow(ax, (0.30, 0.56), (0.36, 0.56), GREEN)
    arrow(ax, (0.30, 0.36), (0.36, 0.36), ORANGE)

    y_positions = [0.70, 0.53, 0.36, 0.19]
    for (_, ev), y in zip(evid.iterrows(), y_positions):
        body = (
            f"id: {shorten(ev.get('evidence_id',''), 23)} | p.{shorten(ev.get('page_number',''), 8)} | rank {shorten(ev.get('rank',''), 4)}\n"
            f"role: {shorten(ev.get('role',''), 22)} | component: {shorten(ev.get('component_family',''), 24)}\n"
            f"FMECA: {shorten(ev.get('role_group',''), 26)} | src: {shorten(ev.get('file_name',''), 30)}\n"
            f"{wrap(shorten(ev.get('text',''), 78), 46)}"
        )
        color = GREEN if str(ev.get("is_accepted_gold", "")).lower() in ["1", "true"] else "#90A4AE"
        card(ax, 0.37, y, 0.31, 0.135, f"Evidence card {int(float(ev.get('rank', 0) or 0))}", body, "#FFFFFF", color, fontsize=5)

    card(ax, 0.73, 0.65, 0.22, 0.15, "Causal / FMECA path", "symptom or condition\n→ component / cause\n→ action / threshold\n→ warning / authority", "#FFFFFF", PURPLE, fontsize=8)
    package_body = (
        f"accepted R@10: {row.get('accepted_recall_10', '')}\n"
        f"package R@10: {row.get('package_recall_10', '')}\n"
        f"grounded score: {row.get('grounded_score', '')}\n"
        f"safety gate: {wrap(row.get('safety_gate_decision', ''), 25)}"
    )
    card(ax, 0.73, 0.42, 0.22, 0.15, "Evidence package", package_body, "#FFFFFF", BLUE, fontsize=8)
    conclusion_color = GREEN if status == "answer" else RED
    conclusion_title = "Answer-ready card" if status == "answer" else "Refusal / review card"
    conclusion_body = "Document-grounded package is sufficient for a concise evidence-cited answer." if status == "answer" else "Evidence package is incomplete or risky; answer should be refused or sent to human review."
    card(ax, 0.73, 0.18, 0.22, 0.15, conclusion_title, wrap(conclusion_body, 45), "#FFFFFF", conclusion_color, fontsize=8)
    arrow(ax, (0.68, 0.79), (0.73, 0.73), PURPLE)
    arrow(ax, (0.84, 0.65), (0.84, 0.57), BLUE)
    arrow(ax, (0.84, 0.42), (0.84, 0.33), conclusion_color)
    fig.text(0.5, 0.025,
             "Key message: case panels show how evidence cards become an answer-ready or refusal-ready package.",
             ha="center", fontsize=9, bbox=dict(boxstyle="round,pad=0.35", fc="#EEF4F7", ec="#C8D7DE"))
    savefig(
        fig,
        path,
        figure_id,
        "case study",
        "current complete QA evidence-chain rows and top-10 evidence table",
        "Main" if status == "answer" else "Supplementary",
        "Evidence-flow case studies expose how retrieved cards support either answer emission or refusal.",
        "Case panels are document-grounded examples and do not establish real-world maintenance correctness.",
    )


def figure_case_studies() -> None:
    rows = read_csv(ROOT / "04_results" / "v21" / "FD_GRAG_RC_V21_complete_QA_evidence_chain_rows.csv")
    top10 = read_csv(ROOT / "04_results" / "v21" / "FD_GRAG_RC_V21_complete_QA_evidence_chain_top10_long.csv")
    if rows is None or top10 is None:
        SKIPPED.append("Figure V21-12 skipped: complete QA rows or top10 long evidence table missing.")
        return
    cases = []
    success = rows[rows["strict_pass"].astype(str).str.lower().eq("true")]
    fail = rows[rows["strict_pass"].astype(str).str.lower().ne("true")]
    boeing = success[success["aircraft_model"].eq("Boeing 737")]
    cessna = success[success["aircraft_model"].eq("Cessna 172")]
    fmeca = success[success["query_type"].isin(["component_to_failure", "threshold_query", "warning_or_limitation", "multi_hop_diagnosis"])]
    limitation = fail[fail["query_type"].isin(["threshold_query", "warning_or_limitation", "cross_page_evidence_query", "component_to_failure"])]
    if not boeing.empty:
        cases.append((boeing.iloc[0], "case_boeing_success_01.png", "V21-12a", "Figure V21-12a. Boeing success evidence-flow case", "answer"))
    if not cessna.empty:
        cases.append((cessna.iloc[0], "case_cessna_success_01.png", "V21-12b", "Figure V21-12b. Cessna success evidence-flow case", "answer"))
    if not fmeca.empty:
        cases.append((fmeca.iloc[0], "case_fmeca_maintenance_action_01.png", "V21-12c", "Figure V21-12c. FMECA-oriented maintenance evidence-flow case", "answer"))
    if not limitation.empty:
        cases.append((limitation.iloc[0], "case_failure_or_limitation_01.png", "V21-12d", "Figure V21-12d. Failure or limitation evidence-flow case", "refusal"))
    if len(cases) < 4:
        SKIPPED.append("Figure V21-12 partially generated: fewer than four case categories had matching rows.")
    for row, filename, fid, title, status in cases:
        evid = parse_top_evidence(top10, str(row["query_id"]), 4)
        draw_case(row, evid, CASE_DIR / filename, fid, title, status)


def write_model_intro() -> None:
    md = """# FD-GRAG-RC-V21 FMECA-Aligned Model Introduction

## 1. Overall methodological position

FD-GRAG-RC-V21 is designed to retrieve structured diagnostic and maintenance evidence packages rather than isolated text chunks. The method treats aviation maintenance answering as an evidence organization problem: a useful answer usually requires a traceable symptom, a component boundary, a causal or diagnostic relation, an action or inspection requirement, and any authority, threshold, warning, or limitation that constrains the answer.

## 2. Document-to-evidence transformation

Raw maintenance manuals, fault notes, and procedural documents are transformed into page-, section-, chunk-, episode-, facet-, and facetpoint-level evidence objects. Each object retains document lineage so that later retrieval can be audited back to page and section context.

## 3. Role-constrained evidence modeling

Evidence objects are tagged with diagnostic and maintenance roles such as symptom, confirmed cause, excluded cause, condition, threshold, action, inspection, replacement, warning, limitation, and authority. These role constraints prevent the retriever from treating a warning, a threshold, and a maintenance action as interchangeable semantic text.

## 4. Component and FMECA alignment

The method aligns evidence objects with component families and FMECA-style fields, including failure mode, failure effect, probable cause, detection evidence, maintenance action, and control. This allows the retrieved package to connect a question about a component or failure mode to the evidence roles needed for maintenance-oriented reasoning.

## 5. Causal and diagnostic graph

Component and causal links organize evidence into diagnostic paths. The graph is not used as an oracle; it is used to preserve neighboring evidence and to help retrieve cause-action, threshold-condition, and warning-authority context that may be split across pages or sections.

## 6. Multi-route retrieval and reranking

FD-GRAG-RC-V21 merges lexical, semantic, role-aware, component-aware, FMECA-aware, causal-path, episode/facet, page-neighbor, and safety-control routes into a candidate pool. A scoring and reranking stage then prioritizes evidence that jointly satisfies query intent, component constraints, role coverage, and authority or warning requirements.

## 7. Diagnostic-maintenance evidence package

The final output is an evidence package containing the ranked evidence cards needed to support an answer or refusal. A package may include action evidence, causal evidence, threshold and condition evidence, warning or limitation evidence, and authority references. This package-level view is more informative than a single-span hit because aviation maintenance questions often require multiple evidence roles.

## 8. Evidence-grounded answer boundary

The generation boundary is evidence-grounded. If the package does not sufficiently support the requested answer, the system should refuse or request human review rather than emit an unsupported maintenance statement.

## 9. Claim boundary

The evaluation supports document-grounded evidence retrieval, traceability, and package completeness. It does not prove real aircraft maintenance accuracy, operational safety, expert replacement, or FMECA certification. Recall@10 and related package metrics are evidence retrieval metrics, not diagnostic accuracy.
"""
    (REPORT_DIR / "fmeca_model_intro_rewritten.md").write_text(md, encoding="utf-8")
    tex = r"""\section{FD-GRAG-RC-V21: FMECA-Aligned Evidence-Package Retrieval}

FD-GRAG-RC-V21 is designed to retrieve structured diagnostic and maintenance evidence packages rather than isolated text chunks. The method treats aviation maintenance question answering as an evidence organization problem: a supported answer typically requires a traceable symptom, a component boundary, a causal or diagnostic relation, an action or inspection requirement, and any authority, threshold, warning, or limitation that constrains the answer.

\paragraph{Document-to-evidence transformation.}
Raw maintenance manuals, fault notes, and procedural documents are transformed into page-, section-, chunk-, episode-, facet-, and facetpoint-level evidence objects. Each object retains document lineage so that retrieval can be audited back to page and section context.

\paragraph{Role-constrained evidence modeling.}
Evidence objects are tagged with diagnostic and maintenance roles such as symptom, confirmed cause, excluded cause, condition, threshold, action, inspection, replacement, warning, limitation, and authority. These role constraints prevent the retriever from treating a warning, a threshold, and a maintenance action as interchangeable semantic text.

\paragraph{Component and FMECA alignment.}
The method aligns evidence objects with component families and FMECA-style fields, including failure mode, failure effect, probable cause, detection evidence, maintenance action, and control. This allows the retrieved package to connect a question about a component or failure mode to the evidence roles needed for maintenance-oriented reasoning.

\paragraph{Causal and diagnostic graph.}
Component and causal links organize evidence into diagnostic paths. The graph is not used as an oracle; it is used to preserve neighboring evidence and to help retrieve cause-action, threshold-condition, and warning-authority context that may be split across pages or sections.

\paragraph{Multi-route retrieval and reranking.}
FD-GRAG-RC-V21 merges lexical, semantic, role-aware, component-aware, FMECA-aware, causal-path, episode/facet, page-neighbor, and safety-control routes into a candidate pool. A scoring and reranking stage prioritizes evidence that jointly satisfies query intent, component constraints, role coverage, and authority or warning requirements.

\paragraph{Diagnostic-maintenance evidence package.}
The final output is an evidence package containing the ranked evidence cards needed to support an answer or refusal. A package may include action evidence, causal evidence, threshold and condition evidence, warning or limitation evidence, and authority references. This package-level view is more informative than a single-span hit because aviation maintenance questions often require multiple evidence roles.

\paragraph{Evidence-grounded answer boundary.}
The generation boundary is evidence-grounded. If the package does not sufficiently support the requested answer, the system should refuse or request human review rather than emit an unsupported maintenance statement.

\paragraph{Claim boundary.}
The evaluation supports document-grounded evidence retrieval, traceability, and package completeness. It does not prove real aircraft maintenance accuracy, operational safety, expert replacement, or FMECA certification. Recall@10 and related package metrics are evidence retrieval metrics, not diagnostic accuracy.
"""
    (PAPER_DIR / "fmeca_model_intro_rewritten.tex").write_text(tex, encoding="utf-8")


def write_claim_boundary() -> None:
    md = """# V21 FMECA Claim Boundary

## Supported claims

- The method improves document-grounded evidence retrieval when evaluated against stored gold/accepted evidence references.
- The method organizes maintenance evidence into role-, component-, causal-, and FMECA-aware packages.
- The evaluation supports evidence-level traceability and retrieval completeness.
- Maintenance and FMECA metrics are proxy indicators based on retrieved documents.
- The refusal boundary can reduce unsupported generation from insufficient evidence packages.

## Claims not supported by this study

- The method proves real aircraft maintenance accuracy.
- The method is certified for operational safety.
- The method replaces expert judgement.
- The method guarantees correct diagnosis.
- Recall@10 equals diagnostic accuracy.
- FMECA proxy equals real FMECA certification.

## Required wording discipline

Use "document-grounded evidence retrieval", "evidence-package completeness", and "proxy validation" when discussing results. Avoid wording that implies operational deployment, certified safety, or autonomous maintenance decision-making without expert validation.
"""
    (REPORT_DIR / "fmeca_claim_boundary_rewritten.md").write_text(md, encoding="utf-8")
    tex = r"""\section{Claim Boundary}

The supported claim is that FD-GRAG-RC-V21 improves document-grounded evidence retrieval and organizes retrieved maintenance evidence into role-, component-, causal-, and FMECA-aware packages. The evaluation supports evidence-level traceability, package completeness, and proxy indicators derived from retrieved documents.

The study does not show real aircraft maintenance accuracy, operational safety certification, replacement of expert judgement, or guaranteed diagnosis. Recall@10 is an evidence retrieval metric, not diagnostic accuracy. FMECA proxy metrics are document-grounded indicators, not real FMECA certification or safety approval.

Accordingly, the results should be described as evidence retrieval and evidence-package validation. Any stronger claim about expert maintenance decisions, airworthiness, safety certification, or field deployment requires prospective expert review and operational validation.
"""
    (PAPER_DIR / "fmeca_claim_boundary_rewritten.tex").write_text(tex, encoding="utf-8")


def write_results_discussion() -> None:
    method_df, _ = load_sota_method_metrics()
    summary_sentence = "The available method-level files support a comparison of accepted Recall@10, package Recall@10, grounded score, and hallucination-risk proxy."
    if method_df is not None:
        method_df["n_queries"] = pd.to_numeric(method_df["n_queries"], errors="coerce")
        metrics = ["accepted_recall_10", "evidence_package_recall_10", "grounded_score", "hallucination_risk_proxy"]
        for c in metrics:
            method_df[c] = pd.to_numeric(method_df[c], errors="coerce")
        pooled = weighted_pooled(method_df, ["method"], metrics, "n_queries")
        v21 = pooled[pooled["method"].eq("FD-GRAG-RC-V21-final")]
        if not v21.empty:
            r = v21.iloc[0]
            summary_sentence = (
                f"In the pooled method table, FD-GRAG-RC-V21-final has accepted Recall@10={r['accepted_recall_10']:.3f}, "
                f"package Recall@10={r['evidence_package_recall_10']:.3f}, grounded score={r['grounded_score']:.3f}, "
                f"and hallucination-risk proxy={r['hallucination_risk_proxy']:.3f}."
            )
    results = rf"""\section{{Results Narrative for V21 FMECA Figures}}

Figure V21-04 compares the available method-level evidence retrieval scores. {summary_sentence} The figure should be read as an evidence-retrieval result: it measures whether the stored retrieval outputs recover accepted and package-level evidence, not whether a mechanic would make the correct real-world diagnosis.

Figure V21-05 shifts the view from a single top-10 number to early-rank recovery. The improvement is most meaningful when relevant evidence appears early enough to support package assembly. Because the current top-10 evidence files do not contain Recall@20, the curve is limited to Recall@1, Recall@3, Recall@5, and Recall@10.

Figure V21-06 shows that the method should be evaluated by query family. The improvement is most interpretable for questions requiring role-component coupling, cross-page evidence, threshold conditions, authority references, or warning/limitation evidence. This suggests that the gain is not merely caused by lexical overlap but by the structured evidence organization.

Figure V21-07 evaluates package completeness rather than single-span retrieval. A maintenance answer may require action, condition, threshold, authority, warning, component, and causal context. Therefore, a package-level matrix is more informative than reporting only whether one gold span appeared in the top ten.

Figure V21-08 reports document-grounded maintenance and FMECA proxy metrics. These metrics indicate whether retrieved documents contain maintenance-action, threshold-condition, authority, warning, and diagnostic-chain evidence. They must not be interpreted as real maintenance effectiveness, safety certification, or expert-validated FMECA performance.

Figure V21-09 presents available ablation views. When removing a module decreases grounded score or package completeness, the result supports the mechanism that structured evidence organization contributes to retrieval quality. If any removal produces a non-monotonic improvement, the text should state that the weighting is not uniformly beneficial and requires further calibration.

Figure V21-10 summarizes failure rows as structured evidence gaps. The main point is not that failures disappear, but that the remaining failures are diagnosable: component boundary misses, role confusion, threshold misses, authority misses, warning misses, cross-page misses, and overly generic evidence can be inspected separately.
"""
    (PAPER_DIR / "fmeca_results_narrative.tex").write_text(results, encoding="utf-8")
    discussion = r"""\section{Discussion Narrative}

\paragraph{Why structured evidence matters.}
Maintenance question answering is rarely solved by retrieving a single semantically similar chunk. A useful answer often depends on several evidence roles: the symptom or condition that triggers the question, the component boundary, the cause or failure mode, the required action, and the warning or authority that constrains the action. FD-GRAG-RC-V21 makes these roles explicit and therefore allows retrieval to be evaluated as evidence-package assembly.

\paragraph{Why FMECA alignment matters.}
FMECA alignment provides a vocabulary for connecting failure mode, effect, cause, detection evidence, and maintenance control. This does not certify the system as a FMECA engine, but it makes the retrieved evidence more compatible with how aviation maintenance risks and controls are documented.

\paragraph{Maintenance-action and threshold evidence.}
Maintenance actions and threshold conditions are different from ordinary semantic retrieval targets. An action may be unsafe if retrieved without the condition or threshold that activates it. Similarly, a warning may be relevant even when it is not lexically similar to the question. The package view therefore better matches the evidence needs of maintenance-oriented answers.

\paragraph{Evidence package completeness.}
Single-span recall can hide missing context. A method may retrieve the correct action but miss the warning, authority, or threshold that determines whether the action is applicable. Evidence package completeness captures this broader support structure.

\paragraph{Failure modes.}
The remaining errors are interpretable. They include component boundary misses, role confusion, confirmed/excluded cause confusion, cross-page evidence gaps, threshold or authority misses, warning omissions, boilerplate retrieval, OCR-related gaps, and generic evidence packages. These failures point to specific future work rather than undifferentiated retrieval noise.

\paragraph{Deployment boundary.}
The current system should be treated as an evidence-support and refusal-boundary mechanism. It can organize and retrieve document-grounded evidence for review, but it cannot replace certified maintenance processes or expert judgement.

\paragraph{Limitations and future validation.}
The current evidence metrics are document-grounded proxies. Stronger claims require expert review, task-level human evaluation, prospective maintenance scenarios, robustness checks on additional aircraft manuals, OCR/image-only evidence recovery, and a validated separation between retrieval support and operational safety decisions.
"""
    (PAPER_DIR / "fmeca_discussion_narrative.tex").write_text(discussion, encoding="utf-8")


def write_captions() -> None:
    captions = []
    for fig in FIGURES:
        fid = fig["Figure ID"]
        title = Path(fig["File Path"]).stem.replace("_", " ").title()
        captions.append(
            f"## Figure {fid}. {title}.\n"
            f"This figure shows {fig['Type']} for FD-GRAG-RC-V21 using {fig['Source Data']}.\n"
            f"The result indicates {fig['Key Message']}\n"
            f"This supports the methodological claim that FD-GRAG-RC-V21 should be evaluated as structured evidence-package retrieval.\n"
            f"However, this should not be interpreted as {fig['Claim Boundary']}\n"
        )
    (REPORT_DIR / "fmeca_png_figure_captions.md").write_text("\n".join(captions), encoding="utf-8")


def write_figure_index() -> None:
    header = "| Figure ID | File Path | Type | Source Data | Main/Supplementary | Key Message | Claim Boundary |\n|---|---|---|---|---|---|---|"
    rows = []
    for fig in FIGURES:
        rows.append(
            "| {Figure ID} | {File Path} | {Type} | {Source Data} | {Main/Supplementary} | {Key Message} | {Claim Boundary} |".format(**{
                k: str(v).replace("|", "\\|") for k, v in fig.items()
            })
        )
    (REPORT_DIR / "fmeca_png_figure_index.md").write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")


def write_final_summary() -> None:
    main_figs = [f for f in FIGURES if f["Main/Supplementary"] == "Main"]
    supp_figs = [f for f in FIGURES if f["Main/Supplementary"] == "Supplementary"]
    lines = [
        "# FMECA PNG Plotting and Writing Final Summary",
        "",
        "## 1. Experimental logic not modified",
        "- No model training, retrieval rerun, QA benchmark regeneration, split editing, test-label tuning, metric rewriting, or algorithm-logic change was performed.",
        "- Existing result CSV files under `04_results/`, benchmark files under `03_benchmark/`, and case retrieval outputs were read only.",
        "",
        "## 2. Generated PNG figures",
    ]
    for fig in FIGURES:
        lines.append(f"- {fig['Figure ID']}: `{fig['File Path']}`")
    lines += [
        "",
        "## 3. Case-study folder",
        f"- All case-study PNGs are stored together in `{rel(CASE_DIR)}/`.",
        "",
        "## 4. Figures suitable for main text",
    ]
    for fig in main_figs:
        lines.append(f"- {fig['Figure ID']}: `{fig['File Path']}`")
    lines += [
        "",
        "## 5. Figures suitable for supplementary material",
    ]
    for fig in supp_figs:
        lines.append(f"- {fig['Figure ID']}: `{fig['File Path']}`")
    lines += [
        "",
        "## 6. Figures or panels constrained by missing data",
        "- Current evaluation summary files do not contain MRR, nDCG@10, MAP@10, or Recall@20. Figure V21-04 therefore uses available R@10/package/grounding metrics; Figure V21-05 uses Recall@1/3/5/10 from existing top-10 evidence outputs only.",
        "- Current evaluation files do not contain every requested FMECA proxy subfield such as `cause_action_joint_hit`; Figure V21-08 uses available proxy fields from an auxiliary proxy table and labels them as document-grounded proxies.",
        "- The exact requested ablation rows `no role tags`, `no FMECA alignment`, `no query intent parser`, and similar module names are not all present as explicit ablation rows. Figure V21-09 plots only available ablation views.",
    ]
    if SKIPPED:
        for item in SKIPPED:
            lines.append(f"- {item}")
    else:
        lines.append("- No full figure was skipped; some metric panels were limited by available columns as noted above.")
    lines += [
        "",
        "## 7. New writing files",
        "- `07_reports/fmeca_model_intro_rewritten.md`",
        "- `paper_sections/fmeca_model_intro_rewritten.tex`",
        "- `07_reports/fmeca_png_figure_captions.md`",
        "- `paper_sections/fmeca_results_narrative.tex`",
        "- `paper_sections/fmeca_discussion_narrative.tex`",
        "- `07_reports/fmeca_claim_boundary_rewritten.md`",
        "- `paper_sections/fmeca_claim_boundary_rewritten.tex`",
        "- `07_reports/fmeca_png_figure_index.md`",
        "- `07_reports/fmeca_png_plotting_and_writing_audit.md`",
        "",
        "## 8. Recall@10 claim discipline",
        "- Yes. The writing and captions explicitly state that Recall@10 is an evidence retrieval metric, not real diagnostic accuracy.",
        "",
        "## 9. FMECA proxy claim discipline",
        "- Yes. Proxy metrics are explicitly described as document-grounded indicators and not safety certification, FMECA certification, or real maintenance effectiveness.",
        "",
        "## 10. Remaining work for stronger journal submission",
        "- Add expert-reviewed maintenance decision evaluation separate from retrieval metrics.",
        "- Add prospective held-out maintenance scenarios and stronger aircraft/model diversity.",
        "- Add current evaluation MRR, nDCG@10, MAP@10, Recall@20, and bootstrap CI files if those metrics are desired in the main paper.",
        "- Add explicit ablation rows for role tags, FMECA alignment, causal edges, query intent parser, warning weighting, and package diversification.",
        "- Validate OCR/image-only pages and cross-page evidence under stricter split controls.",
    ]
    (REPORT_DIR / "fmeca_png_plotting_and_writing_final_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_notebook() -> Path:
    nb_path = NOTEBOOK_DIR / "FMECA_PNG_Figures_and_Writing.ipynb"
    nb = nbformat.v4.new_notebook()
    marker = "FMECA-PNG.01 Plotting-only update scope"
    figure_lines = "\n".join([f"- **{f['Figure ID']}**: `{f['File Path']}`" for f in FIGURES])
    image_blocks = "\n\n".join([f"### {f['Figure ID']}\n\n![](../{f['File Path']})" for f in FIGURES])
    cells = [
        nbformat.v4.new_markdown_cell(
            f"""# {marker}

This appended section is a plotting and writing update only. It reads existing result files and generated PNG figures; it does not rerun retrieval, train models, regenerate benchmarks, tune labels, or alter metric CSV files.
"""
        ),
        nbformat.v4.new_markdown_cell(
            """# FMECA-PNG.02 Available result files and figure audit

See `../07_reports/fmeca_png_plotting_and_writing_audit.md`. Key sources include current evaluation metrics, complete QA evidence chains, case method/query-type metrics, top-10 evidence tables, ablation metrics, and auxiliary proxy/rank metrics where current fields are unavailable.
"""
        ),
        nbformat.v4.new_markdown_cell(
            f"""# FMECA-PNG.03-07 PNG Figure Index

{figure_lines}
"""
        ),
        nbformat.v4.new_markdown_cell(
            f"""# FMECA-PNG.03 Method overview figures

{image_blocks}
"""
        ),
        nbformat.v4.new_markdown_cell(
            """# FMECA-PNG.08 Rewritten model introduction

See `../07_reports/fmeca_model_intro_rewritten.md` and `../paper_sections/fmeca_model_intro_rewritten.tex`.
"""
        ),
        nbformat.v4.new_markdown_cell(
            """# FMECA-PNG.09 Results and discussion narrative

See `../paper_sections/fmeca_results_narrative.tex` and `../paper_sections/fmeca_discussion_narrative.tex`.
"""
        ),
        nbformat.v4.new_markdown_cell(
            """# FMECA-PNG.10 Claim boundary

See `../07_reports/fmeca_claim_boundary_rewritten.md`. Recall@10 is treated as evidence retrieval, not real diagnostic accuracy. FMECA proxy metrics are document-grounded indicators, not certification.
"""
        ),
        nbformat.v4.new_markdown_cell(
            """# FMECA-PNG.11 Figure index

See `../07_reports/fmeca_png_figure_index.md` for file paths, source data, main/supplementary placement, key messages, and claim boundaries.
"""
        ),
    ]
    nb.cells.extend(cells)
    nbformat.write(nb, nb_path)
    return nb_path


def generate_all() -> None:
    summary = audit_inputs()
    write_audit_report(summary)
    figure_method_overview()
    figure_evidence_schema()
    figure_candidate_pool()
    figure_main_performance()
    figure_recall_curves()
    figure_query_heatmap()
    figure_package_completeness()
    figure_proxy_metrics()
    figure_ablation()
    figure_failure_matrix()
    figure_claim_boundary()
    figure_case_studies()
    write_model_intro()
    write_claim_boundary()
    write_results_discussion()
    write_captions()
    write_figure_index()
    notebook_path = append_notebook()
    write_final_summary()
    print(f"Generated {len(FIGURES)} PNG figures")
    print(f"Notebook updated: {rel(notebook_path)}")
    if SKIPPED:
        print("Notes:")
        for item in SKIPPED:
            print(f"- {item}")


if __name__ == "__main__":
    generate_all()
