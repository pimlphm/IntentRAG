from __future__ import annotations

import importlib.util
import json
import math
import shutil
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pypdf import PdfReader
from textwrap import fill


SEED = 20260516

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

PALETTE = {
    "blue": "#0F4D92",
    "teal": "#42949E",
    "violet": "#9A4D8E",
    "gold": "#C99700",
    "grey": "#767676",
    "light": "#D8D8D8",
    "green": "#2E9E44",
    "red": "#B64342",
    "black": "#272727",
}


def find_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "Cases").exists() and (parent / "04_results" / "v21").exists():
            return parent
    raise RuntimeError("Cannot locate V21 package root.")


ROOT = find_root()
CASE_SLUGS = ["Boeing_737", "Cessna_172"]
CASE_AIRCRAFT = {"Boeing_737": "Boeing 737", "Cessna_172": "Cessna 172"}


def import_case_module():
    script = ROOT / "10_scripts" / "build_case_sota_evidence_chain_outputs.py"
    spec = importlib.util.spec_from_file_location("case_sota", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


CASE_MOD = import_case_module()


def apply_publication_style() -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["font.size"] = 14.2
    plt.rcParams["axes.labelsize"] = 15.8
    plt.rcParams["xtick.labelsize"] = 14.2
    plt.rcParams["ytick.labelsize"] = 14.2
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["legend.frameon"] = False


def case_dir(slug: str) -> Path:
    return ROOT / "Cases" / slug


def case_test_dir(slug: str) -> Path:
    return next(p for p in case_dir(slug).iterdir() if p.is_dir() and p.name.startswith("7 "))


def case_step_dir(slug: str, prefix: str) -> Path:
    return next(p for p in case_dir(slug).iterdir() if p.is_dir() and p.name.startswith(prefix))


def setup_output(slug: str) -> Tuple[Path, Path, Path]:
    test = case_test_dir(slug)
    tables = test / "deployment_speed_tables"
    figs = test / "deployment_speed_figures"
    excel = test / "complete_evidence_chain_excel"
    for d in [tables, figs]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
    excel.mkdir(parents=True, exist_ok=True)
    return tables, figs, excel


def style_axis(ax, grid_axis: str = "x") -> None:
    ax.set_facecolor("white")
    ax.grid(False)
    ax.set_axisbelow(True)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(PALETTE["black"])
        ax.spines[s].set_linewidth(0.95)
    for s in ["right", "top"]:
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="both", colors=PALETTE["black"], labelsize=14.2, width=1.05, length=5.0)
    ax.annotate(
        "",
        xy=(1.015, 0.0),
        xytext=(0.0, 0.0),
        xycoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", color=PALETTE["black"], linewidth=1.65, mutation_scale=22),
        annotation_clip=False,
        zorder=20,
    )
    ax.annotate(
        "",
        xy=(0.0, 1.015),
        xytext=(0.0, 0.0),
        xycoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", color=PALETTE["black"], linewidth=1.65, mutation_scale=22),
        annotation_clip=False,
        zorder=20,
    )


def readable_label(text: str, width: int = 16) -> str:
    return fill(str(text), width=width, break_long_words=False)


def spread_positions(values: List[float], lower: float, upper: float, min_gap: float) -> List[float]:
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


def annotate_with_leader(
    ax,
    x: float,
    y: float,
    label: str,
    dx: float,
    dy: float,
    color: str = "white",
    edgecolor: str = "#272727",
    fontsize: float = 10.8,
) -> None:
    ax.annotate(
        label,
        xy=(x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        ha="center",
        va="center",
        fontsize=fontsize,
        weight="bold",
        color="#111111",
        bbox=dict(boxstyle="round,pad=0.24", facecolor=color, edgecolor=edgecolor, linewidth=0.75, alpha=0.98),
        arrowprops=dict(arrowstyle="-|>", color=edgecolor, linewidth=1.45, mutation_scale=16, shrinkA=2, shrinkB=5, relpos=(0.5, 0.5)),
        zorder=5,
    )


def annotate_with_leader_data(
    ax,
    x: float,
    y: float,
    label: str,
    tx: float,
    ty: float,
    color: str = "white",
    edgecolor: str = "#272727",
    fontsize: float = 10.6,
) -> None:
    ax.annotate(
        label,
        xy=(x, y),
        xytext=(tx, ty),
        textcoords="data",
        ha="center",
        va="center",
        fontsize=fontsize,
        weight="bold",
        color="#111111",
        bbox=dict(boxstyle="round,pad=0.23", facecolor=color, edgecolor=edgecolor, linewidth=0.75, alpha=0.98),
        arrowprops=dict(arrowstyle="-|>", color=edgecolor, linewidth=1.45, mutation_scale=16, shrinkA=2, shrinkB=5, relpos=(0.5, 0.5)),
        zorder=5,
        clip_on=False,
    )


def save_plot(fig, out_dir: Path, stem: str, source: pd.DataFrame, caption: str) -> None:
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}.png", dpi=600, bbox_inches="tight", facecolor="white", pad_inches=0.25)
    plt.close(fig)
    source.to_csv(out_dir / f"{stem}_source_data.csv", index=False, encoding="utf-8-sig")
    (out_dir / f"{stem}_caption.md").write_text(caption.strip() + "\n", encoding="utf-8")


def read_case_evidence(slug: str) -> pd.DataFrame:
    step1 = case_step_dir(slug, "1 ")
    ev_path = next(step1.glob("*_evidence_objects.csv"))
    return pd.read_csv(ev_path)


def benchmark_pdf_document(pdf_path: Path, slug: str) -> dict:
    start = time.perf_counter()
    status = "ok"
    page_count = 0
    chars = 0
    try:
        reader = PdfReader(str(pdf_path))
        page_count = len(reader.pages)
        for page in reader.pages:
            try:
                chars += len(page.extract_text() or "")
            except Exception:
                status = "partial_page_extract_error"
    except Exception as exc:
        status = f"error: {type(exc).__name__}"
    elapsed = max(time.perf_counter() - start, 1e-9)
    size_mb = pdf_path.stat().st_size / (1024 * 1024)
    chars_per_page = chars / page_count if page_count else 0
    image_or_scan_likely = chars_per_page < 40
    return {
        "case": slug,
        "aircraft_model": CASE_AIRCRAFT[slug],
        "document": pdf_path.name,
        "document_label": short_document_label(pdf_path.name),
        "extractor": "pypdf",
        "status": status,
        "size_mb": size_mb,
        "pages": page_count,
        "extracted_chars": chars,
        "chars_per_page": chars_per_page,
        "elapsed_seconds": elapsed,
        "pages_per_second": page_count / elapsed if elapsed else np.nan,
        "mb_per_second": size_mb / elapsed if elapsed else np.nan,
        "chars_per_second": chars / elapsed if elapsed else np.nan,
        "image_or_scan_likely": image_or_scan_likely,
    }


def short_document_label(name: str) -> str:
    lower = name.lower()
    if "737" in lower:
        return "Boeing 737 fault manual"
    if "parts_catalog" in lower or "parts catalog" in lower:
        return "Cessna parts catalog"
    if "service_manual" in lower or "service manual" in lower:
        return "Cessna service manual"
    if "maintenance_manual" in lower or "maintenance manual" in lower:
        return "Cessna maintenance manual"
    if "poh" in lower or "information_manual" in lower:
        return "Cessna POH"
    return Path(name).stem[:52]


def document_throughput(slug: str) -> pd.DataFrame:
    raw_dir = case_step_dir(slug, "1 ") / "raw_documents"
    pdfs = sorted(raw_dir.glob("*.pdf"))
    rows = [benchmark_pdf_document(pdf, slug) for pdf in pdfs]
    evidence = read_case_evidence(slug)
    grouped = evidence.groupby("file_name", dropna=False).agg(
        evidence_objects=("evidence_id", "nunique"),
        evidence_chars=("text", lambda s: int(s.fillna("").astype(str).str.len().sum())),
    ).reset_index()
    df = pd.DataFrame(rows)
    ev_counts = []
    ev_chars = []
    for doc in df["document"]:
        stem = Path(doc).stem.lower()
        sub = grouped[grouped["file_name"].fillna("").str.lower().str.contains(stem[:32], regex=False)]
        if sub.empty:
            sub = grouped[grouped["file_name"].fillna("").str.lower().str.contains(stem.split("_")[0].lower(), regex=False)]
        ev_counts.append(int(sub["evidence_objects"].sum()) if not sub.empty else 0)
        ev_chars.append(int(sub["evidence_chars"].sum()) if not sub.empty else 0)
    df["evidence_objects"] = ev_counts
    df["evidence_chars"] = ev_chars
    df["evidence_objects_per_page"] = df["evidence_objects"] / df["pages"].replace(0, np.nan)
    return df


def clear_retriever_caches(retriever) -> None:
    for name in ["_base_cache", "_component_cache", "_section_cache", "_role_cache", "_graph_cache", "_coarse_graph_cache", "_method_cache"]:
        if hasattr(retriever, name):
            getattr(retriever, name).clear()


def scores_for_method(retriever, question: str, method: str) -> np.ndarray:
    if method in {"BM25/word", "TFIDF-char", "Hybrid lexical"}:
        base = retriever.base_scores(question)
        if method == "BM25/word":
            return base["word"]
        if method == "TFIDF-char":
            return base["char"]
        return CASE_MOD.bounded_score(0.40 * base["word"] + 0.40 * base["char"] + 0.20 * base["svd"])
    if method == "FD-GRAG-RC-V21-final":
        a = retriever.method_scores(question, "Agentic RAG")
        e = retriever.method_scores(question, "Evidence-flow baseline")
        g = retriever.method_scores(question, "Controlled GraphRAG")
        attr = retriever.method_scores(question, "GaRAGe-style attribution RAG")
        return CASE_MOD.bounded_score(0.34 * a + 0.28 * e + 0.24 * g + 0.14 * attr)
    return retriever.method_scores(question, method)


def select_top10(retriever, question: str, method: str) -> List[str]:
    scores = scores_for_method(retriever, question, method)
    if method in {"FD-GRAG-RC-V21-final", "Evidence-flow baseline", "Agentic RAG"}:
        idx = CASE_MOD.diverse_top_indices(scores, retriever.evidence, k=10)
    elif method == "Controlled GraphRAG":
        idx = CASE_MOD.diverse_top_indices(scores, retriever.evidence, k=10, page_weight=0.05, role_weight=0.0)
    else:
        idx = CASE_MOD.top_indices(scores, k=10)
    return [retriever.evidence_ids[i] for i in idx]


def deployment_benchmark(slug: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    qa = pd.read_csv(ROOT / "04_results" / "v21" / "qa_warehouse_v21_full.csv")
    queries = qa.loc[qa["aircraft_model"] == CASE_AIRCRAFT[slug], "question"].fillna("").astype(str).tolist()
    metrics = pd.read_csv(case_test_dir(slug) / "sota_evidence_chain_tables" / f"{slug}_sota_metrics_by_method.csv")

    load_start = time.perf_counter()
    evidence, profiles, causal, aliases = CASE_MOD.load_case_evidence(slug)
    corpus_load_seconds = time.perf_counter() - load_start

    build_start = time.perf_counter()
    retriever = CASE_MOD.CaseRetriever(evidence, aliases, causal)
    index_build_seconds = time.perf_counter() - build_start

    rows = []
    for method in METHODS:
        clear_retriever_caches(retriever)
        first_start = time.perf_counter()
        _ = select_top10(retriever, queries[0], method)
        first_query_ms = (time.perf_counter() - first_start) * 1000

        clear_retriever_caches(retriever)
        times = []
        batch_start = time.perf_counter()
        for q in queries:
            t0 = time.perf_counter()
            _ = select_top10(retriever, q, method)
            times.append((time.perf_counter() - t0) * 1000)
        batch_seconds = time.perf_counter() - batch_start
        arr = np.asarray(times, dtype=float)
        rows.append(
            {
                "case": slug,
                "aircraft_model": CASE_AIRCRAFT[slug],
                "method": method,
                "method_display": DISPLAY[method],
                "method_id": METHOD_ID[method],
                "method_short": SHORT_NAME[method],
                "n_queries": len(queries),
                "corpus_load_seconds_case": corpus_load_seconds,
                "index_build_seconds_case": index_build_seconds,
                "deployment_ready_seconds_case": corpus_load_seconds + index_build_seconds,
                "first_query_latency_ms": first_query_ms,
                "mean_query_latency_ms": float(arr.mean()),
                "median_query_latency_ms": float(np.median(arr)),
                "p95_query_latency_ms": float(np.quantile(arr, 0.95)),
                "batch_seconds_all_queries": batch_seconds,
                "queries_per_second": len(queries) / batch_seconds if batch_seconds else np.nan,
                "runtime_scope": "route-level runtime over V21 evidence index; final FD-GRAG speed is measured by its V21 route/package proxy",
            }
        )
    speed = pd.DataFrame(rows)
    speed = speed.merge(metrics[["method", "accepted_recall_10", "evidence_package_recall_10", "grounded_score", "hallucination_risk_proxy"]], on="method", how="left")
    stages = pd.DataFrame(
        [
            {
                "case": slug,
                "aircraft_model": CASE_AIRCRAFT[slug],
                "stage": "case evidence load",
                "seconds": corpus_load_seconds,
            },
            {
                "case": slug,
                "aircraft_model": CASE_AIRCRAFT[slug],
                "stage": "retrieval index build",
                "seconds": index_build_seconds,
            },
            {
                "case": slug,
                "aircraft_model": CASE_AIRCRAFT[slug],
                "stage": "deployment ready",
                "seconds": corpus_load_seconds + index_build_seconds,
            },
        ]
    )
    return speed, stages


def write_excel(path: Path, sheets: Dict[str, pd.DataFrame]) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)
        wb = writer.book
        for ws in wb.worksheets:
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions
            for col in ws.columns:
                width = min(45, max(10, max((len(str(c.value)) if c.value is not None else 0) for c in col[:100]) + 2))
                ws.column_dimensions[col[0].column_letter].width = width


def plot_document_throughput(slug: str, out_dir: Path, docs: pd.DataFrame) -> None:
    df = docs.sort_values("pages_per_second")
    fig, ax = plt.subplots(figsize=(10.4, max(3.4, 0.58 * len(df) + 1.3)))
    y = np.arange(len(df))
    colors = np.where(df["image_or_scan_likely"], PALETTE["gold"], PALETTE["blue"])
    ax.barh(y, df["pages_per_second"], color=colors, edgecolor=PALETTE["black"], linewidth=0.5)
    ax.set_yticks(y)
    if "document_label" not in df.columns:
        df["document_label"] = df["document"].map(short_document_label)
    labels = [readable_label(x, 24) for x in df["document_label"].tolist()]
    ax.set_yticklabels(labels, fontsize=13.0)
    xmax = max(float(df["pages_per_second"].max()) * 1.20, 1.0)
    ax.set_xlim(0, xmax)
    for yi, (_, row) in zip(y, df.iterrows()):
        val = float(row["pages_per_second"])
        ax.text(
            val + xmax * 0.015,
            yi,
            f"{val:.1f}",
            ha="left",
            va="center",
            fontsize=15.0,
            weight="bold",
            color=PALETTE["black"],
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.86, pad=1.0),
        )
    ax.set_xlabel("PDF text extraction throughput (pages/s)", fontsize=15.8, labelpad=8)
    ax.set_ylabel("")
    ax.tick_params(axis="y", labelsize=13.0)
    style_axis(ax)
    save_plot(fig, out_dir, f"{slug}_document_pdf_throughput", df, "Per-document PDF text extraction throughput measured with pypdf.")


def plot_deployment_pareto(slug: str, out_dir: Path, speed: pd.DataFrame) -> None:
    df = speed.copy()
    fig, ax = plt.subplots(figsize=(11.4, 7.5))
    fig.subplots_adjust(left=0.12, right=0.96, bottom=0.15, top=0.94)
    df = df.sort_values(["median_query_latency_ms", "grounded_score"]).reset_index(drop=True)
    x_min = max(1.0, float(df["median_query_latency_ms"].min()) * 0.42)
    x_max = float(df["median_query_latency_ms"].max()) * 1.72
    y_lower = max(0.82, float(df["grounded_score"].min()) - 0.020)
    y_upper = min(1.022, float(df["grounded_score"].max()) + 0.018)
    colors = [
        PALETTE["blue"]
        if m == "FD-GRAG-RC-V21-final"
        else PALETTE["teal"]
        if "Graph" in m or m in {"Evidence-flow baseline", "Agentic RAG"}
        else PALETTE["violet"]
        if "GaRAGe" in m
        else PALETTE["grey"]
        for m in df["method"]
    ]
    ax.scatter(df["median_query_latency_ms"], df["grounded_score"], s=112, color=colors, edgecolor=PALETTE["black"], linewidth=0.75, zorder=4, alpha=0.95)
    ax.set_xscale("log")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_lower, y_upper)

    label_text = {
        "Parent-section": "Parent\nsection",
        "Component-profile": "Component\nprofile",
        "Component-graph": "Comp.\ngraph",
        "Causal-graph": "Causal\ngraph",
        "Char TF-IDF": "Char\nTF-IDF",
        "SVD surrogate": "SVD",
        "Evidence-flow": "Evidence\nflow",
        "GaRAGe-deflect": "GaRAGe\ndeflect",
    }
    # Data-space label coordinates are used here deliberately: the log-scaled
    # latency axis produces dense clusters, and fixed near-point positions keep
    # all leader lines short while labeling every circle.
    label_positions = {
        "GaRAGe-style deflective RAG": (7.8, 1.014, "right"),
        "GaRAGe-style relevance RAG": (7.8, 1.002, "right"),
        "GaRAGe-style attribution RAG": (7.8, 0.989, "right"),
        "BM25/word": (7.8, 0.976, "right"),
        "TFIDF-char": (13.4, 1.011, "left"),
        "Hybrid lexical": (13.4, 0.989, "left"),
        "Component-graph informed RAG": (24.2, 0.905, "left"),
        "Causal-graph informed RAG": (24.2, 0.960, "left"),
        "Parent-section retrieval": (24.2, 0.971, "left"),
        "SVD semantic surrogate": (24.2, 1.001, "left"),
        "Component-profile retrieval": (24.2, 0.982, "left"),
        "FD-GRAG-RC-V21-final": (425.0, 1.004, "left"),
        "Evidence-flow baseline": (560.0, 0.982, "left"),
        "Agentic RAG": (1375.0, 0.966, "left"),
        "Controlled GraphRAG": (1480.0, 0.989, "left"),
    }
    label_df = df.sort_values(["median_query_latency_ms", "grounded_score"]).reset_index(drop=True)
    for _, row in label_df.iterrows():
        tx, ty, ha = label_positions.get(
            str(row["method"]),
            (float(row["median_query_latency_ms"]) * 1.18, float(row["grounded_score"]) + 0.006, "left"),
        )
        tx = min(max(float(tx), x_min * 1.04), x_max / 1.05)
        ty = min(max(float(ty), y_lower + 0.006), y_upper - 0.006)
        face = "#E8F1FF" if row["method"] == "FD-GRAG-RC-V21-final" else "white"
        text = label_text.get(str(row["method_short"]), str(row["method_short"]))
        ax.annotate(
            text,
            xy=(float(row["median_query_latency_ms"]), float(row["grounded_score"])),
            xycoords="data",
            xytext=(tx, ty),
            textcoords="data",
            ha=ha,
            va="center",
            fontsize=10.9,
            weight="bold",
            color="#111111",
            bbox=dict(boxstyle="round,pad=0.18", facecolor=face, edgecolor="#59616C", linewidth=0.72, alpha=0.98),
            arrowprops=dict(
                arrowstyle="-",
                color="#68707C",
                linewidth=0.9,
                linestyle=(0, (2.2, 2.0)),
                shrinkA=2,
                shrinkB=5,
                connectionstyle="arc3,rad=0",
            ),
            annotation_clip=False,
            zorder=6,
        )
    ax.plot([x_min, min(x_max, x_min * 32.0)], [0.9, 0.9], color=PALETTE["green"], linestyle=(0, (2.2, 2.0)), linewidth=0.9)
    ax.set_xlabel("Median query latency (ms)", fontsize=15.8, labelpad=8)
    ax.set_ylabel("Grounded score", fontsize=15.8, labelpad=8)
    style_axis(ax)
    save_plot(fig, out_dir, f"{slug}_deployment_latency_quality_pareto", df, "Deployment Pareto plot: median query latency versus grounded score.")


def plot_latency_bars(slug: str, out_dir: Path, speed: pd.DataFrame) -> None:
    df = speed.sort_values("median_query_latency_ms")
    fig, ax = plt.subplots(figsize=(11.2, 8.2))
    y = np.arange(len(df))
    ax.barh(y, df["median_query_latency_ms"], color=[PALETTE["blue"] if m == "FD-GRAG-RC-V21-final" else PALETTE["grey"] for m in df["method"]], edgecolor=PALETTE["black"], linewidth=0.45)
    for yi, p95 in zip(y, df["p95_query_latency_ms"]):
        ax.vlines(float(p95), yi - 0.24, yi + 0.24, color=PALETTE["red"], linewidth=1.7)
    ax.set_yticks(y)
    ax.set_yticklabels([readable_label(x, 17) for x in df["method_short"]], fontsize=13.0)
    ax.set_xscale("log")
    xmax = max(float(df["p95_query_latency_ms"].max()) * 1.9, 10.0)
    ax.set_xlim(max(0.6, float(df["median_query_latency_ms"].min()) * 0.55), xmax)
    for yi, (_, row) in zip(y, df.iterrows()):
        med = float(row["median_query_latency_ms"])
        ax.text(
            med * 1.08,
            yi,
            f"{med:.1f}",
            va="center",
            ha="left",
            fontsize=14.4,
            weight="bold",
            color=PALETTE["black"],
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.86, pad=0.8),
        )
    ax.set_xlabel("Per-query latency (ms)", fontsize=15.8, labelpad=8)
    ax.set_ylabel("")
    ax.text(
        0.985,
        0.04,
        "red tick = p95 latency",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=12.0,
        color=PALETTE["black"],
        bbox=dict(facecolor="white", edgecolor="#B9C0CA", linewidth=0.5, alpha=0.92, pad=2.2),
    )
    style_axis(ax)
    save_plot(fig, out_dir, f"{slug}_method_latency_bars", df, "Median per-query latency with p95 markers.")


def plot_deployment_stages(slug: str, out_dir: Path, stages: pd.DataFrame) -> None:
    df = stages.copy()
    stage_label = {
        "case evidence load": "Evidence\nload",
        "retrieval index build": "Index\nbuild",
        "deployment ready": "Ready\nend-to-end",
    }
    df["stage_label"] = df["stage"].map(stage_label).fillna(df["stage"])
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    y = np.arange(len(df))
    ax.barh(y, df["seconds"], color=[PALETTE["grey"], PALETTE["teal"], PALETTE["blue"]], edgecolor=PALETTE["black"], linewidth=0.55)
    ax.set_yticks(y)
    ax.set_yticklabels(df["stage_label"], fontsize=13.0, weight="bold")
    xmax = max(float(df["seconds"].max()) * 1.22, 0.5)
    ax.set_xlim(0, xmax)
    for yi, (_, row) in zip(y, df.iterrows()):
        ax.text(
            float(row["seconds"]) + xmax * 0.015,
            yi,
            f"{float(row['seconds']):.2f} s",
            va="center",
            ha="left",
            fontsize=15.0,
            weight="bold",
            color=PALETTE["black"],
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=1.0),
        )
    ax.set_xlabel("Seconds", fontsize=15.8, labelpad=8)
    ax.set_ylabel("")
    style_axis(ax, grid_axis="x")
    save_plot(fig, out_dir, f"{slug}_deployment_stage_breakdown", df, "Case-level cold deployment stage timing.")


def write_table_bundle(out_dir: Path, stem: str, df: pd.DataFrame) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{stem}.csv", index=False, encoding="utf-8-sig")
    preview = df.head(80)
    lines = ["| " + " | ".join(preview.columns) + " |", "| " + " | ".join(["---"] * len(preview.columns)) + " |"]
    for _, row in preview.iterrows():
        vals = []
        for col in preview.columns:
            val = str(row[col]).replace("|", "\\|").replace("\n", "<br>")
            if len(val) > 220:
                val = val[:217] + "..."
            vals.append(val)
        lines.append("| " + " | ".join(vals) + " |")
    (out_dir / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_case(slug: str) -> dict:
    tables_dir, figs_dir, excel_dir = setup_output(slug)
    docs = document_throughput(slug)
    speed, stages = deployment_benchmark(slug)
    legend = pd.DataFrame({"method": METHODS, "method_short": [SHORT_NAME[m] for m in METHODS], "method_display": [DISPLAY[m] for m in METHODS]})

    write_table_bundle(tables_dir, f"{slug}_document_throughput_by_pdf", docs)
    write_table_bundle(tables_dir, f"{slug}_deployment_speed_by_method", speed)
    write_table_bundle(tables_dir, f"{slug}_deployment_stage_breakdown", stages)
    write_table_bundle(tables_dir, f"{slug}_deployment_method_legend", legend)

    write_excel(
        excel_dir / f"{slug}_deployment_throughput_benchmark.xlsx",
        {
            "Document_Throughput": docs,
            "Deployment_Speed": speed,
            "Deployment_Stages": stages,
            "Method_Legend": legend,
        },
    )

    plot_document_throughput(slug, figs_dir, docs)
    plot_deployment_pareto(slug, figs_dir, speed)
    plot_latency_bars(slug, figs_dir, speed)
    plot_deployment_stages(slug, figs_dir, stages)

    code_dir = case_dir(slug) / "sota_evidence_chain_code"
    code_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__).resolve(), code_dir / "benchmark_v21_throughput_deployment.py")
    return {
        "case": slug,
        "document_rows": len(docs),
        "method_rows": len(speed),
        "table_dir": str(tables_dir),
        "figure_dir": str(figs_dir),
        "excel": str(excel_dir / f"{slug}_deployment_throughput_benchmark.xlsx"),
    }


def main() -> None:
    apply_publication_style()
    outputs = [build_case(slug) for slug in CASE_SLUGS]
    report = ROOT / "07_reports" / "v21" / "v21_throughput_deployment_benchmark_report.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# V21 throughput and deployment benchmark",
        "",
        "This benchmark adds engineering deployability evidence to the V21 package.",
        "",
        "## Measurement scope",
        "",
        "- PDF document throughput is measured with `pypdf` because PyMuPDF and pdfplumber were not available in the bundled runtime.",
        "- Deployment speed includes case evidence loading, retrieval index construction, first-query latency, all-query batch latency, median latency, p95 latency and QPS.",
        "- The FD-GRAG-RC-V21 speed row is measured with the V21 route/package runtime proxy over the same evidence index; effectiveness metrics are still read from the V21 result CSV.",
        "",
        "## Outputs",
    ]
    for out in outputs:
        lines += [
            "",
            f"### {out['case']}",
            f"- Documents measured: {out['document_rows']}",
            f"- Methods measured: {out['method_rows']}",
            f"- Tables: `{out['table_dir']}`",
            f"- Figures: `{out['figure_dir']}`",
            f"- Excel: `{out['excel']}`",
        ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = ROOT / "04_results" / "v21" / "v21_throughput_deployment_benchmark_manifest.json"
    manifest.write_text(json.dumps(outputs, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(outputs, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
