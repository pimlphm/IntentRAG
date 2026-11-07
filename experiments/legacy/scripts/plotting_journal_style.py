from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from textwrap import fill
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
CASE_IDS = ["Boeing_737", "Cessna_172"]
DPI = 600

METHOD_ORDER = [
    "BM25",
    "Char TF-IDF",
    "SVD surrogate",
    "Hybrid",
    "Parent-section",
    "Component-profile",
    "GraphRAG",
    "Causal-graph",
    "Component-graph",
    "Agentic RAG",
    "Evidence-flow",
    "GaRAGe-rel",
    "GaRAGe-attr",
    "GaRAGe-deflect",
    "FD-GRAG-RC",
]

METHOD_CANONICAL = {
    "BM25/word": "BM25",
    "BM25 lexical": "BM25",
    "TFIDF-char": "Char TF-IDF",
    "TF-IDF character": "Char TF-IDF",
    "SVD semantic surrogate": "SVD surrogate",
    "Hybrid lexical": "Hybrid",
    "Hybrid lexical-semantic": "Hybrid",
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
    "FD-GRAG-RC-V21": "FD-GRAG-RC",
}

ROLE_LABELS = {
    "warning": "Warning",
    "limitation": "Limitation",
    "threshold": "Threshold",
    "condition": "Condition",
    "confirmed_cause": "Confirmed cause",
    "excluded_cause": "Excluded cause",
    "action": "Action",
    "inspection": "Inspection",
    "replacement": "Replacement",
    "authority": "Authority",
}

COLORS = {
    "fd": "#0F4D92",
    "teal": "#2F8F9D",
    "grey": "#9A9A9A",
    "dark_grey": "#333333",
    "light_grey": "#E8EAED",
    "red": "#D95F45",
    "soft_red": "#F4B6A6",
    "green": "#2E9D57",
    "soft_green": "#DFF2E5",
    "orange": "#D99B2B",
    "soft_orange": "#F7E2B7",
    "purple": "#9A4D8E",
}


def apply_journal_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans"],
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.titlesize": 13,
            "axes.labelsize": 11.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.5,
            "axes.linewidth": 0.9,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def case_dir(case_id: str) -> Path:
    return ROOT / "Cases" / case_id


def latest_dir(case_id: str) -> Path:
    return next(p for p in case_dir(case_id).iterdir() if p.is_dir() and p.name.startswith("8"))


def test_dir(case_id: str) -> Path:
    return next(p for p in case_dir(case_id).iterdir() if p.is_dir() and p.name.startswith("7"))


def table_dir(case_id: str) -> Path:
    return test_dir(case_id) / "sota_evidence_chain_tables"


def output_dir(case_id: str) -> Path:
    out = ROOT / "figures" / case_id
    out.mkdir(parents=True, exist_ok=True)
    return out


def reports_dir() -> Path:
    out = ROOT / "reports"
    out.mkdir(parents=True, exist_ok=True)
    return out


def find_artifact(case_id: str, pattern: str) -> Path:
    places = [
        latest_dir(case_id) / "_support_files",
        test_dir(case_id),
        case_dir(case_id),
    ]
    for base in places:
        hits = sorted(base.rglob(pattern))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"No artifact matching {pattern} for {case_id}")


def old_png(case_id: str, name: str) -> Path:
    return latest_dir(case_id) / f"{case_id}_{name}.png"


def wrap_labels(labels: Iterable[str], width: int = 16) -> list[str]:
    return [fill(str(x), width=width, break_long_words=False) for x in labels]


def canon_method(name: str) -> str:
    return METHOD_CANONICAL.get(str(name), str(name))


def method_color(method: str) -> str:
    method = canon_method(method)
    if method == "FD-GRAG-RC":
        return COLORS["fd"]
    if method in {"GraphRAG", "Causal-graph", "Component-graph", "Evidence-flow", "Agentic RAG"}:
        return COLORS["teal"]
    if method in {"GaRAGe-rel", "GaRAGe-attr", "GaRAGe-deflect"}:
        return COLORS["purple"]
    return COLORS["grey"]


def clean_spines(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)
    ax.spines["left"].set_color(COLORS["dark_grey"])
    ax.spines["bottom"].set_color(COLORS["dark_grey"])
    ax.tick_params(colors=COLORS["dark_grey"], width=0.8, length=3.5)
    ax.xaxis.label.set_color(COLORS["dark_grey"])
    ax.yaxis.label.set_color(COLORS["dark_grey"])


def panel_label(ax, label: str = "(a)") -> None:
    ax.text(0.0, 1.02, label, transform=ax.transAxes, ha="left", va="bottom", weight="bold", fontsize=12)


def format_metric_axis(ax, xlim: tuple[float, float] | None = None) -> None:
    if xlim:
        ax.set_xlim(*xlim)
    ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.2f"))
    ax.grid(axis="x", color="#D8DDE3", linewidth=0.55, alpha=0.7)
    clean_spines(ax)


def annotate_selected_values(ax, xs, ys, labels, colors=None, xoffset=0.004, fontsize=8.5) -> None:
    colors = colors or [COLORS["dark_grey"]] * len(labels)
    for x, y, lab, col in zip(xs, ys, labels, colors):
        ax.text(x + xoffset, y, lab, va="center", ha="left", fontsize=fontsize, color=col)


def save_png(
    fig,
    out_path: Path,
    *,
    source_artifact: Path | list[Path],
    notes: str,
    sidecar: dict,
    manifest: list[dict],
    dpi: int = DPI,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.06, facecolor="white")
    plt.close(fig)
    with Image.open(out_path) as img:
        width, height = img.size
    sources = [str(p) for p in source_artifact] if isinstance(source_artifact, list) else [str(source_artifact)]
    sidecar_payload = {
        "output_png": str(out_path),
        "source_artifact": sources,
        "created_time": datetime.now().isoformat(timespec="seconds"),
        "dpi": dpi,
        "width": width,
        "height": height,
        "notes": notes,
        **sidecar,
    }
    sidecar_path = out_path.with_suffix(".json")
    sidecar_path.write_text(json.dumps(sidecar_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest.append(
        {
            "case_id": out_path.parent.name,
            "figure_name": out_path.name,
            "source_artifact": "; ".join(sources),
            "output_png": str(out_path),
            "dpi": dpi,
            "width": width,
            "height": height,
            "created_time": sidecar_payload["created_time"],
            "notes": notes,
        }
    )


def check_png_basic_quality(output_png: Path) -> dict:
    result = {"output_png": str(output_png), "exists": output_png.exists()}
    if not output_png.exists():
        result.update({"ok": False, "reason": "missing"})
        return result
    size_kb = output_png.stat().st_size / 1024
    with Image.open(output_png) as img:
        width, height = img.size
        arr = np.asarray(img.convert("RGB"))
    nonwhite = np.any(arr < 245, axis=2)
    coords = np.argwhere(nonwhite)
    if coords.size:
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)
        content_area = max(1, (y1 - y0 + 1) * (x1 - x0 + 1))
        blank_border_fraction = 1 - content_area / (width * height)
    else:
        blank_border_fraction = 1.0
    ok = size_kb > 50 and width >= 900 and height >= 500 and blank_border_fraction < 0.82
    result.update(
        {
            "ok": bool(ok),
            "size_kb": round(size_kb, 1),
            "width": int(width),
            "height": int(height),
            "blank_border_fraction": round(float(blank_border_fraction), 3),
        }
    )
    return result


def metric_tables(case_id: str) -> dict[str, pd.DataFrame]:
    td = table_dir(case_id)
    return {
        "metrics": pd.read_csv(td / f"{case_id}_sota_metrics_by_method.csv"),
        "query": pd.read_csv(td / f"{case_id}_sota_metrics_by_query_type.csv"),
        "track": pd.read_csv(td / f"{case_id}_sota_metrics_by_track.csv"),
        "ablation": pd.read_csv(td / f"{case_id}_ablation_metrics.csv"),
        "noev": pd.read_csv(td / f"{case_id}_no_evidence_probe_metrics.csv"),
        "pred": pd.read_csv(td / f"{case_id}_complete_sota_evidence_chains.csv", usecols=[
            "query_id", "track", "query_type", "method", "accepted_recall_10",
            "evidence_package_recall_10", "grounded_score", "hallucination_risk_proxy",
            "required_slots", "exact_recall_10", "accepted_page_recall_10", "accepted_section_recall_10",
        ]),
    }


def order_methods(df: pd.DataFrame, col: str = "method_short") -> pd.DataFrame:
    out = df.copy()
    out[col] = out[col].map(canon_method) if col in out.columns else out[col]
    out["_order"] = out[col].map({m: i for i, m in enumerate(METHOD_ORDER)})
    return out.sort_values("_order").drop(columns=["_order"])


def plot_metric_heatmap(df: pd.DataFrame) -> tuple:
    metrics = [
        ("exact_recall_10", "Exact R@10"),
        ("accepted_recall_10", "Accepted R@10"),
        ("evidence_package_recall_10", "Package R@10"),
        ("grounded_score", "Grounded"),
        ("hallucination_risk_proxy", "1 - Risk"),
    ]
    data = df.copy()
    data["method_short"] = data.get("method_short", data["method"]).map(canon_method)
    data = order_methods(data, "method_short")
    mat = pd.DataFrame({"method": data["method_short"]})
    for col, lab in metrics:
        if col == "hallucination_risk_proxy":
            mat[lab] = 1 - data[col].astype(float)
        elif col in data:
            mat[lab] = data[col].astype(float)
    mat = mat.drop_duplicates("method").set_index("method")
    vals = mat.to_numpy(float)
    vmin = 0.85 if np.nanmin(vals) >= 0.85 else max(0.0, np.nanmin(vals) - 0.02)
    if np.nanmin(vals) >= 0.95:
        display = 1 - vals
        cmap = mpl.colormaps["OrRd"]
        label = "1 - score"
        vmin, vmax = 0, max(0.012, float(np.nanmax(display)))
    else:
        display = vals
        cmap = mpl.colormaps["YlGnBu"]
        label = "Evidence score"
        vmax = 1.0
    fig, ax = plt.subplots(figsize=(7.2, 5.9))
    im = ax.imshow(display, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(mat.shape[1]))
    ax.set_xticklabels(wrap_labels(mat.columns, 12), rotation=28, ha="right")
    ax.set_yticks(np.arange(mat.shape[0]))
    ax.set_yticklabels(mat.index)
    fd_idx = list(mat.index).index("FD-GRAG-RC") if "FD-GRAG-RC" in mat.index else None
    best = np.nanmax(vals, axis=0)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            raw = vals[i, j]
            show = mat.index[i] == "FD-GRAG-RC" or raw < 0.95 or abs(raw - best[j]) < 1e-12
            if show:
                weight = "bold" if abs(raw - best[j]) < 1e-12 else "normal"
                ax.text(j, i, f"{raw:.3f}", ha="center", va="center", fontsize=7.6, weight=weight, color="#222222")
    if fd_idx is not None:
        ax.add_patch(mpl.patches.Rectangle((-0.5, fd_idx - 0.5), mat.shape[1], 1, fill=False, edgecolor=COLORS["fd"], linewidth=1.1))
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.015)
    cbar.set_label(label)
    cbar.ax.tick_params(labelsize=8.5)
    ax.set_title("")
    panel_label(ax)
    clean_spines(ax)
    return fig, ax, {"transformation": "risk inverted as 1-risk; saturated tables use 1-score heatmap" if np.nanmin(vals) >= 0.95 else "risk inverted as 1-risk", "metric_columns": [m[0] for m in metrics if m[0] in df.columns]}


def plot_query_family_heatmap(df: pd.DataFrame) -> tuple:
    data = df.copy()
    if "method_short" in data.columns:
        data["method_short"] = data["method_short"].map(canon_method)
    else:
        data["method_short"] = data["method"].map(canon_method)
    metric_col = "grounded_score" if "grounded_score" in data.columns else "accepted_recall_10"
    piv = data.pivot_table(index="method_short", columns="query_type", values=metric_col, aggfunc="mean")
    piv = piv.reindex(METHOD_ORDER).dropna(how="all")
    cols = sorted(piv.columns, key=lambda c: float(piv[c].mean()))
    piv = piv[cols]
    vals = piv.to_numpy(float)
    vmin = 0.85 if np.nanmin(vals) >= 0.85 else max(0, np.nanmin(vals) - 0.02)
    fig, ax = plt.subplots(figsize=(9.2, 6.2))
    im = ax.imshow(vals, aspect="auto", cmap="YlGnBu", vmin=vmin, vmax=1.0)
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(wrap_labels(cols, 14), rotation=32, ha="right")
    ax.set_yticks(np.arange(piv.shape[0]))
    ax.set_yticklabels(piv.index)
    fd_idx = list(piv.index).index("FD-GRAG-RC") if "FD-GRAG-RC" in piv.index else None
    best = np.nanmax(vals, axis=0)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            raw = vals[i, j]
            show = piv.index[i] == "FD-GRAG-RC" or raw < 0.95 or abs(raw - best[j]) < 1e-12
            if show:
                ax.text(j, i, f"{raw:.2f}", ha="center", va="center", fontsize=7.1, weight="bold" if abs(raw - best[j]) < 1e-12 else "normal")
    if fd_idx is not None:
        ax.add_patch(mpl.patches.Rectangle((-0.5, fd_idx - 0.5), piv.shape[1], 1, fill=False, edgecolor=COLORS["fd"], linewidth=1.1))
    cbar = fig.colorbar(im, ax=ax, fraction=0.028, pad=0.012)
    cbar.set_label("Evidence score")
    cbar.ax.tick_params(labelsize=8.5)
    panel_label(ax)
    clean_spines(ax)
    return fig, ax, {"metric_columns": [metric_col], "transformation": "local color scale"}


def plot_primary_metric_forest(df: pd.DataFrame, key_only: bool = False) -> tuple:
    data = df.copy()
    data["method_short"] = data.get("method_short", data["method"]).map(canon_method)
    data = order_methods(data, "method_short")
    if key_only:
        keep = {"BM25", "Char TF-IDF", "GraphRAG", "Agentic RAG", "Evidence-flow", "GaRAGe-deflect", "FD-GRAG-RC"}
        data = data[data["method_short"].isin(keep)]
    metric_cols = [
        ("accepted_recall_10", "Accepted R@10", "o"),
        ("evidence_package_recall_10", "Package R@10", "s"),
        ("grounded_score", "Grounded", "D"),
        ("hallucination_risk_proxy", "1 - Risk", "^"),
    ]
    ybase = np.arange(len(data))[::-1]
    fig, ax = plt.subplots(figsize=(7.8, max(3.8, 0.34 * len(data) + 1.4)))
    offsets = np.linspace(-0.15, 0.15, len(metric_cols))
    for off, (col, lab, marker) in zip(offsets, metric_cols):
        if col not in data:
            continue
        vals = (1 - data[col]) if col == "hallucination_risk_proxy" else data[col]
        colors = [method_color(m) if m == "FD-GRAG-RC" else "#B7BBC0" for m in data["method_short"]]
        ax.scatter(vals, ybase + off, s=42, marker=marker, color=colors, edgecolor="#333333", linewidth=0.45, label=lab, zorder=3)
    ax.axvline(0.90, color=COLORS["green"], linestyle="--", linewidth=0.8, alpha=0.65)
    ax.set_yticks(ybase)
    ax.set_yticklabels(data["method_short"])
    min_val = min(float(data[c].min() if c != "hallucination_risk_proxy" else (1 - data[c]).min()) for c, _, _ in metric_cols if c in data)
    ax.set_xlim(max(0, min_val - 0.02), 1.005)
    ax.set_xlabel("Evidence score")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=4, frameon=False)
    panel_label(ax)
    format_metric_axis(ax)
    return fig, ax, {"metric_columns": [c for c, _, _ in metric_cols], "transformation": "risk inverted as 1-risk", "filter": "key methods" if key_only else "all methods"}


def plot_risk_gate_scatter(metrics: pd.DataFrame, noev: pd.DataFrame) -> tuple:
    data = metrics[["method", "hallucination_risk_proxy", "grounded_score"]].merge(
        noev[["method", "unsupported_generation_risk_proxy"]], on="method", how="left"
    )
    data["method_short"] = data["method"].map(canon_method)
    data = order_methods(data, "method_short")
    key = {"FD-GRAG-RC", "Evidence-flow", "GraphRAG", "Agentic RAG", "BM25", "Char TF-IDF"}
    low = data.loc[data["hallucination_risk_proxy"].idxmin(), "method_short"]
    high = data.loc[data["unsupported_generation_risk_proxy"].idxmax(), "method_short"]
    key |= {low, high}
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    x_max = max(0.14, float(data["hallucination_risk_proxy"].max()) + 0.035)
    ax.set_xlim(0, x_max)
    ax.set_ylim(0.08, 1.10)
    ax.add_patch(
        mpl.patches.Rectangle(
            (0, 0), 0.10, 0.10,
            facecolor=COLORS["soft_green"], edgecolor="none", alpha=0.35, zorder=0
        )
    )
    ax.axvline(0.10, color=COLORS["soft_red"], linestyle="--", linewidth=0.8, zorder=1)
    ax.axhline(0.10, color=COLORS["soft_red"], linestyle="--", linewidth=0.8, zorder=1)
    nonkey = data[~data["method_short"].isin(key)]
    ax.scatter(nonkey["hallucination_risk_proxy"], nonkey["unsupported_generation_risk_proxy"], s=44, color="#B7BBC0", edgecolor="#555555", alpha=0.55, linewidth=0.35)
    keydf = data[data["method_short"].isin(key)]
    ax.scatter(keydf["hallucination_risk_proxy"], keydf["unsupported_generation_risk_proxy"], s=78, color=[method_color(m) for m in keydf["method_short"]], edgecolor="#222222", linewidth=0.55, zorder=3)
    priority = {
        "FD-GRAG-RC": 0,
        "Evidence-flow": 1,
        "GraphRAG": 2,
        "Agentic RAG": 3,
        "BM25": 4,
        "Char TF-IDF": 5,
    }
    top_slots = [(0.22, 0.94), (0.54, 0.94), (0.78, 0.94), (0.22, 0.84), (0.54, 0.84), (0.78, 0.84)]
    low_slots = [(0.18, 0.34), (0.54, 0.34), (0.18, 0.24), (0.54, 0.24), (0.78, 0.30), (0.78, 0.20)]
    keydf = keydf.assign(_priority=keydf["method_short"].map(lambda m: priority.get(m, 10)))
    high_label_rows = keydf[keydf["unsupported_generation_risk_proxy"] >= 0.72].sort_values(["_priority", "hallucination_risk_proxy"])
    low_label_rows = keydf[keydf["unsupported_generation_risk_proxy"] < 0.72].sort_values(["_priority", "hallucination_risk_proxy"])
    label_slots: dict[str, tuple[float, float]] = {}
    for slot, (_, row) in zip(top_slots, high_label_rows.iterrows()):
        label_slots[row["method_short"]] = slot
    for slot, (_, row) in zip(low_slots, low_label_rows.iterrows()):
        label_slots[row["method_short"]] = slot
    fallback_slots = top_slots + low_slots
    for i, (_, row) in enumerate(keydf.sort_values(["_priority", "method_short"]).iterrows()):
        label_slots.setdefault(row["method_short"], fallback_slots[i % len(fallback_slots)])
    for _, row in keydf.sort_values(["_priority", "method_short"]).iterrows():
        tx, ty = label_slots[row["method_short"]]
        ax.annotate(
            row["method_short"],
            xy=(row["hallucination_risk_proxy"], row["unsupported_generation_risk_proxy"]),
            xytext=(tx, ty),
            xycoords="data",
            textcoords=ax.transAxes,
            fontsize=8.5,
            ha="left",
            va="center",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=method_color(row["method_short"]), lw=0.65),
            arrowprops=dict(arrowstyle="-", color="#6B7280", linewidth=0.55, linestyle=(0, (2, 2)), shrinkA=2, shrinkB=4),
        )
    ax.text(0.004, 0.112, "Safe answer\nregion", fontsize=8.1, color=COLORS["green"], ha="left", va="bottom", alpha=0.80)
    ax.text(0.98, 0.76, "High no-evidence risk", transform=ax.transAxes, fontsize=8.2, color=COLORS["red"], ha="right", va="top", alpha=0.68)
    ax.set_xlabel("Evidence-present risk proxy")
    ax.set_ylabel("No-evidence risk proxy")
    ax.text(-0.075, 1.02, "(a)", transform=ax.transAxes, ha="left", va="bottom", weight="bold", fontsize=12)
    ax.grid(color="#E1E5EA", linewidth=0.55, alpha=0.7)
    clean_spines(ax)
    return fig, ax, {"metric_columns": ["hallucination_risk_proxy", "unsupported_generation_risk_proxy"], "filter": sorted(key)}


def plot_flow_panel(df: pd.DataFrame, value_col: str, stem_type: str) -> tuple:
    data = df.copy()
    labels = [str(s).replace("\\n", "\n") for s in data["stage"]]
    values = data[value_col].to_numpy(float)
    n = len(data)
    fig, ax = plt.subplots(figsize=(10.5, 2.25))
    ax.set_xlim(-0.55, n - 0.45)
    ax.set_ylim(-0.35, 0.55)
    ax.axis("off")
    x = np.arange(n)
    for i, (xi, lab, val) in enumerate(zip(x, labels, values)):
        color = COLORS["fd"] if i == n - 1 else COLORS["teal"]
        box = mpl.patches.FancyBboxPatch(
            (xi - 0.34, -0.08), 0.68, 0.28,
            boxstyle="round,pad=0.025,rounding_size=0.035",
            fc="white", ec=color, lw=1.0
        )
        ax.add_patch(box)
        ax.text(xi, 0.055, fill(lab, 14), ha="center", va="center", fontsize=8.8, color=COLORS["dark_grey"])
        badge = f"{val:.2f}" if value_col == "score" else f"{int(val):,}"
        ax.text(xi, -0.18, badge, ha="center", va="center", fontsize=8.3, color=color, weight="bold")
        if i < n - 1:
            ax.annotate("", xy=(xi + 0.52, 0.055), xytext=(xi + 0.36, 0.055),
                        arrowprops=dict(arrowstyle="->", color="#667085", lw=0.8))
    note = "Methodology summary from existing artifacts; values are not simulated scores." if stem_type == "construction" else "Package recovery is summarized from existing evaluation artifacts."
    ax.text(0.5, -0.30, note, transform=ax.transAxes, ha="center", va="top", fontsize=8.2, color="#6B7280")
    panel_label(ax)
    return fig, ax, {"metric_columns": [value_col], "transformation": "none"}


def plot_compact_robustness(df: pd.DataFrame, x_col: str, title_note: str) -> tuple:
    data = df.copy()
    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    metrics = [c for c in ["accepted_pool", "exact_pool", "accepted_recall_10", "evidence_package_recall_10", "grounded_score"] if c in data.columns]
    if not metrics:
        metrics = [c for c in data.columns if c != x_col and pd.api.types.is_numeric_dtype(data[c])]
    for col, marker in zip(metrics[:3], ["o", "s", "D"]):
        grouped = data.groupby(x_col)[col].mean().reset_index()
        ax.plot(grouped[x_col], grouped[col], marker=marker, markersize=4, linewidth=1.0, label=col.replace("_", " "))
    ax.set_ylim(0.95, 1.005)
    ax.set_xlabel(x_col.replace("_", " ").title())
    ax.set_ylabel("Evidence score")
    ax.legend(loc="lower left", fontsize=8.4, frameon=False, ncol=min(3, len(metrics)))
    ax.text(0.99, 0.06, title_note, transform=ax.transAxes, ha="right", va="bottom", fontsize=8.2, color="#6B7280")
    panel_label(ax)
    ax.grid(color="#E1E5EA", linewidth=0.55)
    clean_spines(ax)
    return fig, ax, {"metric_columns": metrics[:3], "transformation": "mean by robustness axis; y shown on 0.95-1.005 range"}


def plot_oncoprint(df: pd.DataFrame, compact: bool = False) -> tuple:
    data = df.copy()
    criteria = [c for c in data.columns if c not in {"query_id", "query_family", "track"}]
    pass_rate = data[criteria].mean(axis=1)
    if compact:
        data = data.loc[pass_rate < 1.0].copy()
        if data.empty:
            data = df.assign(summary_row=True).tail(1).copy()
        criteria = [c for c in data.columns if c not in {"query_id", "query_family", "track", "summary_row"}]
    max_rows = 24
    if not compact and len(data) > max_rows:
        data = data.head(max_rows).copy()
    mat = data[criteria].astype(bool).to_numpy()
    height = max(2.15 if compact else 3.4, (0.48 if compact else 0.22) * len(data) + 1.4)
    fig, ax = plt.subplots(figsize=(8.2, height))
    ok_color = mpl.colors.to_rgb(COLORS["soft_green"])
    miss_color = mpl.colors.to_rgb(COLORS["soft_red"])
    img = np.zeros((mat.shape[0], mat.shape[1], 3))
    img[mat] = ok_color
    img[~mat] = miss_color
    ax.imshow(img, aspect="auto")
    ax.set_xticks(np.arange(len(criteria)))
    ax.set_xticklabels(wrap_labels(criteria, 14), rotation=32, ha="right")
    if compact:
        ylabels = [f"{r.query_id}  {str(r.query_family).replace('_',' ')}" for r in data.itertuples()]
        ylabels = wrap_labels(ylabels, 28)
    else:
        ylabels = [str(r.query_id) for r in data.itertuples()]
    ax.set_yticks(np.arange(len(data)))
    ax.set_yticklabels(ylabels)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if not mat[i, j]:
                ax.text(j, i, "Miss", ha="center", va="center", fontsize=7.1, color="#7A1F13", weight="bold")
    for j in range(mat.shape[1] + 1):
        ax.axvline(j - 0.5, color="white", linewidth=0.8)
    for i in range(mat.shape[0] + 1):
        ax.axhline(i - 0.5, color="white", linewidth=0.8)
    ax2 = ax.inset_axes([1.01, 0.0, 0.08, 1.0])
    pr = data[criteria].mean(axis=1)
    ax2.barh(np.arange(len(data)), pr, color=COLORS["green"], alpha=0.75)
    ax2.set_ylim(ax.get_ylim())
    ax2.set_xlim(0, 1)
    ax2.set_xticks([0, 1])
    ax2.set_yticks([])
    ax2.set_title("Pass", fontsize=8.5)
    clean_spines(ax2)
    panel_label(ax)
    clean_spines(ax)
    return fig, ax, {"metric_columns": criteria, "filter": "failures only" if compact else "full or first 24 rows"}


def plot_horizontal_tree_component(df: pd.DataFrame) -> tuple:
    data = df.copy()
    comp = data.groupby(["component_family_label"], as_index=False)["evidence_objects"].max().sort_values("evidence_objects", ascending=False).head(5)
    fig, ax = plt.subplots(figsize=(11.5, 5.6))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.02, 0.92, "PDF manuals → evidence objects → component-role profiles", fontsize=9, color="#8A8F98")
    root_box = mpl.patches.FancyBboxPatch((0.03, 0.43), 0.16, 0.12, boxstyle="round,pad=0.02", fc="white", ec=COLORS["fd"], lw=1.0)
    ax.add_patch(root_box)
    ax.text(0.11, 0.49, "Evidence\nrepository", ha="center", va="center", fontsize=10.2, weight="bold", color=COLORS["fd"])
    ys = np.linspace(0.78, 0.18, len(comp))
    for y, row in zip(ys, comp.itertuples()):
        ax.plot([0.19, 0.33], [0.49, y], color="#9AA3AF", lw=0.8)
        ax.add_patch(mpl.patches.FancyBboxPatch((0.33, y - 0.04), 0.22, 0.08, boxstyle="round,pad=0.015", fc="white", ec=COLORS["teal"], lw=0.9))
        ax.text(0.44, y, fill(row.component_family_label, 22), ha="center", va="center", fontsize=9.2)
        subset = data[data["component_family_label"] == row.component_family_label].sort_values("role_evidence_count", ascending=False)
        top = subset.head(4)
        other = subset.iloc[4:]["role_evidence_count"].sum()
        roles = list(top[["role", "role_evidence_count"]].itertuples(index=False, name=None))
        if other:
            roles.append(("Other", int(other)))
        xs = np.linspace(0.64, 0.94, len(roles))
        for x, (role, count) in zip(xs, roles):
            ax.plot([0.55, x - 0.04], [y, y], color="#CBD2D9", lw=0.7)
            ax.add_patch(mpl.patches.FancyBboxPatch((x - 0.045, y - 0.03), 0.09, 0.06, boxstyle="round,pad=0.012", fc="#F8FAFC", ec="#9AA3AF", lw=0.7))
            ax.text(x, y + 0.006, fill(str(role), 12), ha="center", va="center", fontsize=7.7)
            ax.text(x, y - 0.020, str(int(count)), ha="center", va="center", fontsize=7.2, color=COLORS["fd"])
    panel_label(ax)
    return fig, ax, {"metric_columns": ["evidence_objects", "role_evidence_count"], "filter": "top 5 components and top 4 roles plus Other"}


def plot_horizontal_tree_causal(df: pd.DataFrame) -> tuple:
    data = df.copy()
    rel = data.groupby("relation", as_index=False)["edge_count"].sum().sort_values("edge_count", ascending=False).head(5)
    fig, ax = plt.subplots(figsize=(11.5, 5.4))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.02, 0.92, "Facet points → diagnostic relations → component anchors", fontsize=9, color="#8A8F98")
    ax.add_patch(mpl.patches.FancyBboxPatch((0.03, 0.43), 0.16, 0.12, boxstyle="round,pad=0.02", fc="white", ec=COLORS["fd"], lw=1.0))
    ax.text(0.11, 0.49, "Diagnostic\ncausal graph", ha="center", va="center", fontsize=10.0, weight="bold", color=COLORS["fd"])
    ys = np.linspace(0.78, 0.18, len(rel))
    for y, row in zip(ys, rel.itertuples()):
        ax.plot([0.19, 0.34], [0.49, y], color="#9AA3AF", lw=0.8)
        ax.add_patch(mpl.patches.FancyBboxPatch((0.34, y - 0.04), 0.22, 0.08, boxstyle="round,pad=0.015", fc="white", ec=COLORS["teal"], lw=0.9))
        ax.text(0.45, y, fill(row.relation, 22), ha="center", va="center", fontsize=9.0)
        subset = data[data["relation"] == row.relation].sort_values("edge_count", ascending=False)
        top = subset.head(5)
        xs = np.linspace(0.64, 0.94, len(top))
        for x, r in zip(xs, top.itertuples()):
            ax.plot([0.56, x - 0.04], [y, y], color="#CBD2D9", lw=0.7)
            ax.add_patch(mpl.patches.FancyBboxPatch((x - 0.045, y - 0.03), 0.09, 0.06, boxstyle="round,pad=0.012", fc="#F8FAFC", ec="#9AA3AF", lw=0.7))
            ax.text(x, y + 0.006, fill(str(r.component_anchor), 11), ha="center", va="center", fontsize=7.6)
            ax.text(x, y - 0.020, str(int(r.edge_count)), ha="center", va="center", fontsize=7.2, color=COLORS["fd"])
    panel_label(ax)
    return fig, ax, {"metric_columns": ["edge_count", "mean_confidence"], "filter": "top 5 relations and top 5 component anchors"}


def plot_pareto(df: pd.DataFrame) -> tuple:
    data = df.copy()
    data["method_short"] = data.get("method_short", data["method"]).map(canon_method)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    key = {"FD-GRAG-RC", "Evidence-flow", "GraphRAG", "Agentic RAG"}
    fastest = data.loc[data["median_query_latency_ms"].idxmin(), "method_short"]
    low_latency = data[data["median_query_latency_ms"] <= data["median_query_latency_ms"].quantile(0.35)]
    best_low = low_latency.loc[low_latency["grounded_score"].idxmax(), "method_short"]
    key |= {fastest, best_low}
    non = data[~data["method_short"].isin(key)]
    ax.scatter(non["median_query_latency_ms"], non["grounded_score"], s=42, color="#B7BBC0", edgecolor="#555555", alpha=0.55, linewidth=0.35)
    kd = data[data["method_short"].isin(key)]
    ax.scatter(kd["median_query_latency_ms"], kd["grounded_score"], s=72, color=[method_color(m) for m in kd["method_short"]], edgecolor="#222222", linewidth=0.55, zorder=3)
    front = data.sort_values("median_query_latency_ms")
    best = []
    current = -np.inf
    for _, row in front.iterrows():
        if row["grounded_score"] >= current:
            best.append(row)
            current = row["grounded_score"]
    front_df = pd.DataFrame(best)
    ax.plot(front_df["median_query_latency_ms"], front_df["grounded_score"], color=COLORS["green"], alpha=0.65, linewidth=1.0, linestyle="--")
    ax.set_xscale("log")
    ax.set_xlabel("Median query latency (ms, log scale)")
    ax.set_ylabel("Evidence-grounded score")
    ax.set_ylim(max(0.80, float(data["grounded_score"].min()) - 0.02), 1.005)
    label_slots = {
        "FD-GRAG-RC": (0.80, 0.83),
        "Evidence-flow": (0.80, 0.90),
        "GraphRAG": (0.80, 0.76),
        "Agentic RAG": (0.80, 0.52),
    }
    left_slots = [(0.06, 0.94), (0.06, 0.86), (0.23, 0.94), (0.23, 0.86)]
    priority = {"FD-GRAG-RC": 0, "Evidence-flow": 1, "GraphRAG": 2, "Agentic RAG": 3}
    left_i = 0
    for _, row in kd.sort_values("method_short", key=lambda s: s.map(lambda m: priority.get(m, 10))).iterrows():
        slot = label_slots.get(row["method_short"])
        if slot is None:
            slot = left_slots[left_i % len(left_slots)]
            left_i += 1
        ax.annotate(row["method_short"], xy=(row["median_query_latency_ms"], row["grounded_score"]),
                    xytext=slot, xycoords="data", textcoords=ax.transAxes,
                    fontsize=8.5, bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=method_color(row["method_short"]), lw=0.65),
                    arrowprops=dict(arrowstyle="-", color="#6B7280", lw=0.55, linestyle=(0, (2, 2)), shrinkA=2, shrinkB=4))
    ax.grid(color="#E1E5EA", linewidth=0.55, alpha=0.8)
    panel_label(ax)
    clean_spines(ax)
    return fig, ax, {"metric_columns": ["median_query_latency_ms", "grounded_score"], "transformation": "log-scaled latency axis"}


def plot_fmeca_risk_matrix(df: pd.DataFrame) -> tuple:
    data = df.copy()
    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    cvar = data["detection_difficulty"].astype(float)
    use_cbar = cvar.nunique() > 1
    sizes = 34 + 8 * np.sqrt(data["evidence_count"].astype(float))
    if use_cbar:
        sc = ax.scatter(data["occurrence"], data["severity"], s=sizes, c=cvar, cmap="YlOrRd", edgecolor="#333333", linewidth=0.45, alpha=0.85)
        cbar = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.015)
        cbar.set_label("Detection difficulty")
    else:
        ax.scatter(data["occurrence"], data["severity"], s=sizes, color=COLORS["orange"], edgecolor="#333333", linewidth=0.45, alpha=0.82)
    ax.axvspan(7.5, 10.5, ymin=0.70, ymax=1.0, color=COLORS["soft_orange"], alpha=0.35, zorder=0)
    top = data.sort_values("rpn", ascending=False).head(5)
    for _, row in top.iterrows():
        ax.annotate(fill(row["component_label"], 18), xy=(row["occurrence"], row["severity"]),
                    xytext=(row["occurrence"] + 0.15, row["severity"] + 0.15), fontsize=8.3,
                    arrowprops=dict(arrowstyle="-", color="#6B7280", lw=0.55, linestyle=(0, (2, 2))),
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="#9AA3AF", lw=0.55))
    ax.set_xlabel("Occurrence proxy")
    ax.set_ylabel("Severity proxy")
    ax.set_xlim(0.5, 10.5)
    ax.set_ylim(0.5, 10.5)
    ax.grid(color="#E1E5EA", linewidth=0.55)
    ax.text(0.98, 0.96, "High-priority quadrant", transform=ax.transAxes, ha="right", va="top", fontsize=8.4, color="#8A5B00")
    panel_label(ax)
    clean_spines(ax)
    return fig, ax, {"metric_columns": ["occurrence", "severity", "evidence_count", "detection_difficulty"], "transformation": "bubble size = evidence count; colorbar only if detection difficulty varies"}


def plot_fmeca_role_control_heatmap(df: pd.DataFrame) -> tuple:
    data = df.copy().set_index("component_label")
    data = data.rename(columns=ROLE_LABELS)
    vals = data.to_numpy(float)
    log_vals = np.log1p(vals)
    cmap = mpl.colormaps["YlOrRd"].copy()
    cmap.set_under("#F7F7F7")
    fig, ax = plt.subplots(figsize=(8.4, max(4.0, 0.36 * data.shape[0] + 1.3)))
    im = ax.imshow(log_vals, aspect="auto", cmap=cmap, vmin=0.001, vmax=max(1, float(np.nanmax(log_vals))))
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_xticklabels(wrap_labels(data.columns, 12), rotation=35, ha="right")
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_yticklabels(wrap_labels(data.index, 24))
    threshold = float(np.nanmax(log_vals)) * 0.55
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = vals[i, j]
            if val > 0:
                ax.text(j, i, str(int(val)), ha="center", va="center", fontsize=7.4, color="white" if log_vals[i, j] > threshold else "#222222")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.015)
    cbar.set_label("log1p(evidence count)")
    panel_label(ax)
    clean_spines(ax)
    return fig, ax, {"metric_columns": list(data.columns), "transformation": "color = log1p(count); zero values shown as white/light gray"}


def plot_ablation_delta(df: pd.DataFrame) -> tuple:
    data = df.copy()
    data["label"] = data["ablation_view"].str.replace("Full FD-GRAG-RC-V21", "Full model", regex=False)
    data = data.sort_values("delta_grounded_vs_full")
    y = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(7.2, max(3.5, 0.34 * len(data) + 1.2)))
    colors = [COLORS["fd"] if v == 0 else COLORS["red"] for v in data["delta_grounded_vs_full"]]
    ax.barh(y, data["delta_grounded_vs_full"], color=colors, alpha=0.82, height=0.62)
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(wrap_labels(data["label"], 28))
    for yi, val in zip(y, data["delta_grounded_vs_full"]):
        ax.text(val - 0.002 if val < 0 else val + 0.002, yi, f"{val:+.3f}", ha="right" if val < 0 else "left", va="center", fontsize=8.2)
    ax.set_xlabel("Delta grounded score vs full model")
    panel_label(ax)
    ax.grid(axis="x", color="#E1E5EA", linewidth=0.55)
    clean_spines(ax)
    return fig, ax, {"metric_columns": ["delta_grounded_vs_full"], "transformation": "delta against full model"}


def plot_evidence_noevidence_dumbbell(source: pd.DataFrame) -> tuple:
    data = source.copy()
    data["method_short"] = data["method"].map(canon_method)
    data = order_methods(data, "method_short").sort_values("answerable_grounded")
    y = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(7.4, max(4.2, 0.32 * len(data) + 1.3)))
    key = {"FD-GRAG-RC", "GaRAGe-deflect", "Agentic RAG"}
    for yi, row in zip(y, data.itertuples()):
        col = method_color(row.method_short) if row.method_short in key else "#B7BBC0"
        ax.plot([row.no_evidence_deflection, row.answerable_grounded], [yi, yi], color=col, linewidth=0.8, alpha=0.75)
        ax.scatter(row.answerable_grounded, yi, marker="o", s=38, color=col, edgecolor="#333333", linewidth=0.45)
        ax.scatter(row.no_evidence_deflection, yi, marker="s", s=34, color=COLORS["orange"] if row.method_short in key else "#D0D3D8", edgecolor="#333333", linewidth=0.45)
    ax.set_yticks(y)
    ax.set_yticklabels(data["method_short"])
    ax.set_xlabel("Score")
    ax.legend(handles=[
        mpl.lines.Line2D([], [], marker="o", color="none", markerfacecolor=COLORS["fd"], markeredgecolor="#333333", label="Evidence-present grounded"),
        mpl.lines.Line2D([], [], marker="s", color="none", markerfacecolor=COLORS["orange"], markeredgecolor="#333333", label="No-evidence deflection"),
    ], loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False)
    ax.set_xlim(-0.02, 1.02)
    panel_label(ax)
    format_metric_axis(ax)
    return fig, ax, {"metric_columns": ["answerable_grounded", "no_evidence_deflection"], "transformation": "paired dumbbell; no-evidence deflection shown on common score scale"}


def build_case(case_id: str) -> tuple[list[dict], list[dict], list[dict]]:
    apply_journal_style()
    out = output_dir(case_id)
    manifests: list[dict] = []
    before_after: list[dict] = []
    qa: list[dict] = []
    tables = metric_tables(case_id)

    jobs = []
    jobs.append(("metric_heatmap", old_png(case_id, "nature_metric_matrix"), table_dir(case_id) / f"{case_id}_sota_metrics_by_method.csv", lambda: plot_metric_heatmap(tables["metrics"]), "Main text"))
    jobs.append(("query_family_heatmap", old_png(case_id, "nature_query_family_matrix"), table_dir(case_id) / f"{case_id}_sota_metrics_by_query_type.csv", lambda: plot_query_family_heatmap(tables["query"]), "Main text"))
    jobs.append(("risk_gate_scatter", old_png(case_id, "nature_risk_gate_scatter"), [table_dir(case_id) / f"{case_id}_sota_metrics_by_method.csv", table_dir(case_id) / f"{case_id}_no_evidence_probe_metrics.csv"], lambda: plot_risk_gate_scatter(tables["metrics"], tables["noev"]), "Main text"))
    jobs.append(("primary_metric_forest_full", old_png(case_id, "nature_primary_metric_forest"), table_dir(case_id) / f"{case_id}_sota_metrics_by_method.csv", lambda: plot_primary_metric_forest(tables["metrics"], False), "Supplementary"))
    jobs.append(("primary_metric_forest_key_methods", old_png(case_id, "nature_primary_metric_forest"), table_dir(case_id) / f"{case_id}_sota_metrics_by_method.csv", lambda: plot_primary_metric_forest(tables["metrics"], True), "Main text"))
    jobs.append(("ablation_delta_grounded", old_png(case_id, "nature_ablation_waterfall"), table_dir(case_id) / f"{case_id}_ablation_metrics.csv", lambda: plot_ablation_delta(tables["ablation"]), "Main text"))
    jobs.append(("evidence_no_evidence_dumbbell", old_png(case_id, "nature_evidence_no_evidence"), find_artifact(case_id, f"{case_id}_nature_evidence_no_evidence_source_data.csv"), lambda: plot_evidence_noevidence_dumbbell(pd.read_csv(find_artifact(case_id, f"{case_id}_nature_evidence_no_evidence_source_data.csv"))), "Main text"))
    jobs.append(("deployment_latency_quality_pareto", old_png(case_id, "deployment_latency_quality_pareto"), find_artifact(case_id, f"{case_id}_deployment_latency_quality_pareto_source_data.csv"), lambda: plot_pareto(pd.read_csv(find_artifact(case_id, f"{case_id}_deployment_latency_quality_pareto_source_data.csv"))), "Supplementary"))
    jobs.append(("evidence_flow_construction_clean", old_png(case_id, "panel_01_evidence_flow_construction"), find_artifact(case_id, f"{case_id}_panel_01_evidence_flow_construction_source_data.csv"), lambda: plot_flow_panel(pd.read_csv(find_artifact(case_id, f"{case_id}_panel_01_evidence_flow_construction_source_data.csv")), "count", "construction"), "Main text"))
    jobs.append(("package_recovery_flow_clean", old_png(case_id, "panel_18_package_recovery_flow"), find_artifact(case_id, f"{case_id}_panel_18_package_recovery_flow_source_data.csv"), lambda: plot_flow_panel(pd.read_csv(find_artifact(case_id, f"{case_id}_panel_18_package_recovery_flow_source_data.csv")), "score", "recovery"), "Main text"))
    jobs.append(("candidate_pool_ceiling_robustness", old_png(case_id, "panel_10_candidate_pool_ceiling"), find_artifact(case_id, f"{case_id}_panel_10_candidate_pool_ceiling_source_data.csv"), lambda: plot_compact_robustness(pd.read_csv(find_artifact(case_id, f"{case_id}_panel_10_candidate_pool_ceiling_source_data.csv")), "k", "Performance remains saturated under candidate budget."), "Supplementary"))
    jobs.append(("slot_complexity_stability_robustness", old_png(case_id, "panel_12_slot_complexity_stability"), find_artifact(case_id, f"{case_id}_panel_12_slot_complexity_stability_source_data.csv"), lambda: plot_compact_robustness(pd.read_csv(find_artifact(case_id, f"{case_id}_panel_12_slot_complexity_stability_source_data.csv")), "n_required_slots", "Performance remains saturated under slot complexity."), "Supplementary"))
    jobs.append(("reliability_oncoprint_full", old_png(case_id, "panel_17_reliability_oncoprint"), find_artifact(case_id, f"{case_id}_panel_17_reliability_oncoprint_source_data.csv"), lambda: plot_oncoprint(pd.read_csv(find_artifact(case_id, f"{case_id}_panel_17_reliability_oncoprint_source_data.csv")), False), "Supplementary"))
    jobs.append(("compact_oncoprint_failures_only", old_png(case_id, "panel_17_reliability_oncoprint"), find_artifact(case_id, f"{case_id}_panel_17_reliability_oncoprint_source_data.csv"), lambda: plot_oncoprint(pd.read_csv(find_artifact(case_id, f"{case_id}_panel_17_reliability_oncoprint_source_data.csv")), True), "Main text"))
    jobs.append(("component_role_tree_horizontal", old_png(case_id, "component_role_tree_highres"), find_artifact(case_id, f"{case_id}_component_role_tree_highres_plot_data.csv"), lambda: plot_horizontal_tree_component(pd.read_csv(find_artifact(case_id, f"{case_id}_component_role_tree_highres_plot_data.csv"))), "Main text"))
    jobs.append(("diagnostic_causal_tree_horizontal", old_png(case_id, "diagnostic_causal_tree_highres"), find_artifact(case_id, f"{case_id}_diagnostic_causal_tree_highres_plot_data.csv"), lambda: plot_horizontal_tree_causal(pd.read_csv(find_artifact(case_id, f"{case_id}_diagnostic_causal_tree_highres_plot_data.csv"))), "Main text"))
    jobs.append(("fmeca_risk_matrix", old_png(case_id, "fmeca_risk_matrix"), find_artifact(case_id, f"{case_id}_fmeca_risk_matrix_plot_data.csv"), lambda: plot_fmeca_risk_matrix(pd.read_csv(find_artifact(case_id, f"{case_id}_fmeca_risk_matrix_plot_data.csv"))), "Main text"))
    jobs.append(("fmeca_role_control_heatmap", old_png(case_id, "fmeca_role_control_heatmap"), find_artifact(case_id, f"{case_id}_fmeca_role_control_heatmap_plot_data.csv"), lambda: plot_fmeca_role_control_heatmap(pd.read_csv(find_artifact(case_id, f"{case_id}_fmeca_role_control_heatmap_plot_data.csv"))), "Supplementary"))

    for stem, old, src, maker, rec in jobs:
        fig, _ax, sidecar = maker()
        out_png = out / f"{case_id.lower()}_{stem}.png"
        save_png(fig, out_png, source_artifact=src, notes=f"Journal-ready redesign; recommended: {rec}.", sidecar=sidecar, manifest=manifests)
        qa.append(check_png_basic_quality(out_png))
        before_after.append(
            {
                "case_id": case_id,
                "figure": stem,
                "old_png": str(old),
                "new_png": str(out_png),
                "source_artifact": "; ".join(str(x) for x in src) if isinstance(src, list) else str(src),
                "recommendation": rec,
            }
        )

    pd.DataFrame(manifests).to_csv(out / "figure_manifest.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(qa).to_csv(out / "visual_qa.csv", index=False, encoding="utf-8-sig")
    return manifests, before_after, qa


def write_report(all_before_after: list[dict], all_qa: list[dict]) -> None:
    report = reports_dir() / "figure_redesign_report.md"
    lines = [
        "# Journal-ready Figure Redesign Report",
        "",
        "Scope: plotting-only redesign. Existing experiment outputs, benchmark files, metric CSV files, retrieval results, and core method code were not modified.",
        "",
        "## Summary",
        f"- New figures generated: {len(all_before_after)}",
        f"- Visual QA passed: {sum(1 for q in all_qa if q.get('ok'))}/{len(all_qa)}",
        "- Output folders: `figures/Boeing_737/` and `figures/Cessna_172/`",
        "",
        "## Per-figure redesign",
    ]
    problem_map = {
        "metric_heatmap": "Old heatmap used dense annotations and saturated color range.",
        "query_family_heatmap": "Old query-family matrix was visually crowded and over-annotated.",
        "risk_gate_scatter": "Old scatter used dense label lists or labels inside points.",
        "primary_metric_forest_full": "Old forest plot carried stronger styling and crowded legend placement.",
        "primary_metric_forest_key_methods": "Old figure did not provide a compact key-method view.",
        "ablation_delta_grounded": "Old waterfall style used heavier visual encoding than needed.",
        "evidence_no_evidence_dumbbell": "Old paired evidence/no-evidence comparison was less direct.",
        "deployment_latency_quality_pareto": "Old Pareto plot used large axis arrows and dense labels.",
        "evidence_flow_construction_clean": "Old flow panel had too much empty space and large node cards.",
        "package_recovery_flow_clean": "Old package recovery flow had oversized cards for near-saturated scores.",
        "candidate_pool_ceiling_robustness": "Old robustness panel overemphasized near-identical saturated curves.",
        "slot_complexity_stability_robustness": "Old slot-complexity panel overemphasized near-identical saturated curves.",
        "reliability_oncoprint_full": "Old oncoprint wrote OK in most cells.",
        "compact_oncoprint_failures_only": "Old oncoprint did not provide a failures-only compact view.",
        "component_role_tree_horizontal": "Old tree used a dashboard-like layout.",
        "diagnostic_causal_tree_horizontal": "Old causal tree used a dashboard-like layout.",
        "fmeca_risk_matrix": "Old risk matrix did not consistently separate occurrence, severity, size, and detection encoding.",
        "fmeca_role_control_heatmap": "Old role-control heatmap made zero cells visually too heavy.",
    }
    improvement_map = {
        "metric_heatmap": "Applied local color scale or 1-score fallback, selective annotations, compact colorbar, and FD-GRAG-RC row outline.",
        "query_family_heatmap": "Applied local color scale, selective annotations, wrapped task labels, and FD-GRAG-RC row outline.",
        "risk_gate_scatter": "Kept only key labels, used grey non-key points, light safety threshold lines, and neutral risk-proxy axis labels.",
        "primary_metric_forest_full": "Used clean spines, muted baselines, external legend, threshold line, and metric jitter.",
        "primary_metric_forest_key_methods": "Generated a compact key-method forest plot for main-text readability.",
        "ablation_delta_grounded": "Replaced waterfall with horizontal delta bars sorted by grounded-score loss.",
        "evidence_no_evidence_dumbbell": "Converted to paired dumbbell plot with highlighted key methods.",
        "deployment_latency_quality_pareto": "Kept log latency axis, removed large arrows, added Pareto front, and labeled only key methods.",
        "evidence_flow_construction_clean": "Compressed flow layout and moved notes below the panel.",
        "package_recovery_flow_clean": "Compressed score badges into a clean horizontal flow.",
        "candidate_pool_ceiling_robustness": "Converted saturated curves into compact robustness panel with y=[0.95,1.005].",
        "slot_complexity_stability_robustness": "Converted saturated curves into compact robustness panel with explicit saturation note.",
        "reliability_oncoprint_full": "Removed OK text, encoded pass cells by light green and only labeled misses.",
        "compact_oncoprint_failures_only": "Added compact failure-focused oncoprint and summary pass bars.",
        "component_role_tree_horizontal": "Redrew as a horizontal mechanism tree with top components and top roles plus Other.",
        "diagnostic_causal_tree_horizontal": "Redrew as a horizontal mechanism tree with top relations and component anchors.",
        "fmeca_risk_matrix": "Redrew as bubble chart: occurrence x severity, bubble size by evidence count, detection color only if variable.",
        "fmeca_role_control_heatmap": "Used light zero cells, sequential colormap, adaptive text color, and log1p count note.",
    }
    for item in all_before_after:
        fig = item["figure"]
        qa = next((q for q in all_qa if q["output_png"] == item["new_png"]), {})
        lines += [
            "",
            f"### {item['case_id']} / {fig}",
            f"- Original issue: {problem_map.get(fig, 'Old figure needed journal-style cleanup.')}",
            f"- Redesign: {improvement_map.get(fig, 'Applied shared journal style and cleaner export logic.')}",
            f"- New PNG: `{item['new_png']}`",
            f"- Source data: `{item['source_artifact']}`",
            f"- Recommendation: {item['recommendation']}",
            f"- QA: {'pass' if qa.get('ok') else 'review'}; size={qa.get('size_kb')} KB; dimensions={qa.get('width')}x{qa.get('height')}; blank_border_fraction={qa.get('blank_border_fraction')}",
        ]
    lines += [
        "",
        "## Non-modification statement",
        "This task changed plotting code, figure assembly, PNG export, manifests, sidecar JSON metadata, and reporting only. It did not modify experimental metric values, retrieval outputs, benchmark definitions, split assignments, model training, scoring logic, or core algorithm code.",
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    all_manifest: list[dict] = []
    all_before_after: list[dict] = []
    all_qa: list[dict] = []
    for case_id in CASE_IDS:
        manifest, before_after, qa = build_case(case_id)
        all_manifest.extend(manifest)
        all_before_after.extend(before_after)
        all_qa.extend(qa)
    pd.DataFrame(all_before_after).to_csv(reports_dir() / "figure_redesign_before_after.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(all_qa).to_csv(reports_dir() / "figure_redesign_visual_qa.csv", index=False, encoding="utf-8-sig")
    write_report(all_before_after, all_qa)
    print(json.dumps({"generated": len(all_manifest), "qa_pass": sum(1 for q in all_qa if q.get("ok")), "report": str(reports_dir() / "figure_redesign_report.md")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
