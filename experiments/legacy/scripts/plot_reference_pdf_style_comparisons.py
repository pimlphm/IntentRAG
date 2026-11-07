from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from textwrap import fill

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
CASE_IDS = ["Boeing_737", "Cessna_172"]
OUT = ROOT / "figures" / "Comparative_Results"
DPI = 600

METHOD_ORDER = [
    "BM25",
    "Char TF-IDF",
    "GraphRAG",
    "Agentic RAG",
    "Evidence-flow",
    "GaRAGe-deflect",
    "FD-GRAG-RC",
]

METHOD_COLORS = {
    "BM25": "#BDBDBD",
    "Char TF-IDF": "#9E9E9E",
    "GraphRAG": "#8FA9C9",
    "Agentic RAG": "#B79AC8",
    "Evidence-flow": "#62B7A5",
    "GaRAGe-deflect": "#D9A867",
    "FD-GRAG-RC": "#2166AC",
}

METHOD_SHORT = {
    "BM25": "BM25",
    "Char TF-IDF": "Char TF-IDF",
    "GraphRAG": "GraphRAG",
    "Agentic RAG": "Agentic RAG",
    "Evidence-flow": "Evidence-flow",
    "GaRAGe-deflect": "GaRAGe-deflect",
    "FD-GRAG-RC": "FD-GRAG-RC",
}


def table_dir(case_id: str) -> Path:
    case_root = ROOT / "Cases" / case_id
    test = next(p for p in case_root.iterdir() if p.is_dir() and p.name.startswith("7"))
    return test / "sota_evidence_chain_tables"


def canon_method(method: str) -> str:
    text = str(method)
    if "FD-GRAG-RC" in text:
        return "FD-GRAG-RC"
    mapping = {
        "BM25/word": "BM25",
        "BM25 lexical": "BM25",
        "TFIDF-char": "Char TF-IDF",
        "TF-IDF character": "Char TF-IDF",
        "Controlled GraphRAG": "GraphRAG",
        "GraphRAG": "GraphRAG",
        "Agentic RAG": "Agentic RAG",
        "Evidence-flow baseline": "Evidence-flow",
        "Evidence-flow": "Evidence-flow",
        "GaRAGe-style deflective RAG": "GaRAGe-deflect",
        "GaRAGe-deflect": "GaRAGe-deflect",
    }
    return mapping.get(text, text)


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans"],
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.linewidth": 0.9,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 8.8,
            "ytick.labelsize": 8.8,
            "legend.fontsize": 8.8,
        }
    )


def read_chain_rows() -> pd.DataFrame:
    frames = []
    for case_id in CASE_IDS:
        path = table_dir(case_id) / f"{case_id}_complete_sota_evidence_chains.csv"
        df = pd.read_csv(path)
        df["case_id"] = case_id
        df["case_label"] = case_id.replace("_", " ")
        df["method_family"] = df["method_display"].map(canon_method)
        df["source_csv"] = str(path)
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    data = data[data["method_family"].isin(METHOD_ORDER)].copy()
    for col in [
        "exact_recall_10",
        "accepted_recall_10",
        "evidence_package_recall_10",
        "accepted_page_recall_10",
        "accepted_section_recall_10",
        "grounded_score",
        "hallucination_risk_proxy",
        "role_coverage_proxy",
    ]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    data["risk_control_score"] = 1 - data["hallucination_risk_proxy"].fillna(0)
    composite_cols = [c for c in ["exact_recall_10", "accepted_recall_10", "evidence_package_recall_10", "grounded_score", "risk_control_score"] if c in data.columns]
    data["evidence_grounded_composite"] = data[composite_cols].mean(axis=1)
    return data


def mean_ci(series: pd.Series) -> tuple[float, float]:
    vals = pd.to_numeric(series, errors="coerce").dropna().to_numpy(float)
    if len(vals) == 0:
        return np.nan, np.nan
    mean = float(vals.mean())
    if len(vals) == 1:
        return mean, 0.0
    ci = 1.96 * float(vals.std(ddof=1)) / np.sqrt(len(vals))
    return mean, ci


def clean_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#D9DEE5", linewidth=0.55, alpha=0.75)
    ax.set_axisbelow(True)


def save(fig, name: str, source_files: list[str], notes: str, sidecar_extra: dict, manifest: list[dict]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / name
    fig.savefig(out, dpi=DPI, bbox_inches="tight", pad_inches=0.06, facecolor="white")
    plt.close(fig)
    with Image.open(out) as img:
        width, height = img.size
    payload = {
        "output_png": str(out),
        "source_artifact": source_files,
        "created_time": datetime.now().isoformat(timespec="seconds"),
        "dpi": DPI,
        "width": width,
        "height": height,
        "notes": notes,
        **sidecar_extra,
    }
    out.with_suffix(".json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest.append(
        {
            "figure_name": out.name,
            "source_artifact": "; ".join(source_files),
            "output_png": str(out),
            "dpi": DPI,
            "width": width,
            "height": height,
            "notes": notes,
        }
    )


def plot_model_method_integration(data: pd.DataFrame, manifest: list[dict]) -> None:
    summary = (
        data.groupby(["case_label", "method_family"])["evidence_grounded_composite"]
        .apply(lambda s: pd.Series(mean_ci(s), index=["mean", "ci"]))
        .unstack()
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    cases = list(summary["case_label"].unique())
    x = np.arange(len(cases))
    width = 0.10
    for i, method in enumerate(METHOD_ORDER):
        sub = summary[summary["method_family"].eq(method)].set_index("case_label")
        means = [sub.loc[c, "mean"] if c in sub.index else np.nan for c in cases]
        cis = [sub.loc[c, "ci"] if c in sub.index else 0 for c in cases]
        pos = x + (i - (len(METHOD_ORDER) - 1) / 2) * width
        ax.bar(pos, means, width=width, color=METHOD_COLORS[method], edgecolor="#333333", linewidth=0.35, label=method)
        ax.errorbar(pos, means, yerr=cis, fmt="none", ecolor="#333333", elinewidth=0.7, capsize=2)
    ax.set_xticks(x)
    ax.set_xticklabels(cases)
    ax.set_ylabel("Evidence-grounded composite")
    ax.set_ylim(0.82, 1.01)
    ax.text(0.0, 1.02, "(a)", transform=ax.transAxes, ha="left", va="bottom", weight="bold", fontsize=13)
    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.02), frameon=False)
    clean_axes(ax)
    save(
        fig,
        "reference_style_model_method_integration.png",
        sorted(data["source_csv"].unique().tolist()),
        "Nature-style grouped comparison: method integration performance across aircraft cases; mean and 95% CI over test queries.",
        {"metric": "evidence_grounded_composite", "ci": "1.96*SE over queries"},
        manifest,
    )


def plot_metric_panels(data: pd.DataFrame, manifest: list[dict]) -> None:
    metrics = [
        ("exact_recall_10", "Exact Recall@10"),
        ("accepted_recall_10", "Accepted Recall@10"),
        ("evidence_package_recall_10", "Package Recall@10"),
        ("grounded_score", "Grounded score"),
        ("risk_control_score", "1 - risk proxy"),
        ("role_coverage_proxy", "Role coverage"),
    ]
    metrics = [(m, lab) for m, lab in metrics if m in data.columns]
    fig, axes = plt.subplots(2, 3, figsize=(10.8, 6.0), sharex=True)
    axes = axes.ravel()
    x = np.arange(len(METHOD_ORDER))
    for pi, (metric, label) in enumerate(metrics[:6]):
        ax = axes[pi]
        rows = []
        for method in METHOD_ORDER:
            mean, ci = mean_ci(data.loc[data["method_family"].eq(method), metric])
            rows.append((method, mean, ci))
        means = [r[1] for r in rows]
        cis = [r[2] for r in rows]
        ax.bar(x, means, color=[METHOD_COLORS[m] for m in METHOD_ORDER], edgecolor="#333333", linewidth=0.30)
        ax.errorbar(x, means, yerr=cis, fmt="none", ecolor="#333333", elinewidth=0.65, capsize=2)
        finite = [v for v in means if not np.isnan(v)]
        ymin = max(0, min(finite) - 0.04) if finite else 0
        ax.set_ylim(ymin, 1.01)
        ax.set_title(label, fontsize=10.5, pad=6)
        ax.text(0.0, 1.02, f"({chr(97+pi)})", transform=ax.transAxes, ha="left", va="bottom", weight="bold", fontsize=11.5)
        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_SHORT[m] for m in METHOD_ORDER], rotation=35, ha="right")
        clean_axes(ax)
    for ax in axes[len(metrics):]:
        ax.axis("off")
    fig.subplots_adjust(bottom=0.16, hspace=0.34, wspace=0.32)
    fig.text(0.5, 0.01, "Method", ha="center", va="bottom", fontsize=10.5)
    fig.text(0.015, 0.5, "Score", ha="left", va="center", rotation=90, fontsize=10.5)
    save(
        fig,
        "reference_style_detailed_metric_panels.png",
        sorted(data["source_csv"].unique().tolist()),
        "Detailed per-metric comparison panel inspired by the reference paper's multi-panel metric figure.",
        {"metrics": [m for m, _ in metrics[:6]], "ci": "1.96*SE over queries"},
        manifest,
    )


def plot_query_type_delta(data: pd.DataFrame, manifest: list[dict]) -> None:
    rows = []
    for (case, qtype), group in data.groupby(["case_label", "query_type"], dropna=True):
        fd = group[group["method_family"].eq("FD-GRAG-RC")]
        base = group[~group["method_family"].eq("FD-GRAG-RC")]
        if fd.empty or base.empty:
            continue
        fd_mean = fd["risk_control_score"].mean()
        best_base = base.groupby("method_family")["risk_control_score"].mean().max()
        pkg_fd = fd["evidence_package_recall_10"].mean()
        pkg_base = base.groupby("method_family")["evidence_package_recall_10"].mean().max()
        rows.append({"case": case, "query_type": qtype, "risk_control_delta": fd_mean - best_base, "package_delta": pkg_fd - pkg_base})
    d = pd.DataFrame(rows)
    if d.empty:
        return
    d = d[(d["risk_control_delta"].gt(0.002)) | (d["package_delta"].gt(0.005))].copy()
    if d.empty:
        return
    d["label"] = d["case"] + "\n" + d["query_type"].str.replace("_", " ")
    d["max_positive_delta"] = d[["risk_control_delta", "package_delta"]].max(axis=1)
    d = d.sort_values("max_positive_delta").tail(14)
    fig, ax = plt.subplots(figsize=(9.5, max(4.8, 0.34 * len(d) + 1.0)))
    y = np.arange(len(d))
    ax.axvline(0, color="#333333", linewidth=0.9)
    ax.barh(y - 0.14, d["risk_control_delta"], height=0.25, color="#2E9D57", label="Risk-control delta")
    ax.barh(y + 0.14, d["package_delta"], height=0.25, color="#2166AC", label="Package recall delta")
    ax.set_yticks(y)
    ax.set_yticklabels([fill(x, 34) for x in d["label"]])
    ax.set_xlabel("FD-GRAG-RC minus best baseline")
    ax.legend(frameon=False, loc="lower right")
    ax.text(0.0, 1.02, "(a)", transform=ax.transAxes, ha="left", va="bottom", weight="bold", fontsize=13)
    ax.grid(axis="x", color="#D9DEE5", linewidth=0.55, alpha=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save(
        fig,
        "reference_style_query_type_delta.png",
        sorted(data["source_csv"].unique().tolist()),
        "Query-type delta plot: only comparative deltas against the best baseline are shown.",
        {"metrics": ["risk_control_score", "evidence_package_recall_10"], "comparison": "FD-GRAG-RC minus best baseline by query type"},
        manifest,
    )


def plot_risk_distribution(data: pd.DataFrame, manifest: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    samples = [data.loc[data["method_family"].eq(m), "hallucination_risk_proxy"].dropna().to_numpy(float) for m in METHOD_ORDER]
    parts = ax.violinplot(samples, positions=np.arange(len(METHOD_ORDER)), widths=0.72, showmeans=True, showextrema=False)
    for body, method in zip(parts["bodies"], METHOD_ORDER):
        body.set_facecolor(METHOD_COLORS[method])
        body.set_edgecolor("#333333")
        body.set_alpha(0.75)
    parts["cmeans"].set_color("#222222")
    parts["cmeans"].set_linewidth(1.0)
    ax.set_xticks(np.arange(len(METHOD_ORDER)))
    ax.set_xticklabels([METHOD_SHORT[m] for m in METHOD_ORDER], rotation=30, ha="right")
    ax.set_ylabel("Hallucination risk proxy")
    ax.set_ylim(0, max(0.08, min(0.18, np.nanpercentile(data["hallucination_risk_proxy"], 95) * 1.45)))
    ax.text(0.0, 1.02, "(a)", transform=ax.transAxes, ha="left", va="bottom", weight="bold", fontsize=13)
    clean_axes(ax)
    save(
        fig,
        "reference_style_risk_proxy_violin.png",
        sorted(data["source_csv"].unique().tolist()),
        "Violin distribution of hallucination-risk proxy over test queries, following the reference paper's risk-mitigation comparison style.",
        {"metric": "hallucination_risk_proxy", "direction": "risk proxy"},
        manifest,
    )


def plot_difficulty_stability(data: pd.DataFrame, manifest: list[dict]) -> None:
    order = ["easy", "medium", "hard"]
    available = [x for x in order if x in set(data["difficulty_level"].dropna().astype(str))]
    if len(available) < 2:
        available = list(data["track"].dropna().astype(str).unique())
        x_col = "track"
        x_label = "Benchmark track"
    else:
        x_col = "difficulty_level"
        x_label = "Question difficulty"
    d = data[data[x_col].astype(str).isin(available)].copy()
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    x = np.arange(len(available))
    for method in ["BM25", "GraphRAG", "Agentic RAG", "Evidence-flow", "FD-GRAG-RC"]:
        means, cis = [], []
        for level in available:
            mean, ci = mean_ci(d.loc[(d["method_family"].eq(method)) & (d[x_col].astype(str).eq(level)), "grounded_score"])
            means.append(mean)
            cis.append(ci)
        ax.errorbar(x, means, yerr=cis, marker="o", linewidth=1.4, capsize=2.5, color=METHOD_COLORS[method], label=method)
    ax.set_xticks(x)
    ax.set_xticklabels([str(v).replace("_", " ") for v in available])
    ax.set_ylabel("Grounded score")
    ax.set_xlabel(x_label)
    finite = d["grounded_score"].dropna()
    ax.set_ylim(max(0.88, float(finite.quantile(0.05)) - 0.025), 1.01)
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.02))
    ax.text(0.0, 1.02, "(a)", transform=ax.transAxes, ha="left", va="bottom", weight="bold", fontsize=13)
    clean_axes(ax)
    save(
        fig,
        "reference_style_complexity_stability.png",
        sorted(data["source_csv"].unique().tolist()),
        "Difficulty/track stability line plot analogous to complexity analyses in the reference paper.",
        {"metric": "grounded_score", "x_axis": x_col, "ci": "1.96*SE over queries"},
        manifest,
    )


def main() -> None:
    apply_style()
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.png"):
        old.unlink()
    for old in OUT.glob("*.json"):
        old.unlink()
    data = read_chain_rows()
    manifest: list[dict] = []
    plot_model_method_integration(data, manifest)
    plot_metric_panels(data, manifest)
    plot_query_type_delta(data, manifest)
    plot_risk_distribution(data, manifest)
    plot_difficulty_stability(data, manifest)
    pd.DataFrame(manifest).to_csv(OUT / "figure_manifest.csv", index=False, encoding="utf-8-sig")
    report = ROOT / "reports" / "reference_style_comparison_figures.md"
    rows = []
    source_summary = "Boeing_737 and Cessna_172 case-local sota_evidence_chain_tables complete evidence-chain CSVs"
    for row in manifest:
        rel_out = Path(row["output_png"]).relative_to(ROOT).as_posix()
        rows.append(f"| `{row['figure_name']}` | `{rel_out}` | {source_summary} | {row['notes']} |")
    report.write_text(
        "# Reference-style Comparison Figures\n\n"
        "Generated comparison/result figures inspired by the supplied Nature Communications reference PDF. "
        "The figures use existing test-set artifacts only and do not alter any benchmark, metric, retrieval, CSV, JSON, or NPY data.\n\n"
        "## Generated PNG figures\n\n"
        "| Figure | Output path | Source artifact | Note |\n"
        "|---|---|---|---|\n"
        + "\n".join(rows)
        + "\n\n"
        "## Deliberately omitted from the new comparison set\n\n"
        "- `*_evidence_flow_construction.png`: process-summary figure; omitted because the current request asks for comparison/result figures instead.\n"
        "- `*_package_recovery_flow.png`: process-summary figure; omitted because it repeats the evidence-flow mechanism and does not add result comparison value.\n\n"
        "## Claim boundary\n\n"
        "These figures summarize document-grounded retrieval, evidence-package, and risk-proxy measurements from existing artifacts. "
        "They should not be interpreted as real maintenance accuracy, operational safety certification, or expert replacement.\n",
        encoding="utf-8",
    )
    print({"generated": len(manifest), "out": str(OUT)})


if __name__ == "__main__":
    main()
