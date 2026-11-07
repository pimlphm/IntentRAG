from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CASE_IDS = ["Boeing_737", "Cessna_172"]

METHOD_ORDER = [
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

CANONICAL_METHOD_ORDER = [
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

METHOD_LABELS = {
    "BM25/word": "BM25",
    "BM25 lexical": "BM25",
    "Lexical sparse retrieval": "BM25",
    "TFIDF-char": "Char TF-IDF",
    "TF-IDF character": "Char TF-IDF",
    "Character n-gram lexical retrieval": "Char TF-IDF",
    "SVD semantic surrogate": "SVD surrogate",
    "Latent semantic dense surrogate": "SVD surrogate",
    "Hybrid lexical": "Hybrid",
    "Hybrid lexical-semantic": "Hybrid",
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
    "Agentic RAG": "Agentic RAG",
    "Query-text route selection": "Agentic RAG",
    "Evidence-flow baseline": "Evidence-flow",
    "Role-component-section evidence flow": "Evidence-flow",
    "GaRAGe-style relevance RAG": "GaRAGe-rel",
    "Grounding relevance representative": "GaRAGe-rel",
    "GaRAGe-style attribution RAG": "GaRAGe-attr",
    "Attribution/citation representative": "GaRAGe-attr",
    "GaRAGe-style deflective RAG": "GaRAGe-deflect",
    "Deflective grounding representative": "GaRAGe-deflect",
    "FD-GRAG-RC-V21-final": "FD-GRAG-RC",
    "FD-GRAG-RC-V21": "FD-GRAG-RC",
    "Proposed FD-GRAG-RC": "FD-GRAG-RC",
    "FD-GRAG-RC evidence package": "FD-GRAG-RC",
}

QWEN_LABELS = {
    "qwen_direct_no_rag": "Qwen direct",
    "qwen_bm25_context": "Qwen + BM25",
    "qwen_hybrid_context": "Qwen + Hybrid",
    "qwen_graphrag_context": "Qwen + GraphRAG",
    "qwen_evidence_flow_context": "Qwen + Evidence-flow",
    "qwen_fd_grag_rc_context": "Qwen + FD-GRAG-RC",
    "qwen_fd_gragrc_citation_repaired": "Qwen + FD-GRAG-RC (repaired)",
    "fd_gragrc_answer_assembled": "FD-GRAG-RC evidence-card assembly",
}

QWEN_ORDER = [
    "qwen_direct_no_rag",
    "qwen_bm25_context",
    "qwen_hybrid_context",
    "qwen_graphrag_context",
    "qwen_evidence_flow_context",
    "qwen_fd_grag_rc_context",
    "qwen_fd_gragrc_citation_repaired",
    "fd_gragrc_answer_assembled",
]

QWEN_TO_SOTA_METHOD = {
    "qwen_bm25_context": "BM25/word",
    "qwen_hybrid_context": "Hybrid lexical",
    "qwen_graphrag_context": "Controlled GraphRAG",
    "qwen_evidence_flow_context": "Evidence-flow baseline",
    "qwen_fd_grag_rc_context": "FD-GRAG-RC-V21-final",
    "qwen_fd_gragrc_citation_repaired": "FD-GRAG-RC-V21-final",
    "fd_gragrc_answer_assembled": "FD-GRAG-RC-V21-final",
}

COLORS = {
    "fd": "#1F5A99",
    "fd_repair": "#7C3AED",
    "fd_ref": "#2E8B57",
    "grey": "#8F8F8F",
    "light_grey": "#D5DAE3",
    "teal": "#2F8F9D",
    "orange": "#F59E0B",
    "green": "#2E8B57",
    "dark": "#222222",
}


def latest_case_dir(case_id: str) -> Path:
    return next(p for p in (ROOT / "Cases" / case_id).iterdir() if p.is_dir() and p.name.startswith("8"))


def to_float(series: pd.Series | str | float | int, default: float = 0.0) -> float:
    try:
        return float(series)
    except (TypeError, ValueError):
        return default


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def canonical_method_label(value: object) -> str:
    text = str(value).strip()
    if "FD-GRAG-RC" in text:
        return "FD-GRAG-RC"
    return METHOD_LABELS.get(text, text)


def add_axis_arrows(ax, xmin: float, xmax: float, ymin: float, ymax: float) -> None:
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(False)
    ax.annotate(
        "",
        xy=(xmax, ymin),
        xytext=(xmin, ymin),
        arrowprops=dict(arrowstyle="-|>", lw=1.25, color=COLORS["dark"]),
        clip_on=False,
    )
    ax.annotate(
        "",
        xy=(xmin, ymax),
        xytext=(xmin, ymin),
        arrowprops=dict(arrowstyle="-|>", lw=1.25, color=COLORS["dark"]),
        clip_on=False,
    )


def load_sota(case_id: str) -> pd.DataFrame:
    latest = latest_case_dir(case_id)
    preferred = latest / f"{case_id}_nature_primary_metric_forest_source_data.csv"
    fallback = latest / f"{case_id}_sota_metrics_by_method.csv"
    df = pd.read_csv(preferred if preferred.exists() else fallback).fillna("")
    if "method_short" not in df.columns:
        df["method_short"] = df["method"].astype(str)
    if "method_display" not in df.columns:
        df["method_display"] = df["method"].astype(str)
    order_map = {m: i for i, m in enumerate(METHOD_ORDER)}
    df["method_order"] = df["method"].map(order_map).fillna(999).astype(int)
    df["method_short"] = df["method"].map(canonical_method_label)
    df["method_display"] = df["method"].map(canonical_method_label)
    canonical_order_map = {m: i for i, m in enumerate(CANONICAL_METHOD_ORDER)}
    df["canonical_method_order"] = df["method_short"].map(canonical_order_map).fillna(999).astype(int)
    df = df.sort_values(["canonical_method_order", "method_order", "method"]).reset_index(drop=True)
    return df


def load_qwen(case_id: str) -> pd.DataFrame:
    latest = latest_case_dir(case_id)
    qwen = pd.read_csv(latest / f"{case_id}_qwen_generation_summary.csv").fillna("")
    repair = pd.read_csv(latest / f"{case_id}_qwen_fd_gragrc_before_after_citation_repair_summary.csv").fillna("")
    repaired = repair[repair["label"].astype(str).str.startswith("After")].copy()
    repaired["paradigm"] = "qwen_fd_gragrc_citation_repaired"
    repaired["source_method_display"] = QWEN_LABELS["qwen_fd_gragrc_citation_repaired"]

    assembled = repair[repair["label"].astype(str).str.startswith("Reference")].copy()
    assembled["paradigm"] = "fd_gragrc_answer_assembled"
    assembled["source_method_display"] = QWEN_LABELS["fd_gragrc_answer_assembled"]

    keep = [
        "qwen_direct_no_rag",
        "qwen_bm25_context",
        "qwen_hybrid_context",
        "qwen_graphrag_context",
        "qwen_evidence_flow_context",
        "qwen_fd_grag_rc_context",
    ]
    qwen = qwen[qwen["paradigm"].isin(keep)].copy()
    out = pd.concat([qwen, repaired, assembled], ignore_index=True)
    out["qwen_label"] = out["paradigm"].map(QWEN_LABELS).fillna(out["paradigm"])
    out["order"] = out["paradigm"].map({p: i for i, p in enumerate(QWEN_ORDER)}).fillna(999).astype(int)
    out["sota_method"] = out["paradigm"].map(QWEN_TO_SOTA_METHOD).fillna("")
    return out.sort_values(["order", "paradigm"]).reset_index(drop=True)


def build_joined(case_id: str) -> pd.DataFrame:
    sota = load_sota(case_id)
    qwen = load_qwen(case_id)
    qwen_small = qwen[
        [
            "paradigm",
            "qwen_label",
            "sota_method",
            "n",
            "mean_answer_evidence_grounded_score_proxy",
            "mean_citation_precision_proxy",
            "mean_citation_coverage_proxy",
            "mean_char_bigram_f1_to_gold_proxy",
        ]
    ].copy()
    rows = []
    for _, s in sota.iterrows():
        matches = qwen_small[qwen_small["sota_method"].astype(str).eq(str(s["method"]))]
        if matches.empty:
            row = {**s.to_dict()}
            row.update(
                {
                    "qwen_paradigm": "",
                    "qwen_label": "",
                    "qwen_answer_proxy": "",
                    "qwen_citation_precision": "",
                    "qwen_citation_coverage": "",
                    "qwen_char_f1_proxy": "",
                }
            )
            rows.append(row)
        else:
            for _, q in matches.iterrows():
                row = {**s.to_dict()}
                row.update(
                    {
                        "qwen_paradigm": q["paradigm"],
                        "qwen_label": q["qwen_label"],
                        "qwen_answer_proxy": q["mean_answer_evidence_grounded_score_proxy"],
                        "qwen_citation_precision": q["mean_citation_precision_proxy"],
                        "qwen_citation_coverage": q["mean_citation_coverage_proxy"],
                        "qwen_char_f1_proxy": q["mean_char_bigram_f1_to_gold_proxy"],
                    }
                )
                rows.append(row)
    direct = qwen_small[qwen_small["paradigm"].eq("qwen_direct_no_rag")]
    for _, q in direct.iterrows():
        rows.append(
            {
                "aircraft_model": case_id.replace("_", " "),
                "method": "Direct / no RAG",
                "method_short": "Direct",
                "method_display": "Direct / no RAG",
                "exact_recall_10": 0.0,
                "accepted_recall_10": 0.0,
                "evidence_package_recall_10": 0.0,
                "accepted_page_recall_10": 0.0,
                "accepted_section_recall_10": 0.0,
                "grounded_score": 0.0,
                "hallucination_risk_proxy": 0.0,
                "n_queries": q["n"],
                "method_order": -1,
                "qwen_paradigm": q["paradigm"],
                "qwen_label": q["qwen_label"],
                "qwen_answer_proxy": q["mean_answer_evidence_grounded_score_proxy"],
                "qwen_citation_precision": q["mean_citation_precision_proxy"],
                "qwen_citation_coverage": q["mean_citation_coverage_proxy"],
                "qwen_char_f1_proxy": q["mean_char_bigram_f1_to_gold_proxy"],
            }
        )
    return pd.DataFrame(rows)


def plot_full_comparison(case_id: str, sota: pd.DataFrame, qwen: pd.DataFrame, out_png: Path) -> None:
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(16.5, 7.2),
        dpi=180,
        gridspec_kw={"width_ratios": [1.28, 1.0], "wspace": 0.34},
    )

    labels = list(reversed(sota["method_short"].astype(str).tolist()))
    y = list(range(len(labels)))
    plot_sota = sota.iloc[::-1].reset_index(drop=True)
    metrics = [
        ("accepted_recall_10", "Accepted R@10", "#4C78A8", "o", -0.20),
        ("evidence_package_recall_10", "Package R@10", "#2A9D8F", "s", 0.0),
        ("grounded_score", "Grounded score", "#2E8B57", "D", 0.20),
    ]
    for yi, method in zip(y, plot_sota["method"].astype(str)):
        if "FD-GRAG-RC" in method:
            ax1.axhspan(yi - 0.45, yi + 0.45, color="#E8F1FF", zorder=0)
    for col, label, color, marker, off in metrics:
        vals = pd.to_numeric(plot_sota[col], errors="coerce").fillna(0.0)
        ax1.scatter(vals, [v + off for v in y], s=74, color=color, marker=marker, edgecolor="#1F2937", linewidth=0.65, label=label, zorder=3)
    fd_rows = plot_sota[plot_sota["method"].astype(str).str.contains("FD-GRAG-RC")]
    if not fd_rows.empty:
        idx = fd_rows.index[0]
        fd_val = float(fd_rows.iloc[0]["grounded_score"])
        ax1.plot([fd_val + 0.004, min(1.04, fd_val + 0.045)], [idx + 0.20, idx + 0.20], linestyle=(0, (2, 2)), color="#7B8794", linewidth=0.8)
        ax1.text(min(1.045, fd_val + 0.048), idx + 0.20, f"{fd_val:.3f}", ha="left", va="center", fontsize=10.5, fontweight="bold")
    xmin1 = max(0.65, min(pd.to_numeric(sota["exact_recall_10"], errors="coerce").min(), pd.to_numeric(sota["grounded_score"], errors="coerce").min()) - 0.04)
    xmax1 = 1.055
    ymin1, ymax1 = -0.7, len(labels) - 0.25
    ax1.set_xlim(xmin1, xmax1)
    ax1.set_ylim(ymin1, ymax1)
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels, fontsize=10.5)
    ax1.set_xlabel("SOTA evidence retrieval / grounding score", fontsize=13.5)
    ax1.grid(axis="x", color="#D9DEE7", linewidth=0.85, alpha=0.9)
    ax1.tick_params(axis="x", labelsize=10.5, length=0)
    ax1.tick_params(axis="y", length=0)
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, 1.10), ncol=3, frameon=False, fontsize=9.7)
    add_axis_arrows(ax1, xmin1, xmax1, ymin1, ymax1)

    q = qwen.copy().sort_values("order", ascending=False).reset_index(drop=True)
    y2 = list(range(len(q)))
    for yi, paradigm in zip(y2, q["paradigm"].astype(str)):
        if "citation_repaired" in paradigm:
            ax2.axhspan(yi - 0.42, yi + 0.42, color="#F2EAFE", zorder=0)
        elif paradigm == "fd_gragrc_answer_assembled":
            ax2.axhspan(yi - 0.42, yi + 0.42, color="#EAF7EF", zorder=0)
    vals = pd.to_numeric(q["mean_answer_evidence_grounded_score_proxy"], errors="coerce").fillna(0.0)
    colors = [
        COLORS["fd_repair"] if "citation_repaired" in p else COLORS["fd_ref"] if p == "fd_gragrc_answer_assembled" else COLORS["fd"] if "fd_grag_rc" in p else COLORS["grey"]
        for p in q["paradigm"].astype(str)
    ]
    ax2.scatter(vals, y2, s=120, color=colors, marker="o", edgecolor="#1F2937", linewidth=0.7, zorder=3)
    for yi, val, paradigm in zip(y2, vals, q["paradigm"].astype(str)):
        if "citation_repaired" in paradigm or paradigm == "fd_gragrc_answer_assembled":
            ax2.plot([val + 0.004, min(1.035, val + 0.045)], [yi, yi], linestyle=(0, (2, 2)), color="#7B8794", linewidth=0.8)
            ax2.text(min(1.04, val + 0.048), yi, f"{val:.3f}", ha="left", va="center", fontsize=10.7, fontweight="bold")
        elif val < 0.2:
            ax2.text(val + 0.018, yi, f"{val:.3f}", ha="left", va="center", fontsize=9.8)
    xmin2, xmax2 = 0.0, 1.06
    ymin2, ymax2 = -0.6, len(q) - 0.3
    ax2.set_xlim(xmin2, xmax2)
    ax2.set_ylim(ymin2, ymax2)
    ax2.set_yticks(y2)
    ax2.set_yticklabels(q["qwen_label"], fontsize=10.8)
    ax2.set_xlabel("Qwen answer evidence-grounded proxy", fontsize=13.5)
    ax2.grid(axis="x", color="#D9DEE7", linewidth=0.85, alpha=0.9)
    ax2.tick_params(axis="x", labelsize=10.5, length=0)
    ax2.tick_params(axis="y", length=0)
    add_axis_arrows(ax2, xmin2, xmax2, ymin2, ymax2)

    fig.savefig(out_png, dpi=600, bbox_inches="tight", pad_inches=0.05, facecolor="white")
    plt.close(fig)


def plot_retrieval_rank(case_id: str, sota: pd.DataFrame, out_png: Path) -> None:
    import matplotlib.pyplot as plt

    df = sota.copy().sort_values("grounded_score", ascending=True).reset_index(drop=True)
    y = list(range(len(df)))
    colors = [COLORS["fd"] if "FD-GRAG-RC" in str(m) else COLORS["grey"] for m in df["method"]]
    fig, ax = plt.subplots(figsize=(10.8, 7.2), dpi=180)
    ax.barh(y, pd.to_numeric(df["grounded_score"], errors="coerce"), color=colors, edgecolor="#1F2937", linewidth=0.6, height=0.64)
    for yi, val, method in zip(y, pd.to_numeric(df["grounded_score"], errors="coerce"), df["method"].astype(str)):
        weight = "bold" if "FD-GRAG-RC" in method else "normal"
        ax.text(min(1.045, float(val) + 0.006), yi, f"{float(val):.3f}", va="center", ha="left", fontsize=10.5, fontweight=weight)
    xmin = max(0.65, float(pd.to_numeric(df["grounded_score"], errors="coerce").min()) - 0.04)
    xmax = 1.06
    ymin, ymax = -0.7, len(df) - 0.2
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_yticks(y)
    ax.set_yticklabels(df["method_short"], fontsize=11.5)
    ax.set_xlabel("Grounded score", fontsize=14.5)
    ax.grid(axis="x", color="#D9DEE7", linewidth=0.85, alpha=0.9)
    ax.tick_params(axis="x", labelsize=11, length=0)
    ax.tick_params(axis="y", length=0)
    add_axis_arrows(ax, xmin, xmax, ymin, ymax)
    fig.savefig(out_png, dpi=600, bbox_inches="tight", pad_inches=0.05, facecolor="white")
    plt.close(fig)


def plot_answer_rank(case_id: str, qwen: pd.DataFrame, out_png: Path) -> None:
    import matplotlib.pyplot as plt

    df = qwen.copy().sort_values("mean_answer_evidence_grounded_score_proxy", ascending=True).reset_index(drop=True)
    y = list(range(len(df)))
    vals = pd.to_numeric(df["mean_answer_evidence_grounded_score_proxy"], errors="coerce").fillna(0.0)
    colors = [
        COLORS["fd_repair"] if "citation_repaired" in p else COLORS["fd_ref"] if p == "fd_gragrc_answer_assembled" else COLORS["fd"] if "fd_grag_rc" in p else COLORS["grey"]
        for p in df["paradigm"].astype(str)
    ]
    fig, ax = plt.subplots(figsize=(10.8, 5.3), dpi=180)
    ax.barh(y, vals, color=colors, edgecolor="#1F2937", linewidth=0.6, height=0.62)
    for yi, val, paradigm in zip(y, vals, df["paradigm"].astype(str)):
        weight = "bold" if "citation_repaired" in paradigm or paradigm == "fd_gragrc_answer_assembled" else "normal"
        ax.text(min(1.045, float(val) + 0.010), yi, f"{float(val):.3f}", va="center", ha="left", fontsize=11.0, fontweight=weight)
    xmin, xmax = 0.0, 1.06
    ymin, ymax = -0.65, len(df) - 0.25
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_yticks(y)
    ax.set_yticklabels(df["qwen_label"], fontsize=12.2)
    ax.set_xlabel("Answer evidence-grounded proxy", fontsize=14.5)
    ax.grid(axis="x", color="#D9DEE7", linewidth=0.85, alpha=0.9)
    ax.tick_params(axis="x", labelsize=11, length=0)
    ax.tick_params(axis="y", length=0)
    add_axis_arrows(ax, xmin, xmax, ymin, ymax)
    fig.savefig(out_png, dpi=600, bbox_inches="tight", pad_inches=0.05, facecolor="white")
    plt.close(fig)


def update_case(case_id: str) -> dict[str, str]:
    latest = latest_case_dir(case_id)
    sota = load_sota(case_id)
    qwen = load_qwen(case_id)
    joined = build_joined(case_id)
    save_csv(joined, latest / f"{case_id}_sota_full_comparison_with_qwen_fd_repair.csv")
    save_csv(qwen, latest / f"{case_id}_qwen_answer_proxy_all_contexts_with_fd_repair.csv")
    full_png = latest / f"{case_id}_sota_full_comparison_with_qwen_fd_repair.png"
    retrieval_png = latest / f"{case_id}_sota_retrieval_grounded_rank_with_fd.png"
    answer_png = latest / f"{case_id}_qwen_answer_proxy_rank_with_fd_repair.png"
    if full_png.exists():
        full_png.unlink()
    plot_retrieval_rank(case_id, sota, retrieval_png)
    plot_answer_rank(case_id, qwen, answer_png)
    return {
        "case_id": case_id,
        "latest_dir": str(latest),
        "retrieval_png": str(retrieval_png),
        "answer_png": str(answer_png),
        "joined_csv": str(latest / f"{case_id}_sota_full_comparison_with_qwen_fd_repair.csv"),
    }


def write_report(outputs: list[dict[str, str]]) -> None:
    lines = [
        "# Latest SOTA + Qwen FD Repair Figure Update",
        "",
        f"- Created time: {datetime.now().isoformat(timespec='seconds')}",
        "- Source retrieval metrics: case-local `*_sota_metrics_by_method.csv` or `*_nature_primary_metric_forest_source_data.csv`.",
        "- Source Qwen metrics: case-local Qwen generation summary and Qwen FD citation-repair summary.",
        "- The former combined comparison PNG is removed; the update now exports two independent no-title figures per case.",
        "- No benchmark rows, retrieval results, model weights, or original metric values were modified.",
        "- Boundary: Qwen repaired rows are post-processed/citation-repaired answers, not pure free-form Qwen generation.",
        "",
        "## Outputs",
        "",
    ]
    for out in outputs:
        lines += [
            f"### {out['case_id']}",
            f"- `{out['retrieval_png']}`",
            f"- `{out['answer_png']}`",
            f"- `{out['joined_csv']}`",
            "",
        ]
    (ROOT / "reports" / "sota_qwen_fd_repair_latest_figure_update.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    outputs = [update_case(case_id) for case_id in CASE_IDS]
    write_report(outputs)
    print(json.dumps({"status": "completed", "outputs": outputs}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
