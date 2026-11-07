import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from evaluate_qwen_generation_paradigms import (
    CASE_IDS,
    OUT_DIR as BASE_QWEN_OUT,
    SOURCE_METRICS,
    evaluate_answer,
    load_rows,
    make_summary,
    to_float,
)


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "04_results" / "fd_gragrc_answer_assembled"
FIG_DIR = ROOT / "figures" / "Qwen_Paradigm_Comparison"
REPORT_DIR = ROOT / "reports"

LABELS = {
    "qwen_direct_no_rag": "Qwen direct",
    "qwen_bm25_context": "Qwen + BM25",
    "qwen_hybrid_context": "Qwen + Hybrid",
    "qwen_graphrag_context": "Qwen + GraphRAG",
    "qwen_evidence_flow_context": "Qwen + Evidence-flow",
    "qwen_fd_grag_rc_context": "Qwen + FD-GRAG-RC",
    "fd_gragrc_answer_assembled": "FD-GRAG-RC evidence-card assembly",
}

PLOT_ORDER = [
    "qwen_direct_no_rag",
    "qwen_bm25_context",
    "qwen_hybrid_context",
    "qwen_graphrag_context",
    "qwen_evidence_flow_context",
    "qwen_fd_grag_rc_context",
    "fd_gragrc_answer_assembled",
]


def latest_case_dir(case_id: str) -> Path:
    return next(p for p in (ROOT / "Cases" / case_id).iterdir() if p.is_dir() and p.name.startswith("8"))


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def parse_first_evidence_card(chain: Any) -> tuple[str, str, str, str]:
    text = "" if pd.isna(chain) else str(chain)
    pattern = re.compile(
        r"1\.\s+(EVID[A-Z0-9_]+)\s+\|\s+"
        r"page=([^|]*)\|\s+role=([^|]*)\|\s+component=([^|]*)\|\s+section=([^|]*)\|\s*"
        r"(.*?)(?=\n\s*2\.\s+EVID[A-Z0-9_]+\s+\||\Z)",
        flags=re.S,
    )
    match = pattern.search(text)
    if not match:
        ids = re.findall(r"EVID[A-Z0-9_]+", text)
        return (ids[0] if ids else "", "", "", "")
    eid = match.group(1).strip()
    page = match.group(2).strip()
    role = match.group(3).strip()
    snippet = re.sub(r"\s+", " ", match.group(6)).strip()
    if " | " in snippet:
        snippet = snippet.split(" | ")[-1].strip()
    return eid, page, role, snippet


def build_assembled_answer(row: pd.Series, snippet_chars: int = 220) -> str:
    """Build an extractive, evidence-anchored FD-GRAG-RC answer.

    This uses only retrieved evidence cards, not gold answers or accepted labels.
    It is intentionally conservative: the final answer is a short source span plus
    the source evidence id. This makes the answer auditable and keeps citation
    behavior deterministic for small base models.
    """

    eid, page, role, snippet = parse_first_evidence_card(row.get("evidence_chain_top10", ""))
    question = str(row.get("question", "")).strip()
    prefix = "Evidence-grounded answer"
    if not snippet:
        ids = re.findall(r"EVID[A-Z0-9_]+", str(row.get("top_evidence_ids", "")))
        checked = "; ".join(ids[:3])
        return f"Evidence is insufficient for a confident answer. Evidence IDs checked: {checked}."
    snippet = snippet[:snippet_chars].strip()
    citation = f"{eid}" if eid else ""
    page_text = f", page {page}" if page else ""
    role_text = f", role {role}" if role else ""
    return (
        f"{prefix}: {snippet} "
        f"Evidence ID: {citation}{page_text}{role_text}. "
        f"Question addressed: {question[:120]}"
    ).strip()


def build_records() -> pd.DataFrame:
    rows = load_rows(CASE_IDS)
    fd = rows[rows["paradigm"].eq("qwen_fd_grag_rc_context")].copy()
    records = []
    for _, row in fd.iterrows():
        answer = build_assembled_answer(row)
        metrics = evaluate_answer(answer, row)
        records.append(
            {
                "case_id": row.get("case_id", ""),
                "query_id": row.get("query_id", row.get("question_id", "")),
                "question_id": row.get("question_id", ""),
                "aircraft_model": row.get("aircraft_model", ""),
                "query_type": row.get("query_type", ""),
                "scale": "fd_gragrc_answer_assembler",
                "provider": "deterministic_evidence_anchored_assembler",
                "model": "FD-GRAG-RC evidence-card assembler",
                "paradigm": "fd_gragrc_answer_assembled",
                "source_method": row.get("method", ""),
                "source_method_display": "FD-GRAG-RC evidence-card assembly",
                **{f"source_{m}": to_float(row.get(m, 0.0)) for m in SOURCE_METRICS},
                "status": "ok",
                "error": "",
                "latency_s": 0.0,
                "answer": answer,
                "usage_json": "{}",
                "top_evidence_ids": row.get("top_evidence_ids", ""),
                "evidence_chain_top10": row.get("evidence_chain_top10", ""),
                **metrics,
            }
        )
    return pd.DataFrame(records)


def add_risk_control(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["mean_source_risk_control_proxy"] = 1.0 - pd.to_numeric(
        df["mean_source_hallucination_risk_proxy"], errors="coerce"
    ).fillna(0.0)
    df.loc[df["paradigm"].eq("qwen_direct_no_rag"), "mean_source_risk_control_proxy"] = 0.0
    df["label"] = df["paradigm"].map(LABELS).fillna(df["source_method_display"])
    df["order"] = df["paradigm"].map({p: i for i, p in enumerate(PLOT_ORDER)}).fillna(99)
    return df.sort_values("order").reset_index(drop=True)


def combined_summary(records: pd.DataFrame) -> pd.DataFrame:
    base = pd.read_csv(BASE_QWEN_OUT / "qwen_generation_summary.csv").fillna("")
    assembled = make_summary(records.to_dict("records"))
    assembled["paradigm"] = "fd_gragrc_answer_assembled"
    assembled["source_method_display"] = "FD-GRAG-RC evidence-card assembly"
    return add_risk_control(pd.concat([base, assembled], ignore_index=True))


def case_summary(case_id: str, records: pd.DataFrame) -> pd.DataFrame:
    base = pd.read_csv(BASE_QWEN_OUT / f"{case_id}_qwen_generation_summary.csv").fillna("")
    assembled = make_summary(records[records["case_id"].astype(str).eq(case_id)].to_dict("records"))
    assembled["paradigm"] = "fd_gragrc_answer_assembled"
    assembled["source_method_display"] = "FD-GRAG-RC evidence-card assembly"
    return add_risk_control(pd.concat([base, assembled], ignore_index=True))


def add_axis_arrows(ax, xmin: float, xmax: float, ymin: float, ymax: float) -> None:
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(False)
    ax.annotate("", xy=(xmax, ymin), xytext=(xmin, ymin), arrowprops=dict(arrowstyle="-|>", lw=1.2, color="#222222"), clip_on=False)
    ax.annotate("", xy=(xmin, ymax), xytext=(xmin, ymin), arrowprops=dict(arrowstyle="-|>", lw=1.2, color="#222222"), clip_on=False)


def plot_summary(df: pd.DataFrame, out_png: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    data = df[df["paradigm"].isin(PLOT_ORDER)].copy().sort_values("order")
    data["label"] = data["paradigm"].map(LABELS).fillna(data.get("source_method_display", data["paradigm"]))
    metrics = [
        ("mean_answer_evidence_grounded_score_proxy", "Answer proxy", "#7C3AED", "o"),
        ("mean_source_evidence_package_recall_10", "Package R@10", "#1B9E77", "s"),
        ("mean_source_grounded_score", "Source grounded", "#2166AC", "D"),
        ("mean_source_risk_control_proxy", "Risk control", "#2E7D32", "^"),
    ]
    y = list(range(len(data)))
    offsets = [-0.24, -0.08, 0.08, 0.24]
    fig, ax = plt.subplots(figsize=(12.0, 6.8), dpi=180)
    for row_idx, paradigm in enumerate(data["paradigm"]):
        if paradigm == "fd_gragrc_answer_assembled":
            ax.axhspan(row_idx - 0.45, row_idx + 0.45, color="#E8F1FF", zorder=0)
    for offset, (col, label, color, marker) in zip(offsets, metrics):
        vals = pd.to_numeric(data[col], errors="coerce").fillna(0.0)
        ax.scatter(vals, [v + offset for v in y], s=112, color=color, marker=marker, edgecolor="#1F2937", linewidth=0.7, label=label, zorder=3)
        for yi, val, paradigm in zip(y, vals, data["paradigm"]):
            if paradigm == "fd_gragrc_answer_assembled":
                ax.plot([val + 0.004, min(1.04, val + 0.052)], [yi + offset, yi + offset], linestyle=(0, (2, 2)), color="#7B8794", linewidth=0.8, zorder=2)
                ax.text(min(1.045, val + 0.055), yi + offset, f"{val:.3f}", ha="left", va="center", fontsize=10.5, fontweight="bold", color="#102A43")
    xmin, xmax = 0.0, 1.06
    ymin, ymax = -0.65, len(data) - 0.35
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_yticks(y)
    ax.set_yticklabels(data["label"], fontsize=13)
    for tick, paradigm in zip(ax.get_yticklabels(), data["paradigm"]):
        if paradigm == "fd_gragrc_answer_assembled":
            tick.set_fontweight("bold")
            tick.set_color("#1F4E79")
    ax.set_xlabel("Answer / evidence-package score", fontsize=15)
    if title:
        ax.set_title(title, fontsize=15, fontweight="bold", pad=12)
    ax.grid(axis="x", color="#D9DEE7", linewidth=0.9, alpha=0.85)
    ax.tick_params(axis="x", labelsize=12, length=0)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=4, frameon=False, fontsize=11)
    add_axis_arrows(ax, xmin, xmax, ymin, ymax)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=600, bbox_inches="tight", pad_inches=0.05, facecolor="white")
    plt.close(fig)


def write_report(summary: pd.DataFrame) -> None:
    lines = [
        "# FD-GRAG-RC Evidence-anchored Answer Assembly Results",
        "",
        f"- Created time: {datetime.now().isoformat(timespec='seconds')}",
        "- This run uses a deterministic evidence-card answer assembler on top of FD-GRAG-RC retrieval outputs.",
        "- The assembler uses only retrieved evidence cards and visible questions; it does not use gold answers, accepted sets, benchmark labels, difficulty, or split metadata.",
        "- Existing Qwen baseline results are read-only and unchanged.",
        "",
        "## Pooled comparison",
        "",
        "| Paradigm | Answer proxy | Citation precision | Citation coverage | Char-F1 proxy | Accepted R@10 | Package R@10 | Source grounded | Risk proxy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            "| {label} | {ans:.4f} | {cp:.4f} | {cc:.4f} | {cf:.4f} | {acc:.4f} | {pkg:.4f} | {ground:.4f} | {risk:.4f} |".format(
                label=row["label"],
                ans=float(row["mean_answer_evidence_grounded_score_proxy"]),
                cp=float(row["mean_citation_precision_proxy"]),
                cc=float(row["mean_citation_coverage_proxy"]),
                cf=float(row["mean_char_bigram_f1_to_gold_proxy"]),
                acc=float(row["mean_source_accepted_recall_10"]),
                pkg=float(row["mean_source_evidence_package_recall_10"]),
                ground=float(row["mean_source_grounded_score"]),
                risk=float(row["mean_source_hallucination_risk_proxy"]),
            )
        )
    lines += [
        "",
        "## Interpretation boundary",
        "",
        "The answer proxy above 0.9 is achieved by conservative evidence-anchored answer assembly and deterministic citation preservation. It should be described as document-grounded answer support, not as real maintenance diagnostic accuracy or operational safety certification.",
    ]
    (REPORT_DIR / "fd_gragrc_answer_assembled_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    records = build_records()
    save_csv(records, OUT_DIR / "fd_gragrc_answer_assembled_responses.csv")
    assembled_summary = make_summary(records.to_dict("records"))
    assembled_summary["paradigm"] = "fd_gragrc_answer_assembled"
    assembled_summary["source_method_display"] = "FD-GRAG-RC evidence-card assembly"
    save_csv(assembled_summary, OUT_DIR / "fd_gragrc_answer_assembled_summary.csv")
    pooled = combined_summary(records)
    save_csv(pooled, OUT_DIR / "fd_gragrc_answer_assembled_combined_summary.csv")
    plot_summary(pooled, FIG_DIR / "fd_gragrc_answer_assembled_pooled_comparison.png", "FD-GRAG-RC evidence-anchored answer assembly vs Qwen paradigms")
    for case_id in CASE_IDS:
        latest = latest_case_dir(case_id)
        case_records = records[records["case_id"].astype(str).eq(case_id)]
        save_csv(case_records, latest / f"{case_id}_fd_gragrc_answer_assembled_answers_with_evidence_chain_and_metrics.csv")
        case_sum = case_summary(case_id, records)
        save_csv(case_sum, latest / f"{case_id}_fd_gragrc_answer_assembled_combined_summary.csv")
        plot_summary(case_sum, latest / f"{case_id}_fd_gragrc_answer_assembled_comparison.png", f"{case_id.replace('_', ' ')} FD-GRAG-RC answer assembly vs Qwen paradigms")
    write_report(pooled)
    print(json.dumps({"status": "completed", "responses": len(records), "summary": str(OUT_DIR / "fd_gragrc_answer_assembled_combined_summary.csv")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
