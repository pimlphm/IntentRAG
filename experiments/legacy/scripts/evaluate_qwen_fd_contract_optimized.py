import argparse
import json
import os
import time
import urllib.error
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from evaluate_qwen_generation_paradigms import (
    CASE_IDS,
    OUT_DIR as BASE_QWEN_OUT,
    SOURCE_METRICS,
    build_prompt,
    call_openai_compatible,
    evaluate_answer,
    load_rows,
    make_summary,
    to_float,
)


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "04_results" / "qwen_generation_paradigm_contract_optimized"
REPORT_DIR = ROOT / "reports"
FIG_DIR = ROOT / "figures" / "Qwen_Paradigm_Comparison"


LABELS = {
    "qwen_direct_no_rag": "Qwen direct",
    "qwen_bm25_context": "Qwen + BM25",
    "qwen_hybrid_context": "Qwen + Hybrid",
    "qwen_graphrag_context": "Qwen + GraphRAG",
    "qwen_evidence_flow_context": "Qwen + Evidence-flow",
    "qwen_fd_grag_rc_context": "Qwen + FD-GRAG-RC",
    "qwen_fd_grag_rc_contract": "Qwen + FD-GRAG-RC (contract)",
}


PLOT_ORDER = [
    "qwen_direct_no_rag",
    "qwen_bm25_context",
    "qwen_hybrid_context",
    "qwen_graphrag_context",
    "qwen_evidence_flow_context",
    "qwen_fd_grag_rc_context",
    "qwen_fd_grag_rc_contract",
]


def latest_case_dir(case_id: str) -> Path:
    case_root = ROOT / "Cases" / case_id
    return next(p for p in case_root.iterdir() if p.is_dir() and p.name.startswith("8"))


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def append_report(lines: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_fd_contract(args: argparse.Namespace) -> pd.DataFrame:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.cases, query_limit=args.query_limit_per_case)
    rows = rows[rows["paradigm"].eq("qwen_fd_grag_rc_context")].copy()
    rows["paradigm"] = "qwen_fd_grag_rc_contract"
    rows["source_method_display"] = "FD-GRAG-RC contract"

    responses_path = OUT_DIR / "qwen_fd_contract_optimized_responses.csv"
    checkpoint_path = OUT_DIR / "qwen_fd_contract_optimized_checkpoint.csv"

    records: list[dict[str, Any]] = []
    completed: set[tuple[str, str]] = set()
    if args.resume and responses_path.exists():
        old = pd.read_csv(responses_path).fillna("")
        records = old.to_dict("records")
        completed = {(str(r["case_id"]), str(r["query_id"])) for r in records}
    elif args.resume and checkpoint_path.exists():
        old = pd.read_csv(checkpoint_path).fillna("")
        records = old.to_dict("records")
        completed = {(str(r["case_id"]), str(r["query_id"])) for r in records}

    api_key = os.environ.get("OLLAMA_API_KEY") or "ollama"
    base_url = os.environ.get("OLLAMA_BASE_URL") or "http://localhost:11434/v1"
    pending = [r for _, r in rows.iterrows() if (str(r.get("case_id", "")), str(r.get("query_id", ""))) not in completed]

    for idx, row in enumerate(pending, 1):
        started = time.time()
        messages = build_prompt(row, args.max_context_chars)
        try:
            answer, usage = call_openai_compatible(
                model=args.model,
                messages=messages,
                api_key=api_key,
                base_url=base_url,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
            )
            status = "ok"
            error = ""
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
            answer = ""
            usage = {}
            status = "error"
            error = str(exc)

        record = {
            "case_id": row.get("case_id", ""),
            "query_id": row.get("query_id", row.get("question_id", "")),
            "question_id": row.get("question_id", ""),
            "aircraft_model": row.get("aircraft_model", ""),
            "query_type": row.get("query_type", ""),
            "scale": "qwen_3b_local",
            "provider": "ollama_openai_compatible",
            "model": args.model,
            "paradigm": "qwen_fd_grag_rc_contract",
            "source_method": row.get("method", ""),
            "source_method_display": "FD-GRAG-RC contract",
            **{f"source_{m}": to_float(row.get(m, 0.0)) for m in SOURCE_METRICS},
            "status": status,
            "error": error,
            "latency_s": round(time.time() - started, 3),
            "answer": answer,
            "usage_json": json.dumps(usage, ensure_ascii=False),
            "top_evidence_ids": row.get("top_evidence_ids", ""),
            "evidence_chain_top10": row.get("evidence_chain_top10", ""),
            **evaluate_answer(answer, row),
        }
        records.append(record)
        if args.flush_every and idx % args.flush_every == 0:
            save_csv(pd.DataFrame(records), checkpoint_path)
            print(json.dumps({"completed_new": idx, "total_pending": len(pending)}, ensure_ascii=False), flush=True)

    result = pd.DataFrame(records)
    save_csv(result, responses_path)
    summary = make_summary(result.to_dict("records"))
    summary["paradigm"] = "qwen_fd_grag_rc_contract"
    summary["source_method_display"] = "FD-GRAG-RC contract"
    save_csv(summary, OUT_DIR / "qwen_fd_contract_optimized_summary.csv")
    return result


def combined_summary(opt_records: pd.DataFrame) -> pd.DataFrame:
    base_summary = pd.read_csv(BASE_QWEN_OUT / "qwen_generation_summary.csv").fillna("")
    opt_summary = make_summary(opt_records.to_dict("records"))
    opt_summary["paradigm"] = "qwen_fd_grag_rc_contract"
    opt_summary["source_method_display"] = "FD-GRAG-RC contract"
    combined = pd.concat([base_summary, opt_summary], ignore_index=True)
    combined["mean_source_risk_control_proxy"] = 1.0 - pd.to_numeric(
        combined["mean_source_hallucination_risk_proxy"], errors="coerce"
    ).fillna(0.0)
    combined.loc[combined["paradigm"].eq("qwen_direct_no_rag"), "mean_source_risk_control_proxy"] = 0.0
    combined["label"] = combined["paradigm"].map(LABELS).fillna(combined["source_method_display"])
    combined["order"] = combined["paradigm"].map({p: i for i, p in enumerate(PLOT_ORDER)}).fillna(99)
    return combined.sort_values("order").reset_index(drop=True)


def case_combined_summary(case_id: str, opt_records: pd.DataFrame) -> pd.DataFrame:
    case_base = pd.read_csv(BASE_QWEN_OUT / f"{case_id}_qwen_generation_summary.csv").fillna("")
    case_opt = make_summary(opt_records[opt_records["case_id"].astype(str).eq(case_id)].to_dict("records"))
    case_opt["paradigm"] = "qwen_fd_grag_rc_contract"
    case_opt["source_method_display"] = "FD-GRAG-RC contract"
    combined = pd.concat([case_base, case_opt], ignore_index=True)
    combined["mean_source_risk_control_proxy"] = 1.0 - pd.to_numeric(
        combined["mean_source_hallucination_risk_proxy"], errors="coerce"
    ).fillna(0.0)
    combined.loc[combined["paradigm"].eq("qwen_direct_no_rag"), "mean_source_risk_control_proxy"] = 0.0
    combined["label"] = combined["paradigm"].map(LABELS).fillna(combined["source_method_display"])
    combined["order"] = combined["paradigm"].map({p: i for i, p in enumerate(PLOT_ORDER)}).fillna(99)
    return combined.sort_values("order").reset_index(drop=True)


def add_axis_arrows(ax, xmin: float, xmax: float, ymin: float, ymax: float) -> None:
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(False)
    ax.annotate("", xy=(xmax, ymin), xytext=(xmin, ymin), arrowprops=dict(arrowstyle="-|>", lw=1.2, color="#222222"), clip_on=False)
    ax.annotate("", xy=(xmin, ymax), xytext=(xmin, ymin), arrowprops=dict(arrowstyle="-|>", lw=1.2, color="#222222"), clip_on=False)


def plot_summary(df: pd.DataFrame, out_png: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    data = df[df["paradigm"].isin(PLOT_ORDER)].copy()
    data["label"] = data["paradigm"].map(LABELS).fillna(data.get("source_method_display", data["paradigm"]))
    data = data.sort_values("order")
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
        if paradigm == "qwen_fd_grag_rc_contract":
            ax.axhspan(row_idx - 0.45, row_idx + 0.45, color="#E8F1FF", zorder=0)
    for offset, (col, label, color, marker) in zip(offsets, metrics):
        vals = pd.to_numeric(data[col], errors="coerce").fillna(0.0)
        ax.scatter(vals, [v + offset for v in y], s=110, color=color, marker=marker, edgecolor="#1F2937", linewidth=0.7, label=label, zorder=3)
        for yi, val, paradigm in zip(y, vals, data["paradigm"]):
            if paradigm == "qwen_fd_grag_rc_contract":
                ax.plot([val + 0.004, min(1.04, val + 0.05)], [yi + offset, yi + offset], linestyle=(0, (2, 2)), color="#7B8794", linewidth=0.8, zorder=2)
                ax.text(min(1.045, val + 0.052), yi + offset, f"{val:.3f}", ha="left", va="center", fontsize=10.5, fontweight="bold", color="#102A43")
    xmin = 0.0
    xmax = 1.06
    ymin = -0.65
    ymax = len(data) - 0.35
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_yticks(y)
    ax.set_yticklabels(data["label"], fontsize=13)
    for tick, paradigm in zip(ax.get_yticklabels(), data["paradigm"]):
        if paradigm == "qwen_fd_grag_rc_contract":
            tick.set_fontweight("bold")
            tick.set_color("#1F4E79")
    ax.set_xlabel("Qwen 3B answer / evidence-package score", fontsize=15)
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


def write_outputs(opt_records: pd.DataFrame) -> None:
    combined = combined_summary(opt_records)
    save_csv(combined, OUT_DIR / "qwen_contract_optimized_combined_summary.csv")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_summary(combined, FIG_DIR / "qwen_contract_optimized_pooled_comparison.png", "Pooled Qwen 3B comparison after FD-GRAG-RC contract optimization")

    lines = [
        "# Qwen 3B FD-GRAG-RC Contract Optimization Results",
        "",
        f"- Created time: {datetime.now().isoformat(timespec='seconds')}",
        "- Base model: Ollama `qwen2.5:3b`.",
        "- Scope: only the FD-GRAG-RC generation interface was optimized and rerun; existing baseline results are read from the prior full Qwen 3B run.",
        "- No benchmark labels, splits, retrieval scores, or existing baseline metrics were modified.",
        "",
        "## Pooled comparison",
        "",
        "| Paradigm | Answer proxy | Accepted R@10 | Package R@10 | Source grounded | Role coverage | Risk proxy |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in combined.iterrows():
        lines.append(
            "| {label} | {ans:.4f} | {acc:.4f} | {pkg:.4f} | {ground:.4f} | {role:.4f} | {risk:.4f} |".format(
                label=row["label"],
                ans=float(row["mean_answer_evidence_grounded_score_proxy"]),
                acc=float(row["mean_source_accepted_recall_10"]),
                pkg=float(row["mean_source_evidence_package_recall_10"]),
                ground=float(row["mean_source_grounded_score"]),
                role=float(row["mean_source_role_coverage_proxy"]),
                risk=float(row["mean_source_hallucination_risk_proxy"]),
            )
        )
    lines += ["", "## Figures", "", f"- `{FIG_DIR / 'qwen_contract_optimized_pooled_comparison.png'}`"]

    for case_id in CASE_IDS:
        latest = latest_case_dir(case_id)
        case_records = opt_records[opt_records["case_id"].astype(str).eq(case_id)]
        save_csv(case_records, latest / f"{case_id}_qwen_fd_contract_optimized_answers_with_evidence_chain_and_metrics.csv")
        case_sum = case_combined_summary(case_id, opt_records)
        save_csv(case_sum, latest / f"{case_id}_qwen_contract_optimized_combined_summary.csv")
        fig_path = latest / f"{case_id}_qwen_contract_optimized_generation_comparison.png"
        plot_summary(case_sum, fig_path, f"{case_id.replace('_', ' ')} Qwen 3B comparison after FD-GRAG-RC contract optimization")
        lines.append(f"- `{fig_path}`")

    append_report(lines, REPORT_DIR / "qwen_fd_contract_optimized_results.md")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default=",".join(CASE_IDS))
    parser.add_argument("--model", default="qwen2.5:3b")
    parser.add_argument("--query-limit-per-case", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument("--max-context-chars", type=int, default=7000)
    parser.add_argument("--timeout", type=int, default=150)
    parser.add_argument("--flush-every", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.cases = [c.strip() for c in args.cases.split(",") if c.strip()]
    opt_records = run_fd_contract(args)
    write_outputs(opt_records)
    print(json.dumps({"status": "completed", "responses": len(opt_records), "out_dir": str(OUT_DIR)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
