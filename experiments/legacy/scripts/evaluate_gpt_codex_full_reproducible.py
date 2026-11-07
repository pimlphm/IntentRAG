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
    SOURCE_METRICS,
    build_prompt,
    call_openai_compatible,
    evaluate_answer,
    load_rows,
    make_summary,
    to_float,
)


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "04_results" / "gpt_codex_full_reproducible"
FIG_DIR = ROOT / "figures" / "Qwen_Paradigm_Comparison"
REPORT_DIR = ROOT / "reports"


PARADIGMS = [
    ("direct_model_only", "qwen_direct_no_rag", "Direct / no RAG"),
    ("fd_grag_rc_evidence_package", "qwen_fd_grag_rc_context", "FD-GRAG-RC evidence package"),
]


LABELS = {
    "qwen_direct_no_rag": "Qwen direct",
    "qwen_graphrag_context": "Qwen + GraphRAG",
    "qwen_evidence_flow_context": "Qwen + Evidence-flow",
    "qwen_fd_grag_rc_context": "Qwen + FD-GRAG-RC",
    "fd_gragrc_answer_assembled": "FD-GRAG-RC evidence-card assembly",
    "gpt_oss_direct_model_only": "GPT-OSS direct",
    "gpt_oss_fd_grag_rc_evidence_package": "GPT-OSS + FD-GRAG-RC",
    "codex_direct_model_only": "Codex direct",
    "codex_fd_grag_rc_evidence_package": "Codex + FD-GRAG-RC",
}


PLOT_ORDER = [
    "qwen_direct_no_rag",
    "gpt_oss_direct_model_only",
    "qwen_graphrag_context",
    "qwen_evidence_flow_context",
    "qwen_fd_grag_rc_context",
    "gpt_oss_fd_grag_rc_evidence_package",
    "fd_gragrc_answer_assembled",
]


def latest_case_dir(case_id: str) -> Path:
    return next(p for p in (ROOT / "Cases" / case_id).iterdir() if p.is_dir() and p.name.startswith("8"))


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def model_prefix(provider_label: str) -> str:
    label = provider_label.lower()
    if "codex" in label:
        return "codex"
    if "gpt" in label:
        return "gpt_oss"
    return label.replace("-", "_").replace(" ", "_")


def prepare_rows(case_ids: list[str]) -> pd.DataFrame:
    rows = load_rows(case_ids)
    keep = {source for _, source, _ in PARADIGMS}
    return rows[rows["paradigm"].isin(keep)].copy()


def build_prompt_records(rows: pd.DataFrame, model_specs: list[dict[str, str]], provider_label: str, max_context_chars: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    prefix = model_prefix(provider_label)
    for spec in model_specs:
        for out_paradigm, source_paradigm, display in PARADIGMS:
            for _, row in rows[rows["paradigm"].eq(source_paradigm)].iterrows():
                messages = build_prompt(row, max_context_chars)
                records.append(
                    {
                        "case_id": row.get("case_id", ""),
                        "query_id": row.get("query_id", row.get("question_id", "")),
                        "question_id": row.get("question_id", ""),
                        "aircraft_model": row.get("aircraft_model", ""),
                        "query_type": row.get("query_type", ""),
                        "scale": spec.get("scale", provider_label),
                        "provider": spec.get("provider", provider_label),
                        "model": spec.get("model", ""),
                        "paradigm": f"{prefix}_{out_paradigm}",
                        "source_paradigm": source_paradigm,
                        "source_method": row.get("method", ""),
                        "source_method_display": display,
                        "messages_json": json.dumps(messages, ensure_ascii=False),
                        "source_chain_csv": row.get("source_chain_csv", ""),
                    }
                )
    return records


def parse_model_specs(args: argparse.Namespace) -> list[dict[str, str]]:
    if args.model_config:
        return pd.read_csv(args.model_config).fillna("").to_dict("records")
    models = [x.strip() for x in args.models.split(",") if x.strip()]
    labels = [x.strip() for x in args.scale_labels.split(",") if x.strip()]
    if labels and len(labels) != len(models):
        raise ValueError("--scale-labels must match --models length")
    if not labels:
        labels = [f"model_{i + 1}" for i in range(len(models))]
    return [
        {"scale": scale, "provider": args.backend, "model": model}
        for scale, model in zip(labels, models)
    ]


def backend_connection(args: argparse.Namespace) -> tuple[str, str, str]:
    if args.backend == "ollama":
        return os.environ.get("OLLAMA_API_KEY") or "ollama", os.environ.get("OLLAMA_BASE_URL") or "http://localhost:11434/v1", "GPT-OSS surrogate"
    return os.environ.get("OPENAI_API_KEY", ""), os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1", "Codex/OpenAI API"


def write_status(status: str, reason: str, prompt_manifest: Path, output_files: list[Path], model_specs: list[dict[str, str]]) -> None:
    lines = [
        "# GPT/Codex Full Reproducible Comparison Status",
        "",
        f"- Status: {status}",
        f"- Reason: {reason}",
        f"- Created time: {datetime.now().isoformat(timespec='seconds')}",
        f"- Prompt manifest: `{prompt_manifest}`",
        "- Paradigms: direct/no-RAG and FD-GRAG-RC evidence-package/contract.",
        "- Boundary: the interactive Codex/GPT model in this chat is not directly batch-callable; a fully reproducible Codex baseline requires an API model id and API key.",
        "",
        "## Model specs",
        "",
        "| Scale | Provider | Model |",
        "|---|---|---|",
    ]
    for spec in model_specs:
        lines.append(f"| {spec.get('scale','')} | {spec.get('provider','')} | {spec.get('model','')} |")
    lines += ["", "## Output files", ""]
    for path in output_files:
        lines.append(f"- `{path}`")
    (REPORT_DIR / "gpt_codex_full_reproducible_status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_generation(args: argparse.Namespace) -> pd.DataFrame:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = prepare_rows(args.cases)
    if args.limit > 0:
        keep = rows[["case_id", "query_id"]].drop_duplicates().groupby("case_id", group_keys=False).head(args.limit)
        keys = set(map(tuple, keep[["case_id", "query_id"]].to_records(index=False)))
        rows = rows[rows.apply(lambda r: (r["case_id"], r["query_id"]) in keys, axis=1)].copy()
    model_specs = parse_model_specs(args)
    provider_label = "gpt_oss" if args.backend == "ollama" else "codex"
    prompts = build_prompt_records(rows, model_specs, provider_label, args.max_context_chars)
    prompt_manifest = OUT_DIR / f"{provider_label}_prompt_manifest.csv"
    save_csv(pd.DataFrame(prompts), prompt_manifest)
    output_files = [prompt_manifest]
    if not model_specs:
        write_status("prepared_not_run", "No model ids were provided.", prompt_manifest, output_files, model_specs)
        return pd.DataFrame()
    api_key, base_url, backend_label = backend_connection(args)
    if args.backend == "openai" and not api_key:
        write_status("not_run_missing_api_key", "OPENAI_API_KEY is not set; cannot run official Codex/OpenAI API baseline.", prompt_manifest, output_files, model_specs)
        return pd.DataFrame()
    if not args.run:
        write_status("prepared_not_run", f"{backend_label} prompt manifest prepared. Re-run with --run to execute.", prompt_manifest, output_files, model_specs)
        return pd.DataFrame()

    row_lookup = {
        (str(r.get("case_id", "")), str(r.get("query_id", r.get("question_id", ""))), str(r.get("paradigm", ""))): r
        for _, r in rows.iterrows()
    }
    responses_path = OUT_DIR / f"{provider_label}_responses.csv"
    checkpoint_path = OUT_DIR / f"{provider_label}_checkpoint.csv"
    records: list[dict[str, Any]] = []
    completed: set[tuple[str, str, str, str]] = set()
    if args.resume:
        resume_path = responses_path if responses_path.exists() else checkpoint_path if checkpoint_path.exists() else None
        if resume_path:
            old = pd.read_csv(resume_path).fillna("")
            records = old.to_dict("records")
            completed = {
                (str(r["case_id"]), str(r["query_id"]), str(r["model"]), str(r["paradigm"]))
                for r in records
            }

    pending = [
        p for p in prompts
        if (str(p["case_id"]), str(p["query_id"]), str(p["model"]), str(p["paradigm"])) not in completed
    ]
    for idx, rec in enumerate(pending, 1):
        started = time.time()
        messages = json.loads(rec["messages_json"])
        try:
            answer, usage = call_openai_compatible(
                model=rec["model"],
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
        source_row = row_lookup[(str(rec["case_id"]), str(rec["query_id"]), str(rec["source_paradigm"]))]
        records.append(
            {
                **{k: rec[k] for k in ["case_id", "query_id", "question_id", "aircraft_model", "query_type", "scale", "provider", "model", "paradigm", "source_paradigm", "source_method", "source_method_display"]},
                **{f"source_{m}": to_float(source_row.get(m, 0.0)) for m in SOURCE_METRICS},
                "status": status,
                "error": error,
                "latency_s": round(time.time() - started, 3),
                "answer": answer,
                "usage_json": json.dumps(usage, ensure_ascii=False),
                "top_evidence_ids": source_row.get("top_evidence_ids", ""),
                "evidence_chain_top10": source_row.get("evidence_chain_top10", ""),
                **evaluate_answer(answer, source_row),
            }
        )
        if args.flush_every and idx % args.flush_every == 0:
            save_csv(pd.DataFrame(records), checkpoint_path)
            print(json.dumps({"completed_new": idx, "total_pending": len(pending)}, ensure_ascii=False), flush=True)
    df = pd.DataFrame(records)
    save_csv(df, responses_path)
    summary = make_summary(df.to_dict("records"))
    summary_path = OUT_DIR / f"{provider_label}_summary.csv"
    save_csv(summary, summary_path)
    write_status("completed", f"{backend_label} calls completed.", prompt_manifest, output_files + [responses_path, summary_path], model_specs)
    return df


def add_risk_control(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["mean_source_risk_control_proxy"] = 1.0 - pd.to_numeric(
        df["mean_source_hallucination_risk_proxy"], errors="coerce"
    ).fillna(0.0)
    df.loc[df["paradigm"].astype(str).str.contains("direct"), "mean_source_risk_control_proxy"] = 0.0
    df["label"] = df["paradigm"].map(LABELS).fillna(df.get("source_method_display", df["paradigm"]))
    df["order"] = df["paradigm"].map({p: i for i, p in enumerate(PLOT_ORDER)}).fillna(99)
    return df.sort_values("order").reset_index(drop=True)


def load_existing_qwen_and_assembled(case_id: str | None = None) -> pd.DataFrame:
    qwen_name = "qwen_generation_summary.csv" if case_id is None else f"{case_id}_qwen_generation_summary.csv"
    qwen = pd.read_csv(ROOT / "04_results" / "qwen_generation_paradigm" / qwen_name).fillna("")
    keep = ["qwen_direct_no_rag", "qwen_graphrag_context", "qwen_evidence_flow_context", "qwen_fd_grag_rc_context"]
    qwen = qwen[qwen["paradigm"].isin(keep)].copy()
    frames = [qwen]
    if case_id is None:
        assembled_path = ROOT / "04_results" / "fd_gragrc_answer_assembled" / "fd_gragrc_answer_assembled_summary.csv"
        assembled = pd.read_csv(assembled_path).fillna("") if assembled_path.exists() else pd.DataFrame()
    else:
        assembled_responses = ROOT / "04_results" / "fd_gragrc_answer_assembled" / "fd_gragrc_answer_assembled_responses.csv"
        if assembled_responses.exists():
            assembled_rows = pd.read_csv(assembled_responses).fillna("")
            assembled_rows = assembled_rows[assembled_rows["case_id"].astype(str).eq(case_id)].copy()
            assembled = make_summary(assembled_rows.to_dict("records")) if not assembled_rows.empty else pd.DataFrame()
        else:
            assembled = pd.DataFrame()
    if not assembled.empty:
        assembled["paradigm"] = "fd_gragrc_answer_assembled"
        assembled["source_method_display"] = "FD-GRAG-RC evidence-card assembly"
        frames.append(assembled)
    return pd.concat(frames, ignore_index=True)


def combined_summary(records: pd.DataFrame) -> pd.DataFrame:
    base = load_existing_qwen_and_assembled()
    if not records.empty:
        new = make_summary(records.to_dict("records"))
        # make_summary preserves the unique paradigm names stored in records.
        base = pd.concat([base, new], ignore_index=True)
    return add_risk_control(base)


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
        ("mean_source_grounded_score", "Source grounded", "#2166AC", "D"),
        ("mean_source_risk_control_proxy", "Risk control", "#2E7D32", "^"),
    ]
    y = list(range(len(data)))
    offsets = [-0.18, 0.0, 0.18]
    fig, ax = plt.subplots(figsize=(12.2, 6.7), dpi=180)
    for row_idx, paradigm in enumerate(data["paradigm"]):
        if paradigm in {"gpt_oss_fd_grag_rc_evidence_package", "fd_gragrc_answer_assembled"}:
            ax.axhspan(row_idx - 0.43, row_idx + 0.43, color="#E8F1FF", zorder=0)
    for offset, (col, label, color, marker) in zip(offsets, metrics):
        vals = pd.to_numeric(data[col], errors="coerce").fillna(0.0)
        ax.scatter(vals, [v + offset for v in y], s=110, color=color, marker=marker, edgecolor="#1F2937", linewidth=0.7, label=label, zorder=3)
        for yi, val, paradigm in zip(y, vals, data["paradigm"]):
            if paradigm in {"gpt_oss_fd_grag_rc_evidence_package", "fd_gragrc_answer_assembled"}:
                ax.plot([val + 0.004, min(1.04, val + 0.05)], [yi + offset, yi + offset], linestyle=(0, (2, 2)), color="#7B8794", linewidth=0.8)
                ax.text(min(1.045, val + 0.052), yi + offset, f"{val:.3f}", ha="left", va="center", fontsize=10.2, fontweight="bold")
    xmin, xmax = 0.0, 1.06
    ymin, ymax = -0.65, len(data) - 0.35
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_yticks(y)
    ax.set_yticklabels(data["label"], fontsize=13)
    ax.set_xlabel("Answer / evidence-grounding score", fontsize=15)
    if title:
        ax.set_title(title, fontsize=15, fontweight="bold", pad=12)
    ax.grid(axis="x", color="#D9DEE7", linewidth=0.9, alpha=0.85)
    ax.tick_params(axis="x", labelsize=12, length=0)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=3, frameon=False, fontsize=11)
    add_axis_arrows(ax, xmin, xmax, ymin, ymax)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=600, bbox_inches="tight", pad_inches=0.05, facecolor="white")
    plt.close(fig)


def write_report(summary: pd.DataFrame, records: pd.DataFrame) -> None:
    lines = [
        "# GPT/Codex Reproducible Comparison Results",
        "",
        f"- Created time: {datetime.now().isoformat(timespec='seconds')}",
        "- Official interactive Codex/GPT baseline: not batch-callable from this local project without an OpenAI API key and fixed model id.",
        "- GPT-OSS surrogate: run through local Ollama OpenAI-compatible endpoint when records are present.",
        "- Existing Qwen and FD-GRAG-RC evidence-card assembly rows are read-only comparison rows.",
        "",
        "## Combined summary",
        "",
        "| Row | Answer proxy | Citation precision | Citation coverage | Char-F1 proxy | Source grounded | Risk proxy |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            "| {label} | {ans:.4f} | {cp:.4f} | {cc:.4f} | {cf:.4f} | {ground:.4f} | {risk:.4f} |".format(
                label=row["label"],
                ans=float(row.get("mean_answer_evidence_grounded_score_proxy", 0.0)),
                cp=float(row.get("mean_citation_precision_proxy", 0.0)),
                cc=float(row.get("mean_citation_coverage_proxy", 0.0)),
                cf=float(row.get("mean_char_bigram_f1_to_gold_proxy", 0.0)),
                ground=float(row.get("mean_source_grounded_score", 0.0)),
                risk=float(row.get("mean_source_hallucination_risk_proxy", 0.0)),
            )
        )
    lines += [
        "",
        "## Boundary",
        "",
        "Do not report GPT-OSS surrogate rows as the interactive built-in Codex/GPT model. A true Codex/OpenAI full baseline requires setting `OPENAI_API_KEY`, selecting a fixed model id, and rerunning this script with `--backend openai --run`.",
    ]
    (REPORT_DIR / "gpt_codex_full_reproducible_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_case_outputs(summary: pd.DataFrame, records: pd.DataFrame) -> None:
    for case_id in CASE_IDS:
        latest = latest_case_dir(case_id)
        if not records.empty:
            case_records = records[records["case_id"].astype(str).eq(case_id)]
            save_csv(case_records, latest / f"{case_id}_gpt_oss_surrogate_answers_with_evidence_chain_and_metrics.csv")
            case_new = make_summary(case_records.to_dict("records")) if not case_records.empty else pd.DataFrame()
        else:
            case_new = pd.DataFrame()
        base = load_existing_qwen_and_assembled(case_id)
        if not case_new.empty:
            case_summary = add_risk_control(pd.concat([base, case_new], ignore_index=True))
        else:
            case_summary = add_risk_control(base)
        save_csv(case_summary, latest / f"{case_id}_gpt_codex_surrogate_combined_summary.csv")
        plot_summary(case_summary, latest / f"{case_id}_gpt_codex_surrogate_comparison.png", f"{case_id.replace('_', ' ')} GPT/Codex surrogate comparison")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["openai", "ollama"], default="openai")
    parser.add_argument("--models", default="")
    parser.add_argument("--scale-labels", default="")
    parser.add_argument("--model-config", default="")
    parser.add_argument("--cases", default=",".join(CASE_IDS))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1000)
    parser.add_argument("--max-context-chars", type=int, default=7000)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--flush-every", type=int, default=20)
    args = parser.parse_args()
    args.cases = [c.strip() for c in args.cases.split(",") if c.strip()]
    records = run_generation(args)
    if records.empty and args.backend == "ollama":
        responses_path = OUT_DIR / "gpt_oss_responses.csv"
        if responses_path.exists():
            records = pd.read_csv(responses_path).fillna("")
    summary = combined_summary(records)
    save_csv(summary, OUT_DIR / "gpt_codex_surrogate_combined_summary.csv")
    plot_summary(summary, FIG_DIR / "gpt_codex_surrogate_pooled_comparison.png", "GPT/Codex reproducible comparison boundary and GPT-OSS surrogate")
    write_case_outputs(summary, records)
    write_report(summary, records)
    print(json.dumps({"status": "completed", "records": len(records), "summary": str(OUT_DIR / "gpt_codex_surrogate_combined_summary.csv")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
