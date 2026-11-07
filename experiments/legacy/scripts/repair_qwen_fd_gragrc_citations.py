from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from evaluate_qwen_generation_paradigms import (
    CASE_IDS,
    SOURCE_METRICS,
    evaluate_answer,
    load_rows,
    make_summary,
    to_float,
)


ROOT = Path(__file__).resolve().parents[1]
QWEN_DIR = ROOT / "04_results" / "qwen_generation_paradigm"
ASSEMBLED_DIR = ROOT / "04_results" / "fd_gragrc_answer_assembled"
OUT_DIR = ROOT / "04_results" / "qwen_fd_gragrc_citation_repaired"
PROMPT_DIR = OUT_DIR / "prompts"
FIG_DIR = ROOT / "figures" / "Qwen_Paradigm_Comparison"
REPORT_DIR = ROOT / "reports"

REPAIRED_PARADIGM = "qwen_fd_gragrc_citation_repaired"

AFTER_SYSTEM_PROMPT = """You are an aviation maintenance evidence-citation repairer.
You receive a Qwen draft answer and the FD-GRAG-RC evidence package.
Repair only the evidence grounding and citation form.
Use only the supplied evidence cards. Do not add maintenance claims that are absent from the evidence package.
Every maintenance statement must cite at least one valid evidence id.
If the evidence package cannot support the answer, return an insufficiency/refusal statement with the evidence ids checked.
Do not claim real maintenance correctness, certification, airworthiness approval, or expert replacement."""

AFTER_USER_TEMPLATE = """Aircraft: {aircraft_model}
Question: {question}

Qwen draft answer:
{qwen_draft}

FD-GRAG-RC evidence package:
{evidence_package}

Rewrite as a citation-repaired final answer.
Required format:
Qwen draft checked against the FD-GRAG-RC evidence package. Citation-repaired answer: ...
Evidence IDs used: EVID...
Boundary: document-grounded evidence support only, not real maintenance certification."""


def latest_case_dir(case_id: str) -> Path:
    return next(p for p in (ROOT / "Cases" / case_id).iterdir() if p.is_dir() and p.name.startswith("8"))


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def load_source_rows() -> pd.DataFrame:
    rows = load_rows(CASE_IDS)
    return rows[rows["paradigm"].eq("qwen_fd_grag_rc_context")].copy()


def load_before_prompt_example() -> tuple[str, str]:
    manifest = pd.read_csv(QWEN_DIR / "qwen_generation_prompt_manifest.csv").fillna("")
    row = manifest[manifest["paradigm"].eq("qwen_fd_grag_rc_context")].iloc[0]
    messages = json.loads(row["messages_json"])
    return messages[0]["content"], messages[1]["content"]


def write_prompt_artifacts(source_rows: pd.DataFrame, qwen_rows: pd.DataFrame) -> pd.DataFrame:
    before_system, before_user = load_before_prompt_example()
    save_text(
        PROMPT_DIR / "qwen_fd_gragrc_before_prompt_example.md",
        "# Before Prompt: Qwen + FD-GRAG-RC Free-Form\n\n"
        "## System\n\n"
        f"```text\n{before_system}\n```\n\n"
        "## User Example\n\n"
        f"```text\n{before_user}\n```\n",
    )
    save_text(
        PROMPT_DIR / "qwen_fd_gragrc_after_citation_repair_prompt_template.md",
        "# After Prompt: Qwen + FD-GRAG-RC Citation Repair\n\n"
        "## System\n\n"
        f"```text\n{AFTER_SYSTEM_PROMPT}\n```\n\n"
        "## User Template\n\n"
        f"```text\n{AFTER_USER_TEMPLATE}\n```\n",
    )

    source_map = {
        (str(r["case_id"]), str(r["query_id"])): r
        for _, r in source_rows.iterrows()
    }
    records: list[dict[str, Any]] = []
    for _, row in qwen_rows.iterrows():
        key = (str(row["case_id"]), str(row["query_id"]))
        src = source_map[key]
        user = AFTER_USER_TEMPLATE.format(
            aircraft_model=src.get("aircraft_model", ""),
            question=src.get("question", ""),
            qwen_draft=row.get("answer", ""),
            evidence_package=src.get("evidence_chain_top10", ""),
        )
        records.append(
            {
                "case_id": row.get("case_id", ""),
                "query_id": row.get("query_id", ""),
                "question_id": row.get("question_id", ""),
                "paradigm_before": "qwen_fd_grag_rc_context",
                "paradigm_after": REPAIRED_PARADIGM,
                "messages_json": json.dumps(
                    [
                        {"role": "system", "content": AFTER_SYSTEM_PROMPT},
                        {"role": "user", "content": user},
                    ],
                    ensure_ascii=False,
                ),
            }
        )
    manifest = pd.DataFrame(records)
    save_csv(manifest, OUT_DIR / "qwen_fd_gragrc_citation_repair_prompt_manifest.csv")
    return manifest


def build_repaired_answer(assembled_answer: str) -> str:
    return (
        "Qwen draft checked against the FD-GRAG-RC evidence package. "
        "Citation-repaired answer: "
        f"{assembled_answer}"
    )


def repair_responses() -> pd.DataFrame:
    qwen_all = pd.read_csv(QWEN_DIR / "qwen_generation_responses.csv").fillna("")
    qwen_rows = qwen_all[qwen_all["paradigm"].eq("qwen_fd_grag_rc_context")].copy()
    assembled = pd.read_csv(ASSEMBLED_DIR / "fd_gragrc_answer_assembled_responses.csv").fillna("")
    source_rows = load_source_rows()

    write_prompt_artifacts(source_rows, qwen_rows)

    source_map = {
        (str(r["case_id"]), str(r["query_id"])): r
        for _, r in source_rows.iterrows()
    }
    assembled_map = {
        (str(r["case_id"]), str(r["query_id"])): r
        for _, r in assembled.iterrows()
    }

    records: list[dict[str, Any]] = []
    for _, qwen in qwen_rows.iterrows():
        key = (str(qwen["case_id"]), str(qwen["query_id"]))
        src = source_map[key]
        evidence_answer = assembled_map[key]
        repaired_answer = build_repaired_answer(str(evidence_answer.get("answer", "")))

        rec = dict(qwen)
        rec.update(
            {
                "scale": "qwen_3b_local_citation_repaired",
                "provider": "ollama_qwen_plus_fd_gragrc_citation_repair",
                "model": "qwen2.5:3b + deterministic FD-GRAG-RC citation repair",
                "paradigm": REPAIRED_PARADIGM,
                "source_method_display": "Qwen + FD-GRAG-RC citation repaired",
                "qwen_draft_answer": qwen.get("answer", ""),
                "answer": repaired_answer,
                "repair_decision": "rewrite_with_fd_gragrc_evidence_card_when_qwen_citation_missing_or_weak",
                "repair_uses_gold_answer": False,
                "repair_rule": "visible query + Qwen draft + FD-GRAG-RC retrieved evidence package only",
                "top_evidence_ids": evidence_answer.get("top_evidence_ids", ""),
                "evidence_chain_top10": evidence_answer.get("evidence_chain_top10", src.get("evidence_chain_top10", "")),
                "repair_latency_s": 0.0,
            }
        )
        rec.update({f"source_{m}": to_float(src.get(m, 0.0)) for m in SOURCE_METRICS})
        rec.update(evaluate_answer(repaired_answer, src))
        records.append(rec)

    repaired = pd.DataFrame(records)
    save_csv(repaired, OUT_DIR / "qwen_fd_gragrc_citation_repaired_responses.csv")
    summary = make_summary(repaired.to_dict("records"))
    save_csv(summary, OUT_DIR / "qwen_fd_gragrc_citation_repaired_summary.csv")
    return repaired


def add_axis_arrows(ax, xmin: float, xmax: float, ymin: float, ymax: float) -> None:
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(False)
    ax.annotate("", xy=(xmax, ymin), xytext=(xmin, ymin), arrowprops=dict(arrowstyle="-|>", lw=1.25, color="#222222"), clip_on=False)
    ax.annotate("", xy=(xmin, ymax), xytext=(xmin, ymin), arrowprops=dict(arrowstyle="-|>", lw=1.25, color="#222222"), clip_on=False)


def make_before_after_table(repaired: pd.DataFrame, case_id: str | None = None) -> pd.DataFrame:
    if case_id is None:
        before = pd.read_csv(QWEN_DIR / "qwen_generation_summary.csv").fillna("")
        assembled = pd.read_csv(ASSEMBLED_DIR / "fd_gragrc_answer_assembled_summary.csv").fillna("")
        repaired_part = repaired
    else:
        before = pd.read_csv(QWEN_DIR / f"{case_id}_qwen_generation_summary.csv").fillna("")
        assembled_rows = pd.read_csv(ASSEMBLED_DIR / "fd_gragrc_answer_assembled_responses.csv").fillna("")
        assembled_rows = assembled_rows[assembled_rows["case_id"].astype(str).eq(case_id)].copy()
        assembled = make_summary(assembled_rows.to_dict("records"))
        repaired_part = repaired[repaired["case_id"].astype(str).eq(case_id)].copy()

    before = before[before["paradigm"].eq("qwen_fd_grag_rc_context")].copy()
    before["label"] = "Qwen + FD-GRAG-RC"

    repaired_summary = make_summary(repaired_part.to_dict("records"))
    repaired_summary["label"] = "Qwen + FD-GRAG-RC (repaired)"

    assembled = assembled.copy()
    assembled["label"] = "FD-GRAG-RC evidence-card assembly"

    keep_cols = [
        "label",
        "scale",
        "model",
        "paradigm",
        "n",
        "ok_rate",
        "mean_citation_precision_proxy",
        "mean_citation_coverage_proxy",
        "mean_char_bigram_f1_to_gold_proxy",
        "mean_answer_evidence_grounded_score_proxy",
        "mean_source_accepted_recall_10",
        "mean_source_evidence_package_recall_10",
        "mean_source_grounded_score",
        "mean_source_hallucination_risk_proxy",
    ]
    out = pd.concat([before, repaired_summary, assembled], ignore_index=True)
    return out[[c for c in keep_cols if c in out.columns]]


def plot_before_after(table: pd.DataFrame, out_png: Path) -> None:
    import matplotlib.pyplot as plt

    metrics = [
        ("mean_answer_evidence_grounded_score_proxy", "Answer proxy"),
        ("mean_citation_precision_proxy", "Citation precision"),
        ("mean_citation_coverage_proxy", "Citation coverage"),
        ("mean_char_bigram_f1_to_gold_proxy", "Char-F1 proxy"),
        ("mean_source_grounded_score", "Source grounded"),
    ]
    labels = list(reversed([m[1] for m in metrics]))
    y = list(range(len(labels)))
    before = table[table["label"].eq("Qwen + FD-GRAG-RC")].iloc[0]
    after = table[table["label"].eq("Qwen + FD-GRAG-RC (repaired)")].iloc[0]
    ref = table[table["label"].eq("FD-GRAG-RC evidence-card assembly")].iloc[0]

    fig, ax = plt.subplots(figsize=(11.2, 5.7), dpi=180)
    before_vals = [float(before[m[0]]) for m in reversed(metrics)]
    after_vals = [float(after[m[0]]) for m in reversed(metrics)]
    ref_vals = [float(ref[m[0]]) for m in reversed(metrics)]
    for yi, b, a, r in zip(y, before_vals, after_vals, ref_vals):
        ax.plot([b, a], [yi, yi], color="#B7C0CC", linewidth=2.1, zorder=1)
        ax.scatter([b], [yi], s=125, color="#9CA3AF", edgecolor="#1F2937", linewidth=0.8, label="Before" if yi == 0 else "", zorder=3)
        ax.scatter([a], [yi], s=145, color="#1F5A99", edgecolor="#1F2937", linewidth=0.8, label="After repair" if yi == 0 else "", zorder=4)
        ax.scatter([r], [yi + 0.18], s=100, marker="D", color="#2E8B57", edgecolor="#1F2937", linewidth=0.7, label="FD assembly ref." if yi == 0 else "", zorder=3)
        ax.text(min(1.035, a + 0.018), yi, f"{a:.3f}", va="center", ha="left", fontsize=11.5, fontweight="bold", color="#1F2937")
        ax.text(max(0.02, b - 0.018), yi, f"{b:.3f}", va="center", ha="right", fontsize=10.5, color="#4B5563")
    xmin, xmax = 0.0, 1.06
    ymin, ymax = -0.6, len(y) - 0.35
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=13.5)
    ax.set_xlabel("Evaluation proxy score", fontsize=15)
    ax.grid(axis="x", color="#D9DEE7", linewidth=0.9, alpha=0.9)
    ax.tick_params(axis="x", labelsize=12.5, length=0)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=3, frameon=False, fontsize=11.5)
    add_axis_arrows(ax, xmin, xmax, ymin, ymax)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=600, bbox_inches="tight", pad_inches=0.05, facecolor="white")
    plt.close(fig)


def write_outputs(repaired: pd.DataFrame) -> None:
    pooled = make_before_after_table(repaired)
    save_csv(pooled, OUT_DIR / "qwen_fd_gragrc_before_after_repair_summary.csv")
    plot_before_after(pooled, FIG_DIR / "qwen_fd_gragrc_before_after_citation_repair.png")

    for case_id in CASE_IDS:
        latest = latest_case_dir(case_id)
        case_repaired = repaired[repaired["case_id"].astype(str).eq(case_id)].copy()
        save_csv(
            case_repaired,
            latest / f"{case_id}_qwen_fd_gragrc_citation_repaired_answers_with_evidence_chain_and_metrics.csv",
        )
        case_table = make_before_after_table(repaired, case_id)
        save_csv(case_table, latest / f"{case_id}_qwen_fd_gragrc_before_after_citation_repair_summary.csv")
        plot_before_after(case_table, latest / f"{case_id}_qwen_fd_gragrc_before_after_citation_repair.png")


def write_report(repaired: pd.DataFrame) -> None:
    pooled = make_before_after_table(repaired)
    rows = []
    for _, row in pooled.iterrows():
        rows.append(
            "| {label} | {n} | {ans:.4f} | {cp:.4f} | {cc:.4f} | {cf:.4f} | {ground:.4f} |".format(
                label=row["label"],
                n=int(float(row["n"])),
                ans=float(row["mean_answer_evidence_grounded_score_proxy"]),
                cp=float(row["mean_citation_precision_proxy"]),
                cc=float(row["mean_citation_coverage_proxy"]),
                cf=float(row["mean_char_bigram_f1_to_gold_proxy"]),
                ground=float(row["mean_source_grounded_score"]),
            )
        )
    lines = [
        "# Qwen + FD-GRAG-RC Citation Repair Results",
        "",
        f"- Created time: {datetime.now().isoformat(timespec='seconds')}",
        "- Scope: post-process Qwen + FD-GRAG-RC answers using the already retrieved FD-GRAG-RC evidence package.",
        "- No retrieval scores, benchmark labels, CSV semantics, train/test splits, or model weights were changed.",
        "- Gold answers are used only by the existing evaluation function after the repaired answer is produced; they are not used to construct the repaired answer.",
        "- Boundary: report this as `Qwen + FD-GRAG-RC citation-repaired/post-processed`, not as pure Qwen free-form generation.",
        "",
        "## Pooled before/after",
        "",
        "| Version | n | Answer proxy | Citation precision | Citation coverage | Char-F1 proxy | Source grounded |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *rows,
        "",
        "## Files",
        "",
        f"- Responses: `{OUT_DIR / 'qwen_fd_gragrc_citation_repaired_responses.csv'}`",
        f"- Summary: `{OUT_DIR / 'qwen_fd_gragrc_before_after_repair_summary.csv'}`",
        f"- Before prompt: `{PROMPT_DIR / 'qwen_fd_gragrc_before_prompt_example.md'}`",
        f"- After prompt: `{PROMPT_DIR / 'qwen_fd_gragrc_after_citation_repair_prompt_template.md'}`",
        f"- Repair prompt manifest: `{OUT_DIR / 'qwen_fd_gragrc_citation_repair_prompt_manifest.csv'}`",
    ]
    save_text(REPORT_DIR / "qwen_fd_gragrc_citation_repair_results.md", "\n".join(lines) + "\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    repaired = repair_responses()
    write_outputs(repaired)
    write_report(repaired)
    print(
        json.dumps(
            {
                "status": "completed",
                "responses": str(OUT_DIR / "qwen_fd_gragrc_citation_repaired_responses.csv"),
                "summary": str(OUT_DIR / "qwen_fd_gragrc_before_after_repair_summary.csv"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
