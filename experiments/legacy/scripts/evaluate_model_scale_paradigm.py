from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CASE_IDS = ["Boeing_737", "Cessna_172"]
OUT_DIR = ROOT / "04_results" / "model_scale_paradigm"
REPORT_DIR = ROOT / "reports"

SYSTEM_PROMPT = """You are an aviation maintenance evidence-grounded QA assistant.
Answer only from the provided evidence package when evidence is provided.
If the evidence is insufficient, say that the evidence is insufficient.
Do not claim operational safety certification or real maintenance correctness.
Return a concise answer and cite evidence ids when evidence ids are available."""


def table_dir(case_id: str) -> Path:
    case_root = ROOT / "Cases" / case_id
    test = next(p for p in case_root.iterdir() if p.is_dir() and p.name.startswith("7"))
    return test / "sota_evidence_chain_tables"


def load_fdgragrc_rows(case_ids: list[str]) -> pd.DataFrame:
    rows = []
    for case_id in case_ids:
        path = table_dir(case_id) / f"{case_id}_complete_sota_evidence_chains.csv"
        df = pd.read_csv(path)
        fd = df[
            df["method"].astype(str).str.contains("FD-GRAG-RC", case=False, na=False)
            | df["method_display"].astype(str).str.contains("FD-GRAG-RC", case=False, na=False)
        ].copy()
        fd["case_id"] = case_id
        fd["source_chain_csv"] = str(path)
        rows.append(fd)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def truncate_text(text: Any, max_chars: int) -> str:
    text = "" if pd.isna(text) else str(text)
    return text if len(text) <= max_chars else text[:max_chars] + "\n...[truncated]"


def build_prompt(row: pd.Series, paradigm: str) -> list[dict[str, str]]:
    question = str(row["question"])
    aircraft = str(row.get("aircraft_model", ""))
    if paradigm == "direct_model_only":
        user = (
            f"Aircraft: {aircraft}\n"
            f"Question: {question}\n\n"
            "Answer the question. If you cannot ground the answer in known maintenance evidence, state the uncertainty."
        )
    elif paradigm == "fd_grag_rc_evidence_package":
        evidence = truncate_text(row.get("evidence_chain_top10", ""), 5200)
        user = (
            f"Aircraft: {aircraft}\n"
            f"Question: {question}\n\n"
            "FD-GRAG-RC retrieved evidence package:\n"
            f"{evidence}\n\n"
            "Use only this evidence package. Cite evidence ids such as EVID_... when making a maintenance statement."
        )
    else:
        raise ValueError(f"Unknown paradigm: {paradigm}")
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def parse_model_specs(args: argparse.Namespace) -> list[dict[str, str]]:
    if args.model_config:
        specs = pd.read_csv(args.model_config).fillna("")
        return specs.to_dict("records")
    models = [m.strip() for m in (args.models or "").split(",") if m.strip()]
    labels = [x.strip() for x in (args.scale_labels or "").split(",") if x.strip()]
    if labels and len(labels) != len(models):
        raise ValueError("--scale-labels must have the same length as --models")
    if not labels:
        default = ["small", "medium", "large"]
        labels = [default[i] if i < len(default) else f"scale_{i+1}" for i in range(len(models))]
    return [{"scale": lab, "provider": "openai_compatible", "model": model} for lab, model in zip(labels, models)]


def call_openai_compatible(
    *,
    model: str,
    messages: list[dict[str, str]],
    api_key: str,
    base_url: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
) -> tuple[str, dict[str, Any]]:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    content = data["choices"][0]["message"]["content"]
    return content, data.get("usage", {})


def char_bigram_f1(pred: str, gold: str) -> float:
    pred = re.sub(r"\s+", "", pred or "")
    gold = re.sub(r"\s+", "", gold or "")
    if not pred or not gold:
        return 0.0
    p = Counter(pred[i : i + 2] for i in range(max(1, len(pred) - 1)))
    g = Counter(gold[i : i + 2] for i in range(max(1, len(gold) - 1)))
    overlap = sum((p & g).values())
    precision = overlap / max(1, sum(p.values()))
    recall = overlap / max(1, sum(g.values()))
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def evaluate_answer(answer: str, row: pd.Series) -> dict[str, Any]:
    answer = answer or ""
    evidence_ids = set(re.findall(r"EVID[A-Z0-9_]*", str(row.get("top_evidence_ids", ""))))
    cited_ids = set(re.findall(r"EVID[A-Z0-9_]*", answer))
    valid = cited_ids & evidence_ids
    refusal_terms = ["insufficient", "not enough evidence", "无法", "不足", "不能确定", "缺少证据"]
    return {
        "answer_chars": len(answer),
        "refusal_detected": any(term.lower() in answer.lower() for term in refusal_terms),
        "cited_evidence_count": len(cited_ids),
        "valid_cited_evidence_count": len(valid),
        "citation_precision_proxy": len(valid) / max(1, len(cited_ids)) if cited_ids else 0.0,
        "char_bigram_f1_to_gold_proxy": round(char_bigram_f1(answer, str(row.get("gold_answer", ""))), 4),
    }


def write_template_config(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["scale", "provider", "model"])
        writer.writeheader()
        writer.writerow({"scale": "small", "provider": "openai_compatible", "model": "PUT_SMALL_MODEL_ID_HERE"})
        writer.writerow({"scale": "medium", "provider": "openai_compatible", "model": "PUT_MEDIUM_MODEL_ID_HERE"})
        writer.writerow({"scale": "large", "provider": "openai_compatible", "model": "PUT_LARGE_MODEL_ID_HERE"})


def write_status_report(
    *,
    status: str,
    reason: str,
    prompt_manifest: Path,
    model_specs: list[dict[str, str]],
    rows_count: int,
    output_files: list[Path],
) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Model Scale Paradigm Evaluation Status",
        "",
        f"- Status: {status}",
        f"- Reason: {reason}",
        f"- Test rows prepared: {rows_count}",
        f"- Prompt manifest: `{prompt_manifest}`",
        "- Paradigms: `direct_model_only` vs `fd_grag_rc_evidence_package`",
        "- Data boundary: gold answers and gold evidence labels are used only for post-hoc evaluation, not in prompts.",
        "- Built-in GPT boundary: the interactive Codex/GPT model in this chat is not a reproducible local baseline because it has no fixed callable model id, temperature, version pin, or batch API endpoint exposed to this project.",
        "",
        "## Requested Model Specs",
        "",
        "| Scale | Provider | Model |",
        "|---|---|---|",
    ]
    for spec in model_specs:
        lines.append(f"| {spec.get('scale','')} | {spec.get('provider','')} | {spec.get('model','')} |")
    lines += [
        "",
        "## Output Files",
        "",
    ]
    for path in output_files:
        lines.append(f"- `{path}`")
    lines += [
        "",
        "## Claim Boundary",
        "",
        "These model-scale tests, when run, evaluate answer generation from existing retrieved evidence packages on the current test set. They do not change retrieval metrics, benchmark splits, training, or original CSV/JSON/NPY artifact values.",
    ]
    (REPORT_DIR / "model_scale_paradigm_evaluation_status.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default=",".join(CASE_IDS))
    parser.add_argument("--models", default="")
    parser.add_argument("--scale-labels", default="")
    parser.add_argument("--model-config", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=700)
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    template_config = REPORT_DIR / "model_scale_models_template.csv"
    write_template_config(template_config)

    case_ids = [c.strip() for c in args.cases.split(",") if c.strip()]
    rows = load_fdgragrc_rows(case_ids)
    if args.limit and args.limit > 0:
        rows = rows.head(args.limit).copy()
    model_specs = parse_model_specs(args)
    prompt_records = []
    for spec in model_specs:
        for paradigm in ["direct_model_only", "fd_grag_rc_evidence_package"]:
            for _, row in rows.iterrows():
                prompt_records.append(
                    {
                        "case_id": row["case_id"],
                        "query_id": row["query_id"],
                        "question_id": row.get("question_id", ""),
                        "aircraft_model": row.get("aircraft_model", ""),
                        "query_type": row.get("query_type", ""),
                        "scale": spec.get("scale", ""),
                        "provider": spec.get("provider", ""),
                        "model": spec.get("model", ""),
                        "paradigm": paradigm,
                        "messages_json": json.dumps(build_prompt(row, paradigm), ensure_ascii=False),
                        "source_chain_csv": row.get("source_chain_csv", ""),
                    }
                )
    prompt_manifest = OUT_DIR / "model_scale_paradigm_prompt_manifest.csv"
    pd.DataFrame(prompt_records).to_csv(prompt_manifest, index=False, encoding="utf-8-sig")

    output_files = [prompt_manifest, template_config]
    if not model_specs:
        write_status_report(
            status="not_run",
            reason="No model ids were provided. Fill reports/model_scale_models_template.csv or pass --models.",
            prompt_manifest=prompt_manifest,
            model_specs=model_specs,
            rows_count=len(rows),
            output_files=output_files,
        )
        print(json.dumps({"status": "not_run", "reason": "no models", "prompt_rows": len(prompt_records)}, ensure_ascii=False, indent=2))
        return

    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    if not args.run:
        write_status_report(
            status="prepared_not_run",
            reason="Prompts were prepared. Re-run with --run after setting OPENAI_API_KEY or an OpenAI-compatible base URL.",
            prompt_manifest=prompt_manifest,
            model_specs=model_specs,
            rows_count=len(rows),
            output_files=output_files,
        )
        print(json.dumps({"status": "prepared_not_run", "prompt_rows": len(prompt_records)}, ensure_ascii=False, indent=2))
        return
    if not api_key:
        write_status_report(
            status="not_run",
            reason="OPENAI_API_KEY is not set in the current environment.",
            prompt_manifest=prompt_manifest,
            model_specs=model_specs,
            rows_count=len(rows),
            output_files=output_files,
        )
        print(json.dumps({"status": "not_run", "reason": "missing OPENAI_API_KEY", "prompt_rows": len(prompt_records)}, ensure_ascii=False, indent=2))
        return

    source_rows = {(r["case_id"], r["query_id"]): r for _, r in rows.iterrows()}
    response_records = []
    for rec in prompt_records:
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
        row = source_rows[(rec["case_id"], rec["query_id"])]
        metrics = evaluate_answer(answer, row)
        response_records.append(
            {
                **{k: rec[k] for k in ["case_id", "query_id", "question_id", "aircraft_model", "query_type", "scale", "provider", "model", "paradigm"]},
                "status": status,
                "error": error,
                "latency_s": round(time.time() - started, 3),
                "answer": answer,
                "usage_json": json.dumps(usage, ensure_ascii=False),
                **metrics,
            }
        )
    responses_path = OUT_DIR / "model_scale_paradigm_responses.csv"
    pd.DataFrame(response_records).to_csv(responses_path, index=False, encoding="utf-8-sig")
    ok = pd.DataFrame(response_records)
    summary = (
        ok.groupby(["scale", "model", "paradigm"], dropna=False)
        .agg(
            n=("query_id", "count"),
            ok_rate=("status", lambda s: (s == "ok").mean()),
            mean_latency_s=("latency_s", "mean"),
            mean_refusal_rate=("refusal_detected", "mean"),
            mean_citation_precision_proxy=("citation_precision_proxy", "mean"),
            mean_char_bigram_f1_to_gold_proxy=("char_bigram_f1_to_gold_proxy", "mean"),
        )
        .reset_index()
    )
    summary_path = OUT_DIR / "model_scale_paradigm_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    output_files.extend([responses_path, summary_path])
    write_status_report(
        status="completed",
        reason="Model calls completed through an OpenAI-compatible chat completion endpoint.",
        prompt_manifest=prompt_manifest,
        model_specs=model_specs,
        rows_count=len(rows),
        output_files=output_files,
    )
    print(json.dumps({"status": "completed", "responses": len(response_records), "summary": str(summary_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
