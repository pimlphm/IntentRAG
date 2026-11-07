"""Run a model-scale evidence-state diagnostic on real FD-GRAG-RC packages.

The audit asks whether larger local Qwen models already internalize the
evidence-state capability, or whether aviation maintenance answers still need
an explicit organized evidence state.

It is a controlled scale-probe subset, not a full benchmark replacement.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import re
import statistics
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = ROOT / "_pipeline_cache_reuse_v1" / "run_cache_reuse_diagnostic.py"
DEFAULT_CONFIG = ROOT / "configs" / "model_scale_state_v1" / "model_scale_state_v1.json"
RESULTS = ROOT / "04_results" / "model_scale_state_v1"
FIGURES = ROOT / "05_figures" / "model_scale_state_v1"
TABLES = ROOT / "06_tables" / "model_scale_state_v1"
REPORTS = ROOT / "07_reports" / "model_scale_state_v1"
LOGS = ROOT / "08_logs" / "model_scale_state_v1"
CHECKPOINTS = RESULTS / "_checkpoints"
for directory in (RESULTS, FIGURES, TABLES, REPORTS, LOGS, CHECKPOINTS):
    directory.mkdir(parents=True, exist_ok=True)


def load_base_module() -> Any:
    spec = importlib.util.spec_from_file_location("cache_reuse_base", BASE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load base cache-reuse module from {BASE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = load_base_module()


SYSTEM_PROMPT = """You are an aviation maintenance evidence-grounded QA assistant.
Answer concisely. If no evidence package is provided, do not invent document-specific procedures, thresholds, or evidence IDs.
If evidence is provided, use only that evidence and cite evidence IDs.
If the evidence does not support a requested action, threshold, or authority claim, say that the evidence is insufficient.
Do not claim safety certification or real maintenance correctness."""


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    columns = fieldnames or list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def resolve_tokenizer_json(config: dict[str, Any]) -> Path:
    """Resolve a real tokenizer.json file, avoiding Windows empty-path traps."""
    configured_raw = str(config.get("tokenizer_json", "") or "").strip()
    if configured_raw:
        configured = Path(configured_raw)
        if configured.is_file():
            return configured
        if configured.is_dir():
            nested = configured / "tokenizer.json"
            if nested.is_file():
                return nested

    hint = str(config.get("tokenizer_hint", "Qwen3.5-0.8B")).lower()
    matches: list[Path] = []
    for root in getattr(BASE, "TOKENIZER_SEARCH_ROOTS", []):
        root_path = Path(root)
        if not root_path.exists():
            continue
        try:
            for candidate in root_path.rglob("tokenizer.json"):
                if hint in str(candidate).lower():
                    matches.append(candidate)
        except OSError:
            continue
    if matches:
        return sorted(matches)[0]
    raise FileNotFoundError(f"Unable to locate tokenizer.json with hint '{hint}'")


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        return default if value in ("", None) else float(value)
    except (TypeError, ValueError):
        return default


def prompt_text(messages: list[dict[str, str]]) -> str:
    return "\n\n".join(f"{message['role'].upper()}:\n{message['content']}" for message in messages)


def answer_rules() -> str:
    return (
        "Answer format requirements:\n"
        "- Use at most 4 short bullet points.\n"
        "- Include finding, action or deferral, authority/condition boundary, and evidence IDs.\n"
        "- Every maintenance claim must cite an evidence ID when evidence is provided.\n"
        "- Keep the answer under about 120 words."
    )


def raw_chain_block(chain: str, max_chars: int) -> str:
    return BASE.shorten(str(chain or "").replace("\r\n", "\n"), max_chars)


def build_messages(row: dict[str, str], state_block: str, variant: str, raw_chain_chars: int) -> list[dict[str, str]]:
    aircraft = row.get("aircraft_model", "")
    question = BASE.clean_question_text(row.get("question", ""))
    if variant == "direct_model_only":
        user = (
            f"Aircraft: {aircraft}\n"
            f"Question: {question}\n\n"
            "No evidence package is provided. Answer only if you can do so without inventing document-specific maintenance details; "
            "otherwise state uncertainty.\n\n"
            f"{answer_rules()}"
        )
    elif variant == "question_first_chain":
        user = (
            f"Aircraft: {aircraft}\n"
            f"Question: {question}\n\n"
            f"Retrieved raw evidence chain:\n{raw_chain_block(row.get('evidence_chain_top10', ''), raw_chain_chars)}\n\n"
            f"{answer_rules()}"
        )
    elif variant == "state_first_state":
        user = (
            f"Aircraft: {aircraft}\n"
            f"{state_block}\n\n"
            f"Question: {question}\n\n"
            f"{answer_rules()}"
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def select_families(
    rows: list[dict[str, str]],
    *,
    family_prefix_k: int,
    min_family_size: int,
    max_families_per_aircraft: int,
) -> list[dict[str, str]]:
    families = BASE.build_families(rows, family_prefix_k=family_prefix_k, min_family_size=min_family_size)
    by_aircraft: defaultdict[str, list[tuple[tuple[str, tuple[str, ...]], list[dict[str, str]]]]] = defaultdict(list)
    for key, items in families.items():
        by_aircraft[key[0]].append((key, items))

    selected: list[dict[str, str]] = []
    family_index = 0
    for aircraft in sorted(by_aircraft):
        candidates = sorted(
            by_aircraft[aircraft],
            key=lambda item: (-len(item[1]), item[1][0].get("query_id", "")),
        )[:max_families_per_aircraft]
        for key, items in candidates:
            for row in sorted(items, key=lambda entry: entry.get("query_id", "")):
                selected_row = dict(row)
                selected_row["scale_family_id"] = f"SF{family_index:03d}"
                selected_row["scale_family_aircraft"] = aircraft
                selected_row["scale_family_size"] = str(len(items))
                selected_row["scale_family_prefix_evidence_ids"] = ";".join(key[1])
                selected.append(selected_row)
            family_index += 1
    return selected


def make_prompt_rows(rows: list[dict[str, str]], config: dict[str, Any], tokenizer: Tokenizer) -> list[dict[str, Any]]:
    prompt_rows: list[dict[str, Any]] = []
    for row in rows:
        cards = BASE.parse_evidence_chain(row.get("evidence_chain_top10", ""))
        state_block = BASE.canonical_state_block(
            cards,
            max_cards=config["max_state_cards"],
            max_snippet_chars=config["max_snippet_chars"],
        )
        for model_spec in config["models"]:
            for variant in config["variants"]:
                messages = build_messages(row, state_block, variant, raw_chain_chars=config["raw_chain_chars"])
                text = prompt_text(messages)
                prompt_rows.append(
                    {
                        "model_scale": model_spec["scale"],
                        "model_size_b": model_spec["size_b"],
                        "model": model_spec["model"],
                        "variant": variant,
                        "family_id": row["scale_family_id"],
                        "family_aircraft_model": row["scale_family_aircraft"],
                        "family_size": row["scale_family_size"],
                        "family_prefix_evidence_ids": row["scale_family_prefix_evidence_ids"],
                        "question_id": row.get("question_id", ""),
                        "query_id": row.get("query_id", ""),
                        "query_type": row.get("query_type", ""),
                        "question": BASE.clean_question_text(row.get("question", "")),
                        "gold_answer": row.get("gold_answer", ""),
                        "top_evidence_ids": row.get("top_evidence_ids", ""),
                        "grounded_score": row.get("grounded_score", ""),
                        "accepted_recall_10": row.get("accepted_recall_10", ""),
                        "evidence_package_recall_10": row.get("evidence_package_recall_10", ""),
                        "hallucination_risk_proxy": row.get("hallucination_risk_proxy", ""),
                        "role_coverage_proxy": row.get("role_coverage_proxy", ""),
                        "messages_json": json.dumps(messages, ensure_ascii=False),
                        "prompt_token_count_qwen35_08b": len(tokenizer.encode(text).ids),
                        "prompt_char_count": len(text),
                    }
                )
    return prompt_rows


def filter_prompt_rows(
    prompt_rows: list[dict[str, Any]],
    *,
    model_filter: set[str] | None,
    variant_filter: set[str] | None,
) -> list[dict[str, Any]]:
    filtered = []
    for row in prompt_rows:
        if model_filter and row["model"] not in model_filter and row["model_scale"] not in model_filter:
            continue
        if variant_filter and row["variant"] not in variant_filter:
            continue
        filtered.append(row)
    return filtered


def call_openai_compatible(
    *,
    model: str,
    messages: list[dict[str, str]],
    api_key: str,
    base_url: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
) -> tuple[str, dict[str, Any], str]:
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
    choice = data["choices"][0]
    message = choice["message"]
    return message.get("content", ""), data.get("usage", {}), choice.get("finish_reason", "")


def evaluate_answer(answer: str, row: dict[str, Any], variant: str) -> dict[str, Any]:
    metrics = BASE.evaluate_answer(answer, row)
    metrics["unsupported_citation_flag"] = 1 if variant == "direct_model_only" and metrics["cited_evidence_count"] > 0 else 0
    metrics["action_verb_present_proxy"] = 1 if re.search(r"inspect|check|replace|repair|perform|执行|检查|更换|排故|维修|复核", answer or "", re.I) else 0
    return metrics


def run_generations(
    prompt_rows: list[dict[str, Any]],
    config: dict[str, Any],
    no_resume: bool,
    max_new: int | None = None,
) -> list[dict[str, Any]]:
    final_path = RESULTS / "scale_state_responses.csv"
    checkpoint_path = CHECKPOINTS / "scale_state_responses_checkpoint.csv"
    response_rows: list[dict[str, Any]] = []
    completed: set[tuple[str, str, str, str]] = set()
    if no_resume and checkpoint_path.exists():
        checkpoint_path.unlink()
    if not no_resume:
        resume_path = final_path if final_path.exists() else checkpoint_path if checkpoint_path.exists() else None
        if resume_path:
            response_rows.extend(read_csv(resume_path))
            for row in response_rows:
                completed.add((row["model"], row["variant"], row["family_id"], row["question_id"]))

    api_key = config.get("api_key_env_value", "ollama")
    base_url = config["base_url"]
    generated = 0
    for index, row in enumerate(prompt_rows, start=1):
        key = (row["model"], row["variant"], row["family_id"], row["question_id"])
        if key in completed:
            continue
        if max_new is not None and generated >= max_new:
            break
        started = time.time()
        try:
            answer, usage, finish_reason = call_openai_compatible(
                model=row["model"],
                messages=json.loads(row["messages_json"]),
                api_key=api_key,
                base_url=base_url,
                temperature=config["temperature"],
                max_tokens=config["max_tokens"],
                timeout=config["timeout_s"],
            )
            status = "ok"
            error = ""
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
            answer = ""
            usage = {}
            finish_reason = ""
            status = "error"
            error = str(exc)

        response_rows.append(
            {
                "model_scale": row["model_scale"],
                "model_size_b": row["model_size_b"],
                "model": row["model"],
                "variant": row["variant"],
                "family_id": row["family_id"],
                "family_aircraft_model": row["family_aircraft_model"],
                "family_size": row["family_size"],
                "question_id": row["question_id"],
                "query_id": row["query_id"],
                "query_type": row["query_type"],
                "prompt_token_count_qwen35_08b": row["prompt_token_count_qwen35_08b"],
                "prompt_char_count": row["prompt_char_count"],
                "status": status,
                "error": error,
                "finish_reason": finish_reason,
                "latency_s": round(time.time() - started, 4),
                "usage_prompt_tokens": usage.get("prompt_tokens", ""),
                "usage_completion_tokens": usage.get("completion_tokens", ""),
                "usage_total_tokens": usage.get("total_tokens", ""),
                "answer": answer,
                **evaluate_answer(answer, row, row["variant"]),
            }
        )
        if index % config["checkpoint_every"] == 0:
            write_csv(checkpoint_path, response_rows)
        generated += 1
    write_csv(final_path, response_rows)
    return response_rows


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: defaultdict[tuple[str, float, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["model_scale"], to_float(row["model_size_b"]), row["model"], row["variant"])].append(row)
    summary: list[dict[str, Any]] = []
    for (scale, size_b, model, variant), items in grouped.items():
        summary.append(
            {
                "model_scale": scale,
                "model_size_b": size_b,
                "model": model,
                "variant": variant,
                "n_queries": len(items),
                "ok_rate": statistics.mean(1.0 if item["status"] == "ok" else 0.0 for item in items),
                "mean_latency_s": statistics.mean(to_float(item["latency_s"]) for item in items),
                "mean_prompt_tokens_usage": statistics.mean(to_float(item["usage_prompt_tokens"]) for item in items),
                "mean_completion_tokens_usage": statistics.mean(to_float(item["usage_completion_tokens"]) for item in items),
                "mean_citation_precision_proxy": statistics.mean(to_float(item["citation_precision_proxy"]) for item in items),
                "mean_citation_coverage_proxy": statistics.mean(to_float(item["citation_coverage_proxy"]) for item in items),
                "mean_valid_cited_evidence_count": statistics.mean(to_float(item["valid_cited_evidence_count"]) for item in items),
                "mean_char_bigram_f1_to_gold_proxy": statistics.mean(to_float(item["char_bigram_f1_to_gold_proxy"]) for item in items),
                "mean_answer_evidence_grounded_score_proxy": statistics.mean(to_float(item["answer_evidence_grounded_score_proxy"]) for item in items),
                "mean_refusal_rate": statistics.mean(1.0 if str(item["refusal_detected"]).lower() == "true" else 0.0 for item in items),
                "mean_unsupported_citation_rate": statistics.mean(to_float(item["unsupported_citation_flag"]) for item in items),
                "mean_action_verb_rate": statistics.mean(to_float(item["action_verb_present_proxy"]) for item in items),
            }
        )
    return sorted(summary, key=lambda row: (row["model_size_b"], row["variant"]))


def write_delta_table(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_model: defaultdict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in summary:
        by_model[row["model"]][row["variant"]] = row
    deltas: list[dict[str, Any]] = []
    for model, variants in sorted(by_model.items(), key=lambda item: to_float(next(iter(item[1].values()))["model_size_b"])):
        direct = variants.get("direct_model_only")
        state = variants.get("state_first_state")
        raw = variants.get("question_first_chain")
        if not direct or not state:
            continue
        deltas.append(
            {
                "model": model,
                "model_size_b": direct["model_size_b"],
                "state_minus_direct_groundedness": state["mean_answer_evidence_grounded_score_proxy"] - direct["mean_answer_evidence_grounded_score_proxy"],
                "raw_minus_direct_groundedness": (raw["mean_answer_evidence_grounded_score_proxy"] - direct["mean_answer_evidence_grounded_score_proxy"]) if raw else "",
                "state_minus_direct_citation_coverage": state["mean_citation_coverage_proxy"] - direct["mean_citation_coverage_proxy"],
                "raw_minus_direct_citation_coverage": (raw["mean_citation_coverage_proxy"] - direct["mean_citation_coverage_proxy"]) if raw else "",
                "state_minus_raw_groundedness": (state["mean_answer_evidence_grounded_score_proxy"] - raw["mean_answer_evidence_grounded_score_proxy"]) if raw else "",
                "state_latency_s": state["mean_latency_s"],
                "direct_latency_s": direct["mean_latency_s"],
            }
        )
    write_csv(TABLES / "scale_state_delta_summary.csv", deltas)
    return deltas


def plot_summary(summary: list[dict[str, Any]], deltas: list[dict[str, Any]]) -> None:
    label_map = {
        "direct_model_only": "Direct only",
        "question_first_chain": "Raw chain",
        "state_first_state": "State first",
    }
    color_map = {
        "direct_model_only": "#4C78A8",
        "question_first_chain": "#F58518",
        "state_first_state": "#54A24B",
    }
    variants = ["direct_model_only", "question_first_chain", "state_first_state"]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), dpi=300)
    for variant in variants:
        rows = sorted([row for row in summary if row["variant"] == variant], key=lambda row: row["model_size_b"])
        axes[0].plot(
            [row["model_size_b"] for row in rows],
            [row["mean_answer_evidence_grounded_score_proxy"] for row in rows],
            marker="o",
            linewidth=2,
            color=color_map[variant],
            label=label_map[variant],
        )
        axes[1].plot(
            [row["model_size_b"] for row in rows],
            [row["mean_citation_coverage_proxy"] for row in rows],
            marker="o",
            linewidth=2,
            color=color_map[variant],
            label=label_map[variant],
        )
    axes[0].set_title("Model scale vs answer groundedness proxy")
    axes[0].set_xlabel("Model size (B parameters)")
    axes[0].set_ylabel("Mean groundedness proxy")
    axes[1].set_title("Model scale vs evidence citation coverage")
    axes[1].set_xlabel("Model size (B parameters)")
    axes[1].set_ylabel("Mean citation coverage proxy")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.grid(axis="y", color="#D9D9D9", alpha=0.8)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "scale_intrinsic_vs_evidence_state.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), dpi=300)
    deltas_sorted = sorted(deltas, key=lambda row: row["model_size_b"])
    axes[0].bar(
        [str(row["model_size_b"]) for row in deltas_sorted],
        [row["state_minus_direct_groundedness"] for row in deltas_sorted],
        color="#54A24B",
    )
    axes[0].axhline(0, color="#333333", linewidth=1)
    axes[0].set_title("Evidence-state gain over direct answering")
    axes[0].set_xlabel("Model size (B parameters)")
    axes[0].set_ylabel("Delta groundedness proxy")
    axes[1].scatter(
        [row["mean_latency_s"] for row in summary],
        [row["mean_answer_evidence_grounded_score_proxy"] for row in summary],
        s=[40 + 35 * to_float(row["model_size_b"]) for row in summary],
        c=[color_map[row["variant"]] for row in summary],
        alpha=0.82,
        edgecolors="#222222",
        linewidths=0.5,
    )
    for row in summary:
        axes[1].annotate(
            f"{row['model_scale']} {label_map[row['variant']]}",
            (row["mean_latency_s"], row["mean_answer_evidence_grounded_score_proxy"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
        )
    axes[1].set_title("Latency-quality tradeoff")
    axes[1].set_xlabel("Mean latency (s)")
    axes[1].set_ylabel("Mean groundedness proxy")
    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "scale_state_gain_tradeoff.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--models", default="", help="Comma-separated model names or scale labels to run.")
    parser.add_argument("--variants", default="", help="Comma-separated variant names to run.")
    parser.add_argument("--max-families-per-aircraft", type=int, default=None)
    parser.add_argument("--max-new", type=int, default=None, help="Maximum new generations to add in this invocation.")
    args = parser.parse_args()

    config = read_json(Path(args.config))
    if args.max_families_per_aircraft is not None:
        config["max_families_per_aircraft"] = args.max_families_per_aircraft
    tokenizer_path = resolve_tokenizer_json(config)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    all_rows = BASE.load_fd_rows(config["case_ids"], method_name=config["method_name"])
    selected_rows = select_families(
        all_rows,
        family_prefix_k=config["family_prefix_k"],
        min_family_size=config["min_family_size"],
        max_families_per_aircraft=config["max_families_per_aircraft"],
    )
    all_prompt_rows = make_prompt_rows(selected_rows, config, tokenizer)
    model_filter = {item.strip() for item in args.models.split(",") if item.strip()} or None
    variant_filter = {item.strip() for item in args.variants.split(",") if item.strip()} or None
    prompt_rows = filter_prompt_rows(all_prompt_rows, model_filter=model_filter, variant_filter=variant_filter)
    write_csv(RESULTS / "scale_state_prompt_manifest.csv", prompt_rows)

    if not args.run:
        manifest = {
            "status": "prepared_not_run",
            "selected_queries": len(selected_rows),
            "full_design_prompt_rows": len(all_prompt_rows),
            "prompt_rows": len(prompt_rows),
            "models": config["models"],
            "variants": config["variants"],
            "model_filter": sorted(model_filter) if model_filter else [],
            "variant_filter": sorted(variant_filter) if variant_filter else [],
            "tokenizer_json": str(tokenizer_path),
        }
        (REPORTS / "model_scale_state_v1_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return

    response_rows = run_generations(prompt_rows, config, no_resume=args.no_resume, max_new=args.max_new)
    summary = summarize(response_rows)
    write_csv(TABLES / "scale_state_variant_summary.csv", summary)
    deltas = write_delta_table(summary)
    plot_summary(summary, deltas)
    claim_map = [
        {
            "claim_id": "S1",
            "claim": "Larger local models can be tested for intrinsic aviation-maintenance answering without supplied evidence.",
            "support_status": "Supported as a controlled local scale-probe",
            "result_file": "04_results/model_scale_state_v1/scale_state_responses.csv",
            "caveat": "Direct answering is evaluated only on source-derived diagnostic queries and answer proxies.",
        },
        {
            "claim_id": "S2",
            "claim": "Explicit organized evidence state improves groundedness and citation coverage relative to direct answering across the tested scale ladder.",
            "support_status": "Evaluate from scale_state_delta_summary.csv",
            "result_file": "06_tables/model_scale_state_v1/scale_state_delta_summary.csv",
            "caveat": "This is not proof of true internal KV behavior or natural-query generalization.",
        },
        {
            "claim_id": "S3",
            "claim": "Model scale alone replaces evidence-state construction.",
            "support_status": "Unsupported unless direct-only performance closes the evidence-state gap",
            "result_file": "06_tables/model_scale_state_v1/scale_state_variant_summary.csv",
            "caveat": "Large models may improve verbalization but cannot cite unseen project evidence in direct-only mode.",
        },
    ]
    write_csv(RESULTS / "claim_result_map.csv", claim_map)
    manifest = {
        "experiment_id": "model_scale_state_v1_20260812",
        "selected_queries": len(selected_rows),
        "full_design_prompt_rows": len(all_prompt_rows),
        "last_invocation_prompt_rows": len(prompt_rows),
        "prompt_rows": len(prompt_rows),
        "response_rows": len(response_rows),
        "models": config["models"],
        "variants": config["variants"],
        "tokenizer_json": str(tokenizer_path),
        "claim_boundary": (
            "This is a local Qwen scale-probe over source-derived diagnostic queries. "
            "It tests answer behavior with and without explicit evidence state, not internal model knowledge certification, "
            "true KV-cache hits, or natural maintenance-query generalization."
        ),
    }
    (REPORTS / "model_scale_state_v1_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
