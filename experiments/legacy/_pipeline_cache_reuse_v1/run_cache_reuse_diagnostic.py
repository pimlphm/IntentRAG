"""Run a local-Qwen cache-eligible prefix-reuse diagnostic on FD-GRAG-RC families.

This audit studies prompt/state organization on real project evidence chains and
local Qwen generations. It measures cache-eligible shared-prefix structure
under a local Qwen tokenizer proxy plus answer-side groundedness proxies from a
local Ollama model.

It does not inspect internal layerwise KV tensors, GPU kernel cache hits,
serving-engine prefix cache tables, or real deployment throughput under
multi-request batching.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
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
DEFAULT_CONFIG = ROOT / "configs" / "cache_reuse_v1" / "cache_reuse_v1.json"
RESULTS = ROOT / "04_results" / "cache_reuse_v1"
FIGURES = ROOT / "05_figures" / "cache_reuse_v1"
TABLES = ROOT / "06_tables" / "cache_reuse_v1"
REPORTS = ROOT / "07_reports" / "cache_reuse_v1"
LOGS = ROOT / "08_logs" / "cache_reuse_v1"
CHECKPOINTS = RESULTS / "_checkpoints"
for directory in (RESULTS, FIGURES, TABLES, REPORTS, LOGS, CHECKPOINTS):
    directory.mkdir(parents=True, exist_ok=True)

SYSTEM_PROMPT = """You are an aviation maintenance evidence-grounded QA assistant.
LLM generation is downstream of evidence construction.
Use only the provided evidence state or evidence package.
If the available evidence does not support a requested action, threshold, or authority claim, say that the evidence is insufficient.
Do not claim safety certification or real maintenance correctness."""

ROLE_PRIORITY = [
    "maintenance_action",
    "cause",
    "excluded_cause",
    "threshold_or_condition",
    "warning_or_limitation",
    "authority",
    "component",
    "symptom",
    "supporting_context",
]

TOKENIZER_SEARCH_ROOTS = [
    Path.home() / ".cache" / "huggingface" / "hub",
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows and not fieldnames:
        raise ValueError(f"Cannot write empty CSV without fieldnames: {path}")
    columns = fieldnames or list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def split_semicolon(value: Any) -> list[str]:
    return [part.strip() for part in str(value or "").split(";") if part.strip()]


def shorten(text: str, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    return text if len(text) <= max_chars else text[: max_chars - 3] + "..."


def clean_question_text(text: str) -> str:
    text = str(text or "")
    text = re.split(r"\bEvidence cue:\b", text, maxsplit=1, flags=re.I)[0]
    return re.sub(r"\s+", " ", text).strip()


def resolve_tokenizer_path(config: dict[str, Any]) -> Path:
    configured = Path(str(config.get("tokenizer_json", "") or ""))
    if configured and configured.exists():
        return configured

    hint = str(config.get("tokenizer_hint", "Qwen3.5-0.8B"))
    matches: list[Path] = []
    for root in TOKENIZER_SEARCH_ROOTS:
        if not root.exists():
            continue
        try:
            for candidate in root.rglob("tokenizer.json"):
                if hint.lower() in str(candidate).lower():
                    matches.append(candidate)
        except OSError:
            continue
    if matches:
        return sorted(matches)[0]
    raise FileNotFoundError(f"Unable to locate tokenizer.json with hint '{hint}'")


def role_bucket(card: dict[str, str]) -> str:
    role = f"{card.get('role', '')} {card.get('snippet', '')}".lower()
    if any(x in role for x in ["warning", "caution", "limitation", "safety", "note"]):
        return "warning_or_limitation"
    if any(x in role for x in ["threshold", "limit", "torque", "condition", "psi", "rpm", "temperature"]):
        return "threshold_or_condition"
    if any(x in role for x in ["authority", "manual", "section", "page", "reference", "approved"]):
        return "authority"
    if any(x in role for x in ["action", "inspection", "replace", "repair", "maintenance", "procedure", "check"]):
        return "maintenance_action"
    if any(x in role for x in ["excluded", "rule out", "not due", "negative"]):
        return "excluded_cause"
    if any(x in role for x in ["cause", "failure", "fault", "reason"]):
        return "cause"
    if any(x in role for x in ["symptom", "finding", "phenomenon", "message"]):
        return "symptom"
    if card.get("component", "").strip() and card.get("component", "").strip() != "no_component":
        return "component"
    return "supporting_context"


def parse_evidence_chain(chain: str) -> list[dict[str, str]]:
    pattern = re.compile(
        r"(?:^|\n)\s*(\d+)\.\s+"
        r"(EVID[A-Z0-9_]+)\s+\|\s+"
        r"page=([^|]*)\|\s+role=([^|]*)\|\s+component=([^|]*)\|\s+section=([^|]*)\|\s*"
        r"(.*?)(?=\n\s*\d+\.\s+EVID[A-Z0-9_]+\s+\||\Z)",
        flags=re.S,
    )
    cards: list[dict[str, str]] = []
    for match in pattern.finditer(str(chain or "")):
        cards.append(
            {
                "rank": match.group(1).strip(),
                "evidence_id": match.group(2).strip(),
                "page": match.group(3).strip(),
                "role": match.group(4).strip(),
                "component": match.group(5).strip(),
                "section": match.group(6).strip(),
                "snippet": re.sub(r"\s+", " ", match.group(7)).strip(),
            }
        )
    if cards:
        return cards

    ids = re.findall(r"EVID[A-Z0-9_]+", str(chain or ""))
    return [
        {
            "rank": str(i + 1),
            "evidence_id": evidence_id,
            "page": "",
            "role": "",
            "component": "",
            "section": "",
            "snippet": "",
        }
        for i, evidence_id in enumerate(ids[:10])
    ]


def load_fd_rows(case_ids: list[str], method_name: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for case_id in case_ids:
        path = ROOT / "Cases" / case_id / "7 测试" / "sota_evidence_chain_tables" / f"{case_id}_complete_sota_evidence_chains.csv"
        for row in read_csv(path):
            if row.get("method") == method_name:
                row["case_id"] = case_id
                rows.append(row)
    return rows


def family_key(row: dict[str, str], family_prefix_k: int) -> tuple[str, tuple[str, ...]]:
    ids = tuple(split_semicolon(row.get("top_evidence_ids", ""))[:family_prefix_k])
    return (row.get("aircraft_model", ""), ids)


def build_families(rows: list[dict[str, str]], family_prefix_k: int, min_family_size: int) -> dict[tuple[str, tuple[str, ...]], list[dict[str, str]]]:
    grouped: defaultdict[tuple[str, tuple[str, ...]], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = family_key(row, family_prefix_k)
        if key[1]:
            grouped[key].append(row)
    return {key: items for key, items in grouped.items() if len(items) >= min_family_size}


def canonical_state_block(cards: list[dict[str, str]], max_cards: int, max_snippet_chars: int) -> str:
    grouped: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for card in cards:
        grouped[role_bucket(card)].append(card)
    selected: list[dict[str, str]] = []
    seen: set[str] = set()
    for bucket in ROLE_PRIORITY:
        for card in grouped.get(bucket, []):
            evidence_id = card.get("evidence_id", "")
            if evidence_id and evidence_id not in seen:
                selected.append({**card, "bucket": bucket})
                seen.add(evidence_id)
            if len(selected) >= max_cards:
                break
        if len(selected) >= max_cards:
            break

    lines = [
        "EVIDENCE STATE RECORD",
        "The following evidence cards are the currently assembled bounded evidence state.",
        "Use evidence IDs exactly as written.",
        "",
    ]
    for bucket in ROLE_PRIORITY:
        bucket_cards = [card for card in selected if card["bucket"] == bucket]
        if not bucket_cards:
            continue
        lines.append(f"[{bucket}]")
        for card in bucket_cards:
            lines.append(
                "- "
                + " | ".join(
                    [
                        f"id={card.get('evidence_id', '')}",
                        f"page={card.get('page', '')}",
                        f"role={card.get('role', '')}",
                        f"component={card.get('component', '')}",
                        f"section={shorten(card.get('section', ''), 80)}",
                        f"snippet={shorten(card.get('snippet', ''), max_snippet_chars)}",
                    ]
                )
            )
        lines.append("")
    return "\n".join(lines).strip()


def raw_chain_block(chain: str, max_chars: int) -> str:
    return shorten(str(chain or "").replace("\r\n", "\n"), max_chars)


def answer_rules() -> str:
    return (
        "Answer format requirements:\n"
        "- Use at most 4 short bullet points.\n"
        "- Include one bullet for finding, one for action or deferral, one for authority/condition boundary, and one for evidence IDs.\n"
        "- Every maintenance claim bullet must end with at least one evidence ID in parentheses.\n"
        "- Keep the full answer under about 120 words.\n"
        "- If the evidence is insufficient, explicitly say that the evidence is insufficient and still list the checked evidence IDs."
    )


def build_prompt_variant(
    row: dict[str, str],
    state_block: str,
    variant: str,
    raw_chain_chars: int,
) -> list[dict[str, str]]:
    aircraft = row.get("aircraft_model", "")
    question = clean_question_text(row.get("question", ""))
    raw_chain = raw_chain_block(row.get("evidence_chain_top10", ""), raw_chain_chars)
    if variant == "question_first_chain":
        user = (
            f"Aircraft: {aircraft}\n"
            f"Question: {question}\n\n"
            f"Retrieved evidence package:\n{raw_chain}\n\n"
            f"{answer_rules()}"
        )
    elif variant == "question_first_state":
        user = (
            f"Aircraft: {aircraft}\n"
            f"Question: {question}\n\n"
            f"{state_block}\n\n"
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


def prompt_text(messages: list[dict[str, str]]) -> str:
    return "\n\n".join(f"{message['role'].upper()}:\n{message['content']}" for message in messages)


def lcp_token_count(sequences: list[list[int]]) -> int:
    if not sequences:
        return 0
    min_len = min(len(sequence) for sequence in sequences)
    count = 0
    for i in range(min_len):
        token = sequences[0][i]
        if all(sequence[i] == token for sequence in sequences[1:]):
            count += 1
        else:
            break
    return count


def lcp_pair_count(left: list[int], right: list[int]) -> int:
    limit = min(len(left), len(right))
    count = 0
    for i in range(limit):
        if left[i] == right[i]:
            count += 1
        else:
            break
    return count


def char_bigram_f1(pred: str, gold: str) -> float:
    pred = re.sub(r"\s+", "", pred or "")
    gold = re.sub(r"\s+", "", gold or "")
    if not pred or not gold:
        return 0.0
    pred_bigrams = Counter(pred[i : i + 2] for i in range(max(1, len(pred) - 1)))
    gold_bigrams = Counter(gold[i : i + 2] for i in range(max(1, len(gold) - 1)))
    overlap = sum((pred_bigrams & gold_bigrams).values())
    precision = overlap / max(1, sum(pred_bigrams.values()))
    recall = overlap / max(1, sum(gold_bigrams.values()))
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        return default if value in ("", None) else float(value)
    except (TypeError, ValueError):
        return default


def bounded(value: float) -> float:
    return max(0.0, min(1.0, value))


def evaluate_answer(answer: str, row: dict[str, str]) -> dict[str, Any]:
    available_ids = set(re.findall(r"EVID[A-Z0-9_]*", str(row.get("top_evidence_ids", ""))))
    cited_ids = set(re.findall(r"EVID[A-Z0-9_]*", answer or ""))
    valid_ids = cited_ids & available_ids
    refusal_terms = [
        "insufficient",
        "not enough evidence",
        "manual review",
        "cannot determine",
        "证据不足",
        "无法确定",
        "人工复核",
    ]
    char_f1 = round(char_bigram_f1(answer or "", row.get("gold_answer", "")), 4)
    citation_precision = len(valid_ids) / max(1, len(cited_ids)) if cited_ids else 0.0
    citation_coverage = 1.0 if valid_ids else 0.0
    source_grounded = to_float(row.get("grounded_score", 0.0))
    source_package = to_float(row.get("evidence_package_recall_10", 0.0))
    source_accepted = to_float(row.get("accepted_recall_10", 0.0))
    source_risk = to_float(row.get("hallucination_risk_proxy", 0.0))
    score = bounded(
        0.22 * char_f1
        + 0.18 * citation_precision
        + 0.16 * citation_coverage
        + 0.18 * source_grounded
        + 0.18 * source_package
        + 0.10 * source_accepted
        - 0.12 * source_risk
    )
    return {
        "answer_chars": len(answer or ""),
        "refusal_detected": any(term.lower() in (answer or "").lower() for term in refusal_terms),
        "cited_evidence_count": len(cited_ids),
        "valid_cited_evidence_count": len(valid_ids),
        "citation_precision_proxy": citation_precision,
        "citation_coverage_proxy": citation_coverage,
        "char_bigram_f1_to_gold_proxy": char_f1,
        "answer_evidence_grounded_score_proxy": round(score, 4),
    }


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
    return data["choices"][0]["message"]["content"], data.get("usage", {})


def make_prompt_rows(
    rows: list[dict[str, str]],
    variants: list[str],
    tokenizer: Tokenizer,
    family_prefix_k: int,
    min_family_size: int,
    max_state_cards: int,
    max_snippet_chars: int,
    raw_chain_chars: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    families = build_families(rows, family_prefix_k=family_prefix_k, min_family_size=min_family_size)
    prompt_rows: list[dict[str, Any]] = []
    family_rows: list[dict[str, Any]] = []

    for family_index, (key, family_items) in enumerate(sorted(families.items(), key=lambda item: (item[0][0], item[1][0]["query_id"]))):
        aircraft, evidence_prefix = key
        for variant in variants:
            sequence_rows: list[dict[str, Any]] = []
            token_sequences: list[list[int]] = []
            for item in sorted(family_items, key=lambda row: row.get("query_id", "")):
                cards = parse_evidence_chain(item.get("evidence_chain_top10", ""))
                state_block = canonical_state_block(cards, max_cards=max_state_cards, max_snippet_chars=max_snippet_chars)
                messages = build_prompt_variant(item, state_block=state_block, variant=variant, raw_chain_chars=raw_chain_chars)
                text = prompt_text(messages)
                token_ids = tokenizer.encode(text).ids
                record = {
                    "family_id": f"F{family_index:03d}",
                    "family_aircraft_model": aircraft,
                    "family_size": len(family_items),
                    "family_prefix_evidence_ids": ";".join(evidence_prefix),
                    "variant": variant,
                    "question_id": item.get("question_id", ""),
                    "query_id": item.get("query_id", ""),
                    "query_type": item.get("query_type", ""),
                    "question": clean_question_text(item.get("question", "")),
                    "gold_answer": item.get("gold_answer", ""),
                    "top_evidence_ids": item.get("top_evidence_ids", ""),
                    "evidence_chain_top10": item.get("evidence_chain_top10", ""),
                    "grounded_score": item.get("grounded_score", ""),
                    "accepted_recall_10": item.get("accepted_recall_10", ""),
                    "evidence_package_recall_10": item.get("evidence_package_recall_10", ""),
                    "hallucination_risk_proxy": item.get("hallucination_risk_proxy", ""),
                    "role_coverage_proxy": item.get("role_coverage_proxy", ""),
                    "messages_json": json.dumps(messages, ensure_ascii=False),
                    "prompt_text": text,
                    "prompt_token_count_qwen35_08b": len(token_ids),
                    "prompt_char_count": len(text),
                }
                sequence_rows.append(record)
                token_sequences.append(token_ids)
                prompt_rows.append(record)

            family_prefix_tokens = lcp_token_count(token_sequences)
            mean_prompt_tokens = statistics.mean(record["prompt_token_count_qwen35_08b"] for record in sequence_rows)
            adjacent_prefix_tokens: list[int] = []
            for left, right in zip(token_sequences, token_sequences[1:]):
                adjacent_prefix_tokens.append(lcp_pair_count(left, right))
            family_rows.append(
                {
                    "family_id": sequence_rows[0]["family_id"],
                    "family_aircraft_model": aircraft,
                    "family_size": len(family_items),
                    "family_prefix_evidence_ids": ";".join(evidence_prefix),
                    "variant": variant,
                    "family_shared_prefix_tokens_qwen35_08b": family_prefix_tokens,
                    "mean_prompt_tokens_qwen35_08b": mean_prompt_tokens,
                    "family_shared_prefix_ratio_qwen35_08b": family_prefix_tokens / mean_prompt_tokens if mean_prompt_tokens else math.nan,
                    "mean_adjacent_prefix_tokens_qwen35_08b": statistics.mean(adjacent_prefix_tokens) if adjacent_prefix_tokens else family_prefix_tokens,
                    "reusable_prefix_tokens_after_first_qwen35_08b": family_prefix_tokens * max(0, len(family_items) - 1),
                }
            )

    return prompt_rows, family_rows


def summarize_generation(response_rows: list[dict[str, Any]], family_metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    family_by_variant: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in family_metrics:
        family_by_variant[row["variant"]].append(row)

    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in response_rows:
        grouped[row["variant"]].append(row)

    summary_rows: list[dict[str, Any]] = []
    for variant, items in grouped.items():
        family_items = family_by_variant.get(variant, [])
        summary_rows.append(
            {
                "variant": variant,
                "n_queries": len(items),
                "n_families": len(family_items),
                "mean_family_size": statistics.mean(item["family_size"] for item in family_items) if family_items else math.nan,
                "mean_prompt_tokens_qwen35_08b": statistics.mean(to_float(item.get("prompt_token_count_qwen35_08b")) for item in items),
                "mean_family_shared_prefix_tokens_qwen35_08b": statistics.mean(to_float(item["family_shared_prefix_tokens_qwen35_08b"]) for item in family_items) if family_items else math.nan,
                "mean_family_shared_prefix_ratio_qwen35_08b": statistics.mean(to_float(item["family_shared_prefix_ratio_qwen35_08b"]) for item in family_items) if family_items else math.nan,
                "mean_adjacent_prefix_tokens_qwen35_08b": statistics.mean(to_float(item["mean_adjacent_prefix_tokens_qwen35_08b"]) for item in family_items) if family_items else math.nan,
                "mean_reusable_prefix_tokens_after_first_qwen35_08b": statistics.mean(to_float(item["reusable_prefix_tokens_after_first_qwen35_08b"]) for item in family_items) if family_items else math.nan,
                "ok_rate": statistics.mean(1.0 if item["status"] == "ok" else 0.0 for item in items),
                "mean_latency_s": statistics.mean(to_float(item.get("latency_s")) for item in items),
                "mean_prompt_tokens_usage": statistics.mean(to_float(item.get("usage_prompt_tokens")) for item in items),
                "mean_completion_tokens_usage": statistics.mean(to_float(item.get("usage_completion_tokens")) for item in items),
                "mean_citation_precision_proxy": statistics.mean(to_float(item.get("citation_precision_proxy")) for item in items),
                "mean_citation_coverage_proxy": statistics.mean(to_float(item.get("citation_coverage_proxy")) for item in items),
                "mean_char_bigram_f1_to_gold_proxy": statistics.mean(to_float(item.get("char_bigram_f1_to_gold_proxy")) for item in items),
                "mean_answer_evidence_grounded_score_proxy": statistics.mean(to_float(item.get("answer_evidence_grounded_score_proxy")) for item in items),
                "mean_refusal_rate": statistics.mean(1.0 if str(item.get("refusal_detected")).lower() == "true" else 0.0 for item in items),
            }
        )
    return sorted(summary_rows, key=lambda row: row["variant"])


def figure_prefix(summary_rows: list[dict[str, Any]], family_rows: list[dict[str, Any]], path: Path) -> None:
    order = [row["variant"] for row in summary_rows]
    label_map = {
        "question_first_chain": "Question → raw chain",
        "question_first_state": "Question → state",
        "state_first_state": "State → question",
    }
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2), dpi=300)

    box_data = []
    for variant in order:
        values = [to_float(row["family_shared_prefix_ratio_qwen35_08b"]) for row in family_rows if row["variant"] == variant]
        box_data.append(values)
    axes[0].boxplot(box_data, labels=[label_map.get(variant, variant) for variant in order], patch_artist=True)
    for patch, color in zip(axes[0].artists, ["#A0CBE8", "#72B7B2", "#E45756"]):
        patch.set_facecolor(color)
    axes[0].set_ylabel("Family shared-prefix ratio")
    axes[0].set_title("Cache-eligible shared prefix across query families")
    axes[0].tick_params(axis="x", rotation=15)

    prefix_means = [row["mean_family_shared_prefix_tokens_qwen35_08b"] for row in summary_rows]
    quality_means = [row["mean_answer_evidence_grounded_score_proxy"] for row in summary_rows]
    axes[1].scatter(prefix_means, quality_means, s=120, c=["#4C78A8", "#72B7B2", "#E45756"])
    for row in summary_rows:
        axes[1].annotate(
            label_map.get(row["variant"], row["variant"]),
            (row["mean_family_shared_prefix_tokens_qwen35_08b"], row["mean_answer_evidence_grounded_score_proxy"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    axes[1].set_xlabel("Mean family shared-prefix tokens")
    axes[1].set_ylabel("Mean answer groundedness proxy")
    axes[1].set_title("Reuse vs answer quality proxy")

    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def figure_latency(summary_rows: list[dict[str, Any]], path: Path) -> None:
    order = [row["variant"] for row in summary_rows]
    label_map = {
        "question_first_chain": "Question → raw chain",
        "question_first_state": "Question → state",
        "state_first_state": "State → question",
    }
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.0), dpi=300)
    axes[0].bar([label_map.get(v, v) for v in order], [row["mean_latency_s"] for row in summary_rows], color=["#4C78A8", "#72B7B2", "#E45756"])
    axes[0].set_ylabel("Mean latency (s)")
    axes[0].set_title("Local Qwen latency by prompt organization")
    axes[0].tick_params(axis="x", rotation=15)

    axes[1].bar([label_map.get(v, v) for v in order], [row["mean_prompt_tokens_usage"] for row in summary_rows], color=["#4C78A8", "#72B7B2", "#E45756"])
    axes[1].set_ylabel("Mean prompt tokens (model usage)")
    axes[1].set_title("Prompt token usage by prompt organization")
    axes[1].tick_params(axis="x", rotation=15)

    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--limit-per-variant", type=int, default=0)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    config = read_json(Path(args.config))
    tokenizer_path = resolve_tokenizer_path(config)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    variants = config["variants"]
    rows = load_fd_rows(config["case_ids"], method_name=config["method_name"])
    prompt_rows, family_rows = make_prompt_rows(
        rows=rows,
        variants=variants,
        tokenizer=tokenizer,
        family_prefix_k=config["family_prefix_k"],
        min_family_size=config["min_family_size"],
        max_state_cards=config["max_state_cards"],
        max_snippet_chars=config["max_snippet_chars"],
        raw_chain_chars=config["raw_chain_chars"],
    )

    prompt_manifest = RESULTS / "prompt_family_manifest.csv"
    family_metrics_path = RESULTS / "family_prefix_metrics.csv"
    write_csv(prompt_manifest, prompt_rows)
    write_csv(family_metrics_path, family_rows)

    if not args.run:
        manifest = {
            "status": "prepared_not_run",
            "prompt_rows": len(prompt_rows),
            "family_rows": len(family_rows),
            "tokenizer_json": str(tokenizer_path),
            "claim_boundary": (
                "Prepared prompt/state organization audit. Run with --run to call the local Ollama Qwen model. "
                "Shared-prefix metrics are tokenizer-level cache-eligible reuse diagnostics, not direct KV-cache inspection."
            ),
        }
        (REPORTS / "cache_reuse_v1_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return

    api_key = config.get("api_key_env_value") or os.environ.get(config["api_key_env"], "ollama")
    base_url = config["base_url"]
    model = config["model"]
    pending_rows = prompt_rows[:]
    response_rows: list[dict[str, Any]] = []
    completed_keys: set[tuple[str, str, str]] = set()
    checkpoint_path = CHECKPOINTS / "responses_checkpoint.csv"
    final_path = RESULTS / "generation_responses.csv"

    if not args.no_resume:
        resume_path = final_path if final_path.exists() else checkpoint_path if checkpoint_path.exists() else None
        if resume_path:
            old_rows = read_csv(resume_path)
            response_rows.extend(old_rows)
            for row in old_rows:
                completed_keys.add((row["variant"], row["family_id"], row["question_id"]))

    if args.no_resume and checkpoint_path.exists():
        checkpoint_path.unlink()

    if args.limit_per_variant > 0:
        limited: list[dict[str, Any]] = []
        counts: defaultdict[str, int] = defaultdict(int)
        for row in pending_rows:
            if counts[row["variant"]] < args.limit_per_variant:
                limited.append(row)
                counts[row["variant"]] += 1
        pending_rows = limited

    for index, row in enumerate(pending_rows, start=1):
        key = (row["variant"], row["family_id"], row["question_id"])
        if key in completed_keys:
            continue
        messages = json.loads(row["messages_json"])
        started = time.time()
        try:
            answer, usage = call_openai_compatible(
                model=model,
                messages=messages,
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
            status = "error"
            error = str(exc)

        eval_metrics = evaluate_answer(answer, row)
        response_rows.append(
            {
                "variant": row["variant"],
                "family_id": row["family_id"],
                "family_aircraft_model": row["family_aircraft_model"],
                "family_size": row["family_size"],
                "family_prefix_evidence_ids": row["family_prefix_evidence_ids"],
                "question_id": row["question_id"],
                "query_id": row["query_id"],
                "query_type": row["query_type"],
                "prompt_token_count_qwen35_08b": row["prompt_token_count_qwen35_08b"],
                "prompt_char_count": row["prompt_char_count"],
                "status": status,
                "error": error,
                "latency_s": round(time.time() - started, 4),
                "usage_prompt_tokens": usage.get("prompt_tokens", ""),
                "usage_completion_tokens": usage.get("completion_tokens", ""),
                "usage_total_tokens": usage.get("total_tokens", ""),
                "answer": answer,
                **eval_metrics,
            }
        )

        if index % config["checkpoint_every"] == 0:
            write_csv(checkpoint_path, response_rows)

    write_csv(final_path, response_rows)
    summary_rows = summarize_generation(response_rows, family_rows)
    summary_path = TABLES / "variant_summary.csv"
    write_csv(summary_path, summary_rows)
    figure_prefix(summary_rows, family_rows, FIGURES / "cache_prefix_quality_tradeoff.png")
    figure_latency(summary_rows, FIGURES / "cache_latency_token_usage.png")

    manifest = {
        "experiment_id": "cache_reuse_v1_20260812",
        "model": model,
        "tokenizer_json": str(tokenizer_path),
        "prompt_rows": len(prompt_rows),
        "response_rows": len(response_rows),
        "family_count": len({row["family_id"] for row in family_rows}),
        "claim_boundary": (
            "This diagnostic measures cache-eligible shared-prefix structure under a local Qwen tokenizer and "
            "answer-side proxies from local Ollama generations. It does not inspect internal KV cache hits, "
            "server-side batching behavior, or natural user-query generalization."
        ),
        "summary_path": str(summary_path.relative_to(ROOT)),
    }
    (REPORTS / "cache_reuse_v1_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
