from __future__ import annotations

import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


THIS = Path(__file__).resolve()
WORK_ROOT = THIS.parents[1]
PROJECT_ROOT = THIS.parents[3]

INPUT_ACCESS = PROJECT_ROOT / "04_results" / "aesr_v2" / "cue_free_access_query.csv"
INPUT_GOLD = PROJECT_ROOT / "04_results" / "v21" / "v21_gold_sets.csv"
INPUT_EVIDENCE = PROJECT_ROOT / "01_processed_corpus" / "v5" / "evidence_objects.parquet"

OUT_RESULTS = WORK_ROOT / "results" / "long_short_reuse"
OUT_TABLES = WORK_ROOT / "tables" / "long_short_reuse"
OUT_REPORTS = WORK_ROOT / "reports"
OUT_LOGS = WORK_ROOT / "logs"

SEEDS = [
    2026081600,
    2026081601,
    2026081602,
    2026081603,
    2026081604,
    2026081605,
    2026081606,
    2026081607,
    2026081608,
    2026081609,
    2026081610,
    2026081611,
    2026081612,
    2026081613,
    2026081614,
    2026081615,
    2026081616,
    2026081617,
    2026081618,
    2026081619,
]

POLICIES = [
    "fresh_only",
    "naive_semantic_reuse",
    "fixed_50_50",
    "entropy_adaptive",
    "admissibility_signature_gated_entropy",
]

FORBIDDEN_SELECTION_COLUMNS = {
    "track",
    "query_type",
    "required_slots",
    "required_package_slots",
    "gold_evidence_id",
    "gold_answer",
    "gold_answer_brief",
    "gold_page",
    "gold_page_id",
    "accepted_gold_evidence_ids",
    "accepted_gold_page_ids",
    "accepted_gold_section_ids",
    "exact_gold_evidence_ids",
    "offline_exact_recall_10",
    "offline_accepted_recall_10",
    "offline_page_recall_10",
    "offline_section_recall_10",
    "offline_locked_slot_coverage_10",
    "offline_construction_composite",
}

SELECTION_INPUT_COLUMNS = {
    "question_id",
    "aircraft_model",
    "query_text",
    "base_question",
    "top_evidence_ids",
}

ROLE_LEXICON = {
    "action": [
        "action",
        "procedure",
        "maintenance",
        "recommended",
        "should",
        "处理",
        "处置",
        "维修",
        "动作",
        "措施",
        "检查",
        "更换",
        "执行",
    ],
    "condition": [
        "condition",
        "when",
        "if",
        "during",
        "under",
        "现象",
        "条件",
        "情况",
        "当",
        "如果",
        "出现",
        "发生",
    ],
    "authority": [
        "manual",
        "reference",
        "section",
        "page",
        "authority",
        "approved",
        "shall",
        "must",
        "手册",
        "章节",
        "页",
        "依据",
        "规定",
        "要求",
    ],
    "warning": [
        "warning",
        "caution",
        "risk",
        "hazard",
        "limit",
        "limitation",
        "警告",
        "注意",
        "风险",
        "限制",
        "不得",
    ],
    "cause": [
        "cause",
        "why",
        "reason",
        "failure",
        "故障",
        "原因",
        "导致",
        "成因",
        "失效",
    ],
}


def ensure_dirs() -> None:
    for path in [OUT_RESULTS, OUT_TABLES, OUT_REPORTS, OUT_LOGS, WORK_ROOT / "scripts"]:
        path.mkdir(parents=True, exist_ok=True)


def parse_ids(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    return [part.strip() for part in re.split(r"[;,]", text) if part.strip()]


def clean_text(value: object) -> str:
    text = str(value or "").lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def char_ngrams(value: object, n: int = 4, max_chars: int = 800) -> set[str]:
    text = clean_text(value)
    text = re.sub(r"[\s\W_]+", "", text, flags=re.UNICODE)
    if len(text) > max_chars:
        text = text[:max_chars]
    if not text:
        return set()
    if len(text) <= n:
        return {text}
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


def normalized_entropy(values: Iterable[float]) -> float:
    arr = np.asarray([max(0.0, float(v)) for v in values], dtype=float)
    total = float(arr.sum())
    if total <= 0 or len(arr) <= 1:
        return 1.0
    p = arr / total
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum() / math.log2(len(arr)))


def demand_flags(question: str) -> dict[str, bool]:
    q = clean_text(question)
    flags = {}
    for role, terms in ROLE_LEXICON.items():
        flags[f"needs_{role}"] = any(term.lower() in q for term in terms)
    if flags["needs_action"]:
        flags["needs_authority"] = True
        flags["needs_condition"] = True
    return flags


def evidence_bool(row: pd.Series, field: str) -> bool:
    value = row.get(field, "")
    if isinstance(value, str):
        return bool(value.strip())
    return bool(value)


def evidence_tokens(row: pd.Series) -> str:
    return " ".join(
        [
            str(row.get("text", "")),
            str(row.get("role", "")),
            str(row.get("role_group", "")),
            str(row.get("component_family", "")),
            str(row.get("components", "")),
            str(row.get("condition_terms", "")),
            str(row.get("authority_terms", "")),
            str(row.get("warning_terms", "")),
            str(row.get("file_name", "")),
            str(row.get("section_title", "")),
        ]
    )


@dataclass
class CacheEntry:
    evidence_id: str
    source_question_id: str
    source_aircraft_model: str
    source_query_text: str
    source_query_grams: set[str]
    rank: int
    insertion_step: int
    package_signature: dict[str, object]


def package_signature(evidence_ids: list[str], evidence: pd.DataFrame) -> dict[str, object]:
    rows = evidence.reindex([eid for eid in evidence_ids if eid in evidence.index]).fillna("")
    if rows.empty:
        return {
            "aircraft_model": "",
            "authority_present": False,
            "condition_present": False,
            "warning_present": False,
            "component_families": [],
            "role_groups": [],
            "source_documents": [],
        }
    return {
        "aircraft_model": str(rows["aircraft_model"].mode().iloc[0]) if "aircraft_model" in rows else "",
        "authority_present": bool(rows["authority_terms"].astype(str).str.len().gt(0).any()),
        "condition_present": bool(rows["condition_terms"].astype(str).str.len().gt(0).any()),
        "warning_present": bool(rows["warning_terms"].astype(str).str.len().gt(0).any()),
        "component_families": sorted(set(rows["component_family"].astype(str)) - {""}),
        "role_groups": sorted(set(rows["role_group"].astype(str)) - {""}),
        "source_documents": sorted(set(rows["document_id"].astype(str)) - {""}),
    }


def cache_gate(
    current_aircraft: str,
    current_flags: dict[str, bool],
    current_query_grams: set[str],
    entry: CacheEntry,
    ev_flags: dict[str, object],
    ev_grams: set[str],
) -> bool:
    sig = entry.package_signature
    if str(sig.get("aircraft_model", "")) != current_aircraft:
        return False
    if current_flags.get("needs_authority", False) and not (
        bool(sig.get("authority_present", False)) or bool(ev_flags.get("authority_present", False))
    ):
        return False
    if current_flags.get("needs_condition", False) and not (
        bool(sig.get("condition_present", False)) or bool(ev_flags.get("condition_present", False))
    ):
        return False
    if current_flags.get("needs_warning", False) and not (
        bool(sig.get("warning_present", False)) or bool(ev_flags.get("warning_present", False))
    ):
        return False
    sim = jaccard(current_query_grams, ev_grams)
    return sim >= 0.015


def cache_candidates(
    cache: list[CacheEntry],
    query_text: str,
    aircraft_model: str,
    evidence: pd.DataFrame,
    evidence_grams: dict[str, set[str]],
    evidence_flags: dict[str, dict[str, object]],
    step: int,
    gated: bool,
) -> list[tuple[str, float, CacheEntry]]:
    qgrams = char_ngrams(query_text)
    flags = demand_flags(query_text)
    best: dict[str, tuple[float, CacheEntry]] = {}
    for entry in cache:
        if entry.source_aircraft_model != aircraft_model:
            continue
        if entry.evidence_id not in evidence_grams:
            continue
        ev_gram = evidence_grams.get(entry.evidence_id, set())
        ev_flag = evidence_flags.get(entry.evidence_id, {})
        if gated and not cache_gate(aircraft_model, flags, qgrams, entry, ev_flag, ev_gram):
            continue
        text_sim = jaccard(qgrams, ev_gram)
        source_query_sim = jaccard(qgrams, entry.source_query_grams)
        rank_bonus = 1.0 / math.log2(entry.rank + 2)
        recency_bonus = 1.0 / math.log2(max(2, step - entry.insertion_step + 2))
        score = 0.62 * text_sim + 0.20 * source_query_sim + 0.10 * rank_bonus + 0.08 * recency_bonus
        if entry.evidence_id not in best or score > best[entry.evidence_id][0]:
            best[entry.evidence_id] = (score, entry)
    return [(eid, score, entry) for eid, (score, entry) in sorted(best.items(), key=lambda item: item[1][0], reverse=True)]


def choose_ids(
    policy: str,
    fresh_ids: list[str],
    cache: list[CacheEntry],
    query_text: str,
    aircraft_model: str,
    evidence: pd.DataFrame,
    evidence_grams: dict[str, set[str]],
    evidence_flags: dict[str, dict[str, object]],
    step: int,
) -> tuple[list[str], dict[str, object]]:
    fresh_ids = [eid for eid in fresh_ids if eid in evidence.index][:10]
    if policy == "fresh_only" or not cache:
        return fresh_ids[:10], {
            "target_reuse_slots": 0,
            "actual_reuse_slots": 0,
            "fresh_slots": len(fresh_ids[:10]),
            "cache_pool_size": len(cache),
            "cache_candidate_count": 0,
            "query_entropy": normalized_entropy([1.0]),
            "cache_quality": 0.0,
        }

    gated = policy == "admissibility_signature_gated_entropy"
    candidates = cache_candidates(cache, query_text, aircraft_model, evidence, evidence_grams, evidence_flags, step, gated=gated)
    qgrams = char_ngrams(query_text)
    fresh_scores = [jaccard(qgrams, evidence_grams.get(eid, set())) for eid in fresh_ids]
    query_entropy = normalized_entropy(fresh_scores)
    cache_quality = candidates[0][1] if candidates else 0.0

    if policy == "naive_semantic_reuse":
        target_reuse = 10
    elif policy == "fixed_50_50":
        target_reuse = 5
    elif policy in {"entropy_adaptive", "admissibility_signature_gated_entropy"}:
        quality_factor = min(1.0, cache_quality / 0.16)
        target_reuse = int(round(10 * (1.0 - query_entropy) * quality_factor))
        if candidates and target_reuse == 0 and cache_quality >= 0.06:
            target_reuse = 1
        target_reuse = max(0, min(10, target_reuse))
    else:
        raise ValueError(f"Unknown policy: {policy}")

    selected: list[str] = []
    for eid, _, _ in candidates:
        if len(selected) >= target_reuse:
            break
        if eid not in selected:
            selected.append(eid)

    reused = len(selected)
    for eid in fresh_ids:
        if len(selected) >= 10:
            break
        if eid not in selected:
            selected.append(eid)

    if len(selected) < 10:
        for eid, _, _ in candidates:
            if len(selected) >= 10:
                break
            if eid not in selected:
                selected.append(eid)

    return selected[:10], {
        "target_reuse_slots": target_reuse,
        "actual_reuse_slots": reused,
        "fresh_slots": len(selected[:10]) - reused,
        "cache_pool_size": len(cache),
        "cache_candidate_count": len(candidates),
        "query_entropy": query_entropy,
        "cache_quality": cache_quality,
    }


def choose_ids_from_precomputed(
    policy: str,
    fresh_ids: list[str],
    cache_pool_size: int,
    ungated_candidates: list[tuple[str, float, CacheEntry]],
    gated_candidates: list[tuple[str, float, CacheEntry]],
    query_entropy: float,
) -> tuple[list[str], dict[str, object]]:
    fresh_ids = fresh_ids[:10]
    if policy == "fresh_only" or cache_pool_size == 0:
        return fresh_ids[:10], {
            "target_reuse_slots": 0,
            "actual_reuse_slots": 0,
            "fresh_slots": len(fresh_ids[:10]),
            "cache_pool_size": cache_pool_size,
            "cache_candidate_count": 0,
            "query_entropy": query_entropy,
            "cache_quality": 0.0,
        }

    candidates = gated_candidates if policy == "admissibility_signature_gated_entropy" else ungated_candidates
    cache_quality = candidates[0][1] if candidates else 0.0
    if policy == "naive_semantic_reuse":
        target_reuse = 10
    elif policy == "fixed_50_50":
        target_reuse = 5
    elif policy in {"entropy_adaptive", "admissibility_signature_gated_entropy"}:
        quality_factor = min(1.0, cache_quality / 0.16)
        target_reuse = int(round(10 * (1.0 - query_entropy) * quality_factor))
        if candidates and target_reuse == 0 and cache_quality >= 0.06:
            target_reuse = 1
        target_reuse = max(0, min(10, target_reuse))
    else:
        raise ValueError(f"Unknown policy: {policy}")

    selected: list[str] = []
    for eid, _, _ in candidates:
        if len(selected) >= target_reuse:
            break
        if eid not in selected:
            selected.append(eid)

    reused = len(selected)
    for eid in fresh_ids:
        if len(selected) >= 10:
            break
        if eid not in selected:
            selected.append(eid)

    if len(selected) < 10:
        for eid, _, _ in candidates:
            if len(selected) >= 10:
                break
            if eid not in selected:
                selected.append(eid)

    return selected[:10], {
        "target_reuse_slots": target_reuse,
        "actual_reuse_slots": reused,
        "fresh_slots": len(selected[:10]) - reused,
        "cache_pool_size": cache_pool_size,
        "cache_candidate_count": len(candidates),
        "query_entropy": query_entropy,
        "cache_quality": cache_quality,
    }


def hit_metrics(selected_ids: list[str], gold_row: pd.Series, evidence: pd.DataFrame) -> dict[str, object]:
    exact = set(parse_ids(gold_row.get("exact_gold_evidence_ids", "")))
    accepted = set(parse_ids(gold_row.get("accepted_gold_evidence_ids", ""))) | exact
    accepted_pages = set(parse_ids(gold_row.get("accepted_gold_page_ids", "")))
    accepted_sections = set(parse_ids(gold_row.get("accepted_gold_section_ids", "")))
    selected_set = set(selected_ids)
    selected_rows = evidence.reindex([eid for eid in selected_ids if eid in evidence.index]).fillna("")
    pages = set(selected_rows.get("page_id", pd.Series(dtype=str)).astype(str))
    sections = set(selected_rows.get("detected_section_id", pd.Series(dtype=str)).astype(str))
    return {
        "exact_hit_10": float(bool(selected_set & exact)),
        "accepted_hit_10": float(bool(selected_set & accepted)),
        "page_hit_10": float(bool(pages & accepted_pages)) if accepted_pages else np.nan,
        "section_hit_10": float(bool(sections & accepted_sections)) if accepted_sections else np.nan,
        "n_selected": len(selected_ids),
        "n_accepted_selected": len(selected_set & accepted),
    }


def bootstrap_seed_ci(seed_summary: pd.DataFrame, metrics: list[str], n_boot: int = 5000) -> pd.DataFrame:
    rng = np.random.default_rng(20260816)
    rows = []
    for policy, group in seed_summary.groupby("policy", sort=True):
        for metric in metrics:
            values = group[metric].astype(float).to_numpy()
            boots = []
            for _ in range(n_boot):
                sample = rng.choice(values, size=len(values), replace=True)
                boots.append(float(np.mean(sample)))
            rows.append(
                {
                    "policy": policy,
                    "metric": metric,
                    "mean_of_seed_means": float(np.mean(values)),
                    "ci95_low": float(np.percentile(boots, 2.5)),
                    "ci95_high": float(np.percentile(boots, 97.5)),
                    "n_seeds": int(len(values)),
                    "bootstrap_samples": n_boot,
                    "ci_unit": "seed_mean_resampling",
                }
            )
    return pd.DataFrame(rows)


def summarize(records: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_cols = [
        "accepted_hit_10",
        "page_hit_10",
        "section_hit_10",
        "exact_hit_10",
        "n_accepted_selected",
        "actual_reuse_slots",
        "fresh_slots",
        "target_reuse_slots",
        "cache_candidate_count",
        "query_entropy",
        "cache_quality",
    ]
    seed_summary = (
        records.groupby(["seed", "policy"], sort=True)[metric_cols]
        .mean(numeric_only=True)
        .reset_index()
    )
    overall = (
        seed_summary.groupby("policy", sort=True)[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    overall.columns = ["_".join([str(x) for x in col if str(x)]) for col in overall.columns.to_flat_index()]
    ci = bootstrap_seed_ci(
        seed_summary,
        [
            "accepted_hit_10",
            "page_hit_10",
            "section_hit_10",
            "actual_reuse_slots",
            "fresh_slots",
            "n_accepted_selected",
        ],
    )
    return seed_summary, overall, ci


def make_report(overall: pd.DataFrame, ci: pd.DataFrame, leakage: dict[str, object]) -> str:
    lines = [
        "# Offline long/short-state reuse simulator report",
        "",
        "This run simulates online reuse over 214 cue-free IntentRAG Top-10 packages.",
        "Short-term state is the fresh current package; long-term state is the cache of earlier query evidence-state packages under a randomized query order.",
        "",
        "## Policies",
        "",
        "- `fresh_only`: always use current cue-free IntentRAG Top-10.",
        "- `naive_semantic_reuse`: fill up to 10 slots from previous cached evidence ranked by visible-query/evidence semantic overlap.",
        "- `fixed_50_50`: fill up to 5 cached slots, then fresh slots.",
        "- `entropy_adaptive`: choose cached-slot budget from visible-query/fresh-package entropy and cache semantic quality.",
        "- `admissibility_signature_gated_entropy`: entropy-adaptive reuse after aircraft/authority/condition/warning and semantic compatibility gates.",
        "",
        "## Summary over seed means",
        "",
        "| policy | accepted@10 | page@10 | section@10 | reused slots | fresh slots | accepted selected |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in overall.iterrows():
        lines.append(
            "| {policy} | {acc:.4f} | {page:.4f} | {sec:.4f} | {reuse:.2f} | {fresh:.2f} | {nacc:.3f} |".format(
                policy=row["policy"],
                acc=float(row.get("accepted_hit_10_mean", 0.0)),
                page=float(row.get("page_hit_10_mean", 0.0)),
                sec=float(row.get("section_hit_10_mean", 0.0)),
                reuse=float(row.get("actual_reuse_slots_mean", 0.0)),
                fresh=float(row.get("fresh_slots_mean", 0.0)),
                nacc=float(row.get("n_accepted_selected_mean", 0.0)),
            )
        )
    lines += [
        "",
        "## Bootstrap / CI",
        "",
        "CI is percentile bootstrap over the 20 fixed seed-level means, not over independent natural user traffic.",
        "",
        "| policy | metric | mean | 95% CI |",
        "|---|---|---:|---:|",
    ]
    for _, row in ci.iterrows():
        lines.append(
            f"| {row['policy']} | {row['metric']} | {float(row['mean_of_seed_means']):.4f} | [{float(row['ci95_low']):.4f}, {float(row['ci95_high']):.4f}] |"
        )
    lines += [
        "",
        "## Leakage statement",
        "",
        f"- Selection input columns: `{', '.join(leakage['selection_input_columns'])}`.",
        f"- Forbidden selection columns intersect used columns: `{', '.join(leakage['forbidden_intersection']) or 'none'}`.",
        f"- Cue-free IntentRAG rows: {leakage['n_intentrag_rows']}.",
        f"- Gold/accepted/page/section labels are loaded only after selection for post-hoc hit computation.",
        f"- `track`, `query_type`, `required_slots`, and all `gold/accepted/offline metric` fields are not used by policy scoring, slot budgeting, cache ranking, or gates.",
        "",
        "## Boundary",
        "",
        "This is an offline simulator over existing retrieved packages. It does not rerun a live retriever, measure server-side KV cache, or validate aircraft maintenance correctness.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ensure_dirs()

    access = pd.read_csv(INPUT_ACCESS).fillna("")
    intent = access[access["condition"].eq("cue_off_intentrag")].copy()
    if len(intent) != 214:
        raise RuntimeError(f"Expected 214 cue_off_intentrag rows, found {len(intent)}")

    flag_checks = {
        "gold_used_for_retrieval_feature": set(intent["gold_used_for_retrieval_feature"].astype(str).str.lower()),
        "gold_cue_used_for_retrieval_feature": set(intent["gold_cue_used_for_retrieval_feature"].astype(str).str.lower()),
        "locked_track_used_for_routing": set(intent["locked_track_used_for_routing"].astype(str).str.lower()),
    }
    for col, values in flag_checks.items():
        if not values <= {"false", "0"}:
            raise RuntimeError(f"Input is not clean cue-free for {col}: {values}")

    selection_frame = intent[list(SELECTION_INPUT_COLUMNS)].copy()
    forbidden_intersection = sorted(SELECTION_INPUT_COLUMNS & FORBIDDEN_SELECTION_COLUMNS)
    if forbidden_intersection:
        raise RuntimeError(f"Forbidden columns selected: {forbidden_intersection}")

    all_fresh_ids: set[str] = set()
    for value in selection_frame["top_evidence_ids"]:
        all_fresh_ids.update(parse_ids(value))

    gold = pd.read_csv(INPUT_GOLD).fillna("").set_index("question_id")
    evidence = pd.read_parquet(INPUT_EVIDENCE).fillna("").drop_duplicates("evidence_id")
    evidence = evidence[evidence["evidence_id"].astype(str).isin(all_fresh_ids)].set_index("evidence_id")
    if evidence.empty:
        raise RuntimeError("No evidence metadata matched cue-free IntentRAG Top-10 ids.")
    evidence["_search_text"] = evidence.apply(evidence_tokens, axis=1)
    evidence_grams = {eid: char_ngrams(text) for eid, text in evidence["_search_text"].items()}
    evidence_flags = {
        eid: {
            "authority_present": bool(str(row.get("authority_terms", "")).strip()),
            "condition_present": bool(str(row.get("condition_terms", "")).strip()),
            "warning_present": bool(str(row.get("warning_terms", "")).strip()),
            "aircraft_model": str(row.get("aircraft_model", "")),
            "page_id": str(row.get("page_id", "")),
            "detected_section_id": str(row.get("detected_section_id", "")),
        }
        for eid, row in evidence.iterrows()
    }

    query_records = []
    order_manifest = []

    rows_by_qid = {str(row["question_id"]): row for _, row in selection_frame.iterrows()}
    qids = list(rows_by_qid)

    for seed in SEEDS:
        print(json.dumps({"status": "running_seed", "seed": seed}, ensure_ascii=False), flush=True)
        rng = random.Random(seed)
        order = qids.copy()
        rng.shuffle(order)
        cache_by_evidence: dict[str, CacheEntry] = {}

        for step, qid in enumerate(order):
            cache = list(cache_by_evidence.values())
            row = rows_by_qid[qid]
            query_text = str(row.get("base_question") or row.get("query_text") or "")
            aircraft = str(row.get("aircraft_model", ""))
            fresh_ids = parse_ids(row.get("top_evidence_ids", ""))[:10]
            fresh_ids = [eid for eid in fresh_ids if eid in evidence.index][:10]
            qgrams = char_ngrams(query_text)
            fresh_scores = [jaccard(qgrams, evidence_grams.get(eid, set())) for eid in fresh_ids]
            query_entropy = normalized_entropy(fresh_scores)
            ungated_candidates = cache_candidates(
                cache, query_text, aircraft, evidence, evidence_grams, evidence_flags, step, gated=False
            )
            gated_candidates = cache_candidates(
                cache, query_text, aircraft, evidence, evidence_grams, evidence_flags, step, gated=True
            )

            for policy in POLICIES:
                selected_ids, meta = choose_ids_from_precomputed(
                    policy,
                    fresh_ids,
                    len(cache),
                    ungated_candidates,
                    gated_candidates,
                    query_entropy,
                )
                metrics = hit_metrics(selected_ids, gold.loc[qid], evidence)
                query_records.append(
                    {
                        "seed": seed,
                        "order_index": step,
                        "question_id": qid,
                        "policy": policy,
                        "aircraft_model": aircraft,
                        "selected_evidence_ids": ";".join(selected_ids),
                        "fresh_intentrag_evidence_ids": ";".join(fresh_ids),
                        **meta,
                        **metrics,
                    }
                )

            sig = package_signature(fresh_ids, evidence)
            for rank, eid in enumerate(fresh_ids, 1):
                cache_by_evidence[eid] = CacheEntry(
                    evidence_id=eid,
                    source_question_id=qid,
                    source_aircraft_model=aircraft,
                    source_query_text=query_text,
                    source_query_grams=char_ngrams(query_text),
                    rank=rank,
                    insertion_step=step,
                    package_signature=sig,
                )
            order_manifest.append({"seed": seed, "order_index": step, "question_id": qid})

    records = pd.DataFrame(query_records)
    seed_summary, overall, ci = summarize(records)

    tradeoff = overall[
        [
            "policy",
            "actual_reuse_slots_mean",
            "fresh_slots_mean",
            "accepted_hit_10_mean",
            "page_hit_10_mean",
            "section_hit_10_mean",
            "n_accepted_selected_mean",
        ]
    ].copy()

    leakage = {
        "selection_input_columns": sorted(SELECTION_INPUT_COLUMNS),
        "forbidden_selection_columns": sorted(FORBIDDEN_SELECTION_COLUMNS),
        "forbidden_intersection": forbidden_intersection,
        "n_intentrag_rows": int(len(intent)),
        "seeds": SEEDS,
        "input_files": {
            "cue_free_access_query": str(INPUT_ACCESS),
            "v21_gold_sets_posthoc_only": str(INPUT_GOLD),
            "evidence_objects": str(INPUT_EVIDENCE),
        },
        "policy_inputs_exclude": [
            "gold labels",
            "accepted labels",
            "track",
            "query_type",
            "required_slots",
            "offline metrics",
        ],
    }

    records.to_csv(OUT_RESULTS / "long_short_reuse_query_seed_results.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(order_manifest).to_csv(OUT_RESULTS / "query_order_manifest.csv", index=False, encoding="utf-8-sig")
    seed_summary.to_csv(OUT_RESULTS / "long_short_reuse_seed_summary.csv", index=False, encoding="utf-8-sig")
    overall.to_csv(OUT_TABLES / "long_short_reuse_summary.csv", index=False, encoding="utf-8-sig")
    ci.to_csv(OUT_TABLES / "long_short_reuse_bootstrap_ci.csv", index=False, encoding="utf-8-sig")
    tradeoff.to_csv(OUT_TABLES / "long_short_reuse_tradeoff.csv", index=False, encoding="utf-8-sig")
    (OUT_REPORTS / "long_short_reuse_leakage_statement.json").write_text(
        json.dumps(leakage, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT_REPORTS / "long_short_reuse_report.md").write_text(make_report(overall, ci, leakage), encoding="utf-8")
    (OUT_LOGS / "long_short_reuse_run.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "n_query_seed_policy_rows": int(len(records)),
                "n_seed_summary_rows": int(len(seed_summary)),
                "n_seeds": len(SEEDS),
                "n_queries": len(qids),
                "policies": POLICIES,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "records": len(records),
                "summary": str(OUT_TABLES / "long_short_reuse_summary.csv"),
                "report": str(OUT_REPORTS / "long_short_reuse_report.md"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
