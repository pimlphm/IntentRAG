from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from run_offline_long_short_reuse_simulator import (
    FORBIDDEN_SELECTION_COLUMNS,
    INPUT_ACCESS,
    INPUT_EVIDENCE,
    INPUT_GOLD,
    PROJECT_ROOT,
    SEEDS,
    SELECTION_INPUT_COLUMNS,
    WORK_ROOT,
    CacheEntry,
    cache_candidates,
    cache_gate,
    char_ngrams,
    demand_flags,
    evidence_tokens,
    hit_metrics,
    jaccard,
    normalized_entropy,
    package_signature,
    parse_ids,
    summarize,
)


OUT_RESULTS = WORK_ROOT / "results" / "long_short_reuse_v2_strict"
OUT_TABLES = WORK_ROOT / "tables" / "long_short_reuse_v2_strict"
OUT_REPORTS = WORK_ROOT / "reports"
OUT_LOGS = WORK_ROOT / "logs"

POLICIES_V2 = [
    "fresh_only",
    "naive_semantic_reuse",
    "fixed_50_50",
    "entropy_adaptive",
    "admissibility_signature_gated_entropy",
    "component_document_signature_gated_entropy",
]


def ensure_dirs() -> None:
    for path in [OUT_RESULTS, OUT_TABLES, OUT_REPORTS, OUT_LOGS]:
        path.mkdir(parents=True, exist_ok=True)


def strict_component_document_candidates(
    cache: list[CacheEntry],
    query_text: str,
    aircraft_model: str,
    evidence_grams: dict[str, set[str]],
    evidence_flags: dict[str, dict[str, object]],
    fresh_signature: dict[str, object],
    step: int,
) -> list[tuple[str, float, CacheEntry]]:
    qgrams = char_ngrams(query_text)
    flags = demand_flags(query_text)
    current_components = {
        str(x)
        for x in fresh_signature.get("component_families", [])
        if str(x).strip() and str(x).strip() != "no_component"
    }
    current_docs = {str(x) for x in fresh_signature.get("source_documents", []) if str(x).strip()}
    best: dict[str, tuple[float, CacheEntry]] = {}
    for entry in cache:
        if entry.source_aircraft_model != aircraft_model:
            continue
        if entry.evidence_id not in evidence_grams:
            continue
        ev_flag = evidence_flags.get(entry.evidence_id, {})
        ev_gram = evidence_grams.get(entry.evidence_id, set())
        if not cache_gate(aircraft_model, flags, qgrams, entry, ev_flag, ev_gram):
            continue

        cached_components = {
            str(x)
            for x in entry.package_signature.get("component_families", [])
            if str(x).strip() and str(x).strip() != "no_component"
        }
        ev_component = str(ev_flag.get("component_family", "")).strip()
        if current_components and not (
            (ev_component in current_components) or bool(cached_components & current_components)
        ):
            continue

        cached_docs = {str(x) for x in entry.package_signature.get("source_documents", []) if str(x).strip()}
        ev_doc = str(ev_flag.get("document_id", "")).strip()
        if aircraft_model == "Cessna 172" and current_docs and not (
            (ev_doc in current_docs) or bool(cached_docs & current_docs)
        ):
            continue

        text_sim = jaccard(qgrams, ev_gram)
        source_query_sim = jaccard(qgrams, entry.source_query_grams)
        rank_bonus = 1.0 / math.log2(entry.rank + 2)
        recency_bonus = 1.0 / math.log2(max(2, step - entry.insertion_step + 2))
        score = 0.62 * text_sim + 0.20 * source_query_sim + 0.10 * rank_bonus + 0.08 * recency_bonus
        if entry.evidence_id not in best or score > best[entry.evidence_id][0]:
            best[entry.evidence_id] = (score, entry)
    return [(eid, score, entry) for eid, (score, entry) in sorted(best.items(), key=lambda item: item[1][0], reverse=True)]


def choose_from_candidates(
    policy: str,
    fresh_ids: list[str],
    cache_pool_size: int,
    ungated_candidates: list[tuple[str, float, CacheEntry]],
    gated_candidates: list[tuple[str, float, CacheEntry]],
    strict_candidates: list[tuple[str, float, CacheEntry]],
    query_entropy: float,
) -> tuple[list[str], dict[str, object]]:
    fresh_ids = fresh_ids[:10]
    if policy == "fresh_only" or cache_pool_size == 0:
        return fresh_ids, {
            "target_reuse_slots": 0,
            "actual_reuse_slots": 0,
            "fresh_slots": len(fresh_ids),
            "cache_pool_size": cache_pool_size,
            "cache_candidate_count": 0,
            "query_entropy": query_entropy,
            "cache_quality": 0.0,
        }
    if policy == "component_document_signature_gated_entropy":
        candidates = strict_candidates
    elif policy == "admissibility_signature_gated_entropy":
        candidates = gated_candidates
    else:
        candidates = ungated_candidates

    cache_quality = candidates[0][1] if candidates else 0.0
    if policy == "naive_semantic_reuse":
        target_reuse = 10
    elif policy == "fixed_50_50":
        target_reuse = 5
    elif policy in {
        "entropy_adaptive",
        "admissibility_signature_gated_entropy",
        "component_document_signature_gated_entropy",
    }:
        quality_factor = min(1.0, cache_quality / 0.16)
        target_reuse = int(round(10 * (1.0 - query_entropy) * quality_factor))
        if candidates and target_reuse == 0 and cache_quality >= 0.06:
            target_reuse = 1
        target_reuse = max(0, min(10, target_reuse))
    else:
        raise ValueError(policy)

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


def make_report(overall: pd.DataFrame, ci: pd.DataFrame, leakage: dict[str, object]) -> str:
    rows = [
        "# Offline long/short-state reuse simulator v2 strict report",
        "",
        "Adds `component_document_signature_gated_entropy` without overwriting v1 outputs.",
        "The strict policy uses only current fresh-package metadata: aircraft, authority, condition, warning, non-empty component_family compatibility, and Cessna source-document compatibility.",
        "",
        "| policy | accepted@10 | reused slots | fresh slots |",
        "|---|---:|---:|---:|",
    ]
    for _, row in overall.iterrows():
        rows.append(
            f"| {row['policy']} | {float(row['accepted_hit_10_mean']):.4f} | {float(row['actual_reuse_slots_mean']):.2f} | {float(row['fresh_slots_mean']):.2f} |"
        )
    rows += [
        "",
        "## Leakage boundary",
        "",
        f"- Selection input columns: `{', '.join(leakage['selection_input_columns'])}`.",
        f"- Forbidden selection columns intersect used columns: `{', '.join(leakage['forbidden_intersection']) or 'none'}`.",
        "- Gold/accepted/page/section labels are post-hoc scoring only.",
        "- `track`, `query_type`, `required_slots`, and offline metrics are not used by any policy.",
        "",
        "## Boundary",
        "",
        "Offline reuse simulator over existing cue-free IntentRAG packages; not real KV-cache hits, not live retrieval, not maintenance correctness.",
    ]
    return "\n".join(rows) + "\n"


def main() -> None:
    ensure_dirs()
    access = pd.read_csv(INPUT_ACCESS).fillna("")
    intent = access[access["condition"].eq("cue_off_intentrag")].copy()
    if len(intent) != 214:
        raise RuntimeError(f"Expected 214 cue_off_intentrag rows, found {len(intent)}")
    for col in [
        "gold_used_for_retrieval_feature",
        "gold_cue_used_for_retrieval_feature",
        "locked_track_used_for_routing",
    ]:
        values = set(intent[col].astype(str).str.lower())
        if not values <= {"false", "0"}:
            raise RuntimeError(f"Input is not clean cue-free for {col}: {values}")

    selection_frame = intent[list(SELECTION_INPUT_COLUMNS)].copy()
    forbidden_intersection = sorted(SELECTION_INPUT_COLUMNS & FORBIDDEN_SELECTION_COLUMNS)
    all_fresh_ids = set()
    for value in selection_frame["top_evidence_ids"]:
        all_fresh_ids.update(parse_ids(value))
    gold = pd.read_csv(INPUT_GOLD).fillna("").set_index("question_id")
    evidence = pd.read_parquet(INPUT_EVIDENCE).fillna("").drop_duplicates("evidence_id")
    evidence = evidence[evidence["evidence_id"].astype(str).isin(all_fresh_ids)].set_index("evidence_id")
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
            "component_family": str(row.get("component_family", "")),
            "document_id": str(row.get("document_id", "")),
        }
        for eid, row in evidence.iterrows()
    }

    rows_by_qid = {str(row["question_id"]): row for _, row in selection_frame.iterrows()}
    qids = list(rows_by_qid)
    query_records = []
    order_manifest = []
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
            fresh_ids = [eid for eid in parse_ids(row.get("top_evidence_ids", "")) if eid in evidence.index][:10]
            qgrams = char_ngrams(query_text)
            fresh_scores = [jaccard(qgrams, evidence_grams.get(eid, set())) for eid in fresh_ids]
            query_entropy = normalized_entropy(fresh_scores)
            fresh_sig = package_signature(fresh_ids, evidence)
            ungated_candidates = cache_candidates(
                cache, query_text, aircraft, evidence, evidence_grams, evidence_flags, step, gated=False
            )
            gated_candidates = cache_candidates(
                cache, query_text, aircraft, evidence, evidence_grams, evidence_flags, step, gated=True
            )
            strict_candidates = strict_component_document_candidates(
                cache, query_text, aircraft, evidence_grams, evidence_flags, fresh_sig, step
            )

            for policy in POLICIES_V2:
                selected_ids, meta = choose_from_candidates(
                    policy,
                    fresh_ids,
                    len(cache),
                    ungated_candidates,
                    gated_candidates,
                    strict_candidates,
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

            for rank, eid in enumerate(fresh_ids, 1):
                cache_by_evidence[eid] = CacheEntry(
                    evidence_id=eid,
                    source_question_id=qid,
                    source_aircraft_model=aircraft,
                    source_query_text=query_text,
                    source_query_grams=char_ngrams(query_text),
                    rank=rank,
                    insertion_step=step,
                    package_signature=fresh_sig,
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
        "new_policy": "component_document_signature_gated_entropy",
        "new_policy_boundary": "Uses current visible fresh-package metadata only: component_family and source-document compatibility, not gold labels.",
    }
    records.to_csv(OUT_RESULTS / "long_short_reuse_v2_query_seed_results.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(order_manifest).to_csv(OUT_RESULTS / "query_order_manifest_v2.csv", index=False, encoding="utf-8-sig")
    seed_summary.to_csv(OUT_RESULTS / "long_short_reuse_v2_seed_summary.csv", index=False, encoding="utf-8-sig")
    overall.to_csv(OUT_TABLES / "long_short_reuse_v2_summary.csv", index=False, encoding="utf-8-sig")
    ci.to_csv(OUT_TABLES / "long_short_reuse_v2_bootstrap_ci.csv", index=False, encoding="utf-8-sig")
    tradeoff.to_csv(OUT_TABLES / "long_short_reuse_v2_tradeoff.csv", index=False, encoding="utf-8-sig")
    (OUT_REPORTS / "long_short_reuse_v2_leakage_statement.json").write_text(
        json.dumps(leakage, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT_REPORTS / "long_short_reuse_v2_report.md").write_text(make_report(overall, ci, leakage), encoding="utf-8")
    (OUT_LOGS / "long_short_reuse_v2_run.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "n_query_seed_policy_rows": int(len(records)),
                "n_seeds": len(SEEDS),
                "n_queries": len(qids),
                "policies": POLICIES_V2,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"status": "ok", "records": len(records), "out": str(OUT_RESULTS)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
