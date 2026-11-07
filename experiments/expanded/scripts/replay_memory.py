"""Past-only evidence module LRU replay, with separate routing-index diagnostics.

This experiment reuses the identity and exact source text of already requested
evidence modules. It leaves retrieval output unchanged. Index hits are never
counted as retrieved child evidence, reusable text, or GPU KV-cache hits.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict, Counter
import csv
import hashlib
import json
from pathlib import Path
import random
import re
import sys
import time
from typing import Any

import pandas as pd


TOPICS = [
    ("engine", r"\bengine\b|发动机|引擎"),
    ("hydraulic", r"\bhydraulic\b|液压"),
    ("electrical", r"\belectri|\bbattery\b|电气|电池"),
    ("flight_control", r"\bflap|\belevator\b|\brudder\b|\baileron\b|襟翼|升降舵|方向舵|副翼"),
    ("landing_gear", r"\bgear\b|\bbrake\b|\btire\b|起落架|刹车|轮胎"),
    ("fuel", r"\bfuel\b|燃油"),
    ("air_condition", r"\bpack\b|\bpressuri|\bair condition|空调|增压"),
]


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def parse_ids(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    value = str(value or "").strip()
    if value.startswith("["):
        parsed = json.loads(value)
        if not isinstance(parsed, list):
            raise ValueError("top_evidence_ids JSON must be a list")
        return parse_ids(parsed)
    return [v.strip() for v in value.split(";") if v.strip()]


def token_counter(tokenizer_json: Path | None):
    if tokenizer_json:
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(tokenizer_json))
        return (lambda t: len(tokenizer.encode(t, add_special_tokens=False).ids)), {
            "token_count_kind": "tokenizer_source_text_tokens",
            "tokenizer_json": str(tokenizer_json),
            "tokenizer_sha256": digest_file(tokenizer_json),
            "note": "Source-module token mass, not a rendered chat prefix or KV hit.",
        }
    # Mixed Chinese/English token-mass proxy; no unavailable tokenizer is assumed.
    pattern = re.compile(r"[\u3400-\u9fff]|[A-Za-z0-9_]+|[^\w\s]", re.UNICODE)
    return (lambda t: len(pattern.findall(t))), {
        "token_count_kind": "regex_source_text_token_mass_proxy",
        "tokenizer_json": None,
        "note": "One CJK character, word, or punctuation token. Not model tokenizer tokens.",
    }


class LRU:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.data: OrderedDict[tuple[str, ...], None] = OrderedDict()

    def contains(self, key: tuple[str, ...]) -> bool:
        return key in self.data

    def update(self, keys: list[tuple[str, ...]]) -> int:
        evictions = 0
        for key in keys:
            if key in self.data:
                self.data.move_to_end(key)
            else:
                self.data[key] = None
                if len(self.data) > self.capacity:
                    self.data.popitem(last=False)
                    evictions += 1
        return evictions


def topic_key(question: str, aircraft: str, question_id: str):
    low = re.sub(r"\s+", " ", question.lower()).strip()
    topic = next((name for name, pattern in TOPICS if re.search(pattern, low)), "other")
    return str(aircraft), topic, low, question_id


def load_inputs(retrieval: Path, canonical: Path, corpus: Path, count_tokens):
    questions = pd.read_csv(canonical, keep_default_na=False)
    if "question_id" not in questions:
        raise ValueError("canonical_questions.csv must contain question_id")
    if questions.question_id.duplicated().any():
        raise ValueError("Canonical question_id is duplicated; replay would be ambiguous")
    canonical_map = {str(r["question_id"]): r for r in questions.to_dict("records")}
    runs: dict[tuple[str, str], dict[str, Any]] = {}
    methods: set[str] = set()
    with retrieval.open("r", encoding="utf-8-sig") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            r = json.loads(line)
            qid = str(r.get("question_id", r.get("query_id", "")))
            method = str(r.get("method", ""))
            if not qid or not method:
                raise ValueError(f"Missing question_id/method at line {line_number}")
            key = (qid, method)
            if key in runs:
                raise ValueError(f"Duplicate retrieval row {key}; do not select a favourable row")
            r["question_id"], r["method"] = qid, method
            r["top_evidence_ids"] = parse_ids(r.get("top_evidence_ids", []))
            runs[key] = r
            methods.add(method)

    frame = pd.read_parquet(corpus).fillna("")
    if frame.evidence_id.duplicated().any():
        raise ValueError("Corpus evidence_id is duplicated")
    evidence = {}
    for r in frame.to_dict("records"):
        text = str(r.get("text", ""))
        r["text_hash"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
        r["source_token_mass"] = int(count_tokens(text))
        r["module_key"] = (
            str(r.get("aircraft_model", "")), str(r.get("document_id", "")),
            str(r.get("page_id", "")), str(r["evidence_id"]), r["text_hash"],
        )
        r["page_key"] = (str(r.get("aircraft_model", "")), str(r.get("document_id", "")), str(r.get("page_id", "")))
        r["section_key"] = (str(r.get("aircraft_model", "")), str(r.get("document_id", "")), str(r.get("detected_section_id", "")))
        evidence[str(r["evidence_id"])] = r

    coverage = []
    for qid in sorted(canonical_map):
        for method in sorted(methods):
            r = runs.get((qid, method))
            coverage.append({
                "question_id": qid, "method": method,
                "split": canonical_map[qid].get("split", ""),
                "status": "available" if r is not None else "missing_retrieval_row",
                "n_requested_ids": len(r["top_evidence_ids"]) if r else 0,
                "n_unknown_evidence_ids": sum(v not in evidence for v in r["top_evidence_ids"]) if r else 0,
            })
    extras = [{"question_id": qid, "method": method, "status": "not_in_canonical"}
              for qid, method in runs if qid not in canonical_map]
    return canonical_map, runs, evidence, sorted(methods), coverage, extras


def run_replay(canonical_map, runs, evidence, methods, capacities, seeds):
    orders = [("random", seed) for seed in seeds] + [("topical", None)]
    policies = ["flat_identity_lru", "identity_lru_plus_routing_indices"]
    rows = []
    order_rows = []
    # Each split gets a cold start; no calibration/test cross-boundary warming.
    split_ids: dict[str, list[str]] = {}
    for qid, q in canonical_map.items():
        split_ids.setdefault(str(q.get("split", "unspecified")), []).append(qid)
    for split, all_ids in sorted(split_ids.items()):
        for order_name, seed in orders:
            ids = sorted(all_ids)
            if order_name == "random":
                random.Random(seed).shuffle(ids)
            else:
                ids.sort(key=lambda qid: topic_key(
                    str(canonical_map[qid].get("question", "")),
                    str(canonical_map[qid].get("aircraft", canonical_map[qid].get("aircraft_model", ""))), qid))
            for pos, qid in enumerate(ids):
                order_rows.append({"split": split, "order": order_name, "seed": seed,
                                   "request_index": pos, "question_id": qid})
            for method in methods:
                for capacity in capacities:
                    for policy in policies:
                        module_cache, page_cache, section_cache = LRU(capacity), LRU(capacity), LRU(capacity)
                        for pos, qid in enumerate(ids):
                            run = runs.get((qid, method))
                            if run is None:
                                continue  # Explicit missing cell is recorded in coverage.
                            requested = list(dict.fromkeys(run["top_evidence_ids"]))
                            known = [evidence[v] for v in requested if v in evidence]
                            # All lookups happen against the previous-request state.
                            hits = [r for r in known if module_cache.contains(r["module_key"])]
                            total_mass = sum(r["source_token_mass"] for r in known)
                            reused_mass = sum(r["source_token_mass"] for r in hits)
                            page_keys = list(dict.fromkeys(r["page_key"] for r in known))
                            section_keys = list(dict.fromkeys(r["section_key"] for r in known))
                            with_indices = policy == "identity_lru_plus_routing_indices"
                            page_hits = sum(page_cache.contains(k) for k in page_keys) if with_indices else 0
                            section_hits = sum(section_cache.contains(k) for k in section_keys) if with_indices else 0
                            occupancy_before = len(module_cache.data)
                            module_evictions = module_cache.update([r["module_key"] for r in known])
                            page_evictions = page_cache.update(page_keys) if with_indices else 0
                            section_evictions = section_cache.update(section_keys) if with_indices else 0
                            expected_scope = str(run.get("aircraft", run.get("aircraft_model", "")))
                            # Preserve even duplicate ranks in the frozen retrieval output.
                            output_ids = run["top_evidence_ids"].copy()
                            retained = len(set(requested).intersection(output_ids))
                            rows.append({
                                "question_id": qid, "method": method, "split": split,
                                "aircraft": expected_scope, "order": order_name, "seed": seed,
                                "request_index": pos, "policy": policy, "capacity": capacity,
                                "identity_capacity": capacity,
                                "page_index_capacity": capacity if with_indices else 0,
                                "section_index_capacity": capacity if with_indices else 0,
                                "n_requested_ids": len(requested), "n_input_id_duplicates": len(run["top_evidence_ids"])-len(requested),
                                "n_known_requested_ids": len(known), "n_unknown_evidence_ids": len(requested)-len(known),
                                "n_cached_identity_hits": len(hits),
                                "module_hit_fraction": len(hits)/len(known) if known else None,
                                "requested_token_mass": total_mass, "reused_token_mass": reused_mass,
                                "token_weighted_reused_fraction": reused_mass/total_mass if total_mass else None,
                                "cache_occupancy_before": occupancy_before, "cache_occupancy_after": len(module_cache.data),
                                "module_evictions": module_evictions,
                                "n_requested_pages": len(page_keys), "n_page_index_hits": page_hits,
                                "page_index_hit_fraction": page_hits/len(page_keys) if with_indices and page_keys else None,
                                "n_requested_sections": len(section_keys), "n_section_index_hits": section_hits,
                                "section_index_hit_fraction": section_hits/len(section_keys) if with_indices and section_keys else None,
                                "page_index_evictions": page_evictions, "section_index_evictions": section_evictions,
                                "page_index_occupancy_after": len(page_cache.data),
                                "section_index_occupancy_after": len(section_cache.data),
                                "exact_id_support_retention": retained/len(requested) if requested else 1.0,
                                "ranked_output_unchanged": output_ids == run["top_evidence_ids"],
                                "n_scope_mismatches": sum(bool(expected_scope) and r.get("aircraft_model", "") != expected_scope for r in known),
                                "revision_status": "not_available_in_current_corpus",
                                "authorization_status": "not_evaluated",
                                "gpu_kv_cache_measured": False,
                            })
    return rows, order_rows


def summary_table(rows: list[dict[str, Any]]) -> pd.DataFrame:
    d = pd.DataFrame(rows)
    summaries = []
    for group, part in d.groupby(["split", "method", "capacity", "order", "policy"], dropna=False):
        total = part.requested_token_mass.sum()
        known = part.n_known_requested_ids.sum()
        pg = part.n_requested_pages.sum()
        sec = part.n_requested_sections.sum()
        summaries.append({
            **dict(zip(["split", "method", "capacity", "order", "policy"], group)),
            "n_query_ids": part.question_id.nunique(), "n_requests": len(part),
            "n_order_realizations": part.seed.nunique() if group[3] == "random" else 1,
            "token_weighted_reused_fraction": part.reused_token_mass.sum()/total if total else None,
            "mean_query_token_reused_fraction": part.token_weighted_reused_fraction.mean(),
            "module_hit_fraction": part.n_cached_identity_hits.sum()/known if known else None,
            "requested_token_mass": int(total), "reused_token_mass": int(part.reused_token_mass.sum()),
            "module_evictions": int(part.module_evictions.sum()),
            "page_index_hit_fraction": part.n_page_index_hits.sum()/pg if group[4].endswith("indices") and pg else None,
            "section_index_hit_fraction": part.n_section_index_hits.sum()/sec if group[4].endswith("indices") and sec else None,
            "n_unknown_evidence_ids": int(part.n_unknown_evidence_ids.sum()),
            "min_exact_id_support_retention": part.exact_id_support_retention.min(),
            "all_ranked_outputs_unchanged": bool(part.ranked_output_unchanged.all()),
            "memory_budget_note": "The index policy adds separate page and section metadata slots; source-module identity capacity stays equal.",
        })
    return pd.DataFrame(summaries)


def main():
    started = time.perf_counter()
    ap = argparse.ArgumentParser(description=__doc__)
    exp = Path(__file__).resolve().parents[1]
    project = Path(r'${INTENTRAG_PROJECT_ROOT}')
    ap.add_argument("--retrieval", type=Path, default=exp/"results/retrieval_runs.jsonl")
    ap.add_argument("--canonical", type=Path, default=exp/"data/canonical_questions.csv")
    ap.add_argument("--corpus", type=Path, default=project/"01_processed_corpus/v5/evidence_objects.parquet")
    ap.add_argument("--out", type=Path, default=exp/"results/memory_replay")
    ap.add_argument("--capacities", type=int, nargs="+", default=[16, 64, 256])
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(20261014, 20261019)))
    ap.add_argument("--tokenizer-json", type=Path)
    args = ap.parse_args()
    if not all(v > 0 for v in args.capacities):
        raise ValueError("All capacities must be positive")
    count_tokens, count_info = token_counter(args.tokenizer_json)
    canonical, runs, evidence, methods, coverage, extras = load_inputs(args.retrieval, args.canonical, args.corpus, count_tokens)
    rows, order_rows = run_replay(canonical, runs, evidence, methods, args.capacities, args.seeds)
    if not rows:
        raise ValueError("No canonical retrieval rows available for replay")
    args.out.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(args.out/"replay_query_results.csv", index=False, encoding="utf-8-sig")
    summary_table(rows).to_csv(args.out/"replay_summary.csv", index=False, encoding="utf-8-sig")
    per_order = []
    for key, part in frame.groupby(["split", "method", "capacity", "order", "policy", "seed"], dropna=False):
        mass = part.requested_token_mass.sum()
        per_order.append({
            **dict(zip(["split", "method", "capacity", "order", "policy", "seed"], key)),
            "n_queries": len(part),
            "token_weighted_reused_fraction": part.reused_token_mass.sum()/mass if mass else None,
            "module_hit_fraction": part.n_cached_identity_hits.sum()/part.n_known_requested_ids.sum() if part.n_known_requested_ids.sum() else None,
            "module_evictions": int(part.module_evictions.sum()),
        })
    pd.DataFrame(per_order).to_csv(args.out/"replay_order_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(coverage).to_csv(args.out/"requested_query_method_coverage.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(extras, columns=["question_id", "method", "status"]).to_csv(args.out/"noncanonical_input_rows.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(order_rows).to_csv(args.out/"request_orders.csv", index=False, encoding="utf-8-sig")
    assertions = {
        "ranked_outputs_unchanged": bool(frame.ranked_output_unchanged.all()),
        "all_retained_identity_support_equal_one": bool(frame.exact_id_support_retention.eq(1).all()),
        "reuse_mass_never_exceeds_requested_mass": bool(frame.reused_token_mass.le(frame.requested_token_mass).all()),
        "occupancy_within_capacity": bool(frame.cache_occupancy_after.le(frame.capacity).all()),
        "every_cold_start_has_zero_identity_hits": bool(frame.loc[frame.request_index.eq(0), "n_cached_identity_hits"].eq(0).all()),
    }
    manifest = {
        "experiment": "past_only_exact_source_identity_module_replay",
        "inputs": [{"path": str(p), "sha256": digest_file(p)} for p in [args.retrieval, args.canonical, args.corpus]],
        "capacities_evidence_modules": args.capacities, "random_order_seeds": args.seeds,
        "extra_order": "visible_question_topic_then_aircraft_offline_workload",
        "methods": methods, "n_canonical_question_ids": len(canonical), "n_replay_rows": len(rows),
        "coverage_counts": dict(Counter(r["status"] for r in coverage)), "n_noncanonical_input_rows": len(extras),
        "token_mass_measurement": count_info, "validation": assertions,
        "runtime_seconds": time.perf_counter()-started,
        "cache_key_fields": ["aircraft_model", "document_id", "page_id", "evidence_id", "exact_source_text_sha256"],
        "routing_index_budget": {
            "flat_identity_lru": "No page or section routing-node storage",
            "identity_lru_plus_routing_indices": "Adds capacity page keys and capacity section keys; identity-module capacity remains equal",
            "page_node_universe": len({r["page_key"] for r in evidence.values()}),
            "section_node_universe": len({r["section_key"] for r in evidence.values()}),
            "cached_metadata": "Only source identity lookup keys; no page/section full text or all-child content",
            "not_measured": "Index construction cost, graph-edge storage, process RAM, GPU VRAM",
        },
        "lookup_before_update": True, "no_future_cache_content": True,
        "split_cold_start": True, "gold_or_query_type_used_for_cache_selection": False,
        "revision": "unknown/not encoded in current source metadata", "authority": "not inferred from cache key",
        "interpretation": [
            "Identity-support retention is a replay invariant, not retrieval accuracy or expert correctness.",
            "Module reuse saves potential repeated source-module loading/token mass, not measured GPU prefill or KV hits.",
            "Page/section indices add independent metadata capacity. Their hits imply routing-node familiarity, not child evidence support.",
            "Repeated request orders are workload realizations, not additional independent questions.",
            "Retrieval ranking is replayed as frozen; memory policies cannot improve or degrade gold retrieval coverage in this design.",
        ],
    }
    (args.out/"replay_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if not all(assertions.values()):
        raise AssertionError(f"Replay validation failed: {assertions}")
    print(json.dumps({"status": "pass", "out": str(args.out), "rows": len(rows), "coverage": manifest["coverage_counts"], "validation": assertions}, ensure_ascii=False))


if __name__ == "__main__":
    main()
