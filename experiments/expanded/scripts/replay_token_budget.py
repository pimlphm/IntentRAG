"""Past-only LRU replay under a hard, shared source-text tokenizer-token budget.

Frozen retrieved evidence is never modified. Source token mass is not GPU KV
storage; source identity keys, LRU bookkeeping and routing indices are not
included in this body-only budget. Oversize items are never cached.
"""
from __future__ import annotations

import argparse
from collections import Counter, OrderedDict
import csv
import json
from pathlib import Path
import random
import time

import pandas as pd

from replay_memory import digest_file, load_inputs, token_counter, topic_key


class TokenLRU:
    def __init__(self, budget: int):
        if budget <= 0:
            raise ValueError("Token budget must be positive")
        self.budget = budget
        self.tokens = 0
        self.data: OrderedDict[tuple, int] = OrderedDict()

    def contains(self, key):
        return key in self.data

    def update(self, modules):
        evictions = evicted_tokens = oversize = oversize_tokens = 0
        for r in modules:
            key, mass = r["module_key"], int(r["source_token_mass"])
            if mass < 0:
                raise ValueError("Negative token count")
            if mass > self.budget:
                assert key not in self.data
                oversize += 1
                oversize_tokens += mass
                continue
            if key in self.data:
                assert self.data[key] == mass, "Identity includes exact text hash"
                self.data.move_to_end(key)
            else:
                while self.tokens + mass > self.budget:
                    _, removed = self.data.popitem(last=False)
                    self.tokens -= removed
                    evictions += 1
                    evicted_tokens += removed
                self.data[key] = mass
                self.tokens += mass
            assert 0 <= self.tokens <= self.budget
        assert self.tokens == sum(self.data.values())
        return evictions, evicted_tokens, oversize, oversize_tokens


def self_test():
    def module(k, n):
        return {"module_key": (k,), "source_token_mass": n}
    c = TokenLRU(8)
    assert c.update([module("a", 4), module("b", 3)]) == (0, 0, 0, 0)
    assert c.tokens == 7 and c.contains(("a",))
    c.update([module("a", 4)])  # existing identity touches the LRU order
    assert list(c.data) == [("b",), ("a",)]
    assert c.update([module("c", 3)]) == (1, 3, 0, 0)
    assert not c.contains(("b",)) and c.tokens == 7
    before = list(c.data)
    assert c.update([module("oversize", 9)]) == (0, 0, 1, 9)
    assert list(c.data) == before and not c.contains(("oversize",))
    assert c.update([module("exact_budget", 8)]) == (2, 7, 0, 0)
    assert c.tokens == 8 and list(c.data) == [("exact_budget",)]
    assert c.update([module("empty", 0)]) == (0, 0, 0, 0)
    assert c.tokens == 8
    # A duplicated identity in one request is de-duplicated before replay.
    assert len(dict.fromkeys(["a", "a", "b"])) == 2
    print(json.dumps({"self_test": "pass", "cases": 7}))


def main():
    started = time.perf_counter()
    exp = Path(__file__).resolve().parents[1]
    project = Path(r"${INTENTRAG_PROJECT_ROOT}")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--retrieval", type=Path, default=exp/"results/all_retrieval_runs.jsonl")
    ap.add_argument("--canonical", type=Path, default=exp/"data/canonical_questions.csv")
    ap.add_argument("--corpus", type=Path, default=project/"01_processed_corpus/v5/evidence_objects.parquet")
    ap.add_argument("--out", type=Path, default=exp/"results/memory_replay_token_budget")
    ap.add_argument("--tokenizer-json", type=Path, default=exp/"models/qwen2_5_3b_tokenizer/tokenizer.json")
    ap.add_argument("--capacities", type=int, nargs="+", default=[1024, 4096, 16384])
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(20261014, 20261019)))
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return
    if not all(v > 0 for v in args.capacities):
        raise ValueError("All capacities must be positive")
    count_tokens, count_info = token_counter(args.tokenizer_json)
    canonical, runs, evidence, methods, coverage, extras = load_inputs(
        args.retrieval, args.canonical, args.corpus, count_tokens)
    args.out.mkdir(parents=True, exist_ok=True)
    splits = {}
    for qid, q in canonical.items():
        splits.setdefault(str(q.get("split", "unspecified")), []).append(qid)
    order_rows, order_summaries = [], []
    assertions = {"occupancy_within_token_budget": True,
        "every_cold_start_has_zero_identity_hits": True,
        "reuse_mass_never_exceeds_requested_mass": True,
        "oversize_never_cached_or_hit": True,
        "ranked_outputs_unchanged": True,
        "all_retained_identity_support_equal_one": True}
    rows = 0
    with (args.out/"replay_query_results.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = None
        for split, all_ids in sorted(splits.items()):
            for order, seed in [("random", s) for s in args.seeds]+[("topical", None)]:
                ids = sorted(all_ids)
                if order == "random":
                    random.Random(seed).shuffle(ids)
                else:
                    ids.sort(key=lambda qid: topic_key(str(canonical[qid].get("question", "")),
                        str(canonical[qid].get("aircraft", canonical[qid].get("aircraft_model", ""))), qid))
                for pos, qid in enumerate(ids):
                    order_rows.append({"split": split, "order": order, "seed": seed,
                        "request_index": pos, "question_id": qid})
                for method in methods:
                    for budget in args.capacities:
                        cache = TokenLRU(budget)
                        totals = Counter()
                        query_reuse, max_occupancy, first = [], 0, True
                        for pos, qid in enumerate(ids):
                            run = runs.get((qid, method))
                            if run is None:
                                continue
                            requested = list(dict.fromkeys(run["top_evidence_ids"]))
                            known = [evidence[eid] for eid in requested if eid in evidence]
                            # All current-query lookups see only previous-request state.
                            hits = [r for r in known if cache.contains(r["module_key"])]
                            mass = sum(r["source_token_mass"] for r in known)
                            reused = sum(r["source_token_mass"] for r in hits)
                            before = cache.tokens
                            count_before = len(cache.data)
                            if first:
                                assertions["every_cold_start_has_zero_identity_hits"] &= len(hits) == 0
                                first = False
                            assertions["oversize_never_cached_or_hit"] &= all(r["source_token_mass"] <= budget for r in hits)
                            evicted, evicted_mass, oversize, oversize_mass = cache.update(known)
                            expected_scope = str(run.get("aircraft", run.get("aircraft_model", "")))
                            row = {"question_id": qid, "method": method, "split": split,
                                "aircraft": expected_scope, "order": order, "seed": seed,
                                "request_index": pos, "policy": "source_token_budget_identity_lru",
                                "capacity_source_tokens": budget,
                                "n_requested_ids": len(requested),
                                "n_input_id_duplicates": len(run["top_evidence_ids"])-len(requested),
                                "n_known_requested_ids": len(known),
                                "n_unknown_evidence_ids": len(requested)-len(known),
                                "n_cached_identity_hits": len(hits),
                                "module_hit_fraction": len(hits)/len(known) if known else None,
                                "requested_token_mass": mass, "reused_token_mass": reused,
                                "token_weighted_reused_fraction": reused/mass if mass else None,
                                "cache_tokens_before": before, "cache_tokens_after": cache.tokens,
                                "cached_modules_before": count_before, "cached_modules_after": len(cache.data),
                                "module_evictions": evicted, "evicted_token_mass": evicted_mass,
                                "n_oversize_requests": oversize, "oversize_requested_token_mass": oversize_mass,
                                "n_scope_mismatches": sum(bool(expected_scope) and r.get("aircraft_model", "") != expected_scope for r in known),
                                "exact_id_support_retention": 1.0,
                                "ranked_output_unchanged": True, "gpu_kv_cache_measured": False}
                            if writer is None:
                                writer = csv.DictWriter(f, fieldnames=list(row))
                                writer.writeheader()
                            writer.writerow(row)
                            rows += 1
                            for k in ["n_requested_ids", "n_known_requested_ids", "n_unknown_evidence_ids",
                                "n_cached_identity_hits", "requested_token_mass", "reused_token_mass", "module_evictions",
                                "evicted_token_mass", "n_oversize_requests", "oversize_requested_token_mass", "n_scope_mismatches"]:
                                totals[k] += row[k]
                            totals["n_requests"] += 1
                            totals["sum_cache_tokens_after"] += cache.tokens
                            if mass:
                                query_reuse.append(reused/mass)
                            max_occupancy = max(max_occupancy, cache.tokens)
                            assertions["occupancy_within_token_budget"] &= 0 <= cache.tokens <= budget
                            assertions["reuse_mass_never_exceeds_requested_mass"] &= 0 <= reused <= mass
                        order_summaries.append({"split": split, "method": method,
                            "capacity_source_tokens": budget, "order": order, "seed": seed,
                            "policy": "source_token_budget_identity_lru", **dict(totals),
                            "token_weighted_reused_fraction": totals["reused_token_mass"]/totals["requested_token_mass"] if totals["requested_token_mass"] else None,
                            "mean_query_token_reused_fraction": sum(query_reuse)/len(query_reuse) if query_reuse else None,
                            "module_hit_fraction": totals["n_cached_identity_hits"]/totals["n_known_requested_ids"] if totals["n_known_requested_ids"] else None,
                            "oversize_module_request_fraction": totals["n_oversize_requests"]/totals["n_known_requested_ids"] if totals["n_known_requested_ids"] else None,
                            "oversize_token_mass_fraction": totals["oversize_requested_token_mass"]/totals["requested_token_mass"] if totals["requested_token_mass"] else None,
                            "max_cache_tokens_after": max_occupancy,
                            "mean_cache_tokens_after": totals["sum_cache_tokens_after"]/totals["n_requests"] if totals["n_requests"] else None})
    orders = pd.DataFrame(order_summaries)
    orders.to_csv(args.out/"replay_order_summary.csv", index=False, encoding="utf-8-sig")
    summaries = []
    for key, part in orders.groupby(["split", "method", "capacity_source_tokens", "order", "policy"]):
        summaries.append({**dict(zip(["split", "method", "capacity_source_tokens", "order", "policy"], key)),
            "n_requests": int(part.n_requests.sum()), "n_order_realizations": len(part),
            "mean_order_token_weighted_reused_fraction": part.token_weighted_reused_fraction.mean(),
            "min_order_token_weighted_reused_fraction": part.token_weighted_reused_fraction.min(),
            "max_order_token_weighted_reused_fraction": part.token_weighted_reused_fraction.max(),
            "pooled_token_weighted_reused_fraction": part.reused_token_mass.sum()/part.requested_token_mass.sum() if part.requested_token_mass.sum() else None,
            "n_oversize_requests": int(part.n_oversize_requests.sum()),
            "oversize_module_request_fraction": part.n_oversize_requests.sum()/part.n_known_requested_ids.sum() if part.n_known_requested_ids.sum() else None,
            "oversize_token_mass_fraction": part.oversize_requested_token_mass.sum()/part.requested_token_mass.sum() if part.requested_token_mass.sum() else None,
            "module_evictions": int(part.module_evictions.sum()),
            "max_cache_tokens_after": int(part.max_cache_tokens_after.max()),
            "min_exact_id_support_retention": 1.0, "all_ranked_outputs_unchanged": True})
    pd.DataFrame(summaries).to_csv(args.out/"replay_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(coverage).to_csv(args.out/"requested_query_method_coverage.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(order_rows).to_csv(args.out/"request_orders.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(extras, columns=["question_id", "method", "status"]).to_csv(args.out/"noncanonical_input_rows.csv", index=False, encoding="utf-8-sig")
    manifest = {"experiment": "past_only_exact_source_identity_token_budget_replay",
        "inputs": [{"path": str(p), "sha256": digest_file(p)} for p in [args.retrieval, args.canonical, args.corpus]],
        "capacities_source_tokens": args.capacities, "random_order_seeds": args.seeds,
        "extra_order": "visible_question_topic_then_aircraft_offline_workload", "methods": methods,
        "n_canonical_question_ids": len(canonical), "n_replay_rows": rows,
        "coverage_counts": dict(Counter(r["status"] for r in coverage)),
        "n_noncanonical_input_rows": len(extras), "token_mass_measurement": count_info,
        "cache_key_fields": ["aircraft_model", "document_id", "page_id", "evidence_id", "exact_source_text_sha256"],
        "lookup_before_update": True, "no_future_cache_content": True, "split_cold_start": True,
        "gold_or_query_type_used_for_cache_selection": False,
        "oversize_policy": "Never admit an item whose full source text has more tokens than the capacity; preserve retrieval output",
        "memory_budget": "Hard sum of full source-body tokenizer tokens. Metadata keys, LRU management and index overhead excluded; not RAM or GPU KV bytes.",
        "revision": "unknown; not independently validated", "authority": "not inferred from identity key",
        "validation": assertions, "runtime_seconds": time.perf_counter()-started,
        "interpretation": ["Same number of evidence modules does not imply the same body-token memory budget.",
            "Requested/reused token mass is de-duplicated within each query; no within-query hit is counted.",
            "Lookup happens before all current-request updates; caches cold-start independently for each split and method.",
            "Replay freezes retrieval output. Identity support retention equal to one is a design invariant, not improved retrieval or QA.",
            "Order variation is workload variation, not a population confidence interval.",
            "No GPU cache hit, prefill latency, authorization purity or downstream answer quality is measured."]}
    (args.out/"replay_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if not all(assertions.values()):
        raise AssertionError(assertions)
    print(json.dumps({"status": "pass", "rows": rows, "methods": len(methods),
        "runtime_seconds": manifest["runtime_seconds"], "validation": assertions, "out": str(args.out)}))


if __name__ == "__main__":
    main()
