"""Audit and summarize actual engine counters; paired timings are non-isolated."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1]


def char_bigram_f1(left: str, right: str) -> float:
    def grams(text: str) -> Counter:
        text = re.sub(r"\s+", "", text)
        return Counter(text[i:i+2] for i in range(max(0, len(text)-1)))
    a, b = grams(left), grams(right)
    denominator = sum(a.values()) + sum(b.values())
    return 2 * sum((a & b).values()) / denominator if denominator else float(left == right)


def grouped(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    rows = []
    for values, group in frame.groupby(keys, sort=True):
        values = values if isinstance(values, tuple) else (values,)
        row = dict(zip(keys, values))
        row.update({
            "n_profiles": len(group), "n_pairs": group.pair_index.nunique(),
            "mean_native_prompt_tokens": group.prompt_token_count_native.mean(),
            "mean_reused_prompt_tokens": group.warm_actual_reused_prompt_tokens.mean(),
            "mean_newly_evaluated_prompt_tokens": group.warm_timing_prompt_n.mean(),
            "mean_cache_fraction": group.warm_prompt_cache_fraction.mean(),
            "token_weighted_cache_fraction": group.warm_actual_reused_prompt_tokens.sum() / group.prompt_token_count_native.sum(),
            "mean_shared_evidence_id_count": group.shared_evidence_count.mean(),
            "exact_cold_warm_text_match_fraction": group.outputs_identical.mean(),
            "mean_char_bigram_f1_cold_warm": group.char_bigram_f1_cold_warm.mean(),
            "empty_answer_profile_fraction": group.answer_empty.mean(),
            "mean_cold_prefill_ms_under_contention": group.cold_timing_prompt_ms.mean(),
            "mean_warm_prefill_ms_under_contention": group.warm_timing_prompt_ms.mean(),
            "median_prefill_ratio_under_contention": group.prefill_speedup_ratio.median(),
            "faster_warm_prefill_fraction_under_contention": (group.prefill_ms_saved > 0).mean(),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=OUT / "results" / "real_kv")
    args = parser.parse_args()
    frame = pd.read_csv(args.results / "kv_pairs.csv").fillna("")
    inputs = [json.loads(line) for line in (OUT / "data" / "real_kv_prompts.jsonl").read_text(encoding="utf-8").splitlines() if line]
    assert len(frame) == len(inputs) == 594, "Full, fixed 6 pair x33 method x3 layout protocol required."
    assert frame.row_index.tolist() == list(range(594)), "Missing/reordered result rows."
    for key in ("outputs_identical", "cold_token_accounting_conserved", "warm_token_accounting_conserved", "cold_cache_zero", "warm_reuse_bounded_by_seed_lcp", "warm_reuse_bounded_by_seed_context_lcp"):
        frame[key] = frame[key].map(lambda value: str(value).lower() == "true")
    assert frame.cold_token_accounting_conserved.all() and frame.warm_token_accounting_conserved.all(), "Native token-accounting failed."
    assert frame.cold_cache_zero.all(), "A cold request reused prior state."
    frame["pair_index"] = [row["pair_index"] for row in inputs]
    frame["aircraft_model"] = [row["aircraft_model"] for row in inputs]
    frame["topic"] = [row["topic"] for row in inputs]
    frame["shared_evidence_count"] = [len(row["shared_evidence_ids"]) for row in inputs]
    assert all(frame.iloc[i].method == inputs[i]["method"] and frame.iloc[i].namespace == inputs[i]["namespace"] for i in range(len(frame)))
    frame["char_bigram_f1_cold_warm"] = [char_bigram_f1(str(row.cold_answer), str(row.warm_answer)) for row in frame.itertuples()]
    frame["answer_empty"] = [(not str(row.cold_answer).strip()) or (not str(row.warm_answer).strip()) for row in frame.itertuples()]
    layout = grouped(frame, ["namespace"])
    method = grouped(frame, ["method", "namespace"])
    aircraft = grouped(frame, ["aircraft_model", "namespace"])
    pair = grouped(frame, ["pair_index", "aircraft_model", "namespace"])
    for name, table in (("layout", layout), ("method_layout", method), ("aircraft_layout", aircraft), ("pair_layout", pair)):
        table.to_csv(args.results / f"kv_summary_{name}.csv", index=False, encoding="utf-8-sig")
    rng = np.random.default_rng(20261014)
    pivot = pair.pivot(index="pair_index", columns="namespace", values="mean_cache_fraction")
    contrasts = []
    for name, left, right in (("canonical_minus_query", "canonical_state_first", "query_first"),
            ("hierarchical_minus_query", "hierarchical_state_first", "query_first"),
            ("hierarchical_minus_canonical", "hierarchical_state_first", "canonical_state_first")):
        values = (pivot[left] - pivot[right]).to_numpy()
        draws = values[rng.integers(0, len(values), size=(4000, len(values)))].mean(axis=1)
        contrasts.append({"contrast": name, "n_independent_pairs": len(values), "mean_paired_cache_fraction_delta": float(values.mean()),
            "pair_cluster_bootstrap_ci_low": float(np.quantile(draws, .025)), "pair_cluster_bootstrap_ci_high": float(np.quantile(draws, .975))})
    pd.DataFrame(contrasts).to_csv(args.results / "kv_paired_layout_contrasts.csv", index=False, encoding="utf-8-sig")
    report = {
        "complete": True, "n_profiles": len(frame), "n_methods": frame.method.nunique(), "n_layouts": frame.namespace.nunique(), "n_query_pairs": frame.pair_index.nunique(),
        "all_cold_and_warm_token_accounting_conserved": True, "all_cold_cache_zero": True,
        "warm_prompt_lcp_bound_violation_count": int((~frame.warm_reuse_bounded_by_seed_lcp).sum()),
        "warm_seed_context_lcp_bound_violation_count": int((~frame.warm_reuse_bounded_by_seed_context_lcp).sum()),
        "layout_summaries": layout.to_dict("records"), "pair_cluster_contrasts": contrasts,
        "global_ram_cache_disabled": True,
        "limitations": [
            "GPU Qwen 0.5B decoder; 32-token generation probe. This is not full-answer correctness evaluation.",
            "Only 6 common query pairs. The 33 methods are repeated measurements within pairs, not 198 independent query samples.",
            "cache_n includes common system/scope/metadata tokens as well as any evidence text; it is not evidence-only reuse.",
            "Qwen 3B QA runs concurrently on the GPU. Paired prefill timing cannot establish isolated throughput or uncontended speedup.",
            "Revision and action-authority fields are unverified; namespace is a layout label, not validated admissibility-signature purity.",
            "The hierarchical ancestor index varies with retrieved nodes and may break an otherwise shared leaf prefix.",
            "Cold/warm exact-text differences are observed computational stability diagnostics, not semantic or safety judgments.",
        ],
    }
    (args.results / "kv_summary_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
