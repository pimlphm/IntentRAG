# Offline long/short-state reuse simulator report

This run simulates online reuse over 214 cue-free IntentRAG Top-10 packages.
Short-term state is the fresh current package; long-term state is the cache of earlier query evidence-state packages under a randomized query order.

## Policies

- `fresh_only`: always use current cue-free IntentRAG Top-10.
- `naive_semantic_reuse`: fill up to 10 slots from previous cached evidence ranked by visible-query/evidence semantic overlap.
- `fixed_50_50`: fill up to 5 cached slots, then fresh slots.
- `entropy_adaptive`: choose cached-slot budget from visible-query/fresh-package entropy and cache semantic quality.
- `admissibility_signature_gated_entropy`: entropy-adaptive reuse after aircraft/authority/condition/warning and semantic compatibility gates.

## Summary over seed means

| policy | accepted@10 | page@10 | section@10 | reused slots | fresh slots | accepted selected |
|---|---:|---:|---:|---:|---:|---:|
| admissibility_signature_gated_entropy | 0.7946 | 0.7991 | 0.8005 | 1.30 | 8.70 | 2.273 |
| entropy_adaptive | 0.7914 | 0.7958 | 0.7974 | 1.64 | 8.36 | 2.254 |
| fixed_50_50 | 0.7598 | 0.7907 | 0.7951 | 4.95 | 5.05 | 2.059 |
| fresh_only | 0.7991 | 0.7991 | 0.7991 | 0.00 | 10.00 | 2.313 |
| naive_semantic_reuse | 0.3290 | 0.3449 | 0.3682 | 9.91 | 0.09 | 0.721 |

## Bootstrap / CI

CI is percentile bootstrap over the 20 fixed seed-level means, not over independent natural user traffic.

| policy | metric | mean | 95% CI |
|---|---|---:|---:|
| admissibility_signature_gated_entropy | accepted_hit_10 | 0.7946 | [0.7944, 0.7951] |
| admissibility_signature_gated_entropy | page_hit_10 | 0.7991 | [0.7991, 0.7991] |
| admissibility_signature_gated_entropy | section_hit_10 | 0.8005 | [0.7995, 0.8014] |
| admissibility_signature_gated_entropy | actual_reuse_slots | 1.3044 | [1.2946, 1.3126] |
| admissibility_signature_gated_entropy | fresh_slots | 8.6956 | [8.6874, 8.7054] |
| admissibility_signature_gated_entropy | n_accepted_selected | 2.2734 | [2.2706, 2.2762] |
| entropy_adaptive | accepted_hit_10 | 0.7914 | [0.7900, 0.7928] |
| entropy_adaptive | page_hit_10 | 0.7958 | [0.7944, 0.7972] |
| entropy_adaptive | section_hit_10 | 0.7974 | [0.7956, 0.7993] |
| entropy_adaptive | actual_reuse_slots | 1.6360 | [1.6304, 1.6416] |
| entropy_adaptive | fresh_slots | 8.3640 | [8.3584, 8.3696] |
| entropy_adaptive | n_accepted_selected | 2.2537 | [2.2505, 2.2575] |
| fixed_50_50 | accepted_hit_10 | 0.7598 | [0.7572, 0.7626] |
| fixed_50_50 | page_hit_10 | 0.7907 | [0.7890, 0.7923] |
| fixed_50_50 | section_hit_10 | 0.7951 | [0.7928, 0.7979] |
| fixed_50_50 | actual_reuse_slots | 4.9533 | [4.9533, 4.9533] |
| fixed_50_50 | fresh_slots | 5.0467 | [5.0467, 5.0467] |
| fixed_50_50 | n_accepted_selected | 2.0586 | [2.0488, 2.0689] |
| fresh_only | accepted_hit_10 | 0.7991 | [0.7991, 0.7991] |
| fresh_only | page_hit_10 | 0.7991 | [0.7991, 0.7991] |
| fresh_only | section_hit_10 | 0.7991 | [0.7991, 0.7991] |
| fresh_only | actual_reuse_slots | 0.0000 | [0.0000, 0.0000] |
| fresh_only | fresh_slots | 10.0000 | [10.0000, 10.0000] |
| fresh_only | n_accepted_selected | 2.3131 | [2.3131, 2.3131] |
| naive_semantic_reuse | accepted_hit_10 | 0.3290 | [0.3224, 0.3360] |
| naive_semantic_reuse | page_hit_10 | 0.3449 | [0.3369, 0.3528] |
| naive_semantic_reuse | section_hit_10 | 0.3682 | [0.3593, 0.3771] |
| naive_semantic_reuse | actual_reuse_slots | 9.9065 | [9.9065, 9.9065] |
| naive_semantic_reuse | fresh_slots | 0.0935 | [0.0935, 0.0935] |
| naive_semantic_reuse | n_accepted_selected | 0.7213 | [0.7068, 0.7355] |

## Leakage statement

- Selection input columns: `aircraft_model, base_question, query_text, question_id, top_evidence_ids`.
- Forbidden selection columns intersect used columns: `none`.
- Cue-free IntentRAG rows: 214.
- Gold/accepted/page/section labels are loaded only after selection for post-hoc hit computation.
- `track`, `query_type`, `required_slots`, and all `gold/accepted/offline metric` fields are not used by policy scoring, slot budgeting, cache ranking, or gates.

## Boundary

This is an offline simulator over existing retrieved packages. It does not rerun a live retriever, measure server-side KV cache, or validate aircraft maintenance correctness.
