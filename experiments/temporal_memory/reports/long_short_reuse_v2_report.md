# Offline long/short-state reuse simulator v2 strict report

Adds `component_document_signature_gated_entropy` without overwriting v1 outputs.
The strict policy uses only current fresh-package metadata: aircraft, authority, condition, warning, non-empty component_family compatibility, and Cessna source-document compatibility.

| policy | accepted@10 | reused slots | fresh slots |
|---|---:|---:|---:|
| admissibility_signature_gated_entropy | 0.7946 | 1.30 | 8.70 |
| component_document_signature_gated_entropy | 0.7946 | 1.30 | 8.70 |
| entropy_adaptive | 0.7914 | 1.64 | 8.36 |
| fixed_50_50 | 0.7598 | 4.95 | 5.05 |
| fresh_only | 0.7991 | 0.00 | 10.00 |
| naive_semantic_reuse | 0.3290 | 9.91 | 0.09 |

## Leakage boundary

- Selection input columns: `aircraft_model, base_question, query_text, question_id, top_evidence_ids`.
- Forbidden selection columns intersect used columns: `none`.
- Gold/accepted/page/section labels are post-hoc scoring only.
- `track`, `query_type`, `required_slots`, and offline metrics are not used by any policy.

## Boundary

Offline reuse simulator over existing cue-free IntentRAG packages; not real KV-cache hits, not live retrieval, not maintenance correctness.
