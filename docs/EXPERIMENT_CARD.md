# Experiment Card

## Retrieval benchmark

- Records: 690
- Methods: 33
- Complete matrix: 22,770 method-record outputs
- Fresh test: 214 source-derived questions
- Primary retrieval endpoint: accepted evidence Hit@10
- Evidence-coverage endpoint: complete required-quote support

Selected fresh-test results:

| Method | Accepted Hit@10 | Complete required-quote support |
|---|---:|---:|
| True BM25 | 0.972 | 0.916 |
| IntentRAG balanced fair | 0.949 | 0.902 |
| IntentRAG multi + mMARCO CE@50 | 0.930 | 0.869 |
| Dense + mMARCO CE@50 | 0.827 | 0.780 |

The multi-view candidate pool improves the matched dense neural route in this corpus but does not exceed the strongest lexical method.

## Multi-granularity evidence memory

Detail and Page are treated as identity-preserving query-local views. Section and Profile are persistent navigation views that must resolve back to traceable source atoms before supporting a claim.

Long-term fixed retrieval reaches 0.248 accepted Hit@10 and Profile-resolved retrieval 0.019 on the fresh endpoint. Their recurrence therefore cannot be interpreted as sufficient query-local evidence.

## Memory replay

Exact-source LRU replay is provided under both module-count and source-token capacity controls. Token-budget replay uses 1,024, 4,096 and 16,384 complete source-body tokens, starts every split cold and refuses oversize admission. It is a documentary-state recurrence experiment, not GPU KV-cache measurement.

## Generated-output mechanics

- Questions: 64 (32 legacy, 32 fresh)
- Methods: 33
- Cells: 2,112
- Generator: local Qwen2.5 3B configuration
- Successful API attempts: 1,933
- Exact-payload aliases: 179
- Retained transport failure: one

JSON parsing, citation-ID presence and verbatim quote matching are mechanical endpoints. Across all cells, only 11 answers contained a parser-recognized quotation and eight matched supplied source text.

## Semantic audit status

The local 7B judge returned zero model responses because Windows blocked the socket with error 10013. The audit is marked blocked and its score fields must not be used. This release reports no semantic answer accuracy.

## Actual GPU input-KV reuse

- Profiles: 594
- Independent visible-query pairs: six
- Methods: 33
- Layouts: query-first, canonical state-first, hierarchical metadata first
- Measurement: engine-reported reused input tokens in a fixed slot

| Layout | Mean reused-input fraction |
|---|---:|
| Query first | 0.039 |
| Canonical state first | 0.125 |
| Hierarchy metadata first | 0.052 |

Canonical state-first increases measured prefix reuse while reducing materialized reference-anchor visibility from 0.721 to 0.593. The source bodies for canonical and hierarchical layouts are identical in the paired profiles; layout differences arise from leading metadata and total input structure.

## Interpretation boundary

Measured computational reuse is not admissible evidence reuse. A compatibility signature can restrict cache reuse to observed equivalence classes, but it cannot create missing authority labels, resolve stale revisions or prove that a maintenance action is authorized.

