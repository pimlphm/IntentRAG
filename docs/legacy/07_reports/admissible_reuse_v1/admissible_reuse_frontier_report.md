# Admissible Reuse Frontier Diagnostic

This run converts the earlier cache/reuse observation into a more defensible
scientific object: the difference between computationally reusable state and
evidence that remains compatible with visible admissibility proxies.

## Main numeric findings

- Raw evidence-state reuse exists in the current corpus: all-query mean raw reuse
  ratio proxy = 0.623
  [0.574, 0.672].
- After scope, provenance, authority, condition, and action-prerequisite proxies,
  the all-query retained reuse ratio proxy = 0.242
  [0.195, 0.291].
- Boeing has higher raw reuse in this current corpus:
  0.874 versus Cessna
  0.448. Treat this as a corpus
  and representation effect, not an aircraft-type law.
- Across the local Qwen scale ladder, explicit state-first prompting improves
  answer-groundedness proxy over direct model-only prompting by a mean of
  0.301. This supports the claim that scale did not replace
  project evidence-state construction in this diagnostic.

## Paper-safe interpretation

The contribution should be framed as an admissible-reuse frontier: for
aviation maintenance support, the useful pre-generation object is not just a
retrieved context or a larger model's internal memory, but a reusable evidence
state whose reuse is bounded by scope, provenance, authority, and condition
compatibility. The present outputs are proxy diagnostics over source-derived
queries and retrieved packages; they do not prove layerwise KV-cache hits,
mutual information, or expert action authorization.

## Claim map

| claim_id   | manuscript_claim                                                                                                                                                                                 | support_status                                                              | key_value                                                      | result_file                                                          | figure                                                                | caveat                                                                                                                 |
|:-----------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------|:---------------------------------------------------------------|:---------------------------------------------------------------------|:----------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------|
| ARF1       | The current project contains substantial reusable evidence-state mass, but raw computational reuse and admissible evidence reuse must be separated.                                              | Supported as proxy diagnostic on 214 source-derived V21 queries             | All-query action-prerequisite retained reuse ratio proxy=0.242 | 04_results/admissible_reuse_v1/admissible_reuse_frontier_summary.csv | 05_figures/admissible_reuse_v1/figure_a_admissible_reuse_frontier.png | Proxy over visible package factors; not real layerwise KV cache, not expert action authorization.                      |
| ARF2       | Boeing shows higher raw reuse than Cessna in this corpus because its current evidence packages are more repetitive and concentrated.                                                             | Supported as corpus-specific explanation                                    | Boeing raw=0.874; Cessna raw=0.448                             | 06_tables/admissible_reuse_v1/corpus_reuse_explanation.csv           | 05_figures/admissible_reuse_v1/figure_b_corpus_reuse_explanation.png  | Do not generalize to all Boeing/Cessna manuals; this is a current-corpus representation effect.                        |
| ARF3       | State-first organization creates far more cache-eligible shared-prefix structure than question-first raw RAG chains.                                                                             | Supported as tokenizer-level prefix proxy                                   | State-first shared prefix ratio=0.718; raw chain=0.036         | 06_tables/cache_reuse_v1/variant_summary.csv                         | 05_figures/admissible_reuse_v1/figure_d_cache_vs_admissible_reuse.png | No internal server KV-cache table or layerwise tensor inspection.                                                      |
| ARF4       | Increasing model scale alone did not reconstruct the project-specific evidence state in the tested local Qwen ladder.                                                                            | Supported as controlled local model-scale diagnostic                        | Mean state-minus-direct groundedness proxy=0.301               | 06_tables/admissible_reuse_v1/model_scale_state_reuse_matrix.csv     | 05_figures/admissible_reuse_v1/figure_c_scale_state_reuse_matrix.png  | Local Qwen models, selected source-derived queries, answer-grounding proxies; not a universal RLM impossibility claim. |
| ARF5       | RLM/direct answering, question-first RAG chains, and state-first evidence assembly differ empirically along groundedness, citation availability, prompt burden, and cache-eligible prefix reuse. | Supported as a local diagnostic comparison across existing result artifacts | See paradigm_reuse_comparison.csv                              | 06_tables/admissible_reuse_v1/paradigm_reuse_comparison.csv          | 05_figures/admissible_reuse_v1/figure_f_paradigm_reuse_comparison.png | Direct-model condition is local Qwen only and cannot represent all retrieval-free language models.                     |
