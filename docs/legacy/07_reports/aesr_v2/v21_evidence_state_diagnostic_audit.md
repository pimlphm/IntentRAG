# V21 Evidence-State Diagnostic Audit

This run is a diagnostic audit toward AESR. It is not a five-state classifier, an action-authorization validation, a safety assessment, or evidence that AESR is access-agnostic.

## Cue-dependence audit

Across 214 split-held-out, synthetic source-derived queries, removing the appended gold-derived cue while retaining legacy track routing changed accepted R@10 by -0.1869 (paired bootstrap 95% interval -0.2430 to -0.1402). Removing both cue and locked-track routing changed it by -0.1963 (95% interval -0.2523 to -0.1402). The legacy condition is therefore a gold-dependent diagnostic upper reference, not natural cold retrieval.

## Cue-free access settings

| condition                           |   mean_accepted_recall_10 |
|:------------------------------------|--------------------------:|
| cue_off_track_fixed                 |                  0.808411 |
| cue_off_intentrag                   |                  0.799065 |
| cue_off_word_tfidf                  |                  0.766355 |
| cue_off_hybrid_lexical              |                  0.747664 |
| cue_off_char_tfidf                  |                  0.696262 |
| cue_off_dense_cross_encoder         |                  0.453271 |
| cue_off_dense_multilingual          |                  0.453271 |
| cue_off_local_original_order_window |                  0.313084 |

The dense-plus-cross-encoder row is a real multilingual retrieval/reranking run. The local original-order row is only a bounded-window access proxy; no long-context generator was run.

## Unmodified-package factor proxies

The factor table contains 214 unmodified packages retrieved without cue or locked-track routing for synthetic source-derived queries. Mean provenance-anchor validity was 1.0000, mean scope match was 1.0000, and mean pairwise redundancy proxy was 0.2376. The saturated scope and provenance fields reflect a coarse aircraft-family mask and locator availability, not validated effectivity or revision control. These are observable package proxies, not latent-state probabilities.

## Synthetic controlled package interventions

| intervention                           |       mean |   count |
|:---------------------------------------|-----------:|--------:|
| authority_removal                      | 1          |     214 |
| condition_removal                      | 0.586854   |     213 |
| conflict_injection                     | 1          |     214 |
| duplicate_injection                    | 0.953271   |     214 |
| evidence_order_permutation             | 1          |     214 |
| procedural_to_descriptive_substitution | 0.00649351 |     154 |
| provenance_masking                     | 1          |     214 |
| scope_swap                             | 1          |     214 |
| warning_removal                        | 1          |     212 |

Every intervention row is machine-labelled synthetic/counterfactual. Directional response consistency verifies implementation sensitivity of the named proxy only; it is not five-state accuracy or causal evidence. The most important negative result is that procedural-to-descriptive substitution changed its target proxy in only 1 of 154 eligible cases, exposing a weakness in the current lexical procedure proxy.

## Action-related documentary prerequisites

Among 105 unmodified cue- and track-free packages selected by a visible-text action-like query proxy, 0.9810 simultaneously contained an action/procedure cue, an authority cue, matching aircraft scope, a condition cue, and valid provenance. The near-saturated value reflects permissive lexical prerequisites; it is an evidence-availability proxy, not proof that an action is authorized.

## Claim boundary

| claim_id   | manuscript_claim                                                                                                          | support_status                                     | caveat                                                                                                                                                                                           |
|:-----------|:--------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| C1         | Legacy V21 held-out retrieval is materially dependent on gold-derived evidence cues.                                      | Supported                                          | All benchmark questions are template-generated from source evidence. This diagnoses shortcut dependence; it is not a natural-query generalization estimate.                                      |
| C2         | Removing both the appended gold cue and locked evaluation-track routing further defines the fair no-gold-input reference. | Supported                                          | The split groups exact-gold IDs only; accepted/page/section alternatives can still cross splits.                                                                                                 |
| C3         | A real multilingual dense plus cross-encoder setting was evaluated without appended gold cues.                            | Supported                                          | This is an access comparison, not a full long-context generation or AESR evaluation.                                                                                                             |
| C4         | Retriever-visible factor proxies can be audited on unmodified cue- and track-free packages.                               | Supported                                          | These are observable proxies, not calibrated latent knowledge states.                                                                                                                            |
| C5         | Controlled package interventions change their preregistered target proxies in the expected direction.                     | Supported for deterministic proxy sensitivity only | Does not establish five-state classification accuracy or causal mechanism. Response was uneven: procedural-to-descriptive substitution changed its target proxy in only 1 of 154 eligible cases. |
| C6         | Action-like requests can be audited for simultaneous documentary prerequisites.                                           | Partially supported as an availability proxy       | Authority cues do not prove legal or operational authorization; no expert action-authority labels exist, and the heuristic prerequisites are near-saturated.                                     |
| C7         | AESR is empirically access-agnostic across retrieval, long-context, cache, memory, and tools.                             | Unsupported / not evaluated                        | Only retrieval-oriented access settings and a local original-order window proxy were evaluated.                                                                                                  |
| C8         | The system accurately classifies Supported, Insufficient, Inapplicable, Conflicted, and OutOfScope.                       | Unsupported / not evaluated                        | Synthetic interventions report proxy signatures, not state predictions.                                                                                                                          |
| C9         | A GNN captures higher-order evidence dependencies beyond node attributes.                                                 | Unsupported / not implemented                      | The current document-supported graph remains a routing index.                                                                                                                                    |
