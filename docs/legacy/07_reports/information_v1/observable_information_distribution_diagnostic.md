# Observable information-distribution diagnostic v1

All reported distributions are normalized empirical counts, visible-text lexicon matches, word-TFIDF affinity masses, or rank-discounted package masses. Entropy, Jensen-Shannon divergence, observable alignment shift, and fixed-package leave-one-out values describe these estimators only. They are not calibrated epistemic probabilities, mutual information, true information gain, causal effects, answer faithfulness, or aircraft-safety evidence. Locked reference roles and query types are offline, source-derived, construction-coupled audit sensitivity variables and never feed retrieval.

## Estimator contract

The primary role-space analysis uses four normalized masses on the same repository role support: corpus-count prior `p_D`, visible-question bilingual-lexicon demand `p_q`, rank-discounted fair package `p_R`, and uniformly counted locked accepted-evidence reference `p_G`. The source-document sensitivity replaces the visible lexicon with non-negative word-TFIDF affinity mass. The generator is absent.

`H_norm(p) = H_2(p)/log2(m)`, `G_align_obs = JS(p_q,p_D)-JS(p_q,p_R)`, and `Delta_i = JS(p_q,p_R_without_i)-JS(p_q,p_R)` for fixed-package leave-one-out. All logarithms are base 2.

## Primary balanced-IntentRAG role result

Across 214 cue- and track-free synthetic source-derived queries, mean package normalized entropy was 0.4990. Mean package-to-repository selection shift was 0.3341; mean package-to-visible-demand mismatch was 0.5319. The observable alignment shift was -0.0459 (paired-query bootstrap 95% interval -0.0756 to -0.0153).

The shift was non-positive for 64.0% of queries, and 53.7% of defined item-level leave-one-out contributions were negative. These negative observations are retained: selection can concentrate a package without moving it closer to the visible-demand estimator, and some selected items make the realized rank-mass profile less aligned.

The visible lexicon fell back to `general` for 7.5% of queries. Mean divergence from the locked query-type role sensitivity was 0.3114; the sensitivity is construction-coupled and never enters retrieval.

## Method comparison on the role axis

| method                |   n_queries |   mean_package_k |   mean_hnorm_package |   mean_jsd_package_repository |   mean_jsd_package_query_demand |   mean_observable_alignment_shift |   mean_jsd_package_reference |   mean_loo_alignment_positive_rate |
|:----------------------|------------:|-----------------:|---------------------:|------------------------------:|--------------------------------:|----------------------------------:|-----------------------------:|-----------------------------------:|
| Char TF-IDF           |         214 |         10       |             0.443233 |                      0.339394 |                        0.535274 |                        -0.0493445 |                     0.394338 |                           0.296729 |
| Dense + cross-encoder |         214 |         10       |             0.385426 |                      0.362138 |                        0.546658 |                        -0.0607289 |                     0.456793 |                           0.293458 |
| Dense multilingual    |         214 |         10       |             0.379158 |                      0.342787 |                        0.526676 |                        -0.0407474 |                     0.466048 |                           0.306542 |
| Hybrid lexical        |         214 |         10       |             0.420515 |                      0.356334 |                        0.542497 |                        -0.0565674 |                     0.380977 |                           0.300935 |
| IntentRAG (balanced)  |         214 |         10       |             0.498976 |                      0.334079 |                        0.531863 |                        -0.045934  |                     0.360208 |                           0.357477 |
| Original-order window |         214 |          8.95794 |             0.461941 |                      0.297861 |                        0.499381 |                        -0.013452  |                     0.477599 |                           0.406909 |
| Word TF-IDF           |         214 |         10       |             0.42474  |                      0.340428 |                        0.531348 |                        -0.0454193 |                     0.364253 |                           0.288318 |

The comparison is descriptive. A lower divergence or higher alignment shift does not by itself imply a better answer, safer action, or more admissible claim.

## Source-document sensitivity

Cessna 172 has three candidate source documents. For balanced IntentRAG, mean source-axis JS(p_R,p_D) was 0.2583, source-demand mismatch was 0.2117, and observable source alignment shift was -0.1983. Boeing 737 has one ingested source document, so its source entropy and source divergences are structurally zero and carry no discrimination.

## Locked benchmark demand mix (offline description only)

| category                      |   count |      mass |   normalized_entropy |
|:------------------------------|--------:|----------:|---------------------:|
| authority_or_manual_reference |      11 | 0.0514019 |             0.820753 |
| component_to_failure          |      24 | 0.11215   |             0.820753 |
| condition_to_action           |      69 | 0.32243   |             0.820753 |
| cross_page_evidence_query     |      12 | 0.0560748 |             0.820753 |
| excluded_cause                |       4 | 0.0186916 |             0.820753 |
| multi_hop_diagnosis           |      10 | 0.046729  |             0.820753 |
| symptom_to_action             |      55 | 0.257009  |             0.820753 |
| symptom_to_cause              |      15 | 0.0700935 |             0.820753 |
| threshold_query               |       3 | 0.0140187 |             0.820753 |
| warning_or_limitation         |      11 | 0.0514019 |             0.820753 |

## Leave-one-out boundary

There are 2140 defined role-axis item removals for balanced IntentRAG. A positive Delta_i means removal worsens `p_q`/package alignment under fixed original rank weights; a negative Delta_i means removal improves it. Retrieval is not rerun, so Delta_i is not a Shapley value, causal effect, entailment score, or proof that an item is dispensable.

## Claim map

| claim_id   | claim                                                                                                                                        | support_status                                       | metric                                                  |      value | caveat                                                                                                                                              |
|:-----------|:---------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------|:--------------------------------------------------------|-----------:|:----------------------------------------------------------------------------------------------------------------------------------------------------|
| I1         | Repository, visible-query-demand, fair-package, and locked-reference category masses can be reconstructed as explicit normalized estimators. | Supported as an observable estimator audit           | distribution normalization and audit coverage           |   1        | Masses are empirical counts, lexicon matches, TF-IDF affinity, or rank-discounted package weights; they are not calibrated epistemic probabilities. |
| I2         | The balanced IntentRAG package changes role alignment relative to the repository prior under the visible-text demand estimator.              | Non-positive observable shift                        | mean observable alignment shift JS(p_q,p_D)-JS(p_q,p_R) |  -0.045934 | Bootstrap interval [-0.0756, -0.0153]; this is not information gain or answer correctness.                                                          |
| I3         | The realized package exhibits measurable selection shift from the candidate-repository role prior.                                           | Supported descriptively                              | mean JS(p_R,p_D)                                        |   0.334079 | A distribution shift is not evidence of harmful bias, causal mechanism, or improved support.                                                        |
| I4         | Individual items have positive, zero, or negative leave-one-out alignment contribution inside the fixed realized package.                    | Supported as a post-hoc set-function diagnostic      | fraction of positive item contributions                 |   0.357477 | Retrieval is not rerun and ranks are not reassigned; values are not causal effects or evidence entailment scores.                                   |
| I5         | The fair package can be compared with the locked source-derived accepted-evidence role distribution after retrieval.                         | Supported only as construction-coupled offline audit | mean JS(p_R,p_G)                                        |   0.360208 | Accepted alternatives and required slots are source-derived and construction-coupled; p_G never enters retrieval.                                   |
| I6         | The audit estimates mutual information or true information gain between a query and external knowledge.                                      | Unsupported / not estimated                          | latent joint-probability information quantity           | nan        | No calibrated joint distribution or intervention-based causal estimator exists in this run.                                                         |
| I7         | An LLM supplies an independent knowledge source or is responsible for the reported distribution changes.                                     | Unsupported / outside this run                       | generator intervention                                  | nan        | No generator is run. All estimators stop at evidence-package construction, keeping any LLM downstream.                                              |
| I8         | Observed entropy or divergence causally determines accepted retrieval recall.                                                                | Unsupported; exploratory associations only           | causal effect                                           | nan        | Queries are synthetic and source-derived, estimators share the realized package, and no randomized selection mechanism is present.                  |
