# Multi-granularity evidence-memory v1

## What was implemented

Four source-grounded evidence views were indexed: detail and page as query-local working evidence, and section and component profile as persistent repository memory. A normalized-entropy router weights the four views. A separate deterministic assembler reserves evidence roles only from visible query language. Gold, accepted sets, benchmark track, query type, and required slots are excluded from ranking and assembly.

## Locked-blind results

| method                                |   offline_accepted_hit_10_mean |   offline_exact_hit_10_mean |   offline_page_hit_10_mean |   offline_section_hit_10_mean |   package_approx_tokens_mean |   short_term_weight_mean |   long_term_weight_mean |
|:--------------------------------------|-------------------------------:|----------------------------:|---------------------------:|------------------------------:|-----------------------------:|-------------------------:|------------------------:|
| entropy_admissibility_assembly        |                       0.780374 |                    0.751269 |                   0.813084 |                      0.827103 |                      989.061 |                 0.738762 |                0.261238 |
| entropy_multigranularity              |                       0.775701 |                    0.730964 |                   0.803738 |                      0.808411 |                      976.206 |                 0.738762 |                0.261238 |
| existing_cue_free_intentrag_reference |                       0.799065 |                    0.761421 |                   0.799065 |                      0.799065 |                      nan     |               nan        |              nan        |
| fixed_multigranularity                |                       0.757009 |                    0.730964 |                   0.794393 |                      0.799065 |                     1007.08  |                 0.65     |                0.35     |
| flat_detail_hybrid                    |                       0.742991 |                    0.730964 |                   0.780374 |                      0.794393 |                      875.486 |                 1        |                0        |
| long_memory_fixed                     |                       0.490654 |                    0.360406 |                   0.542056 |                      0.579439 |                      973.902 |                 0        |                1        |
| page_only                             |                       0.672897 |                    0.563452 |                   0.686916 |                      0.705607 |                      861.397 |                 1        |                0        |
| profile_only                          |                       0.200935 |                    0.187817 |                   0.205607 |                      0.219626 |                      990.86  |                 0        |                1        |
| section_only                          |                       0.546729 |                    0.436548 |                   0.570093 |                      0.588785 |                      945.551 |                 0        |                1        |
| short_memory_fixed                    |                       0.742991 |                    0.725888 |                   0.766355 |                      0.780374 |                      983.78  |                 1        |                0        |

## Paired comparisons

| metric                  | proposed                       | baseline                              |   n |   mean_delta |      ci_low |   ci_high |   wins |   ties |   losses |
|:------------------------|:-------------------------------|:--------------------------------------|----:|-------------:|------------:|----------:|-------:|-------:|---------:|
| offline_accepted_hit_10 | entropy_admissibility_assembly | flat_detail_hybrid                    | 214 |    0.0373832 |  0          | 0.0747664 |     12 |    198 |        4 |
| offline_accepted_hit_10 | entropy_admissibility_assembly | fixed_multigranularity                | 214 |    0.0233645 |  0          | 0.046729  |      6 |    207 |        1 |
| offline_accepted_hit_10 | entropy_admissibility_assembly | entropy_multigranularity              | 214 |    0.0046729 | -0.00934579 | 0.0186916 |      2 |    211 |        1 |
| offline_accepted_hit_10 | entropy_admissibility_assembly | existing_cue_free_intentrag_reference | 214 |   -0.0186916 | -0.0514019  | 0.0140187 |      5 |    200 |        9 |

## Interpretation boundary

The short/long distinction is a state-lifetime and document-structure hierarchy: detail/page are query-local working evidence; section/component profiles are persistent repository memory. It is not a natural maintenance-event timeline or a deployed conversational memory backend. Entropy is a retrieval-concentration proxy, not safety confidence or mutual information.

The benchmark questions are synthetic and source-derived. Accepted/page/section sets are construction-coupled and are used only post hoc. Results do not establish natural-user performance, expert action authorization, semantic mutual information, calibrated refusal, or an online long-term memory backend.