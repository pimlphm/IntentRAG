| diagnostic                                             |     value | interpretive_boundary                                                         |
|:-------------------------------------------------------|----------:|:------------------------------------------------------------------------------|
| role_alignment_shift_mean                              | -0.045934 | 95% bootstrap interval [-0.075648, -0.015269]                                 |
| role_queries_with_nonpositive_alignment_shift_fraction |  0.640187 | balanced IntentRAG; retained without selective filtering                      |
| role_loo_negative_item_fraction                        |  0.536916 | fixed package; no retrieval rerun                                             |
| role_loo_zero_item_fraction                            |  0.107477 | fixed package; no retrieval rerun                                             |
| visible_vs_locked_role_sensitivity_jsd_mean            |  0.311364 | locked query type is offline and construction-coupled                         |
| package_vs_locked_reference_role_jsd_mean              |  0.360208 | p_G is accepted-evidence-derived and post-hoc only                            |
| cessna_source_alignment_shift_mean                     | -0.198253 | three-document source axis; word-TFIDF source-affinity demand proxy           |
| boeing_source_axis_category_count                      |  1        | degenerate single-document source axis; entropy and JSD are structurally zero |