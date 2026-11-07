# FMECA PNG Plotting and Writing Final Summary

## 1. Experimental logic not modified
- No model training, retrieval rerun, QA benchmark regeneration, split editing, test-label tuning, metric rewriting, or algorithm-logic change was performed.
- Existing result CSV files under `04_results/`, benchmark files under `03_benchmark/`, and case retrieval outputs were read only.

## 2. Generated PNG figures
- V21-01: `05_figures/fmeca_png/main_figures/figure_v21_01_method_overview.png`
- V21-02: `05_figures/fmeca_png/main_figures/figure_v21_02_evidence_schema_fmeca.png`
- V21-03: `05_figures/fmeca_png/main_figures/figure_v21_03_candidate_pool_reranking.png`
- V21-04: `05_figures/fmeca_png/result_figures/figure_v21_04_main_performance_forest.png`
- V21-05: `05_figures/fmeca_png/result_figures/figure_v21_05_recall_at_k_curves.png`
- V21-06: `05_figures/fmeca_png/result_figures/figure_v21_06_query_type_fmeca_heatmap.png`
- V21-07: `05_figures/fmeca_png/result_figures/figure_v21_07_evidence_package_completeness.png`
- V21-08: `05_figures/fmeca_png/result_figures/figure_v21_08_maintenance_fmeca_proxy_metrics.png`
- V21-09: `05_figures/fmeca_png/ablation_figures/figure_v21_09_ablation_delta_waterfall.png`
- V21-10: `05_figures/fmeca_png/ablation_figures/figure_v21_10_failure_analysis_matrix.png`
- V21-11: `05_figures/fmeca_png/writing_support/figure_v21_11_claim_boundary_evidence_hierarchy.png`
- V21-12a: `05_figures/fmeca_png/case_studies/case_boeing_success_01.png`
- V21-12b: `05_figures/fmeca_png/case_studies/case_cessna_success_01.png`
- V21-12c: `05_figures/fmeca_png/case_studies/case_fmeca_maintenance_action_01.png`
- V21-12d: `05_figures/fmeca_png/case_studies/case_failure_or_limitation_01.png`

## 3. Case-study folder
- All case-study PNGs are stored together in `05_figures/fmeca_png/case_studies/`.

## 4. Figures suitable for main text
- V21-01: `05_figures/fmeca_png/main_figures/figure_v21_01_method_overview.png`
- V21-02: `05_figures/fmeca_png/main_figures/figure_v21_02_evidence_schema_fmeca.png`
- V21-03: `05_figures/fmeca_png/main_figures/figure_v21_03_candidate_pool_reranking.png`
- V21-04: `05_figures/fmeca_png/result_figures/figure_v21_04_main_performance_forest.png`
- V21-05: `05_figures/fmeca_png/result_figures/figure_v21_05_recall_at_k_curves.png`
- V21-06: `05_figures/fmeca_png/result_figures/figure_v21_06_query_type_fmeca_heatmap.png`
- V21-07: `05_figures/fmeca_png/result_figures/figure_v21_07_evidence_package_completeness.png`
- V21-11: `05_figures/fmeca_png/writing_support/figure_v21_11_claim_boundary_evidence_hierarchy.png`
- V21-12a: `05_figures/fmeca_png/case_studies/case_boeing_success_01.png`
- V21-12b: `05_figures/fmeca_png/case_studies/case_cessna_success_01.png`
- V21-12c: `05_figures/fmeca_png/case_studies/case_fmeca_maintenance_action_01.png`

## 5. Figures suitable for supplementary material
- V21-08: `05_figures/fmeca_png/result_figures/figure_v21_08_maintenance_fmeca_proxy_metrics.png`
- V21-09: `05_figures/fmeca_png/ablation_figures/figure_v21_09_ablation_delta_waterfall.png`
- V21-10: `05_figures/fmeca_png/ablation_figures/figure_v21_10_failure_analysis_matrix.png`
- V21-12d: `05_figures/fmeca_png/case_studies/case_failure_or_limitation_01.png`

## 6. Figures or panels constrained by missing data
- Current evaluation summary files do not contain MRR, nDCG@10, MAP@10, or Recall@20. Figure V21-04 therefore uses available R@10/package/grounding metrics; Figure V21-05 uses Recall@1/3/5/10 from existing top-10 evidence outputs only.
- Current evaluation files do not contain every requested FMECA proxy subfield such as `cause_action_joint_hit`; Figure V21-08 uses available proxy fields from an auxiliary proxy table and labels them as document-grounded proxies.
- The exact requested ablation rows `no role tags`, `no FMECA alignment`, `no query intent parser`, and similar module names are not all present as explicit ablation rows. Figure V21-09 plots only available ablation views.
- No full figure was skipped; some metric panels were limited by available columns as noted above.

## 7. New writing files
- `07_reports/fmeca_model_intro_rewritten.md`
- `paper_sections/fmeca_model_intro_rewritten.tex`
- `07_reports/fmeca_png_figure_captions.md`
- `paper_sections/fmeca_results_narrative.tex`
- `paper_sections/fmeca_discussion_narrative.tex`
- `07_reports/fmeca_claim_boundary_rewritten.md`
- `paper_sections/fmeca_claim_boundary_rewritten.tex`
- `07_reports/fmeca_png_figure_index.md`
- `07_reports/fmeca_png_plotting_and_writing_audit.md`

## 8. Recall@10 claim discipline
- Yes. The writing and captions explicitly state that Recall@10 is an evidence retrieval metric, not real diagnostic accuracy.

## 9. FMECA proxy claim discipline
- Yes. Proxy metrics are explicitly described as document-grounded indicators and not safety certification, FMECA certification, or real maintenance effectiveness.

## 10. Remaining work for stronger journal submission
- Add expert-reviewed maintenance decision evaluation separate from retrieval metrics.
- Add prospective held-out maintenance scenarios and stronger aircraft/model diversity.
- Add current evaluation MRR, nDCG@10, MAP@10, Recall@20, and bootstrap CI files if those metrics are desired in the main paper.
- Add explicit ablation rows for role tags, FMECA alignment, causal edges, query intent parser, warning weighting, and package diversification.
- Validate OCR/image-only pages and cross-page evidence under stricter split controls.
