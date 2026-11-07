# FMECA PNG Plotting and Writing Audit

Scope: this update is limited to plotting and writing. It reads existing result files and does not retrain, rerun retrieval, alter benchmark splits, or modify metric CSV values.

## Data files found for plotting
- 04_results/v21/v21_metrics_by_case.csv: found. 04_results/v21/v21_metrics_by_case.csv
- 04_results/v21/v21_metrics_by_track.csv: found. 04_results/v21/v21_metrics_by_track.csv
- 04_results/v21/v21_metrics_by_case_track.csv: found. 04_results/v21/v21_metrics_by_case_track.csv
- 04_results/v21/v21_retrieval_predictions_full.csv: found. 04_results/v21/v21_retrieval_predictions_full.csv
- 04_results/v21/v21_target_attainment_matrix.csv: found. 04_results/v21/v21_target_attainment_matrix.csv
- 04_results/v21/v21_safety_gate_metrics.csv: found. 04_results/v21/v21_safety_gate_metrics.csv
- 04_results/v21/FD_GRAG_RC_V21_complete_QA_evidence_chain_rows.csv: found. 04_results/v21/FD_GRAG_RC_V21_complete_QA_evidence_chain_rows.csv
- 04_results/v21/FD_GRAG_RC_V21_complete_QA_evidence_chain_top10_long.csv: found. 04_results/v21/FD_GRAG_RC_V21_complete_QA_evidence_chain_top10_long.csv
- Cases/*/7 */sota_evidence_chain_tables/*_sota_metrics_by_method.csv: 2 match(es). Cases/Boeing_737/7 测试/sota_evidence_chain_tables/Boeing_737_sota_metrics_by_method.csv; Cases/Cessna_172/7 测试/sota_evidence_chain_tables/Cessna_172_sota_metrics_by_method.csv
- Cases/*/7 */sota_evidence_chain_tables/*_sota_metrics_by_query_type.csv: 2 match(es). Cases/Boeing_737/7 测试/sota_evidence_chain_tables/Boeing_737_sota_metrics_by_query_type.csv; Cases/Cessna_172/7 测试/sota_evidence_chain_tables/Cessna_172_sota_metrics_by_query_type.csv
- Cases/*/7 */sota_evidence_chain_tables/*_ablation_metrics.csv: 2 match(es). Cases/Boeing_737/7 测试/sota_evidence_chain_tables/Boeing_737_ablation_metrics.csv; Cases/Cessna_172/7 测试/sota_evidence_chain_tables/Cessna_172_ablation_metrics.csv
- 04_results/historical_weak_sota/*retrieval_metrics_full.csv: 4 match(es). auxiliary retrieval-metric tables found

## Existing figures found
- Existing PNG figures under `05_figures/` and case folders: 15.
- Existing figures include SOTA main-result plots, reference-style evidence panels, FMECA role/control plots, ablation plots, and tree visualizations.

## Figures that can be redrawn as paper-style PNG
- Method overview, evidence schema, reranking architecture, and claim-boundary figures can be generated from method structure rather than experimental metrics.
- Main performance, query-type heatmap, evidence package completeness, ablation, and failure-analysis figures can be generated from existing v21/case result CSV files.
- Case-study evidence-flow panels can be generated from the existing complete QA evidence-chain rows and top-10 evidence tables.

## Data gaps that constrain interpretation
- Current evaluation summary files do not contain MRR, nDCG@10, MAP@10, or Recall@20 fields; those appear only in auxiliary rank/proxy tables.
- Current evaluation files do not expose every requested maintenance/FMECA proxy subfield such as `cause_action_joint_hit`; the available proxy fields are used without inventing missing metrics.
- Requested ablations such as `no role tags`, `no FMECA alignment`, and `no query intent parser` are not all present as explicit ablation rows; available ablation views are plotted and missing module tests are reported.

## Writing sections needing update
- Model introduction should position FD-GRAG-RC-V21 as structured evidence-package retrieval rather than ordinary chunk retrieval.
- Results and discussion should distinguish evidence retrieval, package completeness, proxy metrics, and real maintenance/safety claims.
- Claim boundary needs explicit language that Recall@10 is not diagnostic accuracy and FMECA proxy is not certification.

## Over-claiming risk
- Evidence retrieval scores can be mistaken for real aircraft maintenance accuracy if not bounded.
- FMECA-aligned proxy metrics can be mistaken for safety certification unless labelled as document-grounded proxies.
- Gated/refusal behavior should be described as an evidence-support boundary, not a guarantee of correct diagnosis.

## Experiment files intentionally not modified
- `00_raw_documents/`, `01_processed_corpus/`, `03_benchmark/`, `04_results/`, `06_tables/`, case retrieval outputs, benchmark splits, and algorithm/runtime logic are left unchanged.
