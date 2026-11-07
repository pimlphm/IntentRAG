## Figure V21-01. Figure V21 01 Method Overview.
This figure shows method overview for FD-GRAG-RC-V21 using method design; no metric values.
The result indicates V21 is organized around structured evidence packages rather than single text chunks.
This supports the methodological claim that FD-GRAG-RC-V21 should be evaluated as structured evidence-package retrieval.
However, this should not be interpreted as Conceptual method figure; it does not report maintenance effectiveness.

## Figure V21-02. Figure V21 02 Evidence Schema Fmeca.
This figure shows schema for FD-GRAG-RC-V21 using method design; evidence object fields from existing corpus schema.
The result indicates Structured evidence connects text spans to role, component, causal, and FMECA fields.
This supports the methodological claim that FD-GRAG-RC-V21 should be evaluated as structured evidence-package retrieval.
However, this should not be interpreted as Schema figure; it does not imply certified FMECA completeness.

## Figure V21-03. Figure V21 03 Candidate Pool Reranking.
This figure shows architecture for FD-GRAG-RC-V21 using method design; existing candidate-pool diagnostics support the route concept.
The result indicates The architecture diversifies evidence before ranking and assembling the answer package.
This supports the methodological claim that FD-GRAG-RC-V21 should be evaluated as structured evidence-package retrieval.
However, this should not be interpreted as Architecture figure; does not add or tune retrieval results.

## Figure V21-04. Figure V21 04 Main Performance Forest.
This figure shows forest/dot plot for FD-GRAG-RC-V21 using Cases/Boeing_737/7 测试/sota_evidence_chain_tables/Boeing_737_sota_metrics_by_method.csv; Cases/Cessna_172/7 测试/sota_evidence_chain_tables/Cessna_172_sota_metrics_by_method.csv.
The result indicates Accepted and package-level evidence metrics show the effect of structured retrieval.
This supports the methodological claim that FD-GRAG-RC-V21 should be evaluated as structured evidence-package retrieval.
However, this should not be interpreted as R@10 and grounding are evidence retrieval metrics, not real maintenance diagnostic accuracy; MRR/nDCG are unavailable in current evaluation summary files.

## Figure V21-05. Figure V21 05 Recall At K Curves.
This figure shows recall curve for FD-GRAG-RC-V21 using Cases/Boeing_737/7 测试/sota_evidence_chain_tables/Boeing_737_complete_sota_top10_evidence_long.csv; Cases/Cessna_172/7 测试/sota_evidence_chain_tables/Cessna_172_complete_sota_top10_evidence_long.csv.
The result indicates Top-10 evidence tables support Recall@1/3/5/10 curves from existing retrieved evidence.
This supports the methodological claim that FD-GRAG-RC-V21 should be evaluated as structured evidence-package retrieval.
However, this should not be interpreted as Curves are evidence hit rates from stored retrieval outputs; they are not real maintenance accuracy and do not include Recall@20.

## Figure V21-06. Figure V21 06 Query Type Fmeca Heatmap.
This figure shows heatmap for FD-GRAG-RC-V21 using Cases/Boeing_737/7 测试/sota_evidence_chain_tables/Boeing_737_sota_metrics_by_query_type.csv; Cases/Cessna_172/7 测试/sota_evidence_chain_tables/Cessna_172_sota_metrics_by_query_type.csv.
The result indicates Task-level heatmaps reveal where structured evidence packaging helps or remains weak.
This supports the methodological claim that FD-GRAG-RC-V21 should be evaluated as structured evidence-package retrieval.
However, this should not be interpreted as Task scores are document-grounded evidence metrics, not direct maintenance outcomes.

## Figure V21-07. Figure V21 07 Evidence Package Completeness.
This figure shows matrix dot plot for FD-GRAG-RC-V21 using Cases/Boeing_737/7 测试/sota_evidence_chain_tables/Boeing_737_complete_sota_top10_evidence_long.csv; Cases/Cessna_172/7 测试/sota_evidence_chain_tables/Cessna_172_complete_sota_top10_evidence_long.csv.
The result indicates Package completeness evaluates role coverage across retrieved evidence, not only a single hit.
This supports the methodological claim that FD-GRAG-RC-V21 should be evaluated as structured evidence-package retrieval.
However, this should not be interpreted as Role presence is derived from stored evidence tags and should not be interpreted as clinical or operational correctness.

## Figure V21-08. Figure V21 08 Maintenance Fmeca Proxy Metrics.
This figure shows proxy metric heatmap for FD-GRAG-RC-V21 using auxiliary document-grounded proxy metrics table.
The result indicates Available proxy fields quantify document-grounded evidence completeness for maintenance-oriented answer support.
This supports the methodological claim that FD-GRAG-RC-V21 should be evaluated as structured evidence-package retrieval.
However, this should not be interpreted as Proxy metrics are not real maintenance effectiveness, aircraft safety certification, or expert validation.

## Figure V21-09. Figure V21 09 Ablation Delta Waterfall.
This figure shows ablation delta plot for FD-GRAG-RC-V21 using Cases/Boeing_737/7 测试/sota_evidence_chain_tables/Boeing_737_ablation_metrics.csv; Cases/Cessna_172/7 测试/sota_evidence_chain_tables/Cessna_172_ablation_metrics.csv.
The result indicates Available ablation rows show which stored module-removal views reduce grounded evidence support.
This supports the methodological claim that FD-GRAG-RC-V21 should be evaluated as structured evidence-package retrieval.
However, this should not be interpreted as Only available ablation rows are plotted; missing requested module removals are not fabricated.

## Figure V21-10. Figure V21 10 Failure Analysis Matrix.
This figure shows failure matrix for FD-GRAG-RC-V21 using current complete QA evidence-chain rows.
The result indicates Gated-review cases can be categorized by interpretable evidence failure attributes.
This supports the methodological claim that FD-GRAG-RC-V21 should be evaluated as structured evidence-package retrieval.
However, this should not be interpreted as Attribute coding is derived from existing review/metric fields for analysis; it is not a new benchmark label or algorithm change.

## Figure V21-11. Figure V21 11 Claim Boundary Evidence Hierarchy.
This figure shows claim boundary for FD-GRAG-RC-V21 using writing support; no metric values.
The result indicates The manuscript should bound claims at evidence retrieval and document-grounded package support.
This supports the methodological claim that FD-GRAG-RC-V21 should be evaluated as structured evidence-package retrieval.
However, this should not be interpreted as Explicitly excludes operational safety certification, expert replacement, or guaranteed diagnosis.

## Figure V21-12a. Case Boeing Success 01.
This figure shows case study for FD-GRAG-RC-V21 using current complete QA evidence-chain rows and top-10 evidence table.
The result indicates Evidence-flow case studies expose how retrieved cards support either answer emission or refusal.
This supports the methodological claim that FD-GRAG-RC-V21 should be evaluated as structured evidence-package retrieval.
However, this should not be interpreted as Case panels are document-grounded examples and do not establish real-world maintenance correctness.

## Figure V21-12b. Case Cessna Success 01.
This figure shows case study for FD-GRAG-RC-V21 using current complete QA evidence-chain rows and top-10 evidence table.
The result indicates Evidence-flow case studies expose how retrieved cards support either answer emission or refusal.
This supports the methodological claim that FD-GRAG-RC-V21 should be evaluated as structured evidence-package retrieval.
However, this should not be interpreted as Case panels are document-grounded examples and do not establish real-world maintenance correctness.

## Figure V21-12c. Case Fmeca Maintenance Action 01.
This figure shows case study for FD-GRAG-RC-V21 using current complete QA evidence-chain rows and top-10 evidence table.
The result indicates Evidence-flow case studies expose how retrieved cards support either answer emission or refusal.
This supports the methodological claim that FD-GRAG-RC-V21 should be evaluated as structured evidence-package retrieval.
However, this should not be interpreted as Case panels are document-grounded examples and do not establish real-world maintenance correctness.

## Figure V21-12d. Case Failure Or Limitation 01.
This figure shows case study for FD-GRAG-RC-V21 using current complete QA evidence-chain rows and top-10 evidence table.
The result indicates Evidence-flow case studies expose how retrieved cards support either answer emission or refusal.
This supports the methodological claim that FD-GRAG-RC-V21 should be evaluated as structured evidence-package retrieval.
However, this should not be interpreted as Case panels are document-grounded examples and do not establish real-world maintenance correctness.
