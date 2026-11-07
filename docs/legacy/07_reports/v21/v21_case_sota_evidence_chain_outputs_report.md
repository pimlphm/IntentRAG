# V21 case-level SOTA evidence-chain output report

This run expands the V21 package with complete per-case QA evidence chains, controlled representative SOTA paradigms, GaRAGe-style grounding/deflection baselines, ablation summaries, and evidence-present versus no-evidence probe figures.

## Scientific boundary

- GaRAGe-style baselines are implemented as controlled representatives of relevance-aware grounding, attribution support, and deflective behavior. They are not claimed to be an official reproduction of Sorodoc et al. (ACL Findings 2025).
- Retrieval code uses only visible query text and aircraft constraint. Gold evidence, expected roles/components, query type, difficulty, and split are used only after retrieval for evaluation and grouping.
- No-evidence probes are out-of-corpus stress tests for deflection behavior, not expert-validated field cases.

## Outputs

### Boeing 737
- Queries: 197
- Methods: 15
- Prediction rows: 2955
- Top-10 evidence rows: 29550
- Excel workbook: `${USER_HOME}\Desktop\合作_西工大\师叔\第二篇\FD_GRAG_RC_V21_Runnable_Methodology_Package_20260517_FMECA\Cases\Boeing_737\7 测试\complete_evidence_chain_excel\Boeing_737_complete_QA_SOTA_evidence_chains.xlsx`
- Tables: `${USER_HOME}\Desktop\合作_西工大\师叔\第二篇\FD_GRAG_RC_V21_Runnable_Methodology_Package_20260517_FMECA\Cases\Boeing_737\7 测试\sota_evidence_chain_tables`
- Figures: `${USER_HOME}\Desktop\合作_西工大\师叔\第二篇\FD_GRAG_RC_V21_Runnable_Methodology_Package_20260517_FMECA\Cases\Boeing_737\7 测试\sota_evidence_chain_figures`

### Cessna 172
- Queries: 279
- Methods: 15
- Prediction rows: 4185
- Top-10 evidence rows: 41850
- Excel workbook: `${USER_HOME}\Desktop\合作_西工大\师叔\第二篇\FD_GRAG_RC_V21_Runnable_Methodology_Package_20260517_FMECA\Cases\Cessna_172\7 测试\complete_evidence_chain_excel\Cessna_172_complete_QA_SOTA_evidence_chains.xlsx`
- Tables: `${USER_HOME}\Desktop\合作_西工大\师叔\第二篇\FD_GRAG_RC_V21_Runnable_Methodology_Package_20260517_FMECA\Cases\Cessna_172\7 测试\sota_evidence_chain_tables`
- Figures: `${USER_HOME}\Desktop\合作_西工大\师叔\第二篇\FD_GRAG_RC_V21_Runnable_Methodology_Package_20260517_FMECA\Cases\Cessna_172\7 测试\sota_evidence_chain_figures`
