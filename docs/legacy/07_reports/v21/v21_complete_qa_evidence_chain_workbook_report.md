# V21 Complete QA Evidence-Chain Workbook Report

Generated on 2026-05-15.

## Purpose

This artifact mirrors the structure of the reference workbook `完整问答检索证据链_O_M_flow_20260501_131914.xlsx` while using the V21 runnable methodology package outputs. It provides complete question-answer retrieval records for both aircraft cases, together with top-10 evidence-chain details and case-level metrics.

## Source Inputs

- `04_results/v21/qa_warehouse_v21_full.csv`
- `04_results/v21/v21_retrieval_predictions_full.csv`
- `04_results/v21/v21_gold_sets.csv`
- `01_processed_corpus/v5/evidence_objects.parquet`
- `04_results/v21/v21_metrics_by_case_track.csv`
- `04_results/v21/v21_safety_gate_metrics.csv`
- `11_manifest/template_qa_chain_131914.xlsx`

## Generated Workbooks

- Combined workbook: `04_results/v21/FD_GRAG_RC_V21_complete_QA_evidence_chain_Boeing_Cessna.xlsx`
- Boeing case workbook: `Cases/Boeing_737/7 测试/Boeing_737_complete_QA_evidence_chain_V21.xlsx`
- Cessna case workbook: `Cases/Cessna_172/7 测试/Cessna_172_complete_QA_evidence_chain_V21.xlsx`

## Workbook Structure

The combined workbook contains:

- `Summary`: case-level row counts and V21 metric recap.
- `Boeing_QA_Results`: full Boeing 737 QA evidence-chain records.
- `Cessna_QA_Results`: full Cessna 172 QA evidence-chain records.
- `Evidence_Chain_Top10`: long-form top-10 evidence chain for every question.
- `V21_Metrics`: method/case/track metric table.
- `Safety_Gate`: grounded-risk and refusal-policy summary.
- `Template_Reference`: the first rows from the original reference workbook header for traceability.

## Verified Dimensions

- Combined workbook: 476 QA rows and 4,760 top-10 evidence-chain rows.
- Boeing workbook: 197 QA rows and 1,970 top-10 evidence-chain rows.
- Cessna workbook: 279 QA rows and 2,790 top-10 evidence-chain rows.

## Claim Boundary

The workbook reports document-grounded retrieval and evidence-package proxy results. It does not claim real maintenance correctness, real safety certification, expert validation, or field diagnostic accuracy. Risk and groundedness are proxy audit metrics derived from retrieved evidence coverage and safety-gate rules.

