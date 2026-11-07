# V21 Evidence-Sufficient Target Attainment Report

V21 is the evidence-sufficient locked benchmark for the target: case-level performance above 0.85, evidence-available performance above 0.95, and zero unsupported generation risk after safety gating.

## FD-GRAG-RC-V21 Final Metrics

| aircraft_model | track | exact_recall_10 | accepted_recall_10 | evidence_package_recall_10 | accepted_page_recall_10 | accepted_section_recall_10 | grounded_score | hallucination_risk_proxy | n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Boeing 737 | broad_package |  | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 4.000 |
| Boeing 737 | unique_exact | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 84.000 |
| Cessna 172 | broad_package |  | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 13.000 |
| Cessna 172 | unique_exact | 0.965 | 0.991 | 1.000 | 0.991 | 0.991 | 0.983 | 0.017 | 113.000 |

## Target Attainment

| aircraft_model | track | n | exact_recall_10 | accepted_recall_10 | evidence_package_recall_10 | grounded_score | ungated_hallucination_risk_proxy | gated_unsupported_generation_risk_proxy | answer_coverage_at_gate_0_95 | exact_target_0_95_met | accepted_target_0_95_met | package_target_0_95_met | grounded_target_0_95_met | gated_zero_hallucination_met |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Boeing 737 | broad_package | 4.000 |  | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.981 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Boeing 737 | unique_exact | 84.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.981 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Cessna 172 | broad_package | 13.000 |  | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.981 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Cessna 172 | unique_exact | 113.000 | 0.965 | 0.991 | 1.000 | 0.983 | 0.017 | 0.000 | 0.981 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

## Safety Gate

| method | threshold | n | answer_coverage | refusal_rate | ungated_hallucination_risk_proxy | gated_unsupported_generation_risk_proxy | emitted_answer_risk_proxy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FD-GRAG-RC-V21-final | 0.850 | 214.000 | 0.981 | 0.019 | 0.009 | 0.000 | 0.000 |
| FD-GRAG-RC-V21-final | 0.900 | 214.000 | 0.981 | 0.019 | 0.009 | 0.000 | 0.000 |
| FD-GRAG-RC-V21-final | 0.950 | 214.000 | 0.981 | 0.019 | 0.009 | 0.000 | 0.000 |
| FD-GRAG-RC-V21-final | 1.000 | 214.000 | 0.981 | 0.019 | 0.009 | 0.000 | 0.000 |

## Interpretation

- Unique-evidence exact retrieval is evaluated with Exact R@10; broad diagnostic package retrieval does not report single-ID exact as a primary claim.
- All V21 locked case-track retrieval metrics required by the evidence-sufficient target exceed 0.95, except Exact is intentionally not applicable to broad-package questions.
- Ungated retrieval risk is already below 0.02 overall, but the publishable zero-hallucination claim is only the gated system-risk proxy: the system refuses when accepted evidence/package support is incomplete.
- This is still document-grounded retrieval/generation-readiness, not real aircraft maintenance correctness, expert validation, or safety certification.
