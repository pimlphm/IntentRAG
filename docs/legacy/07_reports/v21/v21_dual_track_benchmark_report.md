# V21 Dual-Track Locked Benchmark Result

V21 separates identifiable exact-evidence retrieval from broad diagnostic package retrieval. Broad component/cross-page questions are not forced into single-ID Exact R@10 claims.

## Split Counts

| split | aircraft_model | track | n |
| --- | --- | --- | --- |
| v21_calibration | Boeing 737 | broad_package | 2 |
| v21_calibration | Boeing 737 | unique_exact | 66 |
| v21_calibration | Cessna 172 | broad_package | 8 |
| v21_calibration | Cessna 172 | unique_exact | 67 |
| v21_locked_blind_test | Boeing 737 | broad_package | 4 |
| v21_locked_blind_test | Boeing 737 | unique_exact | 84 |
| v21_locked_blind_test | Cessna 172 | broad_package | 13 |
| v21_locked_blind_test | Cessna 172 | unique_exact | 113 |
| v21_validation | Boeing 737 | broad_package | 3 |
| v21_validation | Boeing 737 | unique_exact | 38 |
| v21_validation | Cessna 172 | broad_package | 8 |
| v21_validation | Cessna 172 | unique_exact | 70 |

## FD-GRAG-RC-V21 Final Metrics by Case and Track

| aircraft_model | track | exact_recall_10 | accepted_recall_10 | evidence_package_recall_10 | grounded_score | hallucination_risk_proxy | n |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Boeing 737 | broad_package |  | 1.0 | 1.0 | 1.0 | 0.0 | 4 |
| Boeing 737 | unique_exact | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 84 |
| Cessna 172 | broad_package |  | 1.0 | 1.0 | 1.0 | 0.0 | 13 |
| Cessna 172 | unique_exact | 0.9646017699115044 | 0.991150442477876 | 1.0 | 0.9831858407079646 | 0.0168141592920354 | 113 |

## Safety Gate

| method | threshold | n | answer_coverage | refusal_rate | ungated_hallucination_risk_proxy | gated_unsupported_generation_risk_proxy | emitted_answer_risk_proxy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FD-GRAG-RC-V21-final | 0.85 | 214 | 0.9813084112149532 | 0.0186915887850467 | 0.0088785046728971 | 0.0 | 0.0 |
| FD-GRAG-RC-V21-final | 0.9 | 214 | 0.9813084112149532 | 0.0186915887850467 | 0.0088785046728971 | 0.0 | 0.0 |
| FD-GRAG-RC-V21-final | 0.95 | 214 | 0.9813084112149532 | 0.0186915887850467 | 0.0088785046728971 | 0.0 | 0.0 |
| FD-GRAG-RC-V21-final | 1.0 | 214 | 0.9813084112149532 | 0.0186915887850467 | 0.0088785046728971 | 0.0 | 0.0 |

## Claim Boundary

- Exact R@10 is primary only for `unique_exact` track.
- Broad diagnostic questions are evaluated with accepted/package/grounded metrics.
- Retrieval metrics are not diagnosis accuracy or maintenance safety certification.