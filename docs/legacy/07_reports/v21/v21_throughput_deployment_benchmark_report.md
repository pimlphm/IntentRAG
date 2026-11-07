# V21 throughput and deployment benchmark

This benchmark adds engineering deployability evidence to the V21 package.

## Measurement scope

- PDF document throughput is measured with `pypdf` because PyMuPDF and pdfplumber were not available in the bundled runtime.
- Deployment speed includes case evidence loading, retrieval index construction, first-query latency, all-query batch latency, median latency, p95 latency and QPS.
- The FD-GRAG-RC-V21 speed row is measured with the V21 route/package runtime proxy over the same evidence index; effectiveness metrics are still read from the V21 result CSV.

## Outputs

### Boeing_737
- Documents measured: 1
- Methods measured: 15
- Tables: `${USER_HOME}\Desktop\合作_西工大\FD_GRAG_RC_V21_Runnable_Methodology_Package_20260515\Cases\Boeing_737\7 测试\deployment_speed_tables`
- Figures: `${USER_HOME}\Desktop\合作_西工大\FD_GRAG_RC_V21_Runnable_Methodology_Package_20260515\Cases\Boeing_737\7 测试\deployment_speed_figures`
- Excel: `${USER_HOME}\Desktop\合作_西工大\FD_GRAG_RC_V21_Runnable_Methodology_Package_20260515\Cases\Boeing_737\7 测试\complete_evidence_chain_excel\Boeing_737_deployment_throughput_benchmark.xlsx`

### Cessna_172
- Documents measured: 4
- Methods measured: 15
- Tables: `${USER_HOME}\Desktop\合作_西工大\FD_GRAG_RC_V21_Runnable_Methodology_Package_20260515\Cases\Cessna_172\7 测试\deployment_speed_tables`
- Figures: `${USER_HOME}\Desktop\合作_西工大\FD_GRAG_RC_V21_Runnable_Methodology_Package_20260515\Cases\Cessna_172\7 测试\deployment_speed_figures`
- Excel: `${USER_HOME}\Desktop\合作_西工大\FD_GRAG_RC_V21_Runnable_Methodology_Package_20260515\Cases\Cessna_172\7 测试\complete_evidence_chain_excel\Cessna_172_deployment_throughput_benchmark.xlsx`
