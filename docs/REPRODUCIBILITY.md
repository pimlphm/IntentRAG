# Reproducibility

## Level 1: verify the published aggregate release

```bash
python tools/verify_public_release.py
```

This requires only the Python standard library. It verifies the 33-method registry, 214-question fresh split, selected retrieval values, 594-profile KV design, blocked semantic-judge status and required figures.

## Level 2: regenerate figures from published source tables

Install the plotting environment:

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
```

The plotting scripts are under `experiments/expanded/scripts/`. Their local path constants have been sanitized in this release. Where a script uses `${INTENTRAG_EXPERIMENT_ROOT}`, `${INTENTRAG_PROJECT_ROOT}`, `${NATURE_FIGURE_SKILL}`, `${MODEL_PATH}` or `${LLAMA_SERVER}`, supply an equivalent local path before execution.

Figure source tables are under `results/figure_source_data/`. The committed figures are reference outputs; regenerated hashes can differ because of Matplotlib, font and renderer versions.

## Level 3: rerun retrieval and memory replay

The principal scripts include:

- `run_expanded_retrieval.py`
- `run_neural_retrieval.py`
- `summarize_expanded.py`
- `replay_memory.py`
- `replay_token_budget.py`
- `make_expanded_figures.py`
- `make_token_budget_figures.py`

A full rerun requires lawful local access to the underlying manuals and the project-specific processed corpus. Those inputs are not distributed. A researcher must map local evidence objects to the fields expected by the scripts and record new input hashes; otherwise the result is not the same benchmark.

## Level 4: rerun local generation diagnostics

Relevant scripts include:

- `evaluate_expanded_qa.py`
- `judge_expanded_qa.py`
- `make_qa_kv_figures.py`

The public release intentionally omits raw prompts, answers and judge payloads. Any new run must record the exact model, tokenizer, runtime, quantization, context limit, seed and generation settings.

The historical semantic-judge run is blocked and must not be converted into a score. A successful rerun should preserve the states `valid`, `unavailable`, `failed` and `not_evaluated` rather than silently filling missing scores.

## Level 5: rerun actual GPU input-KV reuse

Relevant scripts include:

- `build_kv_prompts.py`
- `measure_real_kv.py`
- `analyze_prefix_materialization_support.py`
- `make_prefix_materialization_figures.py`

Record the GPU, driver, CUDA/runtime version, inference engine build, model file hash, tokenizer, slot configuration, input length and prompt-cache settings. The published result used cold target, cold seed and warm target requests in one fixed slot and read the engine's reused-input count. Do not substitute a tokenizer-overlap proxy and label it actual KV reuse.

## Integrity files

- `PUBLICATION_MANIFEST.csv`: source class, destination, byte size and staged SHA-256.
- `PUBLICATION_AUDIT.json`: secret/path/size scan.
- `SHA256SUMS.txt`: repository payload checksums.
- Release `SHA256SUMS.json`: downloadable-asset checksums.

