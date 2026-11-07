# V21 Evidence-State Diagnostic Audit

This directory contains the code-backed diagnostic audit used by the revised manuscript. It is an audit toward admissible evidence-state reconstruction (AESR), not a validation of a five-state AESR classifier.

## What the audit tests

1. Dependence of legacy V21 retrieval on appended gold-passage cues and the locked unique-versus-package track.
2. A fair retrieval lane in which both the cue and track are removed before retrieval.
3. Cue- and track-free word/character TF-IDF, hybrid lexical, balanced-route IntentRAG, multilingual dense, dense-plus-cross-encoder, and local original-order-window settings.
4. Retriever-visible factor proxies on unmodified packages.
5. Explicitly synthetic post-retrieval interventions with preregistered proxy directions.

All benchmark questions are synthetic and source-derived. Exact/accepted/page/section/slot labels are used only for post-hoc audit. The fair retrieval path never receives the appended cue, locked track, query type, required slots, gold identifiers, accepted alternatives, or split metadata.

## Run

From the package root:

```powershell
.\10_scripts\run_aesr_v2_audit.ps1
```

The run uses the configuration in `configs/aesr_v2/aesr_v2.json`. Transformer checkpoints are loaded from the local Hugging Face cache when `offline_models_only` is true.

## Outputs

- Query-level access results: `04_results/aesr_v2/`
- Figures: `05_figures/aesr_v2/`
- Tables: `06_tables/aesr_v2/`
- Report and manifest: `07_reports/aesr_v2/`
- Claim traceability: `04_results/aesr_v2/claim_result_map.csv`
- Leakage treatment: `04_results/aesr_v2/split_leakage_audit.csv`

Every synthetic package mutation records its base and donor evidence, changed fields, seed, expected direction, and machine-readable synthetic flag.

## Claim boundary

This audit can support statements about cue/track dependence, cue- and track-free retrieval behavior, observable package-proxy distributions, and deterministic proxy responses to synthetic interventions. It does not establish natural-question accuracy, five-state inference, action authorization, claim entailment, access-agnostic efficacy, an online calibrated refusal gate, graph-causal mechanism, or aircraft safety.
