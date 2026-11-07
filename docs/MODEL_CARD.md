# Model and Runtime Card

## Models referenced by the experiments

- Qwen2.5 local generator configurations were used for answer-generation diagnostics.
- A local 7B judge was planned for semantic auditing but produced zero valid responses because the socket connection was blocked.
- Sentence-transformer and mMARCO cross-encoder models were used by neural retrieval routes.
- A small GGUF runtime was used for the actual input-KV reuse measurement.

Model weights are not included. Users must obtain models from their original providers and comply with the corresponding licenses.

## Runtime disclosure

The public scripts preserve the experimental logic but local absolute model and executable paths have been replaced by placeholders. Full reruns require explicit local paths, runtime versions and hardware configuration.

## Intended use

The models are research instruments for measuring retrieval, evidence packaging, mechanical output behavior and cache reuse. They are not certified maintenance decision systems.

## Out-of-scope use

Do not use this release to authorize maintenance actions, replace approved manuals, infer airworthiness, or generate operational instructions without qualified human review and valid source authority.

## Limitations

- Model-scale comparisons do not show that a larger language model reconstructs aircraft-specific evidence state from the question alone.
- Citation formatting is not entailment.
- Prefix reuse is runtime- and serialization-dependent.
- No model in this release has been validated as an independent evidentiary source.

