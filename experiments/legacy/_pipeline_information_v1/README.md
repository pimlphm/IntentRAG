# Observable information-distribution diagnostic v1

This pipeline audits how a fair cue- and track-free retrieval package changes observable evidence distributions before any LLM generation. It reads the existing AESR-v2 artifacts but writes only to new `information_v1` paths.

## Four distribution estimators

On a common category support, the audit constructs:

- `p_D`: repository prior, obtained by normalizing candidate evidence-object counts.
- `p_q`: query-demand proxy. On the primary evidence-role axis it normalizes bilingual visible-question lexicon matches. On the source-document axis it normalizes non-negative word-TFIDF affinity mass over candidate evidence objects.
- `p_R`: retrieved-package distribution, obtained from the realized fair Top-K order with rank mass `1/log2(rank+1)`.
- `p_G`: locked reference distribution, obtained by uniformly counting accepted source-derived evidence identifiers. This is offline and construction-coupled.

The locked query-type-to-role profile is recorded only as a sensitivity comparison for `p_q`; it never enters package construction.

## Observable statistics

For mass vector `p` on `m` repository categories:

```text
H_norm(p) = -sum_c p(c) log2 p(c) / log2(m)
```

For two normalized mass vectors, the code uses base-2 Jensen-Shannon divergence. The observable alignment shift is:

```text
G_align_obs = JS(p_q, p_D) - JS(p_q, p_R)
```

Positive values mean only that the realized package is closer to the selected query-demand estimator than the repository prior. They are not information gain.

For evidence item `i` in the fixed realized package:

```text
Delta_i = JS(p_q, p_R_without_i) - JS(p_q, p_R)
```

Positive values mean that removing the item worsens estimator alignment. Retrieval is not rerun and ranks are not reassigned, so this is a post-hoc package contribution, not a causal retrieval effect.

## Run

From the package root:

```powershell
.\10_scripts\run_information_v1_diagnostic.ps1
```

## Outputs

- Raw estimator values: `04_results/information_v1/raw/`
- Query/method diagnostics and claim map: `04_results/information_v1/`
- Figures: `05_figures/information_v1/`
- Tables: `06_tables/information_v1/`
- Report and manifest: `07_reports/information_v1/`
- Run and validation logs: `08_logs/information_v1/`

No generator is run, and no independent knowledge source is attributed to an LLM.
