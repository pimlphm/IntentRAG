# Admissibility-Signed KV Prefix Diagnostic

## Constructive answer

Computational reuse and admissible evidence reuse can be made closer by changing
the unit of reuse.  Instead of caching an arbitrary prompt prefix, cache a
canonical evidence-state prefix only under an admissibility signature.  The
signature encodes visible scope, provenance, authority, condition, action
prerequisite, source-diversity, role-diversity, and evidence identity proxies.

## Main results

- State-prefix reuse has high cache opportunity but imperfect admissibility
  purity: hit-rate proxy = 1.000, admissibility
  vector purity = 0.565.
- Full admissibility signing raises admissibility purity to
  1.000, with hit-rate proxy
  0.991.
- Exact-evidence signing is the strict reference: hit-rate proxy =
  0.243, purity =
  1.000.  It is closest to the
  condition "computational reuse ~= admissible evidence reuse", but costs more
  reuse opportunity.

## Paper claim

The contribution on KV cache should not be phrased as measuring real GPU KV
cache internals.  It should be phrased as a cache-key and prefix-organization
principle: admissibility-signed KV reuse.  The empirical result is the observed
reuse-purity frontier on project data.

## Claim map

| claim_id   | claim                                                                                                                       | support_status                                            | key_value                                                   | result_file                                               | figure                                                        | caveat                                                                                              |
|:-----------|:----------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------|:------------------------------------------------------------|:----------------------------------------------------------|:--------------------------------------------------------------|:----------------------------------------------------------------------------------------------------|
| SC1        | Admissibility-signed prefixes make cache reuse conditional on evidence-state compatibility.                                 | Supported as a constructive tokenizer/grouping diagnostic | Full signature purity=1.000                                 | 06_tables/signed_cache_v1/signed_cache_policy_summary.csv | 05_figures/signed_cache_v1/figure_a_signed_cache_tradeoff.png | Proxy grouping; no server-side KV cache table inspection.                                           |
| SC2        | Increasing admissibility binding trades cache-hit opportunity for admissibility purity.                                     | Supported as observed trade-off across cache-key policies | State hit=1.000, purity=0.565; full hit=0.991, purity=1.000 | 06_tables/signed_cache_v1/signed_cache_policy_summary.csv | 05_figures/signed_cache_v1/figure_a_signed_cache_tradeoff.png | The design goal is controlled reuse, not maximal hit rate.                                          |
| SC3        | Exact-evidence signed prefixes approximate computational reuse equals admissible evidence reuse at the cost of lower reuse. | Supported as a strict-boundary reference policy           | Exact-evidence hit=0.243, purity=1.000                      | 06_tables/signed_cache_v1/signed_cache_policy_summary.csv | 05_figures/signed_cache_v1/figure_a_signed_cache_tradeoff.png | Approximate because top-4 evidence and proxy admissibility are used, not expert equivalence labels. |
