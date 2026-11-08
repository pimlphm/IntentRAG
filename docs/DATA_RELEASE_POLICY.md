# Data Release Policy

## Classes

| Class | Examples | Policy |
|---|---|---|
| A | Aggregate metrics, method configurations, plots | Track in Git after automated scan |
| B | De-identified derived records | Release only after field-level review |
| C | Query-source mappings, clause identifiers, embeddings | Withhold pending rights and reconstruction-risk review |
| D | Manuals, long excerpts, raw prompts/answers, chats, credentials | Never publish in this repository |

## Current release

The v1.0.0 public research preview contains Class A material only. The sanitized Release archive mirrors the same boundary. The internal 104.7 MB validation package is not uploaded because it contains text-bearing raw responses and query-level records.

## Automated checks

The publication build checks:

- common credential patterns;
- Windows user and external-drive absolute paths;
- the GitHub 100 MiB single-object limit;
- staged-file hashes and provenance labels.

Automated checks do not establish copyright permission. Public visibility does not relax the stated data boundary or grant reuse rights; further licensing still requires rights-holder, co-author and institutional review.

## History rule

Restricted content must never be committed and later deleted as a cleanup step. Git history, forks, caches and release assets may preserve earlier versions.

