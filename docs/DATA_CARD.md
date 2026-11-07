# Data Card

## Scope

This repository publishes metadata, aggregate results, figure source tables and audit records derived from the IntentRAG experiments. It does not publish the underlying aviation-maintenance manuals or text-bearing query-level evidence packages.

## Evaluation units

The project uses several distinct units that must not be conflated:

| Unit | Count | Purpose |
|---|---:|---|
| Retrieval records | 690 | Four frozen question-text splits |
| Retrieval method-record cells | 22,770 | 33 methods evaluated for every record |
| Fresh source-derived questions | 214 | Retrieval and evidence-coverage validation |
| Local generation questions | 64 | 32 legacy and 32 fresh questions |
| Local generation cells | 2,112 | 64 questions × 33 methods |
| Actual GPU prefix profiles | 594 | Six query pairs × 33 methods × three layouts |

## Fresh-test aircraft distribution

| Aircraft family | Questions | Percentage |
|---|---:|---:|
| Cessna 172 | 212 | 99.07% |
| Boeing 737 | 2 | 0.93% |
| Total | 214 | 100% |

The fresh test is not aircraft-balanced. The Boeing sample is insufficient for stable cross-aircraft generalization claims.

## Public fields

Public assets are limited to:

- method names and implementation-status labels;
- aggregate retrieval, evidence-support, memory-replay and KV-reuse statistics;
- plot source tables that do not reproduce manual passages;
- experimental manifests without personal paths or credentials;
- independently generated figures and audit reports.

## Withheld fields and records

The following are withheld because they may disclose copyrighted, identifying or security-sensitive content:

- original manual files, source chunks and long quotations;
- natural-language questions tied to source passages;
- raw prompts, model answers, judge payloads and server logs;
- query-level evidence packages and source-selection mappings;
- embeddings, vector indexes and model weights;
- chat history, account data and local absolute paths.

## Labels and metrics

Accepted evidence and required-quote groups are post-retrieval audit assets. Hit@10 and quote support measure retrieval-side documentary coverage. They are not equivalent to semantic answer correctness, action authorization, regulatory compliance or expert certification.

## Known limitations

- Fresh questions are source-derived rather than a natural-user sample.
- The legacy and fresh splits use project-specific equivalence rules.
- The attempted semantic judge produced no valid judgments.
- Some method labels denote controlled local implementations rather than official baseline reproductions.

## Access to restricted sources

Full reruns require source materials that are not redistributed here. Researchers must obtain lawful access independently and map their local data to the schemas described by the experiment scripts.

