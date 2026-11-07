SEPIA REVIEW — academic manuscript, local in-place revision
Loaded: professional-pass.md; style-pass.md; model-fingerprints.md
Model: author=unknown executor=GPT version=unknown
Prose layer: author=none executor=prior
Venue corpus: existing verified IntentRAG manuscript and its September 2026 edit log
Style scan: the draft insert initially repeated contrast frames, reused three-part sentence structures, and announced scope before giving results.
Failed: #2 density — several early sentences repeated the distinction among retrieval, reuse, and admissibility after the paper had already established it.
Failed: #6 formatting tells — the first outline used equal-length method, result, and limitation paragraphs with repeated three-item lists.
Failed: #8 templatedness — successive figure explanations began with the same sentence frame.
Failed: #9 sameness of rhythm — consecutive result sentences used similar clause counts and length.
Protected: equations, references, Tables 1-7, Figures 1-27, quoted method names, numeric results, and all unselected paragraphs.
Passed: 1, 3, 4, 5, 7, 10
Verdict: cluster → refactor

Stage 2 applied
- Removed repeated claims already established in the existing manuscript.
- Replaced generic section tours with measured sample sizes and endpoints.
- Varied figure explanations according to their inferential role rather than a common template.
- Preserved negative results, caveats, method identities, and all reported numbers.
- Kept the changes local and blue; no reference, formula, or pre-existing figure was rewritten.

Deletion and reversion tests
- Added words were retained only when they specify a protocol, a measured result, or an interpretation boundary.
- Existing sound language was left unchanged. Replacements were limited to paragraphs made outdated by the new retrieval and GPU experiments.
