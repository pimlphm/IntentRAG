# FD-GRAG-RC-V21 FMECA-Aligned Model Introduction

## 1. Overall methodological position

FD-GRAG-RC-V21 is designed to retrieve structured diagnostic and maintenance evidence packages rather than isolated text chunks. The method treats aviation maintenance answering as an evidence organization problem: a useful answer usually requires a traceable symptom, a component boundary, a causal or diagnostic relation, an action or inspection requirement, and any authority, threshold, warning, or limitation that constrains the answer.

## 2. Document-to-evidence transformation

Raw maintenance manuals, fault notes, and procedural documents are transformed into page-, section-, chunk-, episode-, facet-, and facetpoint-level evidence objects. Each object retains document lineage so that later retrieval can be audited back to page and section context.

## 3. Role-constrained evidence modeling

Evidence objects are tagged with diagnostic and maintenance roles such as symptom, confirmed cause, excluded cause, condition, threshold, action, inspection, replacement, warning, limitation, and authority. These role constraints prevent the retriever from treating a warning, a threshold, and a maintenance action as interchangeable semantic text.

## 4. Component and FMECA alignment

The method aligns evidence objects with component families and FMECA-style fields, including failure mode, failure effect, probable cause, detection evidence, maintenance action, and control. This allows the retrieved package to connect a question about a component or failure mode to the evidence roles needed for maintenance-oriented reasoning.

## 5. Causal and diagnostic graph

Component and causal links organize evidence into diagnostic paths. The graph is not used as an oracle; it is used to preserve neighboring evidence and to help retrieve cause-action, threshold-condition, and warning-authority context that may be split across pages or sections.

## 6. Multi-route retrieval and reranking

FD-GRAG-RC-V21 merges lexical, semantic, role-aware, component-aware, FMECA-aware, causal-path, episode/facet, page-neighbor, and safety-control routes into a candidate pool. A scoring and reranking stage then prioritizes evidence that jointly satisfies query intent, component constraints, role coverage, and authority or warning requirements.

## 7. Diagnostic-maintenance evidence package

The final output is an evidence package containing the ranked evidence cards needed to support an answer or refusal. A package may include action evidence, causal evidence, threshold and condition evidence, warning or limitation evidence, and authority references. This package-level view is more informative than a single-span hit because aviation maintenance questions often require multiple evidence roles.

## 8. Evidence-grounded answer boundary

The generation boundary is evidence-grounded. If the package does not sufficiently support the requested answer, the system should refuse or request human review rather than emit an unsupported maintenance statement.

## 9. Claim boundary

The evaluation supports document-grounded evidence retrieval, traceability, and package completeness. It does not prove real aircraft maintenance accuracy, operational safety, expert replacement, or FMECA certification. Recall@10 and related package metrics are evidence retrieval metrics, not diagnostic accuracy.
