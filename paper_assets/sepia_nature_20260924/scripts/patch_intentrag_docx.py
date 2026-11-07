from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from lxml import etree
from PIL import Image


NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "pic": "http://schemas.openxmlformats.org/drawingml/2006/picture",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "m": "http://schemas.openxmlformats.org/officeDocument/2006/math",
}

EXPECTED_BASE_SHA256 = "70FDEF5AD693835757BC2A0FA10CA2E02A69240BC768058BE879776EDAAFDADD"
BLUE = "0000FF"


REPLACEMENTS = [
    (
        "External knowledge access in aviation maintenance may be implemented by retrieval, long-context loading, cached prompts, or memory-oriented backends, yet each leaves the same pre-language question: which assembled state may safely support the current engineering claim? We formulate this problem as admissibility-constrained state reuse and treat IntentRAG as one retrieval-based operationalization. The LLM remains downstream: it verbalizes an already bounded state rather than supplying missing aircraft-specific evidence. We contribute (i) an observable distributional audit that separates evidence-role selection frequency from fixed-package conditional support; (ii) identity-aware multi-granularity organization in which Detail/Page provide query-local evidence and Section/Profile provide persistent navigation, with entropy-based routing used only for review allocation; and (iii) an admissibility-signed prefix interface that restricts computational reuse to evidence-compatible classes. Across 214 fair-path, source-derived queries, mean reusable evidence-token mass is 62.3% overall, 87.4% for Boeing, and 44.8% for Cessna, whereas documentary action-authority prerequisite proxies are 0.330 and 0.587, respectively. Additional audits show that persistent views complement rather than replace query-local evidence. In a post-hoc signed-prefix grouping diagnostic, the full observed-proxy signature retains a 0.991 reuse-opportunity proxy with sample-internal purity 1.000, whereas exact-evidence grouping reduces the opportunity proxy to 0.243. These results do not establish mutual information, natural-query generalization, expert action authorization, or actual Transformer KV-cache hits or latency savings; they identify auditable constraints on evidence-state organization before language.",
        "External knowledge access in aviation maintenance may use retrieval, long-context loading, cached prompts, or memory-oriented backends. All four leave the same pre-language question: which assembled state may support the current engineering claim? We formulate this as admissibility-constrained state reuse and use IntentRAG as a retrieval-based operationalization. The LLM remains downstream and verbalizes a bounded state; it does not supply missing aircraft-specific evidence. The paper makes three contributions: an observable distributional audit that separates evidence-role selection frequency from fixed-package conditional support; identity-aware multi-granularity organization, with Detail/Page carrying query-local evidence and Section/Profile supporting persistent navigation; and a deployment-target signed-prefix interface that confines computational reuse to evidence-compatible classes. Entropy routes review effort and does not certify evidence. Across 214 fair-path, source-derived queries, mean reusable evidence-token mass is 62.3% overall, 87.4% for Boeing, and 44.8% for Cessna, while the documentary action-authority prerequisite proxies are 0.330 and 0.587. Additional audits show that persistent views complement query-local evidence. In the post-hoc grouping diagnostic, the observed-feature signature places 0.991 of queries in non-singleton groups and reaches within-group observed-feature agreement of 1.000, with an idealized first-touch/no-eviction ceiling of 0.893; its top-4 evidence-identity purity is only 0.00935. Grouping by aircraft, top-4 evidence identity, and action bucket lowers non-singleton membership to 0.243 and the first-touch ceiling to 0.140. These results define auditable constraints on evidence-state organization. They do not establish mutual information, natural-query generalization, expert action authorization, actual Transformer KV-cache hits, or latency savings.",
    ),
    (
        "From this angle, evidence-state quality is governed by three non-equivalent properties: distributional alignment with visible query demand, recoverability of documentary identity after granularity expansion, and compatibility with the target admissibility relation. Retrieval score, entropy, or prefix overlap alone can diagnose at most one layer and cannot substitute for the others.",
        "Three properties determine evidence-state quality, and none can be inferred from the others: observable alignment with visible query demand, recovery of documentary identity after granularity expansion, and compatibility with the target admissibility relation. Retrieval score, entropy, and prefix overlap describe different layers of the problem.",
    ),
    (
        "(1) An observable evidence-distribution mechanism that separates how frequently an evidence role is selected from how strongly it supports the realized package. Role transport measures over- or under-allocation within a fixed retrieval budget, while fixed-package leave-one-out sensitivity measures conditional support without rerunning retrieval.",
        "(1) Observable evidence distribution. We distinguish selection frequency from support within the realized package. Role transport measures allocation above or below the visible-demand profile under a fixed retrieval budget. Fixed-package leave-one-out sensitivity measures conditional support without rerunning retrieval.",
    ),
    (
        "(2) Identity-aware multi-granularity evidence organization in which Detail/Page retain query-local documentary identity and Section/Profile provide persistent cross-page navigation and conditional recovery. Granularity expansion is treated as a coverage-resolution trade-off, and final support must return to a traceable page, clause, or evidence identifier.",
        "(2) Identity-aware multi-granularity evidence. Detail/Page retain query-local documentary identity, while Section/Profile provide persistent cross-page navigation and conditional recovery. Granularity expansion trades resolution for coverage. Any evidence used for final support must resolve to a traceable page, clause, or evidence identifier.",
    ),
    (
        "(3) Admissibility-constrained prefix reuse as a deployment-target interface between evidence organization and computational reuse. Canonical evidence states may enter the same reuse group only within an admissibility-compatible class; the reported signed-prefix audit measures an input-side proxy frontier rather than actual GPU KV-cache reuse or independently validated maintenance authorization.",
        "(3) Admissibility-constrained prefix reuse. We define a deployment-target interface in which canonical evidence states share a reuse class only when their admissibility signatures are compatible. The executed analysis is a post-hoc input-side grouping diagnostic; it does not measure GPU KV-cache reuse or independently validated maintenance authorization.",
    ),
    (
        "The rest of the paper first repositions related work around knowledge access, multi-granularity memory, and state reuse; then formalizes distributional alignment, documentary identity recovery, and admissibility; presents the retrieval-based operationalization and bounded audits; and closes with the distinction between repeated computation, reusable evidence, and reusable action authority.",
        "The remainder of the paper relates these claims to work on knowledge access, multi-granularity memory, and state reuse, then presents the retrieval implementation and its bounded audits. The discussion returns to the distinction among repeated computation, reusable evidence, and reusable action authority.",
    ),
    (
        "The repository is exposed through four documentary views. Detail identifies a paragraph, table row, facet, or evidence unit; Page restores adjacent context and local procedural continuity; Section supplies a cross-page structural view; and Profile supplies a component- or document-family view. Detail and Page form the query-local path because they preserve the identity needed for the current claim. Section and Profile form the persistent path because their structures may be reused across queries and may recover evidence missed by a narrow path. Persistent does not mean temporal conversational memory or an additional knowledge source.",
        "The repository is exposed through four documentary views. Detail identifies a paragraph, table row, facet, or evidence unit. Page restores adjacent context and local procedural continuity. Section supplies a cross-page structural view, and Profile supplies a component- or document-family view. Detail and Page form the query-local path because they preserve the identity needed for the current claim. Section and Profile form the persistent path because their structures can be reused across queries and can recover evidence missed by a narrow path. Here, persistent refers to reusable documentary structure, not temporal conversational memory or a separate knowledge source.",
    ),
    (
        "Evidence therefore precedes language as an interface constraint, not as a claim that the current system has solved online epistemic control. The executed audits stop before generation and do not test claim entailment, action correctness, semantic entropy, calibrated refusal, or safety certification.",
        "Here, evidence precedes language as an interface constraint. The executed audits stop before generation and leave online epistemic control unresolved; they do not test claim entailment, action correctness, semantic entropy, calibrated refusal, or safety certification.",
    ),
    (
        "In summary, the scientific object is the constrained evidence state: distributional diagnostics describe what the access operator selected, multi-granularity views determine how documentary identity is expanded and recovered, and admissibility determines what may be carried into language. IntentRAG remains one executed retrieval-based approximation of this object.",
        "The analysis treats the constrained evidence state as its scientific object. Distributional diagnostics describe what the access operator selects. Multi-granularity views expand and recover documentary identity, while the admissibility relation determines what may enter the generator. IntentRAG is one retrieval-based approximation evaluated here.",
    ),
    (
        "The combined results identify three separations that a retrieval-only account obscures. First, evidence that frequently occupies a package is not necessarily the evidence that conditionally supports the realized state. Second, a broader and more reusable documentary view is not necessarily a sufficiently resolved source for the current claim. Third, a repeated computational prefix is not automatically reusable evidence. Evidence-state quality is consequently governed jointly by distributional alignment, recoverable documentary identity, and admissibility compatibility rather than by retrieval score, entropy, or prefix overlap alone.",
        "The results expose three breaks in the chain from retrieval to generation. Selection frequency and conditional support diverge within the realized package. Documentary views can be broad and reusable while leaving the current claim unresolved. Repeated prefixes can likewise combine evidence-incompatible requests. Evidence-state quality therefore depends on distributional alignment, recoverable documentary identity, and admissibility compatibility; retrieval score, entropy, and prefix overlap each describe only part of that object.",
    ),
    (
        "This interpretation does not position multi-granularity organization as an alternative knowledge source or claim that RAG has been replaced. IntentRAG remains the executed access mechanism. Detail/Page and Section/Profile determine where the system searches and how much documentary structure is exposed; the gate determines what may be used; and the LLM verbalizes the admitted result. A larger model may consume a supplied state more effectively, but scale alone does not establish current aircraft, revision, condition, provenance, or authority boundaries.",
        "IntentRAG remains the executed retrieval mechanism, and the multi-granularity layer does not create a separate knowledge source. Detail/Page and Section/Profile govern search and documentary exposure; the gate limits what may be used; and the LLM verbalizes the admitted result. A larger model may consume a supplied state more effectively, but scale alone cannot establish the current aircraft, revision, condition, provenance, or authority boundaries.",
    ),
    (
        "For deployment, the admitted state is serialized into a canonical prefix and bound to an admissibility signature. A target cache-group key is defined as the hash of the model, tokenizer, prompt template, observed admissibility signature, and canonical evidence state. Thus byte-level prefix equality is necessary for reuse but is not sufficient for evidentiary compatibility.",
        "For deployment, we propose serializing each admitted state into a canonical prefix and binding that prefix to an admissibility signature. The target cache key hashes the model, tokenizer, prompt template, observed admissibility signature, and canonical admitted evidence state. This is the proposed interface, not the key executed in the post-hoc audit, which groups records from query-level fields and never verifies byte- or token-identical prefixes. At deployment, byte-level prefix equality remains necessary for reuse, and the signature adds an evidentiary compatibility requirement.",
    ),
    (
        "The post-hoc grouping diagnostic compares three policies. A plain state prefix has reuse-opportunity proxy 1.000, sample-internal admissibility purity 0.565, and token retention 1.000. The full observed-proxy signature changes these values to 0.991, 1.000, and 0.996. Exact-evidence grouping also reaches purity 1.000 but reduces the opportunity proxy to 0.243 and token retention to 0.286. Among these three tested policies, the full signature lies on the empirical reuse-opportunity-proxy-purity frontier; its purity is partly coupled to the proxy family used for grouping and is not independent expert validation.",
        "The post-hoc diagnostic compares three query groupings rather than implemented cache keys. Aircraft plus query type places 1.000 of queries in non-singleton groups, has 0.565 within-group observed-feature agreement, retains 1.000 of pairable token mass, and has an idealized first-touch/no-eviction reuse ceiling of 0.921. The observed-feature signature changes these values to 0.991, 1.000, 0.996, and 0.893. Yet its within-group top-4 evidence-identity purity is only 0.00935, so agreement on the observed fields does not establish identical evidence. Grouping by aircraft, top-4 evidence identity, and action bucket yields 0.243 non-singleton membership, 1.000 observed-feature agreement, 0.286 pairable token-mass retention, and a 0.140 first-touch ceiling. Among these policies, the observed-feature signature best preserves pairability while enforcing agreement on the proxy fields used to define it. Its 1.000 agreement is sample-internal and partly tautological, not independent expert validation.",
    ),
    (
        "Figure 27 isolates the central signed-prefix trade-off: the objective is not to maximize raw prefix reuse, but to define an evidence-compatible equivalence class that preserves most observable reuse opportunity without mixing incompatible support states. The plotted values are input-side grouping proxies and must not be interpreted as measured GPU cache performance.",
        "Figure 27 places the three groupings on an observed-feature-agreement versus reusable-query-opportunity plane. Filled circles use the non-singleton query fraction, hollow triangles use the idealized first-touch/no-eviction ceiling, marker size represents pairable token-mass retention, and horizontal whiskers reproduce the reported 95% interval for the non-singleton fraction. The figure tests whether a grouping preserves observable reuse opportunity while excluding disagreements on the measured admissibility fields. Because both axes are derived from input-side grouping, no point is a measured cache hit rate.",
    ),
    (
        "Fig. 27. Post-hoc reuse-opportunity, sample-internal admissibility-purity, and token-retention frontier for three prefix-equivalence policies. Full observed-proxy signing raises within-bucket purity from 0.565 to 1.000 while retaining a 0.991 reuse-opportunity proxy and 0.996 token mass; exact-evidence signing also reaches purity 1.000 but reduces the two reuse proxies to 0.243 and 0.286. The contribution is the definition of an admissibility-compatible reuse class, not a claim of more cache hits: no request-order trace, layerwise KV identity, GPU saving, prefill FLOPs, TTFT, or independent expert authorization was measured.",
        "Fig. 27. Input-side equivalence-class diagnostic for three query groupings. The vertical coordinate gives within-group agreement on the observed admissibility features. Filled circles use the non-singleton query fraction as the horizontal coordinate, hollow triangles use the idealized first-touch/no-eviction ceiling at the same agreement level, horizontal whiskers reproduce the reported 95% interval for the non-singleton fraction, and marker area represents pairable token-mass retention. Aircraft plus query type has 0.565 agreement, a 1.000 non-singleton fraction, a 0.921 first-touch ceiling, and 1.000 token-mass retention. The observed-feature signature reaches 1.000 agreement with a 0.991 non-singleton fraction, a 0.893 first-touch ceiling, and 0.996 token-mass retention; its top-4 evidence-identity purity remains 0.00935. Grouping by aircraft, top-4 evidence identity, and action bucket raises top-4 identity purity to 1.000 but reduces the three reuse proxies to 0.243, 0.140, and 0.286. The agreement values are defined on the fields used to form each signature, and singleton groups contribute agreement 1.000; they are not independent authorization evidence. The audit contains no request-order trace, layerwise KV identity, GPU saving, prefill FLOPs, TTFT, or independent expert authorization.",
    ),
    (
        "This work reframes aviation-maintenance RAG as admissibility-constrained state reuse before language. Its three contributions comprise two empirical mechanism audits and one deployment-target interface supported by a post-hoc proxy diagnostic: an observable separation between role allocation and conditional package support; an identity-aware multi-granularity organization in which persistent views supplement rather than replace query-local evidence; and a signed-prefix interface that restricts computational reuse to evidence-compatible classes. The LLM remains downstream of these operations and cannot promote repeated but unsupported context into a stronger maintenance claim.",
        "This work treats aviation-maintenance RAG as a problem of admissibility-constrained state reuse before language. Two empirical mechanism audits and a deployment target support three contributions. The first distinguishes evidence-role allocation from conditional package support. The second organizes multiple documentary granularities while requiring persistent views to resolve back to query-local evidence. The third defines a signed-prefix interface whose reuse classes must satisfy an admissibility relation; the present evidence is a post-hoc grouping diagnostic, not an implemented cache. The LLM receives the admitted state downstream and cannot use repeated but unsupported context to strengthen a maintenance claim.",
    ),
    (
        "The reported results are bounded to source-derived fair-path queries, pre-language package diagnostics, documentary identity proxies, and post-hoc signed-prefix grouping. They do not establish semantic mutual information, natural-query generalization, expert action authorization, actual Transformer KV-cache hits, or runtime acceleration. Future work should jointly evaluate natural maintenance questions, independent revision and authority labels, stale-cache invalidation, admissible-support retention, downstream claim entailment, and real prefill/KV-cache behavior under production request sequences.",
        "The reported results are bounded to source-derived fair-path queries, pre-language package diagnostics, documentary identity proxies, and post-hoc grouping statistics. The signed-prefix analysis measures non-singleton group membership, idealized first-touch ceilings, observed-feature agreement, and pairable token mass. It establishes neither semantic mutual information nor natural-query generalization, expert action authorization, actual Transformer KV-cache hits, or runtime acceleration. Future work should jointly evaluate natural maintenance questions, independent revision and authority labels, stale-cache invalidation, admissible-support retention, downstream claim entailment, and real prefill/KV-cache behavior under production request sequences.",
    ),
]


FIGURES = [
    (14, "image23.png", "fig18_01_role_mass_transport_volcano.png", "Observable evidence-role mass transport relative to the visible-demand proxy, with BH-FDR diagnostics."),
    (15, "image24.png", "fig18_02_fixed_package_loo_sensitivity_volcano.png", "Fixed-package leave-one-out alignment sensitivity by evidence role, with BH-FDR diagnostics."),
    (16, "image25.png", "fig18_03_role_transport_vs_loo_sensitivity.png", "Joint plane separating evidence-role selection transport from fixed-package alignment sensitivity."),
    (17, "image26.png", "fig18_04_exact_text_cluster_loo_raincloud.png", "Raincloud distributions of exact-query-text-cluster leave-one-out sensitivity by evidence role."),
    (18, "image27.png", "fig19_01_documentary_ambiguity_raincloud.png", "Documentary identity ambiguity across Page, Section, and Component-family groupings for Boeing and Cessna."),
    (19, "image28.png", "fig19_02_view_overlap_and_conditional_rescue_matrix.png", "Matrix of cross-view Top-10 overlap and direction-specific conditional rescue rates."),
    (20, "image29.png", "fig19_03_single_view_hit_intersections.png", "UpSet analysis of accepted-hit intersections across Detail, Page, Section, and Profile views."),
    (21, "image30.png", "fig19_04_persistent_minus_query_local_forest.png", "Forest plot of persistent-minus-query-local hit differences with cluster-bootstrap intervals."),
    (22, "image31.png", "fig20_01_router_numerator_outcome_plane.png", "Query-local and persistent router numerators stratified by observed hit outcome."),
    (23, "image32.png", "fig20_02_persistent_weight_quintile_outcomes.png", "Outcome composition across persistent-weight quintiles."),
    (24, "image33.png", "fig20_03_persistent_rescue_precision_recall.png", "Stepwise precision-recall curves for sample-internal identification of persistent-only rescue cases."),
    (25, "image34.png", "fig20_04_persistent_rescue_branch_budget.png", "Stepwise branch-budget trade-off between persistent-only rescue capture and precision."),
    (26, "image35.png", "fig20_05_entropy_discrimination_forest.png", "Within-view negative-entropy ROC-AUC estimates with separate AP and prevalence statistics."),
    (27, "image36.png", "fig27_equivalence_class_pairability_plane.png", "Input-side equivalence-class diagnostic comparing non-singleton query membership, idealized one-cold-start ceilings, observed-feature agreement, and pairable token-mass retention; not a measured GPU KV-cache hit rate."),
]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def visible_text(paragraph: etree._Element) -> str:
    return "".join(paragraph.xpath(".//w:t/text()", namespaces=NS))


def patch_document_xml(xml_bytes: bytes) -> tuple[bytes, list[dict[str, object]]]:
    parser = etree.XMLParser(remove_blank_text=False, resolve_entities=False)
    root = etree.fromstring(xml_bytes, parser)
    body_paragraphs = root.xpath("/w:document/w:body/w:p", namespaces=NS)
    audit: list[dict[str, object]] = []

    for old, new in REPLACEMENTS:
        matches = [(index, paragraph) for index, paragraph in enumerate(body_paragraphs) if visible_text(paragraph) == old]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one paragraph match, found {len(matches)} for: {old[:120]!r}")
        index, paragraph = matches[0]
        if paragraph.xpath(".//m:oMath | .//m:oMathPara | .//w:hyperlink | .//w:bookmarkStart | .//w:bookmarkEnd", namespaces=NS):
            raise RuntimeError(f"Protected structure found in target paragraph {index}")
        text_nodes = paragraph.xpath("./w:r/w:t", namespaces=NS)
        if len(text_nodes) != 1:
            raise RuntimeError(f"Paragraph {index} has {len(text_nodes)} direct text nodes; expected one")
        run = text_nodes[0].getparent()
        colors = run.xpath("./w:rPr/w:color/@w:val", namespaces=NS)
        if colors != [BLUE]:
            raise RuntimeError(f"Paragraph {index} is not explicitly blue: {colors}")
        text_nodes[0].text = new
        audit.append({"body_paragraph_index": index, "old_chars": len(old), "new_chars": len(new)})

    for figure_no, _media_name, _source_name, description in FIGURES:
        matches = root.xpath(f"//wp:docPr[@name='Figure {figure_no}']", namespaces=NS)
        if len(matches) != 1:
            raise RuntimeError(f"Expected one wp:docPr for Figure {figure_no}, found {len(matches)}")
        docpr = matches[0]
        inline = docpr.getparent()
        docpr.set("descr", description)
        cnvpr = inline.xpath(".//pic:cNvPr", namespaces=NS)
        if len(cnvpr) != 1:
            raise RuntimeError(f"Expected one pic:cNvPr for Figure {figure_no}, found {len(cnvpr)}")
        cnvpr[0].set("descr", description)

    return etree.tostring(root, encoding="UTF-8", xml_declaration=True, standalone=True), audit


def verify_figure_aspects(base_docx: Path, figures_dir: Path) -> list[dict[str, object]]:
    parser = etree.XMLParser(remove_blank_text=False, resolve_entities=False)
    with ZipFile(base_docx, "r") as archive:
        root = etree.fromstring(archive.read("word/document.xml"), parser)
    checks: list[dict[str, object]] = []
    for figure_no, _media_name, source_name, _description in FIGURES:
        docpr = root.xpath(f"//wp:docPr[@name='Figure {figure_no}']", namespaces=NS)[0]
        inline = docpr.getparent()
        extent = inline.xpath("./wp:extent", namespaces=NS)[0]
        stored_ratio = int(extent.get("cx")) / int(extent.get("cy"))
        source = figures_dir / source_name
        with Image.open(source) as image:
            width, height = image.size
        image_ratio = width / height
        relative_error = abs(image_ratio - stored_ratio) / stored_ratio
        if relative_error > 0.001:
            raise RuntimeError(
                f"Figure {figure_no} aspect mismatch: image={image_ratio:.6f}, stored={stored_ratio:.6f}"
            )
        checks.append(
            {
                "figure": figure_no,
                "source": source_name,
                "pixels": [width, height],
                "image_ratio": image_ratio,
                "stored_ratio": stored_ratio,
                "relative_error": relative_error,
            }
        )
    return checks


def patch_docx(base_docx: Path, figures_dir: Path, output_docx: Path, audit_json: Path) -> None:
    if sha256_file(base_docx) != EXPECTED_BASE_SHA256:
        raise RuntimeError("Base DOCX hash does not match the audited manuscript")
    aspect_checks = verify_figure_aspects(base_docx, figures_dir)
    figure_payload = {
        f"word/media/{media_name}": (figures_dir / source_name).read_bytes()
        for _figure_no, media_name, source_name, _description in FIGURES
    }

    output_docx.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(base_docx, "r") as source_archive:
        patched_xml, prose_audit = patch_document_xml(source_archive.read("word/document.xml"))
        modified = {"word/document.xml", *figure_payload.keys()}
        with tempfile.NamedTemporaryFile(
            prefix=f".{output_docx.stem}.", suffix=".tmp.docx", dir=output_docx.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
        try:
            with ZipFile(temporary, "w", compression=ZIP_DEFLATED) as output_archive:
                output_archive.comment = source_archive.comment
                for info in source_archive.infolist():
                    if info.filename == "word/document.xml":
                        payload = patched_xml
                    elif info.filename in figure_payload:
                        payload = figure_payload[info.filename]
                    else:
                        payload = source_archive.read(info.filename)
                    output_archive.writestr(info, payload)
            with ZipFile(temporary, "r") as check_archive:
                if check_archive.testzip() is not None:
                    raise RuntimeError("Generated DOCX failed ZIP integrity testing")
                if len(check_archive.namelist()) != len(set(check_archive.namelist())):
                    raise RuntimeError("Generated DOCX contains duplicate ZIP members")
            temporary.replace(output_docx)
        finally:
            temporary.unlink(missing_ok=True)

    audit = {
        "base_docx": str(base_docx),
        "base_sha256": sha256_file(base_docx),
        "output_docx": str(output_docx),
        "output_sha256": sha256_file(output_docx),
        "modified_zip_members": sorted(modified),
        "prose_replacements": prose_audit,
        "figure_aspect_checks": aspect_checks,
    }
    audit_json.parent.mkdir(parents=True, exist_ok=True)
    audit_json.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch the audited IntentRAG DOCX without reconstructing Word layout.")
    parser.add_argument("--base-docx", type=Path, required=True)
    parser.add_argument("--figures-dir", type=Path, required=True)
    parser.add_argument("--output-docx", type=Path, required=True)
    parser.add_argument("--audit-json", type=Path, required=True)
    args = parser.parse_args()
    patch_docx(
        args.base_docx.resolve(),
        args.figures_dir.resolve(),
        args.output_docx.resolve(),
        args.audit_json.resolve(),
    )
    print(json.dumps({"status": "pass", "output": str(args.output_docx.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
