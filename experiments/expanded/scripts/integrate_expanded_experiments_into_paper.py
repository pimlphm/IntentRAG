from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path

import pandas as pd
from PIL import Image
from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor
from docx.text.paragraph import Paragraph


EXP = Path(__file__).resolve().parents[1]
DEFAULT_BASE = Path(
    r"${USER_HOME}\Desktop\合作_西工大\师叔\第二篇\新建文件夹"
    r"\IntentRAG_SEPia_Nature_20260924"
    r"\IntentRAG-0802-v3_SEPia_Nature英文图优化_蓝字版.docx"
)
EXPECTED_BASE_SHA256 = "1A0FC8CB96EA73AB795228F5475BFBADA77EFBE12B1753C067C95DE203658FE4"
BLUE = RGBColor(0x00, 0x00, 0xFF)


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def zip_media_hashes(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with zipfile.ZipFile(path) as zf:
        for name in zf.namelist():
            if name.startswith("word/media/"):
                out[name] = hashlib.sha256(zf.read(name)).hexdigest()
    return out


def blue_run(paragraph: Paragraph, text: str, *, bold: bool = False, italic: bool = False):
    run = paragraph.add_run(text)
    run.font.color.rgb = BLUE
    run.bold = bold
    run.italic = italic
    return run


def replace_text(paragraph: Paragraph, text: str) -> None:
    for run in list(paragraph.runs):
        paragraph._p.remove(run._r)
    blue_run(paragraph, text)


def paragraph_before(anchor: Paragraph, text: str = "", style: str | None = None) -> Paragraph:
    element = OxmlElement("w:p")
    anchor._p.addprevious(element)
    paragraph = Paragraph(element, anchor._parent)
    if style:
        paragraph.style = style
    if text:
        blue_run(paragraph, text)
    return paragraph


def page_break_before(anchor: Paragraph) -> Paragraph:
    p = paragraph_before(anchor, style="Normal")
    run = p.add_run()
    run.add_break(WD_BREAK.PAGE)
    return p


def find_exact(doc: Document, text: str) -> Paragraph:
    matches = [p for p in doc.paragraphs if p.text.strip() == text]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one paragraph {text!r}; found {len(matches)}")
    return matches[0]


def find_prefix(doc: Document, prefix: str) -> Paragraph:
    matches = [p for p in doc.paragraphs if p.text.strip().startswith(prefix)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one paragraph beginning {prefix!r}; found {len(matches)}")
    return matches[0]


def set_cell_shading(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def set_cell_margins(cell, value: int = 75) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_mar = OxmlElement("w:tcMar")
    for edge in ("top", "left", "bottom", "right"):
        node = OxmlElement(f"w:{edge}")
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")
        tc_mar.append(node)
    tc_pr.append(tc_mar)


def set_table_borders(table) -> None:
    for row in table.rows:
        for cell in row.cells:
            tc_pr = cell._tc.get_or_add_tcPr()
            borders = OxmlElement("w:tcBorders")
            for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
                node = OxmlElement(f"w:{edge}")
                node.set(qn("w:val"), "single")
                node.set(qn("w:sz"), "4")
                node.set(qn("w:color"), "D9D9D9")
                borders.append(node)
            tc_pr.append(borders)


def build_method_table(doc: Document, anchor: Paragraph) -> None:
    summary = pd.read_csv(EXP / "results" / "all_retrieval_summary.csv")
    summary = summary[summary["split"].eq("fresh_locked_test")].copy()
    replay = pd.read_csv(
        EXP / "results" / "memory_replay_token_budget" / "replay_order_summary.csv"
    )
    replay = replay[
        replay["split"].eq("fresh_locked_test")
        & replay["capacity_source_tokens"].eq(4096)
        & replay["order"].eq("random")
    ]
    reuse = replay.groupby("method", as_index=True)["token_weighted_reused_fraction"].mean()

    def class_name(method: str) -> str:
        if method.startswith("Neural") or "mMARCO" in method:
            return "Pretrained neural"
        if any(
            key in method
            for key in (
                "Agentic",
                "GraphRAG",
                "GaRAGe",
                "graph informed",
                "Evidence-flow",
                "FD-GRAG",
            )
        ):
            return "Project control"
        if any(key in method for key in ("Multi-granularity", "IntentRAG", "Short-term", "Long-term")):
            return "State/granularity"
        return "Lexical/structural"

    rows = []
    for row in summary.sort_values("accepted_hit_at_10", ascending=False).itertuples():
        rows.append(
            [
                row.method,
                class_name(row.method),
                f"{row.accepted_hit_at_10:.3f}",
                f"{row.all_quote_support:.3f}",
                f"{reuse.get(row.method, float('nan')):.3f}",
            ]
        )

    table = doc.add_table(rows=1, cols=5)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    widths = [Cm(5.0), Cm(2.8), Cm(2.1), Cm(2.3), Cm(2.3)]
    headers = [
        "Method",
        "Implementation class",
        "Accepted Hit@10",
        "All required quotes",
        "Reuse at 4,096 source tokens",
    ]
    for i, (cell, width, text) in enumerate(zip(table.rows[0].cells, widths, headers)):
        cell.width = width
        set_cell_shading(cell, "D9EAF7")
        set_cell_margins(cell)
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT if i < 2 else WD_ALIGN_PARAGRAPH.CENTER
        run = blue_run(p, text, bold=True)
        run.font.size = Pt(7.5)
    header_prop = OxmlElement("w:tblHeader")
    table.rows[0]._tr.get_or_add_trPr().append(header_prop)

    for row_index, values in enumerate(rows, start=1):
        cells = table.add_row().cells
        for i, (cell, width, value) in enumerate(zip(cells, widths, values)):
            cell.width = width
            set_cell_margins(cell)
            set_cell_shading(cell, "F5F9FC" if row_index % 2 == 0 else "FFFFFF")
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if i < 2 else WD_ALIGN_PARAGRAPH.CENTER
            run = blue_run(p, str(value))
            run.font.size = Pt(7.2)
    set_table_borders(table)
    anchor._p.addprevious(table._tbl)


FIGURES = [
    {
        "number": 28,
        "root": "figures",
        "stem": "legacy_test_accepted_hit_forest_word",
        "lead": "The legacy test was rerun without exposing target-derived cues or a locked retrieval track. The interval unit is the normalized question-text cluster, so duplicated records do not acquire extra inferential weight.",
        "caption": "All 33 executed methods on the legacy locked test after removal of target-derived access shortcuts. Points are question-cluster means of accepted evidence Hit@10 and bars are 95% cluster-bootstrap intervals over 191 clusters. The historical project labels denote local controls rather than official reproductions.",
    },
    {
        "number": 29,
        "root": "figures",
        "stem": "fresh_test_accepted_hit_forest_word",
        "lead": "The fresh source-derived test gives a more difficult distribution shift. Balanced IntentRAG remains competitive, although True BM25 is stronger on this endpoint; the result does not support a universal ranking of architectures.",
        "caption": "Accepted evidence Hit@10 for all 33 methods on 214 fresh source-derived questions. True BM25 reaches 0.972, balanced IntentRAG 0.949, multi-granularity plus mMARCO CE@50 0.930, and the matched dense plus CE@50 route 0.827. The fresh set contains 212 Cessna and two Boeing questions.",
    },
    {
        "number": 30,
        "root": "figures",
        "stem": "fresh_required_quote_support_forest_word",
        "lead": "A single acceptable hit can hide incomplete multi-source support. This endpoint measures the fraction of frozen required source-quote groups that the returned package actually covers.",
        "caption": "Required-quote support on the fresh source-derived test. The scoring assets are applied after retrieval and distinguish partial from complete support. This is an evidence-coverage endpoint, not answer correctness or action authorization.",
    },
    {
        "number": 31,
        "root": "figures",
        "stem": "fresh_granularity_paired_support_shift_word",
        "lead": "The same fresh questions were compared under detail-only, persistent-only, fixed multi-view, entropy-routed, and constraint-routed access. Coarse persistent views lose local support, and the tested dynamic routers do not create an automatic gain over detail retrieval.",
        "caption": "Paired change in required-source support relative to Detail-only access. Bootstrap resampling preserves the pairing by question cluster. Negative shifts for persistent and several multi-view policies show that granularity expansion requires a leaf-resolution step; entropy concentration alone does not establish useful information gain.",
    },
    {
        "number": 32,
        "root": "figures",
        "stem": "entropy_support_fresh_locked_test",
        "lead": "Retrieval-score entropy describes how concentrated the candidate distribution is. A wrong candidate can also dominate its competitors, so entropy must be tested against support rather than treated as confidence by definition.",
        "caption": "Binned association between balanced IntentRAG detail-score entropy and post-retrieval accepted support on the fresh test. Empty bins remain empty, marker area records the number of clusters, and intervals are within-bin cluster bootstraps. The quantity is neither semantic entropy nor mutual information.",
    },
    {
        "number": 33,
        "root": "figures",
        "stem": "reuse_capacity_mechanistic_fresh_locked_test_word",
        "lead": "Past-only source-module replay exposes a second distribution: which retrieved sources recur under a request sequence. Repeated profile nodes can yield high reuse while carrying little query-specific support.",
        "caption": "Exact-source identity reuse under 16, 64, and 256 cached modules for selected mechanisms. Lookup precedes update and every split starts cold. The observed range is over five randomized request orders. Equal module count is not equal storage cost, so this analysis is supplementary to the token-budget replay.",
    },
    {
        "number": 34,
        "root": "figures",
        "stem": "quality_reuse_fresh_locked_test",
        "lead": "Support and recurrence form separate axes. The high-reuse, low-support profile route makes the failure mode visible: a cache can repeatedly return stable background while missing the clause needed by the current query.",
        "caption": "Fresh-test accepted evidence Hit@10 versus exact-source token-mass reuse at the 64-module replay setting. Every method remains present. Reuse is a workload property and does not certify answer quality, admissibility, or GPU prefix reuse.",
    },
    {
        "number": 35,
        "root": "figures",
        "stem": "fresh_test_accepted_support_granularity_word",
        "lead": "This conditional composition shows the documentary levels of sources that were actually accepted after retrieval. It does not turn a returned parent node into support for all of its descendants.",
        "caption": "Source-token composition of post-hoc accepted evidence by documentary granularity on the fresh test. Right-hand counts give pooled source-unit occurrences. The denominator contains accepted support only, so the plot must be read with the coverage forests rather than as an unconditional success rate.",
    },
    {
        "number": 36,
        "root": "figures_qa_kv",
        "stem": "fresh_answer_quote_support_word",
        "lead": "The generation experiment produced real local-model responses, but the system prompt allowed an empty quote list. The resulting all-answer score jointly reflects quote emission, parser recognition, and verbatim support in the supplied context.",
        "caption": "Mean context-supported quote score for the 32 fresh questions per method; an answer with no parser-recognized quotation receives zero. Across all 2,112 legacy and fresh cells, only 11 contained a recognized quotation and eight matched supplied source text. The plot is not semantic answer accuracy or groundedness.",
    },
    {
        "number": 37,
        "root": "figures_qa_kv",
        "stem": "actual_kv_reuse_by_prefix_layout",
        "lead": "The runtime study replaces proxy overlap with the engine's reused-input count. Canonical state-first ordering raises reuse on average, whereas query-first and hierarchy-first layouts expose different breakpoints in the shared prefix.",
        "caption": "Actual input-KV reuse fraction for 594 profiles from six visible-query pairs, 33 retrieval methods, and three prefix layouts. Means are 0.039 for query-first, 0.125 for canonical state-first, and 0.052 for hierarchical state-first. Shared system and scope headers are included, and only six query pairs are independent clusters.",
    },
    {
        "number": 38,
        "root": "figures_qa_kv",
        "stem": "native_prefix_overlap_vs_actual_kv",
        "lead": "Native token-level longest-common-prefix length predicts the amount of input computation that can be reused, but source identity and documentary compatibility do not guarantee a continuous matching prefix.",
        "caption": "Native seed-target prefix overlap versus engine-reported reused input tokens. Cold-target, cold-seed, and warm-target requests use a fixed GPU slot with the global RAM prompt cache disabled. The equality line checks the counting mechanism; it does not test evidence validity or isolated speedup.",
    },
    {
        "number": 39,
        "root": "figures_token_budget",
        "stem": "fresh_quality_reuse_4096_source_tokens",
        "lead": "A fixed count of cache objects favors large recurring nodes, so the stricter replay holds stored source-body tokens constant. This reverses part of the apparent reuse advantage of large causal-graph objects while leaving the profile route's high-reuse, low-support pattern.",
        "caption": "Fresh accepted evidence Hit@10 versus past-only reuse under a hard 4,096-source-token LRU budget. Items larger than the capacity are not admitted, but retrieval output remains unchanged. The budget excludes keys and metadata and therefore is not a RAM- or KV-byte budget.",
    },
    {
        "number": 40,
        "root": "figures_token_budget",
        "stem": "fresh_reuse_capacity_source_tokens",
        "lead": "Increasing the source-token budget changes which evidence objects survive between requests. The curve separates capacity effects from the identity and length distribution of each method's retrieved sources.",
        "caption": "Past-only exact-source reuse under 1,024, 4,096, and 16,384 stored source-body tokens. Points are means across five randomized histories and bars show the observed range. Oversize non-admission and cache occupancy are audited at every request.",
    },
    {
        "number": 41,
        "root": "figures_qa_kv",
        "stem": "actual_kv_vs_materialized_reference_support",
        "lead": "Prefix organization also changes which material fits before the context limit. Canonical ordering moves the computation point rightward while shifting source-body mass toward stable component profiles and away from some query-local reference anchors.",
        "caption": "Actual input-KV reuse versus the materialized reference-anchor visibility proxy. Points are means over 33 methods within each of six query pairs and three layouts. Canonical state-first increases mean cache reuse while reducing reference-anchor visibility from 0.721 to 0.593. Visibility is an input-support proxy, not answer correctness or admissibility.",
    },
    {
        "number": 42,
        "root": "figures_qa_kv",
        "stem": "paired_cache_gain_vs_materialized_support_loss",
        "lead": "The paired view gives the central systems trade-off: the same layout change can reuse more prefill computation and expose less of the reference support required by the new query.",
        "caption": "Within-method and within-query-pair change in actual cache reuse and materialized reference support. Positive horizontal values indicate reuse gains; positive vertical values indicate support loss. The canonical and hierarchical source bodies are identical for 198 of 198 profiles, so their cache difference is caused by leading metadata and total input structure rather than improved evidence content.",
    },
]


def resolve_figure(spec: dict) -> Path:
    path = EXP / spec["root"] / f"{spec['stem']}.png"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def add_figure_before(doc: Document, anchor: Paragraph, spec: dict) -> None:
    page_break_before(anchor)
    lead = paragraph_before(anchor, spec["lead"], "SCI正文")
    lead.paragraph_format.space_after = Pt(4)

    image_path = resolve_figure(spec)
    width_px, height_px = Image.open(image_path).size
    max_width_cm = 14.35
    max_height_cm = 19.4
    width_cm = min(max_width_cm, max_height_cm * width_px / height_px)

    image_p = paragraph_before(anchor, style="Normal")
    image_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    image_p.paragraph_format.keep_with_next = True
    image_p.paragraph_format.space_after = Pt(2)
    run = image_p.add_run()
    run.add_picture(str(image_path), width=Cm(width_cm))
    doc_pr = run._r.xpath(".//wp:docPr")
    if doc_pr:
        doc_pr[0].set("descr", f"Figure {spec['number']}: {spec['caption']}")

    caption = paragraph_before(anchor, style="SCI正文")
    caption.paragraph_format.space_after = Pt(6)
    blue_run(caption, f"Fig. {spec['number']}. {spec['caption']}")


def add_methodology(doc: Document, anchor: Paragraph) -> None:
    heading = paragraph_before(anchor, "Expanded benchmark, answer-execution, reuse, and GPU protocol", "SCI 三级标题")
    heading.paragraph_format.keep_with_next = True
    paragraphs = [
        "The expanded evaluation freezes 690 query records from four non-overlapping question-text splits and executes 33 methods for every record, yielding 22,770 retrieval outputs with no missing query-method cell. The fresh test contains 214 source-derived questions generated from fixed source groups and retained after deterministic quote checks and a local-model support screen. Its primary target pages do not overlap the legacy primary pages, although nine fresh targets occur in the broader legacy accepted-equivalence union. The fresh set is heavily Cessna-weighted (212 Cessna, two Boeing), so aggregate results are not an aircraft-balanced estimate.",
        "The comparison registry retains every legacy project control and adds true Okapi BM25, local documentary views, fixed and entropy-routed multi-granularity policies, a real multilingual dense encoder, and an mMARCO cross-encoder. All methods return at most ten distinct source objects. Labels such as Controlled GraphRAG, Agentic RAG, and GaRAGe-style RAG refer to the project's executable controls rather than official reproductions of the named systems. Point estimates are reported both by record and by deduplicated question-text cluster, and paired comparisons resample the same clusters for both methods.",
        "Answer execution uses 64 frozen questions (32 legacy and 32 fresh), one Qwen2.5 3B generator, temperature zero, a fixed seed, 3,000 actual Qwen source tokens, and a 128-token output limit. The run contains 2,112 query-method cells, 1,933 API attempts, 179 exact-payload aliases, and one retained transport failure. JSON-and-answer parsing and verbatim quote support are mechanical endpoints. The subsequent local 7B semantic audit produced zero model responses because Windows blocked the socket with error 10013, so no semantic answer-accuracy estimate is reported.",
        "Reuse is evaluated at three levels. Exact-source LRU replay processes every method under six request orders and cold-starts each split; a second replay enforces hard capacities of 1,024, 4,096, and 16,384 complete source-body tokens and refuses oversize admission. The GPU study executes six fixed visible-query pairs for every method under query-first, canonical state-first, and hierarchical state-first layouts. A cold target, cold seed, and warm target are issued in one fixed slot with global RAM prompt-cache restoration disabled. The reused input count is read from the engine and checked against newly computed input tokens for all 594 profiles.",
        "The GPU and answer experiments use different frozen serialization rules and context budgets. Their values therefore cannot be joined as if they measured one identical state. The materialization audit parses every emitted source block, verifies source-text hashes, records truncation, and tests whether long reference anchors remain in the actual input. This makes the cache-versus-support trade-off observable while preserving the distinction between input visibility, answer correctness, and evidence admissibility.",
    ]
    for text in paragraphs:
        p = paragraph_before(anchor, text, "SCI正文")
        p.paragraph_format.space_after = Pt(5)


def add_expanded_results(doc: Document, anchor: Paragraph) -> None:
    page_break_before(anchor)
    heading = paragraph_before(anchor, "Expanded cross-benchmark and runtime validation", "SCI 二级标题")
    heading.paragraph_format.keep_with_next = True
    intro = [
        "The expanded study separates four endpoints that were conflated in the earlier pipeline: retrieval support, generated-output mechanics, source-state recurrence, and actual GPU prefix reuse. Balanced IntentRAG reaches 0.949 accepted Hit@10 and 0.902 complete required-quote support on the fresh set. True BM25 reaches 0.972 and 0.916. Multi-granularity plus mMARCO CE@50 reaches 0.930 and 0.869, compared with 0.827 and 0.780 for the matched dense plus CE@50 route. The multi-view candidate pool therefore improves the matched neural baseline in this corpus, but it does not dominate the strongest lexical method.",
        "The memory result is more selective. Long-term fixed retrieval reaches 0.248 accepted Hit@10 and Profile-resolved retrieval 0.019, whereas the profile route can still exhibit high historical source reuse. Under the hard 4,096-source-token budget, profile reuse remains about 0.782 while balanced IntentRAG and True BM25 reuse only about 0.004 and 0.001, respectively. The persistent layer concentrates repeated background; it cannot replace the leaf evidence needed by the current query.",
        "Actual GPU measurements confirm that serialization matters. Canonical state-first ordering raises mean reused-input fraction from 0.039 to 0.125, a 3.24-fold ratio in this workload. The same ordering reduces the materialized reference-anchor proxy from 0.721 to 0.593 and the emitted accepted-ID proxy from 0.732 to 0.586. The empirical contribution is consequently a constrained design problem: computation should be reused only within compatible evidence states, and prefix optimization must preserve the query-local support atoms before it rearranges the remaining state.",
    ]
    for text in intro:
        p = paragraph_before(anchor, text, "SCI正文")
        p.paragraph_format.space_after = Pt(5)

    caption = paragraph_before(
        anchor,
        "Table 8. Fresh retrieval support and past-only reuse for all 33 executed methods.",
        "Normal",
    )
    caption.paragraph_format.keep_with_next = True
    build_method_table(doc, anchor)
    note = paragraph_before(
        anchor,
        "The final column uses the hard 4,096-source-token replay averaged over five randomized histories. It controls stored source-body tokens rather than total RAM or KV bytes. Project-control labels identify the local implementation family and do not imply official-system reproduction.",
        "SCI正文",
    )
    note.paragraph_format.space_after = Pt(5)

    for spec in FIGURES:
        add_figure_before(doc, anchor, spec)

    synthesis = paragraph_before(
        anchor,
        "Taken together, Figures 28-42 reject a one-dimensional account of state quality. Retrieval support can remain high while cross-query source reuse is low; persistent profiles can be highly recurrent while failing to supply the local evidence; and canonical ordering can increase actual KV reuse while decreasing the reference material that survives context materialization. An admissibility-signed prefix is useful only after the admissible evidence atoms have been selected and retained. The signature restricts reuse classes; it does not create missing authority labels, repair an unsuitable source package, or guarantee that the resulting answer is correct.",
        "SCI正文",
    )
    synthesis.paragraph_format.space_after = Pt(6)


def build_sep_review(path: Path) -> None:
    text = """SEPIA REVIEW — academic manuscript, local in-place revision
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
"""
    path.write_text(text, encoding="utf-8")


def audit_document(base: Path, output: Path, audit_path: Path) -> dict:
    base_media = zip_media_hashes(base)
    output_media = zip_media_hashes(output)
    base_hashes = set(base_media.values())
    output_hashes = set(output_media.values())
    figure_hashes = {file_hash(resolve_figure(spec)) for spec in FIGURES}

    doc = Document(output)
    all_text = "\n".join(p.text for p in doc.paragraphs)
    captions = []
    for paragraph in doc.paragraphs:
        match = re.match(r"^Fig\.\s+(\d+)\.\s", paragraph.text.strip())
        if match:
            captions.append(int(match.group(1)))

    blue_body_paragraphs = 0
    for paragraph in doc.paragraphs:
        text_runs = [run for run in paragraph.runs if run.text.strip()]
        if text_runs and all(run.font.color.rgb == BLUE for run in text_runs):
            blue_body_paragraphs += 1

    with zipfile.ZipFile(output) as zf:
        xml = zf.read("word/document.xml")
        math_objects = xml.count(b"<m:oMath")
        drawing_count = xml.count(b"<w:drawing")

    report = {
        "status": "pass",
        "base": str(base),
        "base_sha256": file_hash(base),
        "output": str(output),
        "output_sha256": file_hash(output),
        "paragraphs": len(doc.paragraphs),
        "tables": len(doc.tables),
        "drawings": drawing_count,
        "math_objects_raw_xml_count": math_objects,
        "figure_caption_numbers": captions,
        "captions_1_to_42_in_order": captions == list(range(1, 43)),
        "blue_body_paragraphs": blue_body_paragraphs,
        "base_media_count": len(base_media),
        "output_media_count": len(output_media),
        "all_base_media_preserved_by_hash": base_hashes.issubset(output_hashes),
        "all_15_new_figure_images_embedded_by_hash": figure_hashes.issubset(output_hashes),
        "placeholder_evid_code_count": len(re.findall(r"EVID_[A-Z0-9]+", all_text)),
        "render_status": "pending_visual_render_gate",
    }
    checks = [
        report["captions_1_to_42_in_order"],
        len(doc.tables) == 8,
        drawing_count == 51,
        report["all_base_media_preserved_by_hash"],
        report["all_15_new_figure_images_embedded_by_hash"],
        report["placeholder_evid_code_count"] == 0,
    ]
    if not all(checks):
        report["status"] = "fail"
    audit_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if report["status"] != "pass":
        raise RuntimeError(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument(
        "--out-root",
        type=Path,
        default=EXP / "latest_paper_integration_20261015",
    )
    args = parser.parse_args()
    base = args.base.resolve()
    out_root = args.out_root.resolve()
    deliverables = out_root / "deliverables"
    reports = out_root / "reports"
    deliverables.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    actual_base_hash = file_hash(base)
    if actual_base_hash.lower() != EXPECTED_BASE_SHA256.lower():
        raise RuntimeError(
            f"Base manuscript hash mismatch: {actual_base_hash} != {EXPECTED_BASE_SHA256}"
        )

    doc = Document(base)

    abstract = find_prefix(doc, "External knowledge access in aviation maintenance")
    replace_text(
        abstract,
        "Aviation-maintenance language systems must decide which retrieved state may support the current claim before generation begins. We study this problem as admissibility-constrained state reuse and use IntentRAG as a retrieval-based implementation. The expanded evaluation executes 33 access methods over 690 records (22,770 query-method outputs), generates 2,112 local-model answer cells on 64 frozen questions, replays exact-source memory under object and source-token budgets, and measures actual GPU input-KV reuse for 594 prefix profiles. On 214 fresh source-derived questions, balanced IntentRAG reaches 0.949 accepted evidence Hit@10 and 0.902 complete required-quote support. Multi-granularity retrieval with a matched cross-encoder reaches 0.930 Hit@10 versus 0.827 for dense retrieval with the same reranker, while True BM25 remains stronger at 0.972. Persistent profile retrieval is highly reusable under some request histories but provides little query-local support. Canonical state-first ordering raises mean measured input-KV reuse from 0.039 to 0.125, yet reduces materialized reference-anchor visibility from 0.721 to 0.593. These results support three conclusions: evidence allocation and conditional support are distinct; persistent views require resolution to local documentary evidence; and computational reuse must be optimized inside, rather than in place of, the evidence-compatibility boundary. The questions are source-derived and the attempted semantic answer audit returned no valid model judgments, so the reported endpoints do not constitute expert-certified maintenance accuracy.",
    )

    contribution3 = find_prefix(doc, "(3) Admissibility-constrained prefix reuse.")
    replace_text(
        contribution3,
        "(3) Admissibility-constrained prefix reuse. We define a deployment interface in which computation can be reused only inside an evidence-compatibility class, then test the computational side with a real GPU prefix protocol. Canonical state-first ordering increases reused input tokens in the measured workload, but the materialization audit shows that a higher cache fraction can coincide with lower visibility of query-local support. The signature therefore constrains reuse; it cannot compensate for missing revision, configuration, condition, provenance, or authority labels.",
    )

    old_comparator = find_prefix(doc, "The legacy comparison atlas includes word TF-IDF")
    replace_text(
        old_comparator,
        "The expanded registry executes all 15 legacy project controls together with true Okapi BM25, local documentary views, fixed and entropy-routed multi-granularity policies, a real multilingual dense encoder, and an mMARCO cross-encoder. The old BM25/word label remains in the registry for traceability but denotes word TF-IDF cosine similarity. Controlled GraphRAG, Agentic RAG, and GaRAGe-style routes remain project implementations rather than official reproductions. The common Top-10 budget permits method-level comparison inside this corpus; it does not establish a state-of-the-art ranking across external systems.",
    )

    old_limit = find_prefix(doc, "The paper still does not establish semantic mutual information")
    replace_text(
        old_limit,
        "The expanded study measures real input-KV reuse, but it still does not establish semantic mutual information, calibrated online refusal, claim entailment, action correctness, expert authorization, or safety certification. The GPU endpoint is an engine count under six fixed query pairs and includes shared system and scope text. It must be interpreted with the materialized-support audit rather than as a direct measure of evidentiary value.",
    )

    old_kv_limit = find_prefix(doc, "These values are not actual Transformer KV-cache hits")
    replace_text(
        old_kv_limit,
        "The post-hoc equivalence-class values in Figure 27 remain input-side grouping diagnostics. The separate GPU experiment reported below measures actual reused input tokens, cache residency within one fixed slot, and native prefix overlap. It does not validate the completeness of the admissibility signature, isolated throughput acceleration, stale-revision invalidation, or maintenance authorization.",
    )

    results_heading = find_exact(doc, "Results")
    add_methodology(doc, results_heading)

    conclusion_heading = find_exact(doc, "Conclusion")
    add_expanded_results(doc, conclusion_heading)

    conclusion1 = find_prefix(doc, "This work treats aviation-maintenance RAG")
    replace_text(
        conclusion1,
        "This work treats aviation-maintenance RAG as a problem of selecting and reusing a bounded evidence state before language. The first contribution separates evidence-role allocation from conditional support inside the realized package. The second models Detail and Page as identity-preserving query-local evidence and Section and Profile as persistent navigation views that must resolve back to traceable source atoms. The third places actual prefix reuse inside an evidence-compatibility constraint. Across 33 methods, the multi-granularity neural route improves its matched dense baseline but does not exceed True BM25 on the fresh retrieval endpoint. Persistent profiles can be easy to reuse and poor at local support, while canonical state-first ordering can increase actual KV reuse and simultaneously remove relevant material under a fixed context budget.",
    )
    conclusion2 = find_prefix(doc, "The reported results are bounded to source-derived fair-path queries")
    replace_text(
        conclusion2,
        "The evidence remains bounded. The fresh questions are source-derived, aircraft coverage is imbalanced, project controls are not official external-system reproductions, and the 64-question generation study currently supports only mechanical output analysis. The attempted local semantic audit produced zero valid judgments because the model endpoint was blocked. Future work needs expert-written questions, independent revision and authority labels, new locked tests for any post-hoc interface improvement, and a controlled deployment study that measures support retention, stale-state invalidation, answer entailment, TTFT, memory use, and cache eviction together.",
    )

    output = deliverables / "IntentRAG_完整论文_扩展实验与状态复用验证_蓝字版_20261015.docx"
    doc.save(output)

    sep_report = reports / "sepia_refactor_stage1_and_changes.md"
    build_sep_review(sep_report)
    audit_path = reports / "integrated_docx_structural_audit.json"
    audit = audit_document(base, output, audit_path)

    lineage = {
        "status": "working_source_created_structural_audit_passed_visual_render_pending",
        "base": str(base),
        "base_sha256": actual_base_hash,
        "output": str(output),
        "output_sha256": audit["output_sha256"],
        "experiment_root": str(EXP),
        "retrieval_rows": 22770,
        "methods": 33,
        "qa_cells": 2112,
        "real_kv_profiles": 594,
        "new_figures": 15,
        "figure_numbers": [spec["number"] for spec in FIGURES],
        "all_new_or_replaced_text_blue": True,
        "semantic_answer_accuracy_status": "no_valid_judgments",
        "visual_render_status": "pending_permission_or_bundled_renderer",
        "delivery_status": "do_not_ship_as_final_until_every_page_is_rendered_and_inspected",
    }
    (reports / "paper_integration_lineage.json").write_text(
        json.dumps(lineage, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"output": str(output), "audit": audit}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
