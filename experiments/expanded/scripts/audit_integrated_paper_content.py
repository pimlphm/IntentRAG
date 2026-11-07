from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn


EXPECTED_FIGURES = list(range(1, 43))
EXPECTED_NEW_FIGURES = list(range(28, 43))
REQUIRED_CLAIMS = {
    "complete retrieval matrix": "33 methods for every record, yielding 22,770 retrieval outputs",
    "fresh split size": "The fresh test contains 214 source-derived questions",
    "fresh split imbalance": "212 Cessna, two Boeing",
    "answer matrix": "The run contains 2,112 query-method cells",
    "semantic audit boundary": "zero model responses because Windows blocked the socket with error 10013",
    "real KV profiles": "all 594 profiles",
    "balanced retrieval": "Balanced IntentRAG reaches 0.949 accepted Hit@10",
    "lexical comparator": "True BM25 reaches 0.972",
    "long-term boundary": "Long-term fixed retrieval reaches 0.248 accepted Hit@10",
    "profile boundary": "Profile-resolved retrieval 0.019",
    "measured cache shift": "from 0.039 to 0.125, a 3.24-fold ratio",
    "materialization trade-off": "from 0.721 to 0.593",
    "expert-certification disclaimer": "do not constitute expert-certified maintenance accuracy",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("docx", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    doc = Document(args.docx)
    paragraphs = [p.text.strip() for p in doc.paragraphs]
    body = "\n".join(paragraphs)
    figure_numbers = [
        int(match.group(1))
        for text in paragraphs
        for match in [re.match(r"^Fig\.\s+(\d+)\.", text)]
        if match
    ]
    new_caption_counts = {str(n): figure_numbers.count(n) for n in EXPECTED_NEW_FIGURES}

    last_table = doc.tables[-1]
    last_table_rows = [[cell.text.strip() for cell in row.cells] for row in last_table.rows]
    method_names = [row[0] for row in last_table_rows[1:] if row and row[0]]

    blue_paragraphs = 0
    for paragraph in doc.paragraphs:
        has_blue = False
        for run in paragraph.runs:
            color = run.font.color.rgb
            if color is not None and str(color).upper() in {"0000FF", "0070C0"}:
                has_blue = True
                break
        blue_paragraphs += int(has_blue)

    drawing_count = len(doc.element.xpath(".//w:drawing"))
    math_count = len(doc.element.xpath(".//m:oMath | .//m:oMathPara"))
    claim_checks = {label: needle in body for label, needle in REQUIRED_CLAIMS.items()}
    checks = {
        "captions_1_to_42_exactly_once_and_ordered": figure_numbers == EXPECTED_FIGURES,
        "new_captions_28_to_42_exactly_once": all(value == 1 for value in new_caption_counts.values()),
        "table_8_has_33_unique_methods": len(method_names) == 33 and len(set(method_names)) == 33,
        "all_required_claims_present": all(claim_checks.values()),
        "no_EVID_placeholders": not re.search(r"EVID_[A-Z0-9]+", body),
        "expected_51_drawings": drawing_count == 51,
        "expected_11_math_objects": math_count == 11,
        "blue_revision_text_present": blue_paragraphs > 0,
    }
    report = {
        "status": "pass" if all(checks.values()) else "fail",
        "docx": str(args.docx.resolve()),
        "sha256": sha256(args.docx),
        "paragraphs": len(doc.paragraphs),
        "tables": len(doc.tables),
        "drawings": drawing_count,
        "math_objects": math_count,
        "blue_body_paragraphs": blue_paragraphs,
        "figure_caption_numbers": figure_numbers,
        "new_caption_counts": new_caption_counts,
        "table_8_rows_including_header": len(last_table_rows),
        "table_8_unique_methods": len(set(method_names)),
        "claim_checks": claim_checks,
        "checks": checks,
        "visual_render_status": "not_assessed_by_this_structural_content_audit",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
