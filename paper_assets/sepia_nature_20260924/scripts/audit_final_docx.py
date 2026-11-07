from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from zipfile import ZipFile

from lxml import etree

from patch_intentrag_docx import BLUE, FIGURES, NS, REPLACEMENTS, sha256_bytes, visible_text


REL_NS = {"pr": "http://schemas.openxmlformats.org/package/2006/relationships"}


def canonical_hashes(nodes: list[etree._Element]) -> list[str]:
    return [
        hashlib.sha256(etree.tostring(node, method="c14n", exclusive=False, with_comments=False)).hexdigest().upper()
        for node in nodes
    ]


def package_bytes(path: Path) -> dict[str, bytes]:
    with ZipFile(path, "r") as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def analyse_docx(base: Path, final: Path, figures_dir: Path) -> dict[str, object]:
    base_entries = package_bytes(base)
    final_entries = package_bytes(final)
    expected_changed = {"word/document.xml", *[f"word/media/{item[1]}" for item in FIGURES]}
    changed = {
        name
        for name in set(base_entries) | set(final_entries)
        if base_entries.get(name) != final_entries.get(name)
    }

    parser = etree.XMLParser(remove_blank_text=False, resolve_entities=False)
    base_root = etree.fromstring(base_entries["word/document.xml"], parser)
    final_root = etree.fromstring(final_entries["word/document.xml"], parser)
    final_body = final_root.xpath("/w:document/w:body/w:p", namespaces=NS)
    all_paragraphs = final_root.xpath("//w:p", namespaces=NS)
    texts = [visible_text(paragraph) for paragraph in final_body]

    captions = []
    for paragraph in all_paragraphs:
        text = visible_text(paragraph)
        match = re.match(r"^Fig\.\s+(\d+)\.", text)
        if match:
            captions.append(int(match.group(1)))

    rel_root = etree.fromstring(final_entries["word/_rels/document.xml.rels"], parser)
    rels = {
        rel.get("Id"): rel.get("Target")
        for rel in rel_root.xpath("/pr:Relationships/pr:Relationship", namespaces=REL_NS)
    }

    figure_checks: list[dict[str, object]] = []
    for figure_no, media_name, source_name, description in FIGURES:
        docpr_matches = final_root.xpath(f"//wp:docPr[@name='Figure {figure_no}']", namespaces=NS)
        assert len(docpr_matches) == 1
        docpr = docpr_matches[0]
        inline = docpr.getparent()
        rid = inline.xpath(".//a:blip/@r:embed", namespaces=NS)[0]
        target = rels[rid]
        cnvpr = inline.xpath(".//pic:cNvPr", namespaces=NS)[0]
        source_bytes = (figures_dir / source_name).read_bytes()
        member = f"word/media/{media_name}"
        figure_checks.append(
            {
                "figure": figure_no,
                "rid": rid,
                "relationship_target": target,
                "expected_media": media_name,
                "media_matches_source": final_entries[member] == source_bytes,
                "wp_alt_text_matches": docpr.get("descr") == description,
                "pic_alt_text_matches": cnvpr.get("descr") == description,
            }
        )

    replacement_checks: list[dict[str, object]] = []
    for old, new in REPLACEMENTS:
        matches = [paragraph for paragraph in final_body if visible_text(paragraph) == new]
        replacement_checks.append(
            {
                "old_absent": old not in texts,
                "new_count": len(matches),
                "new_is_blue": len(matches) == 1
                and matches[0].xpath("./w:r/w:rPr/w:color/@w:val", namespaces=NS) == [BLUE],
            }
        )

    base_math = base_root.xpath("//m:oMath", namespaces=NS)
    final_math = final_root.xpath("//m:oMath", namespaces=NS)
    base_inline = base_root.xpath("//wp:inline", namespaces=NS)
    final_inline = final_root.xpath("//wp:inline", namespaces=NS)

    invariants = {
        "zip_entries": len(final_entries),
        "zip_entry_set_unchanged": set(base_entries) == set(final_entries),
        "changed_members_exact": changed == expected_changed,
        "changed_members": sorted(changed),
        "drawings": len(final_root.xpath("//w:drawing", namespaces=NS)),
        "blips": len(final_root.xpath("//a:blip[@r:embed]", namespaces=NS)),
        "unique_blip_rids": len(set(final_root.xpath("//a:blip/@r:embed", namespaces=NS))),
        "media_files": len([name for name in final_entries if name.startswith("word/media/")]),
        "captions": captions,
        "captions_monotonic_1_to_27": captions == list(range(1, 28)),
        "math_objects": len(final_math),
        "math_canonical_unchanged": canonical_hashes(base_math) == canonical_hashes(final_math),
        "tables": len(final_root.xpath("//w:tbl", namespaces=NS)),
        "direct_body_paragraphs": len(final_body),
        "tracked_insertions": len(final_root.xpath("//w:ins", namespaces=NS)),
        "tracked_deletions": len(final_root.xpath("//w:del", namespaces=NS)),
        "hyperlinks": len(final_root.xpath("//w:hyperlink", namespaces=NS)),
        "fields": len(final_root.xpath("//w:instrText", namespaces=NS)),
        "inline_drawings": len(final_inline),
        "drawing_extents_and_rids_unchanged": [
            (
                node.xpath("./wp:extent/@cx", namespaces=NS),
                node.xpath("./wp:extent/@cy", namespaces=NS),
                node.xpath(".//a:blip/@r:embed", namespaces=NS),
            )
            for node in base_inline
        ]
        == [
            (
                node.xpath("./wp:extent/@cx", namespaces=NS),
                node.xpath("./wp:extent/@cy", namespaces=NS),
                node.xpath(".//a:blip/@r:embed", namespaces=NS),
            )
            for node in final_inline
        ],
    }

    required_values = {
        "zip_entries": 54,
        "drawings": 36,
        "blips": 36,
        "unique_blip_rids": 36,
        "media_files": 36,
        "math_objects": 11,
        "tables": 7,
        "direct_body_paragraphs": 375,
        "tracked_insertions": 0,
        "tracked_deletions": 0,
        "hyperlinks": 52,
        "fields": 1,
        "inline_drawings": 36,
    }
    scalar_ok = all(invariants[key] == value for key, value in required_values.items())
    boolean_ok = all(
        invariants[key]
        for key in [
            "zip_entry_set_unchanged",
            "changed_members_exact",
            "captions_monotonic_1_to_27",
            "math_canonical_unchanged",
            "drawing_extents_and_rids_unchanged",
        ]
    )
    figures_ok = all(
        row["media_matches_source"] and row["wp_alt_text_matches"] and row["pic_alt_text_matches"]
        and row["relationship_target"] == f"media/{row['expected_media']}"
        for row in figure_checks
    )
    replacements_ok = all(
        row["old_absent"] and row["new_count"] == 1 and row["new_is_blue"]
        for row in replacement_checks
    )
    status = "pass" if scalar_ok and boolean_ok and figures_ok and replacements_ok else "fail"
    return {
        "status": status,
        "base": str(base),
        "base_sha256": sha256_bytes(base.read_bytes()),
        "final": str(final),
        "final_sha256": sha256_bytes(final.read_bytes()),
        "invariants": invariants,
        "required_values": required_values,
        "figure_checks": figure_checks,
        "replacement_checks": replacement_checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--final", type=Path, required=True)
    parser.add_argument("--figures-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = analyse_docx(args.base.resolve(), args.final.resolve(), args.figures_dir.resolve())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "report": str(args.report.resolve())}, ensure_ascii=False))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
