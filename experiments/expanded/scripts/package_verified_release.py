from __future__ import annotations

import csv
import hashlib
import json
import os
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "deliverables"
ZIP_PATH = OUT_DIR / "IntentRAG_扩展实验数据_独立英文图与复现代码_20261015.zip"
MANIFEST_CSV = OUT_DIR / "verified_release_manifest.csv"
INDEX_TXT = OUT_DIR / "DELIVERY_INDEX_20261015.txt"

INCLUDE_TOP = {
    "data",
    "figures",
    "figures_qa_kv",
    "figures_token_budget",
    "models",
    "reports",
    "results",
    "scripts",
}

EXCLUDE_DIR_NAMES = {
    "__pycache__",
    ".git",
    ".venv",
    "node_modules",
    "figures_core28",
    "memory_replay_core28",
    "word_render_qa",
    "collision",
    "overlays",
}

EXCLUDE_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".tmp",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def is_included(path: Path) -> bool:
    rel = path.relative_to(ROOT)
    if not rel.parts or rel.parts[0] not in INCLUDE_TOP:
        return False
    if any(part in EXCLUDE_DIR_NAMES for part in rel.parts[:-1]):
        return False
    if path.suffix.lower() in EXCLUDE_SUFFIXES:
        return False
    name = path.name.lower()
    if name.endswith(".overlay.pdf") or name.endswith(".183mm_preview.png"):
        return False
    if name.endswith(".collision-audit.pdf") or name.endswith(".alignment.svg"):
        return False
    if path.resolve() in {ZIP_PATH.resolve(), MANIFEST_CSV.resolve(), INDEX_TXT.resolve()}:
        return False
    return True


def collect_files() -> list[Path]:
    files: list[Path] = []
    for top in sorted(INCLUDE_TOP):
        base = ROOT / top
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path.is_file() and is_included(path):
                files.append(path)
    return sorted(files, key=lambda p: p.relative_to(ROOT).as_posix().lower())


def write_index(file_count: int, total_bytes: int) -> None:
    text = f"""IntentRAG expanded evaluation verified release
Date: 2026-10-15 (Asia/Shanghai)

Scope
- 33 retrieval methods x 690 records = 22,770 complete retrieval rows.
- 64-question local Qwen 3B generation subset = 2,112 query-method cells; 1,933 API attempts plus 179 exact-payload aliases; one transport failure retained.
- Exact-source LRU replay and equal source-token-budget replay for all 33 methods.
- 594 real GPU prefix-layout profiles from six visible-query pairs, 33 methods, and three layouts.
- Independent English figures without in-chart titles, with PNG/PDF/SVG, source data, plotting code, and QA reports.

Interpretation boundaries
- Fresh questions are source-derived and not expert-certified natural maintenance questions.
- Retrieval Hit@10, required-quote support, answer JSON parsing, quote emission/support, source reuse, and GPU prefix reuse are different endpoints.
- No valid semantic blind-judge scores are available: the attempted 7B audit was blocked by Windows socket error 10013, with zero model responses and zero valid judgments.
- Project-labelled GraphRAG, Agentic, and GaRAGe-style methods are controlled local implementations, not official reproductions.
- Fixed module count is not equal storage cost; use the source-token-budget replay for the stricter storage comparison.
- Real GPU cache counts include shared system/scope text and do not establish evidence admissibility, answer correctness, or isolated throughput speedup.
- The unrendered working DOCX is deliberately excluded from this verified release.

Key empirical results
- Fresh accepted evidence Hit@10: True BM25 0.972; balanced IntentRAG 0.949; multi-granularity plus mMARCO CE@50 0.930; matched dense plus CE@50 0.827.
- Fresh complete required-quote support: True BM25 0.916; balanced IntentRAG 0.902; multi-granularity plus CE@50 0.869; matched dense plus CE@50 0.780.
- Long-term fixed Hit@10 0.248 and Profile-resolved Hit@10 0.019 show that compressed persistent views cannot replace query-local evidence in this implementation.
- Mean real input-KV reuse: query-first 0.039; canonical state-first 0.125; hierarchical state-first 0.052. Canonicalization also reduced the materialized reference-anchor proxy from 0.721 to 0.593, so cache gain and evidence visibility must be evaluated jointly.

Package contents
- data: frozen queries, method registry, and GPU pair manifests.
- results: full retrieval, generation, replay, token-budget, interface-control, protected-prefix, and GPU records.
- figures, figures_token_budget, figures_qa_kv: authoritative independent figures and source data.
- scripts: experiment, analysis, plotting, audit, and packaging code.
- reports: protocols, audits, limitations, and execution manifests.
- models: small local tokenizer/calibration artefacts needed by the included scripts; large external model weights and original source manuals are not duplicated.

Inventory
- Included files: {file_count:,}
- Uncompressed bytes: {total_bytes:,}
- Per-file SHA-256 values: verified_release_manifest.csv
"""
    INDEX_TXT.write_text(text, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = collect_files()
    rows = []
    total_bytes = 0
    for path in files:
        stat = path.stat()
        rel = path.relative_to(ROOT).as_posix()
        total_bytes += stat.st_size
        rows.append(
            {
                "relative_path": rel,
                "bytes": stat.st_size,
                "mtime_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                "sha256": sha256(path),
            }
        )

    with MANIFEST_CSV.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    write_index(len(files), total_bytes)

    packaged = files + [MANIFEST_CSV, INDEX_TXT]
    with zipfile.ZipFile(
        ZIP_PATH,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
        allowZip64=True,
    ) as zf:
        for path in packaged:
            zf.write(path, arcname=path.relative_to(ROOT).as_posix())

    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        bad = zf.testzip()
        names = zf.namelist()
    if bad is not None:
        raise RuntimeError(f"ZIP CRC failure: {bad}")
    if len(names) != len(packaged):
        raise RuntimeError(f"ZIP entry mismatch: {len(names)} != {len(packaged)}")

    report = {
        "status": "pass",
        "zip": str(ZIP_PATH),
        "zip_bytes": ZIP_PATH.stat().st_size,
        "zip_sha256": sha256(ZIP_PATH),
        "included_files_excluding_generated_index_and_manifest": len(files),
        "zip_entries": len(names),
        "uncompressed_source_bytes": total_bytes,
        "crc_test": "pass",
        "unverified_docx_excluded": True,
        "large_external_model_weights_included": False,
        "original_source_manuals_included": False,
    }
    (OUT_DIR / "verified_release_validation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
