from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CASE_IDS = ["Boeing_737", "Cessna_172"]

LABEL_COLUMNS = {
    "label",
    "qwen_label",
    "method_short",
    "method_display",
    "source_method_display",
    "runner_up_label",
}

LABEL_MAP = {
    "BM25 context": "Qwen + BM25",
    "BM25 ctx": "Qwen + BM25",
    "Hybrid context": "Qwen + Hybrid",
    "Hybrid ctx": "Qwen + Hybrid",
    "GraphRAG context": "Qwen + GraphRAG",
    "GraphRAG ctx": "Qwen + GraphRAG",
    "FD-GRAG-RC old": "Qwen + FD-GRAG-RC",
    "FD-GRAG-RC free-form": "Qwen + FD-GRAG-RC",
    "Qwen + FD free-form": "Qwen + FD-GRAG-RC",
    "Qwen + FD-GRAG-RC citation repaired": "Qwen + FD-GRAG-RC (repaired)",
    "Qwen + FD-GRAG-RC repaired": "Qwen + FD-GRAG-RC (repaired)",
    "After: citation-repaired Qwen + FD-GRAG-RC": "Qwen + FD-GRAG-RC (repaired)",
    "Before: Qwen + FD-GRAG-RC": "Qwen + FD-GRAG-RC",
    "FD evidence-card assembly": "FD-GRAG-RC evidence-card assembly",
    "FD-GRAG-RC assembled": "FD-GRAG-RC evidence-card assembly",
    "Reference: FD evidence-card assembly": "FD-GRAG-RC evidence-card assembly",
    "GPT-OSS + FD contract": "GPT-OSS + FD-GRAG-RC",
    "Codex + FD contract": "Codex + FD-GRAG-RC",
    "FD-GRAG-RC contract": "Qwen + FD-GRAG-RC (contract)",
    "Qwen direct / no RAG": "Qwen direct",
    "TF-IDF": "Char TF-IDF",
    "SVD": "SVD surrogate",
    "Parent": "Parent-section",
    "CompP": "Component-profile",
    "Causal": "Causal-graph",
    "CompG": "Component-graph",
    "Agentic": "Agentic RAG",
    "E-flow": "Evidence-flow",
    "Rel": "GaRAGe-rel",
    "Attr": "GaRAGe-attr",
    "Deflect": "GaRAGe-deflect",
}


def latest_dir(case_id: str) -> Path:
    return next(p for p in (ROOT / "Cases" / case_id).iterdir() if p.is_dir() and p.name.startswith("8"))


def normalize_value(value: object) -> object:
    if pd.isna(value):
        return value
    text = str(value)
    return LABEL_MAP.get(text, text)


def normalize_csv(path: Path) -> int:
    df = pd.read_csv(path)
    changed = 0
    for col in LABEL_COLUMNS.intersection(df.columns):
        before = df[col].astype(str).tolist()
        df[col] = df[col].map(normalize_value)
        after = df[col].astype(str).tolist()
        changed += sum(1 for old, new in zip(before, after) if old != new)
    if changed:
        df.to_csv(path, index=False, encoding="utf-8-sig")
    return changed


def main() -> None:
    rows = []
    for case_id in CASE_IDS:
        latest = latest_dir(case_id)
        for csv_path in sorted(latest.glob("*.csv")):
            changed = normalize_csv(csv_path)
            if changed:
                rows.append({"case_id": case_id, "csv": str(csv_path), "changed_label_cells": changed})
    report = pd.DataFrame(rows)
    report_path = ROOT / "reports" / "latest_figure_method_label_normalization.csv"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(report_path, index=False, encoding="utf-8-sig")
    print(report.to_string(index=False) if not report.empty else "No label changes needed.")


if __name__ == "__main__":
    main()
