from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

REQUIRED = [
    "_pipeline_v21/p0_build_eval_v21.py",
    "00_raw_documents/source_manifest.csv",
    "04_results/qa_warehouse_full.csv",
    "04_results/all_cases_evidence_objects.csv",
    "04_results/v5/v5_evidence_equivalence_sets.csv",
    "01_processed_corpus/v5/evidence_objects.parquet",
    "04_results/v21/v21_metrics_by_case_track.csv",
    "04_results/v21/v21_target_attainment_matrix.csv",
    "09_notebook/FD_GRAG_RC_V21_Runnable_Methodology_Notebook.ipynb",
]


def main() -> int:
    missing = [p for p in REQUIRED if not (ROOT / p).exists()]
    if missing:
        print("Missing required package files:")
        for p in missing:
            print(" -", p)
        return 1

    metrics = pd.read_csv(ROOT / "04_results/v21/v21_metrics_by_case_track.csv")
    final = metrics[metrics["method"].eq("FD-GRAG-RC-V21-final")]
    if final.empty:
        print("No FD-GRAG-RC-V21-final rows found.")
        return 2
    if len(final) < 4:
        print("Expected four case-track rows for FD-GRAG-RC-V21-final.")
        return 3

    print("Package smoke check passed.")
    print(final[[
        "aircraft_model",
        "track",
        "exact_recall_10",
        "accepted_recall_10",
        "evidence_package_recall_10",
        "grounded_score",
        "hallucination_risk_proxy",
        "n",
    ]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
