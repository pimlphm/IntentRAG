from __future__ import annotations

import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def close(actual: float, expected: float, tolerance: float = 5e-4) -> None:
    if not math.isclose(actual, expected, abs_tol=tolerance):
        raise AssertionError(f"expected {expected}, observed {actual}")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    required = [
        ROOT / "README.md",
        ROOT / "PUBLICATION_AUDIT.json",
        ROOT / "PUBLICATION_MANIFEST.csv",
        ROOT / "paper" / "IntentRAG_working_manuscript_20261015.docx",
        ROOT / "figures" / "figures_qa_kv" / "actual_kv_vs_materialized_reference_support.png",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    if missing:
        raise AssertionError(f"missing required files: {missing}")

    publication_audit = json.loads((ROOT / "PUBLICATION_AUDIT.json").read_text(encoding="utf-8"))
    assert publication_audit["status"] == "pass"
    assert publication_audit["secret_findings"] == []
    assert publication_audit["absolute_path_hits"] == []
    assert publication_audit["max_file_under_100_mib"] is True

    registry = read_csv(ROOT / "data" / "metadata" / "method_registry.csv")
    # The registry records 28 named method families; the frozen matrix contains
    # 33 executed variants after parameterized neural/multi-view configurations.
    assert len(registry) == 28, len(registry)

    retrieval = read_csv(ROOT / "results" / "aggregates" / "all_retrieval_summary.csv")
    fresh = [row for row in retrieval if row["split"] == "fresh_locked_test"]
    assert len(fresh) == 33, len(fresh)
    assert {int(row["n_records"]) for row in fresh} == {214}
    by_method = {row["method"]: row for row in fresh}
    close(float(by_method["True BM25"]["accepted_hit_at_10"]), 0.971963)
    close(float(by_method["IntentRAG balanced fair"]["accepted_hit_at_10"]), 0.948598)
    close(float(by_method["IntentRAG multi + mMARCO CE@50"]["accepted_hit_at_10"]), 0.929907)
    close(float(by_method["Neural dense + mMARCO CE@50"]["accepted_hit_at_10"]), 0.827103)
    close(float(by_method["True BM25"]["all_quote_support"]), 0.915888)
    close(float(by_method["IntentRAG balanced fair"]["all_quote_support"]), 0.901869)

    kv = read_csv(ROOT / "results" / "aggregates" / "real_kv" / "layout_summary.csv")
    assert sum(int(row["n_profiles"]) for row in kv) == 594
    kv_by_layout = {row["namespace"]: row for row in kv}
    close(float(kv_by_layout["query_first"]["mean_cache_fraction"]), 0.038664)
    close(float(kv_by_layout["canonical_state_first"]["mean_cache_fraction"]), 0.125378)
    close(float(kv_by_layout["hierarchical_state_first"]["mean_cache_fraction"]), 0.052250)

    semantic = json.loads(
        (
            ROOT
            / "results"
            / "aggregates"
            / "qa_expanded_main"
            / "blind_semantic_audit_failed_network_20261014"
            / "audit_manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert semantic["status"].startswith("blocked")
    assert semantic["actual_model_responses"] == 0
    assert semantic["scores_must_not_be_used"] is True

    print(
        json.dumps(
            {
                "status": "pass",
                "method_registry_entries": len(registry),
                "executed_method_variants": len(fresh),
                "fresh_questions": 214,
                "retrieval_cells": 22770,
                "actual_kv_profiles": 594,
                "semantic_judgments": 0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

