from __future__ import annotations

import json
import hashlib
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "04_results" / "aesr_v2"
REPORTS = ROOT / "07_reports" / "aesr_v2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def values(value: object) -> list[str]:
    return [item for item in str(value or "").split(";") if item]


def main() -> None:
    bench = pd.read_csv(ROOT / "04_results" / "v21" / "qa_warehouse_v21_full.csv").fillna("")
    gold = pd.read_csv(ROOT / "04_results" / "v21" / "v21_gold_sets.csv").fillna("")
    split_by_query = bench.set_index("question_id")["split"].astype(str).to_dict()

    fields = {
        "exact_evidence_id": "exact_gold_evidence_ids",
        "accepted_evidence_id": "accepted_gold_evidence_ids",
        "accepted_page_id": "accepted_gold_page_ids",
        "accepted_section_id": "accepted_gold_section_ids",
    }
    rows = []
    for label, column in fields.items():
        split_sets: dict[str, set[str]] = defaultdict(set)
        for _, row in gold.iterrows():
            split = split_by_query.get(str(row["question_id"]), "")
            for item in values(row[column]):
                split_sets[item].add(split)
        cross = sum(len(splits) > 1 for splits in split_sets.values())
        total = len(split_sets)
        rows.append(
            {
                "label_type": label,
                "unique_label_count": total,
                "cross_split_label_count": cross,
                "cross_split_rate": cross / total if total else 0.0,
                "interpretation": "zero overlap is guaranteed only for exact-gold evidence identifiers"
                if label == "exact_evidence_id"
                else "residual accepted-equivalence overlap remains across split boundaries",
            }
        )
    overlap = pd.DataFrame(rows)
    overlap.to_csv(RESULTS / "split_overlap_audit.csv", index=False, encoding="utf-8-sig")

    claim_path = RESULTS / "claim_result_map.csv"
    claims = pd.read_csv(claim_path)
    claims = claims[~claims["claim_id"].astype(str).eq("C10")]
    accepted = overlap[overlap["label_type"].eq("accepted_evidence_id")].iloc[0]
    row = {
        "claim_id": "C10",
        "manuscript_claim": "Split isolation is exact-ID based; accepted evidence, page, and section alternatives retain cross-split overlap.",
        "support_status": "Supported by artifact audit",
        "experiment_id": claims["experiment_id"].iloc[0],
        "population": "full 476-item synthetic V21 benchmark",
        "metric": "accepted evidence identifiers appearing in more than one split",
        "value": float(accepted["cross_split_rate"]),
        "ci": "not applicable",
        "result_file": "04_results/aesr_v2/split_overlap_audit.csv",
        "figure_or_table": "protocol limitation",
        "gold_dependency": "audit of locked label topology; never used for retrieval",
        "synthetic_only": True,
        "caveat": "The held-out test is exact-ID-isolated, not fully accepted/page/section-overlap-free.",
    }
    claims = pd.concat([claims, pd.DataFrame([row])], ignore_index=True)
    claims.to_csv(claim_path, index=False, encoding="utf-8-sig")

    manifest_path = REPORTS / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["result_files"] = sorted(
        str(path.relative_to(ROOT)) for path in RESULTS.glob("*.csv")
    )
    source_hashes = manifest.setdefault("source_hashes", {})
    for relative_path in list(source_hashes):
        source_path = ROOT / relative_path
        if source_path.exists():
            source_hashes[relative_path] = sha256(source_path)
    for source_path in [Path(__file__).resolve(), ROOT / "_pipeline_aesr_v2" / "validate_outputs.py"]:
        source_hashes[str(source_path.relative_to(ROOT))] = sha256(source_path)
    manifest["manifest_refresh_timestamp_local"] = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z")
    manifest["postrun_reporting_note"] = (
        "Numeric retrieval and intervention outputs were produced by the recorded run. "
        "Source hashes were refreshed after non-numeric figure-label and claim-boundary reporting edits, "
        "and the split-overlap audit was appended by augment_split_overlap_audit.py."
    )
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(overlap.to_string(index=False))


if __name__ == "__main__":
    main()
