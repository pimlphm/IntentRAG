"""Validate observable information-distribution diagnostic artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "information_v1" / "information_v1.json"
RESULTS = ROOT / "04_results" / "information_v1"
RAW = RESULTS / "raw"
FIGURES = ROOT / "05_figures" / "information_v1"
TABLES = ROOT / "06_tables" / "information_v1"
REPORTS = ROOT / "07_reports" / "information_v1"
LOGS = ROOT / "08_logs" / "information_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_mass_sums(frame: pd.DataFrame, groups: list[str], column: str, label: str) -> None:
    sums = frame.groupby(groups, dropna=False)[column].sum()
    if not np.allclose(sums.to_numpy(dtype=float), 1.0, atol=1e-9, rtol=0):
        bad = sums[~np.isclose(sums, 1.0, atol=1e-9, rtol=0)].head().to_dict()
        raise AssertionError(f"{label} does not sum to one: {bad}")


def assert_between(frame: pd.DataFrame, columns: list[str], low: float, high: float) -> None:
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce").dropna().to_numpy(dtype=float)
        if ((values < low - 1e-10) | (values > high + 1e-10)).any():
            raise AssertionError(f"{column} outside [{low}, {high}]")


def validate(config_path: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    required = [
        RAW / "repository_prior.csv",
        RAW / "query_demand_distribution.csv",
        RAW / "locked_reference_distribution.csv",
        RAW / "retrieved_package_distribution.csv",
        RAW / "benchmark_demand_mix.csv",
        RAW / "source_file_manifest.csv",
        RAW / "aesr_v2_immutability_audit.csv",
        RESULTS / "distribution_diagnostics.csv",
        RESULTS / "selection_bias.csv",
        RESULTS / "loo_contributions.csv",
        RESULTS / "method_axis_summary.csv",
        RESULTS / "metric_outcome_associations.csv",
        RESULTS / "claim_result_map.csv",
        REPORTS / "observable_information_distribution_diagnostic.md",
        REPORTS / "run_manifest.json",
        LOGS / "run.log",
    ]
    required.extend(
        [
            FIGURES / "figure_a_observable_distribution_composite.png",
            FIGURES / "figure_b_role_selection_bias_heatmap.png",
            FIGURES / "figure_c_loo_alignment_contributions.png",
            FIGURES / "figure_d_source_distribution_composite.png",
            TABLES / "table_a_estimator_definitions.csv",
            TABLES / "table_b_method_role_diagnostics.csv",
            TABLES / "table_c_primary_role_distributions.csv",
            TABLES / "table_d_exploratory_outcome_associations.csv",
            TABLES / "table_e_retained_negative_results.csv",
        ]
    )
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    if missing:
        raise AssertionError(f"Missing artifacts: {missing}")

    repository = pd.read_csv(RAW / "repository_prior.csv").fillna("")
    query_demand = pd.read_csv(RAW / "query_demand_distribution.csv").fillna("")
    reference = pd.read_csv(RAW / "locked_reference_distribution.csv").fillna("")
    retrieved = pd.read_csv(RAW / "retrieved_package_distribution.csv").fillna("")
    benchmark_mix = pd.read_csv(RAW / "benchmark_demand_mix.csv").fillna("")
    diagnostics = pd.read_csv(RESULTS / "distribution_diagnostics.csv").fillna("")
    selection = pd.read_csv(RESULTS / "selection_bias.csv").fillna("")
    loo = pd.read_csv(RESULTS / "loo_contributions.csv").fillna("")
    summary = pd.read_csv(RESULTS / "method_axis_summary.csv").fillna("")
    claims = pd.read_csv(RESULTS / "claim_result_map.csv").fillna("")
    manifest = json.loads((REPORTS / "run_manifest.json").read_text(encoding="utf-8"))

    n_queries = diagnostics["question_id"].nunique()
    n_conditions = diagnostics["condition"].nunique()
    n_axes = diagnostics["axis"].nunique()
    if n_queries != 214:
        raise AssertionError(f"Expected 214 queries, found {n_queries}")
    if n_conditions != len(config["access_conditions"]):
        raise AssertionError(f"Expected {len(config['access_conditions'])} methods, found {n_conditions}")
    if n_axes != len(config["distribution_axes"]):
        raise AssertionError(f"Expected {len(config['distribution_axes'])} axes, found {n_axes}")
    if len(diagnostics) != n_queries * n_conditions * n_axes:
        raise AssertionError("Diagnostic row count is not query x condition x axis")
    if diagnostics.duplicated(["question_id", "condition", "axis"]).any():
        raise AssertionError("Duplicate query-condition-axis diagnostic rows")

    assert_mass_sums(repository, ["aircraft_model", "axis"], "repository_mass", "repository prior")
    assert_mass_sums(query_demand, ["question_id", "axis"], "query_demand_mass", "query demand")
    assert_mass_sums(reference, ["question_id", "axis"], "reference_mass", "reference")
    assert_mass_sums(
        retrieved,
        ["question_id", "condition", "axis"],
        "package_mass",
        "retrieved package",
    )
    assert_mass_sums(
        benchmark_mix,
        ["population", "facet"],
        "mass",
        "benchmark mix",
    )

    assert_between(
        diagnostics,
        [
            "hnorm_repository",
            "hnorm_query_demand",
            "hnorm_package",
            "hnorm_reference",
            "jsd_query_demand_repository",
            "jsd_package_repository",
            "jsd_package_query_demand",
            "jsd_package_reference",
            "jsd_query_demand_reference",
            "jsd_visible_query_locked_sensitivity",
            "demand_mass_covered",
            "reference_mass_covered",
            "package_top_category_mass",
            "loo_alignment_positive_rate",
            "loo_alignment_negative_rate",
            "loo_alignment_zero_rate",
        ],
        0.0,
        1.0,
    )
    assert_between(
        diagnostics,
        ["observable_alignment_shift", "normalized_entropy_shift_query_to_package"],
        -1.0,
        1.0,
    )

    selection_key = ["question_id", "condition", "axis", "category"]
    if selection.duplicated(selection_key).any():
        raise AssertionError("Duplicate selection category rows")
    selection_sums = selection.groupby(["question_id", "condition", "axis"])[
        ["repository_mass", "query_demand_mass", "package_mass", "reference_mass"]
    ].sum()
    if not np.allclose(selection_sums.to_numpy(dtype=float), 1.0, atol=1e-9, rtol=0):
        raise AssertionError("Selection-bias component masses do not sum to one")

    defined = loo[loo["loo_defined"].astype(str).str.lower().isin({"true", "1"})].copy()
    full = pd.to_numeric(defined["full_alignment_utility"], errors="raise")
    without = pd.to_numeric(defined["without_item_alignment_utility"], errors="raise")
    contribution = pd.to_numeric(defined["loo_alignment_contribution"], errors="raise")
    if not np.allclose(contribution, full - without, atol=1e-12, rtol=0):
        raise AssertionError("LOO alignment contribution identity failed")
    if defined["retrieval_rerun"].astype(str).str.lower().isin({"true", "1"}).any():
        raise AssertionError("LOO unexpectedly reran retrieval")
    if defined["ranks_reassigned"].astype(str).str.lower().isin({"true", "1"}).any():
        raise AssertionError("LOO unexpectedly reassigned ranks")
    if diagnostics["gold_used_for_retrieval_feature"].astype(str).str.lower().isin({"true", "1"}).any():
        raise AssertionError("Gold entered retrieval")
    if diagnostics["locked_track_used_for_routing"].astype(str).str.lower().isin({"true", "1"}).any():
        raise AssertionError("Locked track entered routing")
    if diagnostics["mutual_information_estimated"].astype(str).str.lower().isin({"true", "1"}).any():
        raise AssertionError("Run incorrectly marks mutual information as estimated")
    if diagnostics["causal_effect_estimated"].astype(str).str.lower().isin({"true", "1"}).any():
        raise AssertionError("Run incorrectly marks a causal effect as estimated")

    role_query = query_demand[query_demand["axis"].eq("evidence_role")]
    if role_query["gold_used_in_primary_query_demand"].astype(str).str.lower().isin({"true", "1"}).any():
        raise AssertionError("Gold entered primary visible role demand")
    if reference["used_for_retrieval"].astype(str).str.lower().isin({"true", "1"}).any():
        raise AssertionError("Locked reference entered retrieval")
    if not reference["construction_coupled"].astype(str).str.lower().isin({"true", "1"}).all():
        raise AssertionError("Reference construction coupling is not fully disclosed")

    source_boeing = diagnostics[
        diagnostics["axis"].eq("source_document") & diagnostics["aircraft_model"].eq("Boeing 737")
    ]
    for column in [
        "hnorm_repository",
        "hnorm_query_demand",
        "hnorm_package",
        "jsd_query_demand_repository",
        "jsd_package_repository",
        "jsd_package_query_demand",
    ]:
        values = pd.to_numeric(source_boeing[column], errors="raise")
        if not np.allclose(values, 0.0, atol=1e-12, rtol=0):
            raise AssertionError(f"Degenerate Boeing source-axis {column} is not zero")

    immutability = pd.read_csv(RAW / "aesr_v2_immutability_audit.csv").fillna("")
    if not immutability["unchanged"].astype(str).str.lower().isin({"true", "1"}).all():
        raise AssertionError("AESR-v2 immutability audit failed")
    source_manifest = pd.read_csv(RAW / "source_file_manifest.csv").fillna("")
    for _, row in source_manifest.iterrows():
        path = ROOT / str(row["path"])
        if not path.exists() or sha256(path) != str(row["sha256"]):
            raise AssertionError(f"Source hash mismatch: {row['path']}")

    claim_i6 = claims[claims["claim_id"].eq("I6")]
    if len(claim_i6) != 1 or "Unsupported" not in str(claim_i6.iloc[0]["support_status"]):
        raise AssertionError("Mutual-information boundary claim is missing")
    claim_i7 = claims[claims["claim_id"].eq("I7")]
    if len(claim_i7) != 1 or "outside" not in str(claim_i7.iloc[0]["support_status"]).lower():
        raise AssertionError("Downstream-LLM boundary claim is missing")

    report_text = (REPORTS / "observable_information_distribution_diagnostic.md").read_text(encoding="utf-8")
    for phrase in [
        "normalized masses",
        "visible-question bilingual-lexicon demand",
        "negative observations are retained",
        "Retrieval is not rerun",
        "The generator is absent",
    ]:
        if phrase not in report_text:
            raise AssertionError(f"Report boundary phrase missing: {phrase}")

    for figure_path in FIGURES.glob("*.png"):
        image = mpimg.imread(figure_path)
        if image.shape[0] < 600 or image.shape[1] < 900:
            raise AssertionError(f"Figure resolution too small: {figure_path.name} {image.shape}")
        if float(np.nanstd(image)) < 0.02:
            raise AssertionError(f"Figure appears blank: {figure_path.name}")

    if manifest["experiment_id"] != config["experiment_id"]:
        raise AssertionError("Manifest experiment ID mismatch")
    if not manifest.get("aesr_v2_result_files_verified_unchanged", False):
        raise AssertionError("Manifest does not affirm AESR-v2 immutability")
    if manifest["n_distribution_diagnostic_rows"] != len(diagnostics):
        raise AssertionError("Manifest diagnostic row count mismatch")
    if manifest["n_loo_rows"] != len(loo):
        raise AssertionError("Manifest LOO row count mismatch")

    validation = {
        "status": "ok",
        "experiment_id": config["experiment_id"],
        "n_queries": n_queries,
        "n_conditions": n_conditions,
        "n_axes": n_axes,
        "n_diagnostics": len(diagnostics),
        "n_selection_rows": len(selection),
        "n_loo_rows": len(loo),
        "n_defined_loo_rows": len(defined),
        "n_summary_rows": len(summary),
        "n_figures": len(list(FIGURES.glob("*.png"))),
        "aesr_v2_unchanged": True,
        "distribution_sums_valid": True,
        "metric_bounds_valid": True,
        "loo_identity_valid": True,
        "claim_boundaries_valid": True,
    }
    (REPORTS / "validation_report.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (LOGS / "validation.log").write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    return validation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    print(json.dumps(validate(args.config.resolve()), ensure_ascii=False))


if __name__ == "__main__":
    main()
