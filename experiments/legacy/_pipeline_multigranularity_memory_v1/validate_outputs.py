"""
Validate multigranularity_memory_v1 outputs and paper figures.

The validator is intentionally strict about provenance and leakage boundaries:
it checks real pipeline CSV schemas, verifies that locked/gold fields are not
marked as used during ranking/router stages, and confirms figure manifests and
single-figure image/PDF outputs. It does not modify the manuscript.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "_pipeline_multigranularity_memory_v1"
RESULTS = ROOT / "04_results" / "multigranularity_memory_v1"
TABLES = ROOT / "06_tables" / "multigranularity_memory_v1"
FIGURES = ROOT / "05_figures" / "multigranularity_memory_v1"
REPORTS = ROOT / "07_reports" / "multigranularity_memory_v1"


REQUIRED_CSV_COLUMNS = {
    RESULTS / "query_results.csv": {
        "experiment_id",
        "question_id",
        "evaluation_split",
        "aircraft_model",
        "benchmark_track_posthoc",
        "benchmark_query_type_posthoc",
        "method",
        "query_text",
        "gold_cue_removed",
        "is_synthetic_query",
        "gold_used_for_ranking",
        "accepted_used_for_ranking",
        "locked_track_used_for_ranking",
        "query_type_used_for_ranking",
        "required_slots_used_for_ranking",
        "top_evidence_ids",
        "short_term_weight",
        "long_term_weight",
        "offline_exact_hit_10",
        "offline_accepted_hit_10",
        "offline_page_hit_10",
        "offline_section_hit_10",
        "package_action_present",
        "package_authority_present",
        "package_condition_present",
        "package_warning_present",
        "package_provenance_valid_rate",
        "package_role_diversity",
        "package_page_diversity",
        "package_section_diversity",
        "package_approx_tokens",
    },
    RESULTS / "router_weights.csv": {
        "experiment_id",
        "question_id",
        "evaluation_split",
        "aircraft_model",
        "query_text",
        "detail_entropy",
        "page_entropy",
        "section_entropy",
        "profile_entropy",
        "detail_weight",
        "page_weight",
        "section_weight",
        "profile_weight",
        "short_term_weight",
        "long_term_weight",
        "gold_used_for_router",
        "locked_track_used_for_router",
        "query_type_used_for_router",
    },
    RESULTS / "summary.csv": {
        "evaluation_split",
        "aircraft_model",
        "method",
        "offline_accepted_hit_10_mean",
        "offline_page_hit_10_mean",
        "offline_section_hit_10_mean",
        "package_approx_tokens_mean",
        "short_term_weight_mean",
        "long_term_weight_mean",
    },
    RESULTS / "paired_bootstrap.csv": {
        "baseline",
        "proposed",
        "metric",
        "mean_delta",
        "ci_low",
        "ci_high",
        "n",
    },
    RESULTS / "claim_result_map.csv": {
        "claim_id",
        "manuscript_claim",
        "support_status",
        "population",
        "metric",
        "value",
        "result_file",
        "gold_dependency",
        "synthetic_only",
        "caveat",
    },
    RESULTS / "retrieval_visibility_audit.csv": {
        "field",
        "stage",
        "online_visible",
        "locked_or_gold",
        "treatment",
    },
}

EXPECTED_METHODS = {
    "flat_detail_hybrid",
    "fixed_multigranularity",
    "entropy_multigranularity",
    "entropy_admissibility_assembly",
}

REQUIRED_FIGURES = {
    "fig_mgm_conceptual_mechanism",
    "fig_mgm_locked_test_method_comparison",
    "fig_mgm_short_long_weight_allocation",
    "fig_mgm_entropy_vs_admissibility_query_delta",
    "fig_mgm_granularity_ablation",
}


def read_csv(path: Path, required: set[str]) -> tuple[pd.DataFrame | None, list[str]]:
    problems: list[str] = []
    if not path.exists():
        return None, [f"missing: {path.relative_to(ROOT)}"]
    frame = pd.read_csv(path)
    if frame.empty:
        problems.append(f"empty: {path.relative_to(ROOT)}")
    missing_columns = sorted(required - set(frame.columns))
    if missing_columns:
        problems.append(f"{path.relative_to(ROOT)} missing columns: {missing_columns}")
    return frame, problems


def validate_pipeline_outputs() -> tuple[dict[str, pd.DataFrame], list[str], list[str]]:
    frames: dict[str, pd.DataFrame] = {}
    problems: list[str] = []
    warnings: list[str] = []
    for path, columns in REQUIRED_CSV_COLUMNS.items():
        frame, path_problems = read_csv(path, columns)
        problems.extend(path_problems)
        if frame is not None:
            frames[path.name] = frame

    query = frames.get("query_results.csv")
    if query is not None and not query.empty:
        locked = query[query["evaluation_split"].eq("locked_blind_test")]
        if locked.empty:
            problems.append("query_results.csv has no locked_blind_test rows")
        methods = set(query["method"].astype(str))
        missing_methods = sorted(EXPECTED_METHODS - methods)
        if missing_methods:
            problems.append(f"query_results.csv missing expected methods: {missing_methods}")
        for column in [
            "gold_used_for_ranking",
            "accepted_used_for_ranking",
            "locked_track_used_for_ranking",
            "query_type_used_for_ranking",
            "required_slots_used_for_ranking",
        ]:
            if column in query and query[column].astype(bool).any():
                problems.append(f"leakage boundary failed: {column} contains True")
        if "existing_cue_free_intentrag_reference" not in methods:
            warnings.append("existing AESR-v2 reference method not present; figures will compare only MGM methods")

    router = frames.get("router_weights.csv")
    if router is not None and not router.empty:
        for column in ["gold_used_for_router", "locked_track_used_for_router", "query_type_used_for_router"]:
            if column in router and router[column].astype(bool).any():
                problems.append(f"router leakage boundary failed: {column} contains True")
        weight_sum = router[["detail_weight", "page_weight", "section_weight", "profile_weight"]].sum(axis=1)
        max_error = float((weight_sum - 1.0).abs().max())
        if max_error > 1e-6:
            problems.append(f"router view weights do not sum to 1; max error={max_error:g}")

    return frames, problems, warnings


def validate_figures() -> tuple[list[str], list[str]]:
    problems: list[str] = []
    warnings: list[str] = []
    manifest_path = REPORTS / "paper_figure_manifest.json"
    if not manifest_path.exists():
        return [f"missing figure manifest: {manifest_path.relative_to(ROOT)}"], warnings
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("rule_summary", {}).get("single_panel"):
        problems.append("manifest does not assert single_panel=True")
    if not manifest.get("rule_summary", {}).get("no_in_figure_titles"):
        problems.append("manifest does not assert no_in_figure_titles=True")
    figures = manifest.get("figures", [])
    seen = {item.get("figure_id") for item in figures}
    missing = sorted(REQUIRED_FIGURES - seen)
    if missing:
        problems.append(f"figure manifest missing required figures: {missing}")
    for item in figures:
        for output in item.get("outputs", []):
            path = ROOT / output["path"]
            if not path.exists():
                problems.append(f"missing figure output: {output['path']}")
            elif path.stat().st_size < 10_000:
                warnings.append(f"small figure output, inspect manually: {output['path']}")
        if int(item.get("source_rows", 0)) <= 0:
            problems.append(f"figure {item.get('figure_id')} has no source rows")
        if not item.get("sources"):
            problems.append(f"figure {item.get('figure_id')} has no source files")
    if not (TABLES / "paper_figure_manifest.csv").exists():
        problems.append("missing machine-readable CSV figure manifest")
    if not (TABLES / "entropy_vs_admissibility_query_delta.csv").exists():
        problems.append("missing per-query delta source table")
    return problems, warnings


def run_make_figures() -> tuple[int, str]:
    script = PIPELINE / "make_paper_figures.py"
    proc = subprocess.run(
        [sys.executable, str(script), "--json"],
        cwd=str(PIPELINE),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return proc.returncode, proc.stdout


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait", type=int, default=0, help="seconds to wait for required CSVs")
    parser.add_argument("--interval", type=int, default=30, help="poll interval while waiting")
    parser.add_argument("--make-figures", action="store_true", help="run make_paper_figures.py before figure validation")
    parser.add_argument("--json", action="store_true", help="print machine-readable status")
    args = parser.parse_args()

    deadline = time.time() + max(0, args.wait)
    frames: dict[str, pd.DataFrame] = {}
    problems: list[str] = []
    warnings: list[str] = []
    while True:
        frames, problems, warnings = validate_pipeline_outputs()
        missing_only = problems and all(item.startswith("missing:") for item in problems)
        if not missing_only or time.time() >= deadline:
            break
        time.sleep(max(1, args.interval))

    make_output = ""
    if args.make_figures and not problems:
        code, make_output = run_make_figures()
        if code != 0:
            problems.append(f"make_paper_figures.py failed with exit code {code}: {make_output.strip()}")

    figure_problems: list[str] = []
    figure_warnings: list[str] = []
    # If upstream CSVs are still missing, figure files are expected to be absent.
    # Keep the run in "pending" rather than misclassifying it as a figure failure.
    if (not problems and args.make_figures) or (REPORTS / "paper_figure_manifest.json").exists():
        figure_problems, figure_warnings = validate_figures()
    warnings.extend(figure_warnings)
    problems.extend(figure_problems)

    status = "ok" if not problems else ("pending" if all(p.startswith("missing:") for p in problems) else "failed")
    result = {
        "status": status,
        "root": str(ROOT),
        "checked_csvs": sorted(frames),
        "problems": problems,
        "warnings": warnings,
        "make_figures_output": make_output.strip(),
        "docx_modified": False,
    }
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if status == "ok" else (2 if status == "pending" else 1)


if __name__ == "__main__":
    raise SystemExit(main())
