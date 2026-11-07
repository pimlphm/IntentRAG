from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path


CASES = ["Boeing_737", "Cessna_172"]

BASE_FINAL_SUFFIXES = {
    "nature_primary_metric_forest.png",
    "nature_metric_matrix.png",
    "nature_query_family_matrix.png",
    "nature_risk_gate_scatter.png",
    "nature_evidence_no_evidence.png",
    "nature_ablation_waterfall.png",
    "nature_track_dotplot.png",
    "nature_volcano.png",
    "component_role_tree_highres.png",
    "diagnostic_causal_tree_highres.png",
    "fmeca_component_criticality.png",
    "fmeca_query_failure_mode_controls.png",
    "fmeca_risk_matrix.png",
    "fmeca_role_control_heatmap.png",
    "qwen_3b_generation_paradigm_full.png",
}

# Only Boeing deployment figures were refreshed in the current pass. The Cessna
# deployment benchmark did not finish, so stale deployment figures are excluded.
CASE_EXTRA_SUFFIXES = {
    "Boeing_737": {
        "deployment_latency_quality_pareto.png",
        "method_latency_bars.png",
    },
    "Cessna_172": set(),
}

GENERATED_FIGURE_DIR_NAMES = {
    "sota_evidence_chain_figures",
    "deployment_speed_figures",
    "result_trees_highres",
    "reference_style_figures",
}

SUPPORT_DUPLICATE_DIRS = {
    "axis_arrow_originals",
    "deprecated_duplicate_figures",
    "non_latest_removed_from_top_level",
    "redundant_figures",
    "redundant_figures_round2",
    "replaced_risk_gate_scatter",
    "replaced_risk_gate_scatter_dashed_lines",
    "replaced_risk_gate_scatter_round3",
}


def find_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "Cases").exists() and (parent / "04_results").exists():
            return parent
    raise FileNotFoundError("Cannot locate project root.")


ROOT = find_root()


def numbered_dir(case_root: Path, prefix: str) -> Path:
    matches = sorted(p for p in case_root.iterdir() if p.is_dir() and p.name.startswith(prefix))
    if not matches:
        raise FileNotFoundError(f"Cannot find {prefix!r} under {case_root}")
    return matches[0]


def final_png_names(case: str) -> set[str]:
    suffixes = set(BASE_FINAL_SUFFIXES) | set(CASE_EXTRA_SUFFIXES.get(case, set()))
    return {f"{case}_{suffix}" for suffix in suffixes}


def remove_file(path: Path, log: list[dict], reason: str) -> None:
    if not path.exists() or not path.is_file():
        return
    size = path.stat().st_size
    path.unlink()
    log.append({"removed": str(path), "bytes": size, "reason": reason})


def remove_dir(path: Path, log: list[dict], reason: str) -> None:
    if not path.exists() or not path.is_dir():
        return
    file_count = sum(1 for p in path.rglob("*") if p.is_file())
    shutil.rmtree(path)
    log.append({"removed": str(path), "files": file_count, "reason": reason})


def case_cleanup(case: str) -> dict:
    case_root = ROOT / "Cases" / case
    latest = numbered_dir(case_root, "8 ")
    test = numbered_dir(case_root, "7 ")
    allowed = final_png_names(case)
    removed: list[dict] = []

    for png in sorted(latest.glob("*.png")):
        if png.name not in allowed:
            remove_file(png, removed, "non-final or stale top-level PNG")

    # FMECA figures are already copied to the latest-folder top level.
    remove_dir(latest / "07_fmeca_analysis", removed, "duplicate FMECA subfolder; final PNG/CSV copies are at latest-folder top level")

    support = latest / "_support_files"
    if support.exists():
        for name in SUPPORT_DUPLICATE_DIRS:
            remove_dir(support / name, removed, "non-latest backup/duplicate figure folder")

    for child in test.iterdir():
        if child.is_dir() and child.name in GENERATED_FIGURE_DIR_NAMES:
            remove_dir(child, removed, "intermediate generated figure directory; final copies are in latest-folder top level")

    for step in [numbered_dir(case_root, "2 "), numbered_dir(case_root, "3 ")]:
        remove_dir(step / "result_trees_highres", removed, "intermediate tree figure directory; final copies are in latest-folder top level")

    remove_file(case_root / "figures" / "qwen_3b_generation_paradigm_full.png", removed, "duplicate Qwen PNG; final copy is in latest-folder top level")

    remaining_pngs = sorted(p.name for p in latest.glob("*.png"))
    missing = sorted(allowed - set(remaining_pngs))
    unexpected = sorted(set(remaining_pngs) - allowed)
    return {
        "case": case,
        "latest_dir": str(latest),
        "final_png_count": len(remaining_pngs),
        "final_pngs": remaining_pngs,
        "missing_expected_pngs": missing,
        "unexpected_pngs": unexpected,
        "removed": removed,
    }


def remove_global_duplicate_figures(log: list[dict]) -> None:
    for folder in [
        ROOT / "figures",
        ROOT / "05_figures" / "fmeca_png",
    ]:
        remove_dir(folder, log, "old/global figure output directory; final case PNGs are kept only in the two latest folders")
    for png in (ROOT / "04_results").rglob("*.png"):
        remove_file(png, log, "old/generated result PNG outside latest folders")


def main() -> None:
    global_removed: list[dict] = []
    outputs = [case_cleanup(case) for case in CASES]
    remove_global_duplicate_figures(global_removed)
    report = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "scope": "Keep final plotting outputs only in the two case latest-figure folders.",
        "cases": outputs,
        "global_removed": global_removed,
    }
    out = ROOT / "reports" / "latest_figure_folder_final_cleanup_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
