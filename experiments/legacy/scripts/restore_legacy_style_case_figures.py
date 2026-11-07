from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
CASE_IDS = ["Boeing_737", "Cessna_172"]
DPI_NOTE = 600


LEGACY_FIGURES = [
    ("query_family_matrix", "nature_query_family_matrix", "Main text"),
    ("risk_gate_scatter", "nature_risk_gate_scatter", "Main text"),
    ("evidence_no_evidence", "nature_evidence_no_evidence", "Main text"),
    ("fmeca_risk_matrix", "fmeca_risk_matrix", "Main text"),
    ("fmeca_role_control_heatmap", "fmeca_role_control_heatmap", "Supplementary"),
]

OMITTED_FIGURES = [
    ("metric_matrix", "nature_metric_matrix", "Highly saturated matrix; duplicates the clearer primary metric forest."),
    ("primary_metric_forest", "nature_primary_metric_forest", "Mixed benchmark figure; many scales are saturated or tied, so it should not be used as a far-superiority figure."),
    ("deployment_latency_quality_pareto", "deployment_latency_quality_pareto", "Useful for deployment discussion but not central to the method-advantage claim."),
    ("candidate_pool_ceiling", "panel_10_candidate_pool_ceiling", "Near-saturated robustness panel; weak contrast for main method advantage."),
    ("slot_complexity_stability", "panel_12_slot_complexity_stability", "Near-saturated robustness panel; overlaps with package-recovery narrative."),
    ("reliability_oncoprint", "panel_17_reliability_oncoprint", "Mostly pass cells; better discussed briefly than shown as a standalone figure."),
    ("fmeca_query_failure_mode_controls", "fmeca_query_failure_mode_controls", "Mechanism-adjacent but less direct than FMECA role-control and risk-matrix figures."),
    ("ablation_waterfall", "nature_ablation_waterfall", "Replaced by module-impact matrix; waterfall over-emphasizes a single scalar."),
    ("component_role_tree", "component_role_tree_highres", "Standalone tree diagram makes the retrieval-tree design look mechanism-only; replaced by result-backed contribution matrix."),
    ("diagnostic_causal_tree", "diagnostic_causal_tree_highres", "Standalone tree diagram makes the retrieval-tree design look mechanism-only; replaced by result-backed contribution matrix."),
    ("evidence_flow_construction", "panel_01_evidence_flow_construction", "Method-process figure; user requested reference-style comparison/result figures instead."),
    ("package_recovery_flow", "panel_18_package_recovery_flow", "Method-process figure; user requested reference-style comparison/result figures instead."),
]


def latest_dir(case_id: str) -> Path:
    return next(p for p in (ROOT / "Cases" / case_id).iterdir() if p.is_dir() and p.name.startswith("8"))


def output_dir(case_id: str) -> Path:
    out = ROOT / "figures" / case_id
    out.mkdir(parents=True, exist_ok=True)
    return out


def reports_dir() -> Path:
    out = ROOT / "reports"
    out.mkdir(parents=True, exist_ok=True)
    return out


def check_png_basic_quality(output_png: Path) -> dict:
    result = {"output_png": str(output_png), "exists": output_png.exists()}
    if not output_png.exists():
        result.update({"ok": False, "reason": "missing"})
        return result
    size_kb = output_png.stat().st_size / 1024
    with Image.open(output_png) as img:
        width, height = img.size
        arr = np.asarray(img.convert("RGB"))
    nonwhite = np.any(arr < 245, axis=2)
    coords = np.argwhere(nonwhite)
    if coords.size:
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)
        content_area = max(1, (y1 - y0 + 1) * (x1 - x0 + 1))
        blank_border_fraction = 1 - content_area / max(1, width * height)
    else:
        blank_border_fraction = 1.0
    ok = size_kb > 50 and width >= 900 and height >= 600 and blank_border_fraction < 0.72
    result.update(
        {
            "ok": bool(ok),
            "size_kb": round(size_kb, 1),
            "width": width,
            "height": height,
            "blank_border_fraction": round(float(blank_border_fraction), 3),
        }
    )
    return result


def clean_generated_case_outputs(out: Path) -> None:
    resolved = out.resolve()
    expected = (ROOT / "figures").resolve()
    if expected not in resolved.parents:
        raise RuntimeError(f"Refusing to clean outside figures directory: {resolved}")
    for pattern in ("*.png", "*.json"):
        for path in out.glob(pattern):
            path.unlink()


def restore_case(case_id: str) -> tuple[list[dict], list[dict], list[dict]]:
    src_dir = latest_dir(case_id)
    out = output_dir(case_id)
    clean_generated_case_outputs(out)
    created = datetime.now().isoformat(timespec="seconds")
    manifest: list[dict] = []
    qa: list[dict] = []
    before_after: list[dict] = []
    for out_stem, old_stem, recommendation in LEGACY_FIGURES:
        src = src_dir / f"{case_id}_{old_stem}.png"
        if not src.exists():
            raise FileNotFoundError(src)
        dst = out / f"{case_id.lower()}_{out_stem}.png"
        shutil.copy2(src, dst)
        with Image.open(dst) as img:
            width, height = img.size
        sidecar = {
            "output_png": str(dst),
            "source_legacy_png": str(src),
            "created_time": created,
            "dpi_note": DPI_NOTE,
            "width": width,
            "height": height,
            "style_policy": "Restored legacy visual style and layout from the current latest case figure atlas.",
            "data_policy": "No experimental metrics, benchmark files, retrieval results, or artifact values were modified.",
            "recommendation": recommendation,
        }
        dst.with_suffix(".json").write_text(json.dumps(sidecar, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest.append(
            {
                "case_id": case_id,
                "figure_name": dst.name,
                "source_artifact": str(src),
                "output_png": str(dst),
                "dpi": DPI_NOTE,
                "width": width,
                "height": height,
                "created_time": created,
                "notes": f"Legacy-style restored figure; recommended: {recommendation}.",
            }
        )
        qa.append(check_png_basic_quality(dst))
        before_after.append(
            {
                "case_id": case_id,
                "figure": out_stem,
                "legacy_source_png": str(src),
                "current_output_png": str(dst),
                "recommendation": recommendation,
                "change": "Restored legacy style/format in unified case output folder.",
            }
        )
    pd.DataFrame(manifest).to_csv(out / "figure_manifest.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(qa).to_csv(out / "visual_qa.csv", index=False, encoding="utf-8-sig")
    return manifest, qa, before_after


def write_report(all_manifest: list[dict], all_qa: list[dict], before_after: list[dict]) -> None:
    report_dir = reports_dir()
    pd.DataFrame(before_after).to_csv(report_dir / "figure_redesign_before_after.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(all_qa).to_csv(report_dir / "figure_redesign_visual_qa.csv", index=False, encoding="utf-8-sig")
    ok_count = sum(1 for row in all_qa if row.get("ok"))
    lines = [
        "# Figure Style Restore Report",
        "",
        "Scope: plotting/display-only restore to the legacy case-figure style. Existing experiment outputs, benchmark files, metric CSV files, retrieval results, and core method code were not modified.",
        "",
        "## Summary",
        f"- Restored PNG figures: {len(all_manifest)}",
        f"- Visual QA passed: {ok_count}/{len(all_qa)}",
        "- Output folders: `figures/Boeing_737/` and `figures/Cessna_172/`",
        "- Source style: current `Cases/<case>/8 最新图集/` PNG atlas.",
        "- Export policy: PNG only; no SVG/PDF generated.",
        "- Curation policy: figures that do not clearly support the method-advantage or claim-boundary narrative are omitted from the unified output folders.",
        "",
        "## Current Figure Set",
        "",
        "| Case | Figure | Source legacy PNG | Recommendation |",
        "|---|---|---|---|",
    ]
    for row in all_manifest:
        lines.append(
            f"| {row['case_id']} | `{row['output_png']}` | `{row['source_artifact']}` | {row['notes'].split('recommended: ')[-1].rstrip('.')} |"
        )
    lines += [
        "",
        "## Omitted From Unified Output",
        "",
        "| Figure stem | Legacy source stem | Reason |",
        "|---|---|---|",
    ]
    for stem, legacy, reason in OMITTED_FIGURES:
        lines.append(f"| `{stem}` | `{legacy}` | {reason} |")
    lines += [
        "",
        "## Boundary",
        "",
        "This restore only changes the displayed figure files and figure assembly. It does not recalculate metrics, rerun retrieval, alter benchmark splits, or change any CSV/JSON/NPY artifact semantics.",
    ]
    (report_dir / "figure_redesign_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    all_manifest: list[dict] = []
    all_qa: list[dict] = []
    before_after: list[dict] = []
    for case_id in CASE_IDS:
        manifest, qa, ba = restore_case(case_id)
        all_manifest.extend(manifest)
        all_qa.extend(qa)
        before_after.extend(ba)
    write_report(all_manifest, all_qa, before_after)
    print(json.dumps({"restored": len(all_manifest), "qa_pass": sum(1 for row in all_qa if row.get("ok"))}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
