from __future__ import annotations

import csv
import json
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd


CASES = ["Boeing_737", "Cessna_172"]

CURATED_SOTA_FIGURE_SUFFIXES = [
    "nature_primary_metric_forest.png",
    "nature_metric_matrix.png",
    "nature_query_family_matrix.png",
    "nature_risk_gate_scatter.png",
    "nature_evidence_no_evidence.png",
    "nature_ablation_waterfall.png",
    "nature_track_dotplot.png",
    "nature_volcano.png",
]

DEPRECATED_TOP_LEVEL_FIGURES = [
    "panel_01_evidence_flow_construction.png",
    "panel_18_package_recovery_flow.png",
]


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
        raise FileNotFoundError(f"Cannot locate {prefix!r} directory under {case_root}")
    return matches[0]


def case_dirs(case: str) -> dict[str, Path]:
    case_root = ROOT / "Cases" / case
    test = numbered_dir(case_root, "7 ")
    latest = numbered_dir(case_root, "8 ")
    support = latest / "_support_files"
    support.mkdir(parents=True, exist_ok=True)
    return {
        "case": case_root,
        "test": test,
        "latest": latest,
        "support": support,
        "sota_tables": test / "sota_evidence_chain_tables",
        "sota_figures": test / "sota_evidence_chain_figures",
        "deployment_figures": test / "deployment_speed_figures",
        "tree_figures": test / "result_trees_highres",
        "fmeca_tables": test / "fmeca_analysis_tables",
        "fmeca_figures": latest / "07_fmeca_analysis",
        "qwen_figures": case_root / "figures",
    }


def copy_file(src: Path, dst: Path, artifact_type: str, rows: list[dict]) -> None:
    if not src.exists() or not src.is_file():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    rows.append(
        {
            "case_id": dst.name.split("_")[0] if dst.name else "",
            "artifact_type": artifact_type,
            "source_path": str(src),
            "output_path": str(dst),
            "bytes": dst.stat().st_size,
            "created_time": datetime.now().isoformat(timespec="seconds"),
        }
    )


def move_deprecated_figures(paths: dict[str, Path], rows: list[dict]) -> None:
    archive = paths["support"] / "deprecated_duplicate_figures"
    archive.mkdir(parents=True, exist_ok=True)
    for suffix in DEPRECATED_TOP_LEVEL_FIGURES:
        src = paths["latest"] / f"{paths['case'].name}_{suffix}"
        if src.exists():
            dst = archive / src.name
            shutil.move(str(src), str(dst))
            rows.append(
                {
                    "case_id": paths["case"].name,
                    "artifact_type": "deprecated_moved_png",
                    "source_path": str(src),
                    "output_path": str(dst),
                    "bytes": dst.stat().st_size,
                    "created_time": datetime.now().isoformat(timespec="seconds"),
                }
            )


def row_count(path: Path) -> int | str:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            return max(0, sum(1 for _ in f) - 1)
    except Exception:
        return ""


def copy_csv_outputs(paths: dict[str, Path], rows: list[dict]) -> None:
    for source_dir, artifact_type in [
        (paths["sota_tables"], "complete_case_csv"),
        (paths["fmeca_tables"], "fmeca_case_csv"),
        (paths["sota_figures"], "figure_source_csv"),
        (paths["deployment_figures"], "deployment_source_csv"),
        (paths["fmeca_figures"], "fmeca_plot_source_csv"),
    ]:
        if not source_dir.exists():
            continue
        for src in sorted(source_dir.glob("*.csv")):
            dst = paths["latest"] / src.name
            copy_file(src, dst, artifact_type, rows)
            if rows:
                rows[-1]["rows"] = row_count(dst)


def copy_png_outputs(paths: dict[str, Path], rows: list[dict]) -> None:
    case = paths["case"].name
    for suffix in CURATED_SOTA_FIGURE_SUFFIXES:
        src = paths["sota_figures"] / f"{case}_{suffix}"
        copy_file(src, paths["latest"] / src.name, "curated_sota_png", rows)

    for name in [
        f"{case}_deployment_latency_quality_pareto.png",
        f"{case}_method_latency_bars.png",
    ]:
        src = paths["deployment_figures"] / name
        copy_file(src, paths["latest"] / name, "deployment_png", rows)

    if paths["tree_figures"].exists():
        for src in sorted(paths["tree_figures"].glob("*.png")):
            copy_file(src, paths["latest"] / src.name, "tree_png", rows)

    if paths["fmeca_figures"].exists():
        for src in sorted(paths["fmeca_figures"].glob("*.png")):
            copy_file(src, paths["latest"] / src.name, "fmeca_png", rows)

    qwen_src = paths["qwen_figures"] / "qwen_3b_generation_paradigm_full.png"
    copy_file(qwen_src, paths["latest"] / f"{case}_qwen_3b_generation_paradigm_full.png", "qwen_generation_png", rows)


def write_qwen_case_csvs(paths: dict[str, Path], rows: list[dict]) -> None:
    case = paths["case"].name
    qwen_dir = ROOT / "04_results" / "qwen_generation_paradigm"
    response_path = qwen_dir / "qwen_generation_responses.csv"
    summary_path = qwen_dir / f"{case}_qwen_generation_summary.csv"
    if not response_path.exists():
        return

    responses = pd.read_csv(response_path)
    case_responses = responses[responses["case_id"] == case].copy()
    out_response = paths["latest"] / f"{case}_qwen3b_generation_responses.csv"
    case_responses.to_csv(out_response, index=False, encoding="utf-8-sig")
    rows.append(
        {
            "case_id": case,
            "artifact_type": "qwen_generation_case_csv",
            "source_path": str(response_path),
            "output_path": str(out_response),
            "bytes": out_response.stat().st_size,
            "rows": len(case_responses),
            "created_time": datetime.now().isoformat(timespec="seconds"),
        }
    )

    chain_path = paths["sota_tables"] / f"{case}_complete_sota_evidence_chains.csv"
    if chain_path.exists():
        chains = pd.read_csv(chain_path)
        chain_cols = [
            c
            for c in chains.columns
            if c
            in {
                "query_id",
                "question_id",
                "method",
                "top_evidence_ids",
                "top_pages",
                "top_roles",
                "top_components",
                "top_sections",
                "evidence_chain_top10",
                "retrieval_chain_top10",
                "exact_recall_10",
                "accepted_recall_10",
                "evidence_package_recall_10",
                "accepted_page_recall_10",
                "accepted_section_recall_10",
                "grounded_score",
                "hallucination_risk_proxy",
                "role_coverage_proxy",
            }
        ]
        slim = chains[chain_cols].rename(columns={"method": "source_method"})
        merged = case_responses.merge(slim, on=["query_id", "question_id", "source_method"], how="left")
        out_merged = paths["latest"] / f"{case}_qwen3b_answers_with_evidence_chain_and_metrics.csv"
        merged.to_csv(out_merged, index=False, encoding="utf-8-sig")
        rows.append(
            {
                "case_id": case,
                "artifact_type": "qwen_generation_with_evidence_chain_csv",
                "source_path": f"{response_path}; {chain_path}",
                "output_path": str(out_merged),
                "bytes": out_merged.stat().st_size,
                "rows": len(merged),
                "created_time": datetime.now().isoformat(timespec="seconds"),
            }
        )

    copy_file(summary_path, paths["latest"] / summary_path.name, "qwen_generation_summary_csv", rows)


def write_manifest(paths: dict[str, Path], rows: list[dict]) -> None:
    case = paths["case"].name
    out = paths["latest"] / f"{case}_latest_outputs_manifest.csv"
    fieldnames = [
        "case_id",
        "artifact_type",
        "source_path",
        "output_path",
        "bytes",
        "rows",
        "created_time",
    ]
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def write_report(all_rows: dict[str, list[dict]]) -> None:
    report = ROOT / "reports" / "latest_case_figure_set_regeneration_report.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Latest Case Figure Set Regeneration Report",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "Scope: plotting/output assembly only. No model training, benchmark regeneration, split modification, or manual metric editing was performed.",
        "",
    ]
    for case, rows in all_rows.items():
        pngs = [
            r
            for r in rows
            if (r.get("artifact_type", "").endswith("_png") or "png" in r.get("artifact_type", ""))
            and not r.get("artifact_type", "").startswith("deprecated_")
        ]
        csvs = [r for r in rows if r.get("output_path", "").lower().endswith(".csv")]
        lines += [
            f"## {case}",
            "",
            f"- Latest folder: `{ROOT / 'Cases' / case / numbered_dir(ROOT / 'Cases' / case, '8 ').name}`",
            f"- PNG files copied/refreshed in this pass: {len(pngs)}",
            f"- CSV files copied/generated in this pass: {len(csvs)}",
            "- Deprecated duplicate flow/package-recovery panels were moved into `_support_files/deprecated_duplicate_figures` if present.",
            "",
            "### Key PNG outputs",
        ]
        for row in pngs:
            lines.append(f"- `{Path(row['output_path']).name}`")
        lines += ["", "### Key CSV outputs"]
        for row in csvs:
            name = Path(row["output_path"]).name
            if any(token in name for token in ["complete_sota", "top10", "metrics_by_method", "qwen3b"]):
                lines.append(f"- `{name}`")
        lines.append("")
    lines += [
        "## Notes",
        "",
        "- The complete SOTA evidence-chain CSVs contain per-query/per-method retrieval evidence, evidence-chain fields, top evidence identifiers, and evaluation metrics.",
        "- The Qwen 3B CSVs contain final generated answers for the evaluated generation paradigms and a merged variant with retrieval evidence-chain and metric columns where a retrieval source method exists.",
        "- Cessna deployment-speed figures were not fully refreshed in this pass because the deployment benchmark process did not finish within the available runtime; existing latest-folder deployment figures were retained.",
    ]
    report.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    all_rows: dict[str, list[dict]] = {}
    for case in CASES:
        paths = case_dirs(case)
        rows: list[dict] = []
        move_deprecated_figures(paths, rows)
        copy_png_outputs(paths, rows)
        copy_csv_outputs(paths, rows)
        write_qwen_case_csvs(paths, rows)
        write_manifest(paths, rows)
        all_rows[case] = rows
    write_report(all_rows)
    print(json.dumps({case: len(rows) for case, rows in all_rows.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
