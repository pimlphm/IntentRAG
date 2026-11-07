from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from textwrap import wrap

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
CASES = ["Boeing_737", "Cessna_172"]
DPI = 600

COLORS = {
    "fd": "#2155A4",
    "teal": "#1B9E9A",
    "green": "#2E7D32",
    "orange": "#D88C25",
    "red": "#C74E2D",
    "grey": "#8A9099",
    "light_grey": "#EEF1F4",
    "grid": "#DDE3EA",
    "text": "#20242A",
}

MODULE_LABELS = {
    "No full package optimizer": "Package optimizer",
    "No component graph": "Component graph",
    "No causal graph": "Causal graph",
    "No graph structure": "Graph structure",
    "No parent section": "Parent-section context",
    "No component profile": "Component profile",
}

ROUTE_METHODS = [
    ("BM25/word", "BM25"),
    ("Hybrid lexical", "Hybrid lexical"),
    ("Parent-section retrieval", "Parent-section"),
    ("Component-profile retrieval", "Component profile"),
    ("Component-graph informed RAG", "Component graph"),
    ("Causal-graph informed RAG", "Causal graph"),
    ("Evidence-flow baseline", "Evidence-flow"),
]


def case_dir(case_id: str, prefix: str) -> Path:
    return next(p for p in (ROOT / "Cases" / case_id).iterdir() if p.is_dir() and p.name.startswith(prefix))


def out_dir(case_id: str) -> Path:
    out = ROOT / "figures" / case_id
    out.mkdir(parents=True, exist_ok=True)
    return out


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.labelcolor": COLORS["text"],
            "xtick.color": COLORS["text"],
            "ytick.color": COLORS["text"],
            "axes.titlesize": 12,
            "axes.labelsize": 11.5,
            "xtick.labelsize": 10.4,
            "ytick.labelsize": 10.6,
            "legend.fontsize": 9,
            "savefig.facecolor": "white",
        }
    )


def wrap_label(text: str, width: int = 20) -> str:
    return "\n".join(wrap(str(text), width=width, break_long_words=False))


def clean_spines(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_visible(False)


def arrow_axes(ax: plt.Axes) -> None:
    clean_spines(ax)
    ax.tick_params(axis="both", length=4, width=0.8, color=COLORS["text"])
    ax.annotate(
        "",
        xy=(1.015, 0),
        xytext=(0, 0),
        xycoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", color=COLORS["text"], lw=0.9, mutation_scale=9),
        clip_on=False,
    )


def annotate_outside_bubble(ax: plt.Axes, x: float, y: float, value: float, size: float, last_col: bool = False) -> None:
    radius_pt = float(np.sqrt(size / np.pi))
    x_offset = -(radius_pt + 9) if last_col else radius_pt + 9
    ax.annotate(
        f"{value:.1f}",
        xy=(x, y),
        xytext=(x_offset, 0),
        textcoords="offset points",
        ha="right" if last_col else "left",
        va="center",
        fontsize=10.8,
        color="#111111",
        bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.86),
        zorder=5,
    )
    ax.annotate(
        "",
        xy=(0, 1.015),
        xytext=(0, 0),
        xycoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", color=COLORS["text"], lw=0.9, mutation_scale=9),
        clip_on=False,
    )


def save_png(fig: plt.Figure, path: Path) -> dict:
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    with Image.open(path) as img:
        width, height = img.size
    return {"dpi": DPI, "width": width, "height": height}


def check_png(path: Path) -> dict:
    result = {"output_png": str(path), "exists": path.exists()}
    if not path.exists():
        result.update({"ok": False, "reason": "missing"})
        return result
    with Image.open(path) as img:
        width, height = img.size
    size_kb = path.stat().st_size / 1024
    result.update(
        {
            "ok": size_kb > 50 and width >= 1200 and height >= 800,
            "size_kb": round(size_kb, 1),
            "width": width,
            "height": height,
        }
    )
    return result


def plot_ablation_module_impact(ablation: pd.DataFrame, case_id: str) -> tuple[plt.Figure, dict]:
    full = ablation.loc[ablation["ablation_view"].eq("Full FD-GRAG-RC-V21")].iloc[0]
    rows = []
    for raw, label in MODULE_LABELS.items():
        hit = ablation.loc[ablation["ablation_view"].eq(raw)]
        if hit.empty:
            continue
        row = hit.iloc[0]
        rows.append(
            {
                "module": label,
                "Grounded\nloss": max(0.0, (full["grounded_score"] - row["grounded_score"]) * 100),
                "Package recall\nloss": max(0.0, (full["evidence_package_recall_10"] - row["evidence_package_recall_10"]) * 100),
                "Risk proxy\nincrease": max(0.0, (row["hallucination_risk_proxy"] - full["hallucination_risk_proxy"]) * 100),
            }
        )
    data = pd.DataFrame(rows)
    metric_cols = ["Grounded\nloss", "Package recall\nloss", "Risk proxy\nincrease"]
    data["impact_rank"] = data[metric_cols].max(axis=1)
    data = data.sort_values("impact_rank", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(7.9, max(3.7, 0.55 * len(data) + 1.4)))
    x = np.arange(len(metric_cols))
    y = np.arange(len(data))
    vmax = max(1.0, float(data[metric_cols].to_numpy().max()))
    cmap = mpl.colors.LinearSegmentedColormap.from_list("impact", ["#EEF4FA", "#7CBCC0", "#D88C25", "#C74E2D"])

    for yi, row in data.iterrows():
        for xi, metric in enumerate(metric_cols):
            val = float(row[metric])
            size = 90 + 820 * (val / vmax) ** 0.9
            ax.scatter(xi, yi, s=size, color=cmap(val / vmax), edgecolor="#30343A", linewidth=0.45, zorder=3)
            annotate_outside_bubble(ax, xi, yi, val, size, last_col=xi == len(metric_cols) - 1)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_cols)
    ax.set_yticks(y)
    ax.set_yticklabels([wrap_label(v, 22) for v in data["module"]])
    ax.set_xlim(-0.6, len(metric_cols) - 0.25)
    ax.set_ylim(-0.65, len(data) - 0.35)
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.6)
    ax.set_xlabel("")
    arrow_axes(ax)
    fig.tight_layout()
    sidecar = {
        "metric_columns": metric_cols,
        "transformation": "percentage-point loss relative to the full model; non-negative impact values shown",
        "case_id": case_id,
    }
    return fig, sidecar


def plot_structured_retrieval_contribution(metrics: pd.DataFrame, case_id: str) -> tuple[plt.Figure, dict]:
    full = metrics.loc[metrics["method"].eq("FD-GRAG-RC-V21-final")].iloc[0]
    rows = []
    for method, label in ROUTE_METHODS:
        hit = metrics.loc[metrics["method"].eq(method)]
        if hit.empty:
            continue
        row = hit.iloc[0]
        rows.append(
            {
                "retrieval design": label,
                "Grounded score\ngain": max(0.0, (full["grounded_score"] - row["grounded_score"]) * 100),
                "Package recall\ngain": max(0.0, (full["evidence_package_recall_10"] - row["evidence_package_recall_10"]) * 100),
                "Risk proxy\nreduction": max(0.0, (row["hallucination_risk_proxy"] - full["hallucination_risk_proxy"]) * 100),
            }
        )
    data = pd.DataFrame(rows)
    metric_cols = ["Grounded score\ngain", "Package recall\ngain", "Risk proxy\nreduction"]
    data["max_gain"] = data[metric_cols].max(axis=1)
    data = data.sort_values("max_gain", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8.0, max(4.0, 0.5 * len(data) + 1.5)))
    x = np.arange(len(metric_cols))
    y = np.arange(len(data))
    vmax = max(1.0, float(data[metric_cols].to_numpy().max()))
    cmap = mpl.colors.LinearSegmentedColormap.from_list("gain", ["#EEF3F7", "#75BFC4", "#2155A4"])

    for yi, row in data.iterrows():
        for xi, metric in enumerate(metric_cols):
            val = float(row[metric])
            size = 80 + 760 * (val / vmax) ** 0.88
            ax.scatter(xi, yi, s=size, color=cmap(val / vmax), edgecolor="#30343A", linewidth=0.45, zorder=3)
            annotate_outside_bubble(ax, xi, yi, val, size, last_col=xi == len(metric_cols) - 1)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_cols)
    ax.set_yticks(y)
    ax.set_yticklabels([wrap_label(v, 22) for v in data["retrieval design"]])
    ax.set_xlim(-0.6, len(metric_cols) - 0.25)
    ax.set_ylim(-0.65, len(data) - 0.35)
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.6)
    ax.set_xlabel("")
    arrow_axes(ax)
    fig.tight_layout()
    sidecar = {
        "metric_columns": metric_cols,
        "transformation": "percentage-point gain of FD-GRAG-RC over each existing method row",
        "case_id": case_id,
    }
    return fig, sidecar


def remove_weak_outputs(case_id: str) -> list[str]:
    removed: list[str] = []
    out = out_dir(case_id)
    stems = [
        f"{case_id.lower()}_ablation_waterfall",
        f"{case_id.lower()}_component_role_tree",
        f"{case_id.lower()}_diagnostic_causal_tree",
    ]
    for stem in stems:
        for ext in [".png", ".json"]:
            path = out / f"{stem}{ext}"
            if path.exists():
                path.unlink()
                removed.append(str(path))
    return removed


def write_sidecar(path: Path, source: Path, sidecar: dict) -> None:
    payload = {
        "output_png": str(path),
        "source_artifact": str(source),
        "created_time": datetime.now().isoformat(timespec="seconds"),
        **sidecar,
    }
    path.with_suffix(".json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def update_case_manifest(case_id: str, new_rows: list[dict]) -> None:
    out = out_dir(case_id)
    manifest_path = out / "figure_manifest.csv"
    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path)
        drop_patterns = [
            "ablation_waterfall",
            "component_role_tree",
            "diagnostic_causal_tree",
            "ablation_module_impact_matrix",
            "structured_retrieval_contribution",
        ]
        keep = ~manifest["figure_name"].astype(str).str.contains("|".join(drop_patterns), case=False, regex=True)
        manifest = manifest.loc[keep].copy()
    else:
        manifest = pd.DataFrame()
    manifest = pd.concat([manifest, pd.DataFrame(new_rows)], ignore_index=True)
    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")


def update_reports(case_rows: list[dict], removed: list[str]) -> None:
    report = ROOT / "reports" / "ablation_and_structure_contribution_update.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Ablation and Structure-Contribution Figure Update",
        "",
        "This update replaces the old waterfall/tree-style presentation with result-backed contribution matrices.",
        "The figures read existing case-local metric and ablation CSVs only; no experiment result, score, split, or retrieval artifact is changed.",
        "",
        "## New figures",
        "",
        "| Case | Figure | Output path | Source data | Recommendation |",
        "|---|---|---|---|---|",
    ]
    for row in case_rows:
        source_name = Path(row["source_artifact"]).name
        lines.append(
            f"| {row['case_id']} | `{row['figure_name']}` | `{Path(row['output_png']).relative_to(ROOT).as_posix()}` | `case-local sota_evidence_chain_tables/{source_name}` | {row['recommendation']} |"
        )
    lines.extend(
        [
            "",
            "## Removed from current exported figure set",
            "",
            "- `*_ablation_waterfall.png`: replaced by module-impact matrix because a waterfall over-emphasizes a single scalar and weakens the structured-retrieval story.",
            "- `*_component_role_tree.png` and `*_diagnostic_causal_tree.png`: removed from the current result set because standalone tree diagrams make the tree design look like a mechanism claim rather than evidence-backed retrieval benefit.",
            "",
            "## Removed files",
            "",
        ]
    )
    if removed:
        lines.extend(f"- `{Path(p).relative_to(ROOT).as_posix()}`" for p in removed)
    else:
        lines.append("- The old waterfall/tree outputs were already absent from the current exported figure folders.")
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            "The new figures support the claim that FD-GRAG-RC integrates structured routes into a stronger evidence package on document-grounded metrics. They do not claim real maintenance accuracy, operational safety certification, or superiority on metrics not supported by the existing artifacts.",
            "",
        ]
    )
    report.write_text("\n".join(lines), encoding="utf-8")

    current_rows: list[dict] = []
    for case_id in CASES:
        manifest_path = out_dir(case_id) / "figure_manifest.csv"
        if manifest_path.exists():
            manifest = pd.read_csv(manifest_path)
            current_rows.extend(manifest.to_dict("records"))
    current = [
        "# Figure Redesign Report",
        "",
        "Scope: plotting/display-only updates. Existing experiment outputs, benchmark files, metric CSV files, retrieval results, and core method code were not modified.",
        "",
        "## Summary",
        f"- Current exported case PNG figures: {len(current_rows)}",
        "- Output folders: `figures/Boeing_737/` and `figures/Cessna_172/`",
        "- Export policy: PNG only; no SVG/PDF generated.",
        "- Curation policy: process-only, weak-contrast, and potentially misleading standalone tree/waterfall figures are omitted from the current exported figure set.",
        "",
        "## Current Exported Figure Set",
        "",
        "| Case | Figure | Source artifact | Recommendation / note |",
        "|---|---|---|---|",
    ]
    for row in current_rows:
        case_id = row.get("case_id", "")
        fig = row.get("figure_name", "")
        source = row.get("source_artifact", "")
        recommendation = row.get("recommendation", "")
        notes = row.get("notes", "")
        note = recommendation if not pd.isna(recommendation) and str(recommendation).strip() else notes
        if pd.isna(note) or not str(note).strip():
            note = "Current curated export"
        try:
            source_rel = Path(source).relative_to(ROOT).as_posix()
        except Exception:
            source_rel = str(source)
        if "sota_evidence_chain_tables" in source_rel:
            source_rel = f"case-local sota_evidence_chain_tables/{Path(source).name}"
        elif row.get("notes", "").startswith("Legacy-style restored"):
            source_rel = "case-local latest PNG atlas"
        current.append(f"| {case_id} | `{fig}` | `{source_rel}` | {note} |")
    current.extend(
        [
            "",
            "## Recently Removed From Current Export",
            "",
            "- `*_ablation_waterfall.png`: replaced by `*_ablation_module_impact_matrix.png`.",
            "- `*_component_role_tree.png` and `*_diagnostic_causal_tree.png`: replaced by `*_structured_retrieval_contribution.png` as the main evidence-backed structure contribution figure.",
            "",
            "## Boundary",
            "",
            "This report describes figure assembly and visual presentation only. It does not recalculate metrics, rerun retrieval, alter benchmark splits, tune labels, or change any CSV/JSON/NPY artifact semantics.",
            "",
        ]
    )
    (ROOT / "reports" / "figure_redesign_report.md").write_text("\n".join(current), encoding="utf-8")


def build_case(case_id: str) -> tuple[list[dict], list[str], list[dict]]:
    test = case_dir(case_id, "7")
    out = out_dir(case_id)
    ablation_path = test / "sota_evidence_chain_tables" / f"{case_id}_ablation_metrics.csv"
    metrics_path = test / "sota_evidence_chain_tables" / f"{case_id}_sota_metrics_by_method.csv"
    ablation = pd.read_csv(ablation_path)
    metrics = pd.read_csv(metrics_path)

    removed = remove_weak_outputs(case_id)
    rows: list[dict] = []
    qa_rows: list[dict] = []

    jobs = [
        (
            f"{case_id.lower()}_ablation_module_impact_matrix.png",
            ablation_path,
            lambda: plot_ablation_module_impact(ablation, case_id),
            "Main text",
            "Module-impact matrix replacing the old waterfall; values are pp loss/risk increase vs full model.",
        ),
        (
            f"{case_id.lower()}_structured_retrieval_contribution.png",
            metrics_path,
            lambda: plot_structured_retrieval_contribution(metrics, case_id),
            "Main text",
            "FD-GRAG-RC gain over partial/single-route retrieval designs on package and risk-aware metrics.",
        ),
    ]
    for name, source, maker, rec, notes in jobs:
        fig, sidecar = maker()
        out_png = out / name
        info = save_png(fig, out_png)
        write_sidecar(out_png, source, sidecar)
        row = {
            "case_id": case_id,
            "figure_name": name,
            "source_artifact": str(source),
            "output_png": str(out_png),
            "dpi": DPI,
            "width": info["width"],
            "height": info["height"],
            "created_time": datetime.now().isoformat(timespec="seconds"),
            "notes": notes,
            "recommendation": rec,
        }
        rows.append(row)
        qa_rows.append(check_png(out_png))

    update_case_manifest(case_id, rows)
    qa_path = out / "visual_qa.csv"
    if qa_path.exists():
        qa = pd.read_csv(qa_path)
        drop_patterns = [
            "ablation_waterfall",
            "component_role_tree",
            "diagnostic_causal_tree",
            "ablation_module_impact_matrix",
            "structured_retrieval_contribution",
        ]
        keep = ~qa["output_png"].astype(str).str.contains("|".join(drop_patterns), case=False, regex=True)
        qa = qa.loc[keep].copy()
        qa = pd.concat([qa, pd.DataFrame(qa_rows)], ignore_index=True)
    else:
        qa = pd.DataFrame(qa_rows)
    qa.to_csv(qa_path, index=False, encoding="utf-8-sig")
    return rows, removed, qa_rows


def main() -> None:
    apply_style()
    all_rows: list[dict] = []
    all_removed: list[str] = []
    all_qa: list[dict] = []
    for case_id in CASES:
        rows, removed, qa = build_case(case_id)
        all_rows.extend(rows)
        all_removed.extend(removed)
        all_qa.extend(qa)
    update_reports(all_rows, all_removed)
    print({"generated": len(all_rows), "removed": len(all_removed), "qa_pass": sum(bool(r.get("ok")) for r in all_qa)})


if __name__ == "__main__":
    main()
