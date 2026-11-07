from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import textwrap
from pathlib import Path
from typing import Callable, Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath
from matplotlib.patches import Circle, Ellipse, FancyBboxPatch
from matplotlib.patches import PathPatch


mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.weight": "regular",
        "axes.labelweight": "bold",
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


CASE_LABELS = {"Boeing_737": "Boeing 737", "Cessna_172": "Cessna 172"}
METHOD_ORDER = [
    "BM25/word",
    "TFIDF-char",
    "Hybrid lexical",
    "FD-GRAG-RC-V21-exact",
    "FD-GRAG-RC-V21-package",
    "FD-GRAG-RC-V21-cue-route",
    "FD-GRAG-RC-V21-final",
]
METHOD_LABELS = {
    "BM25/word": "BM25",
    "TFIDF-char": "Char TF-IDF",
    "Hybrid lexical": "Hybrid",
    "FD-GRAG-RC-V21-exact": "V21 Exact Route",
    "FD-GRAG-RC-V21-package": "V21 Package",
    "FD-GRAG-RC-V21-cue-route": "V21 Route Ensemble",
    "FD-GRAG-RC-V21-final": "FD-GRAG-RC-V21",
}
METHOD_COLORS = {
    "BM25/word": "#8C8C8C",
    "TFIDF-char": "#B9A44C",
    "Hybrid lexical": "#56B4E9",
    "FD-GRAG-RC-V21-exact": "#7A4C9F",
    "FD-GRAG-RC-V21-package": "#E69F00",
    "FD-GRAG-RC-V21-cue-route": "#2F8F6B",
    "FD-GRAG-RC-V21-final": "#D55E00",
}
METRIC_LABELS = {
    "exact_recall_10": "Exact R@10",
    "accepted_recall_10": "Accepted R@10",
    "evidence_package_recall_10": "Package R@10",
    "accepted_page_recall_10": "Page R@10",
    "accepted_section_recall_10": "Section R@10",
    "grounded_score": "Grounded score",
    "hallucination_risk_proxy": "Risk proxy",
}
FAMILY_LABELS = {
    "engine_oil_powerplant": "Engine / Oil / Powerplant",
    "landing_gear_brake": "Landing Gear / Brake",
    "air_conditioning_pneumatic": "Air Conditioning / Pneumatic",
    "flight_control": "Flight Control",
    "navigation_avionics": "Navigation / Avionics",
    "electrical": "Electrical Power",
    "hydraulic": "Hydraulic",
    "fuel": "Fuel System",
    "other_component": "Other Components",
    "no_component": "Cross-System Context",
    "": "Cross-System Context",
}
ROLE_LABELS = {
    "symptom": "Symptom",
    "condition": "Condition",
    "threshold": "Threshold",
    "warning": "Warning",
    "limitation": "Limitation",
    "note": "Note",
    "confirmed_cause": "Cause",
    "excluded_cause": "Excluded Cause",
    "action": "Maintenance Action",
    "inspection": "Inspection",
    "replacement": "Replacement",
    "authority": "Manual / Authority",
    "evidence": "Evidence",
    "general": "General Context",
}
RELATION_LABELS = {
    "condition_to_action": "Condition -> Action",
    "threshold_to_condition": "Threshold -> Condition",
    "warning_to_action": "Warning -> Action",
    "cause_to_action": "Cause -> Action",
    "cause_to_inspection": "Cause -> Inspection",
    "symptom_to_cause": "Symptom -> Cause",
    "symptom_to_excluded_cause": "Symptom -> Excluded Cause",
    "cause_to_replacement": "Cause -> Replacement",
}
COLORS = {
    "ink": "#17212B",
    "deep_ink": "#0B1320",
    "white": "#FFFFFF",
    "muted": "#657286",
    "grid": "#E6EAF0",
    "blue": "#3E6AA7",
    "green": "#2F8F6B",
    "orange": "#E69F00",
    "red": "#D55E00",
    "purple": "#7A4C9F",
    "cyan": "#56B4E9",
    "pale_blue": "#E8F0FA",
    "pale_green": "#E8F5EF",
    "pale_red": "#F8EDE8",
    "pale_orange": "#FFF4DA",
    "pale_purple": "#F2ECF8",
    "axes_bg": "#FFFFFF",
    "title_band": "#0B1320",
    "subtitle_band": "#FFF4D6",
}


def find_root(start: Path) -> Path:
    for p in [start.resolve(), *start.resolve().parents, Path.cwd().resolve(), *Path.cwd().resolve().parents]:
        if (p / "Cases").exists() and (p / "04_results" / "v21").exists():
            return p
    raise FileNotFoundError("V21 package root not found.")


ROOT = find_root(Path(__file__))
RESULTS = ROOT / "04_results" / "v21"


def first_dir(case_dir: Path, prefix: str) -> Path:
    matches = sorted([p for p in case_dir.glob(f"{prefix}*") if p.is_dir()])
    if not matches:
        raise FileNotFoundError(f"No directory beginning with {prefix!r} in {case_dir}")
    return matches[0]


def first_file(folder: Path, pattern: str) -> Path:
    matches = sorted(folder.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No file matching {pattern} in {folder}")
    return matches[0]


def read_csv(path: Path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


def clean(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = re.sub(r"\s+", " ", str(value)).strip()
    return "" if text.lower() in {"nan", "none", "null"} else text


def wrap_label(text: object, width: int = 18) -> str:
    s = clean(text)
    return "\n".join(textwrap.wrap(s, width=width, break_long_words=False, break_on_hyphens=False)) or s


def natural_query_label(text: object) -> str:
    return wrap_label(clean(text).replace("_", " ").title(), 20)


def compact_query_label(text: object) -> str:
    raw = clean(text).replace("_", " ").strip().lower()
    aliases = {
        "authority or manual reference": "Authority",
        "component to failure": "Component-to-failure",
        "condition to action": "Condition-to-action",
        "cross page evidence query": "Cross-page",
        "excluded cause": "Excluded-cause",
        "multi hop diagnosis": "Multi-hop",
        "symptom to action": "Symptom-to-action",
        "symptom to cause": "Symptom-to-cause",
        "warning or limitation": "Warning/limit",
    }
    return aliases.get(raw, clean(text).replace("_", " ").title())


def case_paths(case_key: str) -> dict[str, Path]:
    case_dir = ROOT / "Cases" / case_key
    test_dir = first_dir(case_dir, "7 ")
    return {
        "case": case_dir,
        "step1": first_dir(case_dir, "1 "),
        "step2": first_dir(case_dir, "2 "),
        "step3": first_dir(case_dir, "3 "),
        "align": first_dir(case_dir, "4 "),
        "test": test_dir,
        "out": test_dir / "reference_style_figures",
        "code": case_dir / "reference_style_figure_code",
    }


def get_data(case_key: str) -> dict[str, pd.DataFrame]:
    paths = case_paths(case_key)
    return {
        "metrics_case": read_csv(RESULTS / "v21_metrics_by_case.csv"),
        "metrics_track": read_csv(RESULTS / "v21_metrics_by_case_track.csv"),
        "pred": read_csv(RESULTS / "v21_retrieval_predictions_full.csv"),
        "track_scores": read_csv(RESULTS / "v21_target_attainment_matrix.csv"),
        "safety": read_csv(RESULTS / "v21_safety_gate_metrics.csv"),
        "pool": read_csv(RESULTS / "v21_candidate_pool_upper_bound.csv"),
        "evidence": read_csv(first_file(paths["step1"], "*_evidence_objects.csv")),
        "causal": read_csv(first_file(paths["step2"], "*_causal_edges.csv")),
        "profiles": read_csv(first_file(paths["step3"], "*_component_profiles.csv")),
        "qa": read_csv(first_file(paths["align"], "*_qa_warehouse_v21_full.csv")),
    }


def method_subset(df: pd.DataFrame) -> pd.DataFrame:
    order = {m: i for i, m in enumerate(METHOD_ORDER)}
    sub = df[df["method"].isin(METHOD_ORDER)].copy()
    sub["method_order"] = sub["method"].map(order)
    sub["method_label"] = sub["method"].map(METHOD_LABELS)
    return sub.sort_values("method_order")


def style_axis(ax, grid_axis: str = "x"):
    ax.set_facecolor(COLORS["axes_bg"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#1C2735")
    ax.spines["bottom"].set_color("#1C2735")
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.tick_params(colors=COLORS["deep_ink"], labelsize=14.2, pad=5, width=1.2, length=5.2)
    ax.grid(False)
    ax.set_axisbelow(True)
    ax.annotate("", xy=(1.018, 0), xytext=(0, 0), xycoords="axes fraction", arrowprops=dict(arrowstyle="-|>", lw=1.75, color="#1C2735", mutation_scale=22), annotation_clip=False, zorder=20)
    ax.annotate("", xy=(0, 1.018), xytext=(0, 0), xycoords="axes fraction", arrowprops=dict(arrowstyle="-|>", lw=1.75, color="#1C2735", mutation_scale=22), annotation_clip=False, zorder=20)


def figure_title(fig, title: str, subtitle: str | None = None):
    return None


def add_reference_band(ax, y0=0.95, y1=1.005, label="reference zone"):
    return None


def save_panel(fig, outbase: Path, source: pd.DataFrame, caption: str, metadata: dict):
    outbase.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outbase.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white", pad_inches=0.18)
    plt.close(fig)
    source.to_csv(outbase.with_name(outbase.name + "_source_data.csv"), index=False, encoding="utf-8-sig")
    outbase.with_name(outbase.name + "_caption.md").write_text(caption, encoding="utf-8")
    outbase.with_name(outbase.name + "_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return outbase.with_suffix(".png")


def card(ax, x, y, w, h, text, fc, ec, fontsize=12.6, weight="bold"):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        transform=ax.transAxes,
        boxstyle="round,pad=0.02,rounding_size=0.035",
        facecolor=fc,
        edgecolor=ec,
        linewidth=1.6,
        zorder=3,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, wrap_label(text, 22), transform=ax.transAxes, ha="center", va="center", fontsize=max(fontsize, 12.6), color=COLORS["ink"], fontweight=weight, linespacing=1.08, zorder=4)


def arrow(ax, start, end, color="#97A3B5"):
    ax.annotate("", xy=end, xytext=start, xycoords=ax.transAxes, textcoords=ax.transAxes, arrowprops=dict(arrowstyle="-|>", lw=3.0, color=color, shrinkA=5, shrinkB=5, mutation_scale=28), zorder=6)


def panel_01_pipeline(case_key: str, data: dict[str, pd.DataFrame]) -> Path:
    label = CASE_LABELS[case_key]
    paths = case_paths(case_key)
    counts = pd.DataFrame(
        [
            {"stage": "Evidence objects", "count": len(data["evidence"])},
            {"stage": "Causal KG edges", "count": len(data["causal"])},
            {"stage": "Component profiles", "count": len(data["profiles"])},
            {"stage": "Benchmark questions", "count": len(data["qa"])},
        ]
    )
    fig, ax = plt.subplots(figsize=(18.0, 5.8))
    figure_title(fig, f"{label}: Evidence-Flow Construction Summary", "Document evidence is structured into role, component, causal and package-level retrieval objects.")
    ax.set_axis_off()
    y = 0.52
    stages = [
        ("Raw manuals\nand pages", COLORS["pale_blue"], COLORS["blue"]),
        (f"{len(data['evidence']):,}\nevidence objects", COLORS["pale_green"], COLORS["green"]),
        (f"{len(data['causal']):,}\ncausal KG edges", "#F0EAF7", COLORS["purple"]),
        (f"{len(data['profiles']):,}\ncomponent profiles", "#FFF3D7", COLORS["orange"]),
        (f"{len(data['qa']):,}\nQA tasks", COLORS["pale_red"], COLORS["red"]),
    ]
    xs = np.linspace(0.04, 0.82, len(stages))
    card_w = 0.14
    for i, (txt, fc, ec) in enumerate(stages):
        card(ax, xs[i], y, card_w, 0.24, txt, fc, ec, fontsize=13.4)
        if i < len(stages) - 1:
            arrow(ax, (xs[i] + card_w * 0.78, y + 0.285), (xs[i + 1] + card_w * 0.22, y + 0.285), ec)
    ax.text(0.06, 0.18, "All counts are read from the case-local V21 artifacts. The diagram is a methodology summary, not a simulated score.", transform=ax.transAxes, fontsize=12.0, color=COLORS["muted"])
    return save_panel(
        fig,
        paths["out"] / f"{case_key}_panel_01_evidence_flow_construction",
        counts,
        f"# {label}: Evidence-flow construction summary\n\nSingle-panel methodology summary with true artifact counts from the V21 case package.\n",
        {"sources": ["case evidence objects", "case causal edges", "case component profiles", "case QA warehouse"]},
    )


def panel_02_roles(case_key: str, data: dict[str, pd.DataFrame]) -> Path:
    label = CASE_LABELS[case_key]
    paths = case_paths(case_key)
    ev = data["evidence"].fillna("")
    counts = ev["role"].map(lambda x: ROLE_LABELS.get(clean(x), clean(x).replace("_", " ").title())).value_counts().head(12).sort_values()
    source = counts.rename_axis("evidence_role").reset_index(name="evidence_objects")
    fig, ax = plt.subplots(figsize=(10.8, max(5.2, 0.42 * len(counts) + 2.0)))
    figure_title(fig, f"{label}: Evidence Role Distribution", "Diagnostic evidence is partitioned into action, authority, warning, threshold, cause and context roles.")
    ax.barh(counts.index, counts.values, color=COLORS["purple"], zorder=3)
    ax.set_xlabel("Evidence objects", fontsize=14.6)
    ax.set_ylabel("")
    style_axis(ax)
    fig.subplots_adjust(left=0.28, right=0.96, top=0.94, bottom=0.14)
    return save_panel(fig, paths["out"] / f"{case_key}_panel_02_evidence_roles", source, f"# {label}: Evidence role distribution\n\nCounts of evidence objects by diagnostic role.\n", {"sources": ["case evidence objects"]})


def panel_03_components(case_key: str, data: dict[str, pd.DataFrame]) -> Path:
    label = CASE_LABELS[case_key]
    paths = case_paths(case_key)
    ev = data["evidence"].fillna("")
    counts = ev["component_family"].map(lambda x: FAMILY_LABELS.get(clean(x), clean(x).replace("_", " ").title())).value_counts().head(10).sort_values()
    source = counts.rename_axis("component_family").reset_index(name="evidence_objects")
    fig, ax = plt.subplots(figsize=(11.8, max(5.2, 0.44 * len(counts) + 2.0)))
    figure_title(fig, f"{label}: Component Evidence Coverage", "The component hierarchy provides the retrieval system with aircraft-system anchors.")
    ax.barh(counts.index, counts.values, color=COLORS["green"], edgecolor=COLORS["deep_ink"], linewidth=0.45, zorder=3)
    ax.set_xlabel("Evidence objects", fontsize=14.6)
    ax.set_ylabel("")
    style_axis(ax)
    fig.subplots_adjust(left=0.30, right=0.96, top=0.94, bottom=0.14)
    return save_panel(fig, paths["out"] / f"{case_key}_panel_03_component_coverage", source, f"# {label}: Component evidence coverage\n\nCounts of evidence objects by component family.\n", {"sources": ["case evidence objects"]})


def panel_04_causal_relations(case_key: str, data: dict[str, pd.DataFrame]) -> Path:
    label = CASE_LABELS[case_key]
    paths = case_paths(case_key)
    causal = data["causal"].fillna("")
    counts = causal["relation"].map(lambda x: RELATION_LABELS.get(clean(x), clean(x).replace("_", " ").title())).value_counts().head(10).sort_values()
    source = counts.rename_axis("causal_relation").reset_index(name="kg_edges")
    fig, ax = plt.subplots(figsize=(11.8, max(5.2, 0.45 * len(counts) + 2.0)))
    figure_title(fig, f"{label}: Causal KG Relation Coverage", "The causal graph links symptoms, thresholds, conditions, warnings, causes and actions.")
    ax.barh(counts.index, counts.values, color=COLORS["blue"], edgecolor=COLORS["deep_ink"], linewidth=0.45, zorder=3)
    ax.set_xlabel("Causal KG edges", fontsize=14.6)
    ax.set_ylabel("")
    style_axis(ax)
    fig.subplots_adjust(left=0.30, right=0.96, top=0.94, bottom=0.14)
    return save_panel(fig, paths["out"] / f"{case_key}_panel_04_causal_relation_coverage", source, f"# {label}: Causal relation coverage\n\nCounts of extracted causal KG edges by relation type.\n", {"sources": ["case causal edges"]})


def panel_05_benchmark_tracks(case_key: str, data: dict[str, pd.DataFrame]) -> Path:
    label = CASE_LABELS[case_key]
    paths = case_paths(case_key)
    qa = data["qa"].fillna("")
    counts = qa["track"].replace({"unique_exact": "Unique-evidence exact retrieval", "broad_package": "Broad diagnostic package retrieval"}).value_counts()
    source = counts.rename_axis("benchmark_track").reset_index(name="questions")
    fig, ax = plt.subplots(figsize=(8.8, 6.4))
    figure_title(fig, f"{label}: Benchmark Track Composition", "Broad diagnostic queries are evaluated as package retrieval rather than single-ID exact retrieval.")
    colors = [COLORS["orange"], COLORS["cyan"], COLORS["purple"]][: len(counts)]
    wedges, texts, autotexts = ax.pie(counts.values, labels=[wrap_label(x, 22) for x in counts.index], autopct=lambda p: f"{p:.1f}%", startangle=90, colors=colors, wedgeprops=dict(edgecolor="white", linewidth=1.2), textprops=dict(fontsize=14.2, color=COLORS["ink"]))
    for t in autotexts:
        t.set_fontweight("bold")
        t.set_fontsize(16.0)
    ax.set_aspect("equal")
    fig.subplots_adjust(left=0.08, right=0.92, top=0.94, bottom=0.08)
    return save_panel(fig, paths["out"] / f"{case_key}_panel_05_benchmark_tracks", source, f"# {label}: Benchmark track composition\n\nQuestion counts by V21 evaluation track.\n", {"sources": ["case QA warehouse"]})


def panel_06_main_metrics(case_key: str, data: dict[str, pd.DataFrame]) -> Path:
    label = CASE_LABELS[case_key]
    paths = case_paths(case_key)
    metrics = method_subset(data["metrics_case"][data["metrics_case"]["aircraft_model"].eq(label)])
    metric_cols = ["exact_recall_10", "accepted_recall_10", "evidence_package_recall_10", "grounded_score"]
    source = metrics[["method", "method_label", *metric_cols]].copy()
    long = source.melt(id_vars=["method", "method_label"], value_vars=metric_cols, var_name="metric", value_name="score")
    fig, ax = plt.subplots(figsize=(13.8, 7.0))
    figure_title(fig, f"{label}: Main Retrieval and Grounding Metrics", "Each bar uses the V21 locked benchmark results; no values are manually edited.")
    x = np.arange(len(metric_cols))
    width = min(0.12, 0.78 / max(1, len(metrics)))
    for i, (_, row) in enumerate(metrics.iterrows()):
        vals = [row[m] for m in metric_cols]
        ax.bar(x + (i - (len(metrics) - 1) / 2) * width, vals, width=width, color=METHOD_COLORS.get(row["method"], "#999999"), edgecolor=COLORS["deep_ink"], linewidth=0.35, label=row["method_label"], zorder=3)
    ymin = max(0.0, float(np.nanmin(metrics[metric_cols].values)) - 0.08)
    if ymin > 0.86:
        ymin = 0.86
    ax.set_ylim(ymin, 1.04)
    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS[m] for m in metric_cols], fontsize=13.4)
    ax.set_ylabel("Score", fontsize=14.6)
    style_axis(ax, "y")
    ax.legend(ncol=4, fontsize=12.2, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18), columnspacing=0.9)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.94, bottom=0.27)
    return save_panel(fig, paths["out"] / f"{case_key}_panel_06_main_method_metrics", long, f"# {label}: Main method metrics\n\nGrouped-bar comparison across exact, accepted, package and grounded metrics.\n", {"sources": ["04_results/v21/v21_metrics_by_case.csv"]})


def panel_07_risk_by_method(case_key: str, data: dict[str, pd.DataFrame]) -> Path:
    label = CASE_LABELS[case_key]
    paths = case_paths(case_key)
    metrics = method_subset(data["metrics_case"][data["metrics_case"]["aircraft_model"].eq(label)])
    source = metrics[["method", "method_label", "hallucination_risk_proxy"]].copy()
    fig, ax = plt.subplots(figsize=(10.6, 6.6))
    figure_title(fig, f"{label}: Unsupported-Generation Risk Proxy by Method", "Lower values indicate more complete document-grounded evidence packages.")
    y = np.arange(len(metrics))
    ax.barh(y, metrics["hallucination_risk_proxy"], color=[METHOD_COLORS.get(m, "#999999") for m in metrics["method"]], edgecolor=COLORS["deep_ink"], linewidth=0.45, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(metrics["method_label"], fontsize=13.4)
    ax.invert_yaxis()
    ax.set_xlabel("Risk proxy", fontsize=14.6)
    ax.set_ylabel("")
    style_axis(ax)
    fig.subplots_adjust(left=0.30, right=0.96, top=0.94, bottom=0.16)
    return save_panel(fig, paths["out"] / f"{case_key}_panel_07_risk_by_method", source, f"# {label}: Risk proxy by method\n\nMean hallucination-risk proxy by method.\n", {"sources": ["04_results/v21/v21_metrics_by_case.csv"]})


def panel_08_track_heatmap(case_key: str, data: dict[str, pd.DataFrame]) -> Path:
    label = CASE_LABELS[case_key]
    paths = case_paths(case_key)
    track = method_subset(data["metrics_track"][data["metrics_track"]["aircraft_model"].eq(label)])
    rows = []
    for method, sub in track.groupby("method_label", sort=False):
        row = {"Method": method}
        for _, r in sub.iterrows():
            prefix = "Unique" if r["track"] == "unique_exact" else "Package"
            row[f"{prefix} accepted"] = r["accepted_recall_10"]
            row[f"{prefix} package"] = r["evidence_package_recall_10"]
        rows.append(row)
    mat = pd.DataFrame(rows).set_index("Method")
    source = mat.reset_index()
    fig, ax = plt.subplots(figsize=(11.8, 7.6))
    figure_title(fig, f"{label}: Track-Level Retrieval Heatmap", "Unique-evidence and broad-package tracks are reported separately.")
    im = ax.imshow(mat.values, aspect="auto", cmap="YlGnBu", vmin=max(0.70, np.nanmin(mat.values) - 0.05), vmax=1.0)
    ax.set_xticks(np.arange(mat.shape[1]))
    ax.set_xticklabels([wrap_label(c, 13) for c in mat.columns], rotation=28, ha="right", fontsize=12.4)
    ax.set_yticks(np.arange(mat.shape[0]))
    ax.set_yticklabels(mat.index, fontsize=12.4)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat.iloc[i, j]
            if not np.isnan(val):
                txt_color = COLORS["white"] if val >= 0.92 else COLORS["deep_ink"]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=13.8, color=txt_color, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.025)
    cbar.ax.tick_params(labelsize=12.4)
    fig.subplots_adjust(left=0.24, right=0.94, top=0.94, bottom=0.25)
    return save_panel(fig, paths["out"] / f"{case_key}_panel_08_track_level_heatmap", source, f"# {label}: Track-level retrieval heatmap\n\nAccepted and package recall split by evaluation track.\n", {"sources": ["04_results/v21/v21_metrics_by_case_track.csv"]})


def panel_09_query_family_grounded(case_key: str, data: dict[str, pd.DataFrame]) -> Path:
    label = CASE_LABELS[case_key]
    paths = case_paths(case_key)
    pred = method_subset(data["pred"][data["pred"]["aircraft_model"].eq(label)])
    final = pred[pred["method"].eq("FD-GRAG-RC-V21-final")]
    q = final.groupby("query_type")["grounded_score"].mean().sort_values().tail(12)
    source = q.rename_axis("query_family").reset_index(name="grounded_score")
    fig, ax = plt.subplots(figsize=(10.8, max(5.4, 0.48 * len(q) + 2.0)))
    figure_title(fig, f"{label}: Grounding by Query Family", "FD-GRAG-RC-V21 is evaluated separately across natural diagnostic query families.")
    ax.barh([natural_query_label(x) for x in q.index], q.values, color=COLORS["green"], edgecolor=COLORS["deep_ink"], linewidth=0.45, zorder=3)
    ax.set_xlim(max(0.0, min(q.values) - 0.08), 1.04)
    ax.set_xlabel("Grounded score", fontsize=14.6)
    ax.set_ylabel("")
    style_axis(ax)
    fig.subplots_adjust(left=0.32, right=0.96, top=0.94, bottom=0.15)
    return save_panel(fig, paths["out"] / f"{case_key}_panel_09_query_family_grounding", source, f"# {label}: Query-family grounding\n\nMean grounded score for FD-GRAG-RC-V21 by query family.\n", {"sources": ["04_results/v21/v21_retrieval_predictions_full.csv"]})


def panel_10_candidate_ceiling(case_key: str, data: dict[str, pd.DataFrame]) -> Path:
    label = CASE_LABELS[case_key]
    paths = case_paths(case_key)
    pool = method_subset(data["pool"][data["pool"]["aircraft_model"].eq(label)])
    final = pool[pool["method"].eq("FD-GRAG-RC-V21-final")]
    if final.empty:
        final = pool
    source = final.copy()
    fig, ax = plt.subplots(figsize=(10.6, 6.4))
    figure_title(fig, f"{label}: Candidate Pool Ceiling", "Accepted and exact pool recall are shown before top-10 ranking.")
    colors = {"unique_exact": COLORS["blue"], "broad_package": COLORS["orange"]}
    for track, sub in final.groupby("track"):
        agg = sub.groupby("k")[["exact_pool", "accepted_pool"]].mean().reset_index().sort_values("k")
        ax.plot(agg["k"], agg["accepted_pool"], marker="o", lw=2.4, color=colors.get(track, COLORS["green"]), label=f"{track.replace('_', ' ').title()} accepted")
        if agg["exact_pool"].notna().any():
            ax.plot(agg["k"], agg["exact_pool"], marker="s", lw=1.8, linestyle="--", color=colors.get(track, COLORS["green"]), alpha=0.75, label=f"{track.replace('_', ' ').title()} exact")
    ax.set_xscale("log")
    ax.set_ylim(0.72, 1.03)
    ax.set_xlabel("Candidate budget", fontsize=14.6)
    ax.set_ylabel("Pool recall", fontsize=14.6)
    style_axis(ax, "y")
    ax.legend(fontsize=12.4, frameon=False, loc="lower right")
    fig.subplots_adjust(left=0.10, right=0.96, top=0.94, bottom=0.16)
    return save_panel(fig, paths["out"] / f"{case_key}_panel_10_candidate_pool_ceiling", source, f"# {label}: Candidate pool ceiling\n\nCandidate-pool upper bound for accepted and exact evidence recall.\n", {"sources": ["04_results/v21/v21_candidate_pool_upper_bound.csv"]})


def panel_11_role_component_atlas(case_key: str, data: dict[str, pd.DataFrame]) -> Path:
    label = CASE_LABELS[case_key]
    paths = case_paths(case_key)
    ev = data["evidence"].fillna("")
    ev["component_family_label"] = ev["component_family"].map(lambda x: FAMILY_LABELS.get(clean(x), clean(x).replace("_", " ").title()))
    ev["role_label"] = ev["role"].map(lambda x: ROLE_LABELS.get(clean(x), clean(x).replace("_", " ").title()))
    table = pd.crosstab(ev["component_family_label"], ev["role_label"])
    families = table.sum(axis=1).sort_values(ascending=False).head(9).index
    roles = table.sum(axis=0).sort_values(ascending=False).head(9).index
    table = table.loc[families, roles]
    source = table.reset_index()
    fig, ax = plt.subplots(figsize=(12.8, 8.4))
    figure_title(fig, f"{label}: Role-Component Evidence Atlas", "A heatmap view of which diagnostic roles are available for each component family.")
    values = np.log1p(table.values)
    im = ax.imshow(values, cmap="YlGnBu", aspect="auto")
    max_val = float(np.nanmax(values)) if values.size else 1.0
    ax.set_xticks(np.arange(len(roles)))
    ax.set_xticklabels([wrap_label(r, 12) for r in roles], rotation=30, ha="right", fontsize=12.4)
    ax.set_yticks(np.arange(len(families)))
    ax.set_yticklabels([wrap_label(f, 18) for f in families], fontsize=12.0)
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            raw = int(table.iloc[i, j])
            if raw:
                txt_color = COLORS["white"] if (values[i, j] / max(max_val, 1e-9)) >= 0.58 else COLORS["deep_ink"]
                ax.text(j, i, str(raw), ha="center", va="center", fontsize=12.8, color=txt_color, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.030, pad=0.02)
    cbar.set_label("log(1 + count)", fontsize=12.4)
    cbar.ax.tick_params(labelsize=12.0)
    fig.subplots_adjust(left=0.27, right=0.94, top=0.94, bottom=0.26)
    return save_panel(fig, paths["out"] / f"{case_key}_panel_11_role_component_atlas", source, f"# {label}: Role-component evidence atlas\n\nRole-by-component evidence count heatmap.\n", {"sources": ["case evidence objects"]})


def panel_12_slot_complexity(case_key: str, data: dict[str, pd.DataFrame]) -> Path:
    label = CASE_LABELS[case_key]
    paths = case_paths(case_key)
    pred = method_subset(data["pred"][data["pred"]["aircraft_model"].eq(label)])
    final = pred[pred["method"].eq("FD-GRAG-RC-V21-final")]
    summary = final.groupby("n_required_slots")[["accepted_recall_10", "evidence_package_recall_10", "grounded_score"]].mean().reset_index().sort_values("n_required_slots")
    fig, ax = plt.subplots(figsize=(9.8, 6.0))
    figure_title(fig, f"{label}: Stability Across Evidence-Slot Complexity", "Package retrieval is evaluated as required evidence slots increase.")
    if not summary.empty:
        ax.plot(summary["n_required_slots"], summary["accepted_recall_10"], marker="o", lw=2.2, color=COLORS["blue"], label="Accepted R@10")
        ax.plot(summary["n_required_slots"], summary["evidence_package_recall_10"], marker="o", lw=2.2, color=COLORS["green"], label="Package R@10")
        ax.plot(summary["n_required_slots"], summary["grounded_score"], marker="o", lw=2.2, color=COLORS["red"], label="Grounded score")
    ax.set_ylim(0.78, 1.03)
    ax.set_xlabel("Required evidence slots", fontsize=14.6)
    ax.set_ylabel("Score", fontsize=14.6)
    style_axis(ax, "y")
    ax.legend(fontsize=12.4, frameon=False, loc="lower right")
    fig.subplots_adjust(left=0.12, right=0.96, top=0.94, bottom=0.16)
    return save_panel(fig, paths["out"] / f"{case_key}_panel_12_slot_complexity_stability", summary, f"# {label}: Evidence-slot complexity stability\n\nFD-GRAG-RC-V21 metrics grouped by required package-slot count.\n", {"sources": ["04_results/v21/v21_retrieval_predictions_full.csv"]})


def panel_13_safety_gate(case_key: str, data: dict[str, pd.DataFrame]) -> Path:
    label = CASE_LABELS[case_key]
    paths = case_paths(case_key)
    gate = data["safety"][data["safety"]["method"].eq("FD-GRAG-RC-V21-final")].sort_values("threshold")
    fig, ax = plt.subplots(figsize=(9.8, 6.0))
    figure_title(fig, f"{label}: Safety-Gated Generation Proxy", "The gate separates answer coverage from unsupported-generation risk.")
    if not gate.empty:
        ax.plot(gate["threshold"], gate["answer_coverage"], marker="o", lw=2.4, color=COLORS["green"], label="Answer coverage")
        ax.plot(gate["threshold"], gate["gated_unsupported_generation_risk_proxy"], marker="s", lw=2.4, color=COLORS["red"], label="Gated unsupported risk")
        ax.plot(gate["threshold"], gate["emitted_answer_risk_proxy"], marker="^", lw=1.8, color=COLORS["orange"], label="Emitted-answer risk")
    ax.set_ylim(-0.04, 1.04)
    ax.set_xlabel("Grounded-score gate threshold", fontsize=14.6)
    ax.set_ylabel("Rate", fontsize=14.6)
    style_axis(ax, "y")
    ax.legend(fontsize=12.4, frameon=False, loc="center right")
    fig.subplots_adjust(left=0.12, right=0.96, top=0.94, bottom=0.16)
    return save_panel(fig, paths["out"] / f"{case_key}_panel_13_safety_gate", gate, f"# {label}: Safety gate\n\nSystem-level proxy metrics under grounded-score refusal thresholds.\n", {"sources": ["04_results/v21/v21_safety_gate_metrics.csv"]})


def panel_14_grounding_risk_scatter(case_key: str, data: dict[str, pd.DataFrame]) -> Path:
    label = CASE_LABELS[case_key]
    paths = case_paths(case_key)
    pred = method_subset(data["pred"][data["pred"]["aircraft_model"].eq(label)])
    final = pred[pred["method"].eq("FD-GRAG-RC-V21-final")].dropna(subset=["grounded_score", "hallucination_risk_proxy"])
    source = final[["query_id", "track", "query_type", "grounded_score", "hallucination_risk_proxy", "n_required_slots"]].copy()
    fig, ax = plt.subplots(figsize=(9.4, 6.4))
    figure_title(fig, f"{label}: Grounding-Risk Coupling", "Each point is one FD-GRAG-RC-V21 query result from the locked benchmark.")
    if not final.empty:
        colors = final["track"].map({"unique_exact": COLORS["blue"], "broad_package": COLORS["orange"]}).fillna(COLORS["green"])
        sizes = 28 + 22 * final["n_required_slots"].fillna(1).clip(1, 6)
        ax.scatter(final["grounded_score"], final["hallucination_risk_proxy"], s=sizes, c=colors, alpha=0.68, edgecolor="white", linewidth=0.6)
    ax.set_xlim(0.70, 1.03)
    ax.set_ylim(-0.025, 0.32)
    ax.set_xlabel("Grounded score", fontsize=14.6)
    ax.set_ylabel("Risk proxy", fontsize=14.6)
    style_axis(ax, "both")
    ax.text(0.705, 0.285, "Lower risk\nupper-right is better", fontsize=12.4, color=COLORS["muted"])
    fig.subplots_adjust(left=0.12, right=0.96, top=0.94, bottom=0.16)
    return save_panel(fig, paths["out"] / f"{case_key}_panel_14_grounding_risk_scatter", source, f"# {label}: Grounding-risk coupling\n\nPer-query grounded score and risk proxy for FD-GRAG-RC-V21.\n", {"sources": ["04_results/v21/v21_retrieval_predictions_full.csv"]})


def panel_15_track_score_matrix(case_key: str, data: dict[str, pd.DataFrame]) -> Path:
    label = CASE_LABELS[case_key]
    paths = case_paths(case_key)
    score_table = data["track_scores"][data["track_scores"]["aircraft_model"].eq(label)].copy()
    score_table["track_label"] = score_table["track"].replace({"unique_exact": "Unique evidence", "broad_package": "Broad package"})
    score_table["track_label"] = score_table["track_label"].astype(str).str.replace(r"\s*-\s*$", "", regex=True)
    cols = [
        ("Exact\nR@10", "exact_recall_10", True),
        ("Accepted\nR@10", "accepted_recall_10", True),
        ("Package\nR@10", "evidence_package_recall_10", True),
        ("Grounded", "grounded_score", True),
        ("Gated\nrisk", "gated_unsupported_generation_risk_proxy", False),
    ]
    mat = []
    labels = []
    for _, r in score_table.iterrows():
        labels.append(r["track_label"])
        row = []
        for _, col, higher in cols:
            val = float(r[col]) if pd.notna(r[col]) else np.nan
            row.append(val if higher else 1 - val)
        mat.append(row)
    mat_arr = np.array(mat, dtype=float) if mat else np.zeros((0, len(cols)))
    source = score_table[["aircraft_model", "track_label", *[c for _, c, _ in cols]]].copy()
    fig, ax = plt.subplots(figsize=(9.2, 3.25))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    cmap = mpl.colormaps.get_cmap("YlGnBu")
    norm = mpl.colors.Normalize(vmin=0.90, vmax=1.0)
    x0, y0 = 0.30, 0.24
    cell_w, cell_h = 0.125, 0.22
    gap_x, gap_y = 0.018, 0.08
    for j, (name, _, _) in enumerate(cols):
        ax.text(x0 + j * (cell_w + gap_x) + cell_w / 2, 0.88, name, ha="center", va="center", fontsize=12.4, fontweight="bold", color=COLORS["deep_ink"], transform=ax.transAxes)
    for i, row_label in enumerate(labels):
        y = y0 + (len(labels) - 1 - i) * (cell_h + gap_y)
        ax.text(0.04, y + cell_h / 2, row_label, ha="left", va="center", fontsize=13.0, fontweight="bold", color=COLORS["deep_ink"], transform=ax.transAxes)
        for j, (_, col, higher) in enumerate(cols):
            x = x0 + j * (cell_w + gap_x)
            raw = source.iloc[i][col]
            score = mat_arr[i, j] if mat_arr.size else np.nan
            face = cmap(norm(score)) if pd.notna(score) else "#E6EAF0"
            ax.add_patch(
                FancyBboxPatch(
                    (x, y),
                    cell_w,
                    cell_h,
                    transform=ax.transAxes,
                    boxstyle="round,pad=0.008,rounding_size=0.018",
                    facecolor=face,
                    edgecolor=COLORS["white"],
                    linewidth=1.0,
                )
            )
            if pd.notna(raw):
                text = f"{raw:.2f}" if higher else f"risk {raw:.2f}"
                txt_color = COLORS["white"] if score >= 0.965 else COLORS["deep_ink"]
                ax.text(x + cell_w / 2, y + cell_h / 2, text, ha="center", va="center", fontsize=13.8, color=txt_color, fontweight="bold", transform=ax.transAxes)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.06)
    return save_panel(fig, paths["out"] / f"{case_key}_panel_15_track_score_matrix", source, f"# {label}: Track score matrix\n\nTrack-level retrieval, grounding, and gated risk results.\n", {"sources": ["04_results/v21/v21_target_attainment_matrix.csv"]})


def panel_16_grounded_distribution(case_key: str, data: dict[str, pd.DataFrame]) -> Path:
    label = CASE_LABELS[case_key]
    paths = case_paths(case_key)
    pred = method_subset(data["pred"][data["pred"]["aircraft_model"].eq(label)])
    final = pred[pred["method"].eq("FD-GRAG-RC-V21-final")].copy()
    final["track_label"] = final["track"].replace({"unique_exact": "Unique evidence", "broad_package": "Broad package"})
    groups = [(name, sub["grounded_score"].dropna().values) for name, sub in final.groupby("track_label")]
    source = final[["query_id", "track_label", "query_type", "grounded_score", "hallucination_risk_proxy", "n_required_slots"]].copy()
    fig, ax = plt.subplots(figsize=(9.8, 6.2))
    figure_title(fig, f"{label}: Query-Level Grounding Distribution", "The violin view checks whether high mean scores hide unstable individual queries.")
    positions = np.arange(1, len(groups) + 1)
    if groups:
        parts = ax.violinplot([vals if len(vals) else np.array([np.nan]) for _, vals in groups], positions=positions, widths=0.55, showmeans=True, showextrema=False)
        for body in parts["bodies"]:
            body.set_facecolor(COLORS["pale_purple"])
            body.set_edgecolor(COLORS["purple"])
            body.set_alpha(0.92)
        if "cmeans" in parts:
            parts["cmeans"].set_color(COLORS["purple"])
            parts["cmeans"].set_linewidth(2.0)
        rng = np.random.default_rng(20260516)
        for pos, (_, vals) in zip(positions, groups):
            if len(vals):
                jitter = rng.uniform(-0.08, 0.08, size=len(vals))
                ax.scatter(np.full(len(vals), pos) + jitter, vals, s=18, color=COLORS["blue"], alpha=0.42, edgecolor="white", linewidth=0.35, zorder=3)
    ax.set_xticks(positions)
    ax.set_xticklabels([name for name, _ in groups], fontsize=12.2)
    ax.set_ylim(0.72, 1.03)
    ax.set_ylabel("Grounded score", fontsize=14.6)
    style_axis(ax, "y")
    fig.subplots_adjust(left=0.12, right=0.96, top=0.94, bottom=0.16)
    return save_panel(fig, paths["out"] / f"{case_key}_panel_16_grounded_distribution", source, f"# {label}: Query-level grounding distribution\n\nPer-query grounded score distribution for FD-GRAG-RC-V21.\n", {"sources": ["04_results/v21/v21_retrieval_predictions_full.csv"]})


def panel_17_reliability_oncoprint(case_key: str, data: dict[str, pd.DataFrame]) -> Path:
    label = CASE_LABELS[case_key]
    paths = case_paths(case_key)
    pred = method_subset(data["pred"][data["pred"]["aircraft_model"].eq(label)])
    final = pred[pred["method"].eq("FD-GRAG-RC-V21-final")].copy()
    final["risk_rank"] = final["hallucination_risk_proxy"].fillna(0) + (1 - final["grounded_score"].fillna(1))
    sample = final.sort_values(["risk_rank", "n_required_slots"], ascending=[False, False]).head(24).reset_index(drop=True)
    checks = [
        ("Exact hit", lambda r: True if r["track"] != "unique_exact" else float(r.get("exact_recall_10", 0)) >= 1),
        ("Accepted hit", lambda r: float(r.get("accepted_recall_10", 0)) >= 1),
        ("Package complete", lambda r: float(r.get("evidence_package_recall_10", 0)) >= 1),
        ("Page hit", lambda r: float(r.get("accepted_page_recall_10", 0)) >= 1),
        ("Section hit", lambda r: float(r.get("accepted_section_recall_10", 0)) >= 1),
        ("Grounded >= 0.95", lambda r: float(r.get("grounded_score", 0)) >= 0.95),
        ("Risk <= 0.10", lambda r: float(r.get("hallucination_risk_proxy", 1)) <= 0.10),
    ]
    mat = np.zeros((len(sample), len(checks)))
    rows = []
    for i, (_, r) in enumerate(sample.iterrows()):
        row = {"query_id": r["query_id"], "query_family": r["query_type"], "track": r["track"]}
        for j, (name, fn) in enumerate(checks):
            ok = bool(fn(r))
            mat[i, j] = 1 if ok else 0
            row[name] = ok
        rows.append(row)
    source = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(12.2, max(8.2, 0.38 * max(1, len(sample)) + 2.8)))
    figure_title(fig, f"{label}: Reliability OncoPrint", "A query-by-property map shows whether failures concentrate in specific evidence-package checks.")
    cmap = mpl.colors.ListedColormap([COLORS["pale_red"], COLORS["pale_green"]])
    ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(checks)))
    ax.set_xticklabels([wrap_label(c[0], 13) for c in checks], rotation=28, ha="right", fontsize=12.4)
    row_labels = [f"Q{i+1:02d} · {compact_query_label(q)}" for i, q in enumerate(sample["query_type"].astype(str))]
    ax.set_yticks(np.arange(len(sample)))
    ax.set_yticklabels(row_labels, fontsize=12.2)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, "OK" if mat[i, j] else "Miss", ha="center", va="center", fontsize=12.8, color="#073B2B" if mat[i, j] else "#7B1E14", fontweight="bold")
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.subplots_adjust(left=0.24, right=0.96, top=0.94, bottom=0.24)
    return save_panel(fig, paths["out"] / f"{case_key}_panel_17_reliability_oncoprint", source, f"# {label}: Reliability OncoPrint\n\nQuery-level evidence-package reliability checks for FD-GRAG-RC-V21.\n", {"sources": ["04_results/v21/v21_retrieval_predictions_full.csv"]})


def panel_18_package_recovery_flow(case_key: str, data: dict[str, pd.DataFrame]) -> Path:
    label = CASE_LABELS[case_key]
    paths = case_paths(case_key)
    metrics = data["metrics_case"][(data["metrics_case"]["aircraft_model"].eq(label)) & (data["metrics_case"]["method"].eq("FD-GRAG-RC-V21-final"))]
    pool = data["pool"][(data["pool"]["aircraft_model"].eq(label)) & (data["pool"]["method"].eq("FD-GRAG-RC-V21-final")) & (data["pool"]["k"].eq(1000))]
    pool_value = float(pool["accepted_pool"].mean()) if not pool.empty else np.nan
    m = metrics.iloc[0] if not metrics.empty else pd.Series(dtype=float)
    stages = [
        ("Candidate\npool", pool_value),
        ("Top-10\naccepted", float(m.get("accepted_recall_10", np.nan))),
        ("Package\ncomplete", float(m.get("evidence_package_recall_10", np.nan))),
        ("Grounded\nevidence", float(m.get("grounded_score", np.nan))),
        ("Supported\noutput", 1 - float(m.get("hallucination_risk_proxy", np.nan))),
    ]
    source = pd.DataFrame({"stage": [s for s, _ in stages], "score": [v for _, v in stages]})
    fig, ax = plt.subplots(figsize=(12.0, 6.2))
    figure_title(fig, f"{label}: Evidence-Package Recovery Flow", "The flow summarizes the conversion from candidate evidence to grounded supported output.")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    xs = np.linspace(0.08, 0.92, len(stages))
    baseline = 0.18
    scale = 0.62
    for i, (stage, val) in enumerate(stages):
        val = 0 if np.isnan(val) else max(0, min(1, val))
        h = 0.10 + scale * val
        y0 = baseline
        fc = [COLORS["pale_blue"], COLORS["pale_green"], COLORS["pale_orange"], COLORS["pale_purple"], COLORS["pale_green"]][i]
        ec = [COLORS["blue"], COLORS["green"], COLORS["orange"], COLORS["purple"], COLORS["green"]][i]
        ax.add_patch(FancyBboxPatch((xs[i] - 0.065, y0), 0.13, h, transform=ax.transAxes, boxstyle="round,pad=0.014,rounding_size=0.025", facecolor=fc, edgecolor=ec, linewidth=1.5, zorder=3))
        ax.text(xs[i], y0 + h + 0.035, stage, ha="center", va="bottom", fontsize=12.2, color=COLORS["ink"], fontweight="bold", transform=ax.transAxes)
        ax.text(xs[i], y0 + h / 2, f"{val:.2f}", ha="center", va="center", fontsize=17.4, color=COLORS["ink"], fontweight="bold", transform=ax.transAxes)
        if i < len(stages) - 1:
            next_val = 0 if np.isnan(stages[i + 1][1]) else max(0, min(1, stages[i + 1][1]))
            y_mid = y0 + 0.10 + scale * min(val, next_val) * 0.52
            verts = [
                (xs[i] + 0.065, y_mid + 0.035),
                (xs[i] + 0.11, y_mid + 0.05),
                (xs[i + 1] - 0.11, y_mid + 0.05),
                (xs[i + 1] - 0.065, y_mid + 0.035),
                (xs[i + 1] - 0.065, y_mid - 0.035),
                (xs[i + 1] - 0.11, y_mid - 0.05),
                (xs[i] + 0.11, y_mid - 0.05),
                (xs[i] + 0.065, y_mid - 0.035),
                (xs[i] + 0.065, y_mid + 0.035),
            ]
            codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4, MplPath.LINETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4, MplPath.CLOSEPOLY]
            ax.add_patch(PathPatch(MplPath(verts, codes), transform=ax.transAxes, facecolor=COLORS["cyan"], edgecolor="none", alpha=0.22, zorder=1))
    ax.text(0.08, 0.08, "Scores are case-level V21 locked benchmark results; this flow is a visualization, not a re-scoring step.", transform=ax.transAxes, fontsize=12.0, color=COLORS["muted"], bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.84))
    fig.subplots_adjust(left=0.05, right=0.97, top=0.94, bottom=0.08)
    return save_panel(fig, paths["out"] / f"{case_key}_panel_18_package_recovery_flow", source, f"# {label}: Evidence-package recovery flow\n\nStage-wise V21 evidence package recovery and supported-output visualization.\n", {"sources": ["04_results/v21/v21_metrics_by_case.csv", "04_results/v21/v21_candidate_pool_upper_bound.csv"]})


PANELS: list[Callable[[str, dict[str, pd.DataFrame]], Path]] = [
    panel_01_pipeline,
    panel_02_roles,
    panel_03_components,
    panel_04_causal_relations,
    panel_05_benchmark_tracks,
    panel_06_main_metrics,
    panel_07_risk_by_method,
    panel_08_track_heatmap,
    panel_09_query_family_grounded,
    panel_10_candidate_ceiling,
    panel_11_role_component_atlas,
    panel_12_slot_complexity,
    panel_13_safety_gate,
    panel_14_grounding_risk_scatter,
    panel_15_track_score_matrix,
    panel_16_grounded_distribution,
    panel_17_reliability_oncoprint,
    panel_18_package_recovery_flow,
]


def clean_output_folder(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for child in out_dir.iterdir():
        if child.is_file() and child.suffix.lower() in {".png", ".svg", ".pdf", ".csv", ".md", ".json"}:
            child.unlink()


def copy_code(case_key: str):
    paths = case_paths(case_key)
    paths["code"].mkdir(parents=True, exist_ok=True)
    source = Path(__file__).resolve()
    shutil.copy2(source, paths["code"] / "build_reference_style_single_panels.py")
    wrapper = paths["code"] / "build_reference_style_case_figures.py"
    wrapper.write_text(
        "from build_reference_style_single_panels import main\n\n"
        "if __name__ == '__main__':\n"
        "    main()\n",
        encoding="utf-8",
    )
    (paths["code"] / "README.md").write_text(
        "# Single-Panel Reference-Style Figure Code\n\n"
        "Run from the V21 package root:\n\n"
        "```powershell\n"
        "& '${USER_HOME}/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' "
        f"'10_scripts/build_reference_style_single_panels.py' --case {case_key}\n"
        "```\n\n"
        "Only PNG figures are generated. Each PNG has source data, metadata, and a short caption.\n",
        encoding="utf-8",
    )


def write_case_readme(case_key: str, pngs: list[Path]):
    paths = case_paths(case_key)
    lines = [
        f"# {CASE_LABELS[case_key]} Single-Panel Reference-Style Figures",
        "",
        "All figures in this directory are independent PNG panels generated from true V21 case data and V21 locked benchmark results.",
        "",
        "No PDF or SVG exports are produced for this single-panel version.",
        "",
        "## Panels",
        "",
    ]
    for p in pngs:
        lines.append(f"- `{p.name}`")
    lines += [
        "",
        "Each panel is accompanied by `_source_data.csv`, `_metadata.json`, and `_caption.md`.",
        "",
        "## Code",
        "",
        "`../reference_style_figure_code/build_reference_style_single_panels.py`",
    ]
    (paths["out"] / "README.md").write_text("\n".join(lines), encoding="utf-8")


def build_case(case_key: str) -> list[Path]:
    paths = case_paths(case_key)
    clean_output_folder(paths["out"])
    data = get_data(case_key)
    pngs = [panel(case_key, data) for panel in PANELS]
    copy_code(case_key)
    write_case_readme(case_key, pngs)
    return pngs


def write_report(outputs: dict[str, list[Path]]):
    report = ROOT / "07_reports" / "v21" / "v21_reference_style_single_panel_figures_report.md"
    lines = [
        "# V21 Reference-Style Single-Panel Figures Report",
        "",
        "Generated on 2026-05-15.",
        "",
        "The former composite reference-style figures were replaced by independent PNG panels. All values are read from case-local V21 artifacts or `04_results/v21` locked benchmark files. No PDF or SVG files are generated by this single-panel script.",
        "",
        "The updated figure style follows the cited Nature Communications article at a layout level: high-contrast analytical backgrounds, heatmaps, distribution views, OncoPrint-style reliability maps, and stage-flow visualizations. The panels are still independent single figures rather than composite multi-panel plates.",
        "",
    ]
    for case_key, pngs in outputs.items():
        lines.append(f"## {CASE_LABELS[case_key]}")
        lines.append("")
        for p in pngs:
            lines.append(f"- `Cases/{case_key}/7 */reference_style_figures/{p.name}`")
        lines.append("")
    lines += [
        "## Claim Boundary",
        "",
        "These figures report document-grounded retrieval and generation-readiness proxy metrics. They do not claim real maintenance correctness, expert validation, field diagnostic accuracy, or safety certification.",
    ]
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")


def selected_cases(raw: str | None) -> Iterable[str]:
    if not raw or raw.lower() in {"all", "*"}:
        return CASE_LABELS.keys()
    if raw in CASE_LABELS:
        return [raw]
    normalized = raw.lower().replace(" ", "_")
    aliases = {v.lower().replace(" ", "_"): k for k, v in CASE_LABELS.items()}
    if normalized in aliases:
        return [aliases[normalized]]
    raise ValueError(f"Unknown case {raw!r}")


def main():
    parser = argparse.ArgumentParser(description="Build single-panel Nature-style figures for each case.")
    parser.add_argument("--case", default="all", help="Boeing_737, Cessna_172, or all")
    args = parser.parse_args()
    outputs = {}
    for case_key in selected_cases(args.case):
        outputs[case_key] = build_case(case_key)
    write_report(outputs)
    print("Generated single-panel PNG figures:")
    for case_key, pngs in outputs.items():
        print(CASE_LABELS[case_key])
        for p in pngs:
            print(" ", p)


if __name__ == "__main__":
    main()
