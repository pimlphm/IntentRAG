from __future__ import annotations

import json
import math
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch


SEED = 20260517
METHOD_OURS = "FD-GRAG-RC-V21-final"
METHOD_OURS_DISPLAY = "FD-GRAG-RC-V21"


ROLE_COLUMNS = [
    "symptom",
    "confirmed_cause",
    "excluded_cause",
    "action",
    "inspection",
    "replacement",
    "warning",
    "limitation",
    "threshold",
    "condition",
    "authority",
    "note",
]

COMPONENT_LABELS = {
    "engine_oil_powerplant": "Engine / powerplant",
    "landing_gear_brake": "Landing gear / brake",
    "flight_control": "Flight controls",
    "electrical": "Electrical",
    "navigation_avionics": "Navigation / avionics",
    "fuel": "Fuel system",
    "air_conditioning_pneumatic": "Pneumatic / air conditioning",
    "hydraulic": "Hydraulic",
    "other_component": "Other components",
    "no_component": "Unspecified component",
}

COMPACT_COMPONENT_LABELS = {
    "Engine / powerplant": "Engine",
    "Landing gear / brake": "Gear / brake",
    "Flight controls": "Flight controls",
    "Electrical": "Electrical",
    "Navigation / avionics": "Nav / avionics",
    "Fuel system": "Fuel",
    "Pneumatic / air conditioning": "Pneumatic / AC",
    "Hydraulic": "Hydraulic",
    "Other components": "Other",
    "Unspecified component": "Unspecified",
}

COMPONENT_CRITICALITY_PRIOR = {
    "engine_oil_powerplant": 2.0,
    "flight_control": 1.8,
    "fuel": 1.7,
    "landing_gear_brake": 1.5,
    "electrical": 1.4,
    "navigation_avionics": 1.2,
    "hydraulic": 1.2,
    "air_conditioning_pneumatic": 1.0,
    "other_component": 0.7,
    "no_component": 0.2,
}

QUERY_FAILURE_MODE = {
    "component_to_failure": "Component-to-failure ambiguity",
    "cross_page_evidence_query": "Cross-page evidence continuity",
    "warning_or_limitation": "Warning / limitation support",
    "threshold_query": "Threshold or limit support",
    "authority_or_manual_reference": "Authority citation support",
    "excluded_cause": "Excluded-cause discrimination",
    "multi_hop_diagnosis": "Multi-hop diagnosis chain",
    "symptom_to_cause": "Symptom-to-cause trace",
    "symptom_to_action": "Symptom-to-action trace",
    "condition_to_action": "Condition-to-action trace",
}


@dataclass
class CasePaths:
    case: str
    root: Path
    case_root: Path
    step1: Path
    step2: Path
    step3: Path
    test: Path
    latest: Path
    tables: Path
    figures: Path
    code: Path


def find_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "Cases").exists() and (parent / "04_results").exists():
            return parent
    raise FileNotFoundError("Cannot locate the FD-GRAG-RC V21 project root from the script path.")


def numbered_dir(case_root: Path, prefix: str) -> Path:
    matches = [p for p in case_root.iterdir() if p.is_dir() and p.name.startswith(prefix)]
    if not matches:
        raise FileNotFoundError(f"Cannot locate folder prefix {prefix} under {case_root}")
    return matches[0]


def build_paths(root: Path, case: str) -> CasePaths:
    case_root = root / "Cases" / case
    step1 = numbered_dir(case_root, "1 ")
    step2 = numbered_dir(case_root, "2 ")
    step3 = numbered_dir(case_root, "3 ")
    test = numbered_dir(case_root, "7 ")
    latest = numbered_dir(case_root, "8 ")
    tables = test / "fmeca_analysis_tables"
    figures = latest / "07_fmeca_analysis"
    code = case_root / "fmeca_analysis_code"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    code.mkdir(parents=True, exist_ok=True)
    return CasePaths(case, root, case_root, step1, step2, step3, test, latest, tables, figures, code)


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False)


def first_match(root: Path, pattern: str) -> Path:
    matches = list(root.rglob(pattern))
    if not matches:
        raise FileNotFoundError(f"No file matched {pattern} under {root}")
    return matches[0]


def safe_float(x, default=0.0) -> float:
    try:
        if x == "" or pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def split_terms(value: str) -> list[str]:
    text = str(value or "")
    if not text:
        return []
    parts = re.split(r"[;,\|/]+", text)
    return [p.strip().lower() for p in parts if p.strip()]


def component_label(family: str) -> str:
    return COMPONENT_LABELS.get(str(family), str(family).replace("_", " ").title())


def compact_component_label(label: str) -> str:
    return COMPACT_COMPONENT_LABELS.get(label, label)


def compact_case_name(case: str) -> str:
    return "Boeing 737" if case == "Boeing_737" else "Cessna 172"


def quantile_score(values: pd.Series, value: float, reverse: bool = False) -> int:
    arr = pd.to_numeric(values, errors="coerce").fillna(0).to_numpy()
    if len(arr) <= 1 or np.nanmax(arr) == np.nanmin(arr):
        score = 5
    else:
        pct = (arr <= value).mean()
        score = int(np.clip(math.ceil(pct * 10), 1, 10))
    return 11 - score if reverse else score


def style_publication() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 15,
            "axes.labelsize": 17,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "axes.linewidth": 1.7,
            "axes.facecolor": "#fffdf7",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "legend.frameon": False,
            "axes.grid": False,
        }
    )


def add_axis_arrows(ax: plt.Axes, xpad=0.02, ypad=0.02) -> None:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    arrow_kw = dict(
        arrowstyle="-|>",
        mutation_scale=25,
        linewidth=1.9,
        color="#111111",
        clip_on=False,
        zorder=10,
    )
    ax.add_patch(FancyArrowPatch((xmin, ymin), (xmax + (xmax - xmin) * xpad, ymin), **arrow_kw))
    ax.add_patch(FancyArrowPatch((xmin, ymin), (xmin, ymax + (ymax - ymin) * ypad), **arrow_kw))
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def save_md_table(df: pd.DataFrame, path: Path, max_rows: int | None = None) -> None:
    show = df if max_rows is None else df.head(max_rows)
    path.write_text(markdown_table(show), encoding="utf-8")


def markdown_table(df: pd.DataFrame, max_col_width: int = 80) -> str:
    if df.empty:
        return "_No rows available._\n"
    cols = list(df.columns)

    def fmt(x) -> str:
        if isinstance(x, float):
            text = f"{x:.4f}".rstrip("0").rstrip(".")
        else:
            text = str(x)
        text = text.replace("\n", " ").replace("|", "/")
        if len(text) > max_col_width:
            text = text[: max_col_width - 1] + "…"
        return text

    rows = [[fmt(v) for v in row] for row in df[cols].itertuples(index=False, name=None)]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, sep, *body]) + "\n"


def save_caption(path: Path, text: str) -> None:
    path.write_text(text.strip() + "\n", encoding="utf-8")


def role_counts(ev: pd.DataFrame) -> pd.DataFrame:
    base = ev.copy()
    base["component_family"] = base["component_family"].replace("", "no_component").fillna("no_component")
    pivot = pd.crosstab(base["component_family"], base["role"])
    for col in ROLE_COLUMNS:
        if col not in pivot.columns:
            pivot[col] = 0
    return pivot[ROLE_COLUMNS].reset_index()


def fd_metrics_by_component(chains: pd.DataFrame, ev: pd.DataFrame) -> pd.DataFrame:
    ours = chains[chains["method"].astype(str).eq(METHOD_OURS)].copy()
    if ours.empty:
        ours = chains[chains["method_display"].astype(str).eq(METHOD_OURS_DISPLAY)].copy()
    ev_map = ev.set_index("evidence_id")["component_family"].to_dict()
    rows = []
    for _, row in ours.iterrows():
        ids = split_terms(row.get("top_evidence_ids", ""))
        families = []
        for evid in ids[:10]:
            fam = ev_map.get(evid)
            if fam:
                families.append(fam)
        if not families:
            top_components = split_terms(row.get("top_evidence_components", ""))
            families = top_components or ["no_component"]
        for fam in sorted(set(families)):
            rows.append(
                {
                    "component_family": fam,
                    "grounded_score": safe_float(row.get("grounded_score")),
                    "accepted_recall_10": safe_float(row.get("accepted_recall_10")),
                    "package_recall_10": safe_float(row.get("evidence_package_recall_10")),
                    "hallucination_risk_proxy": safe_float(row.get("hallucination_risk_proxy")),
                    "query_type": row.get("query_type", ""),
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["component_family", "grounded_score", "accepted_recall_10", "package_recall_10", "hallucination_risk_proxy", "n_queries"])
    return (
        df.groupby("component_family", as_index=False)
        .agg(
            grounded_score=("grounded_score", "mean"),
            accepted_recall_10=("accepted_recall_10", "mean"),
            package_recall_10=("package_recall_10", "mean"),
            hallucination_risk_proxy=("hallucination_risk_proxy", "mean"),
            n_queries=("query_type", "count"),
        )
        .sort_values("n_queries", ascending=False)
    )


def build_component_fmeca(paths: CasePaths, ev: pd.DataFrame, edges: pd.DataFrame, chains: pd.DataFrame) -> pd.DataFrame:
    rc = role_counts(ev)
    counts = ev.groupby("component_family", as_index=False).agg(
        evidence_count=("evidence_id", "count"),
        page_count=("page_id", pd.Series.nunique),
        section_count=("detected_section_id", pd.Series.nunique),
        warning_text_count=("warning_terms", lambda s: (s.astype(str).str.len() > 0).sum()),
        threshold_text_count=("threshold_terms", lambda s: (s.astype(str).str.len() > 0).sum()),
        authority_text_count=("authority_terms", lambda s: (s.astype(str).str.len() > 0).sum()),
    )
    counts["component_family"] = counts["component_family"].replace("", "no_component").fillna("no_component")
    if "component" in edges.columns:
        edge_tmp = edges.copy()
        edge_tmp["component_family"] = edge_tmp["component"].replace("", "no_component").fillna("no_component")
        # Map raw component strings onto known families when they contain the family token.
        known = sorted(counts["component_family"].dropna().unique(), key=len, reverse=True)
        def map_edge_component(x: str) -> str:
            x = str(x).lower()
            for fam in known:
                if fam and fam != "no_component" and fam.lower() in x:
                    return fam
            for fam in known:
                if fam and fam != "no_component" and any(tok in x for tok in fam.lower().split("_")):
                    return fam
            return "no_component"

        edge_tmp["component_family"] = edge_tmp["component_family"].map(map_edge_component)
        edge_counts = edge_tmp.groupby("component_family", as_index=False).agg(
            causal_edge_count=("edge_id", "count"),
            causal_confidence=("confidence", lambda s: pd.to_numeric(s, errors="coerce").fillna(0).mean()),
        )
    else:
        edge_counts = pd.DataFrame({"component_family": [], "causal_edge_count": [], "causal_confidence": []})
    metrics = fd_metrics_by_component(chains, ev)
    df = counts.merge(rc, on="component_family", how="left").merge(edge_counts, on="component_family", how="left").merge(metrics, on="component_family", how="left")
    df = df.fillna(0)
    if df.empty:
        return df
    df["component_label"] = df["component_family"].map(component_label)

    raw_severity = (
        3.0
        + 1.45 * (np.log1p(df["warning"] + df["limitation"] + df["warning_text_count"]) / np.log(1 + max(1, (df["warning"] + df["limitation"] + df["warning_text_count"]).max())))
        + 1.15 * (np.log1p(df["threshold"] + df["threshold_text_count"]) / np.log(1 + max(1, (df["threshold"] + df["threshold_text_count"]).max())))
        + 1.20 * (np.log1p(df["confirmed_cause"] + df["symptom"]) / np.log(1 + max(1, (df["confirmed_cause"] + df["symptom"]).max())))
        + 1.00 * (np.log1p(df["action"] + df["inspection"] + df["replacement"]) / np.log(1 + max(1, (df["action"] + df["inspection"] + df["replacement"]).max())))
        + df["component_family"].map(COMPONENT_CRITICALITY_PRIOR).fillna(0.7)
    )
    df["severity"] = np.clip(np.rint(raw_severity), 1, 10).astype(int)

    exposure = (
        np.log1p(df["evidence_count"])
        + 0.7 * np.log1p(df["page_count"])
        + 0.5 * np.log1p(df["causal_edge_count"])
        + 0.6 * np.log1p(df["n_queries"])
    )
    df["occurrence_proxy_raw"] = exposure
    df["occurrence"] = [quantile_score(pd.Series(exposure), v) for v in exposure]

    # Detection rating follows FMECA convention: high score means harder to detect or control.
    quality = (
        0.45 * pd.to_numeric(df["grounded_score"], errors="coerce").fillna(0.0)
        + 0.25 * pd.to_numeric(df["accepted_recall_10"], errors="coerce").fillna(0.0)
        + 0.20 * pd.to_numeric(df["package_recall_10"], errors="coerce").fillna(0.0)
        + 0.10 * (1.0 - pd.to_numeric(df["hallucination_risk_proxy"], errors="coerce").fillna(1.0))
    )
    if quality.max() <= 0:
        quality = pd.Series(np.full(len(df), 0.5), index=df.index)
    detection = np.clip(np.rint(10 - 9 * quality), 1, 10)
    df["detection_difficulty"] = detection.astype(int)
    df["rpn"] = df["severity"] * df["occurrence"] * df["detection_difficulty"]
    df["criticality_index"] = np.round((df["rpn"] / 1000.0) * 100, 1)
    df["criticality_class"] = pd.cut(
        df["rpn"],
        bins=[-1, 100, 200, 350, 1000],
        labels=["Low", "Moderate", "High", "Critical"],
    ).astype(str)
    df["component_function_proxy"] = df["component_family"].map(component_function_proxy)
    df["dominant_failure_mode"] = df.apply(infer_failure_mode, axis=1)
    df["local_effect_proxy"] = df.apply(infer_local_effect, axis=1)
    df["higher_level_effect_proxy"] = df.apply(infer_higher_level_effect, axis=1)
    df["probable_cause_proxy"] = df.apply(infer_probable_cause, axis=1)
    df["evidence_controls"] = df.apply(infer_controls, axis=1)
    df["recommended_action_proxy"] = df.apply(infer_recommended_action, axis=1)
    df["inherent_criticality_proxy"] = df["severity"] * df["occurrence"] * 10
    df["residual_control_factor"] = np.round(df["detection_difficulty"] / 10.0, 2)
    df["fmeca_scope_note"] = "Document-grounded proxy FMECA; not an expert-certified aircraft safety assessment."
    order = [
        "aircraft_model",
        "component_family",
        "component_label",
        "component_function_proxy",
        "dominant_failure_mode",
        "local_effect_proxy",
        "higher_level_effect_proxy",
        "probable_cause_proxy",
        "severity",
        "occurrence",
        "detection_difficulty",
        "inherent_criticality_proxy",
        "residual_control_factor",
        "rpn",
        "criticality_index",
        "criticality_class",
        "evidence_count",
        "page_count",
        "section_count",
        "causal_edge_count",
        "warning",
        "limitation",
        "threshold",
        "condition",
        "confirmed_cause",
        "excluded_cause",
        "action",
        "inspection",
        "replacement",
        "authority",
        "grounded_score",
        "accepted_recall_10",
        "package_recall_10",
        "hallucination_risk_proxy",
        "evidence_controls",
        "recommended_action_proxy",
        "fmeca_scope_note",
    ]
    df["aircraft_model"] = compact_case_name(paths.case)
    for col in order:
        if col not in df.columns:
            df[col] = 0
    return df[order].sort_values(["rpn", "severity", "occurrence"], ascending=False)


def component_function_proxy(family: str) -> str:
    mapping = {
        "engine_oil_powerplant": "Provide thrust, lubrication, temperature/pressure control, and engine operating continuity.",
        "landing_gear_brake": "Support landing, taxi, braking, ground handling, and tire/wheel integrity.",
        "flight_control": "Provide aerodynamic control, trim, flap, aileron, rudder, and stabilizer response.",
        "fuel": "Store, select, filter, and deliver fuel to the engine under the required operating condition.",
        "electrical": "Provide generation, battery, bus, switching, starting, and electrical distribution support.",
        "navigation_avionics": "Provide indication, navigation, communication, pitot-static, and flight instrument support.",
        "hydraulic": "Transmit hydraulic power or damping where the source document identifies hydraulic context.",
        "air_conditioning_pneumatic": "Provide pneumatic, pressurization, air-conditioning, or bleed-air related support.",
        "other_component": "Support miscellaneous aircraft functions not mapped to a primary component family.",
        "no_component": "Evidence lacks an explicit component function and requires stronger source-context control.",
    }
    return mapping.get(str(family), "Component function inferred from document-local evidence.")


def infer_local_effect(row: pd.Series) -> str:
    mode = str(row.get("dominant_failure_mode", ""))
    if "warning" in mode.lower() or "limitation" in mode.lower():
        return "Safety-critical warning or limitation may be omitted from the retrieved evidence package."
    if "threshold" in mode.lower() or "condition" in mode.lower():
        return "Numerical limit, condition, or applicability cue may be separated from the action evidence."
    if "cause" in mode.lower():
        return "Symptom, confirmed cause, and excluded cause may be conflated during diagnosis support."
    if "component anchor" in mode.lower():
        return "Evidence may be retrieved without a stable component anchor."
    return "Maintenance action or inspection evidence may be retrieved without enough component context."


def infer_higher_level_effect(row: pd.Series) -> str:
    family = str(row.get("component_family", ""))
    if family in {"engine_oil_powerplant", "fuel"}:
        return "Potential degradation of propulsion-related troubleshooting support and answer grounding."
    if family == "flight_control":
        return "Potential degradation of control-surface or configuration-related maintenance support."
    if family == "landing_gear_brake":
        return "Potential degradation of landing, braking, or ground-handling evidence support."
    if family == "electrical":
        return "Potential degradation of electrical fault isolation and source-authority traceability."
    if family == "navigation_avionics":
        return "Potential degradation of indication/navigation evidence support and citation precision."
    return "Potential degradation of document-grounded answer completeness or citation traceability."


def infer_probable_cause(row: pd.Series) -> str:
    counts = {
        "warning/limitation density": safe_float(row.get("warning")) + safe_float(row.get("limitation")),
        "threshold-condition density": safe_float(row.get("threshold")) + safe_float(row.get("condition")),
        "cause/symptom density": safe_float(row.get("confirmed_cause")) + safe_float(row.get("symptom")) + safe_float(row.get("excluded_cause")),
        "action/inspection density": safe_float(row.get("action")) + safe_float(row.get("inspection")) + safe_float(row.get("replacement")),
        "authority density": safe_float(row.get("authority")),
    }
    key = max(counts, key=counts.get)
    if counts[key] <= 0:
        return "Sparse role evidence in the current document representation."
    return f"High {key} in the source evidence."


def infer_recommended_action(row: pd.Series) -> str:
    mode = str(row.get("dominant_failure_mode", ""))
    controls = []
    if "warning" in mode.lower() or safe_float(row.get("warning")) + safe_float(row.get("limitation")) > 0:
        controls.append("require warning/limitation slot")
    if "threshold" in mode.lower() or safe_float(row.get("threshold")) > 0:
        controls.append("require threshold-condition joint evidence")
    if safe_float(row.get("authority")) > 0:
        controls.append("require manual/page/section citation")
    if safe_float(row.get("confirmed_cause")) + safe_float(row.get("symptom")) > 0:
        controls.append("preserve symptom-cause-action chain")
    if safe_float(row.get("detection_difficulty")) >= 4:
        controls.append("tighten refusal gate for incomplete packages")
    if not controls:
        controls.append("retain source-page traceability and component profile coverage")
    return "; ".join(controls[:4])


def infer_failure_mode(row: pd.Series) -> str:
    family = row.get("component_family", "")
    warning_load = safe_float(row.get("warning")) + safe_float(row.get("limitation"))
    threshold_load = safe_float(row.get("threshold"))
    cause_load = safe_float(row.get("confirmed_cause")) + safe_float(row.get("symptom"))
    action_load = safe_float(row.get("action")) + safe_float(row.get("inspection")) + safe_float(row.get("replacement"))
    if warning_load >= max(threshold_load, cause_load, action_load):
        return "Safety limitation or warning not carried into the evidence package"
    if threshold_load >= max(cause_load, action_load):
        return "Threshold or operating condition missed or detached from context"
    if cause_load >= action_load:
        return "Fault indication, cause, or excluded-cause evidence confused"
    if family == "no_component":
        return "Source evidence lacks an explicit component anchor"
    return "Corrective action or inspection step not linked to the component context"


def infer_controls(row: pd.Series) -> str:
    controls = []
    if safe_float(row.get("warning")) + safe_float(row.get("limitation")) > 0:
        controls.append("safety/limitation evidence slot")
    if safe_float(row.get("threshold")) > 0:
        controls.append("threshold-condition joint slot")
    if safe_float(row.get("authority")) > 0:
        controls.append("manual/page/section citation slot")
    if safe_float(row.get("confirmed_cause")) + safe_float(row.get("symptom")) > 0:
        controls.append("symptom-cause chain slot")
    if safe_float(row.get("action")) + safe_float(row.get("inspection")) + safe_float(row.get("replacement")) > 0:
        controls.append("action/inspection slot")
    if not controls:
        controls.append("general source traceability")
    return "; ".join(controls[:4])


def build_query_fmeca(paths: CasePaths, chains: pd.DataFrame) -> pd.DataFrame:
    ours = chains[chains["method"].astype(str).eq(METHOD_OURS)].copy()
    if ours.empty:
        ours = chains[chains["method_display"].astype(str).eq(METHOD_OURS_DISPLAY)].copy()
    if ours.empty:
        return pd.DataFrame()
    agg = (
        ours.groupby(["aircraft_model", "track", "query_type"], as_index=False)
        .agg(
            n_queries=("query_id", "nunique"),
            exact_recall_10=("exact_recall_10", "mean"),
            accepted_recall_10=("accepted_recall_10", "mean"),
            evidence_package_recall_10=("evidence_package_recall_10", "mean"),
            accepted_page_recall_10=("accepted_page_recall_10", "mean"),
            accepted_section_recall_10=("accepted_section_recall_10", "mean"),
            grounded_score=("grounded_score", "mean"),
            hallucination_risk_proxy=("hallucination_risk_proxy", "mean"),
            role_coverage_proxy=("role_coverage_proxy", "mean"),
        )
    )
    agg["failure_mode"] = agg["query_type"].map(QUERY_FAILURE_MODE).fillna(agg["query_type"])
    agg["severity"] = np.clip(
        np.rint(
            4
            + 3.0 * agg["query_type"].isin(["warning_or_limitation", "threshold_query", "excluded_cause"]).astype(float)
            + 1.0 * agg["query_type"].isin(["multi_hop_diagnosis", "cross_page_evidence_query", "component_to_failure"]).astype(float)
            + 2.0 * (1.0 - agg["grounded_score"])
        ),
        1,
        10,
    ).astype(int)
    agg["occurrence"] = [quantile_score(agg["n_queries"], v) for v in agg["n_queries"]]
    detection_quality = (
        0.35 * agg["accepted_recall_10"]
        + 0.35 * agg["evidence_package_recall_10"]
        + 0.20 * agg["grounded_score"]
        + 0.10 * agg["role_coverage_proxy"]
    )
    agg["detection_difficulty"] = np.clip(np.rint(10 - 9 * detection_quality), 1, 10).astype(int)
    agg["rpn"] = agg["severity"] * agg["occurrence"] * agg["detection_difficulty"]
    agg["criticality_class"] = pd.cut(
        agg["rpn"],
        bins=[-1, 100, 200, 350, 1000],
        labels=["Low", "Moderate", "High", "Critical"],
    ).astype(str)
    agg["recommended_control"] = agg.apply(query_control, axis=1)
    return agg.sort_values(["rpn", "hallucination_risk_proxy"], ascending=False)


def query_control(row: pd.Series) -> str:
    qt = row["query_type"]
    if qt == "cross_page_evidence_query":
        return "Require section-continuity and neighboring-page package evidence before answer emission."
    if qt == "component_to_failure":
        return "Use component profile plus cause/failure role evidence; avoid single generic component snippets."
    if qt == "warning_or_limitation":
        return "Require warning/caution/note/limitation slot and cite source page."
    if qt == "threshold_query":
        return "Require threshold plus operating condition in the same package."
    if qt == "excluded_cause":
        return "Separate excluded-cause evidence from confirmed-cause evidence."
    if qt == "authority_or_manual_reference":
        return "Require manual/section/page authority evidence."
    return "Maintain role-complete evidence package and citation coverage."


def build_method_control_table(paths: CasePaths, metrics: pd.DataFrame) -> pd.DataFrame:
    df = metrics.copy()
    df["method_display"] = df["method"].replace({"FD-GRAG-RC-V21-final": METHOD_OURS_DISPLAY})
    df["fmeca_detection_control_score"] = (
        0.30 * df["accepted_recall_10"].astype(float)
        + 0.25 * df["evidence_package_recall_10"].astype(float)
        + 0.20 * df["accepted_page_recall_10"].astype(float)
        + 0.10 * df["accepted_section_recall_10"].astype(float)
        + 0.15 * df["grounded_score"].astype(float)
        - 0.10 * df["hallucination_risk_proxy"].astype(float)
    ).clip(0, 1)
    df["fmeca_detection_difficulty"] = np.clip(np.rint(10 - 9 * df["fmeca_detection_control_score"]), 1, 10).astype(int)
    df["control_interpretation"] = np.where(
        df["method"].eq(METHOD_OURS),
        "Role-component-causal evidence package control",
        "Baseline retrieval control",
    )
    return df[
        [
            "aircraft_model",
            "method",
            "method_display",
            "exact_recall_10",
            "accepted_recall_10",
            "evidence_package_recall_10",
            "grounded_score",
            "hallucination_risk_proxy",
            "fmeca_detection_control_score",
            "fmeca_detection_difficulty",
            "control_interpretation",
        ]
    ].sort_values("fmeca_detection_control_score", ascending=False)


def plot_risk_matrix(paths: CasePaths, comp: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(13.2, 8.2))
    cmap = LinearSegmentedColormap.from_list("risk", ["#2c7bb6", "#ffffbf", "#d7191c"])
    sizes = np.clip(comp["rpn"].astype(float) * 5.8, 140, 2050)
    detection = comp["detection_difficulty"].astype(float)
    use_detection_color = detection.nunique(dropna=True) > 1 and (detection.max() - detection.min()) > 0.75
    scatter_kwargs = dict(
        x=comp["occurrence"],
        y=comp["severity"],
        s=sizes,
        edgecolor="#111111",
        linewidth=1.2,
        alpha=0.92,
        zorder=3,
    )
    if use_detection_color:
        sc = ax.scatter(
            **scatter_kwargs,
            c=comp["detection_difficulty"],
            cmap=cmap,
            vmin=1,
            vmax=10,
        )
    else:
        sc = ax.scatter(**scatter_kwargs, color="#3F8FC3")

    x_min, x_max = 0.6, 12.8
    y_min, y_max = 0.6, 10.8
    labels = comp.copy()
    labels = labels.sort_values(["severity", "occurrence", "rpn"], ascending=[False, False, False]).reset_index(drop=True)
    placed_boxes: list[tuple[float, float, float, float]] = []
    point_boxes = [
        (float(row["occurrence"]) - 0.28, float(row["occurrence"]) + 0.28, float(row["severity"]) - 0.24, float(row["severity"]) + 0.24)
        for _, row in comp.iterrows()
    ]

    def risk_label_text(label: str) -> str:
        text = compact_component_label(label)
        return {
            "Flight controls": "Flight\ncontrols",
            "Gear / brake": "Gear\n/ brake",
            "Nav / avionics": "Nav\navionics",
            "Pneumatic / AC": "Pneumatic\n/ AC",
        }.get(text, text)

    def label_size(text: str) -> tuple[float, float]:
        lines = text.split("\n")
        width = 0.34 + 0.130 * max(len(line) for line in lines)
        height = 0.34 + 0.34 * len(lines)
        return width, height

    def candidate_box(tx: float, ty: float, width: float, height: float, ha: str) -> tuple[float, float, float, float]:
        if ha == "right":
            return (tx - width, tx, ty - height / 2, ty + height / 2)
        if ha == "center":
            return (tx - width / 2, tx + width / 2, ty - height / 2, ty + height / 2)
        return (tx, tx + width, ty - height / 2, ty + height / 2)

    def boxes_overlap(a: tuple[float, float, float, float], b: tuple[float, float, float, float], pad: float = 0.06) -> bool:
        return not (a[1] + pad < b[0] or a[0] - pad > b[1] or a[3] + pad < b[2] or a[2] - pad > b[3])

    label_candidates = [
        (0.42, 0.00, "left"),
        (-0.42, 0.00, "right"),
        (0.00, 0.52, "center"),
        (0.00, -0.52, "center"),
        (0.52, 0.34, "left"),
        (0.52, -0.34, "left"),
        (-0.52, 0.34, "right"),
        (-0.52, -0.34, "right"),
        (0.70, 0.58, "left"),
        (-0.70, 0.58, "right"),
        (0.70, -0.58, "left"),
        (-0.70, -0.58, "right"),
        (0.00, 0.82, "center"),
        (0.00, -0.82, "center"),
    ]

    for i, row in labels.iterrows():
        x = float(row["occurrence"])
        y = float(row["severity"])
        text = risk_label_text(row["component_label"])
        width, height = label_size(text)
        chosen = None
        for dx, dy, ha in label_candidates:
            tx = x + dx
            ty = y + dy
            box = candidate_box(tx, ty, width, height, ha)
            if box[0] < x_min + 0.08 or box[1] > x_max - 0.10 or box[2] < y_min + 0.25 or box[3] > y_max - 0.20:
                continue
            if any(boxes_overlap(box, existing) for existing in placed_boxes):
                continue
            if any(boxes_overlap(box, point_box, pad=0.02) for point_box in point_boxes):
                continue
            chosen = (tx, ty, ha, box)
            break
        if chosen is None:
            tx = min(x_max - width / 2 - 0.12, max(x, x_min + width / 2 + 0.15))
            vertical_offset = 1.02 if y < (y_min + y_max) / 2 else -1.02
            ty = min(y_max - height / 2 - 0.20, max(y + vertical_offset, y_min + height / 2 + 0.25))
            ha = "center"
            box = candidate_box(tx, ty, width, height, ha)
            chosen = (tx, ty, ha, box)
        tx, ty, ha, box = chosen
        placed_boxes.append(box)
        ax.annotate(
            text,
            xy=(x, y),
            xytext=(tx, ty),
            textcoords="data",
            ha=ha,
            va="center",
            fontsize=12.6,
            fontweight="bold",
            color="#111111",
            arrowprops=dict(
                arrowstyle="-",
                lw=0.82,
                color="#59616C",
                linestyle=(0, (2.2, 2.0)),
                shrinkA=2,
                shrinkB=5,
                connectionstyle="arc3,rad=0",
            ),
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#59616C", lw=0.75, alpha=0.97),
            annotation_clip=False,
            zorder=6,
        )
    ax.set_xlabel("Occurrence proxy")
    ax.set_ylabel("Severity proxy")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(range(1, 13))
    ax.set_yticks(range(1, 11))
    if use_detection_color:
        cbar = fig.colorbar(sc, ax=ax, fraction=0.038, pad=0.035)
        cbar.set_label("Detection difficulty")
        cbar.ax.tick_params(labelsize=14)
    add_axis_arrows(ax)
    out = paths.figures / f"{paths.case}_fmeca_risk_matrix.png"
    fig.tight_layout()
    fig.savefig(out, dpi=600)
    plt.close(fig)
    comp.to_csv(paths.figures / f"{paths.case}_fmeca_risk_matrix_plot_data.csv", index=False, encoding="utf-8-sig")
    save_caption(
        paths.figures / f"{paths.case}_fmeca_risk_matrix_caption.md",
        f"FMECA proxy risk matrix for {compact_case_name(paths.case)}. Severity, occurrence and detection are derived from document evidence roles, causal links and retrieval control performance; this is not an expert-certified aircraft safety assessment.",
    )


def plot_criticality_bars(paths: CasePaths, comp: pd.DataFrame) -> None:
    top = comp.head(10).iloc[::-1].copy()
    fig, ax = plt.subplots(figsize=(11.2, 8.0))
    colors = top["criticality_class"].map({"Low": "#4daf4a", "Moderate": "#f6c85f", "High": "#f28e2b", "Critical": "#d73027"}).fillna("#999999")
    ax.barh(top["component_label"], top["rpn"], color=colors, edgecolor="#111111", linewidth=1.0)
    for y, (_, row) in enumerate(top.iterrows()):
        ax.text(row["rpn"] + max(8, comp["rpn"].max() * 0.012), y, f"{int(row['rpn'])}", va="center", fontsize=15.5, fontweight="bold")
    ax.set_xlabel("Risk priority number proxy")
    ax.set_ylabel("")
    ax.set_xlim(0, comp["rpn"].max() * 1.20 + 1)
    ax.tick_params(axis="y", labelsize=15)
    ax.tick_params(axis="x", labelsize=15)
    add_axis_arrows(ax)
    out = paths.figures / f"{paths.case}_fmeca_component_criticality.png"
    fig.tight_layout()
    fig.savefig(out, dpi=600)
    plt.close(fig)
    top.to_csv(paths.figures / f"{paths.case}_fmeca_component_criticality_plot_data.csv", index=False, encoding="utf-8-sig")
    save_caption(
        paths.figures / f"{paths.case}_fmeca_component_criticality_caption.md",
        "Component-level FMECA criticality ranking derived from evidence role density, causal exposure and detection control difficulty.",
    )


def plot_role_control_heatmap(paths: CasePaths, comp: pd.DataFrame) -> None:
    roles = ["warning", "limitation", "threshold", "condition", "confirmed_cause", "excluded_cause", "action", "inspection", "replacement", "authority"]
    top = comp.head(9).copy()
    mat = np.log1p(top[roles].astype(float).to_numpy())
    fig, ax = plt.subplots(figsize=(13.4, 8.2))
    im = ax.imshow(mat, aspect="auto", cmap="magma")
    ax.set_yticks(np.arange(len(top)))
    ax.set_yticklabels(top["component_label"], fontsize=14.5, fontweight="bold")
    ax.set_xticks(np.arange(len(roles)))
    ax.set_xticklabels([r.replace("_", " ").title() for r in roles], rotation=35, ha="right", fontsize=14.2)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            raw = int(top.iloc[i][roles[j]])
            ax.text(j, i, str(raw), ha="center", va="center", color="white" if mat[i, j] > mat.max() * 0.45 else "#111111", fontsize=13.5, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("log evidence count")
    cbar.ax.tick_params(labelsize=13.5)
    out = paths.figures / f"{paths.case}_fmeca_role_control_heatmap.png"
    fig.tight_layout()
    fig.savefig(out, dpi=600)
    plt.close(fig)
    top[["component_label", *roles]].to_csv(paths.figures / f"{paths.case}_fmeca_role_control_heatmap_plot_data.csv", index=False, encoding="utf-8-sig")
    save_caption(
        paths.figures / f"{paths.case}_fmeca_role_control_heatmap_caption.md",
        "Role-control heatmap showing which FMECA controls are evidence-supported for the most critical components.",
    )


def plot_query_failure_modes(paths: CasePaths, query_fmeca: pd.DataFrame) -> None:
    if query_fmeca.empty:
        return
    top = query_fmeca.sort_values(["rpn", "hallucination_risk_proxy"], ascending=False).head(10).iloc[::-1].copy()
    fig, ax = plt.subplots(figsize=(12.4, 8.2))
    y = np.arange(len(top))
    ax.scatter(top["grounded_score"], y, s=260, color="#2c7bb6", edgecolor="#111111", linewidth=1.0, label="Grounded")
    ax.scatter(1 - top["hallucination_risk_proxy"], y, s=260, color="#1a9850", marker="D", edgecolor="#111111", linewidth=1.0, label="Risk control")
    for i, (_, row) in enumerate(top.iterrows()):
        ax.text(0.02, i, row["failure_mode"], va="center", ha="left", fontsize=14.5, fontweight="bold")
        grounded = float(row["grounded_score"])
        risk_control = float(1 - row["hallucination_risk_proxy"])
        ax.text(
            min(1.02, grounded + 0.022),
            i + 0.18,
            f"{grounded:.2f}",
            fontsize=13.4,
            fontweight="bold",
            ha="left",
            va="center",
            bbox=dict(boxstyle="round,pad=0.10", fc="white", ec="none", alpha=0.88),
            zorder=5,
        )
        ax.text(
            min(1.02, risk_control + 0.022),
            i - 0.18,
            f"{risk_control:.2f}",
            fontsize=13.0,
            ha="left",
            va="center",
            bbox=dict(boxstyle="round,pad=0.10", fc="white", ec="none", alpha=0.88),
            zorder=5,
        )
    ax.set_yticks([])
    ax.set_xlim(0, 1.08)
    ax.set_ylim(-0.62, len(top) - 0.34)
    ax.set_xlabel("Evidence control score")
    add_axis_arrows(ax, ypad=0.04)
    out = paths.figures / f"{paths.case}_fmeca_query_failure_mode_controls.png"
    fig.tight_layout()
    fig.savefig(out, dpi=600)
    plt.close(fig)
    top.to_csv(paths.figures / f"{paths.case}_fmeca_query_failure_mode_controls_plot_data.csv", index=False, encoding="utf-8-sig")
    save_caption(
        paths.figures / f"{paths.case}_fmeca_query_failure_mode_controls_caption.md",
        "Query-family FMECA control plot. Broad diagnostic tasks are treated as package-retrieval risks rather than single-evidence failures.",
    )


def write_report(paths: CasePaths, comp: pd.DataFrame, query_fmeca: pd.DataFrame, method_control: pd.DataFrame) -> None:
    top = comp.head(5)
    best_method = method_control.head(1).iloc[0] if not method_control.empty else None
    lines = [
        f"# {compact_case_name(paths.case)} FMECA proxy analysis",
        "",
        "This analysis adds a document-grounded FMECA layer to the V21 evidence-flow RAG package. It derives severity, occurrence and detection-control difficulty from evidence roles, causal-edge exposure, component profiles, safety/threshold/authority cues, and retrieval groundedness. It is an auditable proxy analysis, not an expert-certified FMECA or operational airworthiness assessment.",
        "",
        "## Top component risks",
        "",
        markdown_table(top[["component_label", "dominant_failure_mode", "local_effect_proxy", "higher_level_effect_proxy", "severity", "occurrence", "detection_difficulty", "rpn", "criticality_class", "recommended_action_proxy"]]),
        "",
        "## Method-level detection control",
        "",
    ]
    if best_method is not None:
        lines.append(
            f"The strongest retrieval control by the FMECA detection-control score is **{best_method['method_display']}** with score {best_method['fmeca_detection_control_score']:.3f} and detection difficulty {int(best_method['fmeca_detection_difficulty'])}."
        )
    lines.extend(
        [
            "",
            "## Highest query-family risks",
            "",
            markdown_table(query_fmeca.head(8)[["failure_mode", "n_queries", "severity", "occurrence", "detection_difficulty", "rpn", "grounded_score", "hallucination_risk_proxy", "recommended_control"]]) if not query_fmeca.empty else "No query-family FMECA rows were available.",
            "",
            "## Claim boundary",
            "",
            "- Severity, occurrence and detection are proxy scores derived from the local document benchmark and retrieval evidence chain.",
            "- The table supports research discussion of evidence controls, not certified failure-rate estimation.",
            "- A high RPN indicates where the retrieval/generation system needs stronger evidence slots, refusal behavior, or citation controls.",
        ]
    )
    (paths.tables / f"{paths.case}_fmeca_analysis_report.md").write_text("\n".join(lines), encoding="utf-8")


def clean_latest_fmeca_folder(paths: CasePaths) -> None:
    for p in paths.figures.glob("*"):
        if p.is_file():
            p.unlink()


def process_case(root: Path, case: str) -> dict:
    paths = build_paths(root, case)
    clean_latest_fmeca_folder(paths)
    ev = read_csv(first_match(paths.case_root, f"{case}_evidence_objects.csv"))
    edges = read_csv(first_match(paths.case_root, f"{case}_causal_edges.csv"))
    chains = read_csv(first_match(paths.case_root, f"{case}_complete_sota_evidence_chains.csv"))
    metrics = read_csv(first_match(paths.case_root, f"{case}_sota_metrics_by_method.csv"))

    comp = build_component_fmeca(paths, ev, edges, chains)
    query_fmeca = build_query_fmeca(paths, chains)
    method_control = build_method_control_table(paths, metrics)

    comp_path = paths.tables / f"{case}_fmeca_component_table.csv"
    query_path = paths.tables / f"{case}_fmeca_query_failure_mode_table.csv"
    method_path = paths.tables / f"{case}_fmeca_method_detection_control.csv"
    comp.to_csv(comp_path, index=False, encoding="utf-8-sig")
    query_fmeca.to_csv(query_path, index=False, encoding="utf-8-sig")
    method_control.to_csv(method_path, index=False, encoding="utf-8-sig")
    save_md_table(comp, paths.tables / f"{case}_fmeca_component_table.md")
    save_md_table(query_fmeca, paths.tables / f"{case}_fmeca_query_failure_mode_table.md")
    save_md_table(method_control, paths.tables / f"{case}_fmeca_method_detection_control.md")

    plot_risk_matrix(paths, comp)
    plot_criticality_bars(paths, comp)
    plot_role_control_heatmap(paths, comp)
    plot_query_failure_modes(paths, query_fmeca)
    write_report(paths, comp, query_fmeca, method_control)

    shutil.copy2(root / "10_scripts" / "build_v21_fmeca_analysis.py", paths.code / "build_v21_fmeca_analysis.py")
    return {
        "case": case,
        "component_rows": int(len(comp)),
        "query_failure_rows": int(len(query_fmeca)),
        "top_rpn": int(comp["rpn"].max()) if not comp.empty else 0,
        "critical_or_high_components": int(comp["criticality_class"].isin(["Critical", "High"]).sum()) if not comp.empty else 0,
        "tables": [str(comp_path), str(query_path), str(method_path)],
        "figures": [str(p) for p in sorted(paths.figures.glob("*.png"))],
    }


def write_global_report(root: Path, summaries: list[dict]) -> None:
    report_dir = root / "07_reports" / "v21"
    report_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# V21 FMECA proxy analysis addendum",
        "",
        "This addendum integrates Failure Mode, Effects and Criticality Analysis (FMECA) into the V21 runnable methodology package. The analysis is derived from existing evidence objects, component profiles, causal edges, safety/authority indexes and complete QA evidence-chain results.",
        "",
        "## Outputs",
        "",
    ]
    for s in summaries:
        lines.extend(
            [
                f"### {compact_case_name(s['case'])}",
                f"- Component-level FMECA rows: {s['component_rows']}",
                f"- Query-family FMECA rows: {s['query_failure_rows']}",
                f"- Maximum proxy RPN: {s['top_rpn']}",
                f"- High or critical components: {s['critical_or_high_components']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Scientific boundary",
            "",
            "The FMECA layer is a document-grounded proxy for evidence-control risk. It does not estimate certified failure probability, does not replace expert FMECA review, and does not imply real maintenance authorization. It is intended to support results discussion: which components and query families require stronger evidence packages, citation controls, safety slots, or refusal behavior.",
        ]
    )
    (report_dir / "v21_fmeca_analysis_report.md").write_text("\n".join(lines), encoding="utf-8")
    (report_dir / "v21_fmeca_analysis_manifest.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    np.random.seed(SEED)
    style_publication()
    root = find_root()
    summaries = []
    for case in ["Boeing_737", "Cessna_172"]:
        summaries.append(process_case(root, case))
    write_global_report(root, summaries)
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
