from __future__ import annotations

import argparse
import math
import re
import shutil
import textwrap
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.path import Path as MplPath
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, PathPatch


mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.unicode_minus": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


CASE_LABELS = {
    "Boeing_737": "Boeing 737",
    "Cessna_172": "Cessna 172",
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
    "confirmed_cause": "Confirmed Cause",
    "excluded_cause": "Excluded Cause",
    "action": "Maintenance Action",
    "inspection": "Inspection",
    "replacement": "Replacement",
    "authority": "Manual / Authority",
    "evidence": "Evidence",
    "general": "General Context",
}

RELATION_LABELS = {
    "symptom_to_cause": "Symptom -> Cause",
    "symptom_to_excluded_cause": "Symptom -> Excluded Cause",
    "cause_to_action": "Cause -> Action",
    "cause_to_inspection": "Cause -> Inspection",
    "cause_to_replacement": "Cause -> Replacement",
    "threshold_to_condition": "Threshold -> Condition",
    "condition_to_action": "Condition -> Action",
    "warning_to_action": "Warning -> Action",
}

RELATION_ORDER = [
    "Symptom -> Cause",
    "Symptom -> Excluded Cause",
    "Cause -> Inspection",
    "Cause -> Action",
    "Cause -> Replacement",
    "Threshold -> Condition",
    "Condition -> Action",
    "Warning -> Action",
]

RELATION_FLOW = {
    "Symptom -> Cause": "Symptom\n-> Cause",
    "Symptom -> Excluded Cause": "Symptom\n-> Ruled-out cause",
    "Cause -> Inspection": "Cause\n-> Inspection",
    "Cause -> Action": "Cause\n-> Maintenance action",
    "Cause -> Replacement": "Cause\n-> Replacement",
    "Threshold -> Condition": "Threshold\n-> Condition",
    "Condition -> Action": "Condition\n-> Procedure",
    "Warning -> Action": "Warning\n-> Constrained action",
}

COMPONENT_LABELS = {
    "发动机": "Engine",
    "引擎": "Engine",
    "滑油": "Oil system",
    "燃油": "Fuel system",
    "襟翼": "Flap",
    "副翼": "Aileron",
    "方向舵": "Rudder",
    "升降舵": "Elevator",
    "液压": "Hydraulic",
    "空调": "Air conditioning",
    "气源": "Pneumatic air",
    "刹车": "Brake",
    "起落架": "Landing gear",
    "前轮": "Nose wheel",
    "主轮": "Main wheel",
    "客舱": "Cabin",
    "电源": "Electrical power",
    "电瓶": "Battery",
    "风挡": "Windshield",
    "雷达": "Weather radar",
    "气象雷达": "Weather radar",
    "防冰": "Anti-ice",
    "进气道": "Inlet anti-ice",
    "前缘": "Leading edge",
    "后缘": "Trailing edge",
    "反推": "Thrust reverser",
    "减速板": "Speedbrake",
    "自动刹车": "Autobrake",
    "增压": "Pressurization",
    "座舱": "Cabin",
    "PACK": "PACK",
    "R PACK": "Right PACK",
    "L PACK": "Left PACK",
    "PWS": "Predictive Windshear",
    "MCDU": "MCDU",
    "IDG": "IDG generator",
    "APU": "APU",
    "valve": "Valve",
    "pack": "PACK",
    "engine": "Engine",
    "battery": "Battery",
    "wheel": "Wheel",
    "wing": "Wing",
    "irs": "IRS",
    "IRS": "IRS",
    "alternator": "Alternator",
    "flap": "Flap",
    "cylinder": "Cylinder",
    "seat": "Seat",
    "skin": "Skin",
    "static": "Static System",
    "fuselage": "Fuselage",
    "circuit breaker": "Circuit Breaker",
    "trim tab": "Trim Tab",
    "starter": "Starter",
    "master switch": "Master Switch",
    "mixture": "Mixture Control",
    "magneto": "Magneto",
    "tire": "Tire",
    "brake": "Brake",
    "fuel selector": "Fuel Selector",
    "carburetor": "Carburetor",
    "pitot": "Pitot System",
    "oil": "Oil System",
    "": "Cross-System",
    "nan": "Cross-System",
}

PALETTE = {
    "ink": "#17212B",
    "deep_ink": "#0B1320",
    "white": "#FFFFFF",
    "muted": "#5B677A",
    "background": "#FFFFFF",
    "grid": "#D9DEE7",
    "overview": "#FFF4D6",
    "overview_edge": "#B47905",
    "root": "#1B2430",
    "relation": "#3E6AA7",
    "flow": "#E8F0FA",
    "component": "#2F8F6B",
    "component_light": "#E8F5EF",
    "role": "#7A4C9F",
    "role_light": "#F0EAF7",
    "safety": "#C65F44",
    "safety_light": "#F8EDE8",
    "authority": "#5B5EA6",
    "authority_light": "#ECECFA",
    "edge": "#9AA6B8",
}


def find_package_root(start: Path) -> Path:
    candidates = [start.resolve(), *start.resolve().parents, Path.cwd().resolve(), *Path.cwd().resolve().parents]
    for p in candidates:
        if (p / "Cases").exists() and (p / "04_results").exists():
            return p
    raise FileNotFoundError("Could not locate the V21 runnable package root.")


ROOT = find_package_root(Path(__file__).resolve())


def first_dir(case_dir: Path, prefix: str) -> Path:
    matches = sorted([p for p in case_dir.glob(f"{prefix}*") if p.is_dir()])
    if not matches:
        raise FileNotFoundError(f"No step directory starting with {prefix!r} under {case_dir}")
    return matches[0]


def first_file(folder: Path, pattern: str) -> Path:
    matches = sorted(folder.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No file matching {pattern} under {folder}")
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


def title_label(text: str) -> str:
    text = clean(text)
    if not text:
        return "Cross-System"
    if text in COMPONENT_LABELS:
        return COMPONENT_LABELS[text]
    if text.lower() in COMPONENT_LABELS:
        return COMPONENT_LABELS[text.lower()]
    if re.search(r"[\u4e00-\u9fff]", text):
        return "Localized Component"
    if text.isupper() and len(text) <= 8:
        return text
    return text.replace("_", " ").title()


def family_label(text: str) -> str:
    return FAMILY_LABELS.get(clean(text), clean(text).replace("_", " ").title() or "Cross-System Context")


def role_label(text: str) -> str:
    return ROLE_LABELS.get(clean(text), clean(text).replace("_", " ").title() or "General Context")


def relation_label(text: str) -> str:
    return RELATION_LABELS.get(clean(text), clean(text).replace("_", " ").title() or "Unspecified Relation")


def split_items(text: object) -> list[str]:
    value = clean(text)
    if not value:
        return []
    parts = re.split(r"[;,，、/|]+", value)
    return [clean(p) for p in parts if clean(p)]


def wrap_text(text: str, width: int) -> str:
    parts: list[str] = []
    for line in str(text).split("\n"):
        wrapped = textwrap.wrap(line, width=width, break_long_words=False, break_on_hyphens=False)
        parts.extend(wrapped if wrapped else [""])
    return "\n".join(parts)


def add_box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    face: str,
    edge: str | None = None,
    text_color: str = "#FFFFFF",
    fontsize: float = 11.5,
    weight: str = "bold",
    align: str = "center",
    wrap: int = 20,
    lw: float = 1.2,
):
    patch = FancyBboxPatch(
        (x, y - h / 2),
        w,
        h,
        boxstyle="round,pad=0.018,rounding_size=0.055",
        linewidth=lw,
        edgecolor=edge or face,
        facecolor=face,
        zorder=3,
    )
    ax.add_patch(patch)
    tx = x + (w / 2 if align == "center" else 0.13)
    ha = "center" if align == "center" else "left"
    ax.text(
        tx,
        y,
        wrap_text(text, wrap),
        ha=ha,
        va="center",
        fontsize=fontsize,
        color=text_color,
        fontweight=weight,
        linespacing=1.13,
        zorder=4,
    )
    return patch


def add_edge(ax, start: tuple[float, float], end: tuple[float, float], lw: float = 1.0, color: str = PALETTE["edge"], alpha: float = 0.72):
    sx, sy = start
    ex, ey = end
    arrow = FancyArrowPatch(
        (sx, sy),
        (ex, ey),
        connectionstyle="angle3,angleA=0,angleB=90",
        arrowstyle="-|>",
        mutation_scale=max(18, 12 + lw * 4.2),
        linewidth=max(1.55, lw * 1.35),
        color=color,
        alpha=alpha,
        shrinkA=4,
        shrinkB=8,
        capstyle="round",
        joinstyle="round",
        zorder=1,
    )
    ax.add_patch(arrow)


def add_header(ax, height: float, title: str, subtitle: str):
    return None


def add_document_overview(ax, height: float, width: float, mode: str):
    return None


def finish_figure(fig, ax, outbase: Path, xlim: tuple[float, float], ylim: tuple[float, float]):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    fig.patch.set_facecolor(PALETTE["background"])
    ax.set_facecolor(PALETTE["background"])
    outbase.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".svg", ".pdf"):
        stale = outbase.with_suffix(ext)
        if stale.exists():
            stale.unlink()
    fig.savefig(outbase.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def role_colors(role: str) -> tuple[str, str, str]:
    r = role.lower()
    if any(k in r for k in ["warning", "limitation", "threshold", "condition", "note"]):
        return PALETTE["safety_light"], PALETTE["safety"], PALETTE["ink"]
    if any(k in r for k in ["authority", "manual"]):
        return PALETTE["authority_light"], PALETTE["authority"], PALETTE["ink"]
    return PALETTE["role_light"], PALETTE["role"], PALETTE["ink"]


def case_paths(case_key: str) -> dict[str, Path]:
    case_dir = ROOT / "Cases" / case_key
    return {
        "case": case_dir,
        "step1": first_dir(case_dir, "1 "),
        "step2": first_dir(case_dir, "2 "),
        "step3": first_dir(case_dir, "3 "),
        "test": first_dir(case_dir, "7 "),
    }


def build_causal_data(case_key: str) -> pd.DataFrame:
    paths = case_paths(case_key)
    ce = read_csv(first_file(paths["step2"], "*_causal_edges.csv")).fillna("")
    ce["relation_label"] = ce["relation"].map(relation_label)
    ce["component_label"] = ce["component"].map(title_label)
    grouped = (
        ce.groupby(["relation_label", "component_label"], dropna=False)
        .agg(edge_count=("edge_id", "count"), mean_confidence=("confidence", "mean"))
        .reset_index()
    )
    grouped["case"] = CASE_LABELS[case_key]
    grouped["relation_total"] = grouped.groupby("relation_label")["edge_count"].transform("sum")
    grouped["relation_order"] = grouped["relation_label"].map(lambda x: RELATION_ORDER.index(x) if x in RELATION_ORDER else 99)
    grouped = grouped.sort_values(["relation_order", "relation_total", "edge_count"], ascending=[True, False, False])
    rows = []
    for rel, sub in grouped.groupby("relation_label", sort=False):
        for _, r in sub.sort_values("edge_count", ascending=False).head(5).iterrows():
            rows.append(r.to_dict())
    return pd.DataFrame(rows)


def draw_causal_tree(case_key: str) -> list[Path]:
    label = CASE_LABELS[case_key]
    paths = case_paths(case_key)
    data = build_causal_data(case_key)
    if data.empty:
        return []

    relation_rows = (
        data[["relation_label", "relation_total", "relation_order"]]
        .drop_duplicates()
        .sort_values(["relation_order", "relation_total"], ascending=[True, False])
    )
    n_rel = len(relation_rows)
    row_gap = 1.26
    top_pad = 0.78
    height = max(4.8, top_pad + max(1, n_rel - 1) * row_gap + 1.05)
    width = 18.8
    fig, ax = plt.subplots(figsize=(width, height))
    root_y = height - top_pad - (n_rel - 1) * row_gap / 2
    add_box(ax, 0.35, root_y, 2.05, 0.92, "Diagnostic\ncausal KG", PALETTE["root"], fontsize=12.5, wrap=18)

    max_edges = max(1, int(data["edge_count"].max()))
    plot_rows = []
    for idx, (_, rel_row) in enumerate(relation_rows.iterrows()):
        y = height - top_pad - idx * row_gap
        relation = rel_row["relation_label"]
        total = int(rel_row["relation_total"])
        add_edge(ax, (2.40, root_y), (3.02, y), lw=1.25)
        add_box(ax, 3.03, y, 2.65, 0.72, f"{relation}\n{total} edges", PALETTE["relation"], fontsize=10.8, wrap=22)
        flow = RELATION_FLOW.get(relation, "causal evidence\nsupports retrieval path")
        add_edge(ax, (5.68, y), (6.12, y), lw=1.0)
        add_box(ax, 6.12, y, 2.08, 0.72, flow, PALETTE["flow"], edge=PALETTE["relation"], text_color=PALETTE["ink"], fontsize=9.4, weight="normal", wrap=22)
        sub = data[data["relation_label"].eq(relation)].sort_values("edge_count", ascending=False).head(5)
        x0 = 8.70
        chip_w = 1.82
        chip_gap = 0.18
        for j, (_, r) in enumerate(sub.iterrows()):
            x = x0 + j * (chip_w + chip_gap)
            lw = 0.75 + 2.2 * float(r["edge_count"]) / max_edges
            add_edge(ax, (8.20, y), (x, y), lw=lw, color=PALETTE["component"], alpha=0.55)
            add_box(
                ax,
                x,
                y,
                chip_w,
                0.62,
                f"{r['component_label']}\n{int(r['edge_count'])} links",
                PALETTE["component_light"],
                edge=PALETTE["component"],
                text_color=PALETTE["ink"],
                fontsize=9.4,
                weight="bold",
                wrap=18,
            )
            plot_rows.append(
                {
                    "case": label,
                    "relation": relation,
                    "component_anchor": r["component_label"],
                    "edge_count": int(r["edge_count"]),
                    "mean_confidence": round(float(r.get("mean_confidence", 0) or 0), 4),
                    "relation_total": total,
                }
            )

    out_dir = paths["step2"] / "result_trees_highres"
    out_dir.mkdir(parents=True, exist_ok=True)
    outbase = out_dir / f"{case_key}_diagnostic_causal_tree_highres"
    pd.DataFrame(plot_rows).to_csv(outbase.with_name(outbase.name + "_plot_data.csv"), index=False, encoding="utf-8-sig")
    outbase.with_name(outbase.name + "_caption.md").write_text(
        f"# {label}: Diagnostic causal evidence tree\n\n"
        "This high-resolution tree summarizes the case-specific causal KG used by FD-GRAG-RC. "
        "The tree expands diagnostic relation types, explicit causal direction, and the strongest component anchors. "
        "The figure supports results discussion and is not a claim of field causal validation.\n",
        encoding="utf-8",
    )
    finish_figure(fig, ax, outbase, (0, 18.9), (0, height))
    return [outbase.with_suffix(".png")]


def alias_summary(case_key: str, family: str, profile_row: pd.Series | None, aliases: pd.DataFrame) -> str:
    components: list[str] = []
    if profile_row is not None:
        for item in split_items(profile_row.get("components", "")):
            components.append(title_label(item))
    fam_alias = aliases[aliases["component_family"].fillna("").astype(str).eq(family)] if not aliases.empty else pd.DataFrame()
    for col in ["canonical", "alias"]:
        if col in fam_alias:
            components.extend(title_label(x) for x in fam_alias[col].dropna().astype(str).head(12))
    cleaned = []
    for c in components:
        if c and c not in cleaned and c != "Localized Component":
            cleaned.append(c)
    if not cleaned:
        return "Profile card\nsection-level evidence"
    return "Profile anchors:\n" + ", ".join(cleaned[:5])


def build_component_data(case_key: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = case_paths(case_key)
    ev = read_csv(first_file(paths["step1"], "*_evidence_objects.csv")).fillna("")
    profiles = read_csv(first_file(paths["step3"], "*_component_profiles.csv")).fillna("")
    alias_file = paths["step3"] / "component_aliases.csv"
    aliases = read_csv(alias_file).fillna("") if alias_file.exists() else pd.DataFrame()
    ev["family_label"] = ev["component_family"].map(family_label)
    ev["role_label"] = ev["role"].map(role_label)
    fam_counts = (
        ev[~ev["component_family"].astype(str).eq("no_component")]
        .groupby(["component_family", "family_label"])
        .size()
        .reset_index(name="evidence_objects")
        .sort_values("evidence_objects", ascending=False)
    )
    role_counts = (
        ev[~ev["component_family"].astype(str).eq("no_component")]
        .groupby(["component_family", "family_label", "role_label"])
        .size()
        .reset_index(name="role_evidence_count")
        .sort_values(["component_family", "role_evidence_count"], ascending=[True, False])
    )
    return fam_counts, role_counts, profiles, aliases


def draw_component_tree(case_key: str) -> list[Path]:
    label = CASE_LABELS[case_key]
    paths = case_paths(case_key)
    fam_counts, role_counts, profiles, aliases = build_component_data(case_key)
    if fam_counts.empty:
        return []

    fam_counts = fam_counts.head(9).copy()
    n_fam = len(fam_counts)
    row_gap = 1.22
    top_pad = 0.78
    height = max(5.8, top_pad + max(1, n_fam - 1) * row_gap + 1.05)
    width = 19.6
    fig, ax = plt.subplots(figsize=(width, height))
    root_y = height - top_pad - (n_fam - 1) * row_gap / 2
    add_box(ax, 0.35, root_y, 2.05, 0.92, "Component\nprofile index", PALETTE["root"], fontsize=12.5, wrap=18)

    profiles_by_family = {clean(r.get("component_family", "")): r for _, r in profiles.iterrows()}
    max_role = max(1, int(role_counts["role_evidence_count"].max()))
    plot_rows = []
    for idx, (_, fam) in enumerate(fam_counts.iterrows()):
        y = height - top_pad - idx * row_gap
        family = clean(fam["component_family"])
        fam_label = clean(fam["family_label"])
        total = int(fam["evidence_objects"])
        profile = profiles_by_family.get(family)
        add_edge(ax, (2.40, root_y), (3.05, y), lw=1.15)
        add_box(ax, 3.05, y, 2.70, 0.72, f"{fam_label}\n{total} evidence objects", PALETTE["component"], fontsize=9.6, wrap=22)
        add_edge(ax, (5.75, y), (6.15, y), lw=1.0)
        add_box(
            ax,
            6.15,
            y,
            2.70,
            0.72,
            alias_summary(case_key, family, profile, aliases),
            PALETTE["component_light"],
            edge=PALETTE["component"],
            text_color=PALETTE["ink"],
            fontsize=8.6,
            weight="normal",
            align="left",
            wrap=27,
        )
        sub = role_counts[role_counts["component_family"].eq(family)].sort_values("role_evidence_count", ascending=False).head(4)
        x0 = 9.45
        card_w = 2.05
        card_gap = 0.16
        for j, (_, r) in enumerate(sub.iterrows()):
            x = x0 + j * (card_w + card_gap)
            face, edge_col, text_col = role_colors(clean(r["role_label"]))
            lw = 0.75 + 2.15 * float(r["role_evidence_count"]) / max_role
            add_edge(ax, (8.85, y), (x, y), lw=lw, color=edge_col, alpha=0.58)
            add_box(
                ax,
                x,
                y,
                card_w,
                0.62,
                f"{r['role_label']}\n{int(r['role_evidence_count'])}",
                face,
                edge=edge_col,
                text_color=text_col,
                fontsize=9.4,
                weight="bold",
                wrap=18,
            )
            plot_rows.append(
                {
                    "case": label,
                    "component_family": family,
                    "component_family_label": fam_label,
                    "evidence_objects": total,
                    "role": r["role_label"],
                    "role_evidence_count": int(r["role_evidence_count"]),
                }
            )

    out_dir = paths["step3"] / "result_trees_highres"
    out_dir.mkdir(parents=True, exist_ok=True)
    outbase = out_dir / f"{case_key}_component_role_tree_highres"
    pd.DataFrame(plot_rows).to_csv(outbase.with_name(outbase.name + "_plot_data.csv"), index=False, encoding="utf-8-sig")
    outbase.with_name(outbase.name + "_caption.md").write_text(
        f"# {label}: ATA-oriented component and role evidence tree\n\n"
        "This high-resolution tree summarizes the component profile index used by FD-GRAG-RC. "
        "The tree expands component families, section/profile anchors, and diagnostic role coverage.\n",
        encoding="utf-8",
    )
    finish_figure(fig, ax, outbase, (0, 19.7), (0, height))
    return [outbase.with_suffix(".png")]


def mirror_outputs(case_key: str):
    paths = case_paths(case_key)
    mirror = paths["test"] / "result_trees_highres"
    mirror.mkdir(parents=True, exist_ok=True)
    for stale in mirror.iterdir():
        if stale.is_file() and stale.suffix.lower() in {".png", ".svg", ".pdf", ".jpg", ".jpeg"}:
            stale.unlink()
    for source_dir in [paths["step2"] / "result_trees_highres", paths["step3"] / "result_trees_highres"]:
        if not source_dir.exists():
            continue
        for f in source_dir.iterdir():
            if f.is_file():
                shutil.copy2(f, mirror / f.name)


def copy_code_to_case(case_key: str):
    paths = case_paths(case_key)
    code_dir = paths["case"] / "tree_building_code"
    code_dir.mkdir(parents=True, exist_ok=True)
    target = code_dir / "build_highres_case_trees.py"
    shutil.copy2(Path(__file__).resolve(), target)
    readme = code_dir / "README.md"
    readme.write_text(
        "# High-Resolution Case Tree Builder\n\n"
        "Run from the V21 package root:\n\n"
        "```powershell\n"
        "& '${USER_HOME}/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' "
        f"'10_scripts/build_highres_case_trees.py' --case {case_key}\n"
        "```\n\n"
        "The script reads the case-local causal-edge and component-profile CSV files and regenerates PNG-only tree figures with high-contrast backgrounds and a document-panorama strip.\n",
        encoding="utf-8",
    )


def write_case_index(case_key: str):
    paths = case_paths(case_key)
    label = CASE_LABELS[case_key]
    index = paths["case"] / "tree_figure_index.md"
    text = (
        f"# {label} High-Resolution Result Trees\n\n"
        "Generated tree figures are stored in the case-local step folders and mirrored into the testing result folder.\n\n"
        "## Causal KG Tree\n\n"
        f"- Source data: `{paths['step2'].name}/*_causal_edges.csv`\n"
        f"- Figure: `{paths['step2'].name}/result_trees_highres/{case_key}_diagnostic_causal_tree_highres.png`\n\n"
        "## ATA Component-Role Tree\n\n"
        f"- Source data: `{paths['step3'].name}/*_component_profiles.csv`, `component_aliases.csv`, and step-1 evidence objects\n"
        f"- Figure: `{paths['step3'].name}/result_trees_highres/{case_key}_component_role_tree_highres.png`\n\n"
        "## Code\n\n"
        "- Case-local code copy: `tree_building_code/build_highres_case_trees.py`\n"
        "- Root code: `10_scripts/build_highres_case_trees.py`\n"
    )
    index.write_text(text, encoding="utf-8")


def selected_cases(raw: str | None) -> Iterable[str]:
    if not raw or raw.lower() in {"all", "*"}:
        return CASE_LABELS.keys()
    key = raw.strip()
    if key in CASE_LABELS:
        return [key]
    aliases = {v.lower().replace(" ", "_"): k for k, v in CASE_LABELS.items()}
    if key.lower() in aliases:
        return [aliases[key.lower()]]
    raise ValueError(f"Unknown case {raw!r}. Use one of: {', '.join(CASE_LABELS)}")


def main():
    parser = argparse.ArgumentParser(description="Build high-resolution case-level causal and component trees.")
    parser.add_argument("--case", default="all", help="Case key: Boeing_737, Cessna_172, or all.")
    args = parser.parse_args()
    generated = []
    for case_key in selected_cases(args.case):
        generated.extend(draw_causal_tree(case_key))
        generated.extend(draw_component_tree(case_key))
        mirror_outputs(case_key)
        copy_code_to_case(case_key)
        write_case_index(case_key)
    print("Generated high-resolution case trees:")
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
