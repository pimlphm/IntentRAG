from __future__ import annotations

import math
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "05_figures" / "v21" / "case_trees"
OUT.mkdir(parents=True, exist_ok=True)

CASE_DIRS = {
    "Boeing 737": ROOT / "Cases" / "Boeing_737",
    "Cessna 172": ROOT / "Cases" / "Cessna_172",
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
    "no_component": "Unassigned Context",
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
    "condition_to_action": "Condition -> Action",
    "threshold_to_condition": "Threshold -> Condition",
    "warning_to_action": "Warning -> Action",
    "cause_to_action": "Cause -> Action",
    "cause_to_inspection": "Cause -> Inspection",
    "symptom_to_cause": "Symptom -> Cause",
    "symptom_to_excluded_cause": "Symptom -> Excluded Cause",
    "cause_to_replacement": "Cause -> Replacement",
}

COMPONENT_LABELS = {
    "发动机": "Engine",
    "襟翼": "Flap",
    "液压": "Hydraulic",
    "副翼": "Aileron",
    "空调": "Air Conditioning",
    "刹车": "Brake",
    "起落架": "Landing Gear",
    "客舱": "Cabin",
    "unassigned": "Cross-system Context",
    "": "Cross-system Context",
    "PACK": "PACK",
    "PWS": "Predictive Windshear",
    "MCDU": "MCDU",
    "brake": "Brake",
    "engine": "Engine",
    "battery": "Battery",
    "wheel": "Wheel",
    "wing": "Wing",
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
}

COLORS = {
    "root": "#1b1b1f",
    "relation": "#4477AA",
    "component": "#228833",
    "role": "#AA3377",
    "edge": "#A0A0A0",
    "bg": "#FBFBF8",
}


def first_file(case_dir: Path, pattern: str) -> Path:
    matches = list(case_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No file matching {pattern} in {case_dir}")
    return matches[0]


def wrap(label: str, width: int = 18) -> str:
    return "\n".join(textwrap.wrap(str(label), width=width, break_long_words=False))


def node(ax, x: float, y: float, text: str, color: str, size: float = 0.035, fs: int = 8):
    ax.scatter([x], [y], s=980 * (1 + size), color=color, edgecolor="white", linewidth=1.5, zorder=3)
    ax.text(
        x,
        y,
        wrap(text),
        ha="center",
        va="center",
        fontsize=fs,
        color="white",
        fontweight="bold",
        zorder=4,
    )


def edge(ax, x1: float, y1: float, x2: float, y2: float, width: float = 1.0):
    ax.plot([x1, x2], [y1, y2], color=COLORS["edge"], lw=width, alpha=0.7, zorder=1)


def finish(fig, ax, title: str, outbase: Path):
    ax.set_title(title, fontsize=15, fontweight="bold", pad=14)
    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(-0.08, 1.08)
    ax.axis("off")
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])
    fig.tight_layout()
    for ext in ["png", "svg", "pdf"]:
        fig.savefig(outbase.with_suffix(f".{ext}"), dpi=600 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)


def draw_causal_tree(case: str, case_dir: Path):
    ce = pd.read_csv(first_file(case_dir, "2 */*_causal_edges.csv")).fillna("")
    ce["component_label"] = ce["component"].astype(str).replace("", "unassigned").map(lambda x: COMPONENT_LABELS.get(x, x.title()))
    ce["relation_label"] = ce["relation"].astype(str).map(lambda x: RELATION_LABELS.get(x, x.replace("_", " ").title()))

    rel_counts = ce["relation_label"].value_counts().head(5)
    rows = []
    for rel in rel_counts.index:
        sub = ce[ce["relation_label"].eq(rel)]
        comps = sub["component_label"].value_counts().head(4)
        for comp, n in comps.items():
            rows.append({"case": case, "relation": rel, "component": comp, "edge_count": int(n)})
    data = pd.DataFrame(rows)
    if data.empty:
        return

    fig, ax = plt.subplots(figsize=(11, 7))
    root = (0.08, 0.50)
    node(ax, *root, f"{case}\nDiagnostic Causal Evidence", COLORS["root"], fs=8)

    rel_y = [0.88, 0.68, 0.50, 0.32, 0.12][: len(rel_counts)]
    for y, rel in zip(rel_y, rel_counts.index):
        rx, ry = 0.36, y
        edge(ax, root[0], root[1], rx, ry, 1.4)
        node(ax, rx, ry, rel, COLORS["relation"], fs=8)
        sub = data[data["relation"].eq(rel)].sort_values("edge_count", ascending=False)
        comp_y = list(pd.Series([ry + 0.09, ry + 0.03, ry - 0.03, ry - 0.09]).clip(0.05, 0.95))[: len(sub)]
        for cy, (_, r) in zip(comp_y, sub.iterrows()):
            cx = 0.72
            edge(ax, rx, ry, cx, cy, max(0.7, min(3.2, r["edge_count"] / max(1, data["edge_count"].max()) * 4)))
            label = f"{r['component']}\n{r['edge_count']} edges"
            node(ax, cx, cy, label, COLORS["component"], fs=7)

    outbase = OUT / f"{case.replace(' ', '_').lower()}_causal_tree"
    data.to_csv(outbase.with_name(outbase.name + "_plot_data.csv"), index=False, encoding="utf-8-sig")
    caption = (
        f"# {case} causal evidence tree\n\n"
        "This tree aggregates case-level causal edges into relation types and top component anchors. "
        "It is used to explain how FD-GRAG-RC retrieves diagnostic evidence as condition, threshold, warning, cause, and action flows rather than as flat text chunks."
    )
    outbase.with_name(outbase.name + "_caption.md").write_text(caption, encoding="utf-8")
    finish(fig, ax, f"{case}: Causal Evidence Tree", outbase)


def draw_component_tree(case: str, case_dir: Path):
    ev = pd.read_csv(first_file(case_dir, "1 */*_evidence_objects.csv")).fillna("")
    ev["family_label"] = ev["component_family"].astype(str).map(lambda x: FAMILY_LABELS.get(x, x.replace("_", " ").title()))
    ev["role_label"] = ev["role"].astype(str).map(lambda x: ROLE_LABELS.get(x, x.replace("_", " ").title()))
    fam_counts = ev[~ev["component_family"].eq("no_component")]["family_label"].value_counts().head(7)
    rows = []
    for fam in fam_counts.index:
        sub = ev[ev["family_label"].eq(fam)]
        roles = sub["role_label"].value_counts().head(3)
        for role, n in roles.items():
            rows.append({"case": case, "component_family": fam, "role": role, "evidence_count": int(n)})
    data = pd.DataFrame(rows)
    if data.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 7.2))
    root = (0.07, 0.50)
    node(ax, *root, f"{case}\nComponent Evidence Index", COLORS["root"], fs=8)
    fam_y = [0.91, 0.78, 0.65, 0.52, 0.39, 0.26, 0.13][: len(fam_counts)]
    max_count = max(1, int(data["evidence_count"].max()))
    for y, fam in zip(fam_y, fam_counts.index):
        fx, fy = 0.35, y
        edge(ax, root[0], root[1], fx, fy, 1.3)
        node(ax, fx, fy, f"{fam}\n{int(fam_counts[fam])} objects", COLORS["component"], fs=7)
        sub = data[data["component_family"].eq(fam)].sort_values("evidence_count", ascending=False)
        role_y = list(pd.Series([fy + 0.045, fy, fy - 0.045]).clip(0.05, 0.95))[: len(sub)]
        for ry, (_, r) in zip(role_y, sub.iterrows()):
            rx = 0.72
            edge(ax, fx, fy, rx, ry, max(0.7, min(3.0, r["evidence_count"] / max_count * 4)))
            node(ax, rx, ry, f"{r['role']}\n{r['evidence_count']} evidence", COLORS["role"], fs=7)

    outbase = OUT / f"{case.replace(' ', '_').lower()}_component_tree"
    data.to_csv(outbase.with_name(outbase.name + "_plot_data.csv"), index=False, encoding="utf-8-sig")
    caption = (
        f"# {case} component-role evidence tree\n\n"
        "This tree summarizes how evidence objects are organized by component family and diagnostic role. "
        "It supports the results discussion by showing that V21 retrieval is grounded in role-component structure, not only lexical matching."
    )
    outbase.with_name(outbase.name + "_caption.md").write_text(caption, encoding="utf-8")
    finish(fig, ax, f"{case}: Component-Role Evidence Tree", outbase)


def copy_to_case_results(case: str, case_dir: Path):
    result_dir = first_file(case_dir, "7 *") / "result_trees"
    result_dir.mkdir(parents=True, exist_ok=True)
    prefix = case.replace(" ", "_").lower()
    for f in OUT.glob(f"{prefix}_*"):
        if f.is_file():
            target = result_dir / f.name
            target.write_bytes(f.read_bytes())


def main():
    for case, case_dir in CASE_DIRS.items():
        draw_causal_tree(case, case_dir)
        draw_component_tree(case, case_dir)
        copy_to_case_results(case, case_dir)
    print(f"Saved case trees to {OUT}")


if __name__ == "__main__":
    main()
