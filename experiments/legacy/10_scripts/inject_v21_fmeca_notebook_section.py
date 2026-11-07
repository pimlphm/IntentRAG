from __future__ import annotations

import shutil
from pathlib import Path

import nbformat
import pandas as pd
from nbconvert import HTMLExporter, MarkdownExporter


MARKER = "V21.FMECA"


def find_root() -> Path:
    candidates = list((Path.home() / "Desktop").glob("*/FD_GRAG_RC_V21_Runnable_Methodology_Package_20260515"))
    if not candidates:
        raise FileNotFoundError("Cannot locate V21 package under Desktop.")
    return candidates[0]


def md_table(df: pd.DataFrame, max_rows: int = 8) -> str:
    df = df.head(max_rows).copy()
    if df.empty:
        return "_No rows available._"
    cols = list(df.columns)

    def fmt(x) -> str:
        if isinstance(x, float):
            text = f"{x:.4f}".rstrip("0").rstrip(".")
        else:
            text = str(x)
        text = text.replace("|", "/").replace("\n", " ")
        return text[:95] + "..." if len(text) > 98 else text

    rows = [[fmt(v) for v in row] for row in df[cols].itertuples(index=False, name=None)]
    return "\n".join(
        [
            "| " + " | ".join(cols) + " |",
            "| " + " | ".join(["---"] * len(cols)) + " |",
            *["| " + " | ".join(r) + " |" for r in rows],
        ]
    )


def rel(path: Path, base: Path) -> str:
    return path.relative_to(base).as_posix()


def build_cells(root: Path) -> list:
    cells = []
    intro = f"""## {MARKER} FMECA Proxy Analysis

This section adds a Failure Mode, Effects and Criticality Analysis layer to the V21 evidence-flow package. The scores are **document-grounded proxy scores** derived from evidence objects, component profiles, causal edges, safety/threshold/authority cues, and retrieval detection-control performance.

Important boundary: this is not an expert-certified FMECA, not a certified failure-rate model, and not an airworthiness or maintenance-release conclusion. It is used to discuss which evidence roles and component families require stronger retrieval, citation, package-completeness, and refusal controls.
"""
    cells.append(nbformat.v4.new_markdown_cell(intro))
    for case in ["Boeing_737", "Cessna_172"]:
        case_name = "Boeing 737" if case == "Boeing_737" else "Cessna 172"
        case_root = root / "Cases" / case
        table = next(case_root.rglob(f"{case}_fmeca_component_table.csv"))
        query_table = next(case_root.rglob(f"{case}_fmeca_query_failure_mode_table.csv"))
        comp = pd.read_csv(table)
        qfm = pd.read_csv(query_table)
        cols = [
            "component_label",
            "dominant_failure_mode",
            "severity",
            "occurrence",
            "detection_difficulty",
            "rpn",
            "criticality_class",
        ]
        qcols = [
            "failure_mode",
            "n_queries",
            "severity",
            "occurrence",
            "detection_difficulty",
            "rpn",
            "grounded_score",
            "hallucination_risk_proxy",
        ]
        fig_dir = next(p for p in (case_root / "8 最新图集").iterdir() if p.name == "07_fmeca_analysis")
        risk_fig = fig_dir / f"{case}_fmeca_risk_matrix.png"
        crit_fig = fig_dir / f"{case}_fmeca_component_criticality.png"
        heat_fig = fig_dir / f"{case}_fmeca_role_control_heatmap.png"
        query_fig = fig_dir / f"{case}_fmeca_query_failure_mode_controls.png"
        section = f"""### {case_name}: FMECA Component and Evidence-Control Summary

Top component-level proxy FMECA rows:

{md_table(comp[cols], max_rows=10)}

Highest query-family FMECA risks:

{md_table(qfm[qcols], max_rows=8)}

FMECA figures:

![{case_name} FMECA risk matrix]({rel(risk_fig, root)})

![{case_name} component criticality]({rel(crit_fig, root)})

![{case_name} role-control heatmap]({rel(heat_fig, root)})

![{case_name} query failure-mode controls]({rel(query_fig, root)})
"""
        cells.append(nbformat.v4.new_markdown_cell(section))
    code = """# Reproduce the FMECA addendum from the V21 package root.
# The script reads only existing V21 evidence objects, causal edges, component profiles,
# QA evidence-chain results, and method metrics; it does not change retrieval scores.
import subprocess, sys
from pathlib import Path
root = Path.cwd()
script = root / "10_scripts" / "build_v21_fmeca_analysis.py"
print(script)
# subprocess.run([sys.executable, str(script)], check=True)
"""
    cells.append(nbformat.v4.new_code_cell(code))
    return cells


def main() -> None:
    root = find_root()
    nb_path = root / "09_notebook" / "FD_GRAG_RC_V21_Runnable_Methodology_Notebook.ipynb"
    backup = root / "09_notebook" / "FD_GRAG_RC_V21_Runnable_Methodology_Notebook_pre_fmeca_backup.ipynb"
    if not backup.exists():
        shutil.copy2(nb_path, backup)
    nb = nbformat.read(nb_path, as_version=4)
    nb.cells = [c for c in nb.cells if MARKER not in c.get("source", "")]
    nb.cells.extend(build_cells(root))
    nbformat.write(nb, nb_path)

    html_exporter = HTMLExporter()
    body, _ = html_exporter.from_notebook_node(nb)
    (root / "09_notebook" / "FD_GRAG_RC_V21_Runnable_Methodology_Notebook.html").write_text(body, encoding="utf-8")
    md_exporter = MarkdownExporter()
    body, _ = md_exporter.from_notebook_node(nb)
    (root / "09_notebook" / "FD_GRAG_RC_V21_Runnable_Methodology_Notebook.md").write_text(body, encoding="utf-8")
    print(nb_path)


if __name__ == "__main__":
    main()
