from __future__ import annotations

import math
import re
from pathlib import Path

import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.chart import BarChart, Reference
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.table import Table, TableStyleInfo


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "04_results" / "v21"
CASE_DIR = ROOT / "Cases"
TEMPLATE = ROOT / "11_manifest" / "template_qa_chain_131914.xlsx"

METHOD = "FD-GRAG-RC-V21-final"

TEMPLATE_HEADERS = [
    "query_id",
    "question",
    "gold_answer",
    "gold_pages",
    "gold_episode",
    "required_roles",
    "method",
    "final_answer",
    "review_issues",
    "strict_pass",
    "review_status",
    "answer_accuracy_proxy",
    "answer_precision",
    "answer_recall",
    "answer_f1",
    "evidence_chain_completeness",
]

EXTRA_HEADERS = [
    "aircraft_model",
    "split",
    "track",
    "query_type",
    "difficulty_level",
    "exact_recall_10",
    "accepted_recall_10",
    "package_recall_10",
    "page_recall_10",
    "section_recall_10",
    "grounded_score",
    "hallucination_risk_proxy",
    "safety_gate_decision",
    "top_evidence_ids",
    "top_evidence_pages",
    "top_evidence_roles",
    "top_evidence_components",
    "top_evidence_sections",
    "evidence_chain_top10",
    "exact_gold_evidence_ids",
    "accepted_gold_evidence_ids",
    "accepted_gold_page_ids",
    "accepted_gold_section_ids",
]

HEADERS = TEMPLATE_HEADERS + EXTRA_HEADERS

CASE_SHEET = {
    "Boeing 737": "Boeing_QA_Results",
    "Cessna 172": "Cessna_QA_Results",
}

CASE_FOLDER = {
    "Boeing 737": "Boeing_737",
    "Cessna 172": "Cessna_172",
}


def norm(text: object) -> str:
    return " ".join(str(text or "").split())


def excel_safe(value: object):
    if isinstance(value, str):
        # Excel XML cannot store most ASCII control characters.
        return re.sub(r"[\x00-\x08\x0B-\x0C\x0E-\x1F]", " ", value)
    return value


def clip(text: object, n: int = 360) -> str:
    s = norm(text)
    return s if len(s) <= n else s[: n - 3] + "..."


def split_ids(value: object) -> list[str]:
    return [x for x in str(value or "").split(";") if x and x.lower() != "nan"]


def harmonic(a: float, b: float) -> float:
    if not a or not b:
        return 0.0
    return 2 * a * b / (a + b)


def metric_value(x: object) -> float:
    try:
        if pd.isna(x):
            return math.nan
        return float(x)
    except Exception:
        return math.nan


def strict_pass(row: pd.Series) -> bool:
    accepted = metric_value(row["accepted_recall_10"])
    package = metric_value(row["evidence_package_recall_10"])
    grounded = metric_value(row["grounded_score"])
    if row["track"] == "unique_exact":
        exact = metric_value(row["exact_recall_10"])
        return exact >= 0.95 and accepted >= 0.95 and package >= 0.95 and grounded >= 0.95
    return accepted >= 0.95 and package >= 0.95 and grounded >= 0.95


def review_issues(row: pd.Series) -> str:
    issues: list[str] = []
    if row["track"] == "unique_exact" and metric_value(row["exact_recall_10"]) < 0.95:
        issues.append("exact_below_0.95")
    if metric_value(row["accepted_recall_10"]) < 0.95:
        issues.append("accepted_below_0.95")
    if metric_value(row["evidence_package_recall_10"]) < 0.95:
        issues.append("package_below_0.95")
    if metric_value(row["grounded_score"]) < 0.95:
        issues.append("grounded_below_0.95")
    if metric_value(row["hallucination_risk_proxy"]) > 0.05:
        issues.append("ungated_risk_above_0.05")
    return "|".join(issues)


def safety_decision(row: pd.Series) -> str:
    return "ANSWER_ALLOWED" if strict_pass(row) else "REFUSE_OR_HUMAN_REVIEW"


def build_final_answer(question: str, top_rows: pd.DataFrame, accepted_hit: float, package_hit: float) -> str:
    roles = [r for r in top_rows["role"].astype(str).dropna().unique().tolist() if r]
    comps = [r for r in top_rows["component_family"].astype(str).dropna().unique().tolist() if r and r != "no_component"]
    snippets = []
    for _, ev in top_rows.head(3).iterrows():
        snippets.append(
            f"[{ev['evidence_id']} | p.{ev.get('page_number', '')} | {ev.get('role', '')}] {clip(ev.get('text', ''), 180)}"
        )
    status = "sufficient" if accepted_hit >= 0.95 and package_hit >= 0.95 else "insufficient"
    return (
        f"Evidence-grounded package status: {status}. "
        f"Question: {clip(question, 180)} "
        f"Covered roles: {', '.join(roles[:8]) or 'not detected'}. "
        f"Component families: {', '.join(comps[:6]) or 'not detected'}. "
        f"Top evidence: " + " || ".join(snippets)
    )


def load_data():
    qa = pd.read_csv(ROOT / "04_results" / "v21" / "qa_warehouse_v21_full.csv").fillna("")
    pred = pd.read_csv(ROOT / "04_results" / "v21" / "v21_retrieval_predictions_full.csv").fillna("")
    gold = pd.read_csv(ROOT / "04_results" / "v21" / "v21_gold_sets.csv").fillna("")
    ev = pd.read_parquet(ROOT / "01_processed_corpus" / "v5" / "evidence_objects.parquet").fillna("")
    metrics = pd.read_csv(ROOT / "04_results" / "v21" / "v21_metrics_by_case_track.csv").fillna("")
    gate = pd.read_csv(ROOT / "04_results" / "v21" / "v21_safety_gate_metrics.csv").fillna("")
    final_pred = pred[pred["method"].eq(METHOD)].copy()
    merged = final_pred.merge(
        qa[
            [
                "question_id",
                "question",
                "gold_answer_brief",
                "gold_page",
                "gold_episode_id",
                "required_package_slots",
                "difficulty_level",
            ]
        ],
        on="question_id",
        how="left",
    ).merge(gold, on="question_id", how="left")
    return merged, ev, metrics, gate


def make_rows(merged: pd.DataFrame, ev: pd.DataFrame):
    ev_by_id = ev.set_index("evidence_id", drop=False)
    qa_rows: list[dict] = []
    chain_rows: list[dict] = []

    for _, row in merged.iterrows():
        top_ids = split_ids(row["top_evidence_ids"])[:10]
        top = ev_by_id.loc[[eid for eid in top_ids if eid in ev_by_id.index]].copy()
        if isinstance(top, pd.Series):
            top = top.to_frame().T
        exact_set = set(split_ids(row.get("exact_gold_evidence_ids", "")))
        accepted_set = set(split_ids(row.get("accepted_gold_evidence_ids", ""))) | exact_set
        accepted_pages = set(split_ids(row.get("accepted_gold_page_ids", "")))
        accepted_sections = set(split_ids(row.get("accepted_gold_section_ids", "")))
        accepted = metric_value(row["accepted_recall_10"])
        package = metric_value(row["evidence_package_recall_10"])
        grounded = metric_value(row["grounded_score"])
        precision = accepted
        recall = package
        f1 = harmonic(precision, recall)

        top_pages = ";".join([str(x) for x in top.get("page_id", pd.Series(dtype=str)).astype(str).tolist()])
        top_roles = ";".join([str(x) for x in top.get("role", pd.Series(dtype=str)).astype(str).tolist()])
        top_comps = ";".join([str(x) for x in top.get("component_family", pd.Series(dtype=str)).astype(str).tolist()])
        top_sections = ";".join([str(x) for x in top.get("detected_section_id", pd.Series(dtype=str)).astype(str).tolist()])
        chain_summary = " || ".join(
            [
                f"#{i+1} {evr['evidence_id']} p.{evr.get('page_number','')} {evr.get('role','')} {clip(evr.get('text',''), 160)}"
                for i, (_, evr) in enumerate(top.iterrows())
            ]
        )

        passed = strict_pass(row)
        qa_rows.append(
            {
                "query_id": row["query_id"],
                "question": row["question"],
                "gold_answer": row.get("gold_answer_brief", ""),
                "gold_pages": row.get("accepted_gold_page_ids", "") or row.get("gold_page", ""),
                "gold_episode": row.get("gold_episode_id", ""),
                "required_roles": row.get("required_package_slots", ""),
                "method": METHOD,
                "final_answer": build_final_answer(row["question"], top, accepted, package),
                "review_issues": review_issues(row),
                "strict_pass": bool(passed),
                "review_status": "PASS_GATED_READY" if passed else "NEEDS_HUMAN_REVIEW_OR_REFUSAL",
                "answer_accuracy_proxy": grounded,
                "answer_precision": precision,
                "answer_recall": recall,
                "answer_f1": f1,
                "evidence_chain_completeness": package,
                "aircraft_model": row["aircraft_model"],
                "split": row["split"],
                "track": row["track"],
                "query_type": row["query_type"],
                "difficulty_level": row.get("difficulty_level", ""),
                "exact_recall_10": row["exact_recall_10"],
                "accepted_recall_10": row["accepted_recall_10"],
                "package_recall_10": row["evidence_package_recall_10"],
                "page_recall_10": row["accepted_page_recall_10"],
                "section_recall_10": row["accepted_section_recall_10"],
                "grounded_score": row["grounded_score"],
                "hallucination_risk_proxy": row["hallucination_risk_proxy"],
                "safety_gate_decision": safety_decision(row),
                "top_evidence_ids": ";".join(top_ids),
                "top_evidence_pages": top_pages,
                "top_evidence_roles": top_roles,
                "top_evidence_components": top_comps,
                "top_evidence_sections": top_sections,
                "evidence_chain_top10": chain_summary,
                "exact_gold_evidence_ids": row.get("exact_gold_evidence_ids", ""),
                "accepted_gold_evidence_ids": row.get("accepted_gold_evidence_ids", ""),
                "accepted_gold_page_ids": row.get("accepted_gold_page_ids", ""),
                "accepted_gold_section_ids": row.get("accepted_gold_section_ids", ""),
            }
        )

        for rank, eid in enumerate(top_ids, start=1):
            if eid not in ev_by_id.index:
                continue
            evr = ev_by_id.loc[eid]
            chain_rows.append(
                {
                    "query_id": row["query_id"],
                    "aircraft_model": row["aircraft_model"],
                    "split": row["split"],
                    "track": row["track"],
                    "query_type": row["query_type"],
                    "rank": rank,
                    "evidence_id": eid,
                    "is_exact_gold": eid in exact_set,
                    "is_accepted_gold": eid in accepted_set,
                    "page_id": evr.get("page_id", ""),
                    "page_number": evr.get("page_number", ""),
                    "section_id": evr.get("detected_section_id", ""),
                    "section_title": evr.get("section_title", ""),
                    "role": evr.get("role", ""),
                    "role_group": evr.get("role_group", ""),
                    "component_family": evr.get("component_family", ""),
                    "components": evr.get("components", ""),
                    "granularity": evr.get("granularity", ""),
                    "file_name": evr.get("file_name", ""),
                    "text": evr.get("text", ""),
                    "accepted_page_match": evr.get("page_id", "") in accepted_pages,
                    "accepted_section_match": evr.get("detected_section_id", "") in accepted_sections,
                }
            )
    return pd.DataFrame(qa_rows), pd.DataFrame(chain_rows)


def safe_sheet_name(name: str) -> str:
    return name[:31]


def write_df(ws, df: pd.DataFrame, table_name: str | None = None):
    rows = [list(df.columns)] + df.where(pd.notnull(df), "").values.tolist()
    for r_idx, values in enumerate(rows, start=1):
        for c_idx, value in enumerate(values, start=1):
            ws.cell(r_idx, c_idx, excel_safe(value))
    if len(rows) > 1 and table_name:
        ref = f"A1:{get_column_letter(len(df.columns))}{len(rows)}"
        tab = Table(displayName=table_name, ref=ref)
        style = TableStyleInfo(name="TableStyleMedium2", showFirstColumn=False, showLastColumn=False, showRowStripes=True, showColumnStripes=False)
        tab.tableStyleInfo = style
        ws.add_table(tab)


def style_sheet(ws, freeze: str = "A2"):
    header_fill = PatternFill("solid", fgColor="1F4E78")
    header_font = Font(color="FFFFFF", bold=True)
    thin = Side(style="thin", color="D9E2F3")
    for cell in ws[1]:
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = Border(bottom=thin)
    ws.freeze_panes = freeze
    ws.auto_filter.ref = ws.dimensions
    for col in ws.columns:
        letter = get_column_letter(col[0].column)
        max_len = max(len(str(cell.value or "")) for cell in col[: min(len(col), 80)])
        if letter in {"B", "C", "H", "AI"}:
            width = 54
        elif max_len > 35:
            width = 32
        else:
            width = min(max(12, max_len + 2), 28)
        ws.column_dimensions[letter].width = width
    for row in ws.iter_rows(min_row=2, max_row=min(ws.max_row, 300)):
        for cell in row:
            cell.alignment = Alignment(vertical="top", wrap_text=True)
    ws.sheet_view.showGridLines = False


def add_summary_sheet(wb: Workbook, qa: pd.DataFrame, metrics: pd.DataFrame, gate: pd.DataFrame):
    ws = wb.active
    ws.title = "Summary"
    ws["A1"] = "FD-GRAG-RC V21 Complete QA Evidence Chain Workbook"
    ws["A1"].font = Font(size=16, bold=True, color="1F4E78")
    ws["A3"] = "Workbook purpose"
    ws["B3"] = "Complete QA retrieval evidence-chain audit for Boeing 737 and Cessna 172, following the O-M-flow template fields."
    ws["A4"] = "Claim boundary"
    ws["B4"] = "Document-grounded retrieval and gated generation-readiness proxy; not real maintenance correctness or safety certification."
    ws["A6"] = "Case"
    ws["B6"] = "Questions"
    ws["C6"] = "Strict pass rate"
    ws["D6"] = "Mean accepted R@10"
    ws["E6"] = "Mean package R@10"
    ws["F6"] = "Mean grounded"
    ws["G6"] = "Mean risk"
    start = 7
    for i, (case, sub) in enumerate(qa.groupby("aircraft_model"), start=start):
        ws.cell(i, 1, case)
        ws.cell(i, 2, len(sub))
        ws.cell(i, 3, float(sub["strict_pass"].mean()))
        ws.cell(i, 4, float(sub["accepted_recall_10"].mean()))
        ws.cell(i, 5, float(sub["package_recall_10"].mean()))
        ws.cell(i, 6, float(sub["grounded_score"].mean()))
        ws.cell(i, 7, float(sub["hallucination_risk_proxy"].mean()))
    ws["A11"] = "Safety gate"
    for c_idx, col in enumerate(gate.columns, start=1):
        ws.cell(12, c_idx, col)
    for r_idx, (_, row) in enumerate(gate.iterrows(), start=13):
        for c_idx, col in enumerate(gate.columns, start=1):
            ws.cell(r_idx, c_idx, row[col])
    for row in ws.iter_rows(min_row=1, max_row=20, max_col=8):
        for cell in row:
            cell.alignment = Alignment(vertical="top", wrap_text=True)
    for col in range(1, 9):
        ws.column_dimensions[get_column_letter(col)].width = 22 if col != 2 else 16
    for cell in ws[6]:
        cell.fill = PatternFill("solid", fgColor="1F4E78")
        cell.font = Font(color="FFFFFF", bold=True)
    for cell in ws[12]:
        cell.fill = PatternFill("solid", fgColor="5B9BD5")
        cell.font = Font(color="FFFFFF", bold=True)
    chart = BarChart()
    chart.title = "Case-level V21 QA Evidence Chain Quality"
    chart.y_axis.title = "Score"
    chart.x_axis.title = "Case"
    data = Reference(ws, min_col=3, max_col=6, min_row=6, max_row=start + qa["aircraft_model"].nunique() - 1)
    cats = Reference(ws, min_col=1, min_row=7, max_row=start + qa["aircraft_model"].nunique() - 1)
    chart.add_data(data, titles_from_data=True)
    chart.set_categories(cats)
    chart.height = 7
    chart.width = 16
    ws.add_chart(chart, "I3")
    ws.sheet_view.showGridLines = False


def build_workbook(path: Path, qa: pd.DataFrame, chain: pd.DataFrame, metrics: pd.DataFrame, gate: pd.DataFrame, template_headers: list[str]):
    wb = Workbook()
    add_summary_sheet(wb, qa, metrics, gate)

    for case, sheet_name in CASE_SHEET.items():
        sub = qa[qa["aircraft_model"].eq(case)].copy()
        if sub.empty:
            continue
        ws = wb.create_sheet(sheet_name)
        write_df(ws, sub[HEADERS], table_name=f"T_{sheet_name[:18]}")
        style_sheet(ws)
        # Color score columns.
        for col_name in ["answer_accuracy_proxy", "answer_precision", "answer_recall", "answer_f1", "evidence_chain_completeness", "grounded_score", "hallucination_risk_proxy"]:
            if col_name in sub.columns:
                idx = HEADERS.index(col_name) + 1
                col = get_column_letter(idx)
                ws.conditional_formatting.add(
                    f"{col}2:{col}{ws.max_row}",
                    ColorScaleRule(start_type="num", start_value=0, start_color="F8696B", mid_type="num", mid_value=0.85, mid_color="FFEB84", end_type="num", end_value=1, end_color="63BE7B"),
                )

    ws = wb.create_sheet("Evidence_Chain_Top10")
    chain_cols = [
        "query_id",
        "aircraft_model",
        "split",
        "track",
        "query_type",
        "rank",
        "evidence_id",
        "is_exact_gold",
        "is_accepted_gold",
        "page_id",
        "page_number",
        "section_id",
        "section_title",
        "role",
        "role_group",
        "component_family",
        "components",
        "granularity",
        "file_name",
        "text",
        "accepted_page_match",
        "accepted_section_match",
    ]
    write_df(ws, chain[chain_cols], table_name="T_EvidenceChain")
    style_sheet(ws)

    ws = wb.create_sheet("V21_Metrics")
    write_df(ws, metrics, table_name="T_V21Metrics")
    style_sheet(ws)

    ws = wb.create_sheet("Safety_Gate")
    write_df(ws, gate, table_name="T_SafetyGate")
    style_sheet(ws)

    ws = wb.create_sheet("Template_Reference")
    ws["A1"] = "Template copied from"
    ws["B1"] = str(TEMPLATE)
    ws["A3"] = "Original O-M-flow template headers"
    for i, h in enumerate(template_headers, start=1):
        ws.cell(4, i, h)
    for cell in ws[4]:
        cell.fill = PatternFill("solid", fgColor="1F4E78")
        cell.font = Font(color="FFFFFF", bold=True)
    ws.freeze_panes = "A5"
    ws.sheet_view.showGridLines = False
    for col in range(1, len(template_headers) + 1):
        ws.column_dimensions[get_column_letter(col)].width = min(max(14, len(str(ws.cell(4, col).value)) + 2), 32)

    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)


def load_template_headers() -> list[str]:
    if not TEMPLATE.exists():
        return TEMPLATE_HEADERS
    try:
        wb = load_workbook(TEMPLATE, read_only=True, data_only=True)
        ws = wb[wb.sheetnames[0]]
        return [str(c.value) for c in ws[1] if c.value]
    except Exception:
        return TEMPLATE_HEADERS


def verify(path: Path):
    wb = load_workbook(path, read_only=True, data_only=True)
    expected = {"Summary", "Evidence_Chain_Top10", "V21_Metrics", "Safety_Gate", "Template_Reference"}
    missing = expected - set(wb.sheetnames)
    if missing:
        raise RuntimeError(f"Missing sheets: {missing}")
    qa_sheets = [name for name in ["Boeing_QA_Results", "Cessna_QA_Results"] if name in wb.sheetnames]
    if not qa_sheets:
        raise RuntimeError("No case QA result sheet found.")
    for name in qa_sheets + ["Evidence_Chain_Top10"]:
        ws = wb[name]
        if ws.max_row < 2 or ws.max_column < 10:
            raise RuntimeError(f"Sheet {name} looks empty: {ws.max_row}x{ws.max_column}")


def main():
    merged, ev, metrics, gate = load_data()
    qa, chain = make_rows(merged, ev)
    template_headers = load_template_headers()

    combined = OUT_DIR / "FD_GRAG_RC_V21_complete_QA_evidence_chain_Boeing_Cessna.xlsx"
    build_workbook(combined, qa, chain, metrics, gate, template_headers)
    verify(combined)

    for case, folder in CASE_FOLDER.items():
        sub_qa = qa[qa["aircraft_model"].eq(case)].copy()
        sub_chain = chain[chain["aircraft_model"].eq(case)].copy()
        case_metrics = metrics[metrics["aircraft_model"].eq(case)].copy()
        case_path = CASE_DIR / folder / "7 测试" / f"{folder}_complete_QA_evidence_chain_V21.xlsx"
        build_workbook(case_path, sub_qa, sub_chain, case_metrics, gate, template_headers)
        verify(case_path)
        sub_qa.to_csv(case_path.with_suffix(".csv"), index=False, encoding="utf-8-sig")

    qa.to_csv(OUT_DIR / "FD_GRAG_RC_V21_complete_QA_evidence_chain_rows.csv", index=False, encoding="utf-8-sig")
    chain.to_csv(OUT_DIR / "FD_GRAG_RC_V21_complete_QA_evidence_chain_top10_long.csv", index=False, encoding="utf-8-sig")
    print(f"Combined workbook: {combined}")
    print(f"Rows: QA={len(qa)}, evidence-chain={len(chain)}")


if __name__ == "__main__":
    main()
