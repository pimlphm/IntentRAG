from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports"
OUT_DIR = ROOT / "04_results" / "method_advantage_audit"
CASE_IDS = ["Boeing_737", "Cessna_172"]

HIGHER_BETTER = [
    "exact_recall_10",
    "accepted_recall_10",
    "evidence_package_recall_10",
    "accepted_page_recall_10",
    "accepted_section_recall_10",
    "grounded_score",
    "role_coverage_proxy",
]
LOWER_BETTER = ["hallucination_risk_proxy", "unsupported_generation_risk_proxy"]


def canonical_method(method: str) -> str:
    m = str(method)
    if "FD-GRAG-RC" in m:
        return "FD-GRAG-RC"
    mapping = {
        "BM25/word": "BM25",
        "TFIDF-char": "Char TF-IDF",
        "SVD semantic surrogate": "SVD surrogate",
        "Hybrid lexical-semantic": "Hybrid",
        "Hybrid lexical": "Hybrid",
        "Parent-section retrieval": "Parent-section",
        "Component-profile retrieval": "Component-profile",
        "Controlled GraphRAG": "GraphRAG",
        "Causal-graph informed RAG": "Causal-graph",
        "Component-graph informed RAG": "Component-graph",
        "Evidence-flow baseline": "Evidence-flow",
        "GaRAGe-style relevance RAG": "GaRAGe-rel",
        "GaRAGe-style attribution RAG": "GaRAGe-attr",
        "GaRAGe-style deflective RAG": "GaRAGe-deflect",
    }
    return mapping.get(m, m)


def table_dir(case_id: str) -> Path:
    case_root = ROOT / "Cases" / case_id
    test = next(p for p in case_root.iterdir() if p.is_dir() and p.name.startswith("7"))
    return test / "sota_evidence_chain_tables"


def read_sources() -> list[tuple[str, pd.DataFrame, list[str]]]:
    sources: list[tuple[str, pd.DataFrame, list[str]]] = []
    for case_id in CASE_IDS:
        td = table_dir(case_id)
        case_method = pd.read_csv(td / f"{case_id}_sota_metrics_by_method.csv")
        case_method["case_id"] = case_id
        case_method["dataset_scope"] = "case_sota_by_method"
        sources.append((f"{case_id}:case_sota_by_method", case_method, ["case_id"]))
        case_query = pd.read_csv(td / f"{case_id}_sota_metrics_by_query_type.csv")
        case_query["case_id"] = case_id
        case_query["dataset_scope"] = "case_sota_by_query_type"
        sources.append((f"{case_id}:case_sota_by_query_type", case_query, ["case_id", "query_type"]))
        case_track = pd.read_csv(td / f"{case_id}_sota_metrics_by_track.csv")
        case_track["case_id"] = case_id
        case_track["dataset_scope"] = "case_sota_by_track"
        sources.append((f"{case_id}:case_sota_by_track", case_track, ["case_id", "track"]))
        noev = pd.read_csv(td / f"{case_id}_no_evidence_probe_metrics.csv")
        noev["case_id"] = case_id
        noev["dataset_scope"] = "case_no_evidence_probe"
        sources.append((f"{case_id}:case_no_evidence_probe", noev, ["case_id"]))

    for rel, label, group_cols in [
        ("04_results/v21/v21_metrics_by_case.csv", "global_v21_by_case", ["aircraft_model"]),
        ("04_results/v21/v21_metrics_by_case_track.csv", "global_v21_by_case_track", ["aircraft_model", "track"]),
        ("04_results/v21/v21_metrics_by_track.csv", "global_v21_by_track", ["track"]),
        ("06_tables/v21/table_v21_case_metrics.csv", "paper_table_case_metrics", ["aircraft_model"]),
        ("06_tables/v21/table_v21_case_track_metrics.csv", "paper_table_case_track_metrics", ["aircraft_model", "track"]),
    ]:
        path = ROOT / rel
        if path.exists():
            df = pd.read_csv(path)
            df["dataset_scope"] = label
            sources.append((label, df, group_cols))
    return sources


def classify_margin(metric: str, fd_value: float, best_baseline: float) -> tuple[float, float, str, str]:
    if metric in LOWER_BETTER:
        absolute = best_baseline - fd_value
        relative = absolute / max(abs(best_baseline), 1e-9)
        saturated = fd_value <= 1e-9 and best_baseline <= 1e-9
    else:
        absolute = fd_value - best_baseline
        relative = absolute / max(abs(best_baseline), 1e-9)
        saturated = fd_value >= 0.995 and best_baseline >= 0.995
    if saturated:
        return absolute, relative, "saturated_or_tied", "Do not claim large superiority; present as ceiling/tie."
    if absolute >= 0.05 or relative >= 0.20:
        return absolute, relative, "strong_advantage", "Can support a clear advantage claim for this scale."
    if absolute >= 0.01 or relative >= 0.05:
        return absolute, relative, "modest_advantage", "Use cautious language; avoid 'far superior'."
    if absolute >= -1e-9:
        return absolute, relative, "tie_or_negligible", "Not enough margin for an advantage claim."
    return absolute, relative, "baseline_better", "Must report as limitation or inspect failure mode."


def audit_source(label: str, df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    data = df.copy()
    if "method" not in data.columns:
        return pd.DataFrame()
    data["method_family"] = data["method"].map(canonical_method)
    metrics = [m for m in HIGHER_BETTER + LOWER_BETTER if m in data.columns]
    rows = []
    for keys, group in data.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        fd = group[group["method_family"].eq("FD-GRAG-RC")]
        base = group[~group["method_family"].eq("FD-GRAG-RC")]
        if fd.empty or base.empty:
            continue
        for metric in metrics:
            fd_vals = pd.to_numeric(fd[metric], errors="coerce").dropna()
            base_vals = pd.to_numeric(base[metric], errors="coerce").dropna()
            if fd_vals.empty or base_vals.empty:
                continue
            if metric in LOWER_BETTER:
                fd_idx = fd.loc[pd.to_numeric(fd[metric], errors="coerce").idxmin()]
                base_idx = base.loc[pd.to_numeric(base[metric], errors="coerce").idxmin()]
                fd_value = float(fd_idx[metric])
                best_baseline = float(base_idx[metric])
            else:
                fd_idx = fd.loc[pd.to_numeric(fd[metric], errors="coerce").idxmax()]
                base_idx = base.loc[pd.to_numeric(base[metric], errors="coerce").idxmax()]
                fd_value = float(fd_idx[metric])
                best_baseline = float(base_idx[metric])
            absolute, relative, status, language = classify_margin(metric, fd_value, best_baseline)
            row = {
                "source": label,
                "metric": metric,
                "fd_method": fd_idx["method"],
                "best_baseline_method": base_idx["method"],
                "fd_value": fd_value,
                "best_baseline_value": best_baseline,
                "absolute_margin": absolute,
                "relative_margin": relative,
                "advantage_status": status,
                "claim_language": language,
            }
            for col, value in zip(group_cols, keys):
                row[col] = value
            rows.append(row)
    return pd.DataFrame(rows)


def write_report(audit: pd.DataFrame) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    counts = audit["advantage_status"].value_counts().to_dict() if not audit.empty else {}
    risky = audit[audit["advantage_status"].isin(["saturated_or_tied", "tie_or_negligible", "baseline_better"])]
    strong = audit[audit["advantage_status"].eq("strong_advantage")]
    lines = [
        "# FD-GRAG-RC Cross-scale Method Advantage Audit",
        "",
        "This report checks whether existing benchmark and case-local result tables support strong superiority claims for FD-GRAG-RC across measurement scales. It does not alter any metric values.",
        "",
        "## Summary",
        "",
    ]
    for status in ["strong_advantage", "modest_advantage", "saturated_or_tied", "tie_or_negligible", "baseline_better"]:
        lines.append(f"- {status}: {counts.get(status, 0)} comparisons")
    lines += [
        "",
        "## Recommended Claim Policy",
        "",
        "- Use strong advantage language only for rows marked `strong_advantage`.",
        "- Use cautious language for `modest_advantage`.",
        "- For `saturated_or_tied`, describe the benchmark as ceiling-limited rather than claiming FD-GRAG-RC is far superior.",
        "- For `baseline_better`, report the limitation or omit the figure from method-advantage plots.",
        "- Do not modify benchmark CSV/JSON/NPY files to manufacture superiority.",
        "",
        "## Strong Advantage Scales",
        "",
    ]
    def md_table(df: pd.DataFrame, columns: list[str]) -> str:
        if df.empty:
            return ""
        out = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
        for _, row in df[columns].iterrows():
            cells = []
            for col in columns:
                value = row[col]
                if isinstance(value, float):
                    value = f"{value:.4f}"
                cells.append(str(value).replace("|", "/"))
            out.append("| " + " | ".join(cells) + " |")
        return "\n".join(out)

    if strong.empty:
        lines.append("No comparison met the strong-advantage threshold.")
    else:
        keep = ["source", "metric", "fd_value", "best_baseline_value", "absolute_margin", "relative_margin", "best_baseline_method"]
        lines.append(md_table(strong.head(40), keep))
    lines += [
        "",
        "## Scales That Should Not Be Used As 'Far Superior' Evidence",
        "",
    ]
    if risky.empty:
        lines.append("No tied or baseline-better comparisons found.")
    else:
        keep = ["source", "metric", "fd_value", "best_baseline_value", "absolute_margin", "advantage_status", "best_baseline_method"]
        lines.append(md_table(risky.head(80), keep))
    lines += [
        "",
        "## Interpretation",
        "",
        "The existing locked benchmark contains saturated metrics where multiple methods reach the same ceiling. These rows are useful for showing that FD-GRAG-RC is competitive and safe under the benchmark, but they do not support a claim that it is far superior on that scale. The strongest defensible story should emphasize structured evidence-package behavior, risk control, FMECA/role alignment, and query families where the margin is visible.",
    ]
    (REPORT_DIR / "method_advantage_across_scales_audit.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = [audit_source(label, df, group_cols) for label, df, group_cols in read_sources()]
    audit = pd.concat([f for f in frames if not f.empty], ignore_index=True) if frames else pd.DataFrame()
    audit_path = OUT_DIR / "method_advantage_across_scales_audit.csv"
    audit.to_csv(audit_path, index=False, encoding="utf-8-sig")
    summary_path = OUT_DIR / "method_advantage_status_summary.csv"
    if not audit.empty:
        summary = audit.groupby(["source", "advantage_status"]).size().reset_index(name="n")
        summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    write_report(audit)
    print(
        {
            "comparisons": int(len(audit)),
            "strong_advantage": int((audit["advantage_status"] == "strong_advantage").sum()) if not audit.empty else 0,
            "audit_csv": str(audit_path),
            "report": str(REPORT_DIR / "method_advantage_across_scales_audit.md"),
        }
    )


if __name__ == "__main__":
    main()
