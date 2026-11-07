"""Offline final review of eight mechanical-QA forests and two native-KV plots.

Keep the raw entry-point source linter result, including unresolved-import export
failures. Audit the resolved local drawing/export dependency chain separately.
No model requests or network operations are performed.
"""
from __future__ import annotations
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import fitz
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
FIG = BASE/"figures_qa_kv"
OUT = FIG/"qa/final_qa_kv"
SKILL = Path("${NATURE_FIGURE_SKILL}")


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def run(script, argv, dest):
    result = subprocess.run([sys.executable, str(SKILL/script), *map(str, argv)],
        capture_output=True, text=True, encoding="utf-8")
    dest.write_text(result.stdout, encoding="utf-8")
    if result.stderr:
        dest.with_suffix(".stderr.txt").write_text(result.stderr, encoding="utf-8")
    return {"exit": result.returncode, "output": dest.name}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source = BASE/"scripts/make_qa_kv_figures.py"
    dependency = BASE/"scripts/make_expanded_figures.py"
    text, dep = source.read_text(encoding="utf-8"), dependency.read_text(encoding="utf-8")
    selected = [n for n in ast.parse(dep).body if isinstance(n, (ast.ClassDef, ast.FunctionDef))
        and n.name in {"Exporter", "axes_arrows", "forest", "make_summary"}]
    effective = OUT/"resolved_local_source_chain.py"
    effective.write_text(text+"\n\n"+"\n\n".join(ast.get_source_segment(dep, n) for n in selected)+"\n", encoding="utf-8")
    sources = {
        "raw_entry_point": run("validate_figure.py", [source, "--backend", "python", "--json"], OUT/"source_raw.json"),
        "resolved_dependency_chain": run("validate_figure.py", [effective, "--backend", "python", "--json"], OUT/"source_resolved.json")}
    manifest = json.loads((FIG/"figure_manifest.json").read_text(encoding="utf-8"))
    names = [r["figure"] for r in manifest["figures"]]
    assert len(names) == 10
    assert sum("answer_" in n for n in names) == 8
    assert not any("blinded_answer" in n for n in names)
    assert not list(FIG.glob("*blinded_answer*.pdf")), "Empty semantic graphs must not be silently delivered"
    plots = []
    for name in names:
        p = FIG/(name+".pdf")
        glyph = run("audit_pdf_text.py", [p, "--min-pt", "5", "--json"], OUT/(name+".glyph.json"))
        collision = run("audit_figure_collisions.py", [p, "--json-out", OUT/(name+".collision.json")], OUT/(name+".collision.txt"))
        with fitz.open(p) as document:
            page = document[0]
            spans = [s for b in page.get_text("dict")["blocks"] if "lines" in b for l in b["lines"] for s in l["spans"]]
            width = page.rect.width/72*25.4
            minimum = min(s["size"] for s in spans)
            page.get_pixmap(matrix=fitz.Matrix(183/width*150/72, 183/width*150/72), alpha=False).save(OUT/(name+".183mm.png"))
            plots.append({"figure": name, "sha256_pdf": digest(p), "glyph": glyph, "collision": collision,
                "minimum_native_text_pt": minimum, "minimum_text_at_183mm_pt": minimum*183/width,
                "native_width_mm": width, "native_height_mm": page.rect.height/72*25.4,
                "editable_text": bool(spans), "n_axes": 1, "title_in_image": False})

    q = pd.read_csv(BASE/"results/qa_expanded_main/qa_answer_results.csv")
    methods = sorted(q.method.unique())
    expected_ids = {origin: set(g.question_id) for origin, g in q.groupby("origin")}
    coverage = []
    for origin, expected in expected_ids.items():
        assert len(expected) == 32
        for method in methods:
            g = q[q.origin.eq(origin) & q.method.eq(method)]
            coverage.append({"origin": origin, "method": method, "n_expected": 32,
                "n_rows": len(g), "n_question_ids": g.question_id.nunique(), "n_question_clusters": g.cluster_id.nunique(),
                "n_missing_ids": len(expected-set(g.question_id)),
                "n_duplicate_question_ids": int(g.question_id.duplicated().sum()),
                "n_undefined_quote_scores": int(g.quote_supported_fraction.isna().sum()),
                "n_undefined_json_answer_scores": int(g.structured_parse_ok.isna().sum()),
                "n_quote_recognized_answers": int(g.quote_present.eq(1).sum()),
                "mean_quote_score_all_answers": g.quote_supported_fraction.mean(),
                "mean_quote_precision_given_recognized_quote": g.loc[g.quote_present.eq(1), "quote_supported_fraction"].mean(),
                "n_valid_json_answer": int(g.structured_parse_ok.eq(1).sum()),
                "n_actual_calls": int(g.execution_kind.eq("actual_call").sum()),
                "n_identical_prompt_aliases": int(g.execution_kind.eq("duplicate_alias").sum()),
                "n_failed_executions": int(g.status.ne("ok").sum())})
    coverage = pd.DataFrame(coverage)
    coverage.to_csv(OUT/"qa_fold_method_coverage_and_quote_denominators.csv", index=False, encoding="utf-8-sig")
    assert len(q) == 2112 and len(methods) == 33 and len(coverage) == 66
    assert coverage.n_missing_ids.eq(0).all() and coverage.n_duplicate_question_ids.eq(0).all()
    assert coverage.n_rows.eq(32).all() and coverage.n_question_clusters.eq(32).all()
    assert q.loc[q.quote_present.eq(0), "quote_supported_fraction"].eq(0).all()
    failed_judge = BASE/"results/qa_expanded_main/blind_semantic_audit_failed_network_20261014/answer_audit_results.csv"
    j = pd.read_csv(failed_judge)
    valid_semantic = int(j.answer_correct_supported_judge_proxy.notna().sum())
    assert valid_semantic == 0
    report = {"status": "automated_pass_pending_manual_review", "source_audits": sources,
        "source_sha256": digest(source), "dependency_sha256": digest(dependency),
        "plots": plots, "n_figures": len(plots), "n_mechanical_qa_forests": 8, "n_native_kv_figures": 2,
        "n_glyph_blockers": sum(x["glyph"]["exit"] != 0 for x in plots),
        "n_collision_blockers": sum(x["collision"]["exit"] != 0 for x in plots),
        "qa_denominators": {"methods": len(methods), "unique_question_ids": q.question_id.nunique(),
            "questions_per_fold": {o: len(ids) for o, ids in expected_ids.items()}, "method_fold_rows": 32,
            "requested_cells": len(q), "missing_cells": int(coverage.n_missing_ids.sum()),
            "actual_generation_calls": int(q.execution_kind.eq("actual_call").sum()),
            "identical_prompt_alias_cells": int(q.execution_kind.eq("duplicate_alias").sum()),
            "failed_execution_cells": int(q.status.ne("ok").sum()),
            "recognized_quote_cells": int(q.quote_present.eq(1).sum()),
            "unrecognized_or_absent_quote_cells": int(q.quote_present.eq(0).sum()),
            "quote_mean_given_recognized_quote": q.loc[q.quote_present.eq(1), "quote_supported_fraction"].mean()},
        "valid_semantic_judgments": valid_semantic, "semantic_plot_omitted": True,
        "source_resolutions": {
            "raw_exports": "Entry-point linter does not resolve imported Exporter; raw SVG/PDF/raster FAIL results remain preserved. Resolved local chain is a distinct audit, not relabelled raw PASS.",
            "randomness": "Question-cluster bootstrap resamples observed values and KV horizontal jitter changes display coordinates only; no new observations simulated.",
            "tiff": "PDF/SVG and 600-dpi PNG are supplied for this draft; no exact journal export compliance is claimed.",
            "physical_size": "Use _word forests for 183/180-mm Word placement. Native original 12.8-in masters reduce to smaller type; 89-mm placement is not certified."},
        "scientific_boundary": "Quote score is all-answer mean with absent/unrecognized quote=0, not conditional precision or QA accuracy. Empty quote is allowed. JSON/answer parse does not require complete schema. Native KV counters measure computational reuse, not admissibility, expert correctness or isolated throughput."}
    (OUT/"final_audit_manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"figures": len(plots), "glyph_blockers": report["n_glyph_blockers"],
        "collision_blockers": report["n_collision_blockers"], "qa_cells": len(q),
        "missing_cells": report["qa_denominators"]["missing_cells"], "valid_semantic": valid_semantic}))


if __name__ == "__main__":
    main()
