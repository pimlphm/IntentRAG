from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "04_results" / "aesr_v2"
REPORTS = ROOT / "07_reports" / "aesr_v2"


def truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1", "1.0"})


def main() -> None:
    required = [
        "cue_dependence_query.csv",
        "cue_free_access_query.csv",
        "all_access_query_results.csv",
        "cue_dependence_paired_bootstrap.csv",
        "natural_factor_proxies.csv",
        "intervention_cases.csv",
        "intervention_response.csv",
        "eligible_case_counts.csv",
        "action_authority_proxy.csv",
        "split_leakage_audit.csv",
        "split_overlap_audit.csv",
        "claim_result_map.csv",
        "scientific_claim_audit.csv",
    ]
    for name in required:
        path = RESULTS / name
        assert path.exists() and path.stat().st_size > 0, path

    cue = pd.read_csv(RESULTS / "cue_dependence_query.csv")
    access = pd.read_csv(RESULTS / "cue_free_access_query.csv")
    natural = pd.read_csv(RESULTS / "natural_factor_proxies.csv")
    cases = pd.read_csv(RESULTS / "intervention_cases.csv")
    responses = pd.read_csv(RESULTS / "intervention_response.csv")
    claims = pd.read_csv(RESULTS / "claim_result_map.csv")

    fair = cue[cue["condition"].eq("fair_cue_and_track_off")]
    assert fair["question_id"].nunique() == 214
    assert not truthy(fair["gold_used_for_retrieval_feature"]).any()
    assert not truthy(fair["gold_cue_used_for_retrieval_feature"]).any()
    assert not truthy(fair["locked_track_used_for_routing"]).any()
    assert not fair["query_text"].astype(str).str.contains("Evidence cue:|Include evidence cues:", regex=True).any()

    assert access["question_id"].nunique() == 214
    assert not truthy(access["gold_used_for_retrieval_feature"]).any()
    assert not truthy(access["locked_track_used_for_routing"]).any()
    assert access["condition"].nunique() == 7

    assert natural["question_id"].nunique() == 214
    assert truthy(natural["is_synthetic_query"]).all()
    assert not truthy(natural["is_synthetic_package"]).any()
    assert not truthy(natural["gold_used_for_feature"]).any()

    assert len(cases) > 0 and truthy(cases["is_synthetic_package"]).all()
    assert truthy(cases["synthetic_or_counterfactual"]).all()
    assert len(responses) > 0 and truthy(responses["is_synthetic_package"]).all()
    assert cases["base_evidence_ids"].notna().all()

    assert set(claims["claim_id"]) >= {"C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10"}
    assert (REPORTS / "run_manifest.json").exists()
    assert (REPORTS / "v21_evidence_state_diagnostic_audit.md").exists()

    print(
        {
            "status": "ok",
            "fair_queries": fair["question_id"].nunique(),
            "access_conditions": access["condition"].nunique(),
            "unmodified_packages": natural["question_id"].nunique(),
            "synthetic_interventions": len(cases),
        }
    )


if __name__ == "__main__":
    main()
