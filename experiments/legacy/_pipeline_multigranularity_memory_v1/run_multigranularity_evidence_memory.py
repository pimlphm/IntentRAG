"""Cue-free multi-granularity evidence-memory experiment.

This experiment adapts the *idea* of multi-granularity memory routing to an
aviation-maintenance evidence repository.  It does not reproduce MemGAS and it
does not treat entropy as admissibility.  Four source-grounded views are used:

* detail evidence (query-local / short-lived),
* page context (query-local / short-lived),
* section context (repository-persistent / longer-lived), and
* component profiles (repository-persistent / longer-lived).

The query-dependent entropy weights are used only for candidate ranking.  A
separate deterministic package assembler can reserve slots for visible-query
evidence obligations.  Gold, accepted-equivalent, benchmark-track, query-type,
and required-slot fields are never passed to ranking or assembly.  They enter
only the post-hoc audit function.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "multigranularity_memory_v1" / "multigranularity_memory_v1.json"
RESULTS = ROOT / "04_results" / "multigranularity_memory_v1"
TABLES = ROOT / "06_tables" / "multigranularity_memory_v1"
REPORTS = ROOT / "07_reports" / "multigranularity_memory_v1"
LOGS = ROOT / "08_logs" / "multigranularity_memory_v1"
for _path in (RESULTS, TABLES, REPORTS, LOGS):
    _path.mkdir(parents=True, exist_ok=True)


ACTION_TERMS = {
    "replace", "remove", "install", "adjust", "inspect", "check", "service",
    "clean", "repair", "troubleshoot", "test", "verify", "torque", "lubricate",
    "perform", "procedure", "maintenance", "should", "must", "required",
    "更换", "拆卸", "安装", "调整", "检查", "清洁", "维修", "排故", "测试",
    "确认", "执行", "措施", "步骤", "应该", "需要", "如何",
}
CAUSE_TERMS = {
    "cause", "caused", "because", "due", "probable", "reason", "failure",
    "fault", "malfunction", "trouble", "discrepancy", "why", "原因", "导致",
    "故障", "失效", "为什么", "机理",
}
SAFETY_TERMS = {
    "warning", "caution", "note", "limitation", "shall not", "must not",
    "do not", "not exceed", "maximum", "minimum", "警告", "注意", "限制",
    "不得", "禁止", "不应", "最大", "最小",
}
AUTHORITY_TERMS = {
    "manual", "service manual", "maintenance manual", "poh", "afm", "faa",
    "airworthiness directive", "service bulletin", "chapter", "section",
    "figure", "table", "page", "手册", "维修手册", "适航", "章节", "页",
}
CONDITION_TERMS = {
    "if", "when", "during", "before", "after", "while", "with", "without",
    "condition", "under", "如果", "当", "期间", "之前", "之后", "条件",
    "情况下", "发生时",
}
THRESHOLD_TERMS = {
    "psi", "psig", "rpm", "knot", "knots", "kt", "degree", "deg", "in-lb",
    "ft-lb", "volt", "amp", "gph", "inch", "torque", "limit", "tolerance",
    "threshold", "范围", "阈值", "扭矩", "压力", "温度", "速度", "角度",
}


def norm(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def norm_low(value: object) -> str:
    return norm(value).lower()


def tokenize(value: object) -> list[str]:
    return re.findall(r"[a-z][a-z0-9_\-']+|\d+(?:\.\d+)?|[\u4e00-\u9fff]", norm_low(value))


def contains_any(value: object, terms: Iterable[str]) -> bool:
    text = norm_low(value)
    return any(term.lower() in text for term in terms)


def split_gold_cue(question: object) -> tuple[str, str]:
    q = norm(question)
    match = re.search(r"\s+(?:Evidence cue|Include evidence cues):\s*(.+?)\.?$", q, flags=re.I)
    if not match:
        return q, ""
    return q[: match.start()].strip(), match.group(1).strip().rstrip(".")


def parse_ids(value: object) -> list[str]:
    return [item for item in str(value or "").split(";") if item]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def approx_tokens(value: object) -> int:
    text = norm(value)
    latin = len(re.findall(r"[A-Za-z0-9_\-]+", text))
    cjk = len(re.findall(r"[\u4e00-\u9fff]", text))
    return max(1, latin + math.ceil(cjk / 1.5))


def normalized_entropy(scores: np.ndarray, candidate_k: int, temperature: float) -> float:
    finite = scores[np.isfinite(scores)]
    if len(finite) <= 1:
        return 1.0
    k = min(candidate_k, len(finite))
    top = np.partition(finite, len(finite) - k)[-k:]
    if float(np.max(top) - np.min(top)) <= 1e-12:
        return 1.0
    probs = softmax((top - float(np.max(top))) / max(temperature, 1e-6))
    entropy = float(-(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum())
    return float(entropy / math.log(len(probs))) if len(probs) > 1 else 0.0


def rank_score(scores: np.ndarray, eligible: np.ndarray, pool_k: int, rrf_constant: float) -> np.ndarray:
    out = np.zeros(len(scores), dtype=float)
    idx = np.flatnonzero(eligible & np.isfinite(scores))
    if not len(idx):
        return out
    k = min(pool_k, len(idx))
    local = scores[idx]
    top_local = np.argpartition(-local, k - 1)[:k]
    top_local = top_local[np.argsort(-local[top_local], kind="stable")]
    ranked = idx[top_local]
    for rank, global_idx in enumerate(ranked, start=1):
        out[int(global_idx)] = 1.0 / (rrf_constant + rank)
    return out


def visible_obligations(query: str) -> dict[str, bool]:
    return {
        "action": contains_any(query, ACTION_TERMS),
        "cause": contains_any(query, CAUSE_TERMS),
        "condition": contains_any(query, CONDITION_TERMS),
        "warning": contains_any(query, SAFETY_TERMS),
        "threshold": contains_any(query, THRESHOLD_TERMS),
    }


def evidence_role_flags(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    role = frame["role"].fillna("").astype(str).str.lower()
    text = (
        frame["text"].fillna("").astype(str)
        + " " + frame["authority_terms"].fillna("").astype(str)
        + " " + frame["condition_terms"].fillna("").astype(str)
        + " " + frame["warning_terms"].fillna("").astype(str)
    ).str.lower()
    return {
        "action": (role.isin(["action", "inspection", "replacement"]) | text.map(lambda x: contains_any(x, ACTION_TERMS))).to_numpy(),
        "cause": (role.isin(["confirmed_cause", "excluded_cause", "evidence"]) | text.map(lambda x: contains_any(x, CAUSE_TERMS))).to_numpy(),
        "condition": (role.eq("condition") | frame["condition_terms"].fillna("").astype(str).str.len().gt(0) | text.map(lambda x: contains_any(x, CONDITION_TERMS))).to_numpy(),
        "warning": (role.isin(["warning", "limitation", "note"]) | frame["warning_terms"].fillna("").astype(str).str.len().gt(0) | text.map(lambda x: contains_any(x, SAFETY_TERMS))).to_numpy(),
        "threshold": (role.eq("threshold") | text.map(lambda x: contains_any(x, THRESHOLD_TERMS))).to_numpy(),
        "authority": (role.eq("authority") | frame["authority_terms"].fillna("").astype(str).str.len().gt(0) | text.map(lambda x: contains_any(x, AUTHORITY_TERMS))).to_numpy(),
    }


@dataclass
class ViewIndex:
    name: str
    frame: pd.DataFrame
    id_col: str
    text_col: str
    word: TfidfVectorizer
    word_matrix: object
    char: TfidfVectorizer
    char_matrix: object

    @classmethod
    def build(cls, name: str, frame: pd.DataFrame, id_col: str, text_col: str, max_features_char: int) -> "ViewIndex":
        texts = frame[text_col].fillna("").map(norm).tolist()
        word = TfidfVectorizer(
            tokenizer=tokenize,
            token_pattern=None,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.98,
        )
        word_matrix = normalize(word.fit_transform(texts), copy=False)
        char = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=1,
            max_df=0.98,
            max_features=max_features_char,
        )
        char_matrix = normalize(char.fit_transform(texts), copy=False)
        return cls(name, frame.reset_index(drop=True), id_col, text_col, word, word_matrix, char, char_matrix)

    def score(self, query: str, aircraft: str, word_weight: float, char_weight: float) -> tuple[np.ndarray, np.ndarray]:
        qw = self.word.transform([query])
        qc = self.char.transform([query])
        scores = word_weight * (self.word_matrix @ qw.T).toarray().ravel()
        scores += char_weight * (self.char_matrix @ qc.T).toarray().ravel()
        eligible = self.frame["aircraft_model"].astype(str).eq(aircraft).to_numpy()
        scores[~eligible] = -np.inf
        return scores, eligible


def load_benchmarks() -> pd.DataFrame:
    files = {
        "calibration": ROOT / "04_results" / "v21" / "qa_warehouse_v21_calibration.csv",
        "validation": ROOT / "04_results" / "v21" / "qa_warehouse_v21_validation.csv",
        "locked_blind_test": ROOT / "04_results" / "v21" / "qa_warehouse_v21_locked_blind_test.csv",
    }
    parts: list[pd.DataFrame] = []
    for split_name, path in files.items():
        frame = pd.read_csv(path).fillna("")
        frame["evaluation_split"] = split_name
        frame["base_question"] = frame["question"].map(lambda x: split_gold_cue(x)[0])
        frame["gold_cue_removed"] = frame["question"].map(lambda x: bool(split_gold_cue(x)[1]))
        parts.append(frame)
    benchmark = pd.concat(parts, ignore_index=True)
    assert benchmark["base_question"].str.contains(r"Evidence cue|Include evidence cues", case=False, regex=True).sum() == 0
    return benchmark


def load_labels() -> dict[str, dict[str, set[str]]]:
    frame = pd.read_csv(ROOT / "04_results" / "v21" / "v21_gold_sets.csv").fillna("")
    labels: dict[str, dict[str, set[str]]] = {}
    for _, row in frame.iterrows():
        labels[str(row["question_id"])] = {
            "exact": set(parse_ids(row["exact_gold_evidence_ids"])),
            "accepted": set(parse_ids(row["accepted_gold_evidence_ids"])),
            "pages": set(parse_ids(row["accepted_gold_page_ids"])),
            "sections": set(parse_ids(row["accepted_gold_section_ids"])),
        }
    return labels


def make_views(config: dict) -> tuple[pd.DataFrame, dict[str, ViewIndex]]:
    corpus = ROOT / "01_processed_corpus" / "v5"
    evidence = pd.read_parquet(corpus / "evidence_objects.parquet").fillna("")
    for col in ("is_boilerplate", "is_index_or_toc", "image_only_or_no_text"):
        evidence[col] = pd.to_numeric(evidence[col], errors="coerce").fillna(0).astype(int)
    usable = evidence[
        evidence["is_boilerplate"].eq(0)
        & evidence["is_index_or_toc"].eq(0)
        & evidence["image_only_or_no_text"].eq(0)
        & evidence["text"].astype(str).str.len().gt(0)
    ].copy()
    detail = usable[usable["granularity"].isin(config["detail_granularities"])].copy()
    pages = pd.read_parquet(corpus / "pages.parquet").fillna("")
    sections = pd.read_parquet(corpus / "sections.parquet").fillna("")
    profiles = usable[usable["granularity"].eq("component_profile")].copy()
    views = {
        "detail": ViewIndex.build("detail", detail, "evidence_id", "text", int(config["max_features_char"])),
        "page": ViewIndex.build("page", pages, "page_id", "page_text", int(config["max_features_char"])),
        "section": ViewIndex.build("section", sections, "detected_section_id", "section_text", int(config["max_features_char"])),
        "profile": ViewIndex.build("profile", profiles, "evidence_id", "text", int(config["max_features_char"])),
    }
    return detail.reset_index(drop=True), views


def broadcast_parent_scores(
    detail: pd.DataFrame,
    view: ViewIndex,
    node_scores: np.ndarray,
    parent_col: str,
    key_col: str,
    *,
    use_component_family: bool = False,
) -> np.ndarray:
    if use_component_family:
        best: dict[tuple[str, str], float] = defaultdict(float)
        for idx, row in view.frame.iterrows():
            key = (str(row["aircraft_model"]), str(row["component_family"]))
            if np.isfinite(node_scores[idx]):
                best[key] = max(best[key], float(node_scores[idx]))
        return np.asarray(
            [best.get((str(row.aircraft_model), str(row.component_family)), 0.0) for row in detail.itertuples()],
            dtype=float,
        )
    mapping = {
        str(view.frame.iloc[idx][key_col]): float(node_scores[idx])
        for idx in range(len(view.frame))
        if np.isfinite(node_scores[idx])
    }
    return detail[parent_col].astype(str).map(mapping).fillna(0.0).to_numpy(dtype=float)


def view_scores_for_query(
    config: dict,
    detail: pd.DataFrame,
    views: dict[str, ViewIndex],
    query: str,
    aircraft: str,
) -> tuple[dict[str, np.ndarray], dict[str, float], dict[str, float]]:
    mix = config["score_mix"]
    raw_node_scores: dict[str, np.ndarray] = {}
    entropies: dict[str, float] = {}
    for name, view in views.items():
        scores, _eligible = view.score(query, aircraft, float(mix["word"]), float(mix["character"]))
        raw_node_scores[name] = scores
        entropies[name] = normalized_entropy(
            scores,
            int(config["entropy_candidate_k"]),
            float(config["entropy_temperature"]),
        )

    mapped = {
        "detail": raw_node_scores["detail"],
        "page": broadcast_parent_scores(detail, views["page"], raw_node_scores["page"], "page_id", "page_id"),
        "section": broadcast_parent_scores(
            detail,
            views["section"],
            raw_node_scores["section"],
            "detected_section_id",
            "detected_section_id",
        ),
        "profile": broadcast_parent_scores(
            detail,
            views["profile"],
            raw_node_scores["profile"],
            "component_family",
            "evidence_id",
            use_component_family=True,
        ),
    }
    eligible = detail["aircraft_model"].astype(str).eq(aircraft).to_numpy()
    rank_mapped = {
        name: rank_score(
            values,
            eligible,
            int(config["rank_pool_k"]),
            float(config["rrf_constant"]),
        )
        for name, values in mapped.items()
    }
    priors = {name: float(value) for name, value in config["dynamic_prior_weights"].items()}
    raw_weights = {
        name: priors[name] * (0.05 + max(0.0, 1.0 - entropies[name]))
        for name in rank_mapped
    }
    total = sum(raw_weights.values()) or 1.0
    dynamic_weights = {name: raw_weights[name] / total for name in raw_weights}
    return rank_mapped, entropies, dynamic_weights


def combine_scores(rank_mapped: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    combined = np.zeros(len(next(iter(rank_mapped.values()))), dtype=float)
    for name, values in rank_mapped.items():
        combined += float(weights[name]) * values
    return combined


def top_indices(scores: np.ndarray, eligible: np.ndarray, k: int) -> list[int]:
    idx = np.flatnonzero(eligible & np.isfinite(scores))
    if not len(idx):
        return []
    k = min(k, len(idx))
    local = scores[idx]
    part = np.argpartition(-local, k - 1)[:k]
    part = part[np.argsort(-local[part], kind="stable")]
    return [int(x) for x in idx[part]]


def admissibility_assemble(
    detail: pd.DataFrame,
    combined_scores: np.ndarray,
    eligible: np.ndarray,
    query: str,
    top_k: int,
    reserved_top: int,
    flags: dict[str, np.ndarray],
) -> list[int]:
    ranked = top_indices(combined_scores, eligible, min(len(detail), max(300, top_k * 30)))
    selected: list[int] = ranked[: min(reserved_top, top_k)]
    obligations = visible_obligations(query)
    required: list[str] = []
    if obligations["action"]:
        required.extend(["action", "authority", "condition", "warning"])
    if obligations["cause"]:
        required.append("cause")
    if obligations["threshold"]:
        required.append("threshold")
    if obligations["condition"] and "condition" not in required:
        required.append("condition")
    if obligations["warning"] and "warning" not in required:
        required.append("warning")
    for requirement in required:
        if any(bool(flags[requirement][idx]) for idx in selected):
            continue
        candidate = next((idx for idx in ranked if idx not in selected and bool(flags[requirement][idx])), None)
        if candidate is not None and len(selected) < top_k:
            selected.append(int(candidate))
    seen_sections: defaultdict[str, int] = defaultdict(int)
    seen_pages: defaultdict[str, int] = defaultdict(int)
    for idx in selected:
        seen_sections[str(detail.iloc[idx]["detected_section_id"])] += 1
        seen_pages[str(detail.iloc[idx]["page_id"])] += 1
    for idx in ranked:
        if idx in selected:
            continue
        section_id = str(detail.iloc[idx]["detected_section_id"])
        page_id = str(detail.iloc[idx]["page_id"])
        if seen_pages[page_id] >= 2 and len(selected) < top_k - 1:
            continue
        if seen_sections[section_id] >= 4 and len(selected) < top_k - 1:
            continue
        selected.append(int(idx))
        seen_sections[section_id] += 1
        seen_pages[page_id] += 1
        if len(selected) >= top_k:
            break
    if len(selected) < top_k:
        for idx in ranked:
            if idx not in selected:
                selected.append(int(idx))
            if len(selected) >= top_k:
                break
    return selected[:top_k]


def package_diagnostics(package: pd.DataFrame, labels: dict[str, set[str]], track: str) -> dict[str, float]:
    ids = set(package["evidence_id"].astype(str))
    pages = set(package["page_id"].astype(str))
    sections = set(package["detected_section_id"].astype(str))
    exact = float(bool(ids & labels["exact"]))
    accepted = float(bool(ids & labels["accepted"]))
    page_hit = float(bool(pages & labels["pages"]))
    section_hit = float(bool(sections & labels["sections"]))
    roles = set(package["role"].fillna("").astype(str).str.lower())
    authority = float("authority" in roles or package["authority_terms"].fillna("").astype(str).str.len().gt(0).any())
    condition = float("condition" in roles or package["condition_terms"].fillna("").astype(str).str.len().gt(0).any())
    warning = float(bool(roles & {"warning", "limitation", "note"}) or package["warning_terms"].fillna("").astype(str).str.len().gt(0).any())
    action = float(bool(roles & {"action", "inspection", "replacement"}))
    provenance = float(
        package["document_id"].fillna("").astype(str).str.len().gt(0).mean()
        * package["page_id"].fillna("").astype(str).str.len().gt(0).mean()
    )
    token_count = int(sum(approx_tokens(text) for text in package["text"]))
    return {
        "offline_exact_hit_10": math.nan if track == "broad_package" else exact,
        "offline_accepted_hit_10": accepted,
        "offline_page_hit_10": page_hit,
        "offline_section_hit_10": section_hit,
        "package_action_present": action,
        "package_authority_present": authority,
        "package_condition_present": condition,
        "package_warning_present": warning,
        "package_provenance_valid_rate": provenance,
        "package_role_diversity": float(len(roles)),
        "package_page_diversity": float(package["page_id"].nunique()),
        "package_section_diversity": float(package["detected_section_id"].nunique()),
        "package_approx_tokens": float(token_count),
    }


def run_experiment(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    benchmark = load_benchmarks()
    labels = load_labels()
    detail, views = make_views(config)
    role_flags = evidence_role_flags(detail)
    fixed_weights = {name: float(value) for name, value in config["fixed_view_weights"].items()}
    rows: list[dict] = []
    router_rows: list[dict] = []
    for position, row in benchmark.iterrows():
        query = str(row["base_question"])
        aircraft = str(row["aircraft_model"])
        rank_mapped, entropies, dynamic_weights = view_scores_for_query(config, detail, views, query, aircraft)
        eligible = detail["aircraft_model"].astype(str).eq(aircraft).to_numpy()
        flat_scores = rank_mapped["detail"]
        short_scores = combine_scores(
            rank_mapped,
            {"detail": 0.5, "page": 0.5, "section": 0.0, "profile": 0.0},
        )
        long_scores = combine_scores(
            rank_mapped,
            {"detail": 0.0, "page": 0.0, "section": 0.5, "profile": 0.5},
        )
        fixed_scores = combine_scores(rank_mapped, fixed_weights)
        entropy_scores = combine_scores(rank_mapped, dynamic_weights)
        method_indices = {
            "flat_detail_hybrid": top_indices(flat_scores, eligible, int(config["top_k"])),
            "page_only": top_indices(rank_mapped["page"], eligible, int(config["top_k"])),
            "section_only": top_indices(rank_mapped["section"], eligible, int(config["top_k"])),
            "profile_only": top_indices(rank_mapped["profile"], eligible, int(config["top_k"])),
            "short_memory_fixed": top_indices(short_scores, eligible, int(config["top_k"])),
            "long_memory_fixed": top_indices(long_scores, eligible, int(config["top_k"])),
            "fixed_multigranularity": top_indices(fixed_scores, eligible, int(config["top_k"])),
            "entropy_multigranularity": top_indices(entropy_scores, eligible, int(config["top_k"])),
            "entropy_admissibility_assembly": admissibility_assemble(
                detail,
                entropy_scores,
                eligible,
                query,
                int(config["top_k"]),
                int(config["admissibility_reserved_top"]),
                role_flags,
            ),
        }
        method_memory_weights = {
            "flat_detail_hybrid": (1.0, 0.0),
            "page_only": (1.0, 0.0),
            "section_only": (0.0, 1.0),
            "profile_only": (0.0, 1.0),
            "short_memory_fixed": (1.0, 0.0),
            "long_memory_fixed": (0.0, 1.0),
            "fixed_multigranularity": (
                float(fixed_weights["detail"] + fixed_weights["page"]),
                float(fixed_weights["section"] + fixed_weights["profile"]),
            ),
            "entropy_multigranularity": (
                float(dynamic_weights["detail"] + dynamic_weights["page"]),
                float(dynamic_weights["section"] + dynamic_weights["profile"]),
            ),
            "entropy_admissibility_assembly": (
                float(dynamic_weights["detail"] + dynamic_weights["page"]),
                float(dynamic_weights["section"] + dynamic_weights["profile"]),
            ),
        }
        qid = str(row["question_id"])
        for method, indices in method_indices.items():
            package = detail.iloc[indices].copy()
            short_weight, long_weight = method_memory_weights[method]
            rows.append(
                {
                    "experiment_id": config["experiment_id"],
                    "question_id": qid,
                    "evaluation_split": row["evaluation_split"],
                    "aircraft_model": aircraft,
                    "benchmark_track_posthoc": row["track"],
                    "benchmark_query_type_posthoc": row["query_type"],
                    "method": method,
                    "query_text": query,
                    "gold_cue_removed": bool(row["gold_cue_removed"]),
                    "is_synthetic_query": bool(row.get("synthetic_generated", True)),
                    "gold_used_for_ranking": False,
                    "accepted_used_for_ranking": False,
                    "locked_track_used_for_ranking": False,
                    "query_type_used_for_ranking": False,
                    "required_slots_used_for_ranking": False,
                    "top_evidence_ids": ";".join(package["evidence_id"].astype(str)),
                    "short_term_weight": short_weight,
                    "long_term_weight": long_weight,
                    **package_diagnostics(package, labels[qid], str(row["track"])),
                }
            )
        router_rows.append(
            {
                "experiment_id": config["experiment_id"],
                "question_id": qid,
                "evaluation_split": row["evaluation_split"],
                "aircraft_model": aircraft,
                "query_text": query,
                "detail_entropy": entropies["detail"],
                "page_entropy": entropies["page"],
                "section_entropy": entropies["section"],
                "profile_entropy": entropies["profile"],
                "detail_weight": dynamic_weights["detail"],
                "page_weight": dynamic_weights["page"],
                "section_weight": dynamic_weights["section"],
                "profile_weight": dynamic_weights["profile"],
                "short_term_weight": dynamic_weights["detail"] + dynamic_weights["page"],
                "long_term_weight": dynamic_weights["section"] + dynamic_weights["profile"],
                **{f"visible_{key}_obligation": value for key, value in visible_obligations(query).items()},
                "gold_used_for_router": False,
                "locked_track_used_for_router": False,
                "query_type_used_for_router": False,
            }
        )
        if (position + 1) % 25 == 0:
            print(f"processed {position + 1}/{len(benchmark)} queries", flush=True)
    query_results = pd.DataFrame(rows)
    router = pd.DataFrame(router_rows)
    return query_results, router


def add_existing_test_reference(query_results: pd.DataFrame) -> pd.DataFrame:
    path = ROOT / "04_results" / "aesr_v2" / "cue_free_access_query.csv"
    existing = pd.read_csv(path).fillna("")
    existing = existing[existing["condition"].eq("cue_off_intentrag")].copy()
    if existing.empty:
        return query_results
    rows: list[dict] = []
    for _, row in existing.iterrows():
        rows.append(
            {
                "experiment_id": query_results["experiment_id"].iloc[0],
                "question_id": row["question_id"],
                "evaluation_split": "locked_blind_test",
                "aircraft_model": row["aircraft_model"],
                "benchmark_track_posthoc": row["track"],
                "benchmark_query_type_posthoc": row["query_type"],
                "method": "existing_cue_free_intentrag_reference",
                "query_text": row["base_question"],
                "gold_cue_removed": True,
                "is_synthetic_query": True,
                "gold_used_for_ranking": False,
                "accepted_used_for_ranking": False,
                "locked_track_used_for_ranking": False,
                "query_type_used_for_ranking": False,
                "required_slots_used_for_ranking": False,
                "top_evidence_ids": row["top_evidence_ids"],
                "short_term_weight": math.nan,
                "long_term_weight": math.nan,
                "offline_exact_hit_10": row["offline_exact_recall_10"] if row["offline_exact_recall_10"] != "" else math.nan,
                "offline_accepted_hit_10": float(row["offline_accepted_recall_10"]),
                "offline_page_hit_10": float(row["offline_page_recall_10"]),
                "offline_section_hit_10": float(row["offline_section_recall_10"]),
                "package_action_present": math.nan,
                "package_authority_present": math.nan,
                "package_condition_present": math.nan,
                "package_warning_present": math.nan,
                "package_provenance_valid_rate": math.nan,
                "package_role_diversity": math.nan,
                "package_page_diversity": math.nan,
                "package_section_diversity": math.nan,
                "package_approx_tokens": math.nan,
            }
        )
    return pd.concat([query_results, pd.DataFrame(rows)], ignore_index=True)


def summarize(query_results: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "offline_exact_hit_10", "offline_accepted_hit_10", "offline_page_hit_10",
        "offline_section_hit_10", "package_action_present", "package_authority_present",
        "package_condition_present", "package_warning_present", "package_provenance_valid_rate",
        "package_role_diversity", "package_page_diversity", "package_section_diversity",
        "package_approx_tokens", "short_term_weight", "long_term_weight",
    ]
    summary = (
        query_results.groupby(["evaluation_split", "aircraft_model", "method"], dropna=False)[metrics]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary.columns = [
        "_".join([str(part) for part in column if str(part)]) if isinstance(column, tuple) else str(column)
        for column in summary.columns
    ]
    all_rows = (
        query_results.groupby(["evaluation_split", "method"], dropna=False)[metrics]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    all_rows.insert(1, "aircraft_model", "All")
    all_rows.columns = [
        "_".join([str(part) for part in column if str(part)]) if isinstance(column, tuple) else str(column)
        for column in all_rows.columns
    ]
    return pd.concat([summary, all_rows], ignore_index=True, sort=False)


def paired_bootstrap(query_results: pd.DataFrame, config: dict) -> pd.DataFrame:
    test = query_results[query_results["evaluation_split"].eq("locked_blind_test")].copy()
    pivot = test.pivot_table(index="question_id", columns="method", values="offline_accepted_hit_10", aggfunc="first")
    comparisons = [
        ("entropy_admissibility_assembly", "flat_detail_hybrid"),
        ("entropy_admissibility_assembly", "fixed_multigranularity"),
        ("entropy_admissibility_assembly", "entropy_multigranularity"),
        ("entropy_admissibility_assembly", "existing_cue_free_intentrag_reference"),
    ]
    rng = np.random.default_rng(int(config["seed"]))
    rows: list[dict] = []
    for proposed, baseline in comparisons:
        if proposed not in pivot or baseline not in pivot:
            continue
        pair = pivot[[proposed, baseline]].dropna()
        delta = (pair[proposed] - pair[baseline]).to_numpy(dtype=float)
        if not len(delta):
            continue
        samples = np.empty(int(config["bootstrap_samples"]), dtype=float)
        for b in range(len(samples)):
            idx = rng.integers(0, len(delta), len(delta))
            samples[b] = float(delta[idx].mean())
        rows.append(
            {
                "metric": "offline_accepted_hit_10",
                "proposed": proposed,
                "baseline": baseline,
                "n": len(delta),
                "mean_delta": float(delta.mean()),
                "ci_low": float(np.quantile(samples, 0.025)),
                "ci_high": float(np.quantile(samples, 0.975)),
                "wins": int((delta > 0).sum()),
                "ties": int((delta == 0).sum()),
                "losses": int((delta < 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def write_reports(config: dict, query_results: pd.DataFrame, router: pd.DataFrame, summary: pd.DataFrame, paired: pd.DataFrame, runtime_s: float) -> None:
    test_summary = summary[
        summary["evaluation_split"].eq("locked_blind_test")
        & summary["aircraft_model"].eq("All")
    ].copy()
    fields = [
        "method", "offline_accepted_hit_10_mean", "offline_exact_hit_10_mean",
        "offline_page_hit_10_mean", "offline_section_hit_10_mean",
        "package_approx_tokens_mean", "short_term_weight_mean", "long_term_weight_mean",
    ]
    fields = [field for field in fields if field in test_summary]
    test_summary[fields].to_csv(TABLES / "paper_multigranularity_memory_summary.csv", index=False)

    claim_rows: list[dict] = []
    for _, row in test_summary.iterrows():
        claim_rows.append(
            {
                "claim_id": f"MGM-{len(claim_rows)+1:02d}",
                "manuscript_claim": f"On the source-derived cue-free audit, {row['method']} obtained accepted-hit@10={float(row['offline_accepted_hit_10_mean']):.4f}.",
                "support_status": "supported as a construction-coupled post-hoc retrieval proxy",
                "population": "214 locked-blind synthetic source-derived queries",
                "metric": "offline_accepted_hit_10",
                "value": float(row["offline_accepted_hit_10_mean"]),
                "result_file": str((RESULTS / "query_results.csv").relative_to(ROOT)),
                "gold_dependency": "locked accepted sets used after retrieval only",
                "synthetic_only": True,
                "caveat": "not natural-user accuracy, claim entailment, safety, or action authorization",
            }
        )
    claim_rows.extend(
        [
            {
                "claim_id": "MGM-B01",
                "manuscript_claim": "The entropy router uses only retrieval-distribution concentration and visible aircraft scope.",
                "support_status": "implemented",
                "population": "all 476 source-derived queries",
                "metric": "gold_used_for_router",
                "value": 0,
                "result_file": str((RESULTS / "router_weights.csv").relative_to(ROOT)),
                "gold_dependency": "none before post-hoc audit",
                "synthetic_only": True,
                "caveat": "entropy is not safety confidence or mutual information",
            },
            {
                "claim_id": "MGM-B02",
                "manuscript_claim": "Long-term memory denotes persistent section/profile views, not a natural maintenance-event timeline.",
                "support_status": "conceptual boundary",
                "population": "not applicable",
                "metric": "not applicable",
                "value": "not applicable",
                "result_file": str((CONFIG_PATH).relative_to(ROOT)),
                "gold_dependency": "none",
                "synthetic_only": False,
                "caveat": "no tail-number history or expert revision-supersession labels",
            },
        ]
    )
    pd.DataFrame(claim_rows).to_csv(RESULTS / "claim_result_map.csv", index=False)

    visibility = pd.DataFrame(
        [
            ("base_question after cue removal", "retrieval/router/assembly", True, False, "used"),
            ("aircraft model stated in query", "retrieval/router/assembly", True, False, "hard scope filter"),
            ("document/page/section/component metadata", "retrieval/router/assembly", True, False, "used as source-grounded views"),
            ("track", "benchmark construction", False, True, "offline stratification only"),
            ("query_type", "benchmark construction", False, True, "offline stratification only"),
            ("required_package_slots", "benchmark construction", False, True, "not used"),
            ("gold/accepted evidence IDs", "locked audit", False, True, "post-hoc scoring only"),
            ("gold page/section IDs", "locked audit", False, True, "post-hoc scoring only"),
        ],
        columns=["field", "stage", "online_visible", "locked_or_gold", "treatment"],
    )
    visibility.to_csv(RESULTS / "retrieval_visibility_audit.csv", index=False)

    manifest = {
        "experiment_id": config["experiment_id"],
        "timestamp_utc": pd.Timestamp.utcnow().isoformat(),
        "runtime_seconds": runtime_s,
        "config": config,
        "inputs": {
            str(path.relative_to(ROOT)): {"sha256": sha256(path), "bytes": path.stat().st_size}
            for path in [
                CONFIG_PATH,
                ROOT / "01_processed_corpus" / "v5" / "evidence_objects.parquet",
                ROOT / "01_processed_corpus" / "v5" / "pages.parquet",
                ROOT / "01_processed_corpus" / "v5" / "sections.parquet",
                ROOT / "04_results" / "v21" / "v21_gold_sets.csv",
                ROOT / "04_results" / "v21" / "qa_warehouse_v21_calibration.csv",
                ROOT / "04_results" / "v21" / "qa_warehouse_v21_validation.csv",
                ROOT / "04_results" / "v21" / "qa_warehouse_v21_locked_blind_test.csv",
            ]
        },
        "outputs": [
            str((RESULTS / name).relative_to(ROOT))
            for name in ["query_results.csv", "router_weights.csv", "summary.csv", "paired_bootstrap.csv", "claim_result_map.csv", "retrieval_visibility_audit.csv"]
        ],
        "scientific_boundary": config["interpretation_boundary"],
    }
    (REPORTS / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    report_lines = [
        "# Multi-granularity evidence-memory v1",
        "",
        "## What was implemented",
        "",
        "Four source-grounded evidence views were indexed: detail and page as query-local working evidence, and section and component profile as persistent repository memory. A normalized-entropy router weights the four views. A separate deterministic assembler reserves evidence roles only from visible query language. Gold, accepted sets, benchmark track, query type, and required slots are excluded from ranking and assembly.",
        "",
        "## Locked-blind results",
        "",
        test_summary[fields].to_markdown(index=False) if fields else "No test rows.",
        "",
        "## Paired comparisons",
        "",
        paired.to_markdown(index=False) if not paired.empty else "No paired comparisons.",
        "",
        "## Interpretation boundary",
        "",
        config["interpretation_boundary"],
        "",
        "The benchmark questions are synthetic and source-derived. Accepted/page/section sets are construction-coupled and are used only post hoc. Results do not establish natural-user performance, expert action authorization, semantic mutual information, calibrated refusal, or an online long-term memory backend.",
    ]
    (REPORTS / "scientific_report.md").write_text("\n".join(report_lines), encoding="utf-8")


def main() -> None:
    start = time.perf_counter()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    query_results, router = run_experiment(config)
    query_results = add_existing_test_reference(query_results)
    summary = summarize(query_results)
    paired = paired_bootstrap(query_results, config)
    query_results.to_csv(RESULTS / "query_results.csv", index=False)
    router.to_csv(RESULTS / "router_weights.csv", index=False)
    summary.to_csv(RESULTS / "summary.csv", index=False)
    paired.to_csv(RESULTS / "paired_bootstrap.csv", index=False)
    runtime_s = time.perf_counter() - start
    write_reports(config, query_results, router, summary, paired, runtime_s)
    print(f"completed {config['experiment_id']} in {runtime_s:.1f}s")


if __name__ == "__main__":
    main()
