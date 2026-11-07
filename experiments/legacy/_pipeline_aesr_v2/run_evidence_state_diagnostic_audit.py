"""Run the V21 Evidence-State Diagnostic Audit.

This experiment is deliberately narrower than a full AESR validation. It:

1. quantifies dependence on gold-derived evidence cues in the legacy V21
   benchmark;
2. evaluates cue-free lexical, multilingual dense, dense-plus-cross-encoder,
   and local original-order access settings;
3. describes retriever-visible evidence-factor proxies for cue-free packages;
4. applies clearly labelled synthetic package interventions and measures only
   directional proxy responses; and
5. emits a claim-to-result map so manuscript statements remain traceable.

Locked annotations are used only after retrieval for audit metrics. They never
enter cue-free retrieval, dense encoding, reranking, factor construction, or
intervention response calculation.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch
import transformers
from sklearn.preprocessing import normalize
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "aesr_v2" / "aesr_v2.json"
RESULTS = ROOT / "04_results" / "aesr_v2"
FIGURES = ROOT / "05_figures" / "aesr_v2"
TABLES = ROOT / "06_tables" / "aesr_v2"
REPORTS = ROOT / "07_reports" / "aesr_v2"
LOGS = ROOT / "08_logs" / "aesr_v2"
for directory in (RESULTS, FIGURES, TABLES, REPORTS, LOGS):
    directory.mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def norm(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def norm_low(value: object) -> str:
    return norm(value).lower()


def tokens(value: object) -> set[str]:
    return set(re.findall(r"[a-z][a-z0-9_\-']+|\d+(?:\.\d+)?|[\u4e00-\u9fff]", norm_low(value)))


def approx_tokens(value: object) -> int:
    text = norm(value)
    latin = re.findall(r"[A-Za-z0-9_\-]+", text)
    chinese = re.findall(r"[\u4e00-\u9fff]", text)
    return max(1, len(latin) + len(chinese))


def split_gold_cue(question: str) -> tuple[str, str]:
    q = norm(question)
    m = re.search(r"\s+(?:Evidence cue|Include evidence cues):\s*(.+?)\.?$", q, flags=re.I)
    if not m:
        return q, ""
    return q[: m.start()].strip(), m.group(1).strip().rstrip(".")


def load_v21_module():
    path = ROOT / "_pipeline_v21" / "p0_build_eval_v21.py"
    spec = importlib.util.spec_from_file_location("v21_source", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_set(value: object) -> set[str]:
    return {x for x in str(value or "").split(";") if x}


def load_labels() -> dict[str, dict[str, set[str]]]:
    df = pd.read_csv(ROOT / "04_results" / "v21" / "v21_gold_sets.csv").fillna("")
    labels: dict[str, dict[str, set[str]]] = {}
    for _, row in df.iterrows():
        labels[str(row["question_id"])] = {
            "exact": parse_set(row["exact_gold_evidence_ids"]),
            "accepted": parse_set(row["accepted_gold_evidence_ids"]),
            "pages": parse_set(row["accepted_gold_page_ids"]),
            "sections": parse_set(row["accepted_gold_section_ids"]),
        }
    return labels


def make_permuted_cues(bench: pd.DataFrame, seed: int) -> dict[str, str]:
    rng = np.random.default_rng(seed)
    out: dict[str, str] = {}
    work = bench.copy()
    work[["base_question", "gold_cue"]] = work["question"].apply(
        lambda x: pd.Series(split_gold_cue(str(x)))
    )
    for _, group in work.groupby(["split", "aircraft_model", "track"], sort=False):
        cues = group["gold_cue"].astype(str).tolist()
        if len(cues) > 1:
            order = rng.permutation(len(cues))
            if all(int(order[i]) == i for i in range(len(cues))):
                order = np.roll(order, 1)
            cues = [cues[int(i)] for i in order]
        for qid, cue in zip(group["question_id"].astype(str), cues):
            out[qid] = cue
    return out


def condition_question(row: pd.Series, condition: str, permuted: dict[str, str]) -> str:
    base, cue = split_gold_cue(str(row["question"]))
    if condition == "legacy_cue_on":
        return str(row["question"])
    if condition in {"cue_off_track_fixed", "fair_cue_and_track_off"}:
        return base
    if condition == "cue_only":
        return f"{row['aircraft_model']}. Evidence cue: {cue}." if cue else str(row["aircraft_model"])
    if condition == "cue_permuted":
        other = permuted.get(str(row["question_id"]), "")
        return f"{base} Evidence cue: {other}." if other else base
    raise ValueError(condition)


def v21_final_indices(
    v21,
    row: pd.Series,
    ev: pd.DataFrame,
    indexes,
    question: str,
    top_k: int,
    *,
    use_locked_track: bool,
) -> np.ndarray:
    texts, word, xw, char, xc = indexes
    scores = v21.query_scores(question, str(row["aircraft_model"]), ev, texts, word, xw, char, xc)
    if not use_locked_track:
        # Fair diagnostic path: route selection cannot see the locked
        # unique-versus-package benchmark track.  The same balanced route
        # recipe is used for every query.
        routes = [
            "Hybrid lexical",
            "TFIDF-char",
            "BM25/word",
            "FD-GRAG-RC-V21-package",
            "FD-GRAG-RC-V21-cue-route",
            "FD-GRAG-RC-V21-exact",
        ]
        quotas = [
            ("BM25/word", 2),
            ("TFIDF-char", 2),
            ("Hybrid lexical", 2),
            ("FD-GRAG-RC-V21-package", 2),
            ("FD-GRAG-RC-V21-cue-route", 2),
        ]
    elif str(row["track"]) == "broad_package":
        routes = [
            "FD-GRAG-RC-V21-package",
            "FD-GRAG-RC-V21-cue-route",
            "Hybrid lexical",
            "TFIDF-char",
            "BM25/word",
            "FD-GRAG-RC-V21-exact",
        ]
        quotas = [("FD-GRAG-RC-V21-package", 7), ("FD-GRAG-RC-V21-cue-route", 3)]
    else:
        routes = [
            "BM25/word",
            "FD-GRAG-RC-V21-cue-route",
            "FD-GRAG-RC-V21-package",
            "FD-GRAG-RC-V21-exact",
            "Hybrid lexical",
            "TFIDF-char",
        ]
        quotas = [("BM25/word", 3), ("FD-GRAG-RC-V21-cue-route", 4), ("FD-GRAG-RC-V21-package", 3)]
    return v21.route_union_indices(scores, routes, max(1000, top_k), quotas=quotas)[:top_k]


def audit_package(v21, row: pd.Series, top: pd.DataFrame, labels: dict[str, set[str]]) -> dict[str, float]:
    top_ids = set(top["evidence_id"].astype(str))
    exact_hit = float(bool(top_ids & labels["exact"]))
    accepted_hit = float(bool(top_ids & labels["accepted"]))
    hits = v21.slot_hits(top, row, labels)
    required = [x for x in str(row.get("required_package_slots", "")).split(";") if x]
    slot_coverage = float(np.mean([hits.get(x, False) for x in required])) if required else accepted_hit
    if str(row["track"]) == "unique_exact":
        slot_coverage = max(slot_coverage, accepted_hit)
    page_hit = float(hits["page"] or accepted_hit)
    section_hit = float(hits["section"] or accepted_hit)
    if str(row["track"]) == "broad_package":
        composite = 0.55 * accepted_hit + 0.15 * page_hit + 0.15 * section_hit + 0.15 * slot_coverage
        exact_value = math.nan
    else:
        composite = (
            0.35 * exact_hit
            + 0.30 * accepted_hit
            + 0.15 * page_hit
            + 0.05 * section_hit
            + 0.15 * slot_coverage
        )
        exact_value = exact_hit
    return {
        "offline_exact_recall_10": exact_value,
        "offline_accepted_recall_10": accepted_hit,
        "offline_page_recall_10": page_hit,
        "offline_section_recall_10": section_hit,
        "offline_locked_slot_coverage_10": slot_coverage,
        "offline_construction_composite": float(composite),
    }


def cue_audit(config: dict, v21, bench: pd.DataFrame, ev: pd.DataFrame, labels: dict) -> pd.DataFrame:
    indexes = v21.build_indexes(ev)
    permuted = make_permuted_cues(bench, int(config["seed"]))
    rows: list[dict] = []
    for _, row in bench.iterrows():
        qid = str(row["question_id"])
        base, cue = split_gold_cue(str(row["question"]))
        for condition in config["cue_conditions"]:
            query = condition_question(row, condition, permuted)
            start = time.perf_counter()
            uses_locked_track = condition != "fair_cue_and_track_off"
            idx = v21_final_indices(
                v21,
                row,
                ev,
                indexes,
                query,
                int(config["top_k"]),
                use_locked_track=uses_locked_track,
            )
            latency_ms = 1000.0 * (time.perf_counter() - start)
            top = ev.iloc[idx].copy()
            metrics = audit_package(v21, row, top, labels[qid])
            rows.append(
                {
                    "experiment_id": config["experiment_id"],
                    "question_id": qid,
                    "split": row["split"],
                    "aircraft_model": row["aircraft_model"],
                    "track": row["track"],
                    "query_type": row["query_type"],
                    "condition": condition,
                    "method": "IntentRAG_V21_fixed_routes",
                    "is_gold_derived_input": condition in {"legacy_cue_on", "cue_only", "cue_permuted"}
                    or uses_locked_track,
                    "is_synthetic": True,
                    "is_synthetic_query": True,
                    "is_synthetic_package": False,
                    "is_counterfactual_query": condition == "cue_permuted",
                    "gold_cue_used_for_retrieval_feature": condition
                    in {"legacy_cue_on", "cue_only", "cue_permuted"},
                    "locked_track_used_for_routing": uses_locked_track,
                    "gold_used_for_retrieval_feature": condition
                    in {"legacy_cue_on", "cue_only", "cue_permuted"}
                    or uses_locked_track,
                    "query_text": query,
                    "base_question": base,
                    "cue_char_count": len(cue),
                    "top_evidence_ids": ";".join(top["evidence_id"].astype(str)),
                    "retrieval_latency_ms": latency_ms,
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
    summed = (last_hidden * mask).sum(dim=1)
    return summed / mask.sum(dim=1).clamp(min=1e-9)


@torch.inference_mode()
def encode_texts(tokenizer, model, texts: list[str], batch_size: int, max_length: int) -> np.ndarray:
    vectors = []
    model.eval()
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        output = model(**encoded).last_hidden_state
        pooled = mean_pool(output, encoded["attention_mask"])
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        vectors.append(pooled.cpu().numpy().astype("float32"))
    return np.vstack(vectors)


@torch.inference_mode()
def cross_scores(tokenizer, model, query: str, docs: list[str], batch_size: int, max_length: int) -> np.ndarray:
    scores = []
    model.eval()
    for start in range(0, len(docs), batch_size):
        batch = docs[start : start + batch_size]
        encoded = tokenizer(
            [query] * len(batch),
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        logits = model(**encoded).logits.squeeze(-1)
        scores.extend(logits.cpu().numpy().astype(float).tolist())
    return np.asarray(scores, dtype=float)


def local_original_order_window(ev: pd.DataFrame, anchor_idx: int, token_budget: int, top_k: int) -> np.ndarray:
    anchor = ev.iloc[int(anchor_idx)]
    same_doc = ev[ev["document_id"].astype(str).eq(str(anchor["document_id"]))].copy()
    same_doc["_global_idx"] = same_doc.index.astype(int)
    anchor_page = float(anchor.get("page_number", 0) or 0)
    same_doc["_distance"] = (pd.to_numeric(same_doc["page_number"], errors="coerce").fillna(anchor_page) - anchor_page).abs()
    candidates = same_doc.sort_values(["_distance", "page_number", "chunk_id", "evidence_id"])
    chosen: list[int] = []
    used = 0
    for _, item in candidates.iterrows():
        cost = approx_tokens(item["text"])
        if chosen and used + cost > token_budget:
            continue
        chosen.append(int(item["_global_idx"]))
        used += cost
        if len(chosen) >= top_k or used >= token_budget:
            break
    chosen = sorted(
        chosen,
        key=lambda i: (
            float(ev.iloc[i].get("page_number", 0) or 0),
            str(ev.iloc[i].get("chunk_id", "")),
            str(ev.iloc[i].get("evidence_id", "")),
        ),
    )
    return np.asarray(chosen[:top_k], dtype=int)


def cue_free_access_audit(config: dict, v21, bench: pd.DataFrame, ev: pd.DataFrame, labels: dict) -> pd.DataFrame:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    local_only = bool(config.get("offline_models_only", True))
    dense_name = config["dense_model"]
    cross_name = config["cross_encoder_model"]
    dense_tokenizer = AutoTokenizer.from_pretrained(dense_name, local_files_only=local_only)
    dense_model = AutoModel.from_pretrained(dense_name, local_files_only=local_only)
    cross_tokenizer = AutoTokenizer.from_pretrained(cross_name, local_files_only=local_only)
    cross_model = AutoModelForSequenceClassification.from_pretrained(cross_name, local_files_only=local_only)

    base_queries = [split_gold_cue(str(x))[0] for x in bench["question"]]
    corpus_texts = ev["text"].map(norm).tolist()
    corpus_vec = encode_texts(
        dense_tokenizer,
        dense_model,
        corpus_texts,
        int(config["dense_batch_size"]),
        int(config["max_length"]),
    )
    query_vec = encode_texts(
        dense_tokenizer,
        dense_model,
        base_queries,
        int(config["dense_batch_size"]),
        int(config["max_length"]),
    )

    indexes = v21.build_indexes(ev)
    rows: list[dict] = []
    for qpos, (_, row) in enumerate(bench.iterrows()):
        qid = str(row["question_id"])
        query = base_queries[qpos]
        aircraft_mask = ev["aircraft_model"].astype(str).eq(str(row["aircraft_model"])).to_numpy()

        start = time.perf_counter()
        scores = v21.query_scores(query, str(row["aircraft_model"]), ev, *indexes)
        lexical_conditions = {
            "cue_off_word_tfidf": v21.top_indices(scores["BM25/word"], int(config["top_k"])),
            "cue_off_char_tfidf": v21.top_indices(scores["TFIDF-char"], int(config["top_k"])),
            "cue_off_hybrid_lexical": v21.top_indices(scores["Hybrid lexical"], int(config["top_k"])),
            "cue_off_intentrag": v21_final_indices(
                v21,
                row,
                ev,
                indexes,
                query,
                int(config["top_k"]),
                use_locked_track=False,
            ),
        }
        lexical_elapsed = 1000.0 * (time.perf_counter() - start)
        for condition, idx in lexical_conditions.items():
            top = ev.iloc[idx].copy()
            rows.append(
                {
                    "experiment_id": config["experiment_id"],
                    "question_id": qid,
                    "split": row["split"],
                    "aircraft_model": row["aircraft_model"],
                    "track": row["track"],
                    "query_type": row["query_type"],
                    "condition": condition,
                    "method": condition.removeprefix("cue_off_"),
                    "is_gold_derived_input": False,
                    "is_synthetic": True,
                    "is_synthetic_query": True,
                    "is_synthetic_package": False,
                    "gold_used_for_retrieval_feature": False,
                    "gold_cue_used_for_retrieval_feature": False,
                    "locked_track_used_for_routing": False,
                    "query_text": query,
                    "base_question": query,
                    "cue_char_count": 0,
                    "top_evidence_ids": ";".join(top["evidence_id"].astype(str)),
                    "retrieval_latency_ms": lexical_elapsed,
                    **audit_package(v21, row, top, labels[qid]),
                }
            )

        start = time.perf_counter()
        dense_scores = corpus_vec @ query_vec[qpos]
        dense_scores[~aircraft_mask] = -1e9
        dense_pool = np.argpartition(-dense_scores, min(int(config["dense_candidate_k"]), len(dense_scores)) - 1)[
            : int(config["dense_candidate_k"])
        ]
        dense_pool = dense_pool[np.argsort(-dense_scores[dense_pool])]
        dense_idx = dense_pool[: int(config["top_k"])]
        dense_elapsed = 1000.0 * (time.perf_counter() - start)
        top_dense = ev.iloc[dense_idx].copy()
        rows.append(
            {
                "experiment_id": config["experiment_id"],
                "question_id": qid,
                "split": row["split"],
                "aircraft_model": row["aircraft_model"],
                "track": row["track"],
                "query_type": row["query_type"],
                "condition": "cue_off_dense_multilingual",
                "method": "multilingual_MiniLM_dense",
                "is_gold_derived_input": False,
                "is_synthetic": True,
                "is_synthetic_query": True,
                "is_synthetic_package": False,
                "gold_used_for_retrieval_feature": False,
                "gold_cue_used_for_retrieval_feature": False,
                "locked_track_used_for_routing": False,
                "query_text": query,
                "base_question": query,
                "cue_char_count": 0,
                "top_evidence_ids": ";".join(top_dense["evidence_id"].astype(str)),
                "retrieval_latency_ms": dense_elapsed,
                **audit_package(v21, row, top_dense, labels[qid]),
            }
        )

        rerank_pool = dense_pool[: int(config["rerank_candidate_k"])]
        start = time.perf_counter()
        rerank_scores = cross_scores(
            cross_tokenizer,
            cross_model,
            query,
            ev.iloc[rerank_pool]["text"].map(norm).tolist(),
            int(config["rerank_batch_size"]),
            int(config["max_length"]),
        )
        rerank_idx = rerank_pool[np.argsort(-rerank_scores)[: int(config["top_k"])]]
        rerank_elapsed = 1000.0 * (time.perf_counter() - start)
        top_rerank = ev.iloc[rerank_idx].copy()
        rows.append(
            {
                "experiment_id": config["experiment_id"],
                "question_id": qid,
                "split": row["split"],
                "aircraft_model": row["aircraft_model"],
                "track": row["track"],
                "query_type": row["query_type"],
                "condition": "cue_off_dense_cross_encoder",
                "method": "multilingual_dense_plus_mMARCO_cross_encoder",
                "is_gold_derived_input": False,
                "is_synthetic": True,
                "is_synthetic_query": True,
                "is_synthetic_package": False,
                "gold_used_for_retrieval_feature": False,
                "gold_cue_used_for_retrieval_feature": False,
                "locked_track_used_for_routing": False,
                "query_text": query,
                "base_question": query,
                "cue_char_count": 0,
                "top_evidence_ids": ";".join(top_rerank["evidence_id"].astype(str)),
                "retrieval_latency_ms": dense_elapsed + rerank_elapsed,
                **audit_package(v21, row, top_rerank, labels[qid]),
            }
        )

        budget = sum(approx_tokens(x) for x in top_dense["text"])
        original_idx = local_original_order_window(
            ev,
            int(dense_pool[0]),
            max(1, budget),
            int(config["top_k"]),
        )
        top_original = ev.iloc[original_idx].copy()
        rows.append(
            {
                "experiment_id": config["experiment_id"],
                "question_id": qid,
                "split": row["split"],
                "aircraft_model": row["aircraft_model"],
                "track": row["track"],
                "query_type": row["query_type"],
                "condition": "cue_off_local_original_order_window",
                "method": "dense_anchor_local_original_order_proxy",
                "is_gold_derived_input": False,
                "is_synthetic": True,
                "is_synthetic_query": True,
                "is_synthetic_package": False,
                "gold_used_for_retrieval_feature": False,
                "gold_cue_used_for_retrieval_feature": False,
                "locked_track_used_for_routing": False,
                "query_text": query,
                "base_question": query,
                "cue_char_count": 0,
                "top_evidence_ids": ";".join(top_original["evidence_id"].astype(str)),
                "retrieval_latency_ms": dense_elapsed,
                "matched_token_budget": budget,
                **audit_package(v21, row, top_original, labels[qid]),
            }
        )
    return pd.DataFrame(rows)


def summarize_query_metrics(query_df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "offline_exact_recall_10",
        "offline_accepted_recall_10",
        "offline_page_recall_10",
        "offline_section_recall_10",
        "offline_locked_slot_coverage_10",
        "offline_construction_composite",
        "retrieval_latency_ms",
    ]
    groups = ["condition", "method", "aircraft_model", "track"]
    summary = query_df.groupby(groups, dropna=False)[metrics].mean(numeric_only=True).reset_index()
    counts = query_df.groupby(groups, dropna=False).size().reset_index(name="n")
    return summary.merge(counts, on=groups, how="left")


def paired_bootstrap(cue_df: pd.DataFrame, metric: str, seed: int, samples: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    comparisons = [
        ("legacy_cue_on", "cue_off_track_fixed"),
        ("legacy_cue_on", "fair_cue_and_track_off"),
        ("cue_off_track_fixed", "fair_cue_and_track_off"),
    ]
    for aircraft in ["All", *sorted(cue_df["aircraft_model"].unique())]:
        subset = cue_df if aircraft == "All" else cue_df[cue_df["aircraft_model"].eq(aircraft)]
        pivot = subset.pivot_table(index="question_id", columns="condition", values=metric, aggfunc="first")
        for left, right in comparisons:
            pair = pivot[[left, right]].dropna()
            if pair.empty:
                continue
            delta = (pair[right] - pair[left]).to_numpy(dtype=float)
            boots = []
            for _ in range(samples):
                sample = rng.choice(delta, size=len(delta), replace=True)
                boots.append(float(np.mean(sample)))
            rows.append(
                {
                    "aircraft_model": aircraft,
                    "metric": metric,
                    "reference": left,
                    "comparison": right,
                    "n": len(delta),
                    "reference_mean": float(pair[left].mean()),
                    "comparison_mean": float(pair[right].mean()),
                    "paired_delta": float(delta.mean()),
                    "ci_low": float(np.quantile(boots, 0.025)),
                    "ci_high": float(np.quantile(boots, 0.975)),
                    "wins": int(np.sum(delta > 0)),
                    "losses": int(np.sum(delta < 0)),
                    "ties": int(np.sum(delta == 0)),
                }
            )
    return pd.DataFrame(rows)


def has_text_or_field(df: pd.DataFrame, field: str, patterns: Iterable[str]) -> bool:
    if field in df and df[field].map(norm).ne("").any():
        return True
    text = " ".join(df.get("text", pd.Series([], dtype=str)).map(norm_low))
    return any(re.search(pattern, text, flags=re.I) for pattern in patterns)


def action_like_query(question: str) -> bool:
    return bool(
        re.search(
            r"\b(action|perform|replace|remove|install|repair|inspect|check|procedure|should|recommend)\b|"
            r"维修动作|执行什么|什么操作|如何处置|应当|建议|检查|更换|修理",
            norm_low(question),
        )
    )


def package_proxies(v21, row: pd.Series, package: pd.DataFrame) -> dict[str, float | int | bool]:
    if package.empty:
        return {
            "package_size": 0,
            "approx_token_count": 0,
            "role_diversity": 0,
            "role_diversity_ratio": 0.0,
            "scope_match_rate": 0.0,
            "component_match_proxy": 0.0,
            "condition_present": 0.0,
            "warning_present": 0.0,
            "authority_cue_present": 0.0,
            "action_or_procedure_present": 0.0,
            "provenance_valid_rate": 0.0,
            "revision_metadata_available": 0.0,
            "conflict_marker_proxy": 0.0,
            "redundancy_proxy": 0.0,
            "cross_page_indicator": 0.0,
            "source_diversity": 0,
            "page_diversity": 0,
            "action_like_query_proxy": float(action_like_query(str(row["question"]))),
            "action_authority_evidence_availability_proxy": 0.0,
        }
    roles = {norm_low(x) for x in package.get("role", pd.Series([], dtype=str)) if norm(x)}
    scope = package["aircraft_model"].astype(str).eq(str(row["aircraft_model"]))
    component = v21.component_from_question(str(row["question"]))
    if component:
        component_match = package.get("components", pd.Series([""] * len(package), index=package.index)).map(norm_low).str.contains(
            re.escape(component), regex=True
        ).mean()
    else:
        component_match = math.nan
    condition = has_text_or_field(
        package,
        "condition_terms",
        [r"\b(if|when|during|before|after|while|with|without|condition)\b", r"条件|当.+时|在.+情况下"],
    )
    warning = has_text_or_field(
        package,
        "warning_terms",
        [r"\b(warning|caution|limitation|shall not|must not|do not|not exceed)\b", r"警告|注意|不得|禁止|限制"],
    )
    authority = has_text_or_field(
        package,
        "authority_terms",
        [r"\b(maintenance manual|service manual|manual|poh|afm|faa|airworthiness directive|service bulletin)\b", r"手册|适航指令|服务通告"],
    )
    action = bool(
        roles & {"action", "inspection", "replacement", "procedure"}
        or has_text_or_field(
            package,
            "_missing_field_",
            [r"\b(replace|remove|install|adjust|inspect|check|service|repair|test|verify|perform)\b", r"更换|拆卸|安装|调整|检查|维修|测试|执行"],
        )
    )
    provenance_cols = [c for c in ["document_id", "page_id", "chunk_id", "evidence_id"] if c in package]
    provenance = np.ones(len(package), dtype=bool)
    for col in provenance_cols:
        provenance &= package[col].map(norm).ne("").to_numpy()
    revision = has_text_or_field(
        package,
        "revision",
        [r"\brev(?:ision)?\s*[A-Z0-9]+\b", r"\bR\d+\b", r"版本|改版"],
    )
    conflict = bool(package.get("synthetic_conflict", pd.Series(False, index=package.index)).astype(bool).any())
    texts = package["text"].map(tokens).tolist()
    similarities = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            union = texts[i] | texts[j]
            similarities.append(len(texts[i] & texts[j]) / len(union) if union else 0.0)
    redundancy = float(np.mean(similarities)) if similarities else 0.0
    page_diversity = package.get("page_id", pd.Series([], dtype=str)).map(norm).replace("", np.nan).nunique()
    source_diversity = package.get("document_id", pd.Series([], dtype=str)).map(norm).replace("", np.nan).nunique()
    action_like = action_like_query(str(row["question"]))
    action_availability = float(
        action_like
        and action
        and authority
        and bool(scope.all())
        and condition
        and bool(provenance.all())
        and not conflict
    )
    return {
        "package_size": int(len(package)),
        "approx_token_count": int(sum(approx_tokens(x) for x in package["text"])),
        "role_diversity": int(len(roles)),
        "role_diversity_ratio": float(len(roles) / max(1, len(package))),
        "scope_match_rate": float(scope.mean()),
        "component_match_proxy": float(component_match) if not math.isnan(component_match) else math.nan,
        "condition_present": float(condition),
        "warning_present": float(warning),
        "authority_cue_present": float(authority),
        "action_or_procedure_present": float(action),
        "provenance_valid_rate": float(provenance.mean()),
        "revision_metadata_available": float(revision),
        "conflict_marker_proxy": float(conflict),
        "redundancy_proxy": redundancy,
        "cross_page_indicator": float(page_diversity > 1),
        "source_diversity": int(source_diversity),
        "page_diversity": int(page_diversity),
        "action_like_query_proxy": float(action_like),
        "action_authority_evidence_availability_proxy": action_availability,
    }


def natural_factor_proxies(v21, bench: pd.DataFrame, ev: pd.DataFrame, access_df: pd.DataFrame) -> pd.DataFrame:
    by_id = ev.set_index("evidence_id", drop=False)
    bench_map = bench.set_index("question_id", drop=False)
    selected = access_df[access_df["condition"].eq("cue_off_intentrag")]
    rows = []
    for _, result in selected.iterrows():
        row = bench_map.loc[str(result["question_id"])].copy()
        row["question"] = str(result["base_question"])
        ids = [x for x in str(result["top_evidence_ids"]).split(";") if x in by_id.index]
        package = by_id.loc[ids].copy() if ids else ev.iloc[0:0].copy()
        proxies = package_proxies(v21, row, package)
        rows.append(
            {
                "experiment_id": result["experiment_id"],
                "question_id": result["question_id"],
                "aircraft_model": result["aircraft_model"],
                "track": result["track"],
                "query_type": result["query_type"],
                "is_synthetic": True,
                "is_synthetic_query": True,
                "is_synthetic_package": False,
                "gold_used_for_feature": False,
                "top_evidence_ids": result["top_evidence_ids"],
                **proxies,
            }
        )
    return pd.DataFrame(rows)


def scrub_patterns(text: str, patterns: Iterable[str]) -> str:
    out = str(text)
    for pattern in patterns:
        out = re.sub(pattern, " ", out, flags=re.I)
    return norm(out)


def descriptive_donor(ev: pd.DataFrame, row: pd.Series, used_ids: set[str]) -> pd.Series | None:
    candidates = ev[
        ev["aircraft_model"].astype(str).eq(str(row["aircraft_model"]))
        & ~ev["evidence_id"].astype(str).isin(used_ids)
        & ev.get("role", pd.Series("", index=ev.index)).astype(str).isin(
            ["symptom", "evidence", "confirmed_cause", "excluded_cause", "condition"]
        )
    ]
    if candidates.empty:
        return None
    qtok = tokens(row["question"])
    scored = []
    for idx, item in candidates.iterrows():
        dtok = tokens(item["text"])
        score = len(qtok & dtok) / max(1, len(qtok | dtok))
        scored.append((score, int(idx)))
    _, best = max(scored, key=lambda x: (x[0], -x[1]))
    return ev.loc[best].copy()


def run_interventions(config: dict, v21, bench: pd.DataFrame, ev: pd.DataFrame, natural: pd.DataFrame):
    by_id = ev.set_index("evidence_id", drop=False)
    bench_map = bench.set_index("question_id", drop=False)
    rng = np.random.default_rng(int(config["seed"]))
    proxy_columns = [
        "scope_match_rate",
        "condition_present",
        "warning_present",
        "authority_cue_present",
        "action_or_procedure_present",
        "provenance_valid_rate",
        "conflict_marker_proxy",
        "redundancy_proxy",
        "role_diversity_ratio",
        "action_authority_evidence_availability_proxy",
    ]
    target = {
        "authority_removal": ("authority_cue_present", -1),
        "condition_removal": ("condition_present", -1),
        "warning_removal": ("warning_present", -1),
        "provenance_masking": ("provenance_valid_rate", -1),
        "scope_swap": ("scope_match_rate", -1),
        "duplicate_injection": ("redundancy_proxy", 1),
        "conflict_injection": ("conflict_marker_proxy", 1),
        "procedural_to_descriptive_substitution": ("action_or_procedure_present", -1),
        "evidence_order_permutation": ("action_authority_evidence_availability_proxy", 0),
    }
    conceptual_state = {
        "authority_removal": "factual support may remain; action-evidence prerequisites become insufficient",
        "condition_removal": "Insufficient proxy signature",
        "warning_removal": "Insufficient warning-completeness proxy signature",
        "provenance_masking": "Insufficient auditability proxy signature",
        "scope_swap": "Inapplicable proxy signature",
        "duplicate_injection": "state should remain stable while redundancy increases",
        "conflict_injection": "Conflicted proxy signature",
        "procedural_to_descriptive_substitution": "descriptive evidence remains but action prerequisite is weakened",
        "evidence_order_permutation": "state signature should remain invariant",
    }
    cases = []
    responses = []
    eligibility = defaultdict(lambda: {"eligible": 0, "applied": 0})
    for _, natural_row in natural.iterrows():
        qid = str(natural_row["question_id"])
        row = bench_map.loc[qid].copy()
        base_question, _ = split_gold_cue(str(row["question"]))
        row["question"] = base_question
        ids = [x for x in str(natural_row["top_evidence_ids"]).split(";") if x in by_id.index]
        base = by_id.loc[ids].copy() if ids else ev.iloc[0:0].copy()
        if base.empty:
            continue
        base["synthetic_conflict"] = False
        before = package_proxies(v21, row, base)
        for intervention in config["interventions"]:
            target_proxy, direction = target[intervention]
            eligible = True
            changed_fields = []
            donor_ids: list[str] = []
            after_pkg = base.copy()

            if intervention == "authority_removal":
                eligible = bool(before["authority_cue_present"])
                if eligible:
                    after_pkg["authority_terms"] = ""
                    after_pkg["text"] = after_pkg["text"].map(
                        lambda x: scrub_patterns(
                            x,
                            [r"\b(maintenance manual|service manual|manual|poh|afm|faa|airworthiness directive|service bulletin)\b", r"手册|适航指令|服务通告"],
                        )
                    )
                    after_pkg.loc[after_pkg["role"].astype(str).eq("authority"), "role"] = "evidence"
                    changed_fields = ["authority_terms", "text", "role"]
            elif intervention == "condition_removal":
                eligible = bool(before["condition_present"])
                if eligible:
                    after_pkg["condition_terms"] = ""
                    after_pkg["text"] = after_pkg["text"].map(
                        lambda x: scrub_patterns(x, [r"\b(if|when|during|before|after|while|with|without|condition)\b", r"条件|当.+?时|在.+?情况下"])
                    )
                    after_pkg.loc[after_pkg["role"].astype(str).eq("condition"), "role"] = "evidence"
                    changed_fields = ["condition_terms", "text", "role"]
            elif intervention == "warning_removal":
                eligible = bool(before["warning_present"])
                if eligible:
                    after_pkg["warning_terms"] = ""
                    after_pkg["text"] = after_pkg["text"].map(
                        lambda x: scrub_patterns(x, [r"\b(warning|caution|limitation|shall not|must not|do not|not exceed)\b", r"警告|注意|不得|禁止|限制"])
                    )
                    after_pkg.loc[after_pkg["role"].astype(str).isin(["warning", "limitation", "note"]), "role"] = "evidence"
                    changed_fields = ["warning_terms", "text", "role"]
            elif intervention == "provenance_masking":
                for col in ["document_id", "page_id", "chunk_id", "evidence_id"]:
                    if col in after_pkg:
                        after_pkg[col] = ""
                if "page_number" in after_pkg:
                    after_pkg["page_number"] = np.nan
                changed_fields = ["document_id", "page_id", "chunk_id", "evidence_id", "page_number"]
            elif intervention == "scope_swap":
                other = "Cessna 172" if str(row["aircraft_model"]) == "Boeing 737" else "Boeing 737"
                after_pkg["aircraft_model"] = other
                changed_fields = ["aircraft_model"]
            elif intervention == "duplicate_injection":
                duplicate = after_pkg.iloc[[0]].copy()
                duplicate["evidence_id"] = duplicate["evidence_id"].astype(str) + "_SYNTHETIC_DUPLICATE"
                after_pkg = pd.concat([after_pkg, duplicate], ignore_index=True)
                changed_fields = ["package_membership"]
            elif intervention == "conflict_injection":
                conflict = after_pkg.iloc[[0]].copy()
                conflict["evidence_id"] = conflict["evidence_id"].astype(str) + "_SYNTHETIC_CONFLICT"
                conflict["synthetic_conflict"] = True
                conflict["text"] = conflict["text"].astype(str) + " [SYNTHETIC COUNTERFACTUAL REVISION: an incompatible superseding instruction is inserted for diagnostic testing.]"
                after_pkg = pd.concat([after_pkg, conflict], ignore_index=True)
                changed_fields = ["package_membership", "synthetic_conflict", "text"]
            elif intervention == "procedural_to_descriptive_substitution":
                action_mask = after_pkg["role"].astype(str).isin(["action", "inspection", "replacement", "procedure"])
                eligible = bool(action_mask.any())
                if eligible:
                    donor = descriptive_donor(ev, row, set(after_pkg["evidence_id"].astype(str)))
                    eligible = donor is not None
                    if donor is not None:
                        donor_ids = [str(donor["evidence_id"])]
                        for idx in after_pkg.index[action_mask]:
                            for col in after_pkg.columns:
                                if col in donor.index and col != "synthetic_conflict":
                                    after_pkg.at[idx, col] = donor[col]
                        changed_fields = ["package_membership", "role", "text", "provenance"]
            elif intervention == "evidence_order_permutation":
                order = rng.permutation(len(after_pkg))
                after_pkg = after_pkg.iloc[order].reset_index(drop=True)
                changed_fields = ["package_order"]
            else:
                raise ValueError(intervention)

            eligibility[(str(row["aircraft_model"]), intervention)]["eligible"] += int(eligible)
            if not eligible:
                continue
            eligibility[(str(row["aircraft_model"]), intervention)]["applied"] += 1
            after = package_proxies(v21, row, after_pkg)
            delta = float(after[target_proxy]) - float(before[target_proxy])
            if direction < 0:
                success = delta < 0
            elif direction > 0:
                success = delta > 0
            else:
                success = abs(delta) < 1e-12
            cases.append(
                {
                    "experiment_id": config["experiment_id"],
                    "question_id": qid,
                    "aircraft_model": row["aircraft_model"],
                    "query_type": row["query_type"],
                    "intervention": intervention,
                    "base_evidence_ids": ";".join(base["evidence_id"].astype(str)),
                    "donor_evidence_ids": ";".join(donor_ids),
                    "changed_fields": ";".join(changed_fields),
                    "target_proxy": target_proxy,
                    "expected_direction": direction,
                    "expected_conceptual_state": conceptual_state[intervention],
                    "is_synthetic": True,
                    "is_synthetic_query": True,
                    "is_synthetic_package": True,
                    "synthetic_or_counterfactual": True,
                    "gold_used_for_feature": False,
                    "seed": config["seed"],
                    "directional_success": bool(success),
                }
            )
            for proxy in proxy_columns:
                responses.append(
                    {
                        "experiment_id": config["experiment_id"],
                        "question_id": qid,
                        "aircraft_model": row["aircraft_model"],
                        "intervention": intervention,
                        "proxy": proxy,
                        "before": before[proxy],
                        "after": after[proxy],
                        "delta": float(after[proxy]) - float(before[proxy]),
                        "is_target_proxy": proxy == target_proxy,
                        "expected_direction": direction if proxy == target_proxy else math.nan,
                        "directional_success": bool(success) if proxy == target_proxy else math.nan,
                        "is_synthetic": True,
                        "is_synthetic_query": True,
                        "is_synthetic_package": True,
                        "gold_used_for_feature": False,
                    }
                )
    eligibility_rows = [
        {"aircraft_model": aircraft, "intervention": intervention, **values}
        for (aircraft, intervention), values in sorted(eligibility.items())
    ]
    return pd.DataFrame(cases), pd.DataFrame(responses), pd.DataFrame(eligibility_rows)


def write_split_leakage_audit() -> pd.DataFrame:
    rows = [
        ("question text before appended cue", "V21 benchmark", False, True, True, "retained"),
        ("Evidence cue / Include evidence cues", "gold evidence text", True, True, False, "removed from fair cue-and-track-off retrieval; retained only in diagnostic controls"),
        ("aircraft_model", "user-visible scope", False, True, True, "retained"),
        ("track", "benchmark construction", True, True, False, "removed from fair cue-and-track-off retrieval; retained only to stratify offline metrics"),
        ("query_type", "locked annotation", True, False, False, "offline stratification only"),
        ("required_package_slots", "locked annotation", True, False, False, "offline audit metric only"),
        ("gold and accepted evidence IDs", "locked annotation", True, False, False, "offline audit metric only"),
        ("gold page and section IDs", "locked annotation", True, False, False, "offline audit metric only"),
        ("split metadata", "benchmark construction", True, False, False, "offline stratification only"),
        ("evidence role/scope/authority/provenance fields", "document-derived repository", False, True, True, "used for factor proxies; quality not independently certified"),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "feature",
            "source",
            "gold_derived",
            "online_visible_in_legacy_v21",
            "online_visible_in_fair_audit",
            "aesr_v2_treatment",
        ],
    )


def save_md_table(df: pd.DataFrame, name: str) -> None:
    df.to_csv(TABLES / f"{name}.csv", index=False, encoding="utf-8-sig")
    try:
        text = df.to_markdown(index=False)
    except Exception:
        text = df.to_csv(index=False)
    (TABLES / f"{name}.md").write_text(text, encoding="utf-8")


def make_figures(query_df: pd.DataFrame, paired: pd.DataFrame, natural: pd.DataFrame, responses: pd.DataFrame) -> None:
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9})
    colors = {"Boeing 737": "#2F6B9A", "Cessna 172": "#D97732", "All": "#4E4E4E"}

    # Figure A: paired cue-off change with bootstrap intervals.
    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    data = paired[paired["metric"].eq("offline_accepted_recall_10")].copy()
    y = np.arange(len(data))
    ax.axvline(0, color="#777777", lw=1)
    for pos, (_, item) in enumerate(data.iterrows()):
        color = colors.get(str(item["aircraft_model"]), "#555555")
        ax.errorbar(
            item["paired_delta"],
            pos,
            xerr=[[item["paired_delta"] - item["ci_low"]], [item["ci_high"] - item["paired_delta"]]],
            fmt="o",
            color=color,
            capsize=3,
        )
    condition_labels = {
        "legacy_cue_on": "legacy cue + track",
        "cue_off_track_fixed": "cue removed; track retained",
        "fair_cue_and_track_off": "cue + track removed",
    }
    labels = [
        f"{item.aircraft_model}: {condition_labels.get(item.comparison, item.comparison)}\n"
        f"vs {condition_labels.get(item.reference, item.reference)}"
        for item in data.itertuples(index=False)
    ]
    ax.set_yticks(y, labels)
    ax.set_xlabel("Paired change in accepted R@10")
    ax.set_title("Gold-derived cue and locked-track dependence")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES / "figure_a_cue_dependence_paired.png", dpi=350)
    plt.close(fig)

    # Figure B: cue-free backend comparison.
    access = query_df[query_df["condition"].str.startswith("cue_off_")].copy()
    methods = [
        "cue_off_word_tfidf",
        "cue_off_hybrid_lexical",
        "cue_off_intentrag",
        "cue_off_dense_multilingual",
        "cue_off_dense_cross_encoder",
        "cue_off_local_original_order_window",
    ]
    means = access.groupby(["condition", "aircraft_model"])["offline_accepted_recall_10"].mean().unstack(fill_value=np.nan).reindex(methods)
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    x = np.arange(len(means))
    width = 0.36
    for offset, aircraft in zip([-width / 2, width / 2], ["Boeing 737", "Cessna 172"]):
        ax.bar(x + offset, means.get(aircraft), width, label=aircraft, color=colors[aircraft])
    ax.set_ylim(0, 1.05)
    access_labels = {
        "cue_off_word_tfidf": "Word\nTF-IDF",
        "cue_off_hybrid_lexical": "Hybrid\nlexical",
        "cue_off_intentrag": "Balanced\nIntentRAG",
        "cue_off_dense_multilingual": "Multilingual\ndense",
        "cue_off_dense_cross_encoder": "Dense +\ncross-encoder",
        "cue_off_local_original_order_window": "Original-order\nwindow proxy",
    }
    ax.set_xticks(x, [access_labels.get(m, m) for m in means.index])
    ax.set_ylabel("Accepted R@10 (locked post-hoc audit)")
    ax.set_title("Cue-free access settings; no generator or state-classification claim")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES / "figure_b_cue_free_access_comparison.png", dpi=350)
    plt.close(fig)

    # Figure C: natural factor-proxy distributions.
    proxy_cols = [
        "role_diversity_ratio",
        "scope_match_rate",
        "condition_present",
        "authority_cue_present",
        "provenance_valid_rate",
        "redundancy_proxy",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(9.2, 5.4))
    for ax, proxy in zip(axes.flat, proxy_cols):
        values = [
            natural.loc[natural["aircraft_model"].eq(aircraft), proxy].dropna().to_numpy(dtype=float)
            for aircraft in ["Boeing 737", "Cessna 172"]
        ]
        ax.boxplot(values, tick_labels=["B737", "C172"], showfliers=False)
        ax.set_title(proxy.replace("_", " "))
        ax.set_ylim(-0.05, 1.05 if proxy != "redundancy_proxy" else max(0.25, np.nanmax(natural[proxy]) * 1.1))
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Retriever-visible factor proxies in unmodified cue-free packages", y=1.01)
    fig.tight_layout()
    fig.savefig(FIGURES / "figure_c_natural_factor_proxy_distributions.png", dpi=350, bbox_inches="tight")
    plt.close(fig)

    # Figure D: synthetic intervention response heatmap.
    heat = responses.groupby(["intervention", "proxy"])["delta"].mean().unstack(fill_value=0.0)
    ordered_proxies = [
        "authority_cue_present",
        "condition_present",
        "warning_present",
        "provenance_valid_rate",
        "scope_match_rate",
        "redundancy_proxy",
        "conflict_marker_proxy",
        "action_or_procedure_present",
        "action_authority_evidence_availability_proxy",
    ]
    heat = heat.reindex(columns=[p for p in ordered_proxies if p in heat.columns])
    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    vmax = max(0.01, float(np.nanmax(np.abs(heat.to_numpy()))))
    im = ax.imshow(heat.to_numpy(), cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    intervention_labels = {
        "authority_removal": "authority removal",
        "condition_removal": "condition removal",
        "conflict_injection": "conflict injection",
        "duplicate_injection": "duplicate injection",
        "evidence_order_permutation": "evidence-order permutation",
        "procedural_to_descriptive_substitution": "procedure substitution",
        "provenance_masking": "provenance masking",
        "scope_swap": "scope swap",
        "warning_removal": "warning removal",
    }
    proxy_labels = {
        "authority_cue_present": "authority",
        "condition_present": "condition",
        "warning_present": "warning",
        "provenance_valid_rate": "provenance",
        "scope_match_rate": "scope",
        "redundancy_proxy": "redundancy",
        "conflict_marker_proxy": "conflict",
        "action_or_procedure_present": "procedure",
        "action_authority_evidence_availability_proxy": "prerequisite",
    }
    ax.set_yticks(np.arange(len(heat)), [intervention_labels.get(x, x.replace("_", " ")) for x in heat.index])
    ax.set_xticks(
        np.arange(len(heat.columns)),
        [proxy_labels.get(x, x.replace("_", "\n")) for x in heat.columns],
        rotation=24,
        ha="right",
    )
    for i in range(len(heat)):
        for j in range(len(heat.columns)):
            ax.text(j, i, f"{heat.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
    ax.set_title("Synthetic controlled package interventions: mean proxy change")
    fig.colorbar(im, ax=ax, shrink=0.75, label="after - before")
    fig.tight_layout()
    fig.savefig(FIGURES / "figure_d_synthetic_intervention_heatmap.png", dpi=350)
    plt.close(fig)

    # Figure E: documentary prerequisites for action-like queries.
    action = natural[natural["action_like_query_proxy"].eq(1)].copy()
    prereq = [
        "action_or_procedure_present",
        "authority_cue_present",
        "scope_match_rate",
        "condition_present",
        "provenance_valid_rate",
        "action_authority_evidence_availability_proxy",
    ]
    means = action.groupby("aircraft_model")[prereq].mean().reindex(["Boeing 737", "Cessna 172"])
    fig, ax = plt.subplots(figsize=(8.4, 4.5))
    x = np.arange(len(prereq))
    width = 0.36
    for offset, aircraft in zip([-width / 2, width / 2], means.index):
        ax.bar(x + offset, means.loc[aircraft], width, label=aircraft, color=colors[aircraft])
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x, [p.replace("_", "\n") for p in prereq])
    ax.set_ylabel("Availability rate")
    ax.set_title("Action-like queries: documentary-prerequisite availability proxies")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES / "figure_e_action_authority_prerequisites.png", dpi=350)
    plt.close(fig)


def build_claim_maps(config: dict, query_df: pd.DataFrame, paired: pd.DataFrame, natural: pd.DataFrame, cases: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    accepted_rows = paired[(paired["aircraft_model"].eq("All")) & (paired["metric"].eq("offline_accepted_recall_10"))]
    cue_effect = accepted_rows[
        accepted_rows["reference"].eq("legacy_cue_on")
        & accepted_rows["comparison"].eq("cue_off_track_fixed")
    ].iloc[0]
    fair_effect = accepted_rows[
        accepted_rows["reference"].eq("legacy_cue_on")
        & accepted_rows["comparison"].eq("fair_cue_and_track_off")
    ].iloc[0]
    dense = query_df[query_df["condition"].eq("cue_off_dense_cross_encoder")]["offline_accepted_recall_10"].mean()
    intervention_rate = cases["directional_success"].mean() if len(cases) else math.nan
    action = natural[natural["action_like_query_proxy"].eq(1)]
    action_proxy = action["action_authority_evidence_availability_proxy"].mean() if len(action) else math.nan
    claim_rows = [
        {
            "claim_id": "C1",
            "manuscript_claim": "Legacy V21 held-out retrieval is materially dependent on gold-derived evidence cues.",
            "support_status": "Supported",
            "experiment_id": config["experiment_id"],
            "population": "214 split-held-out cue-augmented audit queries",
            "metric": "paired cue-off minus cue-on accepted R@10",
            "value": float(cue_effect["paired_delta"]),
            "ci": f"[{cue_effect['ci_low']:.4f}, {cue_effect['ci_high']:.4f}]",
            "result_file": "04_results/aesr_v2/cue_dependence_paired_bootstrap.csv",
            "figure_or_table": "Figure A / cue dependence",
            "gold_dependency": "Gold-derived cue appears only in the legacy positive-control condition; scoring is post hoc.",
            "synthetic_only": True,
            "caveat": "All benchmark questions are template-generated from source evidence. This diagnoses shortcut dependence; it is not a natural-query generalization estimate.",
        },
        {
            "claim_id": "C2",
            "manuscript_claim": "Removing both the appended gold cue and locked evaluation-track routing further defines the fair no-gold-input reference.",
            "support_status": "Supported",
            "experiment_id": config["experiment_id"],
            "population": "214 split-held-out, synthetic source-derived audit queries",
            "metric": "paired fair cue-and-track-off minus legacy accepted R@10",
            "value": float(fair_effect["paired_delta"]),
            "ci": f"[{fair_effect['ci_low']:.4f}, {fair_effect['ci_high']:.4f}]",
            "result_file": "04_results/aesr_v2/cue_dependence_paired_bootstrap.csv",
            "figure_or_table": "Figure A / cue and track dependence",
            "gold_dependency": "No gold-derived retrieval input in the comparison condition; locked labels are used only for post-hoc scoring.",
            "synthetic_only": True,
            "caveat": "The split groups exact-gold IDs only; accepted/page/section alternatives can still cross splits.",
        },
        {
            "claim_id": "C3",
            "manuscript_claim": "A real multilingual dense plus cross-encoder setting was evaluated without appended gold cues.",
            "support_status": "Supported",
            "experiment_id": config["experiment_id"],
            "population": "214 cue- and track-free synthetic source-derived audit queries",
            "metric": "mean accepted R@10",
            "value": float(dense),
            "ci": "not estimated",
            "result_file": "04_results/aesr_v2/cue_free_access_query.csv",
            "figure_or_table": "Figure B / cue-free access comparison",
            "gold_dependency": "No gold-derived retrieval feature; locked accepted sets used only after retrieval.",
            "synthetic_only": True,
            "caveat": "This is an access comparison, not a full long-context generation or AESR evaluation.",
        },
        {
            "claim_id": "C4",
            "manuscript_claim": "Retriever-visible factor proxies can be audited on unmodified cue- and track-free packages.",
            "support_status": "Supported",
            "experiment_id": config["experiment_id"],
            "population": f"{len(natural)} unmodified packages retrieved for synthetic source-derived queries",
            "metric": "factor-proxy distributions",
            "value": float(natural["provenance_valid_rate"].mean()),
            "ci": "not estimated",
            "result_file": "04_results/aesr_v2/natural_factor_proxies.csv",
            "figure_or_table": "Figure C",
            "gold_dependency": "None in factor construction.",
            "synthetic_only": True,
            "caveat": "These are observable proxies, not calibrated latent knowledge states.",
        },
        {
            "claim_id": "C5",
            "manuscript_claim": "Controlled package interventions change their preregistered target proxies in the expected direction.",
            "support_status": "Supported for deterministic proxy sensitivity only",
            "experiment_id": config["experiment_id"],
            "population": f"{len(cases)} eligible synthetic/counterfactual package interventions",
            "metric": "directional response consistency",
            "value": float(intervention_rate),
            "ci": "not estimated",
            "result_file": "04_results/aesr_v2/intervention_cases.csv",
            "figure_or_table": "Figure D",
            "gold_dependency": "None in proxy response calculation.",
            "synthetic_only": True,
            "caveat": "Does not establish five-state classification accuracy or causal mechanism. Response was uneven: procedural-to-descriptive substitution changed its target proxy in only 1 of 154 eligible cases.",
        },
        {
            "claim_id": "C6",
            "manuscript_claim": "Action-like requests can be audited for simultaneous documentary prerequisites.",
            "support_status": "Partially supported as an availability proxy",
            "experiment_id": config["experiment_id"],
            "population": f"{len(action)} action-like unmodified packages for synthetic source-derived queries",
            "metric": "action-authority evidence availability proxy",
            "value": float(action_proxy),
            "ci": "not estimated",
            "result_file": "04_results/aesr_v2/action_authority_proxy.csv",
            "figure_or_table": "Figure E",
            "gold_dependency": "None in proxy construction.",
            "synthetic_only": True,
            "caveat": "Authority cues do not prove legal or operational authorization; no expert action-authority labels exist, and the heuristic prerequisites are near-saturated.",
        },
        {
            "claim_id": "C7",
            "manuscript_claim": "AESR is empirically access-agnostic across retrieval, long-context, cache, memory, and tools.",
            "support_status": "Unsupported / not evaluated",
            "experiment_id": config["experiment_id"],
            "population": "not applicable",
            "metric": "backend by state-inference matrix",
            "value": math.nan,
            "ci": "not applicable",
            "result_file": "",
            "figure_or_table": "",
            "gold_dependency": "not applicable",
            "synthetic_only": False,
            "caveat": "Only retrieval-oriented access settings and a local original-order window proxy were evaluated.",
        },
        {
            "claim_id": "C8",
            "manuscript_claim": "The system accurately classifies Supported, Insufficient, Inapplicable, Conflicted, and OutOfScope.",
            "support_status": "Unsupported / not evaluated",
            "experiment_id": config["experiment_id"],
            "population": "no natural five-state gold labels",
            "metric": "five-state accuracy",
            "value": math.nan,
            "ci": "not applicable",
            "result_file": "",
            "figure_or_table": "",
            "gold_dependency": "not applicable",
            "synthetic_only": False,
            "caveat": "Synthetic interventions report proxy signatures, not state predictions.",
        },
        {
            "claim_id": "C9",
            "manuscript_claim": "A GNN captures higher-order evidence dependencies beyond node attributes.",
            "support_status": "Unsupported / not implemented",
            "experiment_id": config["experiment_id"],
            "population": "not applicable",
            "metric": "GNN versus capacity-matched and edge-shuffled controls",
            "value": math.nan,
            "ci": "not applicable",
            "result_file": "",
            "figure_or_table": "",
            "gold_dependency": "not applicable",
            "synthetic_only": False,
            "caveat": "The current document-supported graph remains a routing index.",
        },
    ]
    claims = pd.DataFrame(claim_rows)
    audit = pd.DataFrame(
        [
            ("AESR as a knowledge-access/state-inference separation", "Conceptual formulation only", "No cross-backend state-inference matrix", "Implement matched backends and state labels", "Partially supported"),
            ("Evidence obligations", "Heuristic query parser and required-slot audit", "Required slots are locked and cue-derived benchmark questions remain synthetic", "Independent obligation annotations and parser evaluation", "Partially supported"),
            ("Five epistemic states", "Not implemented; legacy output is binary", "Synthetic proxy signatures only", "Natural and controlled five-state gold labels", "Unsupported"),
            ("Descriptive warrant versus action authorization", "Documentary prerequisite availability proxy", "No expert action-authority labels", "Paired factual/action protocol and expert annotation", "Partially supported"),
            ("Authority and provenance", "Document-derived cue and anchor fields", "Cue-off package distributions and targeted synthetic interventions", "Source-authority validation and revision control", "Partially supported"),
            ("Conflict and revision", "No natural resolver", "Synthetic conflict-marker response only", "Real supersession/conflict annotations", "Unsupported for natural data"),
            ("Evidence redundancy", "Pairwise lexical overlap proxy", "Natural distributions and duplicate-injection sensitivity", "Task-conditioned marginal contribution estimator", "Partially supported"),
            ("Evidence complementarity", "Role-diversity proxy only", "No independent synergy estimator", "Pairwise/higher-order interaction experiment", "Unsupported as synergy claim"),
            ("Modern dense plus reranker", "Multilingual MiniLM plus mMARCO cross-encoder", "Cue-free retrieval audit", "Matched generation/token-budget study", "Supported for retrieval audit"),
            ("Long-context alternative", "Local original-order window proxy only", "Cue-free retrieval audit", "Full matched-budget long-context reader", "Unsupported as DOS-RAG reproduction"),
            ("Online admissibility gate", "Legacy V21 project computes offline readiness from locked metrics", "No calibrated online state target", "Visible-feature-only decision and calibration", "Unsupported in current V21 code"),
        ],
        columns=["current_scientific_claim", "current_implementation", "current_experimental_evidence", "missing_evidence", "support_status"],
    )
    return claims, audit


def write_report(config: dict, query_df: pd.DataFrame, paired: pd.DataFrame, natural: pd.DataFrame, cases: pd.DataFrame, claims: pd.DataFrame) -> None:
    accepted = paired[(paired["aircraft_model"].eq("All")) & (paired["metric"].eq("offline_accepted_recall_10"))]
    cue = accepted[
        accepted["reference"].eq("legacy_cue_on")
        & accepted["comparison"].eq("cue_off_track_fixed")
    ].iloc[0]
    fair = accepted[
        accepted["reference"].eq("legacy_cue_on")
        & accepted["comparison"].eq("fair_cue_and_track_off")
    ].iloc[0]
    access = (
        query_df[query_df["condition"].str.startswith("cue_off_")]
        .groupby("condition")["offline_accepted_recall_10"]
        .mean()
        .sort_values(ascending=False)
    )
    action = natural[natural["action_like_query_proxy"].eq(1)]
    target_success = cases.groupby("intervention")["directional_success"].agg(["mean", "count"]).reset_index()
    procedure_substitution = target_success[
        target_success["intervention"].eq("procedural_to_descriptive_substitution")
    ].iloc[0]
    lines = [
        "# V21 Evidence-State Diagnostic Audit",
        "",
        config["claim_boundary"],
        "",
        "## Cue-dependence audit",
        "",
        (
            f"Across {int(cue['n'])} split-held-out, synthetic source-derived queries, removing the appended gold-derived cue while retaining legacy track routing changed accepted R@10 by "
            f"{cue['paired_delta']:.4f} (paired bootstrap 95% interval {cue['ci_low']:.4f} to {cue['ci_high']:.4f}). "
            f"Removing both cue and locked-track routing changed it by {fair['paired_delta']:.4f} "
            f"(95% interval {fair['ci_low']:.4f} to {fair['ci_high']:.4f}). "
            "The legacy condition is therefore a gold-dependent diagnostic upper reference, not natural cold retrieval."
        ),
        "",
        "## Cue-free access settings",
        "",
        access.rename("mean_accepted_recall_10").reset_index().to_markdown(index=False),
        "",
        "The dense-plus-cross-encoder row is a real multilingual retrieval/reranking run. The local original-order row is only a bounded-window access proxy; no long-context generator was run.",
        "",
        "## Unmodified-package factor proxies",
        "",
        (
            f"The factor table contains {len(natural)} unmodified packages retrieved without cue or locked-track routing for synthetic source-derived queries. Mean provenance-anchor validity was "
            f"{natural['provenance_valid_rate'].mean():.4f}, mean scope match was {natural['scope_match_rate'].mean():.4f}, "
            f"and mean pairwise redundancy proxy was {natural['redundancy_proxy'].mean():.4f}. The saturated scope and provenance fields reflect a coarse aircraft-family mask and locator availability, not validated effectivity or revision control. These are observable package proxies, not latent-state probabilities."
        ),
        "",
        "## Synthetic controlled package interventions",
        "",
        target_success.to_markdown(index=False),
        "",
        (
            "Every intervention row is machine-labelled synthetic/counterfactual. Directional response consistency verifies implementation sensitivity of the named proxy only; it is not five-state accuracy or causal evidence. "
            f"The most important negative result is that procedural-to-descriptive substitution changed its target proxy in only {int(round(procedure_substitution['mean'] * procedure_substitution['count']))} of {int(procedure_substitution['count'])} eligible cases, exposing a weakness in the current lexical procedure proxy."
        ),
        "",
        "## Action-related documentary prerequisites",
        "",
        (
            f"Among {len(action)} unmodified cue- and track-free packages selected by a visible-text action-like query proxy, "
            f"{action['action_authority_evidence_availability_proxy'].mean():.4f} simultaneously contained an action/procedure cue, "
            "an authority cue, matching aircraft scope, a condition cue, and valid provenance. The near-saturated value reflects permissive lexical prerequisites; it is an evidence-availability proxy, not proof that an action is authorized."
        ),
        "",
        "## Claim boundary",
        "",
        claims[["claim_id", "manuscript_claim", "support_status", "caveat"]].to_markdown(index=False),
        "",
    ]
    (REPORTS / "v21_evidence_state_diagnostic_audit.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    np.random.seed(int(config["seed"]))
    torch.manual_seed(int(config["seed"]))
    torch.set_num_threads(min(16, max(1, os.cpu_count() or 1)))
    torch.set_num_interop_threads(2)

    v21 = load_v21_module()
    bench = pd.read_csv(ROOT / "04_results" / "v21" / "qa_warehouse_v21_full.csv").fillna("")
    bench = bench[bench["split"].eq("v21_locked_blind_test")].reset_index(drop=True)
    ev = pd.read_parquet(ROOT / "01_processed_corpus" / "v5" / "evidence_objects.parquet").fillna("")
    labels = load_labels()

    print("[1/7] cue and locked-track dependence audit", flush=True)
    cue_df = cue_audit(config, v21, bench, ev, labels)
    cue_df.to_csv(RESULTS / "cue_dependence_query.csv", index=False, encoding="utf-8-sig")

    print("[2/7] cue- and track-free access audit (dense encoding and reranking)", flush=True)
    access_df = cue_free_access_audit(config, v21, bench, ev, labels)
    access_df.to_csv(RESULTS / "cue_free_access_query.csv", index=False, encoding="utf-8-sig")

    query_df = pd.concat([cue_df, access_df], ignore_index=True, sort=False)
    query_df.to_csv(RESULTS / "all_access_query_results.csv", index=False, encoding="utf-8-sig")
    summary = summarize_query_metrics(query_df)
    summary.to_csv(RESULTS / "cue_dependence_and_access_summary.csv", index=False, encoding="utf-8-sig")

    paired_frames = [
        paired_bootstrap(
            cue_df,
            metric,
            int(config["seed"]) + pos,
            int(config["bootstrap_samples"]),
        )
        for pos, metric in enumerate(
            ["offline_accepted_recall_10", "offline_locked_slot_coverage_10", "offline_construction_composite"]
        )
    ]
    paired = pd.concat(paired_frames, ignore_index=True)
    paired.to_csv(RESULTS / "cue_dependence_paired_bootstrap.csv", index=False, encoding="utf-8-sig")

    print("[3/7] unmodified-package factor proxies", flush=True)
    natural = natural_factor_proxies(v21, bench, ev, access_df)
    natural.to_csv(RESULTS / "natural_factor_proxies.csv", index=False, encoding="utf-8-sig")
    action_proxy = natural[
        [
            "experiment_id",
            "question_id",
            "aircraft_model",
            "query_type",
            "action_like_query_proxy",
            "action_or_procedure_present",
            "authority_cue_present",
            "scope_match_rate",
            "condition_present",
            "provenance_valid_rate",
            "conflict_marker_proxy",
            "action_authority_evidence_availability_proxy",
        ]
    ].copy()
    action_proxy.to_csv(RESULTS / "action_authority_proxy.csv", index=False, encoding="utf-8-sig")

    print("[4/7] synthetic controlled package interventions", flush=True)
    cases, responses, eligible = run_interventions(config, v21, bench, ev, natural)
    cases.to_csv(RESULTS / "intervention_cases.csv", index=False, encoding="utf-8-sig")
    responses.to_csv(RESULTS / "intervention_response.csv", index=False, encoding="utf-8-sig")
    eligible.to_csv(RESULTS / "eligible_case_counts.csv", index=False, encoding="utf-8-sig")

    print("[5/7] leakage and scientific-claim audits", flush=True)
    leakage = write_split_leakage_audit()
    leakage.to_csv(RESULTS / "split_leakage_audit.csv", index=False, encoding="utf-8-sig")

    claims, scientific_audit = build_claim_maps(config, query_df, paired, natural, cases)
    claims.to_csv(RESULTS / "claim_result_map.csv", index=False, encoding="utf-8-sig")
    scientific_audit.to_csv(RESULTS / "scientific_claim_audit.csv", index=False, encoding="utf-8-sig")

    print("[6/7] figures, tables, and report", flush=True)
    make_figures(query_df, paired, natural, responses)
    save_md_table(summary, "table_a_access_summary")
    save_md_table(paired, "table_b_cue_dependence_paired")
    save_md_table(eligible, "table_c_intervention_eligibility")
    save_md_table(
        cases.groupby("intervention")["directional_success"].agg(["mean", "count"]).reset_index(),
        "table_d_intervention_directional_response",
    )
    save_md_table(
        natural.groupby("aircraft_model")[
            [
                "role_diversity_ratio",
                "scope_match_rate",
                "condition_present",
                "warning_present",
                "authority_cue_present",
                "provenance_valid_rate",
                "redundancy_proxy",
                "action_authority_evidence_availability_proxy",
            ]
        ].mean().reset_index(),
        "table_e_natural_factor_proxy_means",
    )
    write_report(config, query_df, paired, natural, cases, claims)

    source_files = [
        Path(__file__).resolve(),
        args.config.resolve(),
        ROOT / "_pipeline_v21" / "p0_build_eval_v21.py",
        ROOT / "04_results" / "v21" / "qa_warehouse_v21_full.csv",
        ROOT / "04_results" / "v21" / "v21_gold_sets.csv",
        ROOT / "01_processed_corpus" / "v5" / "evidence_objects.parquet",
    ]
    print("[7/7] run manifest", flush=True)
    manifest = {
        "experiment_id": config["experiment_id"],
        "run_timestamp_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": config["seed"],
        "top_k": config["top_k"],
        "n_locked_queries": int(len(bench)),
        "n_evidence_objects": int(len(ev)),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "dense_model": config["dense_model"],
        "cross_encoder_model": config["cross_encoder_model"],
        "offline_models_only": config.get("offline_models_only", True),
        "source_hashes": {str(p.relative_to(ROOT) if p.is_relative_to(ROOT) else p): sha256(p) for p in source_files},
        "retrieval_visible_in_fair_cue_and_track_off": ["base question text", "aircraft_model", "document-derived evidence repository"],
        "locked_offline_only": [
            "evaluation track",
            "query_type",
            "required_package_slots",
            "gold/accepted IDs",
            "gold pages/sections",
            "split",
        ],
        "claim_boundary": config["claim_boundary"],
        "result_files": sorted(str(p.relative_to(ROOT)) for p in RESULTS.glob("*.csv")),
        "figure_files": sorted(str(p.relative_to(ROOT)) for p in FIGURES.glob("*.png")),
    }
    (REPORTS / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "complete", "experiment_id": config["experiment_id"], "results": str(RESULTS)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
