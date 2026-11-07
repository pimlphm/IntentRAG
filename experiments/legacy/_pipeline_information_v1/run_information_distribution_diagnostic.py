"""Run observable information-distribution diagnostics on fair AESR-v2 packages.

The program deliberately estimates distributions over observable categories.  It
does not estimate latent epistemic probabilities, mutual information, true
information gain, causal contribution, answer faithfulness, or aircraft safety.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import math
import os
import platform
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "information_v1" / "information_v1.json"
RESULTS = ROOT / "04_results" / "information_v1"
RAW = RESULTS / "raw"
FIGURES = ROOT / "05_figures" / "information_v1"
TABLES = ROOT / "06_tables" / "information_v1"
REPORTS = ROOT / "07_reports" / "information_v1"
LOGS = ROOT / "08_logs" / "information_v1"
for directory in (RESULTS, RAW, FIGURES, TABLES, REPORTS, LOGS):
    directory.mkdir(parents=True, exist_ok=True)


METHOD_LABELS = {
    "cue_off_word_tfidf": "Word TF-IDF",
    "cue_off_char_tfidf": "Char TF-IDF",
    "cue_off_hybrid_lexical": "Hybrid lexical",
    "cue_off_intentrag": "IntentRAG (balanced)",
    "cue_off_dense_multilingual": "Dense multilingual",
    "cue_off_dense_cross_encoder": "Dense + cross-encoder",
    "cue_off_local_original_order_window": "Original-order window",
}

SHORT_METHOD_LABELS = {
    "cue_off_word_tfidf": "Word",
    "cue_off_char_tfidf": "Char",
    "cue_off_hybrid_lexical": "Hybrid",
    "cue_off_intentrag": "IntentRAG",
    "cue_off_dense_multilingual": "Dense",
    "cue_off_dense_cross_encoder": "Dense + CE",
    "cue_off_local_original_order_window": "Order window",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_hashes(directory: Path) -> dict[str, str]:
    if not directory.exists():
        return {}
    return {
        str(path.relative_to(ROOT)): sha256(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def norm(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def norm_low(value: object) -> str:
    return norm(value).lower()


def split_gold_cue(question: str) -> tuple[str, str]:
    text = norm(question)
    match = re.search(r"\s+(?:Evidence cue|Include evidence cues):\s*(.+?)\.?$", text, flags=re.I)
    if not match:
        return text, ""
    return text[: match.start()].strip(), match.group(1).strip().rstrip(".")


def parse_ids(value: object) -> list[str]:
    return [item for item in str(value or "").split(";") if item]


def load_v21_module():
    path = ROOT / "_pipeline_v21" / "p0_build_eval_v21.py"
    spec = importlib.util.spec_from_file_location("v21_information_source", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalized_mass(values: Iterable[float], fallback: np.ndarray | None = None) -> tuple[np.ndarray, bool]:
    array = np.asarray(list(values), dtype=float)
    array[~np.isfinite(array)] = 0.0
    array = np.clip(array, 0.0, None)
    total = float(array.sum())
    if total > 0:
        return array / total, False
    if fallback is not None:
        fallback_array = np.asarray(fallback, dtype=float)
        fallback_array = np.clip(fallback_array, 0.0, None)
        fallback_total = float(fallback_array.sum())
        if fallback_total > 0:
            return fallback_array / fallback_total, True
    if len(array) == 0:
        return array, True
    return np.full(len(array), 1.0 / len(array), dtype=float), True


def normalized_entropy(probabilities: np.ndarray) -> float:
    p = np.asarray(probabilities, dtype=float)
    if len(p) <= 1:
        return 0.0
    positive = p[p > 0]
    entropy = float(-(positive * np.log2(positive)).sum())
    return float(entropy / math.log2(len(p)))


def js_divergence(left: np.ndarray, right: np.ndarray) -> float:
    p = np.asarray(left, dtype=float)
    q = np.asarray(right, dtype=float)
    if p.shape != q.shape:
        raise ValueError(f"JSD shape mismatch: {p.shape} versus {q.shape}")
    midpoint = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float((a[mask] * np.log2(a[mask] / b[mask])).sum())

    return float(0.5 * kl(p, midpoint) + 0.5 * kl(q, midpoint))


def bootstrap_mean_ci(values: Iterable[float], seed: int, samples: int) -> tuple[float, float, float, int]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return math.nan, math.nan, math.nan, 0
    mean = float(array.mean())
    if len(array) == 1 or samples <= 0:
        return mean, mean, mean, len(array)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(samples, len(array)))
    means = array[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return mean, float(low), float(high), len(array)


def role_value(value: object) -> str:
    text = norm_low(value)
    return text if text else "unspecified"


def source_label(file_name: object, document_id: object) -> str:
    stem = Path(str(file_name or "")).stem
    return stem if stem else str(document_id)


def category_columns(axis: str) -> tuple[str, str]:
    if axis == "evidence_role":
        return "_role_category", "_role_label"
    if axis == "source_document":
        return "_source_category", "_source_label"
    raise ValueError(axis)


def count_term_hit(text: str, term: str) -> int:
    candidate = norm_low(term)
    if not candidate:
        return 0
    if re.fullmatch(r"[a-z0-9][a-z0-9 _\-/]*", candidate):
        pattern = r"(?<![a-z0-9])" + re.escape(candidate) + r"(?![a-z0-9])"
        return int(bool(re.search(pattern, text, flags=re.I)))
    return int(candidate in text)


def visible_role_demand(
    query: str,
    support: list[str],
    lexicon: dict[str, list[str]],
    repository_prior: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, str, bool]:
    text = norm_low(query)
    raw = np.zeros(len(support), dtype=float)
    matched_terms: list[str] = []
    for position, role in enumerate(support):
        for term in lexicon.get(role, []):
            if count_term_hit(text, str(term)):
                raw[position] += 1.0
                matched_terms.append(f"{role}:{norm(term)}")
    if raw.sum() <= 0:
        fallback = np.zeros(len(support), dtype=float)
        if "general" in support:
            fallback[support.index("general")] = 1.0
        else:
            fallback = repository_prior.copy()
        mass, _ = normalized_mass(fallback, repository_prior)
        return mass, raw, 0, "", True
    mass, _ = normalized_mass(raw)
    return mass, raw, len(matched_terms), ";".join(matched_terms), False


def locked_role_sensitivity(
    query_type: str,
    support: list[str],
    mapping: dict[str, list[str]],
) -> np.ndarray | None:
    roles = [role for role in mapping.get(str(query_type), []) if role in support]
    if not roles:
        return None
    raw = np.asarray([float(role in roles) for role in support], dtype=float)
    return normalized_mass(raw)[0]


def build_repository_priors(ev: pd.DataFrame, axes: list[str]) -> dict[tuple[str, str], dict]:
    priors: dict[tuple[str, str], dict] = {}
    for aircraft, aircraft_df in ev.groupby("aircraft_model", sort=True):
        for axis in axes:
            category_col, label_col = category_columns(axis)
            counts = aircraft_df[category_col].value_counts().sort_index()
            support = counts.index.astype(str).tolist()
            mass, _ = normalized_mass(counts.to_numpy(dtype=float))
            labels = (
                aircraft_df[[category_col, label_col]]
                .drop_duplicates(category_col)
                .set_index(category_col)[label_col]
                .astype(str)
                .to_dict()
            )
            priors[(str(aircraft), axis)] = {
                "support": support,
                "labels": [labels.get(key, key) for key in support],
                "counts": counts.to_numpy(dtype=int),
                "mass": mass,
                "candidate_count": int(len(aircraft_df)),
            }
    return priors


def aggregate_mass(
    frame: pd.DataFrame,
    axis: str,
    support: list[str],
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    category_col, _ = category_columns(axis)
    categories = frame[category_col].astype(str).tolist()
    raw = np.zeros(len(support), dtype=float)
    counts = np.zeros(len(support), dtype=int)
    positions = {category: index for index, category in enumerate(support)}
    for category, weight in zip(categories, weights):
        if category not in positions:
            raise KeyError(f"Category {category!r} not in {axis} support")
        raw[positions[category]] += float(weight)
        counts[positions[category]] += 1
    return normalized_mass(raw)[0], counts


def source_affinity_demand(
    scores: np.ndarray,
    candidate_frame: pd.DataFrame,
    support: list[str],
    affinity_power: float,
    repository_prior: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, bool, int]:
    category_col, _ = category_columns("source_document")
    weights = np.power(np.clip(np.asarray(scores, dtype=float), 0.0, None), affinity_power)
    raw = np.zeros(len(support), dtype=float)
    positions = {category: index for index, category in enumerate(support)}
    for category, weight in zip(candidate_frame[category_col].astype(str), weights):
        raw[positions[category]] += float(weight)
    mass, fallback = normalized_mass(raw, repository_prior)
    return mass, raw, fallback, int((weights > 0).sum())


def build_reference_mass(
    accepted_ids: list[str],
    ev_by_id: pd.DataFrame,
    axis: str,
    support: list[str],
) -> tuple[np.ndarray, np.ndarray, int]:
    unique_ids = list(dict.fromkeys(accepted_ids))
    missing = [evidence_id for evidence_id in unique_ids if evidence_id not in ev_by_id.index]
    if missing:
        raise KeyError(f"Accepted evidence IDs missing from corpus: {missing[:5]}")
    reference = ev_by_id.loc[unique_ids].copy()
    weights = np.ones(len(reference), dtype=float)
    mass, counts = aggregate_mass(reference, axis, support, weights)
    return mass, counts, len(unique_ids)


def package_frame(top_ids: list[str], ev_by_id: pd.DataFrame) -> pd.DataFrame:
    if not top_ids:
        raise ValueError("Empty package")
    missing = [evidence_id for evidence_id in top_ids if evidence_id not in ev_by_id.index]
    if missing:
        raise KeyError(f"Top evidence IDs missing from corpus: {missing[:5]}")
    frame = ev_by_id.loc[top_ids].copy()
    frame["_rank"] = np.arange(1, len(frame) + 1)
    frame["_rank_weight"] = 1.0 / np.log2(frame["_rank"].to_numpy(dtype=float) + 1.0)
    return frame


def make_benchmark_demand_mix(bench: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    populations = [("All", bench)] + [(str(name), group) for name, group in bench.groupby("aircraft_model")]
    for population, group in populations:
        for facet, series in (
            ("locked_query_type", group["query_type"].astype(str)),
            (
                "locked_required_slot",
                group["required_package_slots"].astype(str).str.split(";").explode().replace("", np.nan).dropna(),
            ),
        ):
            counts = series.value_counts().sort_index()
            masses, _ = normalized_mass(counts.to_numpy(dtype=float))
            entropy = normalized_entropy(masses)
            for (category, count), mass in zip(counts.items(), masses):
                rows.append(
                    {
                        "population": population,
                        "facet": facet,
                        "category": category,
                        "count": int(count),
                        "mass": float(mass),
                        "normalized_entropy": entropy,
                        "locked_offline_only": True,
                        "construction_coupled": True,
                    }
                )
    return pd.DataFrame(rows)


def write_table(frame: pd.DataFrame, stem: str) -> None:
    frame.to_csv(TABLES / f"{stem}.csv", index=False, encoding="utf-8-sig")
    (TABLES / f"{stem}.md").write_text(frame.to_markdown(index=False), encoding="utf-8")


def summarize_diagnostics(diagnostics: pd.DataFrame, config: dict) -> pd.DataFrame:
    metrics = [
        "hnorm_repository",
        "hnorm_query_demand",
        "hnorm_package",
        "jsd_query_demand_repository",
        "jsd_package_repository",
        "jsd_package_query_demand",
        "observable_alignment_shift",
        "demand_mass_covered",
        "jsd_package_reference",
        "reference_mass_covered",
        "jsd_visible_query_locked_sensitivity",
        "loo_alignment_positive_rate",
        "loo_alignment_negative_rate",
    ]
    working = pd.concat(
        [
            diagnostics.assign(population=diagnostics["aircraft_model"]),
            diagnostics.assign(population="All"),
        ],
        ignore_index=True,
    )
    rows: list[dict] = []
    counter = 0
    for keys, group in working.groupby(["population", "axis", "condition", "method"], sort=True):
        population, axis, condition, method = keys
        row = {
            "experiment_id": config["experiment_id"],
            "population": population,
            "axis": axis,
            "condition": condition,
            "method": method,
            "n_queries": int(group["question_id"].nunique()),
            "mean_package_k": float(group["package_k"].mean()),
            "lexicon_fallback_rate": float(group["visible_query_fallback"].mean()),
            "source_axis_degenerate_rate": float(group["axis_degenerate"].mean()),
        }
        for metric in metrics:
            counter += 1
            mean, low, high, count = bootstrap_mean_ci(
                group[metric],
                int(config["seed"]) + counter,
                int(config["bootstrap_samples"]),
            )
            row[f"mean_{metric}"] = mean
            row[f"ci_low_{metric}"] = low
            row[f"ci_high_{metric}"] = high
            row[f"n_{metric}"] = count
        rows.append(row)
    return pd.DataFrame(rows)


def outcome_associations(diagnostics: pd.DataFrame, config: dict) -> pd.DataFrame:
    metrics = [
        "hnorm_package",
        "jsd_package_repository",
        "jsd_package_query_demand",
        "observable_alignment_shift",
        "demand_mass_covered",
        "jsd_package_reference",
    ]
    rows: list[dict] = []
    for keys, group in diagnostics.groupby(["axis", "condition", "method"], sort=True):
        axis, condition, method = keys
        outcome = pd.to_numeric(group["offline_accepted_recall_10"], errors="coerce")
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce")
            valid = values.notna() & outcome.notna()
            x = values[valid]
            y = outcome[valid]
            hit = x[y.eq(1)]
            miss = x[y.eq(0)]
            pearson = float(x.corr(y, method="pearson")) if x.nunique() > 1 and y.nunique() > 1 else math.nan
            spearman = float(x.corr(y, method="spearman")) if x.nunique() > 1 and y.nunique() > 1 else math.nan
            rows.append(
                {
                    "experiment_id": config["experiment_id"],
                    "axis": axis,
                    "condition": condition,
                    "method": method,
                    "metric": metric,
                    "n_queries": int(valid.sum()),
                    "n_hits": int(y.eq(1).sum()),
                    "n_misses": int(y.eq(0).sum()),
                    "mean_when_hit": float(hit.mean()) if len(hit) else math.nan,
                    "mean_when_miss": float(miss.mean()) if len(miss) else math.nan,
                    "hit_minus_miss": float(hit.mean() - miss.mean()) if len(hit) and len(miss) else math.nan,
                    "pearson_r": pearson,
                    "spearman_rho": spearman,
                    "analysis_status": "exploratory association only",
                    "causal_interpretation_allowed": False,
                    "synthetic_source_derived_queries": True,
                }
            )
    return pd.DataFrame(rows)


def make_figures(
    diagnostics: pd.DataFrame,
    selection: pd.DataFrame,
    loo: pd.DataFrame,
    summary: pd.DataFrame,
    config: dict,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
        }
    )
    order = [condition for condition in config["access_conditions"] if condition in diagnostics["condition"].unique()]
    labels = [METHOD_LABELS.get(condition, condition) for condition in order]
    primary = str(config["primary_condition"])

    role_primary = selection[
        selection["axis"].eq("evidence_role") & selection["condition"].eq(primary)
    ]
    role_means = (
        role_primary.groupby("category_label")[["repository_mass", "query_demand_mass", "package_mass", "reference_mass"]]
        .mean()
        .sort_values("query_demand_mass", ascending=False)
    )
    top_roles = role_means.head(9)

    all_role = summary[summary["population"].eq("All") & summary["axis"].eq("evidence_role")].set_index("condition")
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.6), constrained_layout=True)
    x = np.arange(len(top_roles))
    width = 0.2
    for offset, column, label, color in (
        (-1.5, "repository_mass", r"$p_D$ repository", "#8c8c8c"),
        (-0.5, "query_demand_mass", r"$p_q$ visible demand", "#4c78a8"),
        (0.5, "package_mass", r"$p_R$ package", "#f58518"),
        (1.5, "reference_mass", r"$p_G$ offline reference", "#54a24b"),
    ):
        axes[0, 0].bar(x + offset * width, top_roles[column], width=width, label=label, color=color)
    axes[0, 0].set_xticks(x, top_roles.index, rotation=38, ha="right")
    axes[0, 0].set_ylabel("Mean normalized mass")
    axes[0, 0].set_title("a  Observable role distributions (balanced IntentRAG)", loc="left", fontweight="bold")
    axes[0, 0].legend(frameon=False, fontsize=8)

    xvals = [all_role.loc[c, "mean_jsd_package_repository"] for c in order]
    yvals = [all_role.loc[c, "mean_jsd_package_query_demand"] for c in order]
    colors = ["#d62728" if c == primary else "#4c78a8" for c in order]
    axes[0, 1].scatter(xvals, yvals, s=55, c=colors)
    annotation_offsets = {
        "cue_off_word_tfidf": (5, -15),
        "cue_off_char_tfidf": (-18, 8),
        "cue_off_hybrid_lexical": (5, 5),
        "cue_off_intentrag": (-60, 8),
        "cue_off_dense_multilingual": (5, -13),
        "cue_off_dense_cross_encoder": (5, 5),
        "cue_off_local_original_order_window": (5, 5),
    }
    for xv, yv, condition in zip(xvals, yvals, order):
        axes[0, 1].annotate(
            SHORT_METHOD_LABELS.get(condition, condition),
            (xv, yv),
            xytext=annotation_offsets.get(condition, (4, 4)),
            textcoords="offset points",
            fontsize=7,
        )
    axes[0, 1].set_xlabel(r"Selection shift $JS(p_R,p_D)$")
    axes[0, 1].set_ylabel(r"Demand mismatch $JS(p_R,p_q)$")
    axes[0, 1].set_title("b  Method-level shift and mismatch", loc="left", fontweight="bold")

    bar_x = np.arange(len(order))
    axes[1, 0].bar(
        bar_x - 0.22,
        [all_role.loc[c, "mean_hnorm_repository"] for c in order],
        width=0.22,
        color="#8c8c8c",
        label=r"$H_{norm}(p_D)$",
    )
    axes[1, 0].bar(
        bar_x,
        [all_role.loc[c, "mean_hnorm_query_demand"] for c in order],
        width=0.22,
        color="#4c78a8",
        label=r"$H_{norm}(p_q)$",
    )
    axes[1, 0].bar(
        bar_x + 0.22,
        [all_role.loc[c, "mean_hnorm_package"] for c in order],
        width=0.22,
        color="#f58518",
        label=r"$H_{norm}(p_R)$",
    )
    axes[1, 0].set_xticks(bar_x, labels, rotation=40, ha="right")
    axes[1, 0].set_ylim(0, 1.03)
    axes[1, 0].set_ylabel("Normalized entropy")
    axes[1, 0].set_title("c  Entropy is estimator-specific", loc="left", fontweight="bold")
    axes[1, 0].legend(frameon=False, fontsize=8)

    align = [all_role.loc[c, "mean_observable_alignment_shift"] for c in order]
    positive = [all_role.loc[c, "mean_loo_alignment_positive_rate"] for c in order]
    axes[1, 1].bar(bar_x - 0.18, align, width=0.36, color="#4c78a8", label=r"$G_{align}^{obs}$")
    axes[1, 1].bar(bar_x + 0.18, positive, width=0.36, color="#e45756", label="LOO positive fraction")
    axes[1, 1].axhline(0, color="black", lw=0.8)
    axes[1, 1].set_xticks(bar_x, labels, rotation=40, ha="right")
    axes[1, 1].set_ylabel("Observed value")
    axes[1, 1].set_title("d  Alignment shift and fixed-package contributions", loc="left", fontweight="bold")
    axes[1, 1].legend(frameon=False, fontsize=8)
    fig.suptitle("Observable information-distribution audit; no LLM generation or latent-state probability", fontweight="bold")
    fig.savefig(FIGURES / "figure_a_observable_distribution_composite.png", bbox_inches="tight")
    plt.close(fig)

    role_bias = selection[selection["axis"].eq("evidence_role")]
    aggregate_bias = (
        role_bias.groupby(["condition", "category_label"])[["repository_mass", "package_mass"]]
        .mean()
        .reset_index()
    )
    aggregate_bias["aggregate_log2_ratio"] = np.log2(
        (aggregate_bias["package_mass"] + float(config["log_ratio_smoothing"]))
        / (aggregate_bias["repository_mass"] + float(config["log_ratio_smoothing"]))
    )
    heat = aggregate_bias.pivot(
        index="condition", columns="category_label", values="aggregate_log2_ratio"
    ).reindex(order)
    columns = heat.abs().mean().sort_values(ascending=False).head(12).index
    heat = heat[columns].clip(-4, 4)
    fig, ax = plt.subplots(figsize=(12.5, 5.4), constrained_layout=True)
    image = ax.imshow(heat.to_numpy(), cmap="coolwarm", vmin=-4, vmax=4, aspect="auto")
    ax.set_yticks(np.arange(len(order)), labels)
    ax.set_xticks(np.arange(len(columns)), columns, rotation=42, ha="right")
    ax.set_title("Role selection log-ratio of mean masses relative to the repository prior (smoothed, clipped at ±4)", fontweight="bold")
    colorbar = fig.colorbar(image, ax=ax, shrink=0.9)
    colorbar.set_label(r"$\log_2[(\overline{p_R}+\epsilon)/(\overline{p_D}+\epsilon)]$")
    fig.savefig(FIGURES / "figure_b_role_selection_bias_heatmap.png", bbox_inches="tight")
    plt.close(fig)

    loo_role = loo[loo["axis"].eq("evidence_role") & loo["loo_defined"].eq(True)].copy()
    data = [
        loo_role.loc[loo_role["condition"].eq(condition), "loo_alignment_contribution"].dropna().to_numpy()
        for condition in order
    ]
    fig, ax = plt.subplots(figsize=(12.3, 5.4), constrained_layout=True)
    boxes = ax.boxplot(data, tick_labels=labels, showfliers=False, patch_artist=True)
    for patch, condition in zip(boxes["boxes"], order):
        patch.set_facecolor("#e45756" if condition == primary else "#8fb9df")
        patch.set_alpha(0.8)
    ax.axhline(0, color="black", lw=0.9)
    ax.set_xticklabels(labels, rotation=38, ha="right")
    ax.set_ylabel(r"$JS(p_q,p_{R\setminus i})-JS(p_q,p_R)$")
    ax.set_title("Fixed realized-package leave-one-out alignment contributions", fontweight="bold")
    fig.savefig(FIGURES / "figure_c_loo_alignment_contributions.png", bbox_inches="tight")
    plt.close(fig)

    cessna = selection[
        selection["axis"].eq("source_document")
        & selection["condition"].eq(primary)
        & selection["aircraft_model"].eq("Cessna 172")
    ]
    source_means = cessna.groupby("category_label")[["repository_mass", "query_demand_mass", "package_mass", "reference_mass"]].mean()
    fig, ax = plt.subplots(figsize=(10.5, 5.4), constrained_layout=True)
    x = np.arange(len(source_means))
    for offset, column, label, color in (
        (-1.5, "repository_mass", r"$p_D$", "#8c8c8c"),
        (-0.5, "query_demand_mass", r"$p_q$", "#4c78a8"),
        (0.5, "package_mass", r"$p_R$", "#f58518"),
        (1.5, "reference_mass", r"$p_G$", "#54a24b"),
    ):
        ax.bar(x + offset * 0.18, source_means[column], width=0.18, label=label, color=color)
    ax.set_xticks(x, source_means.index, rotation=25, ha="right")
    ax.set_ylabel("Mean normalized mass")
    ax.set_title("Cessna source-document distribution (balanced IntentRAG); Boeing source axis has one document", fontweight="bold")
    ax.legend(frameon=False, ncol=4)
    fig.savefig(FIGURES / "figure_d_source_distribution_composite.png", bbox_inches="tight")
    plt.close(fig)


def build_claim_map(
    diagnostics: pd.DataFrame,
    summary: pd.DataFrame,
    loo: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    primary = summary[
        summary["population"].eq("All")
        & summary["axis"].eq("evidence_role")
        & summary["condition"].eq(config["primary_condition"])
    ].iloc[0]
    primary_diag = diagnostics[
        diagnostics["axis"].eq("evidence_role") & diagnostics["condition"].eq(config["primary_condition"])
    ]
    primary_loo = loo[
        loo["axis"].eq("evidence_role")
        & loo["condition"].eq(config["primary_condition"])
        & loo["loo_defined"].eq(True)
    ]
    g_value = float(primary["mean_observable_alignment_shift"])
    g_status = "Positive observable shift" if g_value > 0 else "Non-positive observable shift"
    claims = [
        {
            "claim_id": "I1",
            "claim": "Repository, visible-query-demand, fair-package, and locked-reference category masses can be reconstructed as explicit normalized estimators.",
            "support_status": "Supported as an observable estimator audit",
            "metric": "distribution normalization and audit coverage",
            "value": 1.0,
            "result_file": "04_results/information_v1/distribution_diagnostics.csv",
            "caveat": "Masses are empirical counts, lexicon matches, TF-IDF affinity, or rank-discounted package weights; they are not calibrated epistemic probabilities.",
        },
        {
            "claim_id": "I2",
            "claim": "The balanced IntentRAG package changes role alignment relative to the repository prior under the visible-text demand estimator.",
            "support_status": g_status,
            "metric": "mean observable alignment shift JS(p_q,p_D)-JS(p_q,p_R)",
            "value": g_value,
            "result_file": "04_results/information_v1/distribution_diagnostics.csv",
            "caveat": f"Bootstrap interval [{primary['ci_low_observable_alignment_shift']:.4f}, {primary['ci_high_observable_alignment_shift']:.4f}]; this is not information gain or answer correctness.",
        },
        {
            "claim_id": "I3",
            "claim": "The realized package exhibits measurable selection shift from the candidate-repository role prior.",
            "support_status": "Supported descriptively",
            "metric": "mean JS(p_R,p_D)",
            "value": float(primary["mean_jsd_package_repository"]),
            "result_file": "04_results/information_v1/selection_bias.csv",
            "caveat": "A distribution shift is not evidence of harmful bias, causal mechanism, or improved support.",
        },
        {
            "claim_id": "I4",
            "claim": "Individual items have positive, zero, or negative leave-one-out alignment contribution inside the fixed realized package.",
            "support_status": "Supported as a post-hoc set-function diagnostic",
            "metric": "fraction of positive item contributions",
            "value": float((primary_loo["loo_alignment_contribution"] > 0).mean()),
            "result_file": "04_results/information_v1/loo_contributions.csv",
            "caveat": "Retrieval is not rerun and ranks are not reassigned; values are not causal effects or evidence entailment scores.",
        },
        {
            "claim_id": "I5",
            "claim": "The fair package can be compared with the locked source-derived accepted-evidence role distribution after retrieval.",
            "support_status": "Supported only as construction-coupled offline audit",
            "metric": "mean JS(p_R,p_G)",
            "value": float(primary["mean_jsd_package_reference"]),
            "result_file": "04_results/information_v1/distribution_diagnostics.csv",
            "caveat": "Accepted alternatives and required slots are source-derived and construction-coupled; p_G never enters retrieval.",
        },
        {
            "claim_id": "I6",
            "claim": "The audit estimates mutual information or true information gain between a query and external knowledge.",
            "support_status": "Unsupported / not estimated",
            "metric": "latent joint-probability information quantity",
            "value": math.nan,
            "result_file": "",
            "caveat": "No calibrated joint distribution or intervention-based causal estimator exists in this run.",
        },
        {
            "claim_id": "I7",
            "claim": "An LLM supplies an independent knowledge source or is responsible for the reported distribution changes.",
            "support_status": "Unsupported / outside this run",
            "metric": "generator intervention",
            "value": math.nan,
            "result_file": "",
            "caveat": "No generator is run. All estimators stop at evidence-package construction, keeping any LLM downstream.",
        },
        {
            "claim_id": "I8",
            "claim": "Observed entropy or divergence causally determines accepted retrieval recall.",
            "support_status": "Unsupported; exploratory associations only",
            "metric": "causal effect",
            "value": math.nan,
            "result_file": "04_results/information_v1/metric_outcome_associations.csv",
            "caveat": "Queries are synthetic and source-derived, estimators share the realized package, and no randomized selection mechanism is present.",
        },
    ]
    return pd.DataFrame(claims)


def write_report(
    diagnostics: pd.DataFrame,
    summary: pd.DataFrame,
    loo: pd.DataFrame,
    claims: pd.DataFrame,
    benchmark_mix: pd.DataFrame,
    config: dict,
) -> None:
    primary = summary[
        summary["population"].eq("All")
        & summary["axis"].eq("evidence_role")
        & summary["condition"].eq(config["primary_condition"])
    ].iloc[0]
    role_all = summary[
        summary["population"].eq("All") & summary["axis"].eq("evidence_role")
    ][
        [
            "method",
            "n_queries",
            "mean_package_k",
            "mean_hnorm_package",
            "mean_jsd_package_repository",
            "mean_jsd_package_query_demand",
            "mean_observable_alignment_shift",
            "mean_jsd_package_reference",
            "mean_loo_alignment_positive_rate",
        ]
    ]
    primary_diag = diagnostics[
        diagnostics["axis"].eq("evidence_role") & diagnostics["condition"].eq(config["primary_condition"])
    ]
    primary_loo = loo[
        loo["axis"].eq("evidence_role")
        & loo["condition"].eq(config["primary_condition"])
        & loo["loo_defined"].eq(True)
    ]
    source_primary = summary[
        summary["population"].eq("Cessna 172")
        & summary["axis"].eq("source_document")
        & summary["condition"].eq(config["primary_condition"])
    ].iloc[0]
    negative_fraction = float((primary_diag["observable_alignment_shift"] <= 0).mean())
    item_negative_fraction = float((primary_loo["loo_alignment_contribution"] < 0).mean())
    locked_mix = benchmark_mix[benchmark_mix["population"].eq("All") & benchmark_mix["facet"].eq("locked_query_type")]
    lines = [
        "# Observable information-distribution diagnostic v1",
        "",
        config["claim_boundary"],
        "",
        "## Estimator contract",
        "",
        "The primary role-space analysis uses four normalized masses on the same repository role support: corpus-count prior `p_D`, visible-question bilingual-lexicon demand `p_q`, rank-discounted fair package `p_R`, and uniformly counted locked accepted-evidence reference `p_G`. The source-document sensitivity replaces the visible lexicon with non-negative word-TFIDF affinity mass. The generator is absent.",
        "",
        "`H_norm(p) = H_2(p)/log2(m)`, `G_align_obs = JS(p_q,p_D)-JS(p_q,p_R)`, and `Delta_i = JS(p_q,p_R_without_i)-JS(p_q,p_R)` for fixed-package leave-one-out. All logarithms are base 2.",
        "",
        "## Primary balanced-IntentRAG role result",
        "",
        (
            f"Across {int(primary['n_queries'])} cue- and track-free synthetic source-derived queries, mean package normalized entropy was "
            f"{primary['mean_hnorm_package']:.4f}. Mean package-to-repository selection shift was {primary['mean_jsd_package_repository']:.4f}; "
            f"mean package-to-visible-demand mismatch was {primary['mean_jsd_package_query_demand']:.4f}. The observable alignment shift was "
            f"{primary['mean_observable_alignment_shift']:.4f} (paired-query bootstrap 95% interval "
            f"{primary['ci_low_observable_alignment_shift']:.4f} to {primary['ci_high_observable_alignment_shift']:.4f})."
        ),
        "",
        (
            f"The shift was non-positive for {negative_fraction:.1%} of queries, and {item_negative_fraction:.1%} of defined item-level leave-one-out contributions were negative. "
            "These negative observations are retained: selection can concentrate a package without moving it closer to the visible-demand estimator, and some selected items make the realized rank-mass profile less aligned."
        ),
        "",
        (
            f"The visible lexicon fell back to `general` for {primary['lexicon_fallback_rate']:.1%} of queries. Mean divergence from the locked query-type role sensitivity was "
            f"{primary['mean_jsd_visible_query_locked_sensitivity']:.4f}; the sensitivity is construction-coupled and never enters retrieval."
        ),
        "",
        "## Method comparison on the role axis",
        "",
        role_all.to_markdown(index=False),
        "",
        "The comparison is descriptive. A lower divergence or higher alignment shift does not by itself imply a better answer, safer action, or more admissible claim.",
        "",
        "## Source-document sensitivity",
        "",
        (
            f"Cessna 172 has three candidate source documents. For balanced IntentRAG, mean source-axis JS(p_R,p_D) was "
            f"{source_primary['mean_jsd_package_repository']:.4f}, source-demand mismatch was {source_primary['mean_jsd_package_query_demand']:.4f}, "
            f"and observable source alignment shift was {source_primary['mean_observable_alignment_shift']:.4f}. Boeing 737 has one ingested source document, so its source entropy and source divergences are structurally zero and carry no discrimination."
        ),
        "",
        "## Locked benchmark demand mix (offline description only)",
        "",
        locked_mix[["category", "count", "mass", "normalized_entropy"]].to_markdown(index=False),
        "",
        "## Leave-one-out boundary",
        "",
        (
            f"There are {len(primary_loo)} defined role-axis item removals for balanced IntentRAG. A positive Delta_i means removal worsens `p_q`/package alignment under fixed original rank weights; "
            f"a negative Delta_i means removal improves it. Retrieval is not rerun, so Delta_i is not a Shapley value, causal effect, entailment score, or proof that an item is dispensable."
        ),
        "",
        "## Claim map",
        "",
        claims[["claim_id", "claim", "support_status", "metric", "value", "caveat"]].to_markdown(index=False),
        "",
    ]
    (REPORTS / "observable_information_distribution_diagnostic.md").write_text("\n".join(lines), encoding="utf-8")


def run(config_path: Path) -> None:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    np.random.seed(int(config["seed"]))
    log_path = LOGS / "run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path, mode="w", encoding="utf-8"), logging.StreamHandler(sys.stdout)],
        force=True,
    )
    logger = logging.getLogger("information_v1")

    aesr_before = tree_hashes(ROOT / "04_results" / "aesr_v2")
    access_path = ROOT / "04_results" / "aesr_v2" / "cue_free_access_query.csv"
    bench_path = ROOT / "04_results" / "v21" / "qa_warehouse_v21_full.csv"
    gold_path = ROOT / "04_results" / "v21" / "v21_gold_sets.csv"
    evidence_path = ROOT / "01_processed_corpus" / "v5" / "evidence_objects.parquet"

    logger.info("[1/7] Load and audit read-only inputs")
    access = pd.read_csv(access_path).fillna("")
    access = access[access["condition"].isin(config["access_conditions"])].copy()
    if access["gold_used_for_retrieval_feature"].astype(str).str.lower().isin({"true", "1"}).any():
        raise AssertionError("Fair access input unexpectedly uses a gold retrieval feature")
    if access["locked_track_used_for_routing"].astype(str).str.lower().isin({"true", "1"}).any():
        raise AssertionError("Fair access input unexpectedly uses locked-track routing")
    expected_rows = access["question_id"].nunique() * len(config["access_conditions"])
    if len(access) != expected_rows:
        raise AssertionError(f"Expected {expected_rows} query-method rows, found {len(access)}")

    bench = pd.read_csv(bench_path).fillna("")
    bench = bench[bench["split"].eq("v21_locked_blind_test")].copy()
    bench = bench[bench["question_id"].astype(str).isin(access["question_id"].astype(str))]
    bench = bench.set_index("question_id", drop=False)
    gold = pd.read_csv(gold_path).fillna("").set_index("question_id")
    ev = pd.read_parquet(evidence_path).fillna("").copy()
    ev["_role_category"] = ev["role"].map(role_value)
    ev["_role_label"] = ev["_role_category"]
    ev["_source_category"] = ev["document_id"].astype(str)
    ev["_source_label"] = [source_label(file_name, document_id) for file_name, document_id in zip(ev["file_name"], ev["document_id"])]
    if ev["evidence_id"].duplicated().any():
        raise AssertionError("Evidence IDs are not unique")
    ev_by_id = ev.set_index("evidence_id", drop=False)
    axes = list(config["distribution_axes"])
    priors = build_repository_priors(ev, axes)

    logger.info("[2/7] Fit exact word-TFIDF space used for source-affinity sensitivity")
    v21 = load_v21_module()
    texts = ev["text"].map(v21.norm).tolist()
    word = TfidfVectorizer(
        tokenizer=v21.tokenize,
        token_pattern=None,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.98,
    )
    xw = normalize(word.fit_transform(texts), copy=False)
    aircraft_positions = {
        str(aircraft): np.flatnonzero(ev["aircraft_model"].astype(str).eq(str(aircraft)).to_numpy())
        for aircraft in ev["aircraft_model"].unique()
    }

    logger.info("[3/7] Reconstruct p_D, p_q, p_R, p_G and query-level statistics")
    repository_rows: list[dict] = []
    for (aircraft, axis), prior in priors.items():
        for category, label, count, mass in zip(prior["support"], prior["labels"], prior["counts"], prior["mass"]):
            repository_rows.append(
                {
                    "experiment_id": config["experiment_id"],
                    "aircraft_model": aircraft,
                    "axis": axis,
                    "category": category,
                    "category_label": label,
                    "candidate_evidence_count": int(count),
                    "repository_mass": float(mass),
                    "distribution_estimator": "normalized candidate evidence-object count",
                    "calibrated_probability": False,
                }
            )

    query_demand_rows: list[dict] = []
    reference_rows: list[dict] = []
    package_rows: list[dict] = []
    selection_rows: list[dict] = []
    diagnostic_rows: list[dict] = []
    loo_rows: list[dict] = []
    query_cache: dict[tuple[str, str], dict] = {}
    epsilon = float(config["log_ratio_smoothing"])

    for query_position, (question_id, bench_row) in enumerate(bench.sort_index().iterrows(), start=1):
        aircraft = str(bench_row["aircraft_model"])
        base_question, removed_cue = split_gold_cue(str(bench_row["question"]))
        candidate_indices = aircraft_positions[aircraft]
        candidate_frame = ev.iloc[candidate_indices]
        query_vector = word.transform([v21.strip_boilerplate(base_question)])
        word_scores = (xw[candidate_indices] @ query_vector.T).toarray().ravel()
        accepted_ids = parse_ids(gold.loc[question_id, "accepted_gold_evidence_ids"])

        for axis in axes:
            prior = priors[(aircraft, axis)]
            support = prior["support"]
            labels = prior["labels"]
            p_d = prior["mass"]
            if axis == "evidence_role":
                p_q, q_raw, match_count, matched_terms, fallback = visible_role_demand(
                    base_question,
                    support,
                    config["role_lexicon"],
                    p_d,
                )
                p_q_locked = locked_role_sensitivity(
                    str(bench_row["query_type"]), support, config["locked_query_type_role_map"]
                )
                query_estimator = "normalized distinct bilingual visible-text role-lexicon matches"
                positive_affinity_items = math.nan
            else:
                p_q, q_raw, fallback, positive_affinity_items = source_affinity_demand(
                    word_scores,
                    candidate_frame,
                    support,
                    float(config["affinity_power"]),
                    p_d,
                )
                p_q_locked = None
                match_count = int(positive_affinity_items)
                matched_terms = ""
                query_estimator = "normalized non-negative word-TFIDF affinity mass aggregated by source"
            p_g, g_counts, reference_size = build_reference_mass(accepted_ids, ev_by_id, axis, support)
            query_cache[(str(question_id), axis)] = {
                "p_d": p_d,
                "p_q": p_q,
                "p_q_locked": p_q_locked,
                "p_g": p_g,
                "q_raw": q_raw,
                "fallback": fallback,
                "match_count": match_count,
                "matched_terms": matched_terms,
                "support": support,
                "labels": labels,
                "reference_size": reference_size,
                "query_estimator": query_estimator,
                "candidate_count": prior["candidate_count"],
            }
            for index, (category, label) in enumerate(zip(support, labels)):
                query_demand_rows.append(
                    {
                        "experiment_id": config["experiment_id"],
                        "question_id": question_id,
                        "aircraft_model": aircraft,
                        "query_type_locked_offline": bench_row["query_type"],
                        "axis": axis,
                        "category": category,
                        "category_label": label,
                        "query_demand_raw_score": float(q_raw[index]),
                        "query_demand_mass": float(p_q[index]),
                        "locked_query_type_sensitivity_mass": (
                            float(p_q_locked[index]) if p_q_locked is not None else math.nan
                        ),
                        "query_estimator": query_estimator,
                        "visible_query_fallback": bool(fallback),
                        "visible_lexicon_or_positive_affinity_count": match_count,
                        "matched_visible_terms": matched_terms,
                        "gold_used_in_primary_query_demand": False,
                        "locked_sensitivity_used_for_retrieval": False,
                        "synthetic_source_derived_query": True,
                    }
                )
                reference_rows.append(
                    {
                        "experiment_id": config["experiment_id"],
                        "question_id": question_id,
                        "aircraft_model": aircraft,
                        "axis": axis,
                        "category": category,
                        "category_label": label,
                        "accepted_evidence_count": int(g_counts[index]),
                        "reference_mass": float(p_g[index]),
                        "accepted_reference_size": int(reference_size),
                        "locked_offline_only": True,
                        "source_derived": True,
                        "construction_coupled": True,
                        "used_for_retrieval": False,
                    }
                )

        if query_position % 50 == 0:
            logger.info("Prepared demand/reference distributions for %d/%d queries", query_position, len(bench))

    for row_position, (_, access_row) in enumerate(access.sort_values(["condition", "question_id"]).iterrows(), start=1):
        question_id = str(access_row["question_id"])
        aircraft = str(access_row["aircraft_model"])
        condition = str(access_row["condition"])
        method = METHOD_LABELS.get(condition, str(access_row["method"]))
        top_ids = parse_ids(access_row["top_evidence_ids"])
        package = package_frame(top_ids, ev_by_id)
        if not package["aircraft_model"].astype(str).eq(aircraft).all():
            raise AssertionError(f"Cross-aircraft package for {question_id}/{condition}")

        for axis in axes:
            cached = query_cache[(question_id, axis)]
            support = cached["support"]
            labels = cached["labels"]
            p_d = cached["p_d"]
            p_q = cached["p_q"]
            p_q_locked = cached["p_q_locked"]
            p_g = cached["p_g"]
            p_r, package_counts = aggregate_mass(
                package, axis, support, package["_rank_weight"].to_numpy(dtype=float)
            )
            h_d = normalized_entropy(p_d)
            h_q = normalized_entropy(p_q)
            h_r = normalized_entropy(p_r)
            h_g = normalized_entropy(p_g)
            js_qd = js_divergence(p_q, p_d)
            js_rd = js_divergence(p_r, p_d)
            js_rq = js_divergence(p_r, p_q)
            js_rg = js_divergence(p_r, p_g)
            js_qg = js_divergence(p_q, p_g)
            js_visible_locked = (
                js_divergence(p_q, p_q_locked) if p_q_locked is not None else math.nan
            )
            alignment_shift = js_qd - js_rq
            demand_coverage = float(p_q[p_r > 0].sum())
            reference_coverage = float(p_g[p_r > 0].sum())
            full_alignment_utility = 1.0 - js_rq
            category_positions = {category: index for index, category in enumerate(support)}
            package_category_col, _ = category_columns(axis)
            raw_rank_mass = np.zeros(len(support), dtype=float)
            for category, weight in zip(
                package[package_category_col].astype(str), package["_rank_weight"].to_numpy(dtype=float)
            ):
                raw_rank_mass[category_positions[category]] += float(weight)
            total_rank_mass = float(raw_rank_mass.sum())

            item_contributions: list[float] = []
            item_coverage_contributions: list[float] = []
            for evidence_id, item in package.iterrows():
                category = str(item[package_category_col])
                category_index = category_positions[category]
                item_weight = float(item["_rank_weight"])
                loo_defined = len(package) > 1 and total_rank_mass - item_weight > 0
                if loo_defined:
                    without_raw = raw_rank_mass.copy()
                    without_raw[category_index] = max(0.0, without_raw[category_index] - item_weight)
                    p_without, _ = normalized_mass(without_raw)
                    js_without = js_divergence(p_q, p_without)
                    alignment_utility_without = 1.0 - js_without
                    contribution = full_alignment_utility - alignment_utility_without
                    coverage_without = float(p_q[p_without > 0].sum())
                    coverage_contribution = demand_coverage - coverage_without
                    entropy_without = normalized_entropy(p_without)
                    entropy_contribution = h_r - entropy_without
                    item_contributions.append(float(contribution))
                    item_coverage_contributions.append(float(coverage_contribution))
                else:
                    js_without = math.nan
                    alignment_utility_without = math.nan
                    contribution = math.nan
                    coverage_without = math.nan
                    coverage_contribution = math.nan
                    entropy_without = math.nan
                    entropy_contribution = math.nan
                loo_rows.append(
                    {
                        "experiment_id": config["experiment_id"],
                        "question_id": question_id,
                        "aircraft_model": aircraft,
                        "condition": condition,
                        "method": method,
                        "axis": axis,
                        "evidence_id": evidence_id,
                        "rank": int(item["_rank"]),
                        "fixed_original_rank_weight": item_weight,
                        "category": category,
                        "category_label": labels[category_index],
                        "evidence_role": item["_role_category"],
                        "source_document": item["_source_label"],
                        "query_demand_mass_at_item_category": float(p_q[category_index]),
                        "repository_mass_at_item_category": float(p_d[category_index]),
                        "package_mass_at_item_category": float(p_r[category_index]),
                        "reference_mass_at_item_category": float(p_g[category_index]),
                        "full_jsd_package_query_demand": js_rq,
                        "without_item_jsd_package_query_demand": js_without,
                        "full_alignment_utility": full_alignment_utility,
                        "without_item_alignment_utility": alignment_utility_without,
                        "loo_alignment_contribution": contribution,
                        "full_demand_mass_covered": demand_coverage,
                        "without_item_demand_mass_covered": coverage_without,
                        "loo_demand_coverage_contribution": coverage_contribution,
                        "full_package_hnorm": h_r,
                        "without_item_package_hnorm": entropy_without,
                        "loo_entropy_contribution": entropy_contribution,
                        "loo_defined": bool(loo_defined),
                        "retrieval_rerun": False,
                        "ranks_reassigned": False,
                        "causal_interpretation_allowed": False,
                    }
                )

            for index, (category, label) in enumerate(zip(support, labels)):
                package_rows.append(
                    {
                        "experiment_id": config["experiment_id"],
                        "question_id": question_id,
                        "aircraft_model": aircraft,
                        "condition": condition,
                        "method": method,
                        "axis": axis,
                        "category": category,
                        "category_label": label,
                        "package_item_count": int(package_counts[index]),
                        "package_mass": float(p_r[index]),
                        "package_k": int(len(package)),
                        "package_estimator": "normalized discounted original-rank mass",
                        "rank_weight_formula": config["rank_weight_formula"],
                        "gold_used_for_package_construction": False,
                    }
                )
                selection_rows.append(
                    {
                        "experiment_id": config["experiment_id"],
                        "question_id": question_id,
                        "aircraft_model": aircraft,
                        "condition": condition,
                        "method": method,
                        "axis": axis,
                        "category": category,
                        "category_label": label,
                        "repository_mass": float(p_d[index]),
                        "query_demand_mass": float(p_q[index]),
                        "package_mass": float(p_r[index]),
                        "reference_mass": float(p_g[index]),
                        "package_minus_repository_mass": float(p_r[index] - p_d[index]),
                        "package_minus_query_demand_mass": float(p_r[index] - p_q[index]),
                        "log2_package_to_repository_ratio": float(
                            math.log2((p_r[index] + epsilon) / (p_d[index] + epsilon))
                        ),
                        "log2_package_to_query_demand_ratio": float(
                            math.log2((p_r[index] + epsilon) / (p_q[index] + epsilon))
                        ),
                        "log_ratio_smoothing": epsilon,
                        "selection_bias_interpretation": "descriptive normalized-mass shift only",
                    }
                )

            defined_contributions = np.asarray(item_contributions, dtype=float)
            diagnostic_rows.append(
                {
                    "experiment_id": config["experiment_id"],
                    "question_id": question_id,
                    "aircraft_model": aircraft,
                    "query_type_locked_offline": access_row["query_type"],
                    "condition": condition,
                    "method": method,
                    "axis": axis,
                    "candidate_evidence_count": int(cached["candidate_count"]),
                    "candidate_category_count": int(len(support)),
                    "package_k": int(len(package)),
                    "package_category_count": int((p_r > 0).sum()),
                    "reference_evidence_count": int(cached["reference_size"]),
                    "query_estimator": cached["query_estimator"],
                    "visible_query_fallback": bool(cached["fallback"]),
                    "visible_lexicon_or_positive_affinity_count": cached["match_count"],
                    "hnorm_repository": h_d,
                    "hnorm_query_demand": h_q,
                    "hnorm_package": h_r,
                    "hnorm_reference": h_g,
                    "jsd_query_demand_repository": js_qd,
                    "jsd_package_repository": js_rd,
                    "jsd_package_query_demand": js_rq,
                    "jsd_package_reference": js_rg,
                    "jsd_query_demand_reference": js_qg,
                    "jsd_visible_query_locked_sensitivity": js_visible_locked,
                    "observable_alignment_shift": alignment_shift,
                    "normalized_entropy_shift_query_to_package": h_q - h_r,
                    "demand_mass_covered": demand_coverage,
                    "reference_mass_covered": reference_coverage,
                    "package_top_category_mass": float(p_r.max()),
                    "loo_alignment_mean": (
                        float(defined_contributions.mean()) if len(defined_contributions) else math.nan
                    ),
                    "loo_alignment_positive_rate": (
                        float((defined_contributions > 0).mean()) if len(defined_contributions) else math.nan
                    ),
                    "loo_alignment_negative_rate": (
                        float((defined_contributions < 0).mean()) if len(defined_contributions) else math.nan
                    ),
                    "loo_alignment_zero_rate": (
                        float(np.isclose(defined_contributions, 0.0, atol=1e-15).mean())
                        if len(defined_contributions)
                        else math.nan
                    ),
                    "axis_degenerate": bool(len(support) <= 1),
                    "offline_accepted_recall_10": access_row["offline_accepted_recall_10"],
                    "offline_construction_composite": access_row["offline_construction_composite"],
                    "is_synthetic_query": True,
                    "gold_used_for_retrieval_feature": False,
                    "locked_track_used_for_routing": False,
                    "locked_reference_used_posthoc_only": True,
                    "calibrated_epistemic_probability": False,
                    "mutual_information_estimated": False,
                    "causal_effect_estimated": False,
                }
            )

        if row_position % 300 == 0:
            logger.info("Processed %d/%d query-method packages", row_position, len(access))

    repository = pd.DataFrame(repository_rows)
    query_demand = pd.DataFrame(query_demand_rows)
    reference = pd.DataFrame(reference_rows)
    retrieved = pd.DataFrame(package_rows)
    selection = pd.DataFrame(selection_rows)
    diagnostics = pd.DataFrame(diagnostic_rows)
    loo = pd.DataFrame(loo_rows)
    benchmark_mix = make_benchmark_demand_mix(bench.reset_index(drop=True))

    repository.to_csv(RAW / "repository_prior.csv", index=False, encoding="utf-8-sig")
    query_demand.to_csv(RAW / "query_demand_distribution.csv", index=False, encoding="utf-8-sig")
    reference.to_csv(RAW / "locked_reference_distribution.csv", index=False, encoding="utf-8-sig")
    retrieved.to_csv(RAW / "retrieved_package_distribution.csv", index=False, encoding="utf-8-sig")
    benchmark_mix.to_csv(RAW / "benchmark_demand_mix.csv", index=False, encoding="utf-8-sig")
    diagnostics.to_csv(RESULTS / "distribution_diagnostics.csv", index=False, encoding="utf-8-sig")
    selection.to_csv(RESULTS / "selection_bias.csv", index=False, encoding="utf-8-sig")
    loo.to_csv(RESULTS / "loo_contributions.csv", index=False, encoding="utf-8-sig")

    logger.info("[4/7] Summaries, exploratory associations, and claim traceability")
    summary = summarize_diagnostics(diagnostics, config)
    associations = outcome_associations(diagnostics, config)
    claims = build_claim_map(diagnostics, summary, loo, config)
    summary.to_csv(RESULTS / "method_axis_summary.csv", index=False, encoding="utf-8-sig")
    associations.to_csv(RESULTS / "metric_outcome_associations.csv", index=False, encoding="utf-8-sig")
    claims.to_csv(RESULTS / "claim_result_map.csv", index=False, encoding="utf-8-sig")

    estimator_definitions = pd.DataFrame(
        [
            ("p_D", "repository prior", "candidate evidence-object counts normalized on the axis support", "observable empirical mass"),
            ("p_q role", "visible query-demand proxy", "distinct bilingual visible-question lexicon matches normalized by evidence role", "observable lexical proxy"),
            ("p_q source", "visible source-affinity proxy", "non-negative word-TFIDF evidence affinity summed by document and normalized", "observable affinity proxy"),
            ("p_R", "retrieved package", "realized Top-K category mass weighted by 1/log2(rank+1)", "observable empirical rank mass"),
            ("p_G", "locked reference", "unique accepted source-derived evidence identifiers counted uniformly by category", "offline construction-coupled audit mass"),
            ("H_norm", "normalized entropy", "-sum p log2(p) / log2(number of repository categories)", "estimator dispersion only"),
            ("JS", "Jensen-Shannon divergence", "base-2 symmetric divergence between two normalized estimator masses", "estimator mismatch only"),
            ("G_align_obs", "observable alignment shift", "JS(p_q,p_D) - JS(p_q,p_R)", "not true information gain"),
            ("Delta_i", "fixed-package LOO contribution", "JS(p_q,p_R_without_i) - JS(p_q,p_R)", "no retrieval rerun; not causal"),
        ],
        columns=["symbol", "name", "estimator", "allowed_interpretation"],
    )
    write_table(estimator_definitions, "table_a_estimator_definitions")
    write_table(
        summary[summary["population"].eq("All") & summary["axis"].eq("evidence_role")],
        "table_b_method_role_diagnostics",
    )
    primary_distribution = selection[
        selection["axis"].eq("evidence_role") & selection["condition"].eq(config["primary_condition"])
    ].groupby("category_label")[["repository_mass", "query_demand_mass", "package_mass", "reference_mass"]].mean().reset_index()
    write_table(primary_distribution, "table_c_primary_role_distributions")
    write_table(associations, "table_d_exploratory_outcome_associations")
    primary_role_diag = diagnostics[
        diagnostics["axis"].eq("evidence_role")
        & diagnostics["condition"].eq(config["primary_condition"])
    ]
    primary_role_loo = loo[
        loo["axis"].eq("evidence_role")
        & loo["condition"].eq(config["primary_condition"])
        & loo["loo_defined"].eq(True)
    ]
    primary_role_summary = summary[
        summary["population"].eq("All")
        & summary["axis"].eq("evidence_role")
        & summary["condition"].eq(config["primary_condition"])
    ].iloc[0]
    cessna_source_summary = summary[
        summary["population"].eq("Cessna 172")
        & summary["axis"].eq("source_document")
        & summary["condition"].eq(config["primary_condition"])
    ].iloc[0]
    negative_results = pd.DataFrame(
        [
            (
                "role_alignment_shift_mean",
                float(primary_role_summary["mean_observable_alignment_shift"]),
                f"95% bootstrap interval [{primary_role_summary['ci_low_observable_alignment_shift']:.6f}, {primary_role_summary['ci_high_observable_alignment_shift']:.6f}]",
            ),
            (
                "role_queries_with_nonpositive_alignment_shift_fraction",
                float((primary_role_diag["observable_alignment_shift"] <= 0).mean()),
                "balanced IntentRAG; retained without selective filtering",
            ),
            (
                "role_loo_negative_item_fraction",
                float((primary_role_loo["loo_alignment_contribution"] < 0).mean()),
                "fixed package; no retrieval rerun",
            ),
            (
                "role_loo_zero_item_fraction",
                float(np.isclose(primary_role_loo["loo_alignment_contribution"], 0.0, atol=1e-15).mean()),
                "fixed package; no retrieval rerun",
            ),
            (
                "visible_vs_locked_role_sensitivity_jsd_mean",
                float(primary_role_summary["mean_jsd_visible_query_locked_sensitivity"]),
                "locked query type is offline and construction-coupled",
            ),
            (
                "package_vs_locked_reference_role_jsd_mean",
                float(primary_role_summary["mean_jsd_package_reference"]),
                "p_G is accepted-evidence-derived and post-hoc only",
            ),
            (
                "cessna_source_alignment_shift_mean",
                float(cessna_source_summary["mean_observable_alignment_shift"]),
                "three-document source axis; word-TFIDF source-affinity demand proxy",
            ),
            (
                "boeing_source_axis_category_count",
                1.0,
                "degenerate single-document source axis; entropy and JSD are structurally zero",
            ),
        ],
        columns=["diagnostic", "value", "interpretive_boundary"],
    )
    write_table(negative_results, "table_e_retained_negative_results")

    logger.info("[5/7] Data-backed composite figures")
    make_figures(diagnostics, selection, loo, summary, config)

    logger.info("[6/7] Report and immutable-input audit")
    write_report(diagnostics, summary, loo, claims, benchmark_mix, config)
    source_files = [
        Path(__file__).resolve(),
        config_path.resolve(),
        ROOT / "_pipeline_information_v1" / "validate_outputs.py",
        ROOT / "_pipeline_v21" / "p0_build_eval_v21.py",
        access_path,
        bench_path,
        gold_path,
        evidence_path,
        ROOT / "07_reports" / "aesr_v2" / "run_manifest.json",
    ]
    source_manifest_rows = []
    for path in source_files:
        source_manifest_rows.append(
            {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
                "size_bytes": path.stat().st_size,
                "purpose": "code/config" if path.suffix in {".py", ".json"} and "results" not in str(path) else "read-only input",
            }
        )
    pd.DataFrame(source_manifest_rows).to_csv(RAW / "source_file_manifest.csv", index=False, encoding="utf-8-sig")

    aesr_after = tree_hashes(ROOT / "04_results" / "aesr_v2")
    all_aesr_paths = sorted(set(aesr_before) | set(aesr_after))
    immutability = pd.DataFrame(
        [
            {
                "path": path,
                "sha256_before": aesr_before.get(path, ""),
                "sha256_after": aesr_after.get(path, ""),
                "unchanged": aesr_before.get(path) == aesr_after.get(path),
            }
            for path in all_aesr_paths
        ]
    )
    immutability.to_csv(RAW / "aesr_v2_immutability_audit.csv", index=False, encoding="utf-8-sig")
    if not immutability["unchanged"].all():
        raise AssertionError("An AESR-v2 result changed during the information-v1 run")

    logger.info("[7/7] Run manifest")
    manifest = {
        "experiment_id": config["experiment_id"],
        "run_timestamp_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": config["seed"],
        "n_queries": int(access["question_id"].nunique()),
        "n_methods": int(access["condition"].nunique()),
        "n_query_method_rows": int(len(access)),
        "n_evidence_objects": int(len(ev)),
        "n_distribution_diagnostic_rows": int(len(diagnostics)),
        "n_selection_category_rows": int(len(selection)),
        "n_loo_rows": int(len(loo)),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "source_hashes": {row["path"]: row["sha256"] for row in source_manifest_rows},
        "aesr_v2_result_files_verified_unchanged": bool(immutability["unchanged"].all()),
        "primary_condition": config["primary_condition"],
        "axes": axes,
        "distribution_estimators": {
            "p_D": "normalized candidate evidence-object count",
            "p_q_role": "normalized distinct bilingual visible-text role-lexicon matches",
            "p_q_source": "normalized non-negative word-TFIDF affinity mass",
            "p_R": "normalized discounted original-rank mass",
            "p_G": "normalized accepted source-derived evidence-object count; offline only",
        },
        "locked_offline_only": ["query_type sensitivity", "accepted evidence IDs", "retrieval outcome metrics"],
        "claim_boundary": config["claim_boundary"],
        "result_files": sorted(str(path.relative_to(ROOT)) for path in RESULTS.rglob("*.csv")),
        "figure_files": sorted(str(path.relative_to(ROOT)) for path in FIGURES.glob("*.png")),
        "table_files": sorted(str(path.relative_to(ROOT)) for path in TABLES.glob("*")),
    }
    (REPORTS / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(
        "complete experiment=%s diagnostics=%d loo=%d",
        config["experiment_id"],
        len(diagnostics),
        len(loo),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    run(args.config.resolve())


if __name__ == "__main__":
    main()
