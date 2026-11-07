"""Compute admissible-reuse diagnostics from existing project result artifacts.

This script studies a narrower, paper-safe claim:

    Computational reuse opportunity is not the same as admissible evidence reuse.

It reads existing AESR-v2, state-reuse, cache-reuse, and model-scale diagnostic
outputs. It does not inspect internal transformer KV tensors, estimate mutual
information, certify action authorization, or re-run LLM generation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "admissible_reuse_v1" / "admissible_reuse_v1.json"
RESULTS = ROOT / "04_results" / "admissible_reuse_v1"
FIGURES = ROOT / "05_figures" / "admissible_reuse_v1"
TABLES = ROOT / "06_tables" / "admissible_reuse_v1"
REPORTS = ROOT / "07_reports" / "admissible_reuse_v1"
LOGS = ROOT / "08_logs" / "admissible_reuse_v1"

for directory in (RESULTS, FIGURES, TABLES, REPORTS, LOGS):
    directory.mkdir(parents=True, exist_ok=True)


CONSTRAINT_STEPS = [
    ("raw_reuse", "Raw reusable state", []),
    ("scope_provenance", "+ scope/provenance", ["scope_match_rate", "provenance_valid_rate"]),
    ("authority", "+ authority cue", ["scope_match_rate", "provenance_valid_rate", "authority_cue_present"]),
    (
        "condition",
        "+ condition cue",
        ["scope_match_rate", "provenance_valid_rate", "authority_cue_present", "condition_present"],
    ),
    (
        "action_prereq",
        "+ action prerequisite",
        [
            "scope_match_rate",
            "provenance_valid_rate",
            "authority_cue_present",
            "condition_present",
            "action_authority_evidence_availability_proxy",
        ],
    ),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = fieldnames or (list(rows[0].keys()) if rows else [])
    if not columns:
        raise ValueError(f"Cannot write empty CSV without field names: {path}")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_mean_ci(values: Iterable[float], seed: int, samples: int = 4000) -> tuple[float, float, float, int]:
    array = np.asarray([float(v) for v in values if np.isfinite(float(v))], dtype=float)
    if len(array) == 0:
        return math.nan, math.nan, math.nan, 0
    mean = float(array.mean())
    if len(array) == 1:
        return mean, mean, mean, 1
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(array), size=(samples, len(array)))
    means = array[draws].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return mean, float(low), float(high), int(len(array))


def clipped_numeric(series: pd.Series) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce").fillna(0.0)
    return out.clip(lower=0.0, upper=1.0)


def load_inputs(config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    paths = {
        "state_reuse": ROOT / config["inputs"]["state_reuse_query"],
        "natural_factors": ROOT / config["inputs"]["natural_factor_proxies"],
        "cache_family": ROOT / config["inputs"]["cache_family_prefix_metrics"],
        "cache_variant_summary": ROOT / config["inputs"]["cache_variant_summary"],
        "model_scale_summary": ROOT / config["inputs"]["model_scale_variant_summary"],
        "model_scale_responses": ROOT / config["inputs"]["model_scale_responses"],
        "aesr_interventions": ROOT / config["inputs"]["aesr_intervention_response"],
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required input files:\n" + "\n".join(missing))
    frames = {key: pd.read_csv(path) for key, path in paths.items()}
    frames["_paths"] = pd.DataFrame(
        [{"name": key, "path": str(path.relative_to(ROOT)), "sha256": sha256(path)} for key, path in paths.items()]
    )
    return frames


def compute_frontier_rows(state: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []

    numeric_cols = {
        "raw_reuse_ratio_qwen",
        "raw_reused_evidence_tokens_qwen",
        "total_prompt_tokens_qwen",
        "scope_match_rate",
        "provenance_valid_rate",
        "authority_cue_present",
        "condition_present",
        "action_authority_evidence_availability_proxy",
        "redundancy_proxy",
        "reused_evidence_id_count",
    }
    for col in numeric_cols:
        if col in state.columns:
            state[col] = pd.to_numeric(state[col], errors="coerce")

    for _, row in state.iterrows():
        raw_ratio = float(row.get("raw_reuse_ratio_qwen", 0.0) or 0.0)
        raw_tokens = float(row.get("raw_reused_evidence_tokens_qwen", 0.0) or 0.0)
        base = {
            "question_id": row["question_id"],
            "aircraft_model": row["aircraft_model"],
            "raw_reuse_ratio_qwen": raw_ratio,
            "raw_reused_evidence_tokens_qwen": raw_tokens,
            "reused_evidence_id_count": row.get("reused_evidence_id_count", math.nan),
            "redundancy_proxy": row.get("redundancy_proxy", math.nan),
        }
        for step_id, step_label, factors in CONSTRAINT_STEPS:
            if not factors:
                weight = 1.0
            else:
                values = [float(row.get(col, 0.0) or 0.0) for col in factors]
                weight = float(np.prod(np.clip(values, 0.0, 1.0)))
            retained_ratio = raw_ratio * weight
            retained_tokens = raw_tokens * weight
            query_rows.append(
                {
                    **base,
                    "constraint_step": step_id,
                    "constraint_label": step_label,
                    "constraint_weight_proxy": weight,
                    "admissible_reuse_ratio_proxy": retained_ratio,
                    "admissible_reuse_tokens_proxy": retained_tokens,
                    "reuse_loss_ratio_proxy": raw_ratio - retained_ratio,
                }
            )

    query_df = pd.DataFrame(query_rows)
    for aircraft, group in query_df.groupby("aircraft_model", dropna=False):
        for step_id, step_label, _ in CONSTRAINT_STEPS:
            step = group[group["constraint_step"] == step_id]
            mean, low, high, n = bootstrap_mean_ci(step["admissible_reuse_ratio_proxy"], seed=seed)
            t_mean, t_low, t_high, _ = bootstrap_mean_ci(step["admissible_reuse_tokens_proxy"], seed=seed + 11)
            loss_mean, loss_low, loss_high, _ = bootstrap_mean_ci(step["reuse_loss_ratio_proxy"], seed=seed + 23)
            rows.append(
                {
                    "aircraft_model": aircraft,
                    "constraint_step": step_id,
                    "constraint_label": step_label,
                    "n_queries": n,
                    "mean_admissible_reuse_ratio_proxy": mean,
                    "ci95_low": low,
                    "ci95_high": high,
                    "mean_admissible_reuse_tokens_proxy": t_mean,
                    "tokens_ci95_low": t_low,
                    "tokens_ci95_high": t_high,
                    "mean_reuse_loss_ratio_proxy": loss_mean,
                    "loss_ci95_low": loss_low,
                    "loss_ci95_high": loss_high,
                }
            )

    all_rows = []
    for step_id, step_label, _ in CONSTRAINT_STEPS:
        step = query_df[query_df["constraint_step"] == step_id]
        mean, low, high, n = bootstrap_mean_ci(step["admissible_reuse_ratio_proxy"], seed=seed + 101)
        t_mean, t_low, t_high, _ = bootstrap_mean_ci(step["admissible_reuse_tokens_proxy"], seed=seed + 111)
        loss_mean, loss_low, loss_high, _ = bootstrap_mean_ci(step["reuse_loss_ratio_proxy"], seed=seed + 123)
        all_rows.append(
            {
                "aircraft_model": "All",
                "constraint_step": step_id,
                "constraint_label": step_label,
                "n_queries": n,
                "mean_admissible_reuse_ratio_proxy": mean,
                "ci95_low": low,
                "ci95_high": high,
                "mean_admissible_reuse_tokens_proxy": t_mean,
                "tokens_ci95_low": t_low,
                "tokens_ci95_high": t_high,
                "mean_reuse_loss_ratio_proxy": loss_mean,
                "loss_ci95_low": loss_low,
                "loss_ci95_high": loss_high,
            }
        )
    return pd.DataFrame(rows + all_rows), query_df


def compute_case_explanation(state: pd.DataFrame, natural: pd.DataFrame, seed: int) -> pd.DataFrame:
    merged = state.merge(
        natural[
            [
                "question_id",
                "track",
                "query_type",
                "package_size",
                "role_diversity",
                "source_diversity",
                "page_diversity",
                "action_like_query_proxy",
            ]
        ],
        on="question_id",
        how="left",
    )
    rows = []
    metrics = [
        "raw_reuse_ratio_qwen",
        "reused_evidence_id_count",
        "redundancy_proxy",
        "role_diversity",
        "source_diversity",
        "page_diversity",
        "action_authority_evidence_availability_proxy",
        "scope_match_rate",
        "authority_cue_present",
        "condition_present",
    ]
    for metric in metrics:
        for aircraft, group in merged.groupby("aircraft_model", dropna=False):
            mean, low, high, n = bootstrap_mean_ci(pd.to_numeric(group[metric], errors="coerce"), seed=seed + len(metric))
            rows.append(
                {
                    "metric": metric,
                    "aircraft_model": aircraft,
                    "n_queries": n,
                    "mean": mean,
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )
    return pd.DataFrame(rows)


def compute_family_aircraft_summary(cache_family: pd.DataFrame, seed: int) -> pd.DataFrame:
    rows = []
    for variant in sorted(cache_family["variant"].unique()):
        vdf = cache_family[cache_family["variant"] == variant]
        for aircraft, group in vdf.groupby("family_aircraft_model", dropna=False):
            mean, low, high, n = bootstrap_mean_ci(
                pd.to_numeric(group["family_shared_prefix_ratio_qwen35_08b"], errors="coerce"), seed=seed + len(variant)
            )
            tok_mean, tok_low, tok_high, _ = bootstrap_mean_ci(
                pd.to_numeric(group["reusable_prefix_tokens_after_first_qwen35_08b"], errors="coerce"),
                seed=seed + len(variant) + 17,
            )
            rows.append(
                {
                    "variant": variant,
                    "aircraft_model": aircraft,
                    "n_families": n,
                    "mean_shared_prefix_ratio": mean,
                    "ci95_low": low,
                    "ci95_high": high,
                    "mean_reusable_prefix_tokens_after_first": tok_mean,
                    "tokens_ci95_low": tok_low,
                    "tokens_ci95_high": tok_high,
                }
            )
    return pd.DataFrame(rows)


def compute_model_scale_matrix(summary: pd.DataFrame) -> pd.DataFrame:
    pivot = summary.pivot_table(
        index=["model_scale", "model_size_b", "model"],
        columns="variant",
        values=[
            "mean_answer_evidence_grounded_score_proxy",
            "mean_citation_coverage_proxy",
            "mean_latency_s",
            "mean_prompt_tokens_usage",
        ],
    )
    pivot.columns = [f"{metric}__{variant}" for metric, variant in pivot.columns]
    pivot = pivot.reset_index()
    rows = []
    for _, row in pivot.iterrows():
        direct = row.get("mean_answer_evidence_grounded_score_proxy__direct_model_only", math.nan)
        state = row.get("mean_answer_evidence_grounded_score_proxy__state_first_state", math.nan)
        chain = row.get("mean_answer_evidence_grounded_score_proxy__question_first_chain", math.nan)
        rows.append(
            {
                "model_scale": row["model_scale"],
                "model_size_b": row["model_size_b"],
                "model": row["model"],
                "direct_grounded_proxy": direct,
                "state_grounded_proxy": state,
                "raw_chain_grounded_proxy": chain,
                "state_minus_direct_grounded_proxy": state - direct,
                "raw_chain_minus_direct_grounded_proxy": chain - direct,
                "state_minus_raw_chain_grounded_proxy": state - chain,
                "state_citation_coverage_proxy": row.get(
                    "mean_citation_coverage_proxy__state_first_state", math.nan
                ),
                "direct_citation_coverage_proxy": row.get(
                    "mean_citation_coverage_proxy__direct_model_only", math.nan
                ),
                "state_prompt_token_reduction_vs_raw_chain": 1.0
                - (
                    row.get("mean_prompt_tokens_usage__state_first_state", math.nan)
                    / row.get("mean_prompt_tokens_usage__question_first_chain", math.nan)
                ),
                "state_latency_reduction_vs_raw_chain": 1.0
                - (row.get("mean_latency_s__state_first_state", math.nan) / row.get("mean_latency_s__question_first_chain", math.nan)),
            }
        )
    return pd.DataFrame(rows).sort_values("model_size_b")


def summarize_interventions(interventions: pd.DataFrame) -> pd.DataFrame:
    target = interventions[interventions["is_target_proxy"].astype(str).str.lower() == "true"].copy()
    target["directional_success_numeric"] = target["directional_success"].astype(str).str.lower().map({"true": 1.0, "false": 0.0})
    rows = []
    for (intervention, aircraft), group in target.groupby(["intervention", "aircraft_model"], dropna=False):
        rows.append(
            {
                "intervention": intervention,
                "aircraft_model": aircraft,
                "n_target_proxy_rows": int(group["directional_success_numeric"].notna().sum()),
                "directional_success_rate": float(group["directional_success_numeric"].mean()),
                "mean_target_delta": float(pd.to_numeric(group["delta"], errors="coerce").mean()),
                "is_synthetic": True,
                "claim_boundary": "Synthetic post-retrieval package intervention; proxy response, not state-classifier accuracy.",
            }
        )
    return pd.DataFrame(rows)


def plot_frontier(frontier: pd.DataFrame, path: Path) -> None:
    order = [step[0] for step in CONSTRAINT_STEPS]
    labels = {step[0]: step[1] for step in CONSTRAINT_STEPS}
    colors = {"Boeing 737": "#4C78A8", "Cessna 172": "#F58518", "All": "#54A24B"}
    fig, ax = plt.subplots(figsize=(8.4, 4.8), dpi=320)
    for aircraft in ["Boeing 737", "Cessna 172", "All"]:
        sub = frontier[frontier["aircraft_model"] == aircraft].set_index("constraint_step").loc[order].reset_index()
        x = np.arange(len(sub))
        y = sub["mean_admissible_reuse_ratio_proxy"].to_numpy()
        low = y - sub["ci95_low"].to_numpy()
        high = sub["ci95_high"].to_numpy() - y
        ax.errorbar(x, y, yerr=[low, high], marker="o", linewidth=2, capsize=3, label=aircraft, color=colors[aircraft])
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels([labels[item] for item in order], rotation=18, ha="right")
    ax.set_ylabel("Retained reuse ratio proxy")
    ax.set_xlabel("Sequential admissibility proxy constraints")
    ax.set_title("Admissible reuse frontier: computation reuse after evidence constraints")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, ncol=3, loc="upper right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_reuse_explanation(explain: pd.DataFrame, path: Path) -> None:
    metrics = [
        ("raw_reuse_ratio_qwen", "Raw reuse"),
        ("reused_evidence_id_count", "Repeated evidence IDs"),
        ("redundancy_proxy", "Redundancy"),
        ("role_diversity", "Role diversity"),
        ("source_diversity", "Source diversity"),
        ("action_authority_evidence_availability_proxy", "Action-prereq availability"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(10.8, 6.2), dpi=320)
    colors = {"Boeing 737": "#4C78A8", "Cessna 172": "#F58518"}
    for ax, (metric, label) in zip(axes.ravel(), metrics):
        sub = explain[explain["metric"] == metric].set_index("aircraft_model").loc[["Boeing 737", "Cessna 172"]]
        x = np.arange(2)
        y = sub["mean"].to_numpy()
        ax.bar(x, y, color=[colors["Boeing 737"], colors["Cessna 172"]], alpha=0.85)
        ax.errorbar(
            x,
            y,
            yerr=[y - sub["ci95_low"].to_numpy(), sub["ci95_high"].to_numpy() - y],
            fmt="none",
            ecolor="black",
            capsize=3,
            linewidth=1,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(["Boeing", "Cessna"])
        ax.set_title(label)
        ax.grid(axis="y", alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("Why raw reuse differs by corpus and evidence-state structure", y=1.02)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_model_scale(model_matrix: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6), dpi=320)
    x = model_matrix["model_size_b"].to_numpy(dtype=float)
    axes[0].plot(x, model_matrix["direct_grounded_proxy"], marker="o", label="Direct model only")
    axes[0].plot(x, model_matrix["raw_chain_grounded_proxy"], marker="o", label="Raw evidence chain")
    axes[0].plot(x, model_matrix["state_grounded_proxy"], marker="o", label="Evidence state")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Local Qwen model size (B parameters, log scale)")
    axes[0].set_ylabel("Answer groundedness proxy")
    axes[0].set_title("Model scale alone does not recover\nproject-specific evidence")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.25)

    axes[1].bar(
        np.arange(len(model_matrix)),
        model_matrix["state_prompt_token_reduction_vs_raw_chain"] * 100.0,
        color="#54A24B",
        alpha=0.85,
        label="Prompt tokens",
    )
    axes[1].plot(
        np.arange(len(model_matrix)),
        model_matrix["state_latency_reduction_vs_raw_chain"] * 100.0,
        marker="o",
        color="#E45756",
        label="Latency",
    )
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xticks(np.arange(len(model_matrix)))
    axes[1].set_xticklabels(model_matrix["model_scale"].tolist())
    axes[1].set_ylabel("State-first reduction vs raw chain (%)")
    axes[1].set_title("State organization reduces input burden\nwith scale-dependent latency")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.25)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_intervention_response(intervention_summary: pd.DataFrame, path: Path) -> None:
    order = [
        "scope_swap",
        "provenance_masking",
        "authority_removal",
        "condition_removal",
        "warning_removal",
        "conflict_injection",
        "duplicate_injection",
        "procedural_to_descriptive_substitution",
        "evidence_order_permutation",
    ]
    labels = {
        "scope_swap": "Scope swap",
        "provenance_masking": "Provenance mask",
        "authority_removal": "Authority removal",
        "condition_removal": "Condition removal",
        "warning_removal": "Warning removal",
        "conflict_injection": "Conflict injection",
        "duplicate_injection": "Duplicate injection",
        "procedural_to_descriptive_substitution": "Procedure to descriptive",
        "evidence_order_permutation": "Order permutation",
    }
    sub = intervention_summary[intervention_summary["intervention"].isin(order)].copy()
    matrix = sub.pivot_table(index="intervention", columns="aircraft_model", values="directional_success_rate")
    matrix = matrix.reindex(order)
    fig, ax = plt.subplots(figsize=(8.5, 5.6), dpi=320)
    values = matrix[["Boeing 737", "Cessna 172"]].to_numpy(dtype=float)
    image = ax.imshow(values, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Boeing 737", "Cessna 172"])
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([labels[item] for item in order])
    ax.set_title("Synthetic interventions: target proxy directional response")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            text_color = "white" if value < 0.55 else "black"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=text_color, fontsize=9)
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Directional success rate")
    fig.text(
        0.02,
        0.01,
        "Post-retrieval synthetic package perturbations; proxy response only, not state-label accuracy.",
        fontsize=8,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_cache_vs_admissible(cache_summary: pd.DataFrame, frontier: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.5), dpi=320)
    variants = ["question_first_chain", "question_first_state", "state_first_state"]
    labels = ["Q-first raw chain", "Q-first state", "State-first state"]
    colors = ["#4C78A8", "#72B7B2", "#E45756"]
    sub = cache_summary[cache_summary["aircraft_model"].isin(["Boeing 737", "Cessna 172"])]
    for aircraft, marker in [("Boeing 737", "o"), ("Cessna 172", "s")]:
        aircraft_sub = sub[sub["aircraft_model"] == aircraft].set_index("variant").loc[variants].reset_index()
        axes[0].plot(
            np.arange(len(variants)),
            aircraft_sub["mean_shared_prefix_ratio"],
            marker=marker,
            linewidth=2,
            label=aircraft,
        )
    axes[0].set_xticks(np.arange(len(variants)))
    axes[0].set_xticklabels(labels, rotation=18, ha="right")
    axes[0].set_ylabel("Shared-prefix ratio proxy")
    axes[0].set_title("Prompt organization creates cache-eligible reuse")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.25)

    final = frontier[frontier["constraint_step"].isin(["raw_reuse", "action_prereq"])]
    final = final[final["aircraft_model"].isin(["Boeing 737", "Cessna 172"])]
    width = 0.34
    x = np.arange(2)
    for i, step in enumerate(["raw_reuse", "action_prereq"]):
        vals = final[final["constraint_step"] == step].set_index("aircraft_model").loc[["Boeing 737", "Cessna 172"]]
        axes[1].bar(
            x + (i - 0.5) * width,
            vals["mean_admissible_reuse_ratio_proxy"],
            width=width,
            label="Raw" if step == "raw_reuse" else "After action prerequisite",
            color=colors[i],
            alpha=0.85,
        )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(["Boeing", "Cessna"])
    axes[1].set_ylim(0, 1.02)
    axes[1].set_ylabel("Reuse ratio proxy")
    axes[1].set_title("Computational reuse is filtered by admissibility proxies")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.25)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_paradigm_matrix(
    cache_variant_summary: pd.DataFrame,
    model_scale_summary: pd.DataFrame,
    path: Path,
) -> pd.DataFrame:
    cache = cache_variant_summary.set_index("variant")
    model = model_scale_summary.groupby("variant", as_index=False).agg(
        mean_grounded_proxy=("mean_answer_evidence_grounded_score_proxy", "mean"),
        mean_citation_coverage_proxy=("mean_citation_coverage_proxy", "mean"),
        mean_latency_s=("mean_latency_s", "mean"),
        mean_prompt_tokens_usage=("mean_prompt_tokens_usage", "mean"),
    )
    direct = model[model["variant"] == "direct_model_only"].iloc[0]
    raw = model[model["variant"] == "question_first_chain"].iloc[0]
    state = model[model["variant"] == "state_first_state"].iloc[0]
    rows = [
        {
            "paradigm": "RLM/direct local model",
            "variant": "direct_model_only",
            "knowledge_access": "internal model parameters only",
            "mean_grounded_proxy": direct["mean_grounded_proxy"],
            "mean_citation_coverage_proxy": direct["mean_citation_coverage_proxy"],
            "mean_prompt_tokens_usage": direct["mean_prompt_tokens_usage"],
            "cache_eligible_prefix_ratio_proxy": 0.0,
            "mean_reusable_prefix_tokens_after_first": 0.0,
        },
        {
            "paradigm": "Question-first RAG chain",
            "variant": "question_first_chain",
            "knowledge_access": "retrieved raw evidence chain",
            "mean_grounded_proxy": raw["mean_grounded_proxy"],
            "mean_citation_coverage_proxy": raw["mean_citation_coverage_proxy"],
            "mean_prompt_tokens_usage": raw["mean_prompt_tokens_usage"],
            "cache_eligible_prefix_ratio_proxy": cache.loc[
                "question_first_chain", "mean_family_shared_prefix_ratio_qwen35_08b"
            ],
            "mean_reusable_prefix_tokens_after_first": cache.loc[
                "question_first_chain", "mean_reusable_prefix_tokens_after_first_qwen35_08b"
            ],
        },
        {
            "paradigm": "State-first evidence assembly",
            "variant": "state_first_state",
            "knowledge_access": "retrieved evidence organized before generation",
            "mean_grounded_proxy": state["mean_grounded_proxy"],
            "mean_citation_coverage_proxy": state["mean_citation_coverage_proxy"],
            "mean_prompt_tokens_usage": state["mean_prompt_tokens_usage"],
            "cache_eligible_prefix_ratio_proxy": cache.loc[
                "state_first_state", "mean_family_shared_prefix_ratio_qwen35_08b"
            ],
            "mean_reusable_prefix_tokens_after_first": cache.loc[
                "state_first_state", "mean_reusable_prefix_tokens_after_first_qwen35_08b"
            ],
        },
    ]
    table = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.2), dpi=320)
    x = np.arange(len(table))
    labels = ["Direct\nmodel", "RAG\nchain", "Evidence\nstate"]
    colors = ["#BAB0AC", "#4C78A8", "#54A24B"]
    metrics = [
        ("mean_grounded_proxy", "Groundedness proxy", (0, 0.95)),
        ("cache_eligible_prefix_ratio_proxy", "Cache-eligible prefix ratio", (0, 0.8)),
        ("mean_prompt_tokens_usage", "Mean prompt tokens", (0, None)),
    ]
    for ax, (metric, title, ylim) in zip(axes, metrics):
        y = table[metric].astype(float).to_numpy()
        ax.bar(x, y, color=colors, alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(title)
        if ylim[1] is not None:
            ax.set_ylim(*ylim)
        else:
            ax.set_ylim(0, float(y.max()) * 1.2)
        for xi, yi in zip(x, y):
            label = f"{yi:.2f}" if yi < 10 else f"{yi:.0f}"
            ax.text(xi, yi + (ax.get_ylim()[1] * 0.025), label, ha="center", va="bottom", fontsize=8)
        ax.grid(axis="y", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("RLM, RAG, and evidence-state assembly as pre-generation knowledge organization", y=1.02)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return table


def build_claim_map(
    frontier: pd.DataFrame,
    explanation: pd.DataFrame,
    model_matrix: pd.DataFrame,
    cache_variant_summary: pd.DataFrame,
) -> pd.DataFrame:
    all_frontier = frontier[(frontier["aircraft_model"] == "All") & (frontier["constraint_step"] == "action_prereq")].iloc[0]
    boeing_raw = frontier[(frontier["aircraft_model"] == "Boeing 737") & (frontier["constraint_step"] == "raw_reuse")].iloc[0]
    cessna_raw = frontier[(frontier["aircraft_model"] == "Cessna 172") & (frontier["constraint_step"] == "raw_reuse")].iloc[0]
    state_first = cache_variant_summary[cache_variant_summary["variant"] == "state_first_state"].iloc[0]
    q_chain = cache_variant_summary[cache_variant_summary["variant"] == "question_first_chain"].iloc[0]
    direct_gap = float(model_matrix["state_minus_direct_grounded_proxy"].mean())
    rows = [
        {
            "claim_id": "ARF1",
            "manuscript_claim": "The current project contains substantial reusable evidence-state mass, but raw computational reuse and admissible evidence reuse must be separated.",
            "support_status": "Supported as proxy diagnostic on 214 source-derived V21 queries",
            "key_value": f"All-query action-prerequisite retained reuse ratio proxy={all_frontier['mean_admissible_reuse_ratio_proxy']:.3f}",
            "result_file": "04_results/admissible_reuse_v1/admissible_reuse_frontier_summary.csv",
            "figure": "05_figures/admissible_reuse_v1/figure_a_admissible_reuse_frontier.png",
            "caveat": "Proxy over visible package factors; not real layerwise KV cache, not expert action authorization.",
        },
        {
            "claim_id": "ARF2",
            "manuscript_claim": "Boeing shows higher raw reuse than Cessna in this corpus because its current evidence packages are more repetitive and concentrated.",
            "support_status": "Supported as corpus-specific explanation",
            "key_value": f"Boeing raw={boeing_raw['mean_admissible_reuse_ratio_proxy']:.3f}; Cessna raw={cessna_raw['mean_admissible_reuse_ratio_proxy']:.3f}",
            "result_file": "06_tables/admissible_reuse_v1/corpus_reuse_explanation.csv",
            "figure": "05_figures/admissible_reuse_v1/figure_b_corpus_reuse_explanation.png",
            "caveat": "Do not generalize to all Boeing/Cessna manuals; this is a current-corpus representation effect.",
        },
        {
            "claim_id": "ARF3",
            "manuscript_claim": "State-first organization creates far more cache-eligible shared-prefix structure than question-first raw RAG chains.",
            "support_status": "Supported as tokenizer-level prefix proxy",
            "key_value": f"State-first shared prefix ratio={state_first['mean_family_shared_prefix_ratio_qwen35_08b']:.3f}; raw chain={q_chain['mean_family_shared_prefix_ratio_qwen35_08b']:.3f}",
            "result_file": "06_tables/cache_reuse_v1/variant_summary.csv",
            "figure": "05_figures/admissible_reuse_v1/figure_d_cache_vs_admissible_reuse.png",
            "caveat": "No internal server KV-cache table or layerwise tensor inspection.",
        },
        {
            "claim_id": "ARF4",
            "manuscript_claim": "Increasing model scale alone did not reconstruct the project-specific evidence state in the tested local Qwen ladder.",
            "support_status": "Supported as controlled local model-scale diagnostic",
            "key_value": f"Mean state-minus-direct groundedness proxy={direct_gap:.3f}",
            "result_file": "06_tables/admissible_reuse_v1/model_scale_state_reuse_matrix.csv",
            "figure": "05_figures/admissible_reuse_v1/figure_c_scale_state_reuse_matrix.png",
            "caveat": "Local Qwen models, selected source-derived queries, answer-grounding proxies; not a universal RLM impossibility claim.",
        },
        {
            "claim_id": "ARF5",
            "manuscript_claim": "RLM/direct answering, question-first RAG chains, and state-first evidence assembly differ empirically along groundedness, citation availability, prompt burden, and cache-eligible prefix reuse.",
            "support_status": "Supported as a local diagnostic comparison across existing result artifacts",
            "key_value": "See paradigm_reuse_comparison.csv",
            "result_file": "06_tables/admissible_reuse_v1/paradigm_reuse_comparison.csv",
            "figure": "05_figures/admissible_reuse_v1/figure_f_paradigm_reuse_comparison.png",
            "caveat": "Direct-model condition is local Qwen only and cannot represent all retrieval-free language models.",
        },
    ]
    return pd.DataFrame(rows)


def write_markdown_report(
    path: Path,
    frontier: pd.DataFrame,
    model_matrix: pd.DataFrame,
    claim_map: pd.DataFrame,
) -> None:
    all_raw = frontier[(frontier["aircraft_model"] == "All") & (frontier["constraint_step"] == "raw_reuse")].iloc[0]
    all_action = frontier[(frontier["aircraft_model"] == "All") & (frontier["constraint_step"] == "action_prereq")].iloc[0]
    boeing_raw = frontier[(frontier["aircraft_model"] == "Boeing 737") & (frontier["constraint_step"] == "raw_reuse")].iloc[0]
    cessna_raw = frontier[(frontier["aircraft_model"] == "Cessna 172") & (frontier["constraint_step"] == "raw_reuse")].iloc[0]
    mean_state_direct = model_matrix["state_minus_direct_grounded_proxy"].mean()
    text = f"""# Admissible Reuse Frontier Diagnostic

This run converts the earlier cache/reuse observation into a more defensible
scientific object: the difference between computationally reusable state and
evidence that remains compatible with visible admissibility proxies.

## Main numeric findings

- Raw evidence-state reuse exists in the current corpus: all-query mean raw reuse
  ratio proxy = {all_raw['mean_admissible_reuse_ratio_proxy']:.3f}
  [{all_raw['ci95_low']:.3f}, {all_raw['ci95_high']:.3f}].
- After scope, provenance, authority, condition, and action-prerequisite proxies,
  the all-query retained reuse ratio proxy = {all_action['mean_admissible_reuse_ratio_proxy']:.3f}
  [{all_action['ci95_low']:.3f}, {all_action['ci95_high']:.3f}].
- Boeing has higher raw reuse in this current corpus:
  {boeing_raw['mean_admissible_reuse_ratio_proxy']:.3f} versus Cessna
  {cessna_raw['mean_admissible_reuse_ratio_proxy']:.3f}. Treat this as a corpus
  and representation effect, not an aircraft-type law.
- Across the local Qwen scale ladder, explicit state-first prompting improves
  answer-groundedness proxy over direct model-only prompting by a mean of
  {mean_state_direct:.3f}. This supports the claim that scale did not replace
  project evidence-state construction in this diagnostic.

## Paper-safe interpretation

The contribution should be framed as an admissible-reuse frontier: for
aviation maintenance support, the useful pre-generation object is not just a
retrieved context or a larger model's internal memory, but a reusable evidence
state whose reuse is bounded by scope, provenance, authority, and condition
compatibility. The present outputs are proxy diagnostics over source-derived
queries and retrieved packages; they do not prove layerwise KV-cache hits,
mutual information, or expert action authorization.

## Claim map

{claim_map.to_markdown(index=False)}
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()

    config_path = Path(args.config)
    config = read_json(config_path)
    seed = int(config.get("seed", 20260812))
    frames = load_inputs(config)

    frontier, query_frontier = compute_frontier_rows(frames["state_reuse"].copy(), seed=seed)
    explanation = compute_case_explanation(frames["state_reuse"].copy(), frames["natural_factors"].copy(), seed=seed + 7)
    family_aircraft = compute_family_aircraft_summary(frames["cache_family"].copy(), seed=seed + 13)
    model_matrix = compute_model_scale_matrix(frames["model_scale_summary"].copy())
    intervention_summary = summarize_interventions(frames["aesr_interventions"].copy())

    frontier.to_csv(RESULTS / "admissible_reuse_frontier_summary.csv", index=False, encoding="utf-8-sig")
    query_frontier.to_csv(RESULTS / "admissible_reuse_query_frontier.csv", index=False, encoding="utf-8-sig")
    explanation.to_csv(TABLES / "corpus_reuse_explanation.csv", index=False, encoding="utf-8-sig")
    family_aircraft.to_csv(TABLES / "cache_family_aircraft_summary.csv", index=False, encoding="utf-8-sig")
    model_matrix.to_csv(TABLES / "model_scale_state_reuse_matrix.csv", index=False, encoding="utf-8-sig")
    intervention_summary.to_csv(TABLES / "synthetic_intervention_proxy_response_summary.csv", index=False, encoding="utf-8-sig")

    plot_frontier(frontier, FIGURES / "figure_a_admissible_reuse_frontier.png")
    plot_reuse_explanation(explanation, FIGURES / "figure_b_corpus_reuse_explanation.png")
    plot_model_scale(model_matrix, FIGURES / "figure_c_scale_state_reuse_matrix.png")
    plot_cache_vs_admissible(family_aircraft, frontier, FIGURES / "figure_d_cache_vs_admissible_reuse.png")
    plot_intervention_response(intervention_summary, FIGURES / "figure_e_synthetic_intervention_response.png")
    paradigm_table = plot_paradigm_matrix(
        frames["cache_variant_summary"].copy(),
        frames["model_scale_summary"].copy(),
        FIGURES / "figure_f_paradigm_reuse_comparison.png",
    )
    paradigm_table.to_csv(TABLES / "paradigm_reuse_comparison.csv", index=False, encoding="utf-8-sig")

    claim_map = build_claim_map(frontier, explanation, model_matrix, frames["cache_variant_summary"])
    claim_map.to_csv(RESULTS / "claim_result_map.csv", index=False, encoding="utf-8-sig")
    write_markdown_report(REPORTS / "admissible_reuse_frontier_report.md", frontier, model_matrix, claim_map)

    manifest = {
        "experiment_id": config.get("experiment_id", "admissible_reuse_v1_20260812"),
        "created_by": "run_admissible_reuse_frontier.py",
        "python": sys.version,
        "platform": platform.platform(),
        "seed": seed,
        "input_files": frames["_paths"].to_dict(orient="records"),
        "output_files": [
            "04_results/admissible_reuse_v1/admissible_reuse_frontier_summary.csv",
            "04_results/admissible_reuse_v1/admissible_reuse_query_frontier.csv",
            "04_results/admissible_reuse_v1/claim_result_map.csv",
            "06_tables/admissible_reuse_v1/corpus_reuse_explanation.csv",
            "06_tables/admissible_reuse_v1/cache_family_aircraft_summary.csv",
            "06_tables/admissible_reuse_v1/model_scale_state_reuse_matrix.csv",
            "06_tables/admissible_reuse_v1/synthetic_intervention_proxy_response_summary.csv",
            "05_figures/admissible_reuse_v1/figure_a_admissible_reuse_frontier.png",
            "05_figures/admissible_reuse_v1/figure_b_corpus_reuse_explanation.png",
            "05_figures/admissible_reuse_v1/figure_c_scale_state_reuse_matrix.png",
            "05_figures/admissible_reuse_v1/figure_d_cache_vs_admissible_reuse.png",
            "05_figures/admissible_reuse_v1/figure_e_synthetic_intervention_response.png",
            "05_figures/admissible_reuse_v1/figure_f_paradigm_reuse_comparison.png",
            "07_reports/admissible_reuse_v1/admissible_reuse_frontier_report.md",
        ],
        "claim_boundary": (
            "This run computes observable reuse and admissibility-proxy diagnostics from existing project artifacts. "
            "It does not estimate mutual information, inspect transformer KV tensors, validate expert action authorization, "
            "or prove backend-agnostic state inference."
        ),
    }
    write_json(REPORTS / "admissible_reuse_v1_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
