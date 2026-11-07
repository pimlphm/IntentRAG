"""Admissibility-signed cache diagnostics.

This run turns "computational reuse != admissible evidence reuse" into a
constructive proposal: bind cache-eligible evidence-state prefixes to an
admissibility signature.  The signature is a visible proxy over scope,
provenance, authority, condition, action prerequisite, source diversity, role
diversity, and top evidence IDs.

The outputs are tokenizer/prefix and grouping proxies.  They do not inspect
server-side KV tensors, GPU cache tables, or expert action authorization.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "signed_cache_v1" / "signed_cache_v1.json"
RESULTS = ROOT / "04_results" / "signed_cache_v1"
FIGURES = ROOT / "05_figures" / "signed_cache_v1"
TABLES = ROOT / "06_tables" / "signed_cache_v1"
REPORTS = ROOT / "07_reports" / "signed_cache_v1"
LOGS = ROOT / "08_logs" / "signed_cache_v1"

for directory in (RESULTS, FIGURES, TABLES, REPORTS, LOGS):
    directory.mkdir(parents=True, exist_ok=True)


POLICY_ORDER = [
    "plain_prefix",
    "state_prefix",
    "aircraft_signed",
    "scope_authority_signed",
    "full_admissibility_signed",
    "exact_evidence_signed",
]

POLICY_LABELS = {
    "plain_prefix": "Plain prompt prefix",
    "state_prefix": "State prefix",
    "aircraft_signed": "Aircraft-signed",
    "scope_authority_signed": "Scope+authority signed",
    "full_admissibility_signed": "Full admissibility signed",
    "exact_evidence_signed": "Exact-evidence signed",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    columns = fieldnames or (list(rows[0].keys()) if rows else [])
    if not columns:
        raise ValueError(f"Cannot write empty CSV without field names: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
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
    return mean, float(low), float(high), len(array)


def parse_ids(value: Any) -> list[str]:
    return [part.strip() for part in str(value or "").split(";") if part.strip()]


def bucket_bool(value: Any, threshold: float = 0.999) -> str:
    try:
        return "1" if float(value) >= threshold else "0"
    except (TypeError, ValueError):
        return "0"


def bucket_count(value: Any, bins: tuple[int, ...]) -> str:
    try:
        number = int(float(value))
    except (TypeError, ValueError):
        return "na"
    for bound in bins:
        if number <= bound:
            return f"le{bound}"
    return f"gt{bins[-1]}"


def digest_signature(parts: list[str], n: int = 12) -> str:
    text = "|".join(parts)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:n]


def load_inputs(config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    paths = {
        "state_reuse": ROOT / config["inputs"]["state_reuse_query"],
        "natural": ROOT / config["inputs"]["natural_factor_proxies"],
        "cache_family": ROOT / config["inputs"]["cache_family_prefix_metrics"],
        "prompt_manifest": ROOT / config["inputs"]["prompt_family_manifest"],
        "model_scale": ROOT / config["inputs"]["model_scale_summary"],
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required input files:\n" + "\n".join(missing))
    frames = {key: pd.read_csv(path) for key, path in paths.items()}
    frames["_paths"] = pd.DataFrame(
        [{"name": key, "path": str(path.relative_to(ROOT)), "sha256": sha256(path)} for key, path in paths.items()]
    )
    return frames


def build_query_table(state: pd.DataFrame, natural: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "question_id",
        "track",
        "query_type",
        "top_evidence_ids",
        "package_size",
        "role_diversity",
        "source_diversity",
        "page_diversity",
        "action_like_query_proxy",
    ]
    table = state.merge(natural[cols], on="question_id", how="left", suffixes=("", "_natural"))
    numeric = [
        "raw_reuse_ratio_qwen",
        "raw_reused_evidence_tokens_qwen",
        "total_prompt_tokens_qwen",
        "scope_match_rate",
        "provenance_valid_rate",
        "authority_cue_present",
        "condition_present",
        "action_authority_evidence_availability_proxy",
        "reused_evidence_id_count",
        "role_diversity",
        "source_diversity",
        "page_diversity",
        "redundancy_proxy",
    ]
    for col in numeric:
        if col in table.columns:
            table[col] = pd.to_numeric(table[col], errors="coerce").fillna(0.0)
    table["top4_evidence_signature"] = table["top_evidence_ids"].map(lambda v: ",".join(parse_ids(v)[:4]))
    table["top10_evidence_signature"] = table["top_evidence_ids"].map(lambda v: ",".join(parse_ids(v)[:10]))
    table["action_prereq_bucket"] = table["action_authority_evidence_availability_proxy"].map(bucket_bool)
    table["scope_bucket"] = table["scope_match_rate"].map(bucket_bool)
    table["provenance_bucket"] = table["provenance_valid_rate"].map(bucket_bool)
    table["authority_bucket"] = table["authority_cue_present"].map(bucket_bool)
    table["condition_bucket"] = table["condition_present"].map(bucket_bool)
    table["source_diversity_bucket"] = table["source_diversity"].map(lambda v: bucket_count(v, (1, 2, 4)))
    table["role_diversity_bucket"] = table["role_diversity"].map(lambda v: bucket_count(v, (3, 5, 7)))
    return table


def policy_parts(row: pd.Series, policy: str) -> list[str]:
    if policy == "plain_prefix":
        return ["plain", "global"]
    if policy == "state_prefix":
        return ["state", str(row["aircraft_model"]), str(row.get("query_type", ""))]
    if policy == "aircraft_signed":
        return ["aircraft", str(row["aircraft_model"])]
    if policy == "scope_authority_signed":
        return [
            "scope_authority",
            str(row["aircraft_model"]),
            f"scope={row['scope_bucket']}",
            f"prov={row['provenance_bucket']}",
            f"auth={row['authority_bucket']}",
        ]
    if policy == "full_admissibility_signed":
        return [
            "full_adm",
            str(row["aircraft_model"]),
            f"scope={row['scope_bucket']}",
            f"prov={row['provenance_bucket']}",
            f"auth={row['authority_bucket']}",
            f"cond={row['condition_bucket']}",
            f"act={row['action_prereq_bucket']}",
            f"src={row['source_diversity_bucket']}",
            f"role={row['role_diversity_bucket']}",
        ]
    if policy == "exact_evidence_signed":
        return [
            "exact_evidence",
            str(row["aircraft_model"]),
            str(row["top4_evidence_signature"]),
            f"act={row['action_prereq_bucket']}",
        ]
    raise ValueError(policy)


def compute_signature_table(query_table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in query_table.iterrows():
        for policy in POLICY_ORDER:
            parts = policy_parts(row, policy)
            rows.append(
                {
                    "question_id": row["question_id"],
                    "aircraft_model": row["aircraft_model"],
                    "query_type": row.get("query_type", ""),
                    "policy": policy,
                    "policy_label": POLICY_LABELS[policy],
                    "signature": digest_signature(parts),
                    "signature_text": "|".join(parts),
                    "raw_reuse_ratio_qwen": row["raw_reuse_ratio_qwen"],
                    "raw_reused_evidence_tokens_qwen": row["raw_reused_evidence_tokens_qwen"],
                    "action_prereq_proxy": row["action_authority_evidence_availability_proxy"],
                    "scope_match_rate": row["scope_match_rate"],
                    "provenance_valid_rate": row["provenance_valid_rate"],
                    "authority_cue_present": row["authority_cue_present"],
                    "condition_present": row["condition_present"],
                    "top4_evidence_signature": row["top4_evidence_signature"],
                    "top10_evidence_signature": row["top10_evidence_signature"],
                }
            )
    sig = pd.DataFrame(rows)
    group_sizes = sig.groupby(["policy", "signature"], dropna=False).size().rename("signature_group_size").reset_index()
    sig = sig.merge(group_sizes, on=["policy", "signature"], how="left")
    sig["cache_hit_opportunity_proxy"] = (sig["signature_group_size"] - 1).clip(lower=0)
    sig["cache_hit_binary_proxy"] = (sig["signature_group_size"] > 1).astype(float)
    sig["signed_reusable_tokens_proxy"] = sig["raw_reused_evidence_tokens_qwen"] * sig["cache_hit_binary_proxy"]
    return sig


def compatibility_for_policy(group: pd.DataFrame, policy: str) -> tuple[float, float, float]:
    if len(group) <= 1:
        return 1.0, 1.0, 1.0
    action_values = set(group["action_prereq_proxy"].round(6).astype(str))
    top4_values = set(group["top4_evidence_signature"].astype(str))
    admissibility_values = set(
        (
            group["scope_match_rate"].round(6).astype(str)
            + "|"
            + group["provenance_valid_rate"].round(6).astype(str)
            + "|"
            + group["authority_cue_present"].round(6).astype(str)
            + "|"
            + group["condition_present"].round(6).astype(str)
            + "|"
            + group["action_prereq_proxy"].round(6).astype(str)
        )
    )
    action_purity = 1.0 if len(action_values) == 1 else 0.0
    evidence_purity = 1.0 if len(top4_values) == 1 else 0.0
    admissibility_purity = 1.0 if len(admissibility_values) == 1 else 0.0
    return action_purity, evidence_purity, admissibility_purity


def summarize_policies(sig: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_rows = []
    for (policy, signature), group in sig.groupby(["policy", "signature"], dropna=False):
        action_purity, evidence_purity, admissibility_purity = compatibility_for_policy(group, policy)
        group_rows.append(
            {
                "policy": policy,
                "policy_label": POLICY_LABELS[policy],
                "signature": signature,
                "signature_text": group["signature_text"].iloc[0],
                "n_queries": len(group),
                "aircraft_models": ";".join(sorted(group["aircraft_model"].astype(str).unique())),
                "mean_raw_reuse_ratio": float(group["raw_reuse_ratio_qwen"].mean()),
                "mean_signed_reusable_tokens": float(group["signed_reusable_tokens_proxy"].mean()),
                "action_prereq_purity": action_purity,
                "top4_evidence_purity": evidence_purity,
                "admissibility_vector_purity": admissibility_purity,
            }
        )
    groups = pd.DataFrame(group_rows)

    rows = []
    for policy in POLICY_ORDER:
        sub = sig[sig["policy"] == policy]
        gsub = groups[groups["policy"] == policy]
        hit_mean, hit_low, hit_high, n = bootstrap_mean_ci(sub["cache_hit_binary_proxy"], seed=seed + len(policy))
        token_mean, token_low, token_high, _ = bootstrap_mean_ci(sub["signed_reusable_tokens_proxy"], seed=seed + len(policy) + 17)
        raw_token_mean, _, _, _ = bootstrap_mean_ci(sub["raw_reused_evidence_tokens_qwen"], seed=seed + len(policy) + 29)
        rows.append(
            {
                "policy": policy,
                "policy_label": POLICY_LABELS[policy],
                "n_queries": n,
                "n_signatures": int(gsub["signature"].nunique()),
                "mean_signature_group_size": float(gsub["n_queries"].mean()),
                "cache_hit_rate_proxy": hit_mean,
                "cache_hit_rate_ci95_low": hit_low,
                "cache_hit_rate_ci95_high": hit_high,
                "mean_signed_reusable_tokens_proxy": token_mean,
                "tokens_ci95_low": token_low,
                "tokens_ci95_high": token_high,
                "mean_raw_reusable_tokens_proxy": raw_token_mean,
                "reuse_retention_vs_raw_tokens": token_mean / raw_token_mean if raw_token_mean else math.nan,
                "weighted_action_prereq_purity": float(np.average(gsub["action_prereq_purity"], weights=gsub["n_queries"])),
                "weighted_top4_evidence_purity": float(np.average(gsub["top4_evidence_purity"], weights=gsub["n_queries"])),
                "weighted_admissibility_vector_purity": float(
                    np.average(gsub["admissibility_vector_purity"], weights=gsub["n_queries"])
                ),
            }
        )
    return pd.DataFrame(rows), groups


def summarize_aircraft(sig: pd.DataFrame, seed: int) -> pd.DataFrame:
    rows = []
    for policy in POLICY_ORDER:
        for aircraft, sub in sig[sig["policy"] == policy].groupby("aircraft_model", dropna=False):
            hit_mean, hit_low, hit_high, n = bootstrap_mean_ci(sub["cache_hit_binary_proxy"], seed=seed + len(policy))
            token_mean, token_low, token_high, _ = bootstrap_mean_ci(sub["signed_reusable_tokens_proxy"], seed=seed + len(policy) + 7)
            rows.append(
                {
                    "policy": policy,
                    "policy_label": POLICY_LABELS[policy],
                    "aircraft_model": aircraft,
                    "n_queries": n,
                    "cache_hit_rate_proxy": hit_mean,
                    "cache_hit_rate_ci95_low": hit_low,
                    "cache_hit_rate_ci95_high": hit_high,
                    "mean_signed_reusable_tokens_proxy": token_mean,
                    "tokens_ci95_low": token_low,
                    "tokens_ci95_high": token_high,
                }
            )
    return pd.DataFrame(rows)


def plot_policy_tradeoff(summary: pd.DataFrame, path: Path) -> None:
    sub = summary.set_index("policy").loc[POLICY_ORDER].reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.5), dpi=320)
    x = np.arange(len(sub))
    labels = ["Plain", "State", "Aircraft", "Scope+\nauthority", "Full\nadmiss.", "Exact\nevidence"]

    axes[0].bar(x, sub["cache_hit_rate_proxy"], color="#4C78A8", alpha=0.9)
    axes[0].set_title("Cache-hit opportunity")
    axes[0].set_ylabel("Hit-rate proxy")
    axes[0].set_ylim(0, 1.05)

    axes[1].bar(x, sub["weighted_admissibility_vector_purity"], color="#54A24B", alpha=0.9)
    axes[1].set_title("Admissibility purity")
    axes[1].set_ylabel("Purity of shared groups")
    axes[1].set_ylim(0, 1.05)

    axes[2].scatter(
        sub["cache_hit_rate_proxy"],
        sub["weighted_admissibility_vector_purity"],
        s=90,
        color="#E45756",
    )
    axes[2].text(0.04, 0.94, "Strict but costly", fontsize=8, ha="left", va="top")
    axes[2].text(0.96, 0.94, "High reuse + high purity", fontsize=8, ha="right", va="top")
    axes[2].text(0.96, 0.08, "High reuse, unsafe sharing", fontsize=8, ha="right", va="bottom")
    axes[2].set_title("KV reuse vs admissible sharing")
    axes[2].set_xlabel("Hit-rate proxy")
    axes[2].set_ylabel("Admissibility purity")
    axes[2].set_xlim(-0.03, 1.03)
    axes[2].set_ylim(-0.03, 1.03)

    for ax in axes[:2]:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=18, ha="right")
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("Admissibility-signed KV prefix: aligning computational and evidence reuse", y=1.02)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_aircraft_policy(aircraft_summary: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.6), dpi=320)
    labels = ["Plain", "State", "Aircraft", "Scope+\nauth.", "Full\nadmiss.", "Exact\nevid."]
    colors = {"Boeing 737": "#4C78A8", "Cessna 172": "#F58518"}
    for aircraft, marker in [("Boeing 737", "o"), ("Cessna 172", "s")]:
        sub = aircraft_summary[aircraft_summary["aircraft_model"] == aircraft].set_index("policy").loc[POLICY_ORDER]
        x = np.arange(len(POLICY_ORDER))
        axes[0].plot(x, sub["cache_hit_rate_proxy"], marker=marker, linewidth=2, label=aircraft, color=colors[aircraft])
        axes[1].plot(
            x,
            sub["mean_signed_reusable_tokens_proxy"],
            marker=marker,
            linewidth=2,
            label=aircraft,
            color=colors[aircraft],
        )
    axes[0].set_ylabel("Hit-rate proxy")
    axes[0].set_title("Signed-cache opportunity by aircraft")
    axes[0].set_ylim(-0.02, 1.02)
    axes[1].set_ylabel("Reusable tokens proxy")
    axes[1].set_title("Reusable token mass after signature binding")
    for ax in axes:
        ax.set_xticks(np.arange(len(POLICY_ORDER)))
        ax.set_xticklabels(labels, rotation=18, ha="right")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_method_diagram(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.6, 3.9), dpi=320)
    ax.axis("off")
    boxes = [
        (0.03, 0.54, 0.17, 0.26, "Retrieved\nevidence cards"),
        (0.25, 0.54, 0.18, 0.26, "Admissibility\nsignature"),
        (0.48, 0.54, 0.18, 0.26, "Canonical\nstate prefix"),
        (0.71, 0.54, 0.22, 0.26, "KV prefix cache\nwithin signature"),
        (0.48, 0.12, 0.18, 0.24, "Query slot\nand verbalizer"),
    ]
    for x, y, w, h, text in boxes:
        rect = plt.Rectangle((x, y), w, h, fill=False, linewidth=1.8, edgecolor="#333333")
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10)
    arrows = [
        ((0.20, 0.67), (0.25, 0.67)),
        ((0.43, 0.67), (0.48, 0.67)),
        ((0.66, 0.67), (0.71, 0.67)),
        ((0.57, 0.54), (0.57, 0.36)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", linewidth=1.5, color="#333333"))
    ax.text(
        0.25,
        0.38,
        "signature fields: aircraft, scope, provenance,\nauthority, condition, action prerequisite, revision proxy",
        fontsize=7.7,
        ha="left",
        va="top",
    )
    ax.text(
        0.73,
        0.38,
        "reuse only when\nsignature matches",
        fontsize=8.2,
        ha="left",
        va="top",
    )
    ax.set_title("Admissibility-signed KV prefix organization", fontsize=14)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def build_claim_map(policy_summary: pd.DataFrame) -> pd.DataFrame:
    state = policy_summary[policy_summary["policy"] == "state_prefix"].iloc[0]
    full = policy_summary[policy_summary["policy"] == "full_admissibility_signed"].iloc[0]
    exact = policy_summary[policy_summary["policy"] == "exact_evidence_signed"].iloc[0]
    rows = [
        {
            "claim_id": "SC1",
            "claim": "Admissibility-signed prefixes make cache reuse conditional on evidence-state compatibility.",
            "support_status": "Supported as a constructive tokenizer/grouping diagnostic",
            "key_value": f"Full signature purity={full['weighted_admissibility_vector_purity']:.3f}",
            "result_file": "06_tables/signed_cache_v1/signed_cache_policy_summary.csv",
            "figure": "05_figures/signed_cache_v1/figure_a_signed_cache_tradeoff.png",
            "caveat": "Proxy grouping; no server-side KV cache table inspection.",
        },
        {
            "claim_id": "SC2",
            "claim": "Increasing admissibility binding trades cache-hit opportunity for admissibility purity.",
            "support_status": "Supported as observed trade-off across cache-key policies",
            "key_value": f"State hit={state['cache_hit_rate_proxy']:.3f}, purity={state['weighted_admissibility_vector_purity']:.3f}; full hit={full['cache_hit_rate_proxy']:.3f}, purity={full['weighted_admissibility_vector_purity']:.3f}",
            "result_file": "06_tables/signed_cache_v1/signed_cache_policy_summary.csv",
            "figure": "05_figures/signed_cache_v1/figure_a_signed_cache_tradeoff.png",
            "caveat": "The design goal is controlled reuse, not maximal hit rate.",
        },
        {
            "claim_id": "SC3",
            "claim": "Exact-evidence signed prefixes approximate computational reuse equals admissible evidence reuse at the cost of lower reuse.",
            "support_status": "Supported as a strict-boundary reference policy",
            "key_value": f"Exact-evidence hit={exact['cache_hit_rate_proxy']:.3f}, purity={exact['weighted_admissibility_vector_purity']:.3f}",
            "result_file": "06_tables/signed_cache_v1/signed_cache_policy_summary.csv",
            "figure": "05_figures/signed_cache_v1/figure_a_signed_cache_tradeoff.png",
            "caveat": "Approximate because top-4 evidence and proxy admissibility are used, not expert equivalence labels.",
        },
    ]
    return pd.DataFrame(rows)


def write_report(path: Path, policy_summary: pd.DataFrame, claim_map: pd.DataFrame) -> None:
    state = policy_summary[policy_summary["policy"] == "state_prefix"].iloc[0]
    full = policy_summary[policy_summary["policy"] == "full_admissibility_signed"].iloc[0]
    exact = policy_summary[policy_summary["policy"] == "exact_evidence_signed"].iloc[0]
    text = f"""# Admissibility-Signed KV Prefix Diagnostic

## Constructive answer

Computational reuse and admissible evidence reuse can be made closer by changing
the unit of reuse.  Instead of caching an arbitrary prompt prefix, cache a
canonical evidence-state prefix only under an admissibility signature.  The
signature encodes visible scope, provenance, authority, condition, action
prerequisite, source-diversity, role-diversity, and evidence identity proxies.

## Main results

- State-prefix reuse has high cache opportunity but imperfect admissibility
  purity: hit-rate proxy = {state['cache_hit_rate_proxy']:.3f}, admissibility
  vector purity = {state['weighted_admissibility_vector_purity']:.3f}.
- Full admissibility signing raises admissibility purity to
  {full['weighted_admissibility_vector_purity']:.3f}, with hit-rate proxy
  {full['cache_hit_rate_proxy']:.3f}.
- Exact-evidence signing is the strict reference: hit-rate proxy =
  {exact['cache_hit_rate_proxy']:.3f}, purity =
  {exact['weighted_admissibility_vector_purity']:.3f}.  It is closest to the
  condition "computational reuse ~= admissible evidence reuse", but costs more
  reuse opportunity.

## Paper claim

The contribution on KV cache should not be phrased as measuring real GPU KV
cache internals.  It should be phrased as a cache-key and prefix-organization
principle: admissibility-signed KV reuse.  The empirical result is the observed
reuse-purity frontier on project data.

## Claim map

{claim_map.to_markdown(index=False)}
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()

    config = read_json(Path(args.config))
    seed = int(config.get("seed", 20260812))
    frames = load_inputs(config)
    query_table = build_query_table(frames["state_reuse"].copy(), frames["natural"].copy())
    sig = compute_signature_table(query_table)
    policy_summary, signature_groups = summarize_policies(sig, seed=seed)
    aircraft_summary = summarize_aircraft(sig, seed=seed + 19)

    query_table.to_csv(RESULTS / "signed_cache_query_inputs.csv", index=False, encoding="utf-8-sig")
    sig.to_csv(RESULTS / "signed_cache_query_signatures.csv", index=False, encoding="utf-8-sig")
    signature_groups.to_csv(RESULTS / "signed_cache_signature_groups.csv", index=False, encoding="utf-8-sig")
    policy_summary.to_csv(TABLES / "signed_cache_policy_summary.csv", index=False, encoding="utf-8-sig")
    aircraft_summary.to_csv(TABLES / "signed_cache_aircraft_summary.csv", index=False, encoding="utf-8-sig")

    plot_policy_tradeoff(policy_summary, FIGURES / "figure_a_signed_cache_tradeoff.png")
    plot_aircraft_policy(aircraft_summary, FIGURES / "figure_b_signed_cache_by_aircraft.png")
    plot_method_diagram(FIGURES / "figure_c_admissibility_signed_kv_prefix.png")

    claim_map = build_claim_map(policy_summary)
    claim_map.to_csv(RESULTS / "claim_result_map.csv", index=False, encoding="utf-8-sig")
    write_report(REPORTS / "signed_cache_v1_report.md", policy_summary, claim_map)

    manifest = {
        "experiment_id": config.get("experiment_id", "signed_cache_v1_20260812"),
        "created_by": "run_signed_cache_diagnostic.py",
        "python": sys.version,
        "platform": platform.platform(),
        "seed": seed,
        "input_files": frames["_paths"].to_dict(orient="records"),
        "output_files": [
            "04_results/signed_cache_v1/signed_cache_query_inputs.csv",
            "04_results/signed_cache_v1/signed_cache_query_signatures.csv",
            "04_results/signed_cache_v1/signed_cache_signature_groups.csv",
            "04_results/signed_cache_v1/claim_result_map.csv",
            "06_tables/signed_cache_v1/signed_cache_policy_summary.csv",
            "06_tables/signed_cache_v1/signed_cache_aircraft_summary.csv",
            "05_figures/signed_cache_v1/figure_a_signed_cache_tradeoff.png",
            "05_figures/signed_cache_v1/figure_b_signed_cache_by_aircraft.png",
            "05_figures/signed_cache_v1/figure_c_admissibility_signed_kv_prefix.png",
            "07_reports/signed_cache_v1/signed_cache_v1_report.md",
        ],
        "claim_boundary": (
            "Admissibility-signed KV prefix is evaluated as a tokenizer/grouping proxy. "
            "This run does not inspect internal transformer KV tensors, production cache hit tables, "
            "or expert action-authorization labels."
        ),
    }
    write_json(REPORTS / "signed_cache_v1_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
