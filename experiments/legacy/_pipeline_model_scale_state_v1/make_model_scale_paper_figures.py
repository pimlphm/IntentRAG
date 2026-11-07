"""Create paper-ready figures for the model-scale and cache-reuse diagnostics.

All plotted values are read from experiment CSVs produced by:
- _pipeline_model_scale_state_v1/run_model_scale_state_diagnostic.py
- _pipeline_cache_reuse_v1/run_cache_reuse_diagnostic.py

The figures intentionally avoid unsupported claims about true internal KV-cache
hits.  Cache reuse is shown as a tokenizer-level cache-eligible prefix proxy.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


ROOT = Path(__file__).resolve().parents[1]
SCALE_TABLE = ROOT / "06_tables" / "model_scale_state_v1" / "scale_state_variant_summary.csv"
DELTA_TABLE = ROOT / "06_tables" / "model_scale_state_v1" / "scale_state_delta_summary.csv"
CACHE_TABLE = ROOT / "06_tables" / "cache_reuse_v1" / "variant_summary.csv"
OUT = ROOT / "05_figures" / "model_scale_state_v1"
OUT.mkdir(parents=True, exist_ok=True)
PAPER_TABLE = ROOT / "06_tables" / "model_scale_state_v1" / "paper_level_interpretive_findings.csv"


LABELS = {
    "direct_model_only": "Model only",
    "question_first_chain": "Raw evidence chain",
    "state_first_state": "Evidence state",
    "question_first_state": "State after question",
}

COLORS = {
    "direct_model_only": "#4C78A8",
    "question_first_chain": "#F58518",
    "state_first_state": "#54A24B",
    "question_first_state": "#B279A2",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def f(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, default)
        return default if value in ("", None) else float(value)
    except (TypeError, ValueError):
        return default


def by_variant(rows: list[dict[str, str]], variant: str) -> list[dict[str, str]]:
    return sorted([row for row in rows if row["variant"] == variant], key=lambda row: f(row, "model_size_b"))


def percent_change(after: float, before: float) -> float:
    return (after - before) / before if before else math.nan


def figure_scale_quality(scale_rows: list[dict[str, str]]) -> None:
    variants = ["direct_model_only", "question_first_chain", "state_first_state"]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0), dpi=320)
    for variant in variants:
        rows = by_variant(scale_rows, variant)
        xs = [f(row, "model_size_b") for row in rows]
        axes[0].plot(
            xs,
            [f(row, "mean_answer_evidence_grounded_score_proxy") for row in rows],
            marker="o",
            linewidth=2.2,
            color=COLORS[variant],
            label=LABELS[variant],
        )
        axes[1].plot(
            xs,
            [f(row, "mean_citation_coverage_proxy") for row in rows],
            marker="o",
            linewidth=2.2,
            color=COLORS[variant],
            label=LABELS[variant],
        )

    axes[0].annotate(
        "scale alone\nstays flat",
        xy=(14, f(by_variant(scale_rows, "direct_model_only")[-1], "mean_answer_evidence_grounded_score_proxy")),
        xytext=(8.3, 0.56),
        arrowprops={"arrowstyle": "->", "lw": 1.0, "color": "#444444"},
        fontsize=9,
    )
    axes[0].annotate(
        "explicit evidence state\nlifts the answer band",
        xy=(3, f(by_variant(scale_rows, "state_first_state")[2], "mean_answer_evidence_grounded_score_proxy")),
        xytext=(4.4, 0.79),
        arrowprops={"arrowstyle": "->", "lw": 1.0, "color": "#444444"},
        fontsize=9,
    )

    axes[0].set_title("A. Grounded answer behavior")
    axes[0].set_ylabel("Mean groundedness proxy")
    axes[1].set_title("B. Evidence citation coverage")
    axes[1].set_ylabel("Mean citation coverage proxy")
    for axis in axes:
        axis.set_xlabel("Model scale (B parameters)")
        axis.set_xticks([0.5, 1.5, 3, 7, 14])
        axis.grid(axis="y", color="#D8D8D8", linewidth=0.8)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[1].legend(frameon=False, loc="lower right")
    fig.suptitle("Model scale does not replace explicit aviation evidence-state construction", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "paper_fig_scale_quality.png", bbox_inches="tight")
    plt.close(fig)


def figure_gain_latency(scale_rows: list[dict[str, str]], delta_rows: list[dict[str, str]]) -> None:
    delta_rows = sorted(delta_rows, key=lambda row: f(row, "model_size_b"))
    variants = ["direct_model_only", "question_first_chain", "state_first_state"]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0), dpi=320)

    x = [f(row, "model_size_b") for row in delta_rows]
    axes[0].plot(
        x,
        [f(row, "raw_minus_direct_groundedness") for row in delta_rows],
        marker="o",
        linewidth=2.0,
        color=COLORS["question_first_chain"],
        label="Raw chain - model only",
    )
    axes[0].plot(
        x,
        [f(row, "state_minus_direct_groundedness") for row in delta_rows],
        marker="o",
        linewidth=2.0,
        color=COLORS["state_first_state"],
        label="Evidence state - model only",
    )
    axes[0].axhline(0, color="#333333", linewidth=1)
    axes[0].set_title("A. Evidence access gain over model-only")
    axes[0].set_xlabel("Model scale (B parameters)")
    axes[0].set_ylabel("Delta groundedness proxy")
    axes[0].legend(frameon=False, loc="lower right")

    for variant in variants:
        rows = by_variant(scale_rows, variant)
        axes[1].plot(
            [f(row, "model_size_b") for row in rows],
            [f(row, "mean_latency_s") for row in rows],
            marker="o",
            linewidth=2.0,
            color=COLORS[variant],
            label=LABELS[variant],
        )
    axes[1].set_title("B. Local generation latency")
    axes[1].set_xlabel("Model scale (B parameters)")
    axes[1].set_ylabel("Mean latency (s)")
    axes[1].legend(frameon=False, loc="upper left")

    for axis in axes:
        axis.set_xticks([0.5, 1.5, 3, 7, 14])
        axis.grid(axis="y", color="#D8D8D8", linewidth=0.8)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.suptitle("Evidence state is an interface gain, not just a larger-model effect", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "paper_fig_gain_latency.png", bbox_inches="tight")
    plt.close(fig)


def figure_cache_reuse(cache_rows: list[dict[str, str]]) -> None:
    order = ["question_first_chain", "question_first_state", "state_first_state"]
    rows = {row["variant"]: row for row in cache_rows}
    xs = list(range(len(order)))
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.75), dpi=320)

    axes[0].bar(xs, [f(rows[v], "mean_family_shared_prefix_ratio_qwen35_08b") for v in order], color=[COLORS.get(v, "#999999") for v in order])
    axes[0].set_title("A. Cache-eligible shared prefix")
    axes[0].set_ylabel("Shared prefix / prompt tokens")
    axes[0].yaxis.set_major_formatter(PercentFormatter(1.0))

    width = 0.35
    axes[1].bar([i - width / 2 for i in xs], [f(rows[v], "mean_prompt_tokens_qwen35_08b") for v in order], width=width, color="#9ecae1", label="Prompt tokens")
    axes[1].bar([i + width / 2 for i in xs], [f(rows[v], "mean_reusable_prefix_tokens_after_first_qwen35_08b") for v in order], width=width, color="#74c476", label="Reusable prefix after first")
    axes[1].set_title("B. Token budget vs reusable state")
    axes[1].set_ylabel("Qwen tokenizer tokens")
    axes[1].legend(frameon=False, fontsize=8)

    axes[2].plot(xs, [f(rows[v], "mean_answer_evidence_grounded_score_proxy") for v in order], marker="o", color="#2b8cbe", linewidth=2.2, label="Groundedness")
    axes[2].plot(xs, [f(rows[v], "mean_latency_s") / 5.0 for v in order], marker="s", color="#e6550d", linewidth=2.0, label="Latency / 5")
    axes[2].set_title("C. Quality retained while latency drops")
    axes[2].set_ylabel("Proxy score / scaled seconds")
    axes[2].legend(frameon=False, fontsize=8)

    for axis in axes:
        axis.set_xticks(xs)
        axis.set_xticklabels(["Raw chain", "Question-first\nstate", "State-first\nstate"], rotation=0)
        axis.grid(axis="y", color="#D8D8D8", linewidth=0.8)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.suptitle("State-first organization converts evidence into a reusable prefix proxy for KV-cache-friendly generation", y=1.04)
    fig.tight_layout()
    fig.savefig(OUT / "paper_fig_cache_reuse_proxy.png", bbox_inches="tight")
    plt.close(fig)


def write_interpretive_table(scale_rows: list[dict[str, str]], delta_rows: list[dict[str, str]], cache_rows: list[dict[str, str]]) -> None:
    direct = by_variant(scale_rows, "direct_model_only")
    state = by_variant(scale_rows, "state_first_state")
    raw = by_variant(scale_rows, "question_first_chain")
    direct_min = min(f(row, "mean_answer_evidence_grounded_score_proxy") for row in direct)
    direct_max = max(f(row, "mean_answer_evidence_grounded_score_proxy") for row in direct)
    state_max = max(f(row, "mean_answer_evidence_grounded_score_proxy") for row in state)
    state_min = min(f(row, "mean_answer_evidence_grounded_score_proxy") for row in state)
    raw_mean_tokens = sum(f(row, "mean_prompt_tokens_usage") for row in raw) / len(raw)
    state_mean_tokens = sum(f(row, "mean_prompt_tokens_usage") for row in state) / len(state)
    raw_14 = [row for row in raw if row["model_scale"] == "14B"][0]
    state_14 = [row for row in state if row["model_scale"] == "14B"][0]
    cache = {row["variant"]: row for row in cache_rows}
    prefix_gain = f(cache["state_first_state"], "mean_family_shared_prefix_ratio_qwen35_08b") / f(cache["question_first_chain"], "mean_family_shared_prefix_ratio_qwen35_08b")
    reusable_gain = f(cache["state_first_state"], "mean_reusable_prefix_tokens_after_first_qwen35_08b") / f(cache["question_first_chain"], "mean_reusable_prefix_tokens_after_first_qwen35_08b")
    token_drop = -percent_change(state_mean_tokens, raw_mean_tokens)
    latency_drop_14 = -percent_change(f(state_14, "mean_latency_s"), f(raw_14, "mean_latency_s"))
    rows = [
        {
            "finding_id": "F1",
            "finding": "Model scale alone does not reconstruct the project evidence state.",
            "metric": f"Direct-only groundedness remains {direct_min:.3f}-{direct_max:.3f} from 0.5B to 14B; citation coverage stays 0.",
            "paper_implication": "Do not present large LLM/RLM scale as a substitute for external evidence-state construction.",
        },
        {
            "finding_id": "F2",
            "finding": "Explicit evidence state turns scale into usable evidence consumption rather than intrinsic knowledge.",
            "metric": f"Evidence-state groundedness spans {state_min:.3f}-{state_max:.3f}; state-minus-direct gains reach {max(f(row, 'state_minus_direct_groundedness') for row in delta_rows):.3f}.",
            "paper_implication": "Scale amplifies downstream verbalization over organized evidence; it does not supply aircraft-specific admissibility by itself.",
        },
        {
            "finding_id": "F3",
            "finding": "Raw evidence chain and evidence state converge in quality for larger local models, but state is shorter and more reusable.",
            "metric": f"State uses {token_drop:.1%} fewer prompt tokens on the scale subset; 14B state is {latency_drop_14:.1%} faster than raw chain.",
            "paper_implication": "The contribution is not just retrieval access, but low-entropy evidence organization.",
        },
        {
            "finding_id": "F4",
            "finding": "State-first prompting exposes a KV-cache-friendly reusable prefix proxy.",
            "metric": f"Shared prefix ratio increases {prefix_gain:.1f}x and reusable prefix tokens after first increase {reusable_gain:.1f}x versus raw chain.",
            "paper_implication": "This supports a cache/reuse view connecting RAG, RLM, and evidence-state assembly without claiming true internal KV-hit inspection.",
        },
    ]
    write_csv(PAPER_TABLE, rows)


def main() -> None:
    scale_rows = read_csv(SCALE_TABLE)
    delta_rows = read_csv(DELTA_TABLE)
    cache_rows = read_csv(CACHE_TABLE)
    figure_scale_quality(scale_rows)
    figure_gain_latency(scale_rows, delta_rows)
    figure_cache_reuse(cache_rows)
    write_interpretive_table(scale_rows, delta_rows, cache_rows)
    print(f"wrote figures to {OUT}")
    print(f"wrote {PAPER_TABLE}")


if __name__ == "__main__":
    main()
