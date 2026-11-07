from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from textwrap import fill

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "04_results" / "method_advantage_audit" / "method_advantage_across_scales_audit.csv"
OUT_DIR = ROOT / "figures" / "Benchmark_Advantage"
REPORT_DIR = ROOT / "reports"


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial"],
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.linewidth": 1.0,
        }
    )


def metric_label(metric: str) -> str:
    return {
        "exact_recall_10": "Exact Recall@10",
        "evidence_package_recall_10": "Package Recall@10",
        "hallucination_risk_proxy": "Risk proxy reduction",
    }.get(metric, metric.replace("_", " ").title())


def build_plot() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    apply_style()
    audit = pd.read_csv(AUDIT)
    strong = audit[audit["advantage_status"].eq("strong_advantage")].copy()
    # Use benchmark/table scopes first, then case-local risk scales; avoid saturated or baseline-better rows.
    preferred = strong[
        strong["source"].astype(str).str.contains("global_v21|paper_table|case_sota_by_method|case_sota_by_track", regex=True)
    ].copy()
    if preferred.empty:
        preferred = strong.copy()
    preferred["display_scope"] = preferred["source"].astype(str).str.replace("_", " ", regex=False)
    preferred["display_metric"] = preferred["metric"].map(metric_label)
    preferred["plot_margin"] = preferred["absolute_margin"].astype(float)
    preferred.loc[preferred["metric"].eq("hallucination_risk_proxy"), "plot_margin"] = preferred.loc[
        preferred["metric"].eq("hallucination_risk_proxy"), "relative_margin"
    ].astype(float)
    preferred["label"] = preferred["display_metric"] + "\n" + preferred["display_scope"]
    preferred = preferred.sort_values("plot_margin", ascending=True).tail(14)

    fig, ax = plt.subplots(figsize=(9.2, max(4.8, 0.45 * len(preferred) + 1.0)))
    colors = ["#0F4D92" if m != "hallucination_risk_proxy" else "#2E9D57" for m in preferred["metric"]]
    y = range(len(preferred))
    ax.barh(y, preferred["plot_margin"], color=colors, alpha=0.88, height=0.68)
    ax.set_yticks(list(y))
    ax.set_yticklabels([fill(x, 34) for x in preferred["label"]])
    ax.set_xlabel("Advantage margin (absolute for recall/package; relative reduction for risk)")
    ax.axvline(0, color="#333333", linewidth=0.9)
    ax.grid(axis="x", color="#D7DCE2", linewidth=0.6, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(0.0, 1.02, "(a)", transform=ax.transAxes, ha="left", va="bottom", weight="bold", fontsize=13)
    for yi, row in enumerate(preferred.itertuples()):
        suffix = " rel." if row.metric == "hallucination_risk_proxy" else " abs."
        ax.text(row.plot_margin + 0.004, yi, f"{row.plot_margin:.3f}{suffix}", va="center", ha="left", fontsize=9)
    ax.text(
        0.99,
        0.04,
        "Only strong-advantage, non-saturated comparisons are shown.",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="#5F6B7A",
    )
    out = OUT_DIR / "benchmark_strong_advantage_scales.png"
    fig.savefig(out, dpi=600, bbox_inches="tight", pad_inches=0.06, facecolor="white")
    plt.close(fig)
    with Image.open(out) as img:
        width, height = img.size
    sidecar = {
        "output_png": str(out),
        "source_audit_csv": str(AUDIT),
        "created_time": datetime.now().isoformat(timespec="seconds"),
        "width": width,
        "height": height,
        "filter": "advantage_status == strong_advantage; saturated/tied/baseline-better rows excluded",
        "claim_boundary": "This figure supports strong advantage only for the shown scales; it does not imply far superiority on saturated metrics.",
    }
    out.with_suffix(".json").write_text(json.dumps(sidecar, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = pd.DataFrame(
        [
            {
                "figure_name": out.name,
                "source_artifact": str(AUDIT),
                "output_png": str(out),
                "dpi": 600,
                "width": width,
                "height": height,
                "notes": "Advantage-only benchmark figure; excludes saturated/tied/baseline-better comparisons.",
            }
        ]
    )
    manifest.to_csv(OUT_DIR / "figure_manifest.csv", index=False, encoding="utf-8-sig")
    return out


def main() -> None:
    out = build_plot()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    note = REPORT_DIR / "benchmark_advantage_figure_note.md"
    note.write_text(
        "# Benchmark Advantage Figure Note\n\n"
        f"Generated `{out}` from the cross-scale advantage audit. The figure includes only comparisons marked `strong_advantage`; saturated/tied and baseline-better benchmark rows are excluded from the plot.\n",
        encoding="utf-8",
    )
    print({"output": str(out)})


if __name__ == "__main__":
    main()
