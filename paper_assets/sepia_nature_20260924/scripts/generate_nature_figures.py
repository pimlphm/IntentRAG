from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from PIL import Image


SEED = 20260819

# One semantic, color-vision-safe system is shared by every manuscript figure.
# The hues are based on the Okabe-Ito family; muted neutrals carry structure.
BLUE = "#0072B2"
ORANGE = "#E69F00"
TEAL = "#009E73"
RED = "#D55E00"
PURPLE = "#CC79A7"
GOLD = "#B58900"
GRAY = "#69727D"
LIGHT_GRAY = "#D9DEE5"
PALE = "#F5F7F9"
INK = "#202428"
NATURE_STYLE_VERSION = "2026-09-24"


PANEL_CATALOG = [
    {
        "panel_id": "fig18_01",
        "stem": "fig18_01_role_mass_transport_volcano",
        "word_order": "Manuscript Fig. 14",
        "name_zh": "证据角色质量迁移火山图",
        "name_en": "Observable role-mass transport volcano plot",
        "meaning_zh": "比较检索后角色质量 p_R 与可见查询需求代理 p_q 的平均差，并以角色族内 BH-FDR 显示偏移的稳定性。warning 与 authority 向检索包富集，而 action 与 condition 相对需求不足。",
        "boundary_zh": "这是可观测组成质量的描述性迁移，不是互信息、因果效应或真实知识分布。主分析排除了 16 条 visible-query fallback 记录。",
        "source_csv": "fig18_role_transport_summary.csv",
    },
    {
        "panel_id": "fig18_02",
        "stem": "fig18_02_fixed_package_loo_sensitivity_volcano",
        "word_order": "Manuscript Fig. 15",
        "name_zh": "固定证据包留一对齐敏感度火山图",
        "name_en": "Fixed-package leave-one-out alignment-sensitivity volcano plot",
        "meaning_zh": "度量从既有证据包中移除一个角色条目时，可见需求对齐代理的变化；action 与 condition 为正，而 warning 与 evidence 为负，说明被大量选择不等于对齐效用为正。",
        "boundary_zh": "LOO 不重新运行检索，不是 Shapley 值或因果条目贡献；显著性在该角色族内进行 BH-FDR。",
        "source_csv": "fig18_role_loo_summary.csv",
    },
    {
        "panel_id": "fig18_03",
        "stem": "fig18_03_role_transport_vs_loo_sensitivity",
        "word_order": "Manuscript Fig. 16",
        "name_zh": "角色迁移—对齐敏感度联合机制图",
        "name_en": "Joint role-transport and alignment-sensitivity mechanism plane",
        "meaning_zh": "把同一证据角色的检索质量迁移和固定包 LOO 敏感度放在二维平面中。warning 位于高迁移但负敏感度象限，action/condition 位于负迁移但正敏感度象限，直接展示选择偏置与下游效用的解耦。",
        "boundary_zh": "点大小只编码已选择条目数；四象限关系是关联性机制诊断，不构成因果方向证明。",
        "source_csv": "fig18_role_transport_summary.csv; fig18_role_loo_summary.csv",
    },
    {
        "panel_id": "fig18_04",
        "stem": "fig18_04_exact_text_cluster_loo_raincloud",
        "word_order": "Manuscript Fig. 17",
        "name_zh": "精确查询文本簇 LOO 敏感度雨云图",
        "name_en": "Exact-query-text-cluster LOO-sensitivity raincloud plot",
        "meaning_zh": "展示六类关键证据角色在归一化精确查询文本簇层面的 LOO 敏感度分布，而不只显示总体均值；长尾说明同一角色在不同查询中可能产生异质贡献。",
        "boundary_zh": "分布单位是 aircraft×normalized exact visible-query text×role 的簇均值，不是 930 条原始角色记录的独立样本。",
        "source_csv": "fig18_exact_text_cluster_role_loo_nonfallback.csv",
    },
    {
        "panel_id": "fig19_01",
        "stem": "fig19_01_documentary_ambiguity_raincloud",
        "word_order": "Manuscript Fig. 18",
        "name_zh": "多粒度文档身份歧义雨云图",
        "name_en": "Multi-granularity documentary-identity ambiguity raincloud plot",
        "meaning_zh": "比较 Page、Section 与 Component-family 分组中包含的 detail identity 数量。粒度越粗，单个分组折叠的细粒度身份越多，尤其 Cessna component-family 层级出现极高歧义。",
        "boundary_zh": "这是确定性的文档身份粗化指标，不是语义信息熵、token 压缩率或互信息损失。",
        "source_csv": "fig19_group_size_distributions.csv",
    },
    {
        "panel_id": "fig19_02",
        "stem": "fig19_02_view_overlap_and_conditional_rescue_matrix",
        "word_order": "Manuscript Fig. 19",
        "name_zh": "多视图重合度—条件救援率矩阵",
        "name_en": "Cross-view overlap and conditional-rescue matrix",
        "meaning_zh": "上三角显示不同粒度 Top-10 证据集合的平均 Jaccard，下三角显示当列视图失败时行视图的条件救援率；Profile 对 Detail miss 的救援率仅 0.09，而 Detail 对 Profile miss 的救援率为 0.71。",
        "boundary_zh": "救援率依赖 locked post-hoc accepted-hit 代理及方向化条件分母，矩阵上下三角不可作为对称距离解释。",
        "source_csv": "fig19_overlap_rescue_matrix.csv",
    },
    {
        "panel_id": "fig19_03",
        "stem": "fig19_03_single_view_hit_intersections",
        "word_order": "Manuscript Fig. 20",
        "name_zh": "单视图命中交集 UpSet 图",
        "name_en": "Single-view accepted-hit intersection UpSet plot",
        "meaning_zh": "把 214 条 locked 记录按 Detail、Page、Section、Profile 的命中组合分解，并按 Boeing/Cessna 堆叠。最大交集是 Detail+Page+Section 命中而 Profile 未命中，说明持久层不能替代局部多粒度证据。",
        "boundary_zh": "交集编码的是规则派生 accepted-hit@10，不是维修正确率或专家授权结论。",
        "source_csv": "fig19_hit_pattern_summary.csv",
    },
    {
        "panel_id": "fig19_04",
        "stem": "fig19_04_persistent_minus_query_local_forest",
        "word_order": "Manuscript Fig. 21",
        "name_zh": "持久层相对查询局部层命中差森林图",
        "name_en": "Persistent-minus-query-local hit-difference forest plot",
        "meaning_zh": "比较 persistent-fixed 与 query-local-fixed 的 hit@10 差异及按精确查询文本簇分层 bootstrap 区间。总体 accepted 差为 −0.289、exact 差为 −0.377，只有少数任务族显示条件互补。",
        "boundary_zh": "区间按 aircraft 分层、normalized exact visible-query-text cluster 重采样；小样本任务族区间较宽，不能解释为普适优越性。",
        "source_csv": "fig19_query_family_paired_effects.csv",
    },
    {
        "panel_id": "fig20_01",
        "stem": "fig20_01_router_numerator_outcome_plane",
        "word_order": "Manuscript Fig. 22",
        "name_zh": "长短期路由分子—命中结果联合平面",
        "name_en": "Query-local versus persistent router-numerator outcome plane",
        "meaning_zh": "把短期局部层与长期持久层的路由分子放在二维平面，并按两种视图的命中结果着色；等权重射线说明路由权重由相对证据集中度决定，而不是由模型规模直接给出。",
        "boundary_zh": "路由分子来自同一相似度构造下的归一化 concentration/entropy 代理；不是跨模型可比的校准概率。",
        "source_csv": "fig20_router_outcome_records.csv",
    },
    {
        "panel_id": "fig20_02",
        "stem": "fig20_02_persistent_weight_quintile_outcomes",
        "word_order": "Manuscript Fig. 23",
        "name_zh": "持久层权重五分位命中组成图",
        "name_en": "Outcome composition across persistent-weight quintiles",
        "meaning_zh": "展示 persistent weight 从 Q1 到 Q5 时 Both、Query-local only、Persistent only 与 Neither 的组成变化。全部 7 条 persistent-only 记录落入 Q5，但 Q5 同时有 48.8% Neither，说明高长期权重只能触发分支，不能替代局部证据。",
        "boundary_zh": "五分位和结果来自同一 214 条 locked 样本，是样本内部描述，不是外部校准或决策阈值验证。",
        "source_csv": "fig20_weight_quintile_outcomes.csv",
    },
    {
        "panel_id": "fig20_03",
        "stem": "fig20_03_persistent_rescue_precision_recall",
        "word_order": "Manuscript Fig. 24",
        "name_zh": "持久层独占救援识别精确率—召回率图",
        "name_en": "Precision-recall analysis for persistent-only rescue",
        "meaning_zh": "评估 long-term weight 对 persistent-only 案例的样本内排序识别能力。All 的 AP 为 0.137，Boeing 为 0.379；相对阳性率基线虽有提升，但绝对精确率仍低。",
        "boundary_zh": "阳性仅 7 条记录、4 个唯一文本且全部来自 Boeing；AP/AUC 区间为样本内簇 bootstrap，Cessna 救援判别不可估。",
        "source_csv": "fig20_persistent_rescue_pr_curve.csv; fig20_persistent_rescue_pr_summary.csv",
    },
    {
        "panel_id": "fig20_04",
        "stem": "fig20_04_persistent_rescue_branch_budget",
        "word_order": "Manuscript Fig. 25",
        "name_zh": "持久层救援分支预算—覆盖率权衡图",
        "name_en": "Branch-budget trade-off for persistent-only rescue",
        "meaning_zh": "沿 long-term weight 排序逐步扩大分支预算，比较 persistent-only 救援覆盖率与候选精确率。前 41/214 个候选可以覆盖全部 7 条 persistent-only 记录，但对应精确率仅 17.1%，显示召回收益与审核负担的权衡。",
        "boundary_zh": "曲线是同一 locked 样本上的排序后累积诊断；它不包含真实运行时成本、延迟、人工审核代价或外部阈值验证。",
        "source_csv": "fig20_branch_budget_curve.csv",
    },
    {
        "panel_id": "fig20_05",
        "stem": "fig20_05_entropy_discrimination_forest",
        "word_order": "Manuscript Fig. 26",
        "name_zh": "多粒度负熵命中判别森林图",
        "name_en": "Within-view negative-entropy discrimination forest plot",
        "meaning_zh": "比较 Detail、Page、Section、Profile 的 −H_v 对各自 accepted-hit 的样本内 ROC-AUC，并同时列出 AP 与阳性率。随着粒度变粗，AUC 从 0.918 下降到 0.688。",
        "boundary_zh": "每个视图预测的是自身命中，标签与分数均来自同一 locked 样本；结果是关联性诊断，不是外部预测性能。",
        "source_csv": "fig20_entropy_auc_ap_summary.csv",
    },
    {
        "panel_id": "fig27",
        "stem": "fig27_equivalence_class_pairability_plane",
        "word_order": "Manuscript Fig. 27",
        "name_zh": "等价类复用机会—观测特征一致性图",
        "name_en": "Equivalence-class reuse opportunity versus observed-feature agreement",
        "meaning_zh": "比较三种证据状态分组策略。实心圆表示进入非单例组的查询比例，空心三角表示在每组首请求必然未命中、无驱逐的理想序列下可达到的上限，点面积表示可配对 token 质量保留代理。观测特征签名以极小的可配对机会损失把组内观测特征一致性从 0.565 提升到 1.000；绑定 aircraft 与 Top-4 evidence 后机会显著收缩。",
        "boundary_zh": "这是基于分组等价类的离线可配对诊断。它没有真实请求顺序、序列化前缀哈希、层内 KV 状态、GPU 命中、TTFT、FLOPs 或显存测量；观测特征一致性也不等于证据身份一致性。",
        "source_csv": "signed_cache_policy_summary.csv",
    },
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 17,
            "axes.labelsize": 19,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 14,
            "axes.linewidth": 1.15,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    sns.set_theme(style="ticks", rc={"axes.facecolor": "white", "figure.facecolor": "white"})


def style_axis(ax: mpl.axes.Axes, grid_axis: str | None = None) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.15)
    ax.spines["bottom"].set_linewidth(1.15)
    ax.tick_params(axis="both", labelsize=16, width=1.0, length=4.0)
    if grid_axis:
        ax.grid(axis=grid_axis, color=LIGHT_GRAY, linewidth=0.75, alpha=0.78, zorder=0)
    # Arrowheads extend the existing spines without adding floating arrows that
    # can collide with tick labels or annotations after Word scaling.
    ax.plot(1.0, 0.0, marker=">", markersize=6.5, color=INK,
            transform=ax.transAxes, clip_on=False, zorder=8)
    ax.plot(0.0, 1.0, marker="^", markersize=6.5, color=INK,
            transform=ax.transAxes, clip_on=False, zorder=8)


def save_panel(fig: mpl.figure.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"Creator": "generate_nature_figures.py", "Title": ""}
    fig.savefig(stem.with_suffix(".png"), dpi=600, facecolor="white", metadata=metadata)
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white", metadata=metadata)
    fig.savefig(stem.with_suffix(".svg"), facecolor="white", metadata=metadata)
    plt.close(fig)


def require_matplotlib_panel_alignment(
    fig: mpl.figure.Figure,
    upper: mpl.axes.Axes,
    lower: mpl.axes.Axes,
    tolerance_pt: float = 1.5,
) -> None:
    """Fail closed if the two linked axes of the UpSet chart drift in x.

    This is the chart-specific equivalent of the Nature workflow's
    ``audit_panel_alignment.py`` render-time gate.  The two axes are not
    separate manuscript panels; they are the count and membership regions of
    one UpSet figure and must share the same physical left/right boundaries.
    """
    fig.canvas.draw()
    width_pt = float(fig.get_size_inches()[0]) * 72.0
    upper_box = upper.get_position(original=False)
    lower_box = lower.get_position(original=False)
    left_delta = abs(upper_box.x0 - lower_box.x0) * width_pt
    right_delta = abs(upper_box.x1 - lower_box.x1) * width_pt
    if max(left_delta, right_delta) > tolerance_pt:
        raise RuntimeError(
            "UpSet axes alignment failed: "
            f"left={left_delta:.3f} pt, right={right_delta:.3f} pt"
        )


def volcano_color(effect: float, significant: bool, practical: bool) -> str:
    if not significant or not practical:
        return GRAY
    return TEAL if effect > 0 else RED


def volcano_legend() -> list[Line2D]:
    return [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=TEAL, markeredgecolor="white", markersize=9, label="Positive"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=RED, markeredgecolor="white", markersize=9, label="Negative"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=GRAY, markeredgecolor="white", markersize=9, label="q≥0.05 or |effect|<δ"),
    ]


def panel_18_01(transport: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    fig.subplots_adjust(left=0.14, right=0.985, bottom=0.18, top=0.80)
    offsets = {
        "action": (8, -18),
        "condition": (8, 7),
        "authority": (8, 6),
        "warning": (-8, 7),
        "threshold": (8, 6),
        "evidence": (8, 7),
    }
    for row in transport.itertuples():
        color = volcano_color(float(row.mean_effect), bool(row.significant), bool(row.practical))
        y = min(float(row.volcano_y), 11.6)
        marker = "^" if float(row.volcano_y) > 11.6 else "o"
        ax.scatter(row.mean_effect, y, s=108, color=color, edgecolor="white", linewidth=0.9, marker=marker, zorder=3)
        role = str(row.role)
        if role in offsets:
            ha = "right" if role == "warning" else "left"
            ax.annotate(role.replace("_", " "), (row.mean_effect, y), xytext=offsets[role], textcoords="offset points", fontsize=14.5, color=INK, ha=ha)
    ax.axhline(-math.log10(0.05), color=GRAY, linestyle="--", linewidth=1.0)
    ax.axvline(0, color=GRAY, linewidth=1.0)
    ax.axvline(-0.02, color=LIGHT_GRAY, linestyle=":", linewidth=1.0)
    ax.axvline(0.02, color=LIGHT_GRAY, linestyle=":", linewidth=1.0)
    ax.set_xlim(-0.155, 0.225)
    ax.set_ylim(0, 12.25)
    # Omit the extreme -0.15 label so it cannot collide with the y=0 label at
    # the lower-left axis corner after journal-scale reduction.
    ax.set_xticks([-0.10, -0.05, 0.00, 0.05, 0.10, 0.15, 0.20])
    ax.set_xlabel("Observable role-mass transport,\nmean $(p_R-p_q)$", fontsize=19)
    ax.set_ylabel(r"$-\log_{10}$(BH-FDR q)", fontsize=20)
    style_axis(ax, None)
    fig.legend(handles=volcano_legend(), loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.51, 0.98), fontsize=13.2, columnspacing=1.2, handletextpad=0.4)
    save_panel(fig, out / "fig18_01_role_mass_transport_volcano")


def panel_18_02(utility: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    fig.subplots_adjust(left=0.14, right=0.985, bottom=0.18, top=0.80)
    offsets = {
        "action": (8, 7),
        "condition": (8, -18),
        "authority": (8, 7),
    }
    fixed_labels = {
        "warning": (0.006, 11.75),
        "evidence": (0.006, 10.10),
        "threshold": (0.006, 5.10),
        "general": (0.006, 3.20),
    }
    for row in utility.itertuples():
        testable = pd.notna(row.bh_q)
        color = volcano_color(float(row.mean_effect), bool(row.significant), bool(row.practical)) if testable else LIGHT_GRAY
        y = min(float(row.volcano_y), 11.6) if testable else 0.15
        marker = "^" if testable and float(row.volcano_y) > 11.6 else ("o" if testable else "X")
        ax.scatter(row.mean_effect, y, s=108, color=color, edgecolor="white", linewidth=0.9, marker=marker, zorder=3)
        role = str(row.role)
        if testable and (bool(row.significant) or abs(float(row.mean_effect)) >= 0.015):
            if role in fixed_labels:
                ax.text(*fixed_labels[role], role.replace("_", " "), fontsize=14.5, color=INK, ha="left", va="center")
            elif role in offsets:
                ax.annotate(role.replace("_", " "), (row.mean_effect, y), xytext=offsets[role], textcoords="offset points", fontsize=14.5, color=INK, ha="left")
    ax.axhline(-math.log10(0.05), color=GRAY, linestyle="--", linewidth=1.0)
    ax.axvline(0, color=GRAY, linewidth=1.0)
    ax.axvline(-0.005, color=LIGHT_GRAY, linestyle=":", linewidth=1.0)
    ax.axvline(0.005, color=LIGHT_GRAY, linestyle=":", linewidth=1.0)
    ax.set_xlim(-0.03, 0.19)
    ax.set_ylim(0, 12.25)
    ax.set_xlabel("Fixed-package LOO\nalignment sensitivity", fontsize=19)
    ax.set_ylabel(r"$-\log_{10}$(BH-FDR q)", fontsize=20)
    style_axis(ax, None)
    fig.legend(handles=volcano_legend(), loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.51, 0.98), fontsize=13.2, columnspacing=1.2, handletextpad=0.4)
    save_panel(fig, out / "fig18_02_fixed_package_loo_sensitivity_volcano")


def panel_18_03(transport: pd.DataFrame, utility: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    fig.subplots_adjust(left=0.21, right=0.985, bottom=0.18, top=0.96)
    joint = transport.merge(utility, on="role", suffixes=("_transport", "_utility"))
    offsets = {
        "action": (9, 13),
        "condition": (9, -31),
        "authority": (9, 11),
        "warning": (-66, 30),
        "evidence": (9, -17),
    }
    for row in joint.itertuples():
        x = float(row.mean_effect_transport)
        y = float(row.mean_effect_utility)
        color = TEAL if x >= 0 and y >= 0 else RED if x >= 0 and y < 0 else BLUE if y >= 0 else PURPLE
        size = 70 + 1.7 * math.sqrt(max(float(row.selected_items), 1.0))
        ax.scatter(x, y, s=size, color=color, alpha=0.92, edgecolor="white", linewidth=0.9, zorder=3)
        role = str(row.role).replace("_", " ")
        if role in offsets:
            ax.annotate(role, (x, y), xytext=offsets[role], textcoords="offset points", fontsize=14.0, color=INK)
    ax.axhline(0, color=GRAY, linewidth=1.0)
    ax.axvline(0, color=GRAY, linewidth=1.0)
    ax.set_xlim(-0.155, 0.225)
    # Give the negative-sensitivity quadrants enough vertical room for their
    # explanatory labels; otherwise the zero reference and bottom axis leave
    # no collision-free text band.
    ax.set_ylim(-0.060, 0.184)
    ax.set_xlabel("Role transport, mean $(p_R-p_q)$", fontsize=19)
    ax.set_ylabel("Role-conditioned LOO\nalignment sensitivity", fontsize=19)
    style_axis(ax, None)
    ax.text(0.98, 0.96, "mass+ / sensitivity+", transform=ax.transAxes, ha="right", va="top", fontsize=12.4, color=TEAL)
    # Right-align the lower-right quadrant label against the empty outer edge so
    # that it cannot collide with the nearby ``evidence`` point annotation.
    ax.text(0.98, 0.035, "mass+ / sensitivity−", transform=ax.transAxes, ha="right", va="bottom", fontsize=12.0, color=RED)
    ax.text(0.02, 0.96, "mass− / sensitivity+", transform=ax.transAxes, ha="left", va="top", fontsize=12.4, color=BLUE)
    ax.text(0.02, 0.04, "mass− / sensitivity−", transform=ax.transAxes, ha="left", va="bottom", fontsize=12.4, color=PURPLE)
    save_panel(fig, out / "fig18_03_role_transport_vs_loo_sensitivity")


def panel_18_04(clustered: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    fig.subplots_adjust(left=0.19, right=0.985, bottom=0.18, top=0.84)
    focus_roles = ["action", "condition", "authority", "warning", "threshold", "evidence"]
    focus = clustered[clustered["evidence_role"].isin(focus_roles)].rename(columns={"cluster_mean_effect": "effect"})
    means = focus.groupby("evidence_role")["effect"].mean().to_dict()
    palette = {role: TEAL if means.get(role, 0) >= 0 else RED for role in focus_roles}
    sns.violinplot(data=focus, y="evidence_role", x="effect", hue="evidence_role", order=focus_roles, palette=palette, inner=None, cut=0, density_norm="width", linewidth=0.85, saturation=0.78, legend=False, ax=ax)
    sns.boxplot(data=focus, y="evidence_role", x="effect", order=focus_roles, width=0.17, showfliers=False, color="white", boxprops={"edgecolor": INK, "linewidth": 0.9}, whiskerprops={"color": INK, "linewidth": 0.9}, medianprops={"color": INK, "linewidth": 1.25}, ax=ax)
    sns.stripplot(data=focus.sample(frac=1, random_state=SEED), y="evidence_role", x="effect", order=focus_roles, color=INK, size=2.4, jitter=0.17, alpha=0.17, ax=ax)
    ax.axvline(0, color=GRAY, linewidth=1.0)
    ax.set_xlim(-0.12, 0.37)
    ax.set_xlabel("Normalized exact-text-cluster\nmean LOO sensitivity", fontsize=18)
    ax.set_ylabel("")
    ax.set_yticks(range(len(focus_roles)), [x.replace("_", " ").title() for x in focus_roles])
    style_axis(ax, "x")
    handles = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor=TEAL, markersize=9, label="Positive role mean"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=RED, markersize=9, label="Negative role mean"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.55, 0.98), fontsize=14)
    save_panel(fig, out / "fig18_04_exact_text_cluster_loo_raincloud")


def panel_19_01(group_sizes: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    fig.subplots_adjust(left=0.22, right=0.985, bottom=0.18, top=0.95)
    grouping_order = ["Page", "Section", "Component family"]
    palette = {"Boeing 737": BLUE, "Cessna 172": ORANGE}
    sns.violinplot(data=group_sizes, y="grouping", x="log10_details_per_group", hue="aircraft_model", order=grouping_order, hue_order=["Boeing 737", "Cessna 172"], split=True, inner=None, cut=0, density_norm="width", palette=palette, linewidth=0.85, saturation=0.8, ax=ax)
    sns.boxplot(data=group_sizes, y="grouping", x="log10_details_per_group", hue="aircraft_model", order=grouping_order, hue_order=["Boeing 737", "Cessna 172"], dodge=True, width=0.20, showfliers=False, palette=palette, boxprops={"facecolor": "white", "edgecolor": INK, "linewidth": 0.8}, whiskerprops={"color": INK, "linewidth": 0.8}, medianprops={"color": INK, "linewidth": 1.15}, ax=ax)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], labels[:2], loc="upper right", frameon=False, fontsize=14)
    ax.set_xlim(-0.2, 4.2)
    ax.set_xlabel("Documentary ambiguity, $\\log_{10}$\n(detail identities per group)", fontsize=18)
    ax.set_ylabel("")
    ax.set_yticks(range(len(grouping_order)), ["Page", "Section", "Component\nfamily"])
    style_axis(ax, None)
    save_panel(fig, out / "fig19_01_documentary_ambiguity_raincloud")


def panel_19_02(overlap: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    fig.subplots_adjust(left=0.16, right=0.985, bottom=0.21, top=0.86)
    view_order = ["Detail", "Page", "Section", "Profile"]
    jaccard = overlap.pivot(index="base_view", columns="added_view", values="mean_top10_jaccard").loc[view_order, view_order]
    rescue = overlap.pivot(index="base_view", columns="added_view", values="conditional_rescue_rate").loc[view_order, view_order]
    norm = Normalize(0, 1)
    cmap_overlap = mpl.colormaps["Blues"]
    cmap_rescue = mpl.colormaps["Oranges"]
    for i, row in enumerate(view_order):
        for j, col in enumerate(view_order):
            x0, y0 = j, len(view_order) - 1 - i
            if i == j:
                ax.add_patch(plt.Rectangle((x0, y0), 1, 1, facecolor=PALE, edgecolor="white"))
                diag = overlap[(overlap.base_view == row) & (overlap.added_view == row)].iloc[0]
                hit_rate = 1 - float(diag.base_miss_records) / 214
                ax.text(x0 + 0.5, y0 + 0.54, f"hit\n{hit_rate:.2f}", ha="center", va="center", fontsize=14.2, color=INK)
            elif i < j:
                value = float(jaccard.loc[row, col])
                ax.add_patch(plt.Rectangle((x0, y0), 1, 1, facecolor=cmap_overlap(norm(value)), edgecolor="white"))
                ax.text(x0 + 0.5, y0 + 0.5, f"{value:.2f}", ha="center", va="center", fontsize=14.2, color="white" if value > 0.45 else INK)
            else:
                value = float(rescue.loc[col, row])
                ax.add_patch(plt.Rectangle((x0, y0), 1, 1, facecolor=cmap_rescue(norm(value)), edgecolor="white"))
                denom = int(overlap[(overlap.base_view == col) & (overlap.added_view == row)].iloc[0].base_miss_records)
                ax.text(x0 + 0.5, y0 + 0.5, f"{value:.2f}\n(n={denom})", ha="center", va="center", fontsize=13.0, color="white" if value > 0.55 else INK)
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 4)
    ax.set_xticks(np.arange(4) + 0.5, view_order, rotation=26, ha="right", rotation_mode="anchor")
    ax.set_yticks(np.arange(4) + 0.5, list(reversed(view_order)))
    ax.tick_params(axis="both", length=0, labelsize=15.2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.text(0.18, 0.94, "upper: mean Top-10 Jaccard", ha="left", va="center", fontsize=13.5, color=BLUE)
    fig.text(0.98, 0.94, "lower: row rescues column miss", ha="right", va="center", fontsize=13.5, color=ORANGE)
    save_panel(fig, out / "fig19_02_view_overlap_and_conditional_rescue_matrix")


def panel_19_03(patterns: pd.DataFrame, out: Path) -> None:
    shown = patterns.sort_values(["records", "pattern"], ascending=[False, False]).reset_index(drop=True)
    fig = plt.figure(figsize=(7.0, 5.7))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.85, 1.0], hspace=0.035)
    fig.subplots_adjust(left=0.18, right=0.985, bottom=0.15, top=0.93)
    ax_bar = fig.add_subplot(gs[0])
    ax_mat = fig.add_subplot(gs[1], sharex=ax_bar)
    x = np.arange(len(shown))
    ax_bar.bar(x, shown["boeing"], color=BLUE, width=0.74, label="Boeing 737")
    ax_bar.bar(x, shown["cessna"], bottom=shown["boeing"], color=ORANGE, width=0.74, label="Cessna 172")
    ymax = float(shown.records.max())
    for i, row in shown.iterrows():
        ax_bar.text(i, row.records + ymax * 0.025, str(int(row.records)), ha="center", va="bottom", fontsize=12.7, color=INK)
    ax_bar.set_ylabel("Accepted-hit records", fontsize=18)
    ax_bar.tick_params(axis="x", bottom=False, labelbottom=False)
    ax_bar.legend(loc="upper right", frameon=False, fontsize=13.5)
    style_axis(ax_bar, None)
    view_order = ["Detail", "Page", "Section", "Profile"]
    for i, row in shown.iterrows():
        active: list[int] = []
        for j, view in enumerate(view_order):
            y = len(view_order) - 1 - j
            is_active = int(row[view])
            ax_mat.scatter(i, y, s=64, color=INK if is_active else LIGHT_GRAY, edgecolor="none", zorder=3)
            if is_active:
                active.append(y)
        if len(active) > 1:
            ax_mat.plot([i, i], [min(active), max(active)], color=INK, linewidth=1.45, zorder=2)
    ax_mat.set_yticks(range(4), list(reversed(view_order)))
    ax_mat.set_xticks(x, [f"P{i + 1}" for i in x], rotation=0, rotation_mode="anchor")
    ax_mat.set_xlabel("Observed single-view hit intersections", fontsize=17)
    ax_mat.set_xlim(-0.6, len(shown) - 0.4)
    ax_mat.set_ylim(-0.6, 3.6)
    ax_mat.spines["top"].set_visible(False)
    ax_mat.spines["right"].set_visible(False)
    ax_mat.spines["left"].set_visible(False)
    ax_mat.tick_params(axis="y", length=0, labelsize=14.0)
    ax_mat.tick_params(axis="x", labelsize=12.0)
    require_matplotlib_panel_alignment(fig, ax_bar, ax_mat)
    save_panel(fig, out / "fig19_03_single_view_hit_intersections")


def panel_19_04(family: pd.DataFrame, out: Path) -> None:
    display = family[family["metric"].eq("accepted")].copy()
    order = [
        "All",
        "symptom_to_action",
        "condition_to_action",
        "authority_or_manual_reference",
        "symptom_to_cause",
        "multi_hop_diagnosis",
        "component_to_failure",
        "cross_page_evidence_query",
        "warning_or_limitation",
    ]
    display["label"] = display["scope"]
    exact_all = family[(family["metric"].eq("exact")) & family["scope"].eq("All")].copy()
    exact_all["label"] = "All (exact)"
    display.loc[display["scope"].eq("All"), "label"] = "All (accepted)"
    display = pd.concat([display, exact_all], ignore_index=True)
    order = ["All (accepted)", "All (exact)", *order[1:]]
    display["order"] = display["label"].map({name: i for i, name in enumerate(order)})
    display = display.dropna(subset=["order"]).sort_values("order", ascending=False)
    labels = {
        "All (accepted)": "All (accepted)",
        "All (exact)": "All (exact)",
        "symptom_to_action": "Symptom → action",
        "condition_to_action": "Condition → action",
        "authority_or_manual_reference": "Authority / manual",
        "symptom_to_cause": "Symptom → cause",
        "multi_hop_diagnosis": "Multi-hop",
        "component_to_failure": "Component → failure",
        "cross_page_evidence_query": "Cross-page",
        "warning_or_limitation": "Warning / limitation",
    }
    fig, ax = plt.subplots(figsize=(7.5, 5.6))
    fig.subplots_adjust(left=0.31, right=0.985, bottom=0.16, top=0.87)
    y = np.arange(len(display))
    for i, row in enumerate(display.itertuples()):
        color = TEAL if row.cluster_mean_delta > 0 else RED
        ax.plot([row.cluster_bootstrap_ci_low, row.cluster_bootstrap_ci_high], [i, i], color=color, linewidth=2.3, zorder=2)
        ax.scatter(row.cluster_mean_delta, i, s=92, color=color, edgecolor="white", linewidth=0.9, zorder=3)
        ax.text(0.988, i, f"{row.persistent_only_records}/{row.short_only_records}; {row.n_records}/{row.n_query_text_clusters}", transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=11.2, color=INK, clip_on=True, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.91, "pad": 0.5})
    ax.set_yticks(y, [labels.get(s, s) for s in display["label"]])
    ax.axvline(0, color=GRAY, linewidth=1.0)
    ax.set_xlim(-1.0, 0.82)
    ax.set_xlabel("Persistent-fixed − query-local-fixed hit@10", fontsize=17.5)
    ax.set_ylabel("")
    ax.tick_params(axis="y", labelsize=13.1)
    style_axis(ax, None)
    fig.text(0.985, 0.955, "rescue/loss; records/texts", ha="right", va="top", fontsize=12.0, color=GRAY)
    save_panel(fig, out / "fig19_04_persistent_minus_query_local_forest")


def panel_20_01(outcome: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    # Reserve two explicit legend bands above the data rectangle: outcome
    # classes on top and contour keys below.  This prevents either legend from
    # being crossed by a contour ray or colliding with the other legend.
    fig.subplots_adjust(left=0.15, right=0.985, bottom=0.18, top=0.69)
    colors = {"Both": TEAL, "Query-local only": BLUE, "Persistent only": ORANGE, "Neither": GRAY}
    for label in ["Neither", "Both", "Query-local only", "Persistent only"]:
        group = outcome[outcome["outcome"].eq(label)]
        ax.scatter(group["short_router_numerator"], group["persistent_router_numerator"], s=48 if label != "Persistent only" else 84, color=colors[label], alpha=0.55 if label != "Persistent only" else 0.96, edgecolor="white", linewidth=0.65, label=f"{label} (n={len(group)})", zorder=3)
    xlim = (max(0.0, float(outcome.short_router_numerator.min()) - 0.02), float(outcome.short_router_numerator.max()) + 0.02)
    ylim = (max(0.0, float(outcome.persistent_router_numerator.min()) - 0.015), float(outcome.persistent_router_numerator.max()) + 0.015)
    xs = np.linspace(*xlim, 200)
    ray_styles = {
        0.2: (0, (1.0, 1.4)),
        0.3: (0, (3.0, 1.5)),
        0.4: "--",
        0.5: "-",
    }
    ray_handles: list[Line2D] = []
    for weight in [0.2, 0.3, 0.4, 0.5]:
        ys = (weight / (1 - weight)) * xs
        mask = (ys >= ylim[0]) & (ys <= ylim[1])
        if np.any(mask):
            ax.plot(xs[mask], ys[mask], color=LIGHT_GRAY, linewidth=1.15, linestyle=ray_styles[weight], zorder=1)
            ray_handles.append(
                Line2D([0], [0], color=GRAY, linewidth=1.6, linestyle=ray_styles[weight], label=f"$w_L={weight:.1f}$")
            )
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("Query-local router numerator", fontsize=19)
    ax.set_ylabel("Persistent router numerator", fontsize=19)
    outcome_handles, outcome_labels = ax.get_legend_handles_labels()
    fig.legend(outcome_handles, outcome_labels, loc="upper center", bbox_to_anchor=(0.52, 0.99), frameon=False, fontsize=13.2, ncol=2, columnspacing=1.2, handletextpad=0.45)
    fig.legend(handles=ray_handles, loc="upper center", bbox_to_anchor=(0.52, 0.845), frameon=False, fontsize=11.0, ncol=4, columnspacing=0.9, handlelength=2.2, handletextpad=0.45)
    style_axis(ax, None)
    save_panel(fig, out / "fig20_01_router_numerator_outcome_plane")


def panel_20_02(quintile: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    fig.subplots_adjust(left=0.14, right=0.985, bottom=0.18, top=0.78)
    colors = {"Both": TEAL, "Query-local only": BLUE, "Persistent only": ORANGE, "Neither": GRAY}
    q_order = ["Q1", "Q2", "Q3", "Q4", "Q5"]
    bottom = np.zeros(len(q_order))
    for label in ["Both", "Query-local only", "Persistent only", "Neither"]:
        values = quintile[quintile.outcome.eq(label)].set_index("weight_quintile")["fraction"].reindex(q_order).fillna(0).to_numpy()
        ax.bar(q_order, values, bottom=bottom, color=colors[label], width=0.72, label=label)
        for i, (base, value) in enumerate(zip(bottom, values)):
            if value >= 0.095:
                ax.text(i, base + value / 2, f"{value:.0%}", ha="center", va="center", fontsize=13.2, color="white" if label != "Neither" else INK)
        bottom += values
    ax.set_ylim(0, 1)
    ax.set_xlabel("Persistent-weight quintile", fontsize=19)
    ax.set_ylabel("Outcome composition", fontsize=19)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.28), frameon=False, fontsize=13.2, ncol=2, columnspacing=1.2)
    style_axis(ax, None)
    save_panel(fig, out / "fig20_02_persistent_weight_quintile_outcomes")


def panel_20_03(pr_curve: pd.DataFrame, pr_summary: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    # Keep the terminal 1.0 tick fully inside the standalone raster/vector
    # canvas when Word scales this panel to a half-column object.
    fig.subplots_adjust(left=0.15, right=0.965, bottom=0.18, top=0.94)
    scope_colors = {"All": PURPLE, "Boeing 737": ORANGE}
    for scope in ["All", "Boeing 737"]:
        curve = pr_curve[pr_curve.scope.eq(scope)]
        summary = pr_summary[pr_summary.scope.eq(scope)].iloc[0]
        ax.step(curve.recall, curve.precision, where="post", color=scope_colors[scope], linewidth=2.5, label=f"{scope}: AP {summary.average_precision:.3f} [{summary.ap_ci_low:.3f}, {summary.ap_ci_high:.3f}]")
        ax.axhline(summary.prevalence, color=scope_colors[scope], linestyle=":", linewidth=1.0, alpha=0.8)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Persistent-only recall", fontsize=19)
    ax.set_ylabel("Precision", fontsize=19)
    ax.legend(loc="upper right", frameon=False, fontsize=12.6)
    style_axis(ax, None)
    save_panel(fig, out / "fig20_03_persistent_rescue_precision_recall")


def panel_20_04(branch: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    # Reserve enough right margin for the terminal 1.0 tick label.
    fig.subplots_adjust(left=0.15, right=0.965, bottom=0.18, top=0.95)
    ax.step(branch.branch_fraction, branch.rescue_capture, where="post", color=GOLD, linewidth=2.7, label="Rescue capture")
    ax.step(branch.branch_fraction, branch.precision_at_budget, where="post", color=GRAY, linewidth=2.2, linestyle="--", label="Precision at budget")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Branch budget (fraction of 214 queries)", fontsize=18.5)
    ax.set_ylabel("Capture / precision", fontsize=19)
    ax.legend(loc="center right", frameon=False, fontsize=14)
    style_axis(ax, None)
    target = branch[branch["rescue_capture"].ge(1.0)].head(1)
    if not target.empty:
        row = target.iloc[0]
        ax.scatter(row.branch_fraction, row.rescue_capture, s=105, color=GOLD, edgecolor="white", linewidth=0.9, zorder=4)
        ax.scatter(row.branch_fraction, row.precision_at_budget, s=90, color=GRAY, edgecolor="white", linewidth=0.9, zorder=4)
        ax.annotate(
            f"{int(row['rank'])}/214 candidates\n100% capture; {row['precision_at_budget']:.1%} precision",
            (row.branch_fraction, row.rescue_capture),
            xytext=(18, -38),
            textcoords="offset points",
            fontsize=13.2,
            ha="left",
            va="top",
            color=INK,
            arrowprops={"arrowstyle": "->", "color": GOLD, "linewidth": 1.25},
            bbox={"facecolor": "white", "edgecolor": LIGHT_GRAY, "alpha": 0.94, "pad": 3.0},
        )
    save_panel(fig, out / "fig20_04_persistent_rescue_branch_budget")


def panel_20_05(auc_summary: pd.DataFrame, out: Path) -> None:
    display = auc_summary[auc_summary.aircraft_model.eq("All")].copy()
    display["order"] = display["view"].map({"Detail": 3, "Page": 2, "Section": 1, "Profile": 0})
    display = display.sort_values("order")
    # This fifth Word object spans both columns.  A deliberately wide, shallow
    # canvas preserves large type at the audited 5.35 x 1.55 inch placement.
    fig, ax = plt.subplots(figsize=(10.8, 3.1))
    # Keep the probability-like ROC-AUC axis in its valid domain and reserve a
    # separate axis-free column for AP/prevalence statistics.
    fig.subplots_adjust(left=0.12, right=0.70, bottom=0.27, top=0.96)
    y = np.arange(len(display))
    colors = {"Detail": BLUE, "Page": TEAL, "Section": ORANGE, "Profile": PURPLE}
    for i, row in enumerate(display.itertuples()):
        ax.plot([row.auc_ci_low, row.auc_ci_high], [i, i], color=colors[row.view], linewidth=2.5)
        ax.scatter(row.auc, i, s=100, color=colors[row.view], edgecolor="white", linewidth=0.9, zorder=3)
        ax.text(1.055, i, f"AP={row.average_precision:.3f}; prev={row.prevalence:.3f}", transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=13.0, color=INK, clip_on=False)
    ax.axvline(0.5, color=GRAY, linestyle="--", linewidth=1.0)
    ax.axvline(1.0, color=LIGHT_GRAY, linewidth=0.9)
    ax.set_yticks(y, display["view"])
    ax.set_xlim(0.45, 1.02)
    ax.set_xticks([0.5, 0.75, 1.0])
    ax.set_xlabel("Within-view ROC-AUC of $-H_v$ for accepted hit", fontsize=18)
    ax.set_ylabel("")
    style_axis(ax, "x")
    ax.tick_params(axis="y", labelsize=16)
    save_panel(fig, out / "fig20_05_entropy_discrimination_forest")


def panel_27_signed_equivalence_frontier(summary: pd.DataFrame, out: Path) -> None:
    """Visualise grouping opportunity without calling it a measured cache hit.

    The filled-circle x coordinate is the source table's
    ``cache_hit_rate_proxy`` field, whose implemented meaning is the fraction of
    queries that belong to a non-singleton equivalence class.  The hollow
    triangle is a stricter optimistic sequence ceiling: one compulsory cold
    start per class and no eviction, (N - K) / N.  Neither is a GPU KV hit.
    """
    policies = [
        ("state_prefix", "Aircraft + query type", BLUE),
        ("full_admissibility_signed", "Observed-feature signature", TEAL),
        ("exact_evidence_signed", "Aircraft + Top-4 evidence", PURPLE),
    ]
    # Match the audited Fig. 27 Word object aspect ratio (about 1.6925) so
    # replacing its PNG cannot stretch the new chart.
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    fig.subplots_adjust(left=0.145, right=0.97, bottom=0.19, top=0.95)

    points: dict[str, tuple[float, float, float, float, str, int, float]] = {}
    for policy, label, color in policies:
        row = summary.loc[summary["policy"].eq(policy)].iloc[0]
        n_queries = int(row["n_queries"])
        n_signatures = int(row["n_signatures"])
        pairable_fraction = float(row["cache_hit_rate_proxy"])
        cold_start_ceiling = (n_queries - n_signatures) / n_queries
        agreement = float(row["weighted_admissibility_vector_purity"])
        token_mass = float(row["reuse_retention_vs_raw_tokens"])
        size = 230 + 620 * token_mass

        ax.plot(
            [cold_start_ceiling, pairable_fraction],
            [agreement, agreement],
            color=color,
            linewidth=2.2,
            alpha=0.72,
            zorder=2,
        )
        ax.errorbar(
            pairable_fraction,
            agreement,
            xerr=np.array(
                [
                    [pairable_fraction - float(row["cache_hit_rate_ci95_low"])],
                    [float(row["cache_hit_rate_ci95_high"]) - pairable_fraction],
                ]
            ),
            fmt="none",
            ecolor=color,
            elinewidth=1.5,
            capsize=4,
            alpha=0.72,
            zorder=3,
        )
        ax.scatter(
            pairable_fraction,
            agreement,
            s=size,
            color=color,
            edgecolor="white",
            linewidth=1.25,
            alpha=0.92,
            zorder=4,
        )
        ax.scatter(
            cold_start_ceiling,
            agreement,
            s=150,
            marker="^",
            facecolor="white",
            edgecolor=color,
            linewidth=2.0,
            zorder=5,
        )
        points[policy] = (
            pairable_fraction,
            cold_start_ceiling,
            agreement,
            token_mass,
            color,
            n_signatures,
            float(row["mean_signature_group_size"]),
        )

    # Direct labels are deliberately placed in white space, with short leaders,
    # so they remain readable at the audited Word insertion size.
    x, c, y, t, color, k, mean_group = points["exact_evidence_signed"]
    ax.annotate(
        f"Aircraft + Top-4 evidence\n{c:.3f} ceiling | {x:.3f} pairable\n{t:.3f} token mass\nK={k}; mean group={mean_group:.2f}",
        xy=(x, y), xytext=(0.34, 0.985), textcoords="data",
        ha="left", va="top", fontsize=13.2, color=color,
        arrowprops={"arrowstyle": "-", "color": color, "linewidth": 1.2},
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.90, "pad": 1.5},
    )
    x, c, y, t, color, k, mean_group = points["full_admissibility_signed"]
    ax.annotate(
        f"Observed-feature signature\n{c:.3f} ceiling | {x:.3f} pairable\n{t:.3f} token mass\nK={k}; mean group={mean_group:.2f}",
        xy=(x, y), xytext=(0.61, 0.88), textcoords="data",
        ha="left", va="top", fontsize=13.2, color=color,
        arrowprops={"arrowstyle": "-", "color": color, "linewidth": 1.2},
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.90, "pad": 1.5},
    )
    x, c, y, t, color, k, mean_group = points["state_prefix"]
    ax.annotate(
        f"Aircraft + query type\n{c:.3f} ceiling | {x:.3f} pairable\n{t:.3f} token mass\nK={k}; mean group={mean_group:.2f}",
        xy=(x, y), xytext=(0.56, 0.69), textcoords="data",
        ha="left", va="top", fontsize=13.2, color=color,
        arrowprops={"arrowstyle": "-", "color": color, "linewidth": 1.2},
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.90, "pad": 1.5},
    )

    ax.set_xlim(0.08, 1.07)
    ax.set_ylim(0.49, 1.055)
    ax.set_xlabel("Reusable-query fraction proxy", fontsize=19)
    ax.set_ylabel("Observed-feature agreement\nwithin grouping", fontsize=18.5)
    # The figure is annotation-dense; omitting the grid prevents reference
    # strokes from running behind labels while preserving the tick scale.
    style_axis(ax, None)
    marker_legend = [
        Line2D([0], [0], marker="o", color=GRAY, markerfacecolor=GRAY,
               markeredgecolor="white", linewidth=0, markersize=10,
               label="Non-singleton membership"),
        Line2D([0], [0], marker="^", color=GRAY, markerfacecolor="white",
               markeredgecolor=GRAY, linewidth=0, markersize=9,
               label="One-cold-start ceiling"),
    ]
    ax.legend(
        handles=marker_legend,
        loc="lower left",
        frameon=False,
        fontsize=12.7,
        borderaxespad=0.5,
        handletextpad=0.5,
    )
    ax.text(
        0.12,
        0.75,
        "Circle area scales with\npairable token-mass retention",
        fontsize=12.7,
        color=GRAY,
        ha="left",
        va="center",
        bbox={"facecolor": "white", "edgecolor": LIGHT_GRAY, "alpha": 0.94, "pad": 3.0},
    )
    save_panel(fig, out / "fig27_equivalence_class_pairability_plane")


def load_inputs(source_dir: Path) -> dict[str, pd.DataFrame]:
    required = sorted({name.strip() for item in PANEL_CATALOG for name in item["source_csv"].split(";")})
    frames: dict[str, pd.DataFrame] = {}
    for name in required:
        path = source_dir / name
        if not path.exists():
            raise FileNotFoundError(path)
        frames[name] = pd.read_csv(path)
    return frames


def generate_panels(frames: dict[str, pd.DataFrame], figures_dir: Path) -> None:
    panel_18_01(frames["fig18_role_transport_summary.csv"], figures_dir)
    panel_18_02(frames["fig18_role_loo_summary.csv"], figures_dir)
    panel_18_03(frames["fig18_role_transport_summary.csv"], frames["fig18_role_loo_summary.csv"], figures_dir)
    panel_18_04(frames["fig18_exact_text_cluster_role_loo_nonfallback.csv"], figures_dir)
    panel_19_01(frames["fig19_group_size_distributions.csv"], figures_dir)
    panel_19_02(frames["fig19_overlap_rescue_matrix.csv"], figures_dir)
    panel_19_03(frames["fig19_hit_pattern_summary.csv"], figures_dir)
    panel_19_04(frames["fig19_query_family_paired_effects.csv"], figures_dir)
    panel_20_01(frames["fig20_router_outcome_records.csv"], figures_dir)
    panel_20_02(frames["fig20_weight_quintile_outcomes.csv"], figures_dir)
    panel_20_03(frames["fig20_persistent_rescue_pr_curve.csv"], frames["fig20_persistent_rescue_pr_summary.csv"], figures_dir)
    panel_20_04(frames["fig20_branch_budget_curve.csv"], figures_dir)
    panel_20_05(frames["fig20_entropy_auc_ap_summary.csv"], figures_dir)
    panel_27_signed_equivalence_frontier(frames["signed_cache_policy_summary.csv"], figures_dir)


def write_catalog(reports_dir: Path) -> None:
    catalog = pd.DataFrame(PANEL_CATALOG)
    catalog.to_csv(reports_dir / "panel_catalog_bilingual.csv", index=False, encoding="utf-8-sig")
    lines = [
        "# IntentRAG Fig.14–27 独立英文图目录与逐图解释",
        "",
        "所有图像均为独立 PNG/PDF/SVG；图内不烧录 `(a)–(e)`，也不含标题。英文图作为 Fig.14–27 的独立图片对象插入 Word，图名与解释由 Word 正文提供。",
        "",
    ]
    for item in PANEL_CATALOG:
        lines.extend(
            [
                f"## {item['panel_id']}　{item['name_zh']}",
                "",
                f"- English: {item['name_en']}",
                f"- Word position: {item['word_order']}",
                f"- File stem: `{item['stem']}`",
                f"- 含义：{item['meaning_zh']}",
                f"- 解释边界：{item['boundary_zh']}",
                f"- 数据：`{item['source_csv']}`",
                "",
            ]
        )
    (reports_dir / "panel_catalog_zh.md").write_text("\n".join(lines), encoding="utf-8")


def validate_assets(source_dir: Path, figures_dir: Path, reports_dir: Path) -> dict[str, object]:
    findings: list[dict[str, object]] = []
    banned_tokens = ["(a)", "(b)", "(c)", "(d)", "(e)"]
    for item in PANEL_CATALOG:
        stem = figures_dir / item["stem"]
        png = stem.with_suffix(".png")
        pdf = stem.with_suffix(".pdf")
        svg = stem.with_suffix(".svg")
        with Image.open(png) as im:
            width, height = im.size
            grayscale = np.asarray(im.convert("L").resize((256, 256)), dtype=np.uint8)
            nonwhite_fraction = float(np.mean(grayscale < 250))
        svg_text = svg.read_text(encoding="utf-8")
        banned_found = [token for token in banned_tokens if token in svg_text]
        findings.append(
            {
                "panel_id": item["panel_id"],
                "stem": item["stem"],
                "png_dimensions_px": [width, height],
                "png_nonwhite_fraction": nonwhite_fraction,
                "png_sha256": sha256(png),
                "pdf_sha256": sha256(pdf),
                "svg_sha256": sha256(svg),
                "formats_present": all(path.exists() and path.stat().st_size > 0 for path in (png, pdf, svg)),
                "burned_panel_labels_found": banned_found,
                "no_burned_panel_labels": not banned_found,
                "not_blank": nonwhite_fraction > 0.025,
            }
        )
    source_files = sorted(source_dir.glob("*.csv"))
    report = {
        "status": "pass" if all(row["formats_present"] and row["no_burned_panel_labels"] and row["not_blank"] for row in findings) else "fail",
        "panels_expected": len(PANEL_CATALOG),
        "asset_files_expected": 3 * len(PANEL_CATALOG),
        "asset_files_found": len(list(figures_dir.glob("*.*"))),
        "font_family": "Arial/Helvetica/DejaVu Sans",
        "png_dpi": 600,
        "word_target": "Fourteen independent English image objects for manuscript Fig.14–27; captions remain editable Word text, not pixels.",
        "source_files": [{"name": p.name, "bytes": p.stat().st_size, "sha256": sha256(p)} for p in source_files],
        "panel_checks": findings,
    }
    (reports_dir / "automated_visual_qa.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def write_manifest(root: Path, source_dir: Path, figures_dir: Path, reports_dir: Path, qa: dict[str, object]) -> None:
    outputs = sorted([p for p in figures_dir.rglob("*") if p.is_file()])
    manifest = {
        "bundle_id": "intentrag_sepia_nature_v1_20260924",
        "source_bundle": "locked mechanism-validation CSVs plus signed-cache grouping summary",
        "seed": SEED,
        "design_change": "Fourteen independent, title-free English figures use a unified color-vision-safe journal style. Fig.27 is redesigned as an equivalence-class pairability diagnostic; no statistic or data scope is changed.",
        "panel_count": len(PANEL_CATALOG),
        "formats_per_panel": ["png", "pdf", "svg"],
        "source_inputs": [{"path": os.path.relpath(p, root), "sha256": sha256(p), "bytes": p.stat().st_size} for p in sorted(source_dir.glob("*.csv"))],
        "outputs": [{"path": str(p.relative_to(root)), "sha256": sha256(p), "bytes": p.stat().st_size} for p in outputs],
        "catalog": PANEL_CATALOG,
        "qa_status": qa["status"],
        "interpretation_boundaries": [
            "No panel estimates mutual information, semantic entropy, causal compression, transformer KV-cache hits, latency, maintenance correctness, action authorization, or safety.",
            "Role transport uses compositional observable masses; LOO sensitivity does not rerun retrieval and is not a Shapley or causal effect.",
            "Documentary ambiguity is deterministic identity coarsening, not semantic information loss.",
            "Accepted/exact hits are rule-derived locked post-hoc audit labels.",
            "Bootstrap intervals retain the original 5,000-resample aircraft-stratified normalized-exact-query-text-cluster design.",
            "Entropy discrimination and persistent-only rescue are sample-internal associations; persistent-only has 7 records / 4 unique texts, all Boeing.",
            "The signed-grouping panel reports non-singleton membership, an ideal one-cold-start-per-class ceiling, and pairable token-mass retention. It is not a measured request-sequence or GPU KV-cache hit rate.",
            "Observed-feature agreement does not establish identical evidence identity; the observed-feature signature has weighted Top-4 evidence purity 0.00935 in the locked summary.",
        ],
    }
    (reports_dir / "split_panel_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build 14 standalone manuscript figures from locked derived CSV files.")
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args()
    source_dir = args.source_dir.resolve()
    root = args.out_root.resolve()
    figures_dir = root / "figures"
    reports_dir = root / "reports"
    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    configure_style()
    frames = load_inputs(source_dir)
    generate_panels(frames, figures_dir)
    write_catalog(reports_dir)
    qa = validate_assets(source_dir, figures_dir, reports_dir)
    write_manifest(root, source_dir, figures_dir, reports_dir, qa)
    print(json.dumps({"status": qa["status"], "panels": len(PANEL_CATALOG), "assets": len(list(figures_dir.glob("*.*"))), "out_root": str(root)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
