# IntentRAG Fig.14–27 独立英文图目录与逐图解释

所有图像均为独立 PNG/PDF/SVG；图内不烧录 `(a)–(e)`，也不含标题。英文图作为 Fig.14–27 的独立图片对象插入 Word，图名与解释由 Word 正文提供。

## fig18_01　证据角色质量迁移火山图

- English: Observable role-mass transport volcano plot
- Word position: Manuscript Fig. 14
- File stem: `fig18_01_role_mass_transport_volcano`
- 含义：比较检索后角色质量 p_R 与可见查询需求代理 p_q 的平均差，并以角色族内 BH-FDR 显示偏移的稳定性。warning 与 authority 向检索包富集，而 action 与 condition 相对需求不足。
- 解释边界：这是可观测组成质量的描述性迁移，不是互信息、因果效应或真实知识分布。主分析排除了 16 条 visible-query fallback 记录。
- 数据：`fig18_role_transport_summary.csv`

## fig18_02　固定证据包留一对齐敏感度火山图

- English: Fixed-package leave-one-out alignment-sensitivity volcano plot
- Word position: Manuscript Fig. 15
- File stem: `fig18_02_fixed_package_loo_sensitivity_volcano`
- 含义：度量从既有证据包中移除一个角色条目时，可见需求对齐代理的变化；action 与 condition 为正，而 warning 与 evidence 为负，说明被大量选择不等于对齐效用为正。
- 解释边界：LOO 不重新运行检索，不是 Shapley 值或因果条目贡献；显著性在该角色族内进行 BH-FDR。
- 数据：`fig18_role_loo_summary.csv`

## fig18_03　角色迁移—对齐敏感度联合机制图

- English: Joint role-transport and alignment-sensitivity mechanism plane
- Word position: Manuscript Fig. 16
- File stem: `fig18_03_role_transport_vs_loo_sensitivity`
- 含义：把同一证据角色的检索质量迁移和固定包 LOO 敏感度放在二维平面中。warning 位于高迁移但负敏感度象限，action/condition 位于负迁移但正敏感度象限，直接展示选择偏置与下游效用的解耦。
- 解释边界：点大小只编码已选择条目数；四象限关系是关联性机制诊断，不构成因果方向证明。
- 数据：`fig18_role_transport_summary.csv; fig18_role_loo_summary.csv`

## fig18_04　精确查询文本簇 LOO 敏感度雨云图

- English: Exact-query-text-cluster LOO-sensitivity raincloud plot
- Word position: Manuscript Fig. 17
- File stem: `fig18_04_exact_text_cluster_loo_raincloud`
- 含义：展示六类关键证据角色在归一化精确查询文本簇层面的 LOO 敏感度分布，而不只显示总体均值；长尾说明同一角色在不同查询中可能产生异质贡献。
- 解释边界：分布单位是 aircraft×normalized exact visible-query text×role 的簇均值，不是 930 条原始角色记录的独立样本。
- 数据：`fig18_exact_text_cluster_role_loo_nonfallback.csv`

## fig19_01　多粒度文档身份歧义雨云图

- English: Multi-granularity documentary-identity ambiguity raincloud plot
- Word position: Manuscript Fig. 18
- File stem: `fig19_01_documentary_ambiguity_raincloud`
- 含义：比较 Page、Section 与 Component-family 分组中包含的 detail identity 数量。粒度越粗，单个分组折叠的细粒度身份越多，尤其 Cessna component-family 层级出现极高歧义。
- 解释边界：这是确定性的文档身份粗化指标，不是语义信息熵、token 压缩率或互信息损失。
- 数据：`fig19_group_size_distributions.csv`

## fig19_02　多视图重合度—条件救援率矩阵

- English: Cross-view overlap and conditional-rescue matrix
- Word position: Manuscript Fig. 19
- File stem: `fig19_02_view_overlap_and_conditional_rescue_matrix`
- 含义：上三角显示不同粒度 Top-10 证据集合的平均 Jaccard，下三角显示当列视图失败时行视图的条件救援率；Profile 对 Detail miss 的救援率仅 0.09，而 Detail 对 Profile miss 的救援率为 0.71。
- 解释边界：救援率依赖 locked post-hoc accepted-hit 代理及方向化条件分母，矩阵上下三角不可作为对称距离解释。
- 数据：`fig19_overlap_rescue_matrix.csv`

## fig19_03　单视图命中交集 UpSet 图

- English: Single-view accepted-hit intersection UpSet plot
- Word position: Manuscript Fig. 20
- File stem: `fig19_03_single_view_hit_intersections`
- 含义：把 214 条 locked 记录按 Detail、Page、Section、Profile 的命中组合分解，并按 Boeing/Cessna 堆叠。最大交集是 Detail+Page+Section 命中而 Profile 未命中，说明持久层不能替代局部多粒度证据。
- 解释边界：交集编码的是规则派生 accepted-hit@10，不是维修正确率或专家授权结论。
- 数据：`fig19_hit_pattern_summary.csv`

## fig19_04　持久层相对查询局部层命中差森林图

- English: Persistent-minus-query-local hit-difference forest plot
- Word position: Manuscript Fig. 21
- File stem: `fig19_04_persistent_minus_query_local_forest`
- 含义：比较 persistent-fixed 与 query-local-fixed 的 hit@10 差异及按精确查询文本簇分层 bootstrap 区间。总体 accepted 差为 −0.289、exact 差为 −0.377，只有少数任务族显示条件互补。
- 解释边界：区间按 aircraft 分层、normalized exact visible-query-text cluster 重采样；小样本任务族区间较宽，不能解释为普适优越性。
- 数据：`fig19_query_family_paired_effects.csv`

## fig20_01　长短期路由分子—命中结果联合平面

- English: Query-local versus persistent router-numerator outcome plane
- Word position: Manuscript Fig. 22
- File stem: `fig20_01_router_numerator_outcome_plane`
- 含义：把短期局部层与长期持久层的路由分子放在二维平面，并按两种视图的命中结果着色；等权重射线说明路由权重由相对证据集中度决定，而不是由模型规模直接给出。
- 解释边界：路由分子来自同一相似度构造下的归一化 concentration/entropy 代理；不是跨模型可比的校准概率。
- 数据：`fig20_router_outcome_records.csv`

## fig20_02　持久层权重五分位命中组成图

- English: Outcome composition across persistent-weight quintiles
- Word position: Manuscript Fig. 23
- File stem: `fig20_02_persistent_weight_quintile_outcomes`
- 含义：展示 persistent weight 从 Q1 到 Q5 时 Both、Query-local only、Persistent only 与 Neither 的组成变化。全部 7 条 persistent-only 记录落入 Q5，但 Q5 同时有 48.8% Neither，说明高长期权重只能触发分支，不能替代局部证据。
- 解释边界：五分位和结果来自同一 214 条 locked 样本，是样本内部描述，不是外部校准或决策阈值验证。
- 数据：`fig20_weight_quintile_outcomes.csv`

## fig20_03　持久层独占救援识别精确率—召回率图

- English: Precision-recall analysis for persistent-only rescue
- Word position: Manuscript Fig. 24
- File stem: `fig20_03_persistent_rescue_precision_recall`
- 含义：评估 long-term weight 对 persistent-only 案例的样本内排序识别能力。All 的 AP 为 0.137，Boeing 为 0.379；相对阳性率基线虽有提升，但绝对精确率仍低。
- 解释边界：阳性仅 7 条记录、4 个唯一文本且全部来自 Boeing；AP/AUC 区间为样本内簇 bootstrap，Cessna 救援判别不可估。
- 数据：`fig20_persistent_rescue_pr_curve.csv; fig20_persistent_rescue_pr_summary.csv`

## fig20_04　持久层救援分支预算—覆盖率权衡图

- English: Branch-budget trade-off for persistent-only rescue
- Word position: Manuscript Fig. 25
- File stem: `fig20_04_persistent_rescue_branch_budget`
- 含义：沿 long-term weight 排序逐步扩大分支预算，比较 persistent-only 救援覆盖率与候选精确率。前 41/214 个候选可以覆盖全部 7 条 persistent-only 记录，但对应精确率仅 17.1%，显示召回收益与审核负担的权衡。
- 解释边界：曲线是同一 locked 样本上的排序后累积诊断；它不包含真实运行时成本、延迟、人工审核代价或外部阈值验证。
- 数据：`fig20_branch_budget_curve.csv`

## fig20_05　多粒度负熵命中判别森林图

- English: Within-view negative-entropy discrimination forest plot
- Word position: Manuscript Fig. 26
- File stem: `fig20_05_entropy_discrimination_forest`
- 含义：比较 Detail、Page、Section、Profile 的 −H_v 对各自 accepted-hit 的样本内 ROC-AUC，并同时列出 AP 与阳性率。随着粒度变粗，AUC 从 0.918 下降到 0.688。
- 解释边界：每个视图预测的是自身命中，标签与分数均来自同一 locked 样本；结果是关联性诊断，不是外部预测性能。
- 数据：`fig20_entropy_auc_ap_summary.csv`

## fig27　等价类复用机会—观测特征一致性图

- English: Equivalence-class reuse opportunity versus observed-feature agreement
- Word position: Manuscript Fig. 27
- File stem: `fig27_equivalence_class_pairability_plane`
- 含义：比较三种证据状态分组策略。实心圆表示进入非单例组的查询比例，空心三角表示在每组首请求必然未命中、无驱逐的理想序列下可达到的上限，点面积表示可配对 token 质量保留代理。观测特征签名以极小的可配对机会损失把组内观测特征一致性从 0.565 提升到 1.000；绑定 aircraft 与 Top-4 evidence 后机会显著收缩。
- 解释边界：这是基于分组等价类的离线可配对诊断。它没有真实请求顺序、序列化前缀哈希、层内 KV 状态、GPU 命中、TTFT、FLOPs 或显存测量；观测特征一致性也不等于证据身份一致性。
- 数据：`signed_cache_policy_summary.csv`
