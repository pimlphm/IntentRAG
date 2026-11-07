# 可写入论文的结果解释：Admissible Reuse Frontier

## 核心科学问题

本文不宜再把贡献表述为“又设计了一套 RAG pipeline”。更稳妥也更有新意的对象是：
在生成之前，系统如何把外部检索到的证据组织成一种可复用的生成状态；以及这种复用是否仍满足航空维护场景中的范围、出处、权威性、条件和动作前提约束。

因此，RAG、无需检索的语言模型以及 evidence-state assembly 可以被放在同一个观察坐标中：它们都是在回答前组织知识状态的方式。但本文的结果表明，真正关键的不是“是否检索”，也不是“模型够不够大”，而是可复用状态中有多少仍然是当前查询可采纳的证据。

## 图 A：Admissible reuse frontier

可写结论：
当前项目中存在明显的 raw evidence-state reuse。214 个 source-derived V21 查询上的平均 raw reuse ratio proxy 为 0.623，95% bootstrap CI 为 [0.574, 0.672]。但是，当进一步要求 reused state 同时满足 scope/provenance/authority/condition/action-prerequisite 代理条件时，保留下来的 reuse ratio proxy 降至 0.242 [0.195, 0.291]。

这说明“可以被 KV/prefix 复用的 token 状态”和“可以被当前维护查询采纳的证据状态”不是同一个量。这个分离就是本文比普通 RAG 压缩或 cache 优化更像科学问题的地方。

不能写成：
真实 KV cache 命中率、真实互信息增益、真实安全风险下降、专家级 action authorization。

## 图 B：为什么 Boeing raw reuse 高于 Cessna

可写结论：
Boeing 737 在当前语料和 evidence-card 表示下 raw reuse 更高：0.874 [0.846, 0.897]，而 Cessna 172 为 0.448 [0.388, 0.511]。原因不是“Boeing 天然更可复用”，而是当前 Boeing evidence packages 更集中、更重复：reused evidence ID count 更高，redundancy proxy 更高，role diversity 也更高；相反，Cessna source diversity 更高，证据分布更碎片化。

同时，Cessna 的 action-prerequisite availability proxy 更高（0.587 vs Boeing 0.330），说明 raw reuse 高不等于动作支持更强。这个结果恰好支持 admissible reuse frontier：高复用可能来自重复和集中，但能否用于动作回答还要看权威性、条件和程序性证据是否齐备。

不能写成：
Boeing 机型普遍比 Cessna 更适合 KV cache，或 Cessna 维护知识更复杂。只能说是当前语料、当前 chunk/evidence-card 表示和当前查询集下的观察。

## 图 C：模型规模不能替代项目证据状态

可写结论：
在本地 Qwen 0.5B 到 14B 的受控诊断中，direct model-only 的 groundedness proxy 保持在约 0.48-0.51，且 citation coverage 为 0。提供 raw evidence chain 或 organized evidence state 后，groundedness proxy 上升到约 0.72-0.85。state-first 相比 direct model-only 的平均 groundedness proxy 增益为 0.301。

这支持一个有限但有价值的结论：在当前航空维护项目知识上，更大模型没有自行重构出项目特定、可引用的证据状态；模型规模主要改善语言表述和消费证据的能力，而不是替代证据构建。

不能写成：
所有 retrieval-free language model 都不行；或 RLM 架构必然不能内化维护知识。这里只是 local Qwen scale ladder 上的项目诊断。

## 图 D：cache-friendly organization 与 admissible filtering 的分离

可写结论：
state-first organization 显著提高 tokenizer-level cache-eligible shared-prefix structure：state-first state 的 shared-prefix ratio proxy 为 0.718，而 question-first raw chain 只有 0.036。这说明 evidence state 的顺序和形式确实更适合 prefix/KV reuse。

但是右图同时显示，高 prefix reuse 并不自动等于高 admissible reuse。Boeing raw reuse 从 0.874 经 action-prerequisite 代理过滤后降至 0.286；Cessna 从 0.448 降至 0.211。也就是说，本文的贡献不是“把 prompt 写得更像 cache”，而是把 cache-friendly state 与 maintenance-admissibility constraints 放在同一前沿上分析。

## 图 E：Synthetic intervention response

可写结论：
post-retrieval synthetic interventions 显示，scope swap、provenance masking、authority removal、conflict injection 等扰动能够按预期改变目标代理指标。这个结果支撑的是“这些约束代理对可采纳证据状态具有方向敏感性”，不是五状态分类准确率。

需要如实说明的负结果：
Boeing 的 condition removal 在当前自然包中没有产生方向响应，procedural-to-descriptive substitution 也几乎没有响应。这说明当前 condition/procedural 代理还不够强，未来需要更细的程序性证据标注和 action-authority annotation。这个负结果应该保留，因为它能让论文显得更可信。

## 图 F：RLM、RAG chain、Evidence-state assembly 对照

可写结论：
在现有本地诊断中，direct local model 的 mean groundedness proxy 为 0.497，citation coverage 为 0；question-first RAG chain 的 groundedness proxy 为 0.813，但 cache-eligible prefix ratio proxy 仅 0.036，mean prompt tokens 约 2598；state-first evidence assembly 的 groundedness proxy 为 0.798，prompt tokens 约 1796，并将 prefix ratio proxy 提高到 0.718。

这给出一个新的视角：RAG 与 RLM 的差别不只是“外挂检索 vs 内化知识”，而是回答前的知识状态是否能在“可引用、可复用、可约束”三个维度上同时成立。本文当前结果显示，state-first evidence assembly 牺牲了极少的 groundedness proxy，却显著降低了输入负担并增加了可复用前缀结构。

## 可作为 Results 小节标题

1. Model scale does not reconstruct project-specific evidence state.
2. State-first evidence assembly creates cache-eligible reusable generation state.
3. Raw reuse and admissible reuse diverge under aviation-maintenance constraints.
4. Corpus structure explains Boeing-Cessna reuse asymmetry.
5. Synthetic perturbations expose which admissibility proxies are direction-sensitive.

## 最稳妥的贡献表述

This study introduces an admissible-reuse perspective on evidence-grounded aviation maintenance support. Rather than treating RAG, retrieval-free language models, and evidence-state assembly as separate engineering paradigms, we analyze them as different ways of organizing pre-generation knowledge state. The empirical contribution is a proxy-based admissible reuse frontier: a measurement of how much reusable evidence-state mass remains after scope, provenance, authority, condition, and action-prerequisite constraints are imposed. The results show that cache-eligible reuse can be large, but admissible reuse is substantially smaller and corpus-dependent; therefore, aviation maintenance support requires not only retrieving or compressing evidence, but organizing reusable evidence states whose reuse is bounded by documentary admissibility.

