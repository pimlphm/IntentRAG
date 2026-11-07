# 原文基线、历史高分与扩大实验的协议审计

审计日期：2026-10-14。输入是第二份全文附件及项目已有脚本和 CSV；没有运行官方 MemGAS、CAG、RAPTOR 或 Microsoft GraphRAG。新建的 adapter 只复用原项目的受控方法。

## 原文实际比较的 15 个设置

| 原方法标签 | 实际实现 | 与官方方法的关系 |
|---|---|---|
| BM25/word | V21 词级 1–2 gram TF-IDF cosine | 不含 BM25 TF 饱和/文长公式，展示名必须写 Word TF-IDF |
| TFIDF-char | V21 char_wb 3–5 gram TF-IDF | 经典字符检索控制 |
| SVD semantic surrogate | 词 TF-IDF + 96 维以内 TruncatedSVD | 不是 SBERT/DPR/ColBERT |
| Hybrid lexical | V21 0.45 word + 0.45 char + 0.10 phrase | 项目受控混合，非 neural hybrid |
| Parent-section retrieval | section-title token 匹配与传播 | 不是 RAPTOR 的递归聚类摘要树 |
| Component-profile retrieval | component/profile/section 权重 | 项目受控结构检索 |
| Controlled GraphRAG | SVD + section + page/paragraph structural bonus | 不含官方实体关系抽取/社区报告流程 |
| Causal-graph informed RAG | query component/role + cause cue mask | 内部结构信号控制，非物理因果验证 |
| Component-graph informed RAG | component/section/role 加权 | 内部结构信号控制 |
| Agentic RAG | 正则识别可见 warning/threshold/action/cause 等线索，固定加权路线 | 不是自主 Agent、ReAct 或 Self-RAG 训练复现 |
| Evidence-flow baseline | hybrid/component/role/section/graph + diversity penalty | 项目受控证据流 |
| GaRAGe-style relevance RAG | relevance/role/section/component 权重 | GaRAGe 是 benchmark，原代码为受启发控制 |
| GaRAGe-style attribution RAG | section/authority/component/role 权重 | 不是官方复现 |
| GaRAGe-style deflective RAG | 固定加权检索 +另设拒答探针 | retrieval 排名本身并未完整实现拒答 |
| FD-GRAG-RC-V21-final | 原 V21 lexical/cue/package 多路线 quota union | 原运行 quota 依赖 benchmark track；新 adapter 使用可见 cue marker，必须注明适配 |

旧实现入口：

- `10_scripts/build_case_sota_evidence_chain_outputs.py`：`METHOD_ORDER`、`CaseRetriever`、11 类结构评分和 Top-K。
- `_pipeline_v21/p0_build_eval_v21.py`：`build_indexes`、`query_scores`、`route_union_indices`，四个已有基础/最终设置。
- `_pipeline_aesr_v2/run_evidence_state_diagnostic_audit.py`：已有无 cue、dense/reranker、original-order window 和干预审计，可用于比较强神经检索，但不能改名为上述官方方法。

## 历史高分的分母与含义

| 数据层 | Boeing | Cessna | 解释 |
|---|---:|---:|---|
| 15 方法 atlas 全量 query | 197 | 279 | 共 476 个问题 × 15 = 7140 行；不是独立 held-out 总分 |
| Locked benchmark | 88 | 126 | 共 214 个 held-out 问题；其中 broad 4/13，unique 84/113 |
| 原 no-cue stress | 59 | 92 | 共151；与 source-cued 定位任务分开报告 |
| 原无证据探针 | 8 | 8 | 样本量很小，不能稳定估计总体错误支持率 |

Atlas 的 IntentRAG 数值（原 CSV，不是新结果）：

| 指标 | Boeing n=197 | Cessna n=279 |
|---|---:|---:|
| Exact hit@10（旧列名 recall） | 1.000000 | 0.946237 |
| Accepted hit@10 | 1.000000 | 0.989247 |
| Package coverage proxy | 0.995262 | 0.994265 |
| Retrieval grounded score proxy | 0.998862 | 0.993557 |
| One-minus-grounded risk proxy | 0.001138 | 0.006443 |

Locked 层 IntentRAG 的 accepted hit 为 1.000000/0.992063，package proxy 为 1.000000/1.000000，Cessna unique exact hit 为 109/113=0.964602。Boeing locked 88 个 query 上，原七种 V21 设置均达到 grounded 1.000，所以该层已经无法区分方法。

历史自由生成 Qwen2.5:3B 的 answer-grounded proxy 为 Boeing 0.599296、Cessna 0.762335；citation coverage 为 0.187817/0.594982。论文中的 0.927/0.917 属于 deterministic evidence-card assembly。上述均不等同于由独立专家标注的端到端维修问答正确率。

数值来源：Cases/Boeing_737 和 Cases/Cessna_172 的 `8 最新图集/*_sota_metrics_by_method.csv`、`*_qwen_generation_summary.csv`；`04_results/v21/v21_metrics_by_case_track.csv` 和 `v21_metrics_by_case.csv`。

## 会造成高分解释失真的两项协议问题

第一，`build_benchmark()` 从 gold evidence 原文生成 short_anchor，再把 `Evidence cue:` 加进 unique 问题；broad 问题把多个 exact gold 原文 anchor 加成 `Include evidence cues:`。在线函数没有读 qrels，但问题在生成阶段包含目标原文定位线索。历史高召回应解释为 source-cued / evidence-sufficient lookup。自然需求、无 cue 改写和跨条件问题需要单列，不能承诺自动保持旧 99%。

第二，两个旧 grounded 公式不同。V21 unique 使用 exact/accepted/page/section/package 的加权组合，并把 package 提升到至少 accepted hit，page/section 也允许 accepted hit 代替。Atlas 使用 0.32 accepted + 0.20 package + 0.17 page + 0.16 section + 0.15 role；package 设置 accepted 时至少 0.65、page 和 section 都 hit 时至少 0.55。Atlas role proxy 是不同角色数除以角色数上限，不能保证覆盖 query 真正需要的角色。扩大实验应单列严格 span hit、严格必需角色覆盖、实际展示来源支持和真实生成支持，而非只比加权 proxy。

原 split 连通分组以 exact IDs 为键；扩大实验建议使用 accepted-equivalence/source overlap groups，并对这些组做 paired cluster bootstrap。改写同一根问题不增加独立样本量。

## 官方强基线核对及复用/粒度统一方式

MemGAS 的官方 Python 实验代码构建 session、turn、summary、keyword 四种表示，GMM 选相关记忆连边，依据 similarity softmax 的 Shannon entropy 做 inverse-H 路由，PPR 再排序并由 LLM 过滤。移植维护文档时，应明确 section 对应 session、atomic source span 对应 turn；summary/keyword 的生成只读 source，保存 leaf provenance。官方来源：[论文](https://arxiv.org/html/2505.19549)、[作者仓库](https://github.com/quqxui/MemGAS)。当前没有运行官方方法。

CAG 将相关资源预装入上下文并缓存实际 KV；其边界是资源必须装入上下文窗口。维护语料超过可用 context 时，可比较受预算限制的静态来源前缀，但应叫 budgeted static-prefix policy，而非完整 CAG 复现。实际 KV 需执行 `kvcache.py` 或有等价公开 cache API 的运行时。来源：[作者仓库](https://github.com/hhhuang/CAG)。

RAPTOR 用递归 embedding、clustering、abstractive summarization 建树，可自定义本地 embedding 和 summarizer。需要保存每个 summary node 的子节点映射，在固定 token 预算下物化 source span；仅按 parent section 标题加权不足以称 RAPTOR。来源：[官方仓库](https://github.com/parthsarthi03/raptor)。

Microsoft GraphRAG local search 使用实体入口及关联 text units、relations、community reports，并在预算内构建 context。官方 indexing 过程和社区摘要没有在旧控制中实现。来源：[官方 local-search 文档](https://microsoft.github.io/graphrag/query/local_search/)、[官方仓库](https://github.com/microsoft/graphrag)。

统一测量须固定 source snapshot、相同可见 query、generator、context token budget、顺序请求流、cache memory budget、cold/warm start 和失效规则。不同方法自然表示可以不同，但检索成功只能按实际送给生成器的 source span 衡量：一个 summary 不能因为挂了整个 section 的 IDs 就把所有子 evidence 算命中。单独报告 routing coverage 和 materialized-support coverage。

复用报三层：相同 token prefix 的可复用长度；实际缓存运行时的 prefill/TTFT（若测到）；authority/scope/condition/revision 兼容支持的保留率及错误支持阻断。若只做离线离散 key replay，就写 simulator/prefix proxy，不能叫 GPU hit-rate。

多粒度消融至少为 atomic-only、section-only、summary-only、keyword-only、固定 hybrid、entropy router、constrained router，并区分跨飞机/组件/条件/来源变更。对 summary-only 的损失不得靠人为降低 baseline；应使用真实固定预算和同一盲评标准。

## 新适配器接口与检查

`scripts/baseline_adapter.py` 的 `build_index(project_root, evidence=online_df)` 只读取 document-derived index、aliases 和 causal support 文件。`index.retrieve(question, aircraft, method, k=10)` 返回原 evidence ID、rank、score、source text、role 和 provenance；`retrieve_all` 提供15方法输出。没有 track/query_type/qrels 参数，投影白名单删除评测列。V21 的 TF-IDF 仍在 joint corpus 拟合，aircraft mask 在评分后施加，保留旧 IDF 定义。

Smoke：两案例各100条 source evidence，注入 `gold_evidence_id=POISON` 后确认 online projection 不含此列；15 方法均输出10条同飞机 evidence，共150条；状态 PASS。该 smoke 是适配器可执行性检查，不是性能实验。
