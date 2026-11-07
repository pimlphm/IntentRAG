# 同源正文词元预算的复用检验：方法、图件与结果边界

## 为什么新增这一检验

原16/64/256容量单位是精确来源证据模块数。同样64个模块可能包含64条短span，也可能包含64个长section；因此只控制模块数不能称为等正文存储量，更不能称为等RAM或KV字节。原模块回放保留作为源对象局部性诊断，本轮另以统一正文词元容量检验预算敏感性。

## 冻结协议

使用与最终检索实验相同的690个question IDs和33种实际执行方法、22,770份冻结Top-10返回列表。每个split独立冷启动，所有方法使用同一五种随机请求顺序（seed=20261014–20261018），另保留一条按可见问题文本的主题及机型排列的离线顺序。重复顺序不增加独立问题样本量。

缓存键严格包含aircraft、document、page、evidence ID与精确源正文SHA256。正文成本由项目固定的官方Qwen2.5 tokenizer计数，不含special tokens；tokenizer路径及hash记录于manifest。总容量为1,024/4,096/16,384源正文词元。每个问题先查询上一问题结束时的缓存，然后按冻结返回次序统一更新LRU；不计本题刚插入内容为本题命中。同题重复ID只计一次源正文量。超过总容量的源条目不准入缓存，其请求数和词元质量单独统计，但检索返回仍保持原样。已存在条目触摸至LRU末端，逐项淘汰直到满足预算。没有page/section索引缓存的额外策略。

脚本内硬断言每次更新后正文词元占用≤容量，并检查每次冷启动为零命中、超长条目不命中、复用量不超过请求量、冻结检索输出不变。7项LRU单元行为测试通过，正式回放409,860行（690×33×3×6），运行37.913秒，6项不变量全通过；requested query-method覆盖22,770，无缺失。权威与revision仍未由现有源元数据验证。

源正文token容量不包括identity key、LRU管理、routing索引、进程内存或GPU KV张量；只能称“等源正文词元预算”。不是测量GPU缓存hit或prefill节省。身份支持保留率恒为1是冻结检索输出的设计不变量，不能作为准确率或安全性提升证据。

## 新鲜来源问题集的真实结果

下表复用值是五个随机工作负载实现的均值，accepted Hit@10及必要原文支持是独立冻结检索结果（fresh214题、214文本cluster）。不能把高复用解释为检索/QA改善。

| 方法 | accepted Hit@10 | 必要原文支持比例 | 4,096-token源正文量复用 | 16,384-token源正文量复用 |
|---|---:|---:|---:|---:|
| Profile-resolved | 0.018692 | 0.014019 | 0.782433 | 0.884004 |
| Causal-graph informed RAG | 0.607477 | 0.544393 | 0.012363 | 0.226472 |
| Long-term fixed | 0.247664 | 0.238318 | 0.011232 | 0.041214 |
| Detail-only | 0.887850 | 0.829439 | 0.007563 | 0.046339 |
| IntentRAG balanced fair | 0.948598 | 0.925234 | 0.004397 | 0.026049 |
| True BM25 | 0.971963 | 0.943925 | 0.001334 | 0.010735 |
| Multi-granularity fixed | 0.803738 | 0.771028 | 0.009042 | 0.042073 |
| Multi-granularity entropy | 0.831776 | 0.801402 | 0.009030 | 0.041171 |
| Multi-granularity entropy+constraints | 0.864486 | 0.813084 | 0.008279 | 0.040821 |
| IntentRAG multi + mMARCO CE@50 | 0.929907 | 0.899533 | 0.002620 | 0.012435 |
| Neural dense + mMARCO CE@50 | 0.827103 | 0.803738 | 0.002874 | 0.013548 |

### 这一公平性控制是否改变原结论

Profile route的“高复用、低支持”并未被等词元预算消除：64模块回放复用0.778680，对应4,096词元为0.782433，而其必要原文支持只有0.014019。其低覆盖与高重复来源对象同时出现，表明复用的集中度不能替代局部依据的充分性。它不证明所有粗粒度状态均无效，只约束本次实际执行route与已冻结任务。

Causal graph control则改变很大：64模块回放复用0.699960，4,096词元容量仅0.012363。该预算下，15.70%的模块请求超长，覆盖请求正文词元质量的83.09%，因此原模块计数容许若干长块占据巨大正文量，抬高了token-weighted reuse。不能据旧值概括“结构化方法普遍有高缓存收益”。16,384容量下全部请求可准入，但复用仍只有0.226472，说明容量、来源对象大小与请求局部性共同决定效率端。

本文主方法在这组精确ID随机工作负载中也没有突出复用增益：4,096词元复用0.004397；多粒度版本均约0.008–0.011，而真实BM25支持仍最高。不能写“多粒度动态路由全面胜出”或“检索高分保证缓存高分”。当前结果支持的是：同一来源支持目标需要分开测量coverage、必要原文充分性、状态集中度和预算受限的复用；query-conditioned粒度选择的收益需通过配对支持差及真实缓存实现分别验证。

## 两张独立图的中文名称与机理解释

1. `figures_token_budget/fresh_quality_reuse_4096_source_tokens`：**等正文词元预算下的检索支持—复用关系**。横轴为4,096源正文词元预算的mean workload reuse，纵轴为accepted Hit@10。完整33种方法保留，颜色/形状表示机制类别，星形为balanced fair。左侧大多数方法有高低不一的support但复用有限；右下的Profile route高重复而几乎不提供fresh必要支持。它将“同一来源对象反复出现”和“当前问题需要的证据被找回”明确分离。多个点重合时，全部method仍在source CSV逐项列出，无抖动造值或删除低分方法。

2. `figures_token_budget/fresh_reuse_capacity_source_tokens`：**源正文词元容量与长短粒度状态复用**。横轴为统一正文token预算，纵轴为复用量。点是五种随机顺序均值，误差线是其min–max范围，不是总体CI。Profile route随容量增长明显复用，但其他route在同预算下复用较低；因此多粒度接口的评价应同时计入对象长度、局部性和支持保留。此图是声明的8项机理比较视图，所有33方法另在上图与全量表中保留。不能从该曲线推断真实GPU KV或问答质量改善。

两图均英文、单坐标区、无图内标题或(a)/(b)；12pt主要标签、13pt轴标签、轴箭头，图例在图外留白；PDF/SVG/600dpi PNG和Source Data已导出。绘图遵循nature-figure的源代码、PDF字体、碰撞和人工审阅流程，不声称符合某个未指定期刊的最终投稿尺寸。

## 可复现入口

- `scripts/replay_token_budget.py`（`--self-test`可运行独立LRU测试）
- `scripts/make_token_budget_figures.py`
- `results/memory_replay_token_budget/replay_manifest.json`
- `results/memory_replay_token_budget/replay_query_results.csv`
- `results/memory_replay_token_budget/replay_order_summary.csv`
- `results/memory_replay_token_budget/replay_summary.csv`
- `figures_token_budget/source_data/`：每图逐方法/逐顺序源数据
- `figures_token_budget/qa/`：源代码与最终PDF验证记录

原模块容量结果与新正文容量结果不能混用为一个横轴。建议论文主图采用本轮正文预算版本，模块数版本作为补充并明示预算边界。
