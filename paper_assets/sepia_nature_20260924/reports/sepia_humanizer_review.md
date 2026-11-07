# SEPia / Humanizer 写作修改审阅记录

## 审阅范围

- 文档类型：航空维护 RAG 学术论文，英文正文。
- 操作：SEPia `review → refactor`，再做 Humanizer 结构性语气扫描。
- 修改范围：只处理 17 个指定段落，不做全文重写；所有修改继续使用蓝色字体。
- 保护内容：公式、表格、引文、文献条目、数值、图号、原始结果文件和未指定段落。图件替换与文本改写分别审计。
- 作者模型：unknown。
- 执行模型：GPT family，精确 release 未提供。
- Prose layer：author=none；executor=prior。
- Venue corpus：未提供目标期刊的近期人工作品，沿用原稿的学术英语和术语系统。
- 加载规则：SEPia `professional-pass.md`、`style-pass.md`、中文校准说明和 GPT prose-layer prior；Humanizer 3.0.0。

## 第一阶段：发现的问题

1. 部分段落先宣布“真正的问题”“科学对象”或“最重要的区别”，再给事实。这样的舞台式结构削弱了方法边界。
2. 三项贡献在多个位置以几乎相同的节奏重复，摘要、引言、讨论和结论之间有复述。
3. `not X but Y`、`not merely`、`not automatically` 等对照句出现较密。有些对照承担必要的科学边界，应保留；只起强调作用的对照被改为直接陈述。
4. signed prefix 部分把离线分组代理写得接近已实现缓存键或真实 KV hit，容易让读者把输入侧诊断误解为 GPU 运行时结果。
5. “purity = 1.000” 缺少构造性解释。该一致性部分由分组字段定义，且 observed-feature signature 的 Top-4 evidence-identity purity 只有 0.00935。
6. 原稿部分长句同时承载方法、结果和限制。改写把主张与边界拆开，但没有通过填充连接词来追求表面流畅。

Deferred：全文未指定段落仍保留原有写法；没有对参考文献真实性、统计重算或自然查询外部效度做独立验证。

Protected：所有公式、引用文本、文献项、表格和未指定段落。

Verdict：局部存在成组的模板化结构和一处重要的 claim-strength 风险，适合最小重构，不需要重写整篇论文。

## 第二阶段：已执行的重构

补丁覆盖正文段落索引：`7, 16–20, 102, 113, 117, 236–237, 299, 302–303, 305, 308–309`。对应位置包括摘要、三项贡献、方法边界、讨论、signed-prefix 小节及结论。

`reports/docx_patch_audit.json` 记录了 17 次替换：9 段缩短，8 段因补入必要的测量边界而变长；字符总数由 9,469 增至 10,883。增加的文字主要用于纠正 Fig. 27 和缓存论证的测量对象，不是修辞性扩写。

主要改动如下：

- 把“三项贡献”写成可区分的研究对象：可观测证据分布、可恢复的多粒度文档身份、admissibility-constrained prefix reuse。
- 将“LLM 是 verbalizer”限制为本文的接口设定，不把它写成模型能力的普遍定理。
- 把“cache key”改为部署目标接口，并明确本次执行的是 query-level post-hoc grouping；没有验证 byte/token-identical prefix。
- 不再把 Fig. 27 的横轴称为 hit-rate。正文和图注改为 non-singleton membership、idealized first-touch/no-eviction ceiling 与 pairable token-mass retention。
- 把 observed-feature agreement 1.000 与 evidence identity 分开，并加入 Top-4 identity purity 0.00935 的反例约束。
- 把“模型规模不能替代证据状态”改写为本实验支持的有限结论：给定现有航空维护数据，规模本身没有建立 aircraft、revision、condition、provenance 或 authority 边界。

## Humanizer 终检

在改动的 17 段中逐项检查：聊天式开场、空泛结尾、机械三段式、连续破折号、装饰性加粗、借用未指明权威、重复结论和宣传性形容词。终稿没有新增事实、引文、参考文献或外部排名。必要的否定句仍被保留，因为它们界定“代理量不等于真实 KV 命中”“观测一致性不等于证据身份一致”等实质性边界。

本次终检只覆盖改动段落，不能据此声称全文没有任何模板化表达。SEPia 与 Humanizer 也不是 AI 文本检测器；这里记录的是编辑过程和保真约束，不是作者身份结论。

## 文件级验证

- 最终稿 SHA-256：`560E28925998F91A9AED98B719FB99E0E5FCE4C2B5F96F9D81FC0FAF81C6C264`
- 基础稿 SHA-256：`70FDEF5AD693835757BC2A0FA10CA2E02A69240BC768058BE879776EDAAFDADD`
- `reports/final_docx_structural_audit.json`：PASS。
- 17 个旧段落均已消失；新段落各出现一次并保持蓝色。
- 公式规范化哈希未变；绘图对象数量、关系 ID 和版面尺寸未变。
- ZIP 中除 `word/document.xml` 与 Fig. 14–27 对应的 14 个媒体文件外，其余成员按字节保持不变。

后续若继续改写，建议仍以段落为单位执行“先列问题、再最小修改、最后核对数值与限制”的流程。全文改写会增加术语漂移和结论升级的风险。
