# 最终问答接口与真实KV图件离线审查

## 本轮范围与结果

本轮仅离线读取已冻结的回答与真实KV数据，修改误导性轴标签、补全图注和Source Data分母，然后重绘重审；没有调用任何模型服务，没有尝试绕过socket 10013错误。语义判断的失败日志保持原样，全部2,112个semantic score仍为缺失，valid=0；未绘制、未交付任何空的语义正确率图。

最终8张机械问答forest（legacy/fresh×引文/JSON解析×原尺寸/Word尺寸）及2张重新导出的真实KV图，均经源依赖审查、最终PDF glyph与collision检查及183mm宽页面proof人工逐张审查。10/10字体和碰撞检查通过（collision FAIL=0、WARN=0），没有图标题、合并(a)–(d)、数字或文字遮盖、图例压线、轴箭头与标注交叠或裁切。

原入口脚本的纯静态检查**没有**被改写成PASS：raw为15 PASS、4 WARN、2 FAIL，两个FAIL来自入口导入Exporter、静态检查器不解析依赖而未找到export语句。完整raw报告保留为`figures_qa_kv/qa/final_qa_kv/source_raw.json`。独立解析本地Exporter、forest、make_summary、axes_arrows后的resolved chain为18 PASS、3 WARN、0 FAIL，记录为`source_resolved.json`；两者不能混称。实际PDF/SVG/600dpi PNG导出与PDF结果另行验证。

resolved WARN涉及没有TIFF、初稿9in宽不是未指定期刊默认宽度、实际观察数据bootstrap或只改变水平显示位置的jitter被误识别为模拟数据。它们已有明确边界：不声称期刊最终稿尺寸合规；bootstrap没有造观察值，KV竖轴数值没有抖动；authoritative PDF/SVG与600dpi PNG均存在。

## 问答分母与实际执行数量

64个冻结问题，legacy和fresh各32个unique question IDs与32个question-text clusters。33种方法每fold恰好32条，总2,112 query-method cells；66个method-fold组缺行=0、重复question ID=0、quote/JSON-parse未定义分母=0。

实际1933个generation calls，另179个identical-prompt alias cells共享已生成结果；status为2111 ok、1 http_error。2,112是请求方法格数，不应写成2,112次独立模型调用。图的32题分母保留该失败执行，机械指标按既定规则为零；不删失败以抬高均值。重复别名也不是新增独立问题样本。

详见`figures_qa_kv/qa/final_qa_kv/qa_fold_method_coverage_and_quote_denominators.csv`。每幅forest的Source Data还新增原有`quote_verbatim_precision_given_quote`、`n_answers_with_quotes`、`n_valid_json`、calls/alias/failure等字段，只补信息，不改变任何原指标值或CI。

## 两类问答图究竟测了什么

### 引文输出与实际上下文支持

中文名称：**旧问题集/新问题集的已识别引文上下文支持得分**。文件stem保留`legacy_answer_quote_support`与`fresh_answer_quote_support`，Word插图使用对应`_word`。

横轴已改为`Mean context-supported quote score (no recognized quotation = 0)`。对一个回答，parser只识别quotes列表中的dictionary项；得分是这些识别项中引用到实际输入source text、且非空quote可在该source中匹配的比例。未识别到dictionary quote时得分为零。再对全部32问题取mean，因而同时受到**引文出现/可识别性**和**已识别引文的原文匹配**影响。它不是仅在“给了引文”条件下的precision，也不是回答语义准确率。

全2,112格中仅11格存在parser-recognized quote，8格匹配原文、3格未匹配；条件均值8/11=0.727273，按全部回答平均只有8/2112=0.003788。这两个值的分母不同，不得互换，也不应在仅11个相关格上宣称可靠泛化精度。Source Data保留每method的条件值和发生次数；无已识别引文的方法，其条件precision保持缺失而非零或一。

system instruction是“至多一个引文”，并不强制非空；因此空quotes可以是合法输出。低all-answer quote score不能自动证明输出契约失败、答案不grounded或不能回答问题。自然文本/字符串引文未被既定parser识别也属于此指标边界，不能擅自改parser提高结果。即使匹配了输入原文，仍可能没有回答问题、证据不充分、机型/条件不适用或无维修授权。

### JSON与answer解析

中文名称：**旧问题集/新问题集的JSON对象与回答字段解析比例**。为兼容已引用文件，stem仍为`legacy_answer_schema_compliance`与`fresh_answer_schema_compliance`，但图内横轴已改为`JSON and answer parse fraction`，不再称完整schema compliance。

`structured_parse_ok`只要求输出能被JSON解析为dictionary且`answer`为字符串。它不检查citations或quotes必须存在、完整或正确，更不检查完整三个字段的schema。此图是输出接口的窄解析端点；高值不能被称为高问答准确率、完整输出契约符合率或维修建议安全性。

所有CI来自32个question-text clusters的2000次bootstrap。对全零或全一观察格得到的退化区间是当前观察样本的bootstrap性质，不是已证明未来必然零/一。

## 真实KV两图的含义与边界

- `actual_kv_reuse_by_prefix_layout`，**不同前缀组织方式的真实输入KV复用分布**。保留594个相关profile（6对visible queries×33方法×3布局，每布局198），竖轴是引擎实际cache计数除以目标prompt的native token count。原始点、violin、median/IQR仅描述样本分布；独立抽样单位为6对query，不是594次独立场景。共同system/scope header也计入cache，不是纯证据token。层级metadata不可作为答案证据。
- `native_prefix_overlap_vs_actual_kv`，**native共享前缀与真实输入KV复用的一致性**。横轴是seed-target native LCP，纵轴是实际复用cache input tokens；虚线为相等而非因果拟合。保留全部profile，冷目标—冷seed—暖目标协议及固定GPU slot、RAM prompt cache关闭均在图注中说明。计数不是最终slot occupancy；不能由重合度认证scope、revision、authority或答案正确性，也不能在并发设备负载下推出隔离GPU吞吐提升。

另两张materialization图沿用此前独立的runtime审查记录，本轮未改其数据、图或manifest。

## Word放置建议

4张`_word` QA forest源图9in宽、主要方法标签12pt，183mm放置的最小text约9.61pt；在180mm约9.45pt。它们适合Word独立页插图，仍需主任务检查图注是否跨页。

4张原尺寸forest是12.8in母图；在183mm缩放最小text约6.47pt，仅作完整可编辑源图，不优先用于Word正文。两个真实KV图在183mm的最小text分别约10.41pt和10.17pt。89mm单栏放置不在本轮认证范围内。

## 审查入口

- `scripts/audit_final_qa_kv_figures.py`：纯离线、无网络
- `figures_qa_kv/qa/final_qa_kv/final_audit_manifest.json`
- `figures_qa_kv/qa/final_qa_kv/source_raw.json`与`source_resolved.json`
- `figures_qa_kv/qa/final_qa_kv/*.glyph.json`、`*.collision.json`、`*.183mm.png`
- `figures_qa_kv/figure_manifest.json`：更新后的英文图注和图件清单

当前应标为“机械问答接口与真实KV验证已完成，语义问答准确率待有效评价”的阶段结果，不能声称所有benchmark问答能力已被证明或全面超越。
