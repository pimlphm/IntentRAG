# 扩大问答实验的实际生成与评分协议

本脚本 `scripts/evaluate_expanded_qa.py` 读取 `data/canonical_questions.json`、`results/retrieval_runs.jsonl`，然后调用本地 Ollama `qwen2.5:3b` 实际生成。正式默认选择96个问题（legacy heldout 48 + fresh heldout 48），各类内按 Boeing/Cessna 分层、固定seed=20261014。`--queries 64` 对应32+32。问题按question cluster去重；source/evidence group数量另报，不能自动把96个问句称96个独立证据样本。

所有已有实际方法均保留。默认API参数 temperature=0、num_predict=128、num_ctx=4096。模型输入只有可见问题、飞机名称、actual Top-10物化原文及文档/页来源，使用[S1]等局部标签。没有gold answers、accepted IDs、required quotes、上游groundedness、benchmark类型或评分字段。每个方法相同system prompt、同样输出JSON协议。

正文预算为最多3000个Qwen2.5 tokenizer token，包含source headers。Tokenizer来自官方 `Qwen/Qwen2.5-3B-Instruct`，revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`；路径/内容SHA写入 `qa_protocol.json`。按原始字符串的字符prefix二分截断，保留真实原文substring，不通过decode改写证据。全部prompt实际token数由Ollama返回并另存。若指定目录没有tokenizer，脚本明确切换到2400字符ceiling并在协议标明fallback。

API原始请求、原始响应、完整prompt、实际物化sources、raw答案、status、done_reason、耗时和返回的token计数均保存。格式错误、拒答、HTTP异常、模型异常、输出达到length限制都保留。没有citation repair、无生成后的答案替换、没有advantage-only方法筛选。

完全相同的模型/参数/system/prompt payload可以复用已经保存的原始响应，其新方法行标记为`duplicate_alias`，并记录`call_origin_id`；不能计为新增LLM调用。断点只跳过完全相同question/method/request SHA。旧responses中异常结果仍可追溯。

评分在原始响应落盘后读取qrels，只给出以下proxy：

- JSON是否有效；声明citation是否存在于真实materialized context。
- 模型声明的quote是否为其指定source的实际原文substring；quote source是否也被citation列声明。
- 有效引用对应的原始evidence是否落在accepted set（这是post-hoc目标匹配proxy）。
- 参考答案数值字符串覆盖、required quote的表面覆盖、character-bigram F1和词/中文字符ROUGE-L。

这些机械检查无法判断模型是否正确理解维修机理，也不能验证作用范围、动作授权和语义完整性。没有独立语义judge，因此结果不能称维修QA准确率或机理认证。固定JSON格式本身也会影响输出；该影响在各方法间相同。

Ollama `prompt_eval_cached_count` 若存在按原值保存，缺失写null。问答运行没有控制cold/warm请求与并发缓存历史；这些计数仅为raw telemetry，不能代替受控KV cache方法实验。

正式入口（retrieval和fresh准备完毕后）：

```powershell
python scripts/evaluate_expanded_qa.py --queries 96
```

Smoke入口：

```powershell
python scripts/evaluate_expanded_qa.py --queries 1 --question-ids QV21E_0000 --methods "BM25/word" "TFIDF-char" --allow-missing-fresh --output results/qa_smoke_token
```

已完成token预算smoke：两个真实调用均返回可解析JSON；actual source context为3000/2195 tokens，完整prompt为3208/2403 tokens。两条输出均未提供机械可验证的quote支持，未修复此结果。第二次同参数执行新增调用数为0，验证了断点。此smoke仅验证API、落盘和评分链路，不属于正式方法性能结论。
