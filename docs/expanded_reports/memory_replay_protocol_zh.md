# 统一来源模块复用协议

本轮对每个实际执行的检索方法使用同一个来源模块 LRU。检索结果先冻结，缓存只决定本次已请求的源条目中哪些可以从过去状态加载；当前查询的返回顺序、条目身份和证据文本均保留。每个 split 冷启动，禁止用 calibration/validation 预热 test。

缓存模块的键由机型、document_id、page_id、evidence_id 和精确源文本 SHA-256 构成。所有查询必须先完成对上一请求结束时缓存的查找，然后才更新 LRU。本次内重复条目仅请求一次模块；未来请求不得进入当前状态。revision 未在现有语料中独立提供，authority 未由缓存键验证，所以结果表示精确来源模块复用，不表示合法授权或真实维修适用性。

容量为 16、64、256 个 evidence modules。请求顺序为五个固定随机种子及一个仅按可见问题主题/机型排序的聚集场景，各方法完全相同。随机顺序是同一查询集的重复工作负载，不是新增独立样本。

官方 Qwen2.5 tokenizer 对每个原始 source text 计算词元量，tokenizer 文件 SHA 与配置写入 replay_manifest.json。词元加权复用率等于当前已请求且缓存命中的精确来源模块的词元量除以当前所有已知请求模块的词元量。它没有测量规范化 chat-template 公共前缀、层内 KV、预填充时长或 GPU FLOPs。

flat_identity_lru 仅保存来源模块。identity_lru_plus_routing_indices 保持相同内容模块容量，另各增加相同数量的 page/section 来源定位节点槽；其额外节点预算和实际 occupancy 单独输出。索引缓存只保存定位身份，不保存完整页面、摘要正文、图边或所有子 evidence。其命中表示已有路由定位机会，不代表页面下所有条目仍是可用支持；不得把 metadata 命中计入正文词元复用率。

原排名保留率为1是此 replay 的设计不变量，不能当检索精度、信息增益或QA质量。更高复用可来自重复/单一内容，未必更高 query-specific support。应同时展示独立检索 Hit@10、set recall、required-quote support 与复用率，不从复用率推断问答改善。

旧集使用214 held-out query IDs。新源文档派生集的 target primary pages 与旧 primary pages 分离；旧 accepted-equivalence union 可能涉及这些页面，不能声称全部 accepted-page/section 标注完全隔离。source overlap audit 应逐条随实验输出。

逐请求结果记录所有实际 method 与 canonical query 的覆盖；缺失执行会出现在 coverage 表，不能从统计中悄然删除。paired 比较对同一五个请求顺序计算完整 workload 比率差，报告均值、范围和种子标准差；范围是固定查询集下的顺序变动，不是自然用户总体置信区间。
