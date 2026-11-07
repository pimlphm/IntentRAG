# 数据图正文建议：8 张独立图

所有图件均为单个坐标区、英文标注、无图内标题；不将多个坐标区拼为(a)–(d)。以下推荐依照检索验证—机理边界—复用诊断的论证顺序，完整方法结果和负结果均保留在文件夹的其余单图与source data中。

| 图文件 stem | 中文名称 | 方法论与具体含义 |
|---|---|---|
| legacy_test_accepted_hit_forest_word | 旧测试集全部方法的可接受证据命中比较 | 去除gold cue、采用可见query路由后，在同214旧测试IDs/191问题文本cluster和Top-10预算下比较全部执行方法。体现旧高分在公平输入下应如何重测，而不把cue-assisted高分直接继承到新主张。CI由问题文本cluster bootstrap得出。 |
| fresh_test_accepted_hit_forest_word | 新来源问题集全部方法的可接受证据命中比较 | 在新源文档派生问题集统一比较全部方法。balanced fair是本文主检索方法；真实BM25及真实神经方法同样完整展示，包括高于本文主方法的结果。训练校准版本失败保留，不以结果排序当作删除依据。 |
| fresh_required_quote_support_forest_word | 新问题必要原文依据的完整支持比较 | 从“找到任意可接受片段”推进至“查询要求的每条原文依据有多少真正返回”。该比例由冻结source quote support groups作检索后评价，不把relatedness或一次Hit当作多证据充分性。 |
| fresh_granularity_paired_support_shift_word | 长短期层级与动态组合的配对支持变化 | 对同一fresh问题比较long-only、short-only、fixed multi、entropy、entropy+constraints与detail-only的差。每次bootstrap对两个方法使用相同cluster，保持配对关系。长期/概况层路线的负差体现粗层级不能替代局部支持；动态组合的零/负差说明熵路由并非必然增益。 |
| entropy_support_fresh_locked_test | 得分熵与实际证据命中的分箱诊断 | 使用balanced fair真实detail-score entropy，按固定0.1区间分箱，与事后accepted Hit关联。空箱留空、不补数据；点面积表示cluster数量，区间为bin内cluster bootstrap。它检验集中度与命中是否一致，不将熵当作正确率、安全置信度或互信息。 |
| figures_token_budget/fresh_reuse_capacity_source_tokens | 源正文词元容量与长短粒度状态复用 | 在1024/4096/16384相同源正文token容量、同五种请求顺序下比较声明的local/long-only/multi层路线及本文主方法。超长对象不缓存，缓存占用硬断言≤预算；误差线是顺序min–max而非总体CI。对象长度与请求局部性共同影响复用，不将同模块数误称等存储。 |
| figures_token_budget/fresh_quality_reuse_4096_source_tokens | 等正文词元预算下的检索支持—复用关系 | 横轴为4096源正文tokens的past-only LRU复用量，纵轴为独立检索accepted Hit。所有33方法保留。Profile route仍是高复用低支持；Causal graph旧高复用大幅降低，暴露模块数预算不公平。高复用不代表QA改善；预算不含管理元数据，更不是实测GPU KV bytes。 |
| fresh_test_accepted_support_granularity_word | 被可接受依据支撑的真实来源粒度组成 | 仅对事后accepted支持条目统计facet/paragraph/table/page/section/profile的真实来源词元量组成。parent-route命中不会把所有children自动变成support，也不会改变返回源条目的粒度。右列单位数呈现实有支持分母，避免条件组成被误读为总体高覆盖。 |

上表前5及最后1图位于`figures/`，新正文词元预算2图位于`figures_token_budget/`。旧/新Exact Hit、全部返回条目粒度、按所有方法family拆开的模块容量曲线、旧集熵诊断等全部保留为独立补充图。模块数量不等于正文词元量，不应把原16/64/256模块曲线当作严格token公平结果。family拆分仅为可读性、按方法机制定义，不按是否获胜筛选。

文字解释应同时给出两个边界：新问题的primary target pages与旧primary pages分离，不等于全部accepted-equivalence页面/章节完全隔离；层级在这里指source结构和状态生命周期，不是已验证的自然维修会话长期记忆。

Word友好版本使用9in宽与12pt方法标签，33条方法的forest约11.96in高；180mm插入约9.4pt标签与239mm图高，宜独立页排版并另行检查图注是否跨页。所有短英文别名在source_data/method_display_aliases.csv逐一映射至完整method；没有删减方法。保留原宽幅版本在图件包，图内不增加(a)/(b)标题。
