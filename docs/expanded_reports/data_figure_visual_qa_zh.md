# 数据图最终审阅记录

本记录只涵盖本子任务的数据图，不替代Word文档的最终页面渲染，也不涵盖主任务新增的QA/KV图。

## 原有正式数据图：36张

`figures/figure_manifest.json`所列24张原尺寸单图与12张Word友好单图均逐张人工审阅了最终PNG。内容包含33方法检索forests、旧/新真实来源粒度、长短路由配对支持差、熵分箱诊断、模块容量曲线及quality–reuse关系。没有图内标题、(a)–(d)拼图、文字/箭头遮盖、legend压线、数字压住点/误差线或裁切；Word友好forests采用唯一英文短别名与12pt标签，完整方法映射另保存。

最终PDF自动检查36/36无collision FAIL或WARN，PDF字体检查36/36通过（源图最小字号≥11pt）；源代码preflight为17 PASS、4 WARN、0 FAIL。WARN分别为未导出TIFF（已有用户要求的PNG/PDF/SVG）、尺寸不是未指定期刊默认89/183mm、实际观察数据的bootstrap随机数语句、log轴guard的静态识别不足（代码有capacity>0硬断言，实际容量16/64/256）。这些均已审阅并保留说明，而非隐藏检查。

Word 180mm宽插入9in源图约使12pt标签变为9.4pt；33行forest高约239mm，宜独立页。PDF原图可读不自动等于带图注Word页面可读，Word页面图注跨页及实际字体大小应由主任务再验证。

## 正文词元预算新增图：2张

`figures_token_budget/figure_manifest.json`所列quality–reuse和capacity两张图均单独人工审阅了最终PNG。12pt主要标签、13pt轴标签，图例位于上方或右侧留白，源数据33方法完整保留。

最终PDF collision为2/2 FAIL=0、WARN=0，字体检查2/2通过；源代码preflight18 PASS、3 WARN、0 FAIL。WARN为未导出TIFF、9in初稿宽度不是期刊默认宽度、静态log positivity检测（实际代码有budget>0断言、容量1024/4096/16384）。SVG保留可编辑text；PNG600dpi；PDF内嵌可编辑字体。

38个数据图最终检查完成，但这不是对科学结论、专家适用性或期刊接受的保证。与数据有关的解释边界分别在主图推荐表、源数据与token-budget协议报告中保留：accepted Hit≠完整Recall；回放reuse≠GPU KV；冻结检索support retention恒1≠性能提升；不同请求顺序≠新增独立题目。
