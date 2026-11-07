"""Build the experiment report from completed measurements with bundled python-docx."""
from pathlib import Path
import json
import math
import pandas as pd
from docx import Document
from docx.shared import Cm, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

EXP=Path(__file__).resolve().parents[1]
OUT=EXP/'deliverables'
OLD='v21_locked_blind_test'
FRESH='fresh_locked_test'

def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def pct(x):return '未定义' if pd.isna(x) else f'{100*x:.1f}%'
def num(x):return '未定义' if pd.isna(x) else f'{x:.3f}'
def method_name(m):
    return {'BM25/word':'Word TF-IDF（原标签BM25/word）','Calibrated lexical control':'Calibrated three-view control',
            'FD-GRAG-RC-V21-final':'IntentRAG visible router'}.get(m,m)

def paragraph(doc,text,style=None):
    p=doc.add_paragraph(text,style)
    p.paragraph_format.space_after=Pt(7)
    return p

def table(doc,headers,rows,widths):
    t=doc.add_table(rows=1,cols=len(headers));t.alignment=WD_TABLE_ALIGNMENT.CENTER;t.autofit=False
    for c,w,h in zip(t.rows[0].cells,widths,headers):c.width=Cm(w);c.text=h
    repeat=OxmlElement('w:tblHeader');t.rows[0]._tr.get_or_add_trPr().append(repeat)
    for row in rows:
        for c,w,value in zip(t.add_row().cells,widths,row):c.width=Cm(w);c.text=str(value)
    for i,row in enumerate(t.rows):
        for j,c in enumerate(row.cells):
            c.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
            tcpr=c._tc.get_or_add_tcPr();b=OxmlElement('w:tcBorders')
            for edge in ['top','left','bottom','right']:
                el=OxmlElement('w:'+edge);el.set(qn('w:val'),'single');el.set(qn('w:sz'),'4');el.set(qn('w:color'),'D9D9D9');b.append(el)
            tcpr.append(b)
            margins=OxmlElement('w:tcMar')
            for edge in ['top','left','bottom','right']:
                el=OxmlElement('w:'+edge);el.set(qn('w:w'),'90');el.set(qn('w:type'),'dxa');margins.append(el)
            tcpr.append(margins)
            shade=OxmlElement('w:shd');shade.set(qn('w:fill'),'E8EFF5' if i==0 else ('F7F8FA' if i%2==0 else 'FFFFFF'));tcpr.append(shade)
            for p in c.paragraphs:
                p.alignment=WD_ALIGN_PARAGRAPH.LEFT if j==0 else WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.space_after=Pt(1);p.paragraph_format.line_spacing=1.05
                for r in p.runs:r.font.size=Pt(9.5);r.bold=(i==0);r.font.color.rgb=RGBColor(0,0,0)
    paragraph(doc,'')
    return t

EXPLANATIONS={
 'legacy_test_accepted_hit_forest':('旧测试全部方法的证据命中比较','此图重新测量旧问题，不继承由目标证据提示或锁定track获得的高分。点和区间按191个去重问题文本簇计算；表格另列214条记录均值，两者不是相同加权口径。方法的差距必须在共同输入、Top-10和源证据身份下解释。'),
 'fresh_test_accepted_hit_forest':('新来源问题全部方法的证据命中比较','此图检验方法能否在新增来源目标上保留检索能力。多粒度加重排优于同重排预算的稠密路线，但BM25和其他强方法并未因其结构简单而被排除。命中一条可接受依据还不能证明多依据问题已得到完整支持。'),
 'fresh_required_quote_support_forest':('新问题必要原文依据的覆盖比较','评价单元从任意命中改为问题所要求的原文支持组。对两来源问题，覆盖一组得到二分之一，而非完整支持。该图把局部相关性和证据充分性区分开来；目标引用仅在检索后用于评分。'),
 'fresh_granularity_paired_support_shift':('长短期层级与动态组合的配对支持变化','同一问题的Detail-only作为共同参照，配对差减少问题难度造成的混淆。长期概况不能替代原文叶子支持；固定和熵动态多粒度在此实现下也不自动优于细节检索。约束组装的收益应按实际正负差报告，不能由方法名称推定。'),
 'entropy_support_fresh_locked_test':('候选得分熵与实际支持的分箱诊断','得分熵描述候选排序质量的集中程度，命中则来自事后来源标签。两者可以不一致：错误候选也可能有很低的得分熵。空分箱不补点。此图检验熵路由所依据的经验联系，不是在估计真实知识熵、互信息或安全置信度。'),
 'reuse_capacity_mechanistic_fresh_locked_test':('精确来源模块容量与历史复用机会','在同16、64、256模块预算下，缓存先读取过去、后写入本次请求。五种随机顺序的范围反映工作负载敏感性，不是新增样本或总体置信区间。概况路线可能因反复请求相同大块文本而获得高复用，仍需联合覆盖率解释。'),
 'quality_reuse_fresh_locked_test':('检索支持与历史模块复用的双轴关系','横轴是精确来源模块的词元量复用，纵轴是独立测量的可接受证据命中。高复用但低支持的点说明频繁出现的信息未必是问题需要的信息。此图不是可采纳性Pareto证明，也没有把模块命中直接换算为GPU KV命中。'),
 'fresh_test_accepted_support_granularity':('实际支持条目的来源粒度组成','仅在事后被判为可接受支持的条目中统计真实原文粒度。因此这是命中后的条件组成，不是命中率。父节点用于定位并不使其全部子证据自动适用；必须与返回和命中条目数及覆盖图联合读取。'),
 'fresh_answer_quote_support':('新问题真实回答的原文引用支持','只检查模型实际给出的quote是否是实际输入来源的子串。它不继承上游检索分数，也不把确定性替换文本当成模型回答。高quote支持仍可能伴随答非所问、条件遗漏或格式失败。'),
 'fresh_blinded_answer_audit':('新问题回答的盲评正确性与支持代理','本地7B评估者仅看到问题、原始目标依据与真实答案，不看到方法名或检索分数。正确且有支持的结果必须同时满足回答问题、来源支持、无矛盾和无无依据行动。该评估仍是同系列模型的机器代理，需要外部专家独立复核。'),
 'actual_kv_reuse_by_prefix_layout':('三种前缀布局的真实GPU KV复用分布','每个点对应一个方法和一个固定查询对。改变的是问题、规范化正文及层级元数据在前缀中的位置；实际复用由引擎cache_n计数。稳定原文可促进连续前缀复用，而查询变化的层级节点可能提前打断它。节点身份重复不等于连续计算历史相同。'),
 'native_prefix_overlap_vs_actual_kv':('原生前缀重合与真实KV命中的对应','横轴是种子到目标的原生词元前缀重合，纵轴是GPU引擎实际沿用的输入KV。禁止全局恢复目标自身缓存，并核对已沿用与新计算词元之和。这一验证确认计算计数的含义，没有确认维修授权或文档修订的有效性。'),
 'fresh_quality_reuse_4096_source_tokens':('严格词元容量下的支持与复用关系','所有方法使用相同4,096个完整源正文词元容量；超过容量的条目不进入缓存，但仍保留原检索输出。此图消除了仅控制模块数量时不同正文长度的预算混淆。高复用低支持的概况路线仍存在，但大块图控制原有的高复用明显下降。源正文词元公平仍不等于RAM或KV字节公平。'),
 'fresh_reuse_capacity_source_tokens':('完整源正文词元预算与复用敏感性','同样的1,024、4,096、16,384词元预算作用于各源层级路线。点为五种历史请求顺序的均值，误差线为观测范围。它检验可存储成本是否改变复用结论，区别于原模块计数预算；超长条目准入率与正文复用分别记录。'),
 'actual_kv_vs_materialized_reference_support':('实际KV复用与物化参考支持的分离','每个点是一个问题对在33种方法上的均值，共六对问题和三种布局，不是594个独立样本。横轴是真实缓存比例，纵轴是实际输入中的参考长文本可见代理。更靠右可能同时更缺少局部参考依据。该图检验预算物化后的信息保留，不是答案正确率或合法可采纳性。'),
 'paired_cache_gain_vs_materialized_support_loss':('配对缓存收益与参考支持损失','先对同方法同问题对计算布局变化，再在每对内求均值。右上方表示缓存提高但参考片段丢失；祖先索引对规范正文的支持差恒为零，因为198对源区域完全相同，其缓存差来自前置元数据及输入长度。两种来源、两组比较及零值全部保留。'),
}

def figure(doc,root,stem,index):
    image=root/(stem+'_word.png') if (root/(stem+'_word.png')).exists() else root/(stem+'.png')
    if not image.exists():return False
    from PIL import Image
    w,h=Image.open(image).size
    name,meaning=EXPLANATIONS.get(stem,(stem,'本图保留实际方法和观测值；应依据随图Source Data的指标、样本单位及不确定性定义解读。'))
    doc.add_page_break()
    p=paragraph(doc,f'图{index}  {name}');p.paragraph_format.keep_with_next=True
    p.runs[0].bold=True
    # Independent inline images, never baked multi-panel arrangements.
    width=min(17.0,21.7*w/h)
    p=doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.paragraph_format.space_after=Pt(4)
    r=p.add_run();r.add_picture(str(image),width=Cm(width))
    drawing=r._r.xpath('.//wp:docPr')
    if drawing:drawing[0].set('descr',name)
    p=paragraph(doc,meaning);p.paragraph_format.line_spacing=1.08
    for r in p.runs:r.font.size=Pt(10)
    return True

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    manifest=load(EXP/'reports/expanded_execution_manifest.json')
    assert manifest['n_missing']==0 and manifest['n_executed_rows']==manifest['n_queries']*manifest['n_methods']
    methods=manifest['methods'];nmethods=len(methods)
    r=pd.read_csv(EXP/'results/all_retrieval_query_results.csv')
    s=pd.read_csv(EXP/'results/all_retrieval_summary.csv').set_index(['split','method'])
    mem=pd.read_csv(EXP/'results/memory_replay/replay_order_summary.csv')
    token_mem=pd.read_csv(EXP/'results/memory_replay_token_budget/replay_order_summary.csv')
    qa_dir=EXP/'results/qa_expanded_main';qaexec=load(qa_dir/'qa_execution_summary.json')
    assert qaexec['rows']==64*nmethods, 'Formal QA incomplete'
    qa=pd.read_csv(qa_dir/'qa_summary_by_origin_method.csv')
    jdir=qa_dir/'blind_semantic_audit';judge=None
    if (jdir/'audit_manifest.json').exists():
        judge_exec=load(jdir/'audit_manifest.json')
        if judge_exec.get('n_rows')==qaexec['rows']:
            judge=pd.read_csv(jdir/'audit_summary_by_method.csv')
    kvdir=EXP/'results/real_kv';kvexec=load(kvdir/'kv_pairs_report.json')
    assert kvexec['rows']==nmethods*6*3, 'Formal GPU KV incomplete'
    kv=pd.read_csv(kvdir/'layout_summary.csv')
    controls=EXP/'results/qa_interface_controls'
    control_summary=pd.read_csv(controls/'blind_semantic_audit/audit_summary_by_method.csv') if (controls/'blind_semantic_audit/audit_summary_by_method.csv').exists() else None
    d=Document();sec=d.sections[0];sec.page_width=Cm(21);sec.page_height=Cm(29.7)
    sec.top_margin=sec.bottom_margin=Cm(1.7);sec.left_margin=sec.right_margin=Cm(1.8)
    for name in ['Normal','Title','Subtitle','Heading 1','Heading 2','Heading 3','Caption']:
        st=d.styles[name];st.font.name='Arial';st.font.color.rgb=RGBColor(0,0,0)
        st._element.get_or_add_rPr().get_or_add_rFonts().set(qn('w:eastAsia'),'宋体')
    d.styles['Normal'].font.size=Pt(11);d.styles['Normal'].paragraph_format.line_spacing=1.15
    d.styles['Title'].font.size=Pt(21)
    d.styles['Heading 1'].font.size=Pt(16);d.styles['Heading 2'].font.size=Pt(13)
    d.core_properties.title='IntentRAG扩展检索问答与状态复用实验'
    d.core_properties.author='Research team'
    p=sec.footer.paragraphs[0];p.alignment=WD_ALIGN_PARAGRAPH.CENTER
    field=OxmlElement('w:fldSimple');field.set(qn('w:instr'),'PAGE');p._p.append(field)
    d.add_paragraph('IntentRAG扩展检索问答与状态复用实验',style='Title')
    paragraph(d,'共同预算下的基线验证  多粒度支持分析  真实GPU前缀测量',style='Subtitle')
    paragraph(d,'2026年10月14日')
    balanced=s.loc[(FRESH,'IntentRAG balanced fair')]
    bm25=s.loc[(FRESH,'True BM25')]
    neural=s.loc[(FRESH,'IntentRAG multi + mMARCO CE@50')]
    dense=s.loc[(FRESH,'Neural dense + mMARCO CE@50')]
    paragraph(d,f'本轮在31,766个真实项目证据对象上执行{nmethods}种检索方法，覆盖690条查询记录、624个不同问题文本簇，得到{manifest["n_executed_rows"]:,}条完整检索记录。新增214条来源派生测试问题中，均衡IntentRAG的可接受证据Hit@10为{pct(balanced.accepted_hit_at_10)}，完整必要引用支持率为{pct(balanced.all_quote_support)}。多粒度加重排相对同预算稠密加重排提高了命中与支持，但BM25仍具有更高的新集命中率。因此，结果支持具体条件下的优势，不支持全面远超所有基线的结论。')
    paragraph(d,f'问答采用冻结的64题和统一3B生成器，共{qaexec["rows"]:,}条问题方法执行记录，API尝试{qaexec["actual_calls"]:,}次，完全相同请求的别名复用{qaexec["duplicate_aliases"]:,}条，其中1条传输失败记录保留。另完成256条接口与政策对照。历史模块回放覆盖全部方法；GPU试验实际执行{kvexec["rows"]}个方法查询对前缀布局组合。以下分别报告检索、答案、粒度和计算状态，避免将其中一项高分解释为其他能力。完整语义准确率尚待验证。')
    d.add_heading('一 数据规模与评价边界',level=1)
    table(d,['集合','记录数','不同问题簇','用途'],[
        ['旧校准集',143,r[r.split.eq('v21_calibration')].cluster_id.nunique(),'仅训练校准排序'],
        ['旧验证集',119,r[r.split.eq('v21_validation')].cluster_id.nunique(),'与测试分开'],
        ['旧锁定测试集',214,191,'重测原文基线'],
        ['新来源测试集',214,214,'142单来源和72双来源问题'],
        ['统一真实问答子集',64,64,'32旧问题及32新问题']], [5.0,2.0,2.9,7.3])
    paragraph(d,'新增问题由本地模型从固定抽样原文生成，保留全部尝试，经原文子串核对和另一模型支持筛查后进入测试。它们不是自然维修用户问题，也没有专家认证。新集的主要目标页与旧出题主要目标页分离，但9条新问题的目标证据曾出现在旧可接受等价集合中，不能称为全来源全标注隔离。新集仅有2条Boeing问题、212条Cessna问题，其总体结果主要反映Cessna，不能据此重新证明两机型的普遍差异。')
    paragraph(d,'所有检索方法只看问题、机型和文档特征。目标证据、track、split和必需支持组仅在输出冻结后用于评价。旧代码的二值R@10在本轮更名为Hit@10；同时单独计算真实集合召回。旧约99%的加权、提示辅助检索值不能继承为本轮问答准确率。旧214记录中存在重复问题，因此表中并列记录均值和去重问题簇均值，图的区间围绕后者计算。')
    paragraph(d,'旧校准、验证、测试分别含121、98、191个不同问题文本簇，彼此无文本簇重合；新集214个问题簇与这三个旧集合也不重合。该检查说明问题文本划分独立，并不自动说明所有来源页面、章节或既有等价证据标注均独立。')
    d.add_heading('二 共同预算与方法身份',level=1)
    paragraph(d,'原文的15种项目控制实现全部重跑。其中原BM25/word实际为词级TF-IDF余弦；本轮另实现真正Okapi BM25。GraphRAG、Agentic及GaRAGe-style标签对应项目内控制路线，不是这些论文官方系统的完整复现。另加入细节、页面、章节、概况、固定长短期、固定多粒度、熵动态路由、约束组装及校准排序对照。失败的校准多粒度版本同样保留。')
    paragraph(d,'五种神经路线使用真实多语MiniLM和mMARCO交叉编码器权重，不以SVD或词法分数冒充神经模型。稠密加重排分别采用30和50候选；多粒度加重排与50候选稠密路线使用相同重排预算。全部方法统一返回至多10个不同原文证据对象。预训练神经模型在本地DirectML GPU上执行，数值与排名校验通过；部分辅助节点在CPU运行。')
    paragraph(d,'问答使用本地Qwen2.5 3B，温度0，固定随机种子，源正文及其来源头统一限制在真实Qwen词元计数的3,000以内，生成上限128词元。保存实际物化来源、截断、原始响应、格式失败和计时，不进行引用修补或答案替换。128词元限制可能截断JSON，格式成功、原文引用支持和语义正确性分别统计，不将HTTP执行成功当成回答成功。')
    paragraph(d,'Boeing问题主要为中文，Cessna主要为英文。统一生成词元预算限制计算量，但65个中文字符和35个英文单词的词元消耗并不相同；飞机之间的JSON截断差异不能直接解释为来源分布或方法机理差异。原始生成词元量、停止原因及分机型结果随表保存。')
    d.add_heading('三 长短期证据层级与信息机制',level=1)
    paragraph(d,'本轮将章节与部件概况视图视为持久定位层，把查询实际返回的原文叶子及必要条件视为局部支持层。长期的含义是可在后续任务中保留其身份与定位关系，不是摘要一经生成就具有支持所有下游问题的资格。粗层路由最终仍需返回可引用的来源条目；页面或章节身份命中不使其所有子条目自动进入有效证据。')
    paragraph(d,'多粒度访问改变候选得分在不同层级上的分布，也改变进入证据状态的原文来源粒度。候选得分熵衡量排序分布是否集中，用于资源分配；它不等于知识的不确定性或答案正确率。错误的概况节点也可能得到集中的分数。配对支持变化和熵分箱命中图分别检验层级选择的实际效果与熵代理的经验依据。当前动态组合的效果存在退化，不能用较低熵推定信息增益更大。')
    paragraph(d,'历史缓存还引入访问频率偏置：反复请求的粗概况更容易留在有限状态中，使源正文词元量复用率升高；查询需要的低频条款可能被驱逐。因而高复用同时可能意味着信息集中于少量重复节点，而不是需求覆盖改善。双轴质量复用图将这两项可观测量分开。本轮没有估计严格互信息、证明最优信息瓶颈，或验证MemGAS的GMM关联及PPR跨会话更新；它验证的是明确实现的源层级路线和只看过去的缓存工作负载。')
    d.add_heading('四 全部方法的检索结果',level=1)
    rows=[]
    for m in methods:
        old=s.loc[(OLD,m)];new=s.loc[(FRESH,m)]
        rows.append([method_name(m),pct(old.accepted_hit_at_10),pct(old.accepted_hit_at_10_cluster_mean),pct(new.accepted_hit_at_10),pct(new.all_quote_support)])
    table(d,['方法','旧记录命中','旧去重命中','新集命中','新集完整引用'],rows,[7.7,2.15,2.35,2.2,2.8])
    paragraph(d,f'均衡IntentRAG在新集保留{pct(balanced.accepted_hit_at_10)}命中；BM25达到{pct(bm25.accepted_hit_at_10)}，完整必要引用支持为{pct(bm25.all_quote_support)}。多粒度加重排的命中为{pct(neural.accepted_hit_at_10)}，对照稠密CE50为{pct(dense.accepted_hit_at_10)}；完整支持分别为{pct(neural.all_quote_support)}和{pct(dense.all_quote_support)}。这一对照证明在该实现与数据上，多视图候选组织能补充纯稠密候选的遗漏，但它没有超过所有词法或融合方法。')
    paragraph(d,'新集Long-term fixed命中约24.8%，Profile-resolved约1.9%，显著弱于Detail-only与Short-term fixed。这支持高度压缩或概况层不能替代查询局部原文的观点。固定、熵动态和约束动态多粒度的配对差还显示，加入层级不必然改善支持。校准模型仅用143条旧校准记录训练后，在新集明显退化，说明训练分布与查询需求迁移仍需处理。')
    paragraph(d,'检索表按固定测试记录完整保留。配对比较使用相同问题簇，报告bootstrap区间、胜负和Holm校正；另外输出来源组聚合的不确定性。新增双来源问题的完整支持率与单来源问题分开汇总，不能用一次命中掩盖另一必需依据缺失。')
    d.add_heading('五 真实问答与盲评审计',level=1)
    merged=qa.merge(judge,on=['origin','method'],suffixes=('_mechanical','_judge')) if judge is not None else qa
    rows=[]
    for m in methods:
        fresh=merged[(merged.method==m)&(merged.origin=='fresh')].iloc[0]
        old=merged[(merged.method==m)&(merged.origin=='legacy')].iloc[0]
        rows.append([method_name(m),pct(fresh.structured_parse_ok),pct(fresh.quote_supported_fraction),
                     pct(fresh.answer_correct_supported_judge_proxy) if judge is not None else pct(old.structured_parse_ok),
                     pct(old.answer_correct_supported_judge_proxy) if judge is not None else pct(old.quote_supported_fraction)])
    headers=['方法','新集JSON解析','新集quote检查','新集盲评代理','旧集盲评代理'] if judge is not None else ['方法','新集JSON解析','新集quote检查','旧集JSON解析','旧集quote检查']
    table(d,headers,rows,[7.7,2.15,2.35,2.2,2.8])
    if judge is None:
        paragraph(d,'本轮已完成实际回答执行，但尚未形成有效语义盲评分数。因此表格只报告格式与实际原文引用检查，不报告完整问答准确率。失败的后续评估尝试未作为0分、正确回答或独立样本纳入统计；语义与行动支持仍需验证。')
    paragraph(d,'JSON解析列仅要求输出能够解析为对象且answer为字符串，不是完整输出模式的强制验证。逐回答quote检查在未给出可识别quote时记0；2,112条记录中只有11条含解析器识别的quote。提示仅要求至多一条quote，并未强制非空，因此低平均值主要受quote给出与识别影响，不能推定所有回答无证据或错误。已给quote的条件精度及其很小分母另列在原始汇总中。')
    paragraph(d,'后续语义审计协议采用本地7B盲评，同时要求回答具体问题、主要事实得到原始参照或实际输入来源支持、不与相关参照矛盾、没有无依据行动；额外但确由实际上下文支持的事实不因未出现在单一目标参照中而自动判错。方法名和上游分数不输入评估者。对于截断JSON，若完整answer字符串已存在，仅提取它作内容审计并记录提取方式，不补全实际输出；否则评估原始输出。引用检查仍严格依赖原始可解析结构。评估者与生成器属于同系列，且新题筛查也用同一7B，即使取得评分也仍是机器代理，不是独立专家认证的维修问答精度。')
    if control_summary is not None:
        rows=[]
        for (origin,m),g in control_summary.groupby(['origin','method']):
            rows.append([origin,method_name(m),pct(g.answer_correct_supported_judge_proxy.iloc[0])])
        table(d,['问题来源','接口对照','正确且有支持的盲评代理'],rows,[3.0,9.0,5.2])
    elif (controls/'qa_summary_by_origin_method.csv').exists():
        cs=pd.read_csv(controls/'qa_summary_by_origin_method.csv')
        table(d,['来源','接口或政策','JSON解析','quote检查'],[[r.origin,r.method,pct(r.structured_parse_ok),pct(r.quote_supported_fraction)] for r in cs.itertuples()],[2.0,9.0,3.1,3.1])
    paragraph(d,'接口对照在相同64题上比较原排名证据、规范化证据状态、无来源证据政策及参数知识作答，共256条记录。规范化排序可能在相同词元上限下改变实际保留内容，故同时保存来源顺序和截断。新32题的必要引用实际显示覆盖从原排名的96.9%降至规范化的46.9%，相邻前缀机会却从约23.8词元升至116.1词元。该前缀值是tokenizer代理，不是GPU实测；引用显示是输入支持代理，不是问答准确率。只有内容保持一致的对照才可将质量差异孤立归于顺序。无来源且要求仅用来源的模型应拒答，其低任务完成率不能证明参数知识缺失。允许参数知识的对照明确采用不同政策，结果仅针对本地3B，不能推出更大模型或特定RLM架构的普遍能力。')
    d.add_heading('查询局部片段保护的探索性接口',level=2)
    paragraph(d,'一个直接的修正是先按查询排名在预算内形成实际保留的原文片段，再仅对这些不可变片段进行规范排序。相反，先对完整Top-10规范排序再截断，会使新的顺序重新决定哪些信息被删除。这两个操作次序并不等价。修正接口逐题核对来源ID、实际文本哈希与溯源多重集完全保持；如排列使词元数超上限，整包回退原排序，而不是根据目标标签裁剪。')
    paragraph(d,'在同64题的离线探索中，保护接口的新集前缀机会从23.81增至31.78词元，必要引用保留仍为96.875%；旧集从286.41增至372.31词元，可接受ID显示保留仍为90.625%。64题均保持片段多重集，3,000词元预算通过，回退为0。温和的约1.3倍前缀机会与朴素规范排序更高但有损的机会形成取舍。这里的无损保留是构造不变量，尚未验证真实KV及新生成回答；原型是在本轮结果后提出，不能当作已锁定测试上确认的新增问答贡献。')
    d.add_heading('六 在线证据模块复用',level=1)
    subset=mem[(mem.split==FRESH)&(mem.capacity==64)&(mem.order=='random')&(mem.policy=='flat_identity_lru')]
    rows=[]
    for m,g in subset.groupby('method'):
        rows.append([method_name(m),pct(g.token_weighted_reused_fraction.mean()),
                     f'{pct(g.token_weighted_reused_fraction.min())}至{pct(g.token_weighted_reused_fraction.max())}'])
    table(d,['方法','64模块词元量复用','五顺序观测范围'],rows,[9.2,3.4,4.6])
    paragraph(d,'全部33方法使用相同精确来源模块LRU、16/64/256内容模块容量、五个随机顺序与一个可见主题顺序；每个数据划分独立冷启动，查找发生在写入之前，共819,720条逐请求记录。来源键包含机型、文档、页面、证据身份和源文本哈希。额外页面与章节定位索引的预算单列，其命中不计正文复用或叶子支持。')
    paragraph(d,'复用率由本次所需且已在历史缓存中的源正文词元量除以本次全部所需源正文词元量计算。它保留原检索输出，输出保留率为1是设计不变量，而非新的准确率贡献。Profile-resolved的高复用和低检索支持共同揭示，过度概况化会把有限缓存集中到重复信息。高支持路线的低复用则表明，扩大来源覆盖需要付出更多新证据读取代价。')
    paragraph(d,'模块数量相同不等于存储正文量相同。因此另执行409,860条严格词元预算回放：容量为1,024、4,096、16,384个完整源正文词元，超出容量的单条来源不准入缓存，逐次核对总占用不超过上限。它控制源正文存储成本，不包含键、索引或管理开销，不等于严格RAM及KV字节预算。旧64模块结果只作补充诊断。')
    rows=[]
    tpart=token_mem[(token_mem.split==FRESH)&(token_mem.capacity_source_tokens==4096)&(token_mem.order=='random')]
    for m,g in tpart.groupby('method'):
        rows.append([method_name(m),pct(g.token_weighted_reused_fraction.mean()),pct(g.oversize_token_mass_fraction.mean())])
    table(d,['方法','4,096正文词元复用','超长未准入的请求词元量'],rows,[9.2,3.4,4.6])
    paragraph(d,'严格4,096词元预算下，Profile-resolved仍约78.2%复用而新集命中仅1.9%；Causal-graph控制的复用则从64模块条件的约70.0%降至约1.2%，其大部分请求词元来自超长条目。均衡IntentRAG约0.44%、BM25约0.13%，说明高覆盖新增问题的实际历史正文重合很少。相对倍数不能掩盖这类很小的绝对量。旧62.3%潜在重复比例、新集有限容量回放和六个固定查询对的GPU前缀结果采用不同工作负载与定义，不能当作同一个指标的前后改善。')
    d.add_heading('七 真实GPU前缀复用与可采纳性边界',level=1)
    rows=[]
    for row in kv.itertuples():
        rows.append([row.namespace,int(row.n_profiles),pct(row.mean_cache_fraction),f'{row.mean_cached_tokens:.1f}',pct(row.exact_output_identical_fraction)])
    table(d,['前缀布局','组合数','平均真实复用','平均KV词元','冷暖文本相同'],rows,[6.3,2.2,3.0,2.6,3.1])
    paragraph(d,'GPU试验使用本地0.5B GGUF模型和独占请求slot，对所有33种检索路线及六个相同可见主题查询对比较三种布局。每个组合执行冷目标、冷种子、暖目标；禁止全局RAM缓存恢复目标自身，并排除启动热身。实际沿用量读取timings.cache_n，新计算量读取timings.prompt_n，二者之和须等于原生目标输入词元数。顶层tokens_cached仅表示最终槽占用，不作命中率。')
    paragraph(d,'GPU试验中的规范化正文按证据ID排序，源正文上限为1,600词元；3B问答接口对照按文档、页面、章节、角色和ID排序，上限为3,000词元。两者分别冻结并公平比较，但不是同一排序及物化规则，不能将其两组数值直接合并为同一证据状态的质量效率曲线。')
    paragraph(d,'计算复用依赖完整的计算历史，包括模型和词元化配置、位置、前缀文字及顺序。证据复用则需要核对来源、适用范围与任务需求。因此，来源集合重合不保证连续前缀相同；将查询变化的层级节点放在稳定正文之前还可能提前截断共享前缀。规范化状态和长期定位层的作用应由真实计数判断，不能由架构图推定。')
    paragraph(d,'进一步读取全部594个真实运行输入，核对实际发出的来源及预算截断。三个布局的上游Top-10相同，但规范排序将component-profile正文份额从9.8%提高至19.7%，实际参考长文本可见代理从72.1%降至59.3%，已发出可接受ID命中代理从73.2%降至58.6%。规范正文与祖先索引布局的198对源区域完全相同，后者较低的缓存比例不对应支持内容改善。这说明缓存优化可能把状态分布推向稳定粗背景，并让查询局部条款落在预算外。长文本与ID可见均只是物化支持代理，不是问答正确或授权有效性。')
    paragraph(d,'signed KV的贡献应表述为带适用签名的状态接口：先核对证据允许保留的条件，再在同一已核对等价类内部组织相同前缀及运行配置。安全目标是允许的计算复用关系包含于已验证的证据兼容关系，而不是强迫所有可采纳证据产生相同KV。标准KV对相同词元前缀的复用应在数值上等价重新预填充；跨任务证据不适用首先是输入选择和支持范围的问题，不能直接归咎于缓存计算。现有语料不能完整验证修订、生效范围、构型与行动授权；未知字段不得被哈希签名自动变成已验证。旧purity达到1的分组结果是按字段定义的兼容性，不是维护安全或GPU实测。')
    paragraph(d,'此次缓存与3B问答曾共享GPU负载，配对计时只作为该运行下的描述，不作为隔离吞吐或普遍加速比例。冷暖贪婪文本有变化时按原样保留，不假定命中缓存就保持答案字面不变。0.5B缓存执行器验证的是前缀计算，不能替代3B问答质量或更大模型的能力验证。')
    d.add_heading('八 目前可成立的论文结论',level=1)
    paragraph(d,'当前最明确的经验结论是：粗概况与局部原文承担不同作用，前者可形成集中且易复用的定位信息，但不能替代当前问题的必要支持；多粒度候选组织在同重排预算下改善了纯稠密检索，仍需面对强词法基线及分布迁移；模块身份复用与连续GPU前缀复用由不同条件控制，新增结构有时提高定位能力，有时打断缓存前缀。')
    paragraph(d,'因此，三个研究贡献可围绕任务需求下的证据保留与补全、多层来源表示的选择机制，以及证据兼容关系与实际计算前缀的交集来组织。实测结果只支持其中已测量的部分。尚不能写入确定性结论的内容包括：99%自然用户问答精度、全面远超官方MemGAS或GraphRAG、低熵保证正确、签名保证合法授权、模型规模无法替代证据，以及所有多粒度记忆都会提高复用。')
    paragraph(d,'后续验证应引入新的专家编写任务、足够的独立Boeing来源与实际修订构型标签，分别考察局部必要支持和跨任务适用性。若改进方法是在观察本轮测试后设计，需要新锁定集，不得回头调低基线或重选问题使本轮出现预设优势。')
    d.add_heading('九 独立图件与方法解释',level=1)
    paragraph(d,'以下图件由各自源结果直接绘制，图内无标题、不合并(a)(b)面板。长幅全部方法图以独立页插入；中文说明解释其方法、机理与边界，图内文字保持英文。完整独立图及Source Data、绘图代码和质量审计随实验目录交付。')
    figroots=[EXP/'figures',EXP/'figures_qa_kv',EXP/'figures_token_budget']
    idx=0
    selected=list(EXPLANATIONS)
    for stem in selected:
        root=next((p for p in figroots if (p/(stem+'.png')).exists()),None)
        if root:
            idx+=1;figure(d,root,stem,idx)
    d.add_page_break();d.add_heading('十 实验文件索引',level=1)
    table(d,['目录或文件','内容'],[
        ['data','冻结问题 来源筛查 方法注册与共同KV查询对'],
        ['results/all_retrieval_query_results.csv','33方法全部690记录及实际返回来源'],
        ['results/paired_method_comparisons.csv','同问题配对差 区间 胜负与校正'],
        ['results/qa_expanded_main','原始问答 来源物化及独立答案审计'],
        ['results/qa_interface_controls','固定64题的参数知识及证据接口对照'],
        ['results/qa_protected_canonical_prototype','无新增模型调用的片段保护原型与完整性验证'],
        ['results/memory_replay','全部在线回放 顺序 容量与来源身份复用'],
        ['results/memory_replay_token_budget','严格完整源正文词元容量及超长准入对照'],
        ['results/real_kv','真实GPU请求 响应 slot快照和计数'],
        ['figures 与 figures_qa_kv','独立英文图 Source Data与图件QA'],
        ['scripts','可重跑的实验 分析 绘图与报告代码']], [7.1,10.1])
    paragraph(d,'原始语料与既有模型权重读取本地项目，不在交付包中重复复制。最终汇总使用当前目录的冻结执行日志。core28阶段快照和smoke仅保留用于溯源，不与完整33方法正式结果混为一组。')
    path=OUT/'IntentRAG_扩展实验与机理验证_20261014.docx';d.save(path)
    (OUT/'report_manifest.json').write_text(json.dumps({'path':str(path),'n_methods':nmethods,'n_retrieval_rows':manifest['n_executed_rows'],
        'qa_rows':qaexec['rows'],'qa_actual_calls':qaexec['actual_calls'],'kv_profiles':kvexec['rows'],'inserted_independent_figures':idx,
        'not_a_replacement_of_original_manuscript':True,'semantic_accuracy_status':'complete_model_proxy' if judge is not None else 'pending_permission_and_independent_validation',
        'render_status':'pending'},ensure_ascii=False,indent=2),encoding='utf-8')
    print(path,flush=True)

if __name__=='__main__':main()
