"""Build a self-contained Chinese literature-search notebook from verified metadata.

Network searches and Crossref queries were performed separately on 2026-10-29.
The builder never contacts the network or writes to the shared literature library.
"""
from pathlib import Path
import csv
import hashlib
import json
import platform
import re
from urllib.parse import quote

import nbformat as nbf
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / '.cache/paper-search/paper-web-search/20261029-csi-pid'
OUT = ROOT / 'jupyter/codex_0001_csi_pid_bandwidth_literature_web.ipynb'
LIB = Path('/media/sf_works/papers')
DATE = '2026-10-29'

INFO = {
    'viet2021': dict(
        author='Nguyen V. H. Viet 等，2021',
        relevance='最接近装置：CsI(Tl)、多个 MPPC 求和；光电子统计、积分门与 PSD 宽度。',
        url='https://ieeexplore.ieee.org/document/9309257/',
        scope='Crossref 正式题录；arXiv v1 摘要及 HTML §II–IV（重点 §III-A、IV-A–D）。未取得正式版正文逐项比对。',
        evidence=['https://arxiv.org/abs/2010.00775', 'https://arxiv.org/html/2010.00775v1'],
        version='arXiv:2010.00775（2020）明确关联正式 DOI；合为同一研究，优先列 2021 卷期版。预印本 DOI 为 10.48550/arXiv.2010.00775。',
        finding='预印本用独立 prompt/delayed 光电子与 Poisson 假设推得 σ²_R≈R(1−R)/N；长门影响电荷收集和 PSD，周期噪声会改变最优门。',
        conditions='五个 S13360-6075CS 求和、大尺寸 CsI(Tl)，主要是天然钍源 α/β；与本项目 S13361 阵列、1 cm³ 晶体及 Am/Co 条件不同。',
        limits='不是当前装置的 Npe 标定；文中比值是尾门/总门。相关雪崩、有限积分、基线误差存在时，简单 Poisson 宽度不再充分。',
        suggestion='在原始事件与原 3 μs 电荷选择固定后，对可用积分窗扫描，比较中心、IQR 和门间协方差；只使用实际记录支持的尾部，不补造长波形。'),
    'coakley2021': dict(
        author='Kevin J. Coakley，2021',
        relevance='直接对应按观测 Q 分箱的比值 PID：条件分布与能量不确定性。',
        url='https://www.nist.gov/publications/ratio-based-pulse-shape-discrimination-analytic-results-gaussian-and-poisson-noise',
        scope='NIST 官方题录与完整摘要、Crossref；正文未精读。',
        evidence=['https://www.nist.gov/publications/ratio-based-pulse-shape-discrimination-analytic-results-gaussian-and-poisson-noise'],
        version='按 NIST 2021 正式论文记录；本轮未定位需要合并的独立预印本。',
        finding='给出 Gaussian/Poisson 模型下，已知观测总积分时 prompt 积分的条件分布，并处理有效沉积能量的不确定性。',
        conditions='假定两类的理论 prompt fraction 随能量关系已知；不是 CsI–SiPM 专门的器件噪声模型。',
        limits='摘要不能证明其模型包含本项目的有色噪声、触发选择及全部长短门协方差；本项目没有独立实验粒子真值。',
        suggestion='细化 Q 箱并匹配箱内 Q 分布，区分条件宽度和均值随 Q/采集段变化造成的混合宽度；不直接使用文中的 ROC 作为实验效率。'),
    'rosado2026': dict(
        author='Víctor Moya、Jaime Rosado，2026',
        relevance='专门研究输出方差：像元恢复、相关噪声、非线性与光子计数分辨。',
        url='https://www.mdpi.com/1424-8220/26/17/5579',
        scope='Crossref 正式题录和出版社提交摘要；出版社检索结果含 §2–3 摘录，直接正文页面受 429 限制。结论以摘要范围为主。',
        evidence=['https://www.mdpi.com/1424-8220/26/17/5579'],
        version='正式发表于 2026-09-02；本轮未检出可确认的独立预印本。不与 2024 年同作者论文合并。',
        finding='从瞬时与长脉冲极限构造方差模型，再用现象学插值处理任意持续时间及相关噪声，并与模拟及光源实验比较。',
        conditions='被动淬灭 SiPM；需要光脉冲持续时间、恢复及占据信息。',
        limits='输出总电荷的方差不等于条件 PID 比值、平均时间或 PCA 的宽度；文献支持检查非线性，不证明本装置由饱和主导。',
        suggestion='先从已有光子/雪崩历史统计每像元短时占据、恢复中再触发和增益损失，再决定是否启用相应模型；同时比较条件方差和均值。'),
    'lu2017': dict(
        author='X. Lu 等，2017',
        relevance='CsI(Tl) 发光形状、分量产额及能量依赖，可检验固定衰减参数是否过强。',
        url='https://journals.aps.org/prapplied/abstract/10.1103/PhysRevApplied.7.014007',
        scope='APS 官方题录和完整摘要、Crossref；正文需授权，未精读。',
        evidence=['https://journals.aps.org/prapplied/abstract/10.1103/PhysRevApplied.7.014007'],
        version='2017 正式版；本轮未定位可确认的独立预印本。',
        finding='载流子输运/反应模型解释多衰减分量、脉形和各分量产额随 γ 能量的变化；摘要提及快、长分量的能量趋势相反。',
        conditions='CsI:Tl 材料发光动力学；要与 SiPM、光输运、电子学响应分开。',
        limits='随能量的反向变化不等于固定能量下逐事件分量反相关；不能将其 γ 参数直接当成 Am α 参数或唯一的材料常数。',
        suggestion='先比较同 Q、同采集段的尾部残差和形状参数分布；扣除拟合误差与观测噪声后，才检验额外事件间形状自由度。'),
    'jha2013': dict(
        author='Abhinav K. Jha 等，2013',
        relevance='完整的光子→雪崩→电路响应框架，支持均值与方差、机制开关对照。',
        url='https://ieeexplore.ieee.org/document/6423840/',
        scope='Crossref 正式题录；PMC 作者稿 HTML 摘要、引言、§II/III/IV/V 相关段落。未逐式复现。',
        evidence=['https://pmc.ncbi.nlm.nih.gov/articles/PMC4519993/'],
        version='PMC4519993 明示对应正式期刊 DOI，归为同一研究。Crossref 卷号是 60；PMC 头部和部分库内引用的 30 为冲突值，不采用。',
        finding='离散时间/事件 MC 处理恢复、串扰、后脉冲与公共读出压降，形成瞬态响应并可计算输出方差。',
        conditions='验证包含 LaBr3:Ce 与 S10362-33-050C；器件及读出参数必须另行约束。',
        limits='不是本项目 CsI–S13361 的验证，旧器件参数不可照搬；器件参数退化不会因为模型更详细而自动消失。',
        suggestion='保持 Geant4 光学链，用同一批光子历史重放不同响应机制，检查对 QDC/平均时间/PCA 的联合影响。'),
    'moya2024': dict(
        author='Víctor Moya-Zamanillo、Jaime Rosado，2024',
        relevance='说明相关噪声可显著改变输出电流形状；平均电荷吻合不足以检验波形。',
        url='https://www.mdpi.com/1424-8220/24/8/2648',
        scope='Crossref 正式题录与摘要；arXiv v1 HTML 引言、§2 及摘要。出版社直接页面 429。',
        evidence=['https://arxiv.org/abs/2401.06581', 'https://arxiv.org/html/2401.06581v1'],
        version='arXiv:2401.06581 与正式版同题、同作者/机构，核心 MC 修改和两类拟合模型一致，归为同一研究；arXiv 未填期刊 DOI。其排版中的 IEEE Sensors Journal 不是正式发表依据；Crossref 确认 Sensors。预印本 DOI：10.48550/arXiv.2401.06581。',
        finding='非线性受光子率和恢复时间之比、光脉形及读出阻抗影响；相关噪声对平均响应非线性的影响较小时仍可明显改变电流形状。',
        conditions='适用于检查恢复与噪声对瞬态形状的影响，需区分光子时序、像元恢复及电子学常数。',
        limits='不是证明当前 Am 带宽缺失一定由后脉冲造成；当前长尾与响应常数仍退化。',
        suggestion='检查延迟相关分量对长时协方差与平均时间的方向性影响，并同时约束平均尾部，防止只把宽度调对。'),
    'rosado2015': dict(
        author='J. Rosado 等，2015',
        relevance='将即时串扰、后脉冲及迟发串扰分开；适合响应机制对照。',
        url='https://www.sciencedirect.com/science/article/pii/S0168900214013862',
        scope='Crossref 正式题录；arXiv 官方摘要/版本 DOI；已有单篇笔记身份、适用范围和采用意见。未重读本地 PDF。',
        evidence=['https://arxiv.org/abs/1409.4564'],
        version='arXiv:1409.4564 明确关联期刊 DOI；正式卷年 2015，原笔记记录在线 2014。预印本 DOI：10.48550/arXiv.1409.4564。',
        finding='相关雪崩的类型与延迟须区分；后脉冲和迟发串扰的统计分解有条件性与非唯一性。',
        conditions='原笔记限定为两种旧 Hamamatsu 器件、暗条件/室温，无晶体。',
        limits='不能直接移植相关噪声概率、陷阱寿命或邻域常数；当前记录未必足以单独辨识这些参数。',
        suggestion='与现有有色残差模型做替代/消融对照；加入物理相关雪崩后须重新分配残差方差，防止重复计数。'),
    'garutti2021': dict(
        author='E. Garutti 等，2021',
        relevance='低占据下逐雪崩 MC、门积分和电子学卷积，便于核查响应模块。',
        url='https://www.sciencedirect.com/science/article/pii/S016890022100838X',
        scope='Crossref 正式题录；出版社检索摘要；本地笔记身份、边界和采用意见。未重读本地 PDF。',
        evidence=['https://www.sciencedirect.com/science/article/abs/pii/S016890022100838X'],
        version='正式版已收录。arXiv:2006.11150 是相关的同作者早稿，未查到明确正式 DOI 对应；另列待核，不据主题相近直接合并。',
        finding='将主放电与相关雪崩转为带幅度/时间的事件表，然后生成积分电荷与电子学卷积波形。',
        conditions='低占据/忽略饱和的线性区；PDE、绝对增益及电路响应需外部输入。',
        limits='不能因为含后脉冲恢复因子就视作完整饱和模型；已有笔记仍是待审草稿。',
        suggestion='用作同历史响应重放和逐机制协方差检查的实现参照；先核实本项目的瞬时像元占据。'),
    'amorini2012': dict(
        author='F. Amorini 等，2012',
        relevance='CsI(Tl) 快慢分量的时间常数与积分强度拟合；约束形状模型解释。',
        url='https://ieeexplore.ieee.org/document/6236256/',
        scope='Crossref 正式题录；作者机构库摘要；本地笔记身份、模型边界及审核状态。未重读本地 PDF。',
        evidence=['https://re.public.polimi.it/handle/11311/666561'],
        version='正式期刊版已收录；本轮未定位需要单独归组的预印本。',
        finding='由带电子学响应的波形拟合研究 CsI(Tl) 快慢积分强度和时间常数随粒子与能量的依赖。',
        conditions='PD/前放、10–240 MeV 束流粒子及外部粒子识别；不同于本项目 SiPM/无前放源测量。',
        limits='积分强度比不等于峰值系数比；现有约 3 μs 主积分窗难唯一确定很长衰减时间。',
        suggestion='比较可辨识的窗内尾部份额与共享 RC 参数族，而非自由给每个事件拟合所有时间常数。'),
}
ORDER = ['viet2021','coakley2021','rosado2026','lu2017','jha2013','moya2024','rosado2015','garutti2021','amorini2012']
OWN = {'rosado2015','garutti2021','amorini2012'}
QUERIES = [
    'CsI Tl SiPM pulse shape discrimination fluctuations simulation photon statistics noise',
    'silicon photomultiplier response model correlated noise recovery scintillation pulse shape discrimination variance',
    '"Pulse Shape Discrimination of CsI(Tl) with a Photomultiplier Tube and MPPCs"',
    'CsI Tl pulse shape discrimination intrinsic fluctuations slow component simulation 2024 2025',
    'pulse shape discrimination figure of merit statistical fluctuations charge integration covariance noise model scintillator',
    'CsI Tl intrinsic pulse shape fluctuations alpha gamma statistical light yield discrimination simulation',
    '"Modeling the Variance of Passive SiPMs" authors 2026',
    '"Energy-Dependent Scintillation Pulse Shape" site:arxiv.org',
    '"Modeling crosstalk and afterpulsing in silicon photomultipliers" site:arxiv.org',
    '"Simulation of the response of SiPMs" Garutti Klanner 2021 arxiv',
    '"10.3390/s24082648" "2401.06581"',
    '"2006.11150" "165853"',
]


def mdlink(label, url):
    return f'[{label}](<{url}>)'


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def make_records():
    lookup = json.loads((CACHE/'library_lookup.json').read_text())
    records = []
    for ident in ORDER:
        p = CACHE/'crossref'/f'{ident}.json'
        d = json.loads(p.read_text())['message']
        row = dict(INFO[ident])
        row.update(id=ident, title=' '.join(d['title'][0].split()), doi=d['DOI'],
                   year=d['published']['date-parts'][0][0], venue=d['container-title'][0],
                   crossref_url='https://api.crossref.org/works/'+quote(d['DOI'], safe=''),
                   crossref_sha256=digest(p), doi_status='已核验（Crossref；题名、作者、年份匹配）')
        evidence = lookup['candidate_hits'][ident]
        row['local_note'] = None
        row['local_raw'] = None
        row['local_reference_mentions'] = [x for x in evidence['text_hits'] if '/sources/' in x['path']]
        row['collection'] = 'new'
        if ident in OWN:
            # These three paths were reviewed against their own bibliographic headers.
            row['local_note'] = next(x['path'] for x in evidence['filename_hits'] if x['path'].endswith('.md'))
            row['local_raw'] = next(x['path'] for x in evidence['filename_hits'] if x['path'].endswith('.pdf'))
            row['local_note_sha256'] = digest(Path(row['local_note']))
            row['local_raw_sha256'] = digest(Path(row['local_raw']))
            assert d['DOI'].lower() in Path(row['local_note']).read_text().lower()
            row['local_status'] = '已收录（正式原文＋单篇笔记）；自身题录 DOI 匹配；笔记仍待审'
            row['collection'] = 'owned'
        elif ident == 'jha2013':
            row['local_status'] = '未检出自身原文/笔记；仅有引用及待读记录'
        elif ident == 'lu2017':
            row['local_status'] = '未检出自身收录记录；仅有他文引用提及'
        else:
            assert not evidence['filename_hits'] and not evidence['text_hits']
            row['local_status'] = '本轮核验范围内未检出'
        row['local_mention_paths'] = [x['path'] for x in row.pop('local_reference_mentions') if x['path'] != row['local_note']]
        records.append(row)
    return records, lookup


def table(records):
    lines = ['| 论文原题 | 作者及年份 | 相关性 | 论文网址 | DOI | 本地收录状态 |',
             '| --- | --- | --- | --- | --- | --- |']
    for r in records:
        status = r['local_status']
        if r.get('local_note'):
            status += '；'+mdlink('笔记',r['local_note'])+' / '+mdlink('原文',r['local_raw'])
        elif r.get('local_mention_paths'):
            status += '；'+mdlink('引用位置（非该文笔记）',r['local_mention_paths'][0])
        doi = mdlink(r['doi'], 'https://doi.org/'+r['doi'])
        if r.get('crossref_url'):
            doi += '（'+mdlink('Crossref 已核验',r['crossref_url'])+'）'
        else:
            doi += '（官方 arXiv 核验；预印本 DOI）'
        vals = [r['title'],r['author'],r['relevance'],mdlink('论文页',r['url']),doi,status]
        lines.append('| '+' | '.join(v.replace('|','\\|') for v in vals)+' |')
    return '\n'.join(lines)


def main():
    records, lookup = make_records()
    pending = dict(id='garutti2020_relation_pending',
        title='A computer program to simulate the response of SiPMs', author='E. Garutti、R. Klanner、J. Rolph、J. Schwandt，2020',
        relevance='与已收录 Part I 的方法接近；同作者早稿的版本关系待核，暂不计作新增研究。',
        url='https://arxiv.org/abs/2006.11150',doi='10.48550/arXiv.2006.11150',year=2020,
        local_status='原文目录/笔记/索引未检出自身记录；与 Part I 的版本关系待核',collection='pending',
        doi_status='已核验（官方 arXiv 条目；不是期刊 DOI）')
    source_csv = ROOT/'simulation/results/joint_constraints_20261029/validation_table.csv'
    current = [r for r in csv.DictReader(source_csv.open()) if r['model']=='joint' and r['group'].endswith('last20')]
    assert len(current)==2
    source_record = ROOT/'simulation/results/joint_constraints_20261029/record.json'
    config = ROOT/'simulation/results/joint_constraints_20261029/validation/run12_Am241/config.json'
    sensor = json.loads(config.read_text())['sipm']['sensor']
    assert sensor['afterpulse_probability']==sensor['crosstalk_probability']==sensor['dark_rate_hz']==0
    audit = {
        'date':DATE, 'topic':'CsI–SiPM 模拟与实测 PID 带宽差异', 'library':str(LIB),
        'counts':{'raw_pdfs':lookup['raw_pdf_count'],'notes':lookup['note_count'],'new_candidates':6,'owned':3,'pending_versions':1},
        'coverage':lookup['coverage_limit'], 'indexes':lookup['indexes'],
        'queries':QUERIES, 'dedup_errors':lookup['read_errors'],
        'read_only_library':True, 'downloaded_remote_pdfs':False,
        'modified_simulation':False,
        'source_record':str(source_record.relative_to(ROOT)), 'source_record_sha256':digest(source_record),
        'source_csv':str(source_csv.relative_to(ROOT)), 'source_csv_sha256':digest(source_csv),
        'source_config':str(config.relative_to(ROOT)), 'source_config_sha256':digest(config),
        'current_last20':current,
        'doi_conflicts_resolved':['Viet: DOI 含 2020，正式卷期为 2021。','Jha: Crossref 卷 60，PMC 头部/一些旧引文卷 30 不采用。','Moya 2024: 正式期刊为 Sensors；不采用预印本排版里的 IEEE Sensors Journal。'],
        'limits':['非系统综述；仅核验与当前差异最相关的候选。','原文目录按文件名与笔记题录核对，没有逐一全文扫描 1880 份 PDF。','部分出版社页面返回 403/429 或 JavaScript 检查；题录用 Crossref，方法只按明确注明的摘要/作者稿范围。','Garutti 2020/2021 关系未确认；不重复计为新研究。'],
    }
    payload={'audit':audit,'papers':records,'pending':[pending]}
    (CACHE/'research_record.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
    cells=[]
    md=lambda text:cells.append(nbf.v4.new_markdown_cell(text))
    code=lambda text:cells.append(nbf.v4.new_code_cell(text))
    md('''# CsI–SiPM 模拟与实测 PID 带宽差异：联网文献核验与下一步

2026-10-29，使用本项目 `paper-web-search`。本轮是文献检索与项目建议，没有改模拟参数或产生新的模拟验证结果。

**判断：应先查条件分布、长时协方差与响应机制，不能把所有带宽差异都归于基线噪声。** 当前平均 ADC 波形已接近，但 Am 平均时间的中心和宽度、短尺度起伏和额外尾部尚不能同时解释。最新阶段已经拟合完整 PID 协方差，本轮建议是在其残差上辨别机制，而不是重复增加同一个拟合目标。

共享库按只读使用。实际检查 `index.md`、`topic-index.md`、`review-index.md`、全部 `sources/*.md` 及 `raw/` 文件名，并用待读队列/审核记录辅助分辨引用提及。实际目录为 **1880 份 PDF、1856 篇笔记**，主索引的 1869 份 PDF 是较早统计。没有逐份全文检索 PDF，也未将缓存、历史记录或参考文献算作收录。''')
    md('主要实际检索式：\n\n'+'\n'.join('- `'+q+'`' for q in QUERIES)+'''

九篇正式论文均完成 Crossref 的题名、作者、年份及 DOI 核对。IEEE/Elsevier/MDPI 部分页面直接访问受限；下文分别标明摘要、预印本 HTML、作者稿或本地笔记的实际阅读范围，未把题录核验说成全文精读。没有下载远程 PDF、入库或修改审核状态。''')
    md('## 当前项目证据与约束\n\n本单元展示冻结结果的数值快照；Co/Am 对应 run3/run12 的源测量主带，不是逐事件实验粒子真值。CI 是固定候选的条件区间，不含拟合、选组和装置不确定性；包含 1 不等于宽度已经等效。')
    code('import json\nfrom IPython.display import display, Markdown\n\nresearch = json.loads('+repr(json.dumps(payload,ensure_ascii=False))+')\nassert research["audit"]["dedup_errors"] == []\nassert len(research["papers"]) == 9\nassert len({p["doi"].casefold() for p in research["papers"]}) == 9\nprint("9 journal DOIs verified; 6 not found, 3 owned; 1 version relation pending.")\nprint("Library: 1880 PDFs and 1856 notes; read-only.")')
    code('''rows = research['audit']['current_last20']
lines = ['| 实测分组 | MC 共同区事件数 | 均 ADC NRMSE | 平均时间偏差 / ns（95% CI） | QDC / 平均时间 / PCA 的 IQR 比（MC/数据） | F50 比 |', '| --- | --- | --- | --- | --- | --- |']
for r in rows:
    ratios=' / '.join(f"{float(r[k]):.3f}" for k in ['QDC_IQR_ratio','mean_time_IQR_ratio','PCA_IQR_ratio'])
    delta=f"{float(r['mean_time_difference_ns']):+.2f} [{float(r['mean_time_CI95_low']):+.2f}, {float(r['mean_time_CI95_high']):+.2f}]"
    lines.append(f"| {r['group']} | {r['n_MC']} | {float(r['ADC_NRMSE_percent']):.2f}% | {delta} | {ratios} | {float(r['F50_ratio']):.3f} |")
display(Markdown('\\n'.join(lines)))
print('Frozen record SHA256:', research['audit']['source_record_sha256'])''')
    md('''装置为 10×10×10 mm³ CsI(Tl)、PTFE 包裹、5 μm Mylar 入射面、S13361-3050 系列多个单元求和，无前放。阵列具体单元数、并联拓扑、实际增益与耦合等仍不完整；模型的 16 单元、PDE、29 ns 恢复和源距均有假设成分。

最新候选的 **DCR、即时串扰和后脉冲参数均为 0**，而信号期起伏由经验有色残差承担。这意味着值得检查机制等效性；不能据此声称真实器件没有相关噪声，也不能把新增相关雪崩与原残差方差简单叠加。

只用已有实验数据。前 60%用于拟合，后段已经多次检查，只能称内部验证。后续模拟的新随机种子提供新的 MC 历史，不会产生新的实验验证样本。''')
    md('## 优先候选：本轮未检出自身收录\n\n“未检出”只针对上述实际检查范围。\n\n'+table([r for r in records if r['collection']=='new']))
    md('## 已收录：可直接利用现有笔记\n\n'+table([r for r in records if r['collection']=='owned'])+'''

三篇自身题录 DOI 与正式原文路径均已匹配。笔记仍是 AI 草稿待审；Rosado/Garutti 的既有意见只确认用户关心“光子到电信号与 Geant4 后响应”这一研究方向，未确认本项目采用其具体模型、参数或数值精度。''')
    md('## 疑似版本关系\n\n'+table([pending])+'''

同作者、方法接近不足以确认正式发表版本关系。本轮将其作为待核的关联早稿，不合并身份，也不计为第七篇新增研究。该预印本的 DOI 来自官方 arXiv 条目，不是期刊 DOI。''')
    md('## 版本与题录核验记录\n\n'+'\n\n'.join(
        '**'+r['author']+'**：'+r['version']+'\n\n核验/阅读范围：'+r['scope']+' 来源：'+'；'.join(mdlink('官方页面',u) for u in r['evidence'])+'；'+mdlink('Crossref',r['crossref_url'])+'。' for r in records))
    md('## 方法的证据、边界与本项目建议\n\n以下“建议”是本次对本项目的推断，不是论文已经验证的结论。')
    for ident in ['viet2021','coakley2021','rosado2026','lu2017','jha2013','moya2024','rosado2015','garutti2021','amorini2012']:
        r=next(x for x in records if x['id']==ident)
        md('**'+r['author']+'**\n\n| 项目 | 内容 |\n| --- | --- |\n'+
           '| 论文结论/既有记录 | '+r['finding']+' '+mdlink('来源',r['evidence'][0])+' |\n'+
           '| 适用条件 | '+r['conditions']+' |\n'+
           '| 局限 | '+r['limits']+' |\n'+
           '| 本项目可验证方向（建议） | '+r['suggestion']+' |')
    md(r'''## 为什么需要协方差，而不只是噪声 RMS

下面是**本次分析推导**，并非上述论文在本装置的测量结论。令 $S=Q_{short}$、$L=Q_{long}$，$R=S/L$；若实际分数用 $1-S/L$，一阶方差相同。围绕局部均值、小相对起伏且 $L$ 远离零时：

$$\operatorname{Var}(R)\simeq\frac{\operatorname{Var}(S)+\bar R^2\operatorname{Var}(L)-2\bar R\operatorname{Cov}(S,L)}{\bar L^2}.$$

两个积分来自同一条波形并共享样点，不能当作独立测量。电子学噪声相同 RMS、不同时间相关性，可以给出不同门间协方差。对平均时间 $T=\sum_i t_iw_i/\sum_iw_i$，固定时间轴下 $J_i=\partial T/\partial w_i=(t_i-\bar T)/\bar L$，故 $\operatorname{Var}(T)\simeq J C_w J^\mathsf T$；实际 $t_{10}$ 与积分门会随噪声变化，还需完整算法前向计算。

进一步，令 $z$ 表示观测电荷、采集段等条件，则宽 Q 箱内：

$$\operatorname{Var}(R\mid B)=\mathbb E[\operatorname{Var}(R\mid z,B)\mid B]+\operatorname{Var}(\mathbb E[R\mid z,B]\mid B).$$

第二项是不同条件均值混合造成的宽度，不等同于单事件噪声。上述方差关系不能直接换算非高斯分布的 IQR；本项目的 IQR、尾部分布和选择效率仍须从原生 ADC 的完整分析输出检查。''')
    md('''## 建议执行顺序（尚未执行）

1. **先区分选择/状态混合与条件宽度。** 复用现有逐 Q、逐段结果，在前 60% 做更细的 Q 匹配，固定事件 ID 和原 3 μs Q 范围，检查中心斜率、箱内 Q 分布、基线状态及触发附近事件；后段只报告。已经看到的 run12 段间变化对块长敏感，暂不归因于温度漂移。
2. **利用现有协方差残差做响应机制对照。** 保存光子历史及同一母事件身份，比较电子学噪声、即时串扰、延迟相关雪崩/恢复、公共响应常数变化对长时相关性和三种 PID 的影响。先检查像元占据，且开启物理噪声时重新分配经验残差，防止重复计入。参数在可辨识范围内共用，不为 Am/Co 分别随意扩大 PID 分数。
3. **仅在残差仍需要时增加形状自由度。** 先做形状参数误差与噪声传播的闭环检查，再试小范围、受约束的尾部份额随 Q 变化或事件间变化；同时匹配中心、宽度、完整协方差、F50/102/202 和尾部。Lu 的能量趋势不支持直接凭空加入逐事件快慢分量反相关。
4. **用现有记录做积分窗敏感性检查。** 在保证同一事件集合且所有样点确实存在的前提下，比较 2.0、2.5、3.0 μs，并视记录支持延伸至 3.5/3.8 μs；原 3 μs 结果保持为冻结基准。不能依据他文较长记录补造当前未测尾部，也不能把减少某类事件后的窄带当作模拟改进。
5. **锁定候选后重新验证完整链路。** 使用新独立 Geant4 母历史，报告共同 Q 区有效统计、块 bootstrap 与参数族敏感性；沿用当前候选/未推广状态，直到中心和宽度等同时满足事先规定的标准。run18 可做阈值/读出条件明确的迁移检查，不宣称同条件重复；Am 中的 γ 和衰变电子仍需保留。

首选下一步是 **第 1–2 项**。现有数据可排除一些机制并约束有效参数，但不足以唯一测定 PDE、增益、像元数、后脉冲参数和材料衰减常数；应保留参数族，而不是追求一套看似唯一的“最佳参数”。''')
    md('## 只读核验与保存检查\n\n下面仅检查快照、DOI 唯一性及已收录记录的实际路径；所有文献元数据嵌入本 notebook，阅读和代码重跑不需要联网。完整 Crossref JSON、候选检索命中及构建输入保存在本项目 `.cache/paper-search/paper-web-search/20261029-csi-pid/`。')
    code('''from pathlib import Path
import hashlib

for p in research['papers']:
    assert p['doi'].startswith('10.') and p['doi_status'].startswith('已核验')
    if p['collection'] == 'owned':
        note = Path(p['local_note'])
        raw = Path(p['local_raw'])
        assert note.is_file() and raw.is_file()
        text = note.read_text(encoding='utf-8')
        assert p['doi'].casefold() in text.casefold()
        assert hashlib.sha256(note.read_bytes()).hexdigest() == p['local_note_sha256']
        assert hashlib.sha256(raw.read_bytes()).hexdigest() == p['local_raw_sha256']
assert sum(p['collection']=='new' for p in research['papers']) == 6
assert sum(p['collection']=='owned' for p in research['papers']) == 3
assert len(research['pending']) == 1
print('PASS: bibliographic identity and local own-record checks; no library writes.')''')
    md('''尚未完成的范围：部分正式正文未获访问，本轮没有对九篇论文逐篇全文精读或复现；Garutti 2020/2021 版本关系待核。目录/题录去重已完成上述范围，没有把引用或队列误算成收录。需要全文实现细节时，应针对方法再核原文，不能以本表替代全文证据。

本 notebook 是研究调研记录，不是论文入库操作，也不是新的模拟一致性或实验 PID 效率证明。''')
    nb=nbf.v4.new_notebook(cells=cells,metadata={
        'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
        'language_info':{'name':'python','version':platform.python_version()},
        'research_audit':{'date':DATE,'skill':'paper-web-search','read_only_library':True,'remote_pdfs_downloaded':False}})
    # Keep the embedded evidence available without dominating the reading view.
    next(c for c in nb.cells if c.cell_type == 'code').metadata['jupyter'] = {'source_hidden': True}
    # Execute simple read-only cells in process; avoid a kernel writing runtime files
    # outside the project or requiring network ports.
    shell=InteractiveShell.instance()
    count=0
    for cell in nb.cells:
        if cell.cell_type!='code':continue
        count+=1
        with capture_output() as captured:
            result=shell.run_cell(cell.source,store_history=False)
        if result.error_before_exec or result.error_in_exec:
            raise RuntimeError(f'Notebook cell {count} failed')
        cell.execution_count=count
        outputs=[]
        if captured.stdout:outputs.append(nbf.v4.new_output('stream',name='stdout',text=captured.stdout))
        if captured.stderr:outputs.append(nbf.v4.new_output('stream',name='stderr',text=captured.stderr))
        for rich in captured.outputs:
            outputs.append(nbf.v4.new_output('display_data',data=rich.data,metadata=rich.metadata))
        cell.outputs=outputs
    nbf.validate(nb)
    nbf.write(nb,OUT)
    reread=nbf.read(OUT,as_version=4)
    nbf.validate(reread)
    assert all(o.output_type!='error' for c in reread.cells if c.cell_type=='code' for o in c.outputs)
    print(json.dumps({'notebook':str(OUT),'code_cells_executed':count,'cells':len(nb.cells),'bytes':OUT.stat().st_size,'sha256':digest(OUT),'bibliography_counts':audit['counts']},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
