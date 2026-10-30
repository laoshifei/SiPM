"""Save a local-only literature update and an executed, self-contained research note."""
import hashlib
import json
import os
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
LIB = Path('/media/sf_works/papers')
CACHE = PROJECT/'.cache/paper-search/2026-10-30-library-update'
OUT = PROJECT/'jupyter/run_records/library_update_20261030'
NOTEBOOK = PROJECT/'jupyter/codex_0001_csi_library_update_20261030.ipynb'
os.environ['IPYTHONDIR'] = str(PROJECT/'.cache/codex_0001_literature_update_ipython')
import nbformat as nbf
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')


def main():
    assert not NOTEBOOK.exists() and not OUT.exists()
    OUT.mkdir(parents=True)
    extraction = read(CACHE/'extraction_manifest.json')
    inventory = read(CACHE/'inventory_delta.json')
    assert inventory['stable_names_during_read']
    batch = LIB/'.cache/paper-read/batch-2026-10-30-P13-parallel16'
    definitions = [
        ('hong2026', 'Reconstruction Fidelity and Confidence Reliability in Overlapped Radiation-Detector Pulse Analysis Under Distribution Shift',
         'Tao Hong、Bo Zhang、Xinjie Yao、Zhoupeng Guo、Guangsheng Xie', 2026, '10.3390/s26206600', batch/'P13-012/draft.md',
         '高：校准转移和验证边界', 'PDF 7、13、18、21；式(1)，表3、7–9，§6.3–6.4',
         '受控白噪声与整条波形常量偏置下，名义分布的概率校准未可靠转移；用匹配扰动的验证数据校准可改善ECE/NLL，但不改变类别准确率。',
         '研究对象是单/双脉冲分类及半合成分量重建；不是α/γ分类，也不是CsI–SiPM的噪声模型。温度缩放中的温度是logit刻度，不是器件温度。',
         '明确区分新读出状态内的条件校准、块留出内部检查和跨状态转移；分别报告均波形、PID中心/协方差和F，不能用一种指标改善代替全部。'),
        ('georgiev2025', 'A deconvolution based signal reconstruction capable of piled-up pulse separation',
         'Georgi Georgiev', 2025, '10.1088/1748-0221/20/07/P07044', LIB/'.cache/paper-read/single-2026-10-30-Georgiev/draft.md',
         '中：响应模板及残差诊断', 'PDF 4–5、7–8、10；式(2.5–2.6)、(3.2)、(3.7–3.9)、(4.1)，图3，§5',
         '以稳定响应的线性叠加和带限逆滤波重建时间、峰高；低通减少高频噪声，也引入展宽与旁瓣。性能扫描使用同一估计模板生成的双脉冲。',
         '塑料闪烁体与SiPM、带整形放大的PADME读出，非本项目无前放CsI阵列。末端归零的基线条件未必适合长尾；固定模板可能改变粒子依赖形状。8 ns是特定测试的分离边界，不是本项目PID或时间分辨率。',
         '先用不同采集块的独立平均形状检查模板稳定性；模板仅作诊断对照。任何反卷积对照同时检查原始ADC中的α/γ形状差异、尾积分和噪声协方差。'),
        ('simpson2019', 'Performance of four digital algorithms for γ–γ timing with LaBr3(Ce) scintillators',
         'G. Simpson、S. Curtoni、D. Dauvergne、M.-L. Gallin-Martel、S. Marcatilli、G. Thiamova', 2019, '10.1016/j.nima.2019.05.015', batch/'P13-007/draft.md',
         '中：前沿定位对照', 'PDF 4–6；§3.4–§6，图8–10，表1',
         '同一套波形比较前沿、拟合前沿、CFD和平均脉冲的方差最小lag；最佳算法随所选能区和已有参数改变。',
         'LaBr3–PMT、40 GS/s等条件与本项目不同；表1是近似单探头FWHM，不能当CsI PID宽度。模板构建与独立测试细节有缺口，文中autocovariance不等于本项目的信号期噪声自协方差。',
         '增加只用前沿估计位置的对照，积分保持原ADC、原事件/Q；对齐模板从校准块产生。检查定位变化是否同时改变平均时间、PCA和尾电荷。'),
        ('kong2026', '高精度数字化定时方法研究 / Study on high-precision digital timing methods',
         '孔誉谦、吴鸿毅、张广鑫', 2026, '10.7498/aps.75.20260283', batch/'P13-014/draft.md',
         '中：亚采样相位审计', 'PDF 2–4、6–7；式(1–6)，图1–3、6，§4',
         '双指数模型、加性高斯噪声和两点线性CFD插值的模拟显示采样相位会影响时间差分布。',
         '没有本篇实测验证，未计量化噪声；4–6点的数值建议受模型和参数缺口限制。已有讨论稿指出100 MHz下原印tr=td组合退化，尚不能照搬数值结论。',
         '在固定Q和读出状态下检查估计t10的样点内相位与PID残差的关系，并用原生模拟控制真实相位；测得t10本身含噪声，不能直接当真值。相位检查不预设能解释约7 ns偏差。'),
        ('windischhofer2021', 'The statistics of electron–hole avalanches',
         'P. Windischhofer、W. Riegler', 2021, '10.1016/j.nima.2021.165327', None,
         '背景：微观雪崩统计', 'PDF 1–2；摘要、§1–2、图1说明；仅定向初筛',
         '讨论电子/空穴倍增的平均发展与统计涨落，以及阈值取时；模型假设一维、局部电离系数，忽略空间电荷反馈。',
         '当前核验没有覆盖全文推导；§2明确把淬灭起始等空间电荷问题排除。不能将其早期雪崩统计直接用作微秒闪烁波形的经验残差。',
         '保留为底层模型依据候选；现阶段不据此新增自由噪声幅度或宣称已解释CsI的带宽。'),
        ('inoue2020', 'Modeling and Analysis of Capacitive Relaxation Quenching in a Single Photon Avalanche Diode (SPAD) Applied to a CMOS Image Sensor',
         'Akito Inoue、Toru Okino、Shinzo Koyama、Yutaka Hirose', 2020, '10.3390/s20103007', None,
         '背景：淬灭模型的电路条件', 'PDF 1–2；摘要、§1、§2开头；仅定向初筛',
         '用载流子方程和电容电路讨论SPAD雪崩到淬灭；实验对象是6 μm像素的CMOS成像器件。',
         '几何、寄生电容和读出方式不同于S13361求和阵列；首页所述电压摆幅结论不应成为本项目增益的直接倍增系数。本轮未核全文数值。',
         '提醒区分雪崩电荷、淬灭/再充电和外部读出传递函数；保持现有有效增益/PDE退化边界。'),
        ('ferrarotto2018', 'Performance of the Prototype of the Charged-Particle Veto System of the PADME Experiment',
         'F. Ferrarotto 等', 2018, '10.1109/TNS.2018.2822724', None,
         '背景：Georgiev的装置来源', 'PDF 1；摘要、题录；仅初筛',
         '报道塑料闪烁体否决系统原型、光传感器和前端的束流测试。',
         '不是CsI装置，也未在本轮核查其完整电路和性能曲线；不把该引用当作Georgiev反卷积算法的独立验证。',
         '当前已找到原文，替代旧讨论稿“未找到原文”的可用性状态；科学采用状态仍未改变。'),
        ('li2023', 'Pulse-shaping method for real-time neutron/gamma discrimination at low sampling rates',
         'Jia-Xin Li、Hui-Liang Hou、Yue-Feng Huang、Mao-Song Cheng、Zhi-Min Dai', 2023, '10.1007/s41365-023-01306-z', None,
         '背景：成形后的PID定义', 'PDF 1；摘要、题录；仅初筛',
         '研究NaI:Tl,6Li与Am-Be场中的梯形成形、零交叉时间判别和FPGA实现。',
         '材料、入射粒子及任务不同；摘要FOM不能移作CsI–Am/Co的预计成绩。改善成形后分离能力不等于解决原始波形模拟的带宽差异。',
         '如以后研究在线判别可另做方法对照；本轮不改变正式PID定义。'),
    ]
    papers = []
    inputs = {}
    visual = {'hong2026':[18], 'georgiev2025':[7], 'kong2026':[3]}
    checked = {'hong2026':[1,7,13,18,21], 'georgiev2025':[1,4,5,7,8,10],
               'simpson2019':[1,4,5,6], 'kong2026':[1,2,3,4,6,7],
               'windischhofer2021':[1,2], 'inoue2020':[1,2], 'ferrarotto2018':[1], 'li2023':[1]}
    for key,title,authors,year,doi,note,priority,location,finding,limits,application in definitions:
        ext = extraction[key]
        assert sha(ext['pdf']) == ext['sha256']
        first = (CACHE/key/'page-01.txt').read_text()
        assert doi.lower() in first.lower(), key
        inputs[ext['pdf']] = ext['sha256']
        if note:
            assert note.exists()
            inputs[str(note)] = sha(note)
        papers.append(dict(id=key,title=title,authors=authors,year=year,doi=doi,
            url='https://doi.org/'+doi,pdf=ext['pdf'],note=str(note) if note else None,
            local_status='原文＋既有精读讨论稿，未正式入库，待人工审核' if note else '原文已存在，未见对应正式笔记；本轮初筛',
            doi_status='本轮核对本地正式PDF首页所印DOI；未联网解析',
            priority=priority,location=location,finding=finding,limits=limits,application=application,
            checked_text_pages=checked[key],checked_image_pages=visual.get(key,[]),
            review='本轮未检出针对本篇科学采用范围的用户确认；不是整篇认可'))
    old_note=LIB/'sources/2019_Calo_NIMA_SiPM-readout-electronics.md'
    inputs[str(old_note)] = sha(old_note)
    for name in ['AGENTS.md','index.md','vocabulary.json','topic-index.md','review-index.md','global-review.md','reading-queue.md']:
        inputs[str(LIB/name)] = sha(LIB/name)
    for name in ['inventory_delta.json','extraction_manifest.json','topic-search.txt','review-search.txt']:
        inputs[str(CACHE/name)] = sha(CACHE/name)
    project_results = PROJECT/'simulation/results/acquisition_state_20261030'
    for name in ['record.json','summary.json','states.json','controls.json','decision.json']:
        inputs[str(project_results/name)] = sha(project_results/name)
    prior=read(PROJECT/'.cache/paper-search/paper-web-search/20261029-csi-pid/research_record.json')
    gaps=[{k:p[k] for k in ['id','title','doi']} for p in prior['papers'] if p['collection']=='new']
    record=dict(date='2026-10-30',library=str(LIB),mode='local-only; shared library read-only including cache',
        inventory=inventory,discussion_update='P13 16篇＋Georgiev 1篇已有缓存讨论稿；不是本轮从头精读，也不代表正式入库或科学审核通过。',
        papers=papers,previous_web_candidates=gaps,
        gap_scope='此前未收录的6篇候选，在本轮新增文件清单及未变的正式笔记目录中仍未找到同身份记录；没有重新全文检索全部PDF或联网。',
        old_readout_evidence=dict(title='SiPM readout electronics',authors='Pietro P. Calò 等',year=2019,
            doi='10.1016/j.nima.2018.09.030',note=str(old_note),
            pdf=str(LIB/'raw/2019_Calo_NIMA_SiPM-readout-electronics.pdf'),
            scope='据既有完整精读笔记；本轮未回查该PDF。相关位置为PDF3–5式(2–8)、图3–6。',
            relevance='输入阻抗、寄生电容/电感和多单元负载能影响读出波形；不能由简化线性电路直接识别本装置参数。',
            review='AI待审，无本篇科学采用确认'),
        review_boundary='search-review SiPM命中的Molina第668项仅认可特定NaI(Tl)–SiPM堆积拒绝工程证据；已回查global-review.md:13461，不转移到上述论文或本项目的PID效率。',
        input_sha256=inputs,
        conclusion='新增资料细化验证策略，尚未提供可直接替代经验信号残差的CsI–S13361参数；优先按读出状态联合约束，再补相位/前沿定位诊断。',
        new_experimental_events=0,new_simulated_mothers=0,simulation_modified=False)
    save(OUT/'research_record.json', record)
    cells=[]
    def md(s):cells.append(nbf.v4.new_markdown_cell(s))
    def code(s):cells.append(nbf.v4.new_code_cell(s))
    md('''# CsI–SiPM 文献库更新：对 PID 带宽问题的影响

2026-10-30；使用项目 `paper-search`，只读 `/media/sf_works/papers`，包括库内缓存。本轮没有联网、下载、修改文献库、运行新的Geant4或调整模拟参数。

**判断：更新有助于完善验证和定位诊断，尚不足以确定带宽差异的微观原因。下一步仍优先分读出状态约束信号期响应，新增亚采样相位和仅前沿定位的对照。**

当前项目依据见[采集状态笔记](codex_0001_csi_acquisition_state.ipynb)：Am电荷匹配基线RMS由3.33变为5.91 ADC；四种基线/定位对照的MC−实测平均时间仍为7.05–7.26 ns；前段状态回归的平均时间交叉MSE为Q-only的1.060倍，后段状态支持比例仅33.7%。这些是本项目结果，不能归为下列论文的结论。

## 检索范围与更新状态

读取AGENTS、索引、词表、主题/审核索引，运行库内主题和审核检索，补查来源、概念、方法、专题、全局审核及当前队列。
库内检索脚本默认可能写缓存，因此本轮明确使用 `--no-cache` 与 `PYTHONDONTWRITEBYTECODE=1`。
原目录初次扫描与复核之间有新增文件；采用11:45–11:47期间文件名稳定的快照，初次不一致计数不作最终依据。
相对于上次1880份PDF、1856篇正式笔记的文件清单，本轮快照为1897份PDF、1856篇正式笔记：新增17份PDF，未减少旧路径。
另有P13的16篇及Georgiev的1篇完整讨论稿保存在库内缓存。它们部分原文此前已存在，因此“新增17份PDF”与“17篇讨论稿”是两个不同集合。
**未正式入库不等于未读；已精读也不等于科学采用通过。** 后续再增补的文件不包含在本次冻结快照。
''')
    code('''from pathlib import Path
import json,hashlib
from IPython.display import display,Markdown
PROJECT=next(p for p in (Path.cwd(),*Path.cwd().parents) if (p/'simulation/main.cc').exists())
OUT=PROJECT/'jupyter/run_records/library_update_20261030'
r=json.loads((OUT/'research_record.json').read_text())
def table(headers,rows):
    return '\\n'.join(['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|']+['|'+'|'.join(str(x).replace('|','／').replace('\\n',' ') for x in row)+'|' for row in rows])
def link(label,path):return f'[{label}](<{path}>)'
print('稳定快照:',r['inventory']['started'],'至',r['inventory']['finished'])
print('PDF:',r['inventory']['raw_pdfs'],'正式笔记:',r['inventory']['source_notes'],'新增PDF:',len(r['inventory']['new']))
display(Markdown(table(['新增原文','对应正式笔记'],[[link(Path(x['path']).stem,x['path']),'未见同名正式笔记' if x['matching_note'] is None else link('笔记',x['matching_note'])] for x in r['inventory']['new']])))''')
    md('''## 本轮相关文献与核验层级

下表DOI都与本轮实际读取的本地正式PDF首页一致；链接是DOI网址，**未联网检查解析**。页码从本地PDF第1页算起。
前四篇复用已有精读讨论稿并回核指定原文位置；其余四篇仅初筛指定页，没有声称完成全文精读。
本轮实际查看原页图：Hong第18页、Georgiev第7页、孔誉谦等第3页。抽取了其他页的文字不等于看过其图，也不等于读过整篇论文。
''')
    code('''display(Markdown(table(['论文原题／作者／年','DOI与本地资料','本轮核验位置／状态'],[[p['title']+'<br>'+p['authors']+'，'+str(p['year']),link(p['doi'],p['url'])+'<br>'+link('原文',p['pdf'])+'<br>'+(link('既有讨论稿',p['note']) if p['note'] else '未见对应正式笔记'),p['location']+'<br>'+p['local_status']] for p in r['papers']])))''')
    md('''## 哪些内容能用于当前项目

“论文结果”和“本项目建议”分列。现有源组标签不是逐事件粒子真值；²⁴¹Am含α、γ和衰变电子。借用论文中的分类或重建框架，不会自动补出实验真值。
''')
    code('''for p in r['papers']:
    display(Markdown('### '+p['authors'].split('、')[0]+'（'+str(p['year'])+'）：'+p['priority']+'\\n\\n'+
        '**论文证据：** '+p['finding']+'\\n\\n**适用限制：** '+p['limits']+'\\n\\n**本项目建议：** '+p['application']+'\\n\\n'+
        link('原文',p['pdf'])+'；'+(link('讨论稿',p['note']) if p['note'] else '本轮仅定向初筛')+'。'+p['review']))''')
    md('''Hong的表7显示，匹配扰动后的概率温度校准改善ECE/NLL，但三行准确率均为60.72%；它支持区分校准与预测任务，不能证明本项目改一次读出参数便能消除PID带宽差异。已有讨论稿还指出二分类温度缩放与E-AURC变化的排序一致性疑点；本项目建议不使用该列作证据。

Georgiev的稳定模板线性叠加、记录末端信号归零及同模板生成测试，是迁移时必须保留的条件。CsI粒子相关形状和长尾会影响这些条件，统一反卷积不能直接充当物理模型验证。

孔誉谦等的“4–6点”经验区不直接采用：原印时间常数定义和100 MHz退化组合尚有缺口，量化噪声也未纳入。采用的是**检查采样相位敏感性这一方向**。

## 与既有读出证据及采用意见的关系

已有Calò综述仍更直接地支撑“读出电路可以改变波形”的讨论，但本轮只复用其完整笔记。新微观雪崩论文并未为本项目确定电路、并联单元数、PDE或温度，不能据此给参数一个唯一物理解释。
''')
    code('''p=r['old_readout_evidence']
display(Markdown(p['title']+'；'+p['authors']+'，'+str(p['year'])+'；'+link(p['doi'],'https://doi.org/'+p['doi'])+'。\\n\\n'+link('既有完整笔记',p['note'])+'；'+link('原文',p['pdf'])+'。'+p['scope']+' '+p['relevance']+' '+p['review']))
display(Markdown(r['review_boundary']+' '+link('原始审核记录','/media/sf_works/papers/global-review.md:13461')))
display(Markdown('此前联网候选的本地缺口：'+r['gap_scope']))
display(Markdown(table(['题名','DOI（沿用前次核验）'],[[p['title'],link(p['doi'],'https://doi.org/'+p['doi'])] for p in r['previous_web_candidates']])))''')
    md('''## 下一步的具体安排（建议，尚未执行）

1. **先建立读出状态内的校准与内部检查。** 冻结现有first60外推结果；若使用后段数据，明确标为新条件校准。按原始事件连续块留出，并在64/128块长下检查结果敏感性，避免同一母波形或相邻重复记录跨组。状态划分及阈值只在校准块确定；前触发状态不足以解释信号期变化，不能仅按RMS拟合后宣称完成校正。
2. **增加亚采样相位与仅前沿定位两个诊断对照。** 先按固定原Q检查估计t10样点内相位、上升沿和PID残差的关联；使用原生模拟中的已知时间控制检查因果响应。任何插值都应报告对噪声协方差的改变；原ADC积分、固定事件ID及原Q作为共同参照。模板只用于定位/残差比较，生成与检查使用不同事件块，不以重新对齐后的单一指标变窄宣称物理带宽问题已经解决。
3. **再比较信号期响应候选。** 对有限个预先列明的读出传递函数变化、低频形状波动与短尺度残差候选，联合检查均ADC、三种PID的中心和完整协方差、F50/102/202及额外尾部。分量比例波动与读出极点变化仍是竞争解释。校准选择和不确定度应在各训练块内传播，模型冻结后再用独立Geant4母事件检查模拟统计；增加重放不能增加独立输运历史。

本轮没有改变默认模型或正式PID，也没有把新论文的FOM、皮秒分辨率或分类准确率移作本项目性能。现有资料尚不能证明PID带宽仅由电子白噪声造成，也不能把经验信号残差直接命名为CsI本征涨落。

## 可复核记录

本笔记包含题录、采用边界和建议；所有来源路径及SHA-256写入同目录下的 `run_records/library_update_20261030/research_record.json`。库内缓存全程只读，新文本和页面图仅写入本项目 `.cache/paper-search/2026-10-30-library-update/`。
''')
    code('''checks=[]
for name,digest in r['input_sha256'].items():
    path=Path(name)
    checks.append(path.exists() and hashlib.sha256(path.read_bytes()).hexdigest()==digest)
assert all(checks)
assert all(Path(p['pdf']).exists() and (p['note'] is None or Path(p['note']).exists()) for p in r['papers'])
print('来源与项目结果哈希核验通过:',len(checks),'项')
print('本轮定向核查:',len(r['papers']),'篇；新实验事件=0；新Geant4母事件=0。')''')
    nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.13.5'}})
    shell=InteractiveShell.instance()
    previous=Path.cwd()
    os.chdir(PROJECT)
    try:
        count=0
        for cell in nb.cells:
            if cell.cell_type!='code':continue
            count+=1
            with capture_output() as captured:
                result=shell.run_cell(cell.source,store_history=False)
            if result.error_before_exec or result.error_in_exec:
                raise RuntimeError(str(result.error_before_exec or result.error_in_exec))
            outputs=[]
            if captured.stdout:outputs.append(nbf.v4.new_output('stream',name='stdout',text=captured.stdout))
            if captured.stderr:outputs.append(nbf.v4.new_output('stream',name='stderr',text=captured.stderr))
            for item in captured.outputs:
                outputs.append(nbf.v4.new_output('display_data',data=item.data,metadata=item.metadata))
            cell.outputs=outputs;cell.execution_count=count
    finally:
        os.chdir(previous)
    nbf.validate(nb)
    nbf.write(nb,NOTEBOOK)
    save(OUT/'verification.json',dict(executed_code_cells=count,cells=len(nb.cells),papers=len(papers),
        original_PDF_DOIs_matched=8,selected_PDFs_unchanged=True,
        input_count=len(inputs),notebook_sha256=sha(NOTEBOOK),record_sha256=sha(OUT/'research_record.json'),
        builder_sha256=sha(__file__),note='No new study, library write, or online verification.'))
    print(NOTEBOOK, 'written;',count,'executed cells;',len(inputs),'hashed inputs.',flush=True)


if __name__=='__main__':
    main()
