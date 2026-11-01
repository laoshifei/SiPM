"""Build and execute the fixed-parameter PID core/tail research note."""
import os
import argparse
from pathlib import Path
PROJECT=Path(__file__).resolve().parents[1]
cache=PROJECT/'.cache/codex_0001_pid_core_tail_ipython';cache.mkdir(parents=True,exist_ok=True)
os.environ['IPYTHONDIR']=str(cache);os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import nbformat as nbf
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output


def main(refresh=False):
    path=PROJECT/'jupyter/codex_0001_csi_pid_core_tail.ipynb';assert refresh or not path.exists()
    cells=[]
    def md(s):cells.append(nbf.v4.new_markdown_cell(s))
    def code(s):cells.append(nbf.v4.new_code_cell(s))
    md('''# CsI(Tl)–SiPM：PID核心、尾部与波形关联

2026-10-30，承接[连续RC与联合噪声校准](codex_0001_csi_refined_rc.ipynb)。按用户要求先完成英文提交，再继续本轮诊断。
**本轮不重新拟合模拟、不删除事件；新增Geant4母事件和原生重放均为0。** 全部实验此前已探索，因此是固定参数下的内部条件检查。
''')
    code('''from pathlib import Path
import json,sys
import numpy as np
from IPython.display import display,Markdown,Image
PROJECT=next(p for p in (Path.cwd(),*Path.cwd().parents) if (p/'simulation/main.cc').exists())
OUT=PROJECT/'simulation/results/pid_core_tail_20261030'
def read(p):return json.loads(Path(p).read_text())
def table(headers,rows):
    return '\\n'.join(['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|']+['|'+'|'.join(map(str,r))+'|' for r in rows])
def fmt(x,n=3):return '不可估计' if x is None else f'{x:.{n}f}'
def nums(x):return ' / '.join(fmt(v) for v in x)
def ci(r):return '不可估计' if r['CI95'] is None else '['+', '.join(fmt(v) for v in r['CI95'])+']'
p=read(OUT/'protocol.json');e=read(OUT/'diagnostics.json');decision=read(OUT/'decision.json')
methods=['QDC','negative_mean_time','PCA'];method_labels=['QDC','平均时间得分−t','PCA']
labels={'frozen':'原冻结','coarse_spectrum':'粗RC+扩展谱','mean_only':'连续RC/继承噪声','inherited_CV_fit':'连续RC+原谱/CV选择','spectrum_CV_fit':'连续RC+扩展谱/CV选择'}
print('继续工作前已提交:',p['checkpoint_commit'])
print('不同原母事件470；原Q支持内456；新增母事件/重放均0。')
display(Markdown(decision['conclusion_zh']))''')
    md('''## 1. 数据、权重与诊断边界

保留run3（⁶⁰Co源）和run12（²⁴¹Am源）的原主带事件、原Q、first60目标Q两分权重及冻结PCA。
Co支持6个Q箱、Am支持前3箱，模拟支持区分别276/180个原母事件。Am含α、γ和衰变电子，实验主带没有逐事件粒子真值。
first60/middle20/last20各段内部以`(原事件ID // 块长) % 2`分两折。块128主分析、块64敏感性分析，MC沿用已有母事件折分及每折对应响应。

每个Q箱/评价折仅从对折实验估计加权中位数及`IQR / 1.3489795`，后者只是稳健尺度单位。
同一边界应用于实验和MC；`|z|>2`与`|z|>3`是预设诊断尾部，不是假定高斯尾概率，也不是粒子筛选。
对折只隔离新增边界的估计，不能隔离此前光学、响应、PCA和Q目标的校准信息。
''')
    code('''rows=[]
for key,g in p['groups'].items():
    ref=read(OUT/'references'/f'{key}_B128.json')
    rows.append([key,len(g['ids']),len(ref['0']['training_ids']),len(ref['0']['evaluation_ids'])])
display(Markdown(table(['组','原事件数','评价折0训练数','评价折0检查数'],rows)))
print('原Q边界:',p['q_edges'])''')
    md('''## 2. 核心与外侧分位宽度

50/80/90/95%宽度分别取加权25–75%、10–90%、5–95%、2.5–97.5%分位跨度。
各Q箱宽度平方等权平均后开根号，与上一轮IQR定义一致；SD来自完整、未裁剪的箱内协方差。
`(IQR/SD)_MC / (IQR/SD)_data`用来描述核心相对整体方差的比例，不是拟合参数，也不直接等于峰度。

512次原始实验块和完整原母事件配对重采样；模拟参数、模型选择、PCA、Q目标和诊断边界均固定。
区间不包含边界及响应拟合的不确定度。没有等效性检验或多重比较校正；区间覆盖1不能证明一致。
''')
    code('''rows=[]
for key,blocks in e['groups'].items():
    for v in ['coarse_spectrum','inherited_CV_fit','spectrum_CV_fit']:
        r=blocks['128']['variants'][v]['point']
        rows.append([key,labels[v],nums([r[f'width50_{m}_ratio'] for m in methods]),nums([r[f'width95_{m}_ratio'] for m in methods]),nums([r[f'core_over_SD_{m}_ratio'] for m in methods])])
display(Markdown(table(['组','模型','QDC/−t/PCA IQR比','95%宽度比','核心/SD相对比'],rows)))
display(Image(filename=str(OUT/'figures/core_tail_quantile_widths.png')))''')
    code('''rows=[]
for B in ['128','64']:
    r=e['groups']['run12_last20'][B]['variants']['spectrum_CV_fit']
    for method,label in zip(methods,method_labels):
        for coverage in [50,80,90,95]:
            n=f'width{coverage}_{method}_ratio';rows.append([B,label,coverage,fmt(r['point'][n]),ci(r['CI95'][n])])
display(Markdown(table(['块长','Am晚段指标','覆盖比例/%','MC/实测宽度','条件95%区间'],rows)))''')
    md('''## 3. 尾部质量和方差贡献

每种得分分别划分尾部。方差贡献使用各评价组自己的Q箱均值作中心：
`E[(score−箱均值)² × 尾部指示] / E[(score−箱均值)²]`，最后按原定义汇总各箱二阶矩。
分母没有剔除尾部，核心贡献与尾部贡献严格相加为全方差。另存“任一指标越界”的联合尾部协方差分解。

原始尾部事件数也报告，不能把重复重采样或不同响应当新事件。零计数时经验bootstrap可能给出[0,0]，这不证明尾部率为零，更不是实验粒子泄漏率上限。
''')
    code('''rows=[]
for source,run in [('Co60',3),('Am241',12)]:
    g=e['groups'][f'run{run}_last20']['128']
    for v in ['data','inherited_CV_fit','spectrum_CV_fit']:
        r=g['data'] if v=='data' else g['variants'][v]['native']
        for k,bound in enumerate([2,3]):
            rows.append([source,'实测' if v=='data' else labels[v],bound,r['supported_events'],r['tail_event_counts'][k],nums(100*np.array(r['tail_probability'][k])),nums(100*np.array(r['tail_variance_share'][k]))])
display(Markdown(table(['晚段源','组','尺度边界','原支持数','QDC/−t/PCA尾部数','加权尾部占比/%','方差贡献/%'],rows)))
display(Image(filename=str(OUT/'figures/core_tail_mass_and_variance.png')))''')
    code('''r=e['groups']['run12_last20']['128']['variants']['spectrum_CV_fit'];rows=[]
for m,label in zip(methods,method_labels):
    for stem,title in [('tail2','尾部质量差/百分点')]:
        n=f'{stem}_{m}_mass_delta_pp';rows.append([label,title,fmt(r['point'][n]),ci(r['CI95'][n])])
    n=f'tail2_{m}_variance_share_delta_pp';rows.append([label,'尾部方差贡献差/百分点',fmt(r['point'][n]),ci(r['CI95'][n])])
display(Markdown(table(['Am晚段','MC−实测','点估计','条件95%区间'],rows)))
display(Image(filename=str(OUT/'figures/core_tail_distributions.png')))
rows=[]
for source,run in [('Co60',3),('Am241',12)]:
    g=e['groups'][f'run{run}_last20']['128']
    for v in ['data','coarse_spectrum','inherited_CV_fit','spectrum_CV_fit']:
        r=g['data'] if v=='data' else g['variants'][v]['native']
        rows.append([source,'实测' if v=='data' else labels[v],nums(100*np.array(r['histogram']['outside_range_mass']))])
display(Markdown(table(['源','组','图示±5范围外QDC/−t/PCA质量/%'],rows)))''')
    md('''## 4. 尾部对应的波形与起伏

每条波形的76个40 ns门电荷份额之和为1。对每个原Q箱，以该评价组的得分均值及波形份额均值居中，计算交叉协方差，再汇总并除以完整得分SD。
虚线为`|z|>2`事件对该协方差的贡献，实线为全体事件；虚线不是“尾部事件的平均波形”。核心与尾部贡献严格相加。

另列PID与前128点基线RMS对数、F50/102/202的箱内相关。PID与F共享样点，门电荷份额还共享归一化，因此这些关联不能单独识别物理机制。
''')
    code('''display(Image(filename=str(OUT/'figures/core_tail_waveform_covariance.png')))
display(Image(filename=str(OUT/'figures/core_tail_noise_associations.png')))''')
    md('''## 5. 现有Geant4祖先标签检查

从原生输运ROOT重新读取CsI能量沉积及雪崩电荷的祖先贡献。祖先类别为unknown、α、γ、衰变电子、其他等；类别不是当前沉积粒子的种类。
某类占总贡献≥90%才称该家族主导，另以能量95%及雪崩电荷90%检查定义敏感性。
雪崩电荷是原窗口中实际雪崩电荷，不等于有限门ADC积分。α主导事件仍可能含次要γ/电子贡献。

在每个Q箱按能量90%主导类分解完整协方差为“类别均值之间”与“类别内”两项。
若一个支持区只有一类，类别间项为0；这只排除这组粗类别混合对已选择MC的贡献，不排除类内能损、光收集和响应变化，也不说明实验纯度。
''')
    code('''rows=[]
for source,run in [('Co60',3),('Am241',12)]:
    r=e['groups'][f'run{run}_last20']['128']['variants']['spectrum_CV_fit']['ancestry']
    for key,s in r['label_sensitivity'].items():rows.append([source,key,s['counts'],nums(s['weighted_fractions'])])
    print(source,'类间均值方差/总方差:',r['between_truth_mean_variance_fraction'])
display(Markdown(table(['源','定义','α/γ/衰变电子/混合其他/无沉积原数','对应加权份额'],rows)))
rows=[]
for source,run in [('Co60',3),('Am241',12)]:
    r=e['groups'][f'run{run}_last20']['128']['variants']['spectrum_CV_fit']['ancestry']
    for key,s in r['counts'].items():
        if s['events']:rows.append([source,key,s['events'],s['joint_tail_events'],fmt(s['tail_probability_given_class']),s['Q_bins_present']])
display(Markdown(table(['源','能量90%类','支持数','任一PID越界原数','类内加权联合尾部率','存在Q箱'],rows)))''')
    md('''## 6. 判断与后续工作
''')
    code('''display(Markdown(decision['conclusion_zh']))
for finding in decision['findings_zh']:display(Markdown(finding))
display(Markdown(decision['next_work_zh']))
for limit in decision['limitations_zh']:display(Markdown('边界：'+limit))
print('默认配置已替换:',decision['default_changed'])
print('复现: prepare_pid_core_tail.py → analyze_pid_core_tail.py → plot_pid_core_tail.py → summarize_pid_core_tail.py')
print('笔记: jupyter/codex_0001_build_pid_core_tail_notebook.py；核验: verify_pid_core_tail.py，随后 --archive')
print('完整结果目录:',OUT)''')
    md('''`protocol.json`在新数值检查之前固定方案、输入哈希和提交号；`references/`保存每折训练/评价ID及边界。
`diagnostics.json`含逐Q箱结果、宽度区间、精确方差和波形协方差分解；`bootstrap.npz`保留配对抽样；`validation_table.csv`为60行全候选数值表。
最后核验IQR/SD点估计及区间与上一轮逐项一致，并直接从事件重算尾部质量和方差、检查祖先能量/电荷守恒、执行笔记和嵌入图。
`verification.json`与`record.json`封存结果及源码哈希。复跑使用新目录，不覆盖当前记录。共享文献库未修改，本轮未联网或新增实验。
''')
    nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}})
    shell=InteractiveShell.instance()
    for c in nb.cells:
        if c.cell_type!='code':continue
        with capture_output() as capture:r=shell.run_cell(c.source,store_history=True)
        if r.error_before_exec or r.error_in_exec:raise RuntimeError(r.error_before_exec or r.error_in_exec)
        c.execution_count=shell.execution_count-1;outputs=[]
        if capture.stdout:outputs.append(nbf.v4.new_output('stream',name='stdout',text=capture.stdout))
        if capture.stderr:outputs.append(nbf.v4.new_output('stream',name='stderr',text=capture.stderr))
        for o in capture.outputs:outputs.append(nbf.v4.new_output('display_data',data=o.data,metadata=o.metadata))
        c.outputs=outputs
    nbf.validate(nb);nbf.write(nb,path);print(path)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--refresh',action='store_true');args=ap.parse_args();main(args.refresh)
