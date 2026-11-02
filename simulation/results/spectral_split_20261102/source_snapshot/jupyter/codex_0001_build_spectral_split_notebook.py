"""Build and execute native complementary spectral-state research notes."""
import os
from pathlib import Path
PROJECT=Path(__file__).resolve().parents[1]
cache=PROJECT/'.cache/codex_0001_spectral_split_ipython';cache.mkdir(parents=True,exist_ok=True)
os.environ['IPYTHONDIR']=str(cache);os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import nbformat as nbf
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output


def main():
    out=PROJECT/'simulation/results/spectral_split_20261102'
    assert (out/'decision.json').exists() and not (out/'record.json').exists()
    cells=[]
    def md(x):cells.append(nbf.v4.new_markdown_cell(x))
    def code(x):cells.append(nbf.v4.new_code_cell(x))
    md('''# CsI–SiPM：保持平均谱的成分方差模型与原生检验

2026-11-02。承接[完整母事件池选择核查](codex_0001_csi_full_pool_selection.ipynb)：原队列截断并没有消除F202过宽，且全局事件方差CV置零会使Co的F50过窄。本轮实现一个最小的新读出模型：让互补成分的事件方差分别变化，保留平均噪声协方差、平均响应和原Geant4雪崩历史。

实现位于`simulation/include/SpectralVarianceSplit.hh`与`simulation/src/SpectralVarianceSplit.cc`，经已有`SignalDependentNoise`进入`csi_sources`和原生重放程序。Python只负责矩预测、校准、提取和统计，最终比较波形由C++读出生成。新选项默认关闭，旧配置不增加新随机流。

现有256000个母事件全部参与当前选择：Co64000、Am192000。原响应r0与零CV r0共512000条控制读出必须与上一轮ADC逐样点相同；候选r0/r1共512000条读出用于检查。**新增独立Geant4输运及实验事件均为0**。旧实验段和MC已参与探索，不能称为盲验证。
''')
    code('''from pathlib import Path
import json
import numpy as np
from IPython.display import display,Markdown,Image
PROJECT=next(p for p in (Path.cwd(),*Path.cwd().parents) if (p/'simulation/main.cc').exists())
OUT=PROJECT/'simulation/results/spectral_split_20261102'
def read(path):return json.loads(Path(path).read_text())
def fmt(x):return '不可估计' if x is None else f'{x:.4g}'
def ci(x):return '不可估计' if x['CI95'] is None else '['+', '.join(fmt(v) for v in x['CI95'])+f"] ({x['valid_draws']}/512)"
def table(head,rows):return '\\n'.join(['|'+'|'.join(head)+'|','|'+'|'.join(['---']*len(head))+'|']+['|'+'|'.join(map(str,r))+'|' for r in rows])
p=read(OUT/'protocol.json');fit=read(OUT/'fit.json');a=read(OUT/'analysis.json');d=read(OUT/'decision.json')
display(Markdown('**本轮结论：**'+d['conclusion_zh']))
for finding in d['findings_zh']:display(Markdown(finding))''')
    md(r'''## 1. 互补成分：保持平均谱，改变事件间高阶起伏

设原有独立标准高斯创新为$w_j$，另外生成与其独立的$v_j$。对预定整数延迟$d$，定义

\[
H_j=\frac{-w_{j-d}+2w_j-w_{j+d}-v_{j-d}+v_{j+d}}4,
\qquad L_j=w_j-H_j.
\]

对所有时滞，$L$与$H$的互协方差为0，且它们联合高斯，因此相互独立。其谱为

\[
P_H(\omega)=\sin^2(\omega d/2),\quad P_L(\omega)=\cos^2(\omega d/2),\quad P_H+P_L=1.
\]

新创新为$L+\sqrt{M_H}H$，其中$M_H$是独立、正、均值为1的对数正态方差乘子。随后使用原来的FIR、无噪声信号包络$\sqrt{v\max(\mu_i,0)}$，以及均值为1的全局方差乘子$M_C$。由于$E[M_H]=E[M_C]=1$，原有二阶协方差和平均谱保留，但事件间方差及四阶矩可以变化。

**差分成分有频率旁瓣，不是互不重叠的物理高频带。** CV也不是实测SiPM增益或光产额波动。未按事件强制固定总功率，未减去每个事件的噪声均值，也未对生成ADC做幅度缩放或注入实测脉冲。边缘使用独立随机延拓，没有周期回绕。

当新增CV为0时，新函数直接返回原创新，保持原随机数和ADC精确不变。新增边缘、辅助创新及方差状态分别使用独立种子域，不消耗输运或雪崩随机数。
''')
    code('''display(read(OUT/'algebra_tests.json'))
display(read(OUT/'build_tests.json'))
rows=[]
for source,g in fit.items():
    for ef,r in g['folds'].items():
        c=r['candidates'][r['chosen_index']]
        rows.append([source,ef,c['delay_samples'],2*c['delay_samples'],fmt(c['common_variance_cv']),fmt(c['high_variance_cv']),len(r['training_data_ids']),len(r['evaluation_data_ids'])])
display(Markdown(table(['源','评价折','延迟/样点','延迟/ns','全局方差CV','差分方差CV','校准实测数','评价实测数'],rows)))
display(Image(filename=str(OUT/'figures/spectral_partition.png')))''')
    md(r'''## 2. 先冻结矩预测和边界，再选候选

主要条件仍为last20、块128。每个评价折仅用另一折的实测F50/F102/F202完整协方差确定新增参数。原MC队列上16次零CV读出先分别计算箱内加权协方差，再平均，作为$C_0$。原半Q箱目标及所有支持箱保留。

在原零CV r0锚点和无噪声脉冲均值分母下，令$Q_i$表示某一Gaussian噪声成分独自产生的三个方差统计。用16组独立创新计算

\[
G=\frac1{N_Q}\sum_q\sum_{i\in q}w_i(1-w_i)\,E[Q_iQ_i^T].
\]

$w_i(1-w_i)$对应有限母事件加权方差的期望，保留加权均值估计的扣除项。对全谱及差分成分分别得到$G_{all},G_H$，从而预测

\[
C(F)\simeq C_0+aG_{all}+bG_H,\qquad
a=CV_C^2,\quad b=(1+a)CV_H^2.
\]

预先固定延迟为1、4、16样点，$0\le a\le CV_{原}^2$、$0\le b\le4$。以实测SD乘积规范化六个上三角协方差残差，做有界最小二乘；保留所有候选，最小目标选择延迟。未根据新原生结果扩展边界、重新选择或重新拟合。前8/后8组创新的矩预测作为敏感性记录，不当成独立母事件。

这些矩预测固定队列、锚点和分母，忽略新读出引起的定位、随机归一化、Q与质量重选，**不能代替原生ADC结果**。所有$G$的对角线非负，所以在该近似中增加非负CV无法把方差降到$C_0$以下；若零CV下限已偏宽，必须如实保留这一限制。
''')
    code('''rows=[]
for source,g in fit.items():
    for ef,r in g['folds'].items():
        for i,c in enumerate(r['candidates']):
            rows.append([source,ef,c['delay_samples'],i==r['chosen_index'],fmt(c['objective']),*[fmt(x) for x in c['predicted_SD_ratio']]])
display(Markdown(table(['源','评价折','延迟/样点','选中','协方差目标','预测F50比','预测F102比','预测F202比'],rows)))
rows=[]
for source,g in fit.items():
    for ef,r in g['folds'].items():rows.append([source,ef,*[fmt(x) for x in r['zero_CV_SD_floor']]])
display(Markdown(table(['源','评价折','零CV F50下限','零CV F102下限','零CV F202下限'],rows)))
display(Image(filename=str(OUT/'figures/spectral_proposal.png')))''')
    md('''## 3. 完整母事件池的实际读出结果

`replay_spectral_pool.cc`遍历全部原母事件，不按旧Q、能量或真值预筛选。平均响应、电子噪声、光学和SiPM雪崩参数固定，只应用本轮冻结的信号噪声方差候选。原参数两个r0控制与上一轮完整池的ADC逐点比较，任何差异都会使程序失败。

候选按实际ADC重新定位并应用原全批次质量参考、幅度≥200、t10≥384和完整积分门要求，再按当前Q进入原源支持范围。Co支持6箱，Am支持3箱，半箱质量目标和PCA不变。所有使用的Q箱必须有足够事件，不能挑选表现好的箱或删除尾部。

实验仍用原512组连续块计数；此前已扩充的MC母事件计数逐项保留，新进入母事件用剩余类别的条件多项分布继续补入。同一母事件的模型与重复读出始终一起抽样。原当前/零CV控制的全部512组统计继续闭合上一轮。r0与r1分别报告，两个流不能作为已收敛的噪声平均。
''')
    code('''rows=[]
for source,g in a.items():
    for v,r in g['variants'].items():
        count=sum(x['supported'] for x in g['selection'][v]);enter=sum(x['enter_vs_current'] for x in g['selection'][v]);leave=sum(x['leave_vs_current'] for x in g['selection'][v])
        rows.append([source,v,count,enter,leave,*[fmt(r['point'][f'F{s}_SD_ratio']) for s in [50,102,202]],ci(r['CI95']['F202_SD_ratio'])])
display(Markdown(table(['源','模型/读出','支持母事件','比当前移入','比当前移出','F50 SD比','F102 SD比','F202 SD比','F202条件95%范围'],rows)))
display(Image(filename=str(OUT/'figures/spectral_native_scales.png')))
rows=[]
for source,g in a.items():
    for r in [0,1]:
        x=g['variants'][f'split_r{r}']['change_vs_current'];rows.append([source,r,*[ci(x['CI95'][f'F{s}_SD_ratio']) for s in [50,102,202]]])
display(Markdown(table(['源','读出','F50配对变化范围','F102配对变化范围','F202配对变化范围'],rows)))''')
    md('''## 4. 联合检查，而不只追求F202变窄

预先设定的数值检查包括：F三尺度的均值与SD比在0.9–1.1，三个PID的SD比在0.9–1.1、中心50%/95%宽度比在0.85–1.15；平均ADC相对L2不得比同一读出原响应恶化超过0.002，PID规范化协方差误差不得恶化超过0.02，并要求F完整协方差误差改善。两个预定读出都报告。它们是候选筛查阈值，不是统计显著性检验或自动替换默认参数的依据。

平均ADC保持绝对幅度，曲线横坐标为整数对齐锚点；统计变化可来自随机实现与当前选择。F202是波形波动指标，不等同于所有PID宽度。各折自己的实测评价结果保存在`fold_checks`中，不能只依赖跨折汇总。

本轮Co在两个流的汇总门槛中通过，但分折F50点估计仍在0.869–1.148之间波动；因此汇总通过不等于每折稳定或已经证明等效。Am两个流均因F202 SD未通过，仍保留约16%–18%的长尺度过宽。
''')
    code('''display(Image(filename=str(OUT/'figures/spectral_native_pid.png')))
display(Image(filename=str(OUT/'figures/spectral_native_mean_adc.png')))
rows=[]
for source,g in a.items():
    for r in [0,1]:
        x=g['variants'][f'split_r{r}']['point'];guard=g['joint_guards'][str(r)]
        rows.append([source,r,fmt(x['mean_ADC_relative_L2']),fmt(x['PID_covariance_relative_error']),fmt(x['F_covariance_relative_error']),guard['passed'],', '.join(guard['failed']) or '无'])
display(Markdown(table(['源','读出','均ADC L2','PID协方差误差','F协方差误差','联合门槛通过','未通过项'],rows)))
rows=[]
for source,g in a.items():
    for ef,variants in g['fold_checks'].items():
        for v in ['split_r0','split_r1']:rows.append([source,ef,v,*[fmt(variants[v][f'F{s}_SD_ratio']) for s in [50,102,202]]])
display(Markdown(table(['源','评价折','候选','F50比','F102比','F202比'],rows)))
for limit in d['limitations_zh']:display(Markdown('边界：'+limit))
display(Markdown('**下一步：**'+d['next_work_zh']))''')
    md('''## 5. 复现和封存

入口：`prepare_spectral_split.py` → `fit_spectral_split.py` → `run_spectral_split.py` → `extract_spectral_split.py` → `analyze_spectral_split.py` → `report_spectral_split.py` → 本笔记生成器 → `verify_spectral_split.py`及`--archive`。先在新的构建目录编译Geant4应用、重放程序和测试。完整重跑需要新结果目录，已有输出拒绝静默覆盖。

`protocol.json`冻结模型、目标、输入、边界和后续联合检查。准备过程中发现旧compact点数组不保存t10，改为读取已封存完整池的同一零CV r0锚点；该字段修正在任何校准完成前进行，保留`pre_fit_protocol_v1.json`和`execution_notes.json`，模型与目标不变。`native_protocol.json`在新原生输出前冻结候选文件与种子，`analysis_protocol.json`在统计前冻结分析代码。

首次统计完成512组抽样和旧结果闭合后，联合检查的NumPy布尔值无法直接写入JSON。修正仅将检查值转为Python布尔值；原分析方案、源码和16个数值产物的哈希分别保存在`pre_export_analysis_protocol_v1.json`、`pre_export_analyze_spectral_split_v1.py`、`pre_export_outputs.json`中。最终核验要求这16个数值文件逐字节不变，没有据此更改模型、目标、门槛或随机种子。

验证包括互补算子的精确协方差、40000次独立矩模拟、7项原生测试、有界最优性、512000条旧控制ADC精确闭合、所有候选共同范围事件的原ADC特征重算、随机/边界母事件质量检查、原抽样闭合及已执行笔记。历史结果保留原样，其旧程序版本在各自源码快照中；新可选模型默认关闭。共享文献库未修改，本轮未联网，也未提交或push新增改动。
''')
    nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}})
    shell=InteractiveShell.instance()
    for c in nb.cells:
        if c.cell_type!='code':continue
        with capture_output() as captured:result=shell.run_cell(c.source,store_history=True)
        if result.error_before_exec or result.error_in_exec:raise RuntimeError(result.error_before_exec or result.error_in_exec)
        c.execution_count=shell.execution_count-1;outputs=[]
        if captured.stdout:outputs.append(nbf.v4.new_output('stream',name='stdout',text=captured.stdout))
        if captured.stderr:outputs.append(nbf.v4.new_output('stream',name='stderr',text=captured.stderr))
        for o in captured.outputs:outputs.append(nbf.v4.new_output('display_data',data=o.data,metadata=o.metadata))
        c.outputs=outputs
    nbf.validate(nb);path=PROJECT/'jupyter/codex_0001_csi_spectral_variance_split.ipynb';nbf.write(nb,path);print(path)


if __name__=='__main__':main()
