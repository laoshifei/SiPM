"""Build and execute the frozen-event Gaussian closure research notebook."""
import os
from pathlib import Path
PROJECT = Path(__file__).resolve().parents[1]
cache = PROJECT/'.cache/codex_0001_gaussian_closure_ipython'
cache.mkdir(parents=True, exist_ok=True)
os.environ['IPYTHONDIR'] = str(cache)
os.environ['OPENBLAS_NUM_THREADS'] = '1'; os.environ['OMP_NUM_THREADS'] = '1'
import nbformat as nbf
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output


def main():
    path = PROJECT/'jupyter/codex_0001_csi_gaussian_closure.ipynb'
    assert not (PROJECT/'simulation/results/gaussian_closure_20261102/record.json').exists()
    assert not (PROJECT/'simulation/results/gaussian_closure_20261102/verification.json').exists()
    cells = []
    def md(s): cells.append(nbf.v4.new_markdown_cell(s))
    def code(s): cells.append(nbf.v4.new_code_cell(s))
    md('''# CsI–SiPM：时间协方差能否解释F202带宽差异？

2026-11-02。先完成英文提交 **94aa3ba**（`Decompose CsI QDC tails and fluctuation variance residuals`），并已push、核对GitHub main与本地SHA一致；此前的独立Geant4统计提交d481527也一并推送。本笔记是该提交之后的继续研究。

承接[QDC尾部与F202残差分解](codex_0001_csi_qdc_fluctuation_residuals.ipynb)：晚段扩展谱的Co/Am F202事件SD比为1.232/1.201，超额方差主要位于信号窗。这里进一步问：**这些差异是否已由完整波形向量的均值、时间协方差解释，还是还有二阶闭合之外的差异？**

全部原ADC、事件ID、Q支持、目标权重、三个既有响应及PID定义固定。新增Geant4母事件、读出重放、实验事件均为0，没有根据本轮结果调参或剔除事件。
''')
    code('''from pathlib import Path
import json
import numpy as np
from IPython.display import display,Markdown,Image
PROJECT=next(p for p in (Path.cwd(),*Path.cwd().parents) if (p/'simulation/main.cc').exists())
OUT=PROJECT/'simulation/results/gaussian_closure_20261102'
def read(path):return json.loads(Path(path).read_text())
def fmt(x):return '不可估计' if x is None else f'{x:.3f}'
def ci(x):return '不可估计' if x['CI95'] is None else '['+', '.join(fmt(v) for v in x['CI95'])+f"] ({x['valid_draws']}/512)"
def table(head,rows):return '\\n'.join(['|'+'|'.join(head)+'|','|'+'|'.join(['---']*len(head))+'|']+['|'+'|'.join(map(str,r))+'|' for r in rows])
p=read(OUT/'protocol.json');e=read(OUT/'analysis.json');decision=read(OUT/'decision.json')
variants=p['variants'];labels={'frozen':'原冻结','inherited_CV_fit':'RC+原谱','spectrum_CV_fit':'RC+扩展谱'}
def rec(g,v='spectrum_CV_fit',cond='Q_half',norm='event_mu',scale=202):return g['cohorts'][v]['contrasts'][cond][norm][str(scale)]
print('方案冻结提交:',p['checkpoint_commit'],'已核对输入:',len(p['inputs_sha256']))
display(Markdown(decision['conclusion_zh']))''')
    md(r'''## 1. 从原ADC到一个精确的二次型

保留原25/51/101点中心滑动均值高通残差（50/102/202 ns）、原`t10`定位和前200点/后1000点窗口。每个窗内先减去该窗均值，记残差为$r_{\rm post},r_{\rm pre}$，则

\[
X=\left(\frac{r_{\rm post}}{\sqrt{999\mu}},\frac{r_{\rm pre}}{\sqrt{199\mu}}\right),\quad
A=\operatorname{diag}(+1_{1000},-1_{200}),\quad F=X^TAX.
\]

分母$\mu$为该事件信号窗平均ADC。这个等式与原`ddof=1`、`F=(Vpost−Vpre)/mu`逐事件完全相等。固定分母对照只将μ换成父记录中对折实测的同Q箱均值μ₀，不改ADC或正式PID。

若X服从同一多元Gaussian分布，均值m、协方差C、二阶矩M=C+mmᵀ，其二次型方差由一、二阶矩决定：

\[
V_G=2\operatorname{tr}(ACAC)+4m^TACAm
=2\{\operatorname{tr}(AMAM)-(m^TAm)^2\}.
\]

这里保留完整时间矩阵和前/后窗交叉项，无平稳性、白噪声或对角协方差假设。对实际非Gaussian向量也计算此闭合量，再把`实际方差−闭合方差`称作**余项**。余项可能来自高阶累积量、信号/幅度/响应混合、选择约束或事件相关性，不等于已识别的某一种物理噪声。
''')
    code('''rows=[]
for run in [3,12]:
    g=e['groups'][f'run{run}_last20']['128']
    for v in variants:
        r=rec(g,v)
        for name in ['observed_SD_ratio','gaussian_SD_ratio','gaussian_excess_over_data','remainder_excess_over_data']:
            rows.append([run,labels[v],name,fmt(r['point'][name]),ci(r['CI95'][name])])
display(Markdown(table(['晚段run','响应','F202指标','点值','条件95%重抽样范围（有效次数）'],rows)))
display(Image(filename=str(OUT/'figures/gaussian_excess_decomposition.png')))''')
    md('''所有方差仍为**原Q箱内方差，再在支持箱之间等权**。主分析的Gaussian闭合在Q半箱内计算，加回与实测直接方差相同的半箱均值差项；这个均值项含有限样本噪声，不是预测出来的物理漂移。

主分析保留响应折混合，细分“Q半箱×响应折”是敏感性检查。在看新派生结果前，样本计数已显示Co晚段块128有一个3事件细格，中段还有2事件细格；这些细分整体结果不可估计，不能删掉该箱后报告剩余箱。主分析所有原支持箱保留。
''')
    code('''rows=[]
for run in [3,12]:
    g=e['groups'][f'run{run}_last20']['128']
    for norm in p['normalizations']:
        for cond in p['conditioning']:
            r=rec(g,cond=cond,norm=norm)
            rows.append([run,norm,cond,fmt(r['point']['observed_SD_ratio']),fmt(r['point']['gaussian_SD_ratio']),
                         fmt(r['point']['remainder_excess_over_data']),ci(r['CI95']['remainder_excess_over_data'])])
display(Markdown(table(['run','分母','条件格','直接SD比','Gaussian SD比','余项超额/实测方差','重抽样范围'],rows)))
display(Image(filename=str(OUT/'figures/gaussian_normalization_controls.png')))''')
    md(r'''## 2. 有限样本偏差与不同原事件的U统计

每个向量有1200个样点，而部分格只有十几个事件。直接把样本协方差代入$V_G$会引入自配对项，不能把它视为可靠的总体Gaussian方差估计。

令$K_{ij}=X_i^TAX_j$。不同事件的$E(K_{ij}^2)=\operatorname{tr}(AMAM)$，四个不同事件的$E(K_{ij}K_{kl})=(m^TAm)^2$，因此可分别用不同原ID的有序对/四元组估计这两项。无权重时记对角置零的K的总和为S、元素平方和为T、行和平方之和为R，则

\[
\widehat V_G=2\left\{\frac{T}{n(n-1)}-\frac{S^2-4R+2T}{n(n-1)(n-2)(n-3)}\right\}.
\]

该U式的总体无偏性质依赖格内独立同分布假设，真实采集数据不保证满足。点估计或某次抽样少于4个不同原ID时不可估计；负U估计不截成零，无法开平方时不报告Gaussian SD比。与父分析的有偏经验方差比较时，格内乘`(N−1)/N`，再加回完全相同的格间均值项。

沿用父分析512组实验原索引块抽样、4批完整母事件抽样的**相同计数和随机流**。U项使用这些计数作为不同原ID对/四元组的乘积权重，分母也是不同原ID的权重和；重复副本不能变成“独立事件”。通常的复制样本直接方差则保持父统计口径。

这些区间是固定参数下的加权U重抽样范围，没有证明覆盖率，也未计入拟合、系统误差或多重比较。稀疏格会令部分抽样不可估计，必须同时看有效次数。`plugin−U`只是有限样本自配对敏感性的诊断；其重抽样范围会因重复权重而偏移，**不能解读成总体偏差的置信区间**。
''')
    code('''display(Markdown('独立算法检查：'))
display(read(OUT/'algebra_tests.json'))
rows=[]
for run in [3,12]:
    g=e['groups'][f'run{run}_last20']['128']
    for v in ['data',*variants]:
        s=g['cohorts'][v]['summary']['Q_half']['event_mu']['202']
        rows.append([run,'实测' if v=='data' else labels[v],*[f'{s[k]:.6g}' if s[k] is not None else '不可估计' for k in
            ['observed','gaussian_U','gaussian_plugin','fourth_remainder']],fmt(s['plugin_bias']/s['observed'])])
display(Markdown(table(['run','队列','直接方差','不同ID闭合','直接代入闭合','余项','(直接代入−U)/直接方差'],rows)))''')
    md('''## 3. 每个电荷箱都检查，并报告稀疏格

以下逐箱比较仍用原Q半箱目标质量，Co保留6箱，Am保留前3箱。不是按本轮结果挑箱，也没有新增粒子真值标签。
''')
    code('''rows=[]
for run in [3,12]:
    g=e['groups'][f'run{run}_last20']['128']
    for c in g['cohorts']['data']['cells']:
        rows.append([run,c['Q_bin'],c['half'],c['events'],*c['fold_counts'],fmt(c['target'])])
display(Markdown(table(['run','Q箱','半箱','实测事件','折0','折1','目标质量'],rows)))
display(Image(filename=str(OUT/'figures/gaussian_charge_controls.png')))''')
    md('''## 4. 描述性的时间协方差曲线

对Q半箱内事件在每个样点减去跨事件均值，平均后窗的`Cov(t,t+lag)`，除以该格的平均零延迟方差，再按冻结半箱/Q权重汇总。使用完整原窗，但图只展示预先固定的9个延迟。

曲线接近或Gaussian SD比接近1，不代表两个完整协方差矩阵等效；只意味着所报告的投影/二次型关系接近。曲线不是平稳噪声功率谱，不直接给出器件相关时间，也不带新的拟合或误差带。
''')
    code('''display(Image(filename=str(OUT/'figures/gaussian_temporal_covariance.png')))''')
    md('''## 5. 前、中、晚采集段与块长敏感性

全部12个源/采集段/块长条件、三个已有响应、三个尺度、两种分母和两种条件划分都保留。已有实验与MC已被多次诊断，此处不是盲验证。
''')
    code('''rows=[]
for group,blocks in e['groups'].items():
    for B,g in blocks.items():
        r=rec(g)
        rows.append([group,B,fmt(r['point']['observed_SD_ratio']),fmt(r['point']['gaussian_SD_ratio']),
                     fmt(r['point']['remainder_excess_over_data']),ci(r['CI95']['remainder_excess_over_data'])])
display(Markdown(table(['组','块长','直接SD比','Gaussian SD比','余项超额','重抽样范围'],rows)))
display(Image(filename=str(OUT/'figures/gaussian_segment_controls.png')))''')
    md('''## 6. 对下一轮建模的约束

F202与直接QDC/时间PID是不同指标。余项差异提供了检验方向，不能据此宣布正式PID带宽已解决，或把实验余项强行匹配到零。主带、电荷条件、触发及粒子混合本身也可能限制波动分布，需要与模型内信号相关方差的事件间异质性一起核查。
''')
    code('''for finding in decision['findings_zh']:display(Markdown(finding))
display(Markdown(decision['next_work_zh']))
for limit in decision['limitations_zh']:display(Markdown('边界：'+limit))
print('结果目录:',OUT)
print('冻结与计算: simulation/analyze_gaussian_closure.py prepare; analyze --workers 2')
print('报告: simulation/report_gaussian_closure.py')
print('笔记: jupyter/codex_0001_build_gaussian_closure_notebook.py')
print('核验/封存: simulation/verify_gaussian_closure.py [--archive]')''')
    md('''`protocol.json`冻结方案、代码与339个输入哈希；`algebra_tests.json`记录不同有序元组的暴力枚举、直接协方差公式及Gaussian总体对照。

`kernels/`保存每个条件格的原事件索引、未归一化带符号Gram矩阵、固定分母与延迟曲线；`evaluation/`保存原512组计数、全部统计量/对比、逐Q点值与区间。`comparison_table.csv`共432行，`per_Q_table.csv`共1944行，`cell_counts.csv`共432行。`verification.json`核验独立卷积窗口、父指标逐次闭合、稀疏格传播与已执行笔记；`record.json`和源码快照封存结果。

本轮没有联网或修改共享文献库。Gaussian二次型代数在此直接推导并数值检验，不引用其他实验的成绩。重新完整运行应在新结果目录冻结方案，避免覆盖已有研究记录。
''')
    nb = nbf.v4.new_notebook(cells=cells, metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python'}})
    shell = InteractiveShell.instance()
    for c in nb.cells:
        if c.cell_type != 'code': continue
        with capture_output() as capture: result = shell.run_cell(c.source, store_history=True)
        if result.error_before_exec or result.error_in_exec: raise RuntimeError(result.error_before_exec or result.error_in_exec)
        c.execution_count = shell.execution_count-1; outputs = []
        if capture.stdout: outputs.append(nbf.v4.new_output('stream', name='stdout', text=capture.stdout))
        if capture.stderr: outputs.append(nbf.v4.new_output('stream', name='stderr', text=capture.stderr))
        for o in capture.outputs: outputs.append(nbf.v4.new_output('display_data', data=o.data, metadata=o.metadata))
        c.outputs = outputs
    nbf.validate(nb); nbf.write(nb, path); print(path)


if __name__ == '__main__': main()
