"""Build and execute charge-conditioning and multiscale covariance notes."""
import os
from pathlib import Path
PROJECT=Path(__file__).resolve().parents[1]
cache=PROJECT/'.cache/codex_0001_charge_scale_ipython';cache.mkdir(parents=True,exist_ok=True)
os.environ['IPYTHONDIR']=str(cache);os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import nbformat as nbf
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output


def main():
    out=PROJECT/'simulation/results/charge_scales_20261102'
    assert not (out/'verification.json').exists() and not (out/'record.json').exists()
    assert (out/'decision.json').exists(),'Complete report_charge_scales.py before building the notebook'
    cells=[]
    def md(x):cells.append(nbf.v4.new_markdown_cell(x))
    def code(x):cells.append(nbf.v4.new_code_cell(x))
    md('''# CsI–SiPM：电荷重新分箱与多时间尺度波动约束

2026-11-02。按用户要求，先将[上一轮固定雪崩历史的噪声重放](codex_0001_csi_noise_replay.ipynb)用英文提交为 **ddd6ca7**（`Audit CsI PID selection and fixed-history noise fluctuations`），随后push并用GitHub main提交号核对一致，再开展本轮分析。

上一轮取消事件方差CV后，Co F202宽度接近实测，但F50明显过窄。本轮先检验这个矛盾是否受电荷选择影响，再给下一轮模型提供跨尺度约束。**新增独立Geant4母事件、读出重放及实验事件均为0**；全部复用已封存的原ADC特征和16次正编号读出。

本轮不修改ADC，不拟合或升级参数，不更新正式PID或实验ROC。已用数据都参与过探索，不能称为盲验证。
''')
    code('''from pathlib import Path
import json
import numpy as np
from IPython.display import display,Markdown,Image
PROJECT=next(p for p in (Path.cwd(),*Path.cwd().parents) if (p/'simulation/main.cc').exists())
OUT=PROJECT/'simulation/results/charge_scales_20261102'
def read(p):return json.loads(Path(p).read_text())
def fmt(x):return '不可估计' if x is None else f'{x:.4g}'
def ci(x):return '不可估计' if x['CI95'] is None else '['+', '.join(fmt(v) for v in x['CI95'])+f"] ({x['valid_draws']}/512)"
def table(head,rows):return '\\n'.join(['|'+'|'.join(head)+'|','|'+'|'.join(['---']*len(head))+'|']+['|'+'|'.join(map(str,r))+'|' for r in rows])
p=read(OUT/'protocol.json');points=read(OUT/'point_analysis.json');repeats=read(OUT/'repeat_analysis.json');d=read(OUT/'decision.json')
display(Markdown('**本轮结论：**'+d['conclusion_zh']))
for x in d['findings_zh']:display(Markdown(x))
print('冻结提交:',p['checkpoint_commit'],'核对输入:',len(p['inputs_sha256']))''')
    md(r'''## 1. 明确区分原电荷与当前电荷

实测原Q就是当前波形的积分电荷；模拟之前统一继承冻结ADC的原Q，响应或噪声改变后仍不重新分箱。现在固定原事件队列、响应折、Q范围、Q半箱物理阈值和目标质量，比较三种条件：

|条件|参与事件|分箱和半箱权重|
|---|---|---|
|original_Q|原支持Q箱中的原事件|沿用原Q|
|retained_original_Q|当前Q仍在该源支持内，同时保留原Q支持|按原Q重权，用于检查移出支持的影响|
|current_Q|原队列中当前Q落入该源支持的事件|按当前Q重权，也允许原先处于非支持箱的已有事件进入|

Co支持原6箱，Am支持前3箱。上下溢出单列，最高总电荷边界包含，不能把越界值强行夹到边箱。每个原Q支持箱及正目标半箱都必须存在；某次抽样缺格则整体不可估计，不能删除该箱。

积分门1520个整数ADC样点，基线为前128点整数ADC的均值，故$Q=\sum ADC-1520\sum ADC_{pre}/128$严格落在1/8 ADC网格。由76个20点均值恢复Q后，仅恢复这一精确网格，并要求浮点残差<10⁻⁷，避免中位数阈值恰好相等时被舍入误差分到另一半箱。所有正编号读出的恢复Q与原生保存的`qcalc`逐条相等；实测恢复Q与原Q逐条相等。

**原电荷范围之外的母事件不在这个已选队列内，无法通过本轮重新分箱补回。** 因此三种结果都是有限队列内的敏感性，不是完整触发、质量选择或探测效率验证。
''')
    code('''rows=[]
for source,g in repeats.items():
    for v,items in g['migration'].items():
        means={n:np.mean([x[n] for x in items]) for n in ['old_support','current_support','leave_support','enter_support','bin_changed','underflow','overflow']}
        rows.append([source,v,*[fmt(means[n]) for n in means]])
display(Markdown(table(['源','响应','原支持','当前支持','移出','移入','换箱','下溢','上溢'],rows)))
display(Image(filename=str(OUT/'figures/charge_migration.png')))
display(read(OUT/'algebra_tests.json'))''')
    md('''## 2. 按每次读出重新加权，再比较波动宽度

主要条件为晚段、块128、当前扩展谱响应及零CV对照的16次正编号读出。每次读出分别按指定Q条件计算箱内协方差、均值、中心50%/95%宽度平方、平均ADC，然后跨读出平均；宽度最终开平方。这样保留“一次读出实验”的统计对象。

512次配对抽样沿用原实验索引块和完整母事件计数；一个母事件的16条读出始终一起抽取。每次重复、每个支持箱都必须有效。**不能把16次读出摊平后当成16倍独立母事件**，也不能在电荷权重依赖噪声时直接使用上一轮固定权重的母事件内/间分解。区间固定已生成的读出及参考，未完整计入有限重复的噪声积分误差；前8/后8对照保留用于检查敏感性。

原Q条件的单次结果逐次闭合上一轮512组对比，重复结果闭合上一轮“一次读出经验协方差的重复平均”；这里没有将其混同于另一个有限重复校正估计。
''')
    code('''rows=[]
for source,g in repeats.items():
    for c,item in g['conditions'].items():
        for v,r in item['variants'].items():
            rows.append([source,c,v,*[fmt(r['point'][f'F{s}_SD_ratio']) for s in [50,102,202]],ci(r['CI95']['F202_SD_ratio'])])
display(Markdown(table(['源','Q条件','响应','F50 SD比','F102 SD比','F202 SD比','F202条件95%范围'],rows)))
display(Image(filename=str(OUT/'figures/charge_width_controls.png')))''')
    md(r'''## 3. 三个尺度的协方差与差分组合

各原评价折/Q箱采用对折实测的加权均值$m_s$，定义$z_s=F_s/m_s-1$。参考在本轮抽样中固定；当前Q条件选用当前箱的参考。再作固定正交变换：

\[
u_c=(z_{50}+z_{102}+z_{202})/\sqrt3,\quad
u_d=(z_{50}-z_{202})/\sqrt2,\quad
u_k=(z_{50}-2z_{102}+z_{202})/\sqrt6.
\]

这三个轴分别描述共同起伏、短长差异和中间尺度的相对弯曲。正交变换保持规范化协方差的迹，但**不会让物理噪声变成互相独立的频带**。F50/F102/F202来自重叠滤波统计，本来就相关；相关系数、各轴方差与完整协方差误差应共同检查。对折规范化也不使已探索实验重新成为独立验证数据。
''')
    code('''rows=[]
for source,g in repeats.items():
    for v,r in g['conditions']['current_Q']['variants'].items():
        for n in ['F_common_SD_ratio','F_short_long_SD_ratio','F_curvature_SD_ratio','F_covariance_relative_error']:
            rows.append([source,v,n,fmt(r['point'][n]),ci(r['CI95'][n])])
display(Markdown(table(['源','响应','联合指标','点值','条件95%范围'],rows)))
display(Image(filename=str(OUT/'figures/charge_scale_joint_modes.png')))''')
    md(r'''## 4. 一个方差方向能否同时满足三个尺度？

令当前CV与零CV对照在某尺度的方差为$V_c,V_0$，实测为$V_d$。只在两个已经计算的方差端点之间作代数诊断：

\[
V(\lambda)=V_0+\lambda(V_c-V_0),\qquad
\lambda_s=\frac{V_{d,s}-V_{0,s}}{V_{c,s}-V_{0,s}}.
\]

λ=0/1分别是零CV/当前响应，超出[0,1]的值原样报告；分母近零则不可估计。配对报告λ50−λ202等差值，判断各尺度对**这条方差直线**的要求是否不同。这里不假设中间原生CV真的遵循这条直线，不将λ换算成器件CV，也不能由此排除所有非线性响应。相关定位、电荷选择和随机归一化都可能造成非线性。
''')
    code('''rows=[]
for source,g in repeats.items():
    for c,item in g['conditions'].items():
        r=item['interpolation'];rows.append([source,c,*[fmt(r['point'][f'lambda_F{s}']) for s in [50,102,202]],ci(r['CI95']['lambda_F50_minus_F202'])])
display(Markdown(table(['源','Q条件','λ50','λ102','λ202','λ50−λ202配对范围'],rows)))
display(Image(filename=str(OUT/'figures/charge_variance_line.png')))''')
    md('''## 5. 采集段检查与平均波形

全部前/中/晚段、块64/128都保留原单次读出的配对分析。平均波形使用基线扣除ADC，不逐事件缩放幅度；重分箱后均值的变化是事件组成/权重变化，不是重新调参。零CV对照也没有根据本轮结果重拟合。
''')
    code('''rows=[]
for context,g in points.items():
    old=g['conditions']['original_Q']['variants']['current'];new=g['conditions']['current_Q']['variants']['current']
    rows.append([context,fmt(old['point']['F202_SD_ratio']),fmt(new['point']['F202_SD_ratio']),ci(new['change_from_original_Q']['CI95']['F202_SD_ratio']),
                 fmt(old['point']['mean_ADC_relative_L2']),fmt(new['point']['mean_ADC_relative_L2'])])
display(Markdown(table(['条件','原Q F202比','当前Q F202比','配对变化范围','原Q均ADC相对L2','当前Q均ADC相对L2'],rows)))
display(Image(filename=str(OUT/'figures/charge_segment_checks.png')))
display(Image(filename=str(OUT/'figures/charge_mean_waveforms.png')))
display(Markdown('**下一步：**'+d['next_work_zh']))
for limit in d['limitations_zh']:display(Markdown('边界：'+limit))''')
    md('''## 6. 记录与复现

`protocol.json`冻结505个输入哈希、代码、三种Q条件、正交组合、16次重复与512次配对计数口径；`references/`保存对折实测尺度均值及训练ID。`point_evaluation/`与`repeat_evaluation/`保存完整协方差、指标、原支持计数、跨箱矩阵和全部抽样对比。

`point_table.csv`有72行，`repeat_table.csv`有12行，`variance_line_table.csv`有42行，`migration_table.csv`有100行。`verification.json`检查原指标闭合、直接加权协方差、原生电荷一致性、条件区间及笔记；`record.json`封存输出和源码快照。

流程为`prepare_charge_scales.py` → `analyze_charge_scales.py points --workers 2` → `analyze_charge_scales.py repeats --workers 2` → `report_charge_scales.py` → 本笔记生成器 → `verify_charge_scales.py`及`--archive`。完整重跑须在新结果目录冻结方案。旧档案保持原样，本轮没有联网或修改共享文献库。
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
    nbf.validate(nb);path=PROJECT/'jupyter/codex_0001_csi_charge_scale_constraints.ipynb';nbf.write(nb,path);print(path)


if __name__=='__main__':main()
