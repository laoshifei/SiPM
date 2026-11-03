"""Build and execute the existing-ADC intensity-conditioning study."""
import os
from pathlib import Path
PROJECT=Path(__file__).resolve().parents[1]
cache=PROJECT/'.cache/codex_0001_intensity_condition_ipython';cache.mkdir(parents=True,exist_ok=True)
os.environ['IPYTHONDIR']=str(cache);os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import nbformat as nbf
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output


def main():
    out=PROJECT/'simulation/results/intensity_condition_20261102';assert (out/'decision.json').exists() and not (out/'record.json').exists()
    cells=[]
    def md(x):cells.append(nbf.v4.new_markdown_cell(x))
    def code(x):cells.append(nbf.v4.new_code_cell(x))
    md('''# CsI–SiPM：脉冲强度能否解释剩余波动差异？

2026-11-02。在英文提交`75641c2a0f52056fd1bc4ae25271f063480715da`（`Diagnose time-resolved CsI SiPM fluctuation differences`）push并核对远端后继续。[上一轮时间分段诊断](codex_0001_csi_time_resolved_fluctuations.ipynb)发现Am241最大的带符号方差差额落在t10后500–1000 ns。本轮检验这种差额是否主要来自脉冲平均信号强度的简单线性变化。

**结论：简单线性强度关系没有解释Am241的差额。** 用首段平均强度扣除线性关系后，整体F202的MC/实测SD比仍为1.171/1.186，对折残差比为1.138/1.162。首段固定分母项本身也仍偏宽。这里计算的是诊断残差，尚未修改任何模拟参数或ADC，也没有改善正式PID。

只使用既有实测与原生C++/Geant4波形的已核验时间分段数组、原Q箱和原512组重采样。没有新实验、新母事件或新增读出。Am241包括α与γ/电子，不能用源标签代替实验逐事件粒子真值。
''')
    code('''from pathlib import Path
import json
import numpy as np
from IPython.display import display, Markdown, Image
PROJECT=next(p for p in (Path.cwd(),*Path.cwd().parents) if (p/'simulation/main.cc').exists())
OUT=PROJECT/'simulation/results/intensity_condition_20261102'
def read(path): return json.loads(Path(path).read_text())
def fmt(x): return '不可估计' if x is None else f'{x:.4g}'
def ci(x): return '不可估计' if x['CI95'] is None else '['+', '.join(fmt(y) for y in x['CI95'])+f"] ({x['valid_draws']}/512)"
def table(head,rows): return '\\n'.join(['|'+'|'.join(head)+'|','|'+'|'.join(['---']*len(head))+'|']+['|'+'|'.join(map(str,r))+'|' for r in rows])
p=read(OUT/'protocol.json');a=read(OUT/'analysis.json');d=read(OUT/'decision.json')
rows=[]
for source,g in a.items():
    for v,r in g['variants'].items():
        for predictor in [p['predictors'][0],p['predictors'][-1]]:
            z=r['global'][predictor]['202'];q=z['point']
            rows.append([source,v,predictor,fmt(q['original_SD_ratio']),fmt(q['residual_SD_ratio']),ci(z['CI95']['residual_SD_ratio']),fmt(q['crossfit_residual_SD_ratio']),ci(z['CI95']['crossfit_residual_SD_ratio'])])
display(Markdown(table(['源','读出','强度预测量','原F202比','同样本残差比','有效组分位范围','对折残差比','有效组分位范围'],rows)))
display(Image(filename=str(OUT/'figures/intensity_global_residuals.png')))''')
    md(r'''## 1. 固定设计：同一Q箱内的五种一维关系

保持Co60 run 3、Am241 run 12的last20、B128、原质量条件、当前Q及支持箱。沿用已核验的四段250样点窗口（每段500 ns），以及整段1000样点窗口。对每个窗口均值$\mu_k$和整体均值$\mu$，分别构造$x=\mu_k/\mu_0$或$x=\mu/\mu_0$。$\mu_0$仍是先前对折、每Q箱的参考；不重新拟合幅度标尺。

五种$x$是**五个分别进行的一维回归**，不是五变量联合拟合。每种关系都保留三个滤波尺度和两个源。每个Q箱保留原上下半箱的目标质量；MC与实测分别做描述性拟合，再按Q箱等权汇总协方差。没有按强度删除或重新加权事件。

回归响应是上一轮七项$\mathbf T$，其和逐事件等于原整体$F$。以每箱加权总体矩记

\[
v_x=\operatorname{Var}(x),\quad \mathbf c=\operatorname{Cov}(\mathbf T,x),\quad
\boldsymbol\beta=\frac{\mathbf c}{v_x},\quad
C_{\rm linear}=\frac{\mathbf c\mathbf c^\top}{v_x},\quad
C_{\rm residual}=C-C_{\rm linear}.
\]

这是带截距加权最小二乘的精确同样本方差分解。先在各Q箱计算，再等权平均，之后对矩阵元素求和得到整体F的方差；不是先把不同Q箱混合后拟合。差额均以实测原整体F方差归一化，线性差额与剩余差额之和严格等于原差额。单个样本的线性方差非负，但MC减实测的线性差额可以为负。

**预测量和响应都来自同一带噪ADC。** 该关系可能包含内生相关、曲率、归一化、源混合等，不能用回归系数直接解释SiPM物理噪声，也不能据此给实验波形做校正。
''')
    code('''rows=[]
for source,g in a.items():
    for v,r in g['variants'].items():
        for scale in ['50','102','202']:
            z=r['global'][p['predictors'][0]][scale];q=z['point']
            rows.append([source,v,scale,fmt(q['variance_excess']),fmt(q['linear_excess']),fmt(q['residual_excess']),fmt(q['data_linear_fraction']),fmt(q['MC_linear_fraction'])])
display(Markdown(table(['源','读出','尺度/ns','原方差差额','线性差额','剩余差额','实测线性方差占比','MC线性方差占比'],rows)))
display(Image(filename=str(OUT/'figures/intensity_variance_partition.png')))''')
    md(r'''## 2. 首段和其他时间窗口的剩余离散

局部检查使用$T_k=(249/999)V_k/\mu_0$，并以本窗口的$\mu_k/\mu_0$为预测量。这里比较的是**七项分解中的固定分母窗口项**，没有减去前触发方差，也没有除以随机局部均值，所以不能与上一轮局部$F_k=(V_k-V_{\rm pre})/\mu_k$的SD比混用。

Am241首段窗口项原SD比为1.119/1.163，扣除同样本线性关系后为1.142/1.183，对折残差为1.124/1.173。这个结果支持继续研究相近强度下的剩余离散及时间相关结构；它没有排除全部非线性强度模型，更没有唯一确定微观机制。
''')
    code('''rows=[]
for source,g in a.items():
    for v,r in g['variants'].items():
        for k,predictor in enumerate(p['predictors'][:4]):
            z=r['local'][predictor]['202'];q=z['point']
            rows.append([source,v,f'{500+500*k}–{1000+500*k}',fmt(q['original_SD_ratio']),fmt(q['residual_SD_ratio']),fmt(q['crossfit_residual_SD_ratio']),ci(z['CI95']['crossfit_residual_SD_ratio'])])
display(Markdown(table(['源','读出','窗口/ns','原窗口项SD比','同样本残差比','对折残差比','有效组分位范围'],rows)))
display(Image(filename=str(OUT/'figures/intensity_local_residuals.png')))''')
    md('''## 3. 对折检查与斜率的边界

为了检查同样本拟合的自动变窄，每个Q箱中用另一原始折拟合截距和斜率，预测本折，再把两折残差按原完整Q箱的半箱权重合并，最后在Q箱内中心化。**折间残差均值差仍计入协方差。** MC和实测分别拟合；这里没有把一个样本的斜率作为另一个样本的校正。

对折结果会包含斜率估计噪声，因此残差可能变宽。尤其Co60，不能把对折SD比更接近1说成模拟获得改善。所有实验样本已经探索过，此检查不是新盲验证。

下图斜率是每Q箱内窗口项对无量纲强度的斜率，再对Q箱等权平均。Am241实测与MC的斜率存在差别，但扣除各自线性关系后过宽仍在。全部每Q箱拟合、两折斜率和截距保存于`analysis.json`的`cells`字段；斜率差不能单独确定新的噪声包络参数。
''')
    code('''rows=[]
for source,g in a.items():
    for k,predictor in enumerate(p['predictors'][:4]):
        r0=g['variants']['split_r0']['local'][predictor]['202'];r1=g['variants']['split_r1']['local'][predictor]['202']
        rows.append([source,f'{500+500*k}–{1000+500*k}',fmt(r0['point']['data_slope']),ci(r0['CI95']['data_slope']),fmt(r0['point']['MC_slope']),ci(r0['CI95']['MC_slope']),fmt(r1['point']['MC_slope']),ci(r1['CI95']['MC_slope'])])
display(Markdown(table(['源','窗口/ns','实测斜率','有效组分位范围','r0斜率','有效组分位范围','r1斜率','有效组分位范围'],rows)))
display(Image(filename=str(OUT/'figures/intensity_local_slopes.png')))''')
    md('''## 4. 保留不可估计的重采样，不改变比较样本

完全沿用原512组实验事件块和MC母事件的配对计数。**每组都重新估计本轮新增的描述性回归**；已有物理拟合、平均响应、参考、切选和噪声实现保持固定。原始统计的全部512组结果仍须复现上一轮。

原完整Q箱至少需要5个按重采样次数计的事件，并且每个正目标质量的半箱都必须有支持。线性拟合还要求预测量方差大于1e-14；对折检查对每个训练折应用相同事件数与支持规则。任何Q箱不满足条件，则整组相应估计量记为不可估计，不丢弃Q箱、窗口或更改阈值。

原统计与同样本分解均为512/512组有效。Co60实测对折仅399/512组有效，Am241为509/512；MC两条流均512/512。**图表中的对折范围是有效组的2.5%–97.5%分位，不是完整bootstrap的95%置信区间。** Co的缺失比例约22%，限制了对折结果的说服力。所有尺度、窗口共享事件，范围未作多重比较校正，也未覆盖系统误差。

另报告每个Q箱内MC落在实测加权5%–95%强度区间的质量比例。该范围按累计权重插值计算，只描述覆盖情况；没有据此做强度切选或重加权。覆盖良好也不能证明条件分布一致。
''')
    code('''rows=[]
for source,g in a.items():
    for v,n in g['valid_draws'].items():
        rows.append([source,v,n['original'],n['linear'],n['crossfit'],512-n['crossfit']])
display(Markdown(table(['源','样本','原统计有效组','同样本有效组','对折有效组','对折不可估计组'],rows)))
display(Image(filename=str(OUT/'figures/intensity_range_overlap.png')))
display(Markdown('**下一步：**'+d['next_work_zh']))
for text in d['limitations_zh']:display(Markdown('边界：'+text))''')
    md('''## 5. 复现和封存

流程为`analyze_intensity_condition.py` → `report_intensity_condition.py` → 本笔记生成器 → `verify_intensity_condition.py`及`--archive`。分析程序在结果计算前保存方案、输入与源码哈希；`global_table.csv`含60行整体结果，`local_table.csv`含48行窗口结果，全部尺度与预测量均保留。数据来自上一轮已验证的实际ADC分解数组，没有产生新波形。

初次运行的闭合检查发现：新增对折支持要求使部分重采样无效，原实现把这种无效标记也传播给本应保持有效的原统计。该次尚未写出统计结果。修正后分别记录原统计、同样本拟合、对折拟合的有效性，保持原512组统计完整；新回归仍按原规则标记不可估计。未修改阈值、拟合公式、计数或候选。旧方案及三份源码保存在`pre_export_fix_protocol.json`和`pre_export_fix_source/`，修改原因与新旧哈希见`export_fix_record.json`；实际恢复命令使用`--resume-export-fix`。

独立核验使用显式重复事件、带权最小二乘与直接残差协方差，检查点估计及第0/127/511组重采样；同时核对全部512组原统计闭合、所有区间、协方差半正定性、输入哈希、表格行数和已执行笔记。`verification.json`给出实际检查数，`record.json`保存输出哈希和源码快照。五幅图均进行可视检查。

默认参数、正式PID、原ADC和共享文献库均未修改。前一轮时间分段研究已经commit和push，本轮强度条件诊断为随后新增的本地工作。
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
        for output in captured.outputs:outputs.append(nbf.v4.new_output('display_data',data=output.data,metadata=output.metadata))
        c.outputs=outputs
    nbf.validate(nb);path=PROJECT/'jupyter/codex_0001_csi_intensity_conditioning.ipynb';nbf.write(nb,path);print(path)


if __name__=='__main__':main()
