"""Build and execute the fixed-history repeated-noise research notebook."""
import os
from pathlib import Path
PROJECT=Path(__file__).resolve().parents[1]
cache=PROJECT/'.cache/codex_0001_noise_replay_ipython'
cache.mkdir(parents=True,exist_ok=True)
os.environ['IPYTHONDIR']=str(cache)
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import nbformat as nbf
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output


def main():
    out=PROJECT/'simulation/results/noise_replay_20261102'
    assert not (out/'record.json').exists() and not (out/'verification.json').exists()
    cells=[]
    def md(s):cells.append(nbf.v4.new_markdown_cell(s))
    def code(s):cells.append(nbf.v4.new_code_cell(s))
    md('''# CsI–SiPM：固定雪崩历史的重复噪声与事件方差对照

2026-11-02。承接已提交并推送的 **82d42f8**（`Diagnose CsI PID fluctuation differences with Gaussian closure`）及[Gaussian闭合笔记](codex_0001_csi_gaussian_closure.ipynb)。原F202方差差异主要落在二阶闭合之外；本轮检验主带选择与读出随机性，并测试只去掉信号噪声的事件方差乘子是否能改善联合结果。

原有256000个Geant4母事件中，复用3661个已选母事件（Co 2175、Am 1486；原支持Q箱内为2175/1406）。新增198819条科学分析读出记录，来自788个物理输出文件，**新增独立Geant4母事件和实验事件均为0**。读出重复不增加独立输运统计量。API核验另有240条测试记录，不计入科学分析条数。

仍使用原CH0、53.2 V的run3与run12。Am源含α及γ/电子贡献；来源标签和QDC主带不能当作实验真值。旧数据及本MC池都已参与探索，本轮不是盲验证。
''')
    code('''from pathlib import Path
import json
import numpy as np
from IPython.display import display,Markdown,Image
PROJECT=next(p for p in (Path.cwd(),*Path.cwd().parents) if (p/'simulation/main.cc').exists())
OUT=PROJECT/'simulation/results/noise_replay_20261102'
def read(path):return json.loads(Path(path).read_text())
def fmt(x):return '不可估计' if x is None else f'{x:.4g}'
def ci(x):return '不可估计' if x['CI95'] is None else '['+', '.join(fmt(v) for v in x['CI95'])+f"] ({x['valid_draws']}/512)"
def table(head,rows):return '\\n'.join(['|'+'|'.join(head)+'|','|'+'|'.join(['---']*len(head))+'|']+['|'+'|'.join(map(str,r))+'|' for r in rows])
p=read(OUT/'replay_protocol.json');selection=read(OUT/'selection_analysis.json')
points=read(OUT/'point_analysis.json');repeated=read(OUT/'repeat_analysis.json');decision=read(OUT/'decision.json')
display(Markdown('**本轮结论：**'+decision['conclusion_zh']))
for finding in decision['findings_zh']:display(Markdown(finding))
print('固定母事件:',p['fixed_selected_mothers'],'科学读出记录:',p['physical_replay_entries'])''')
    md('''## 1. 主带选择是否造成不公平比较？

原比较保留实测主带（Co QDC≥0.4，Am QDC<0.4），MC没有这一步删除。本轮在全部前/中/晚段、块64/128及三个已有响应中，比较：完整已选实测队列、继承主带、双方按当前响应的同一0.4条件筛选、MC按冻结响应固定主带。每种情况仍按原Q半箱目标质量归一，缺失支持格就报告不可估计，不能偷偷删箱。

“完整已选”只是主带切分之前的队列，仍经过原质量和共同电荷范围选择。该对照不等于未触发、未筛选的真实源总体。
''')
    code('''rows=[]
for run in [3,12]:
    g=selection['groups'][f'run{run}_last20']['128']
    for case in ['all_selected','inherited','matched_band','fixed_MC_band']:
        r=g['comparisons']['spectrum_CV_fit'][case]
        rows.append([run,case,fmt(r['point']['F202_SD_ratio']),ci(r['CI95']['F202_SD_ratio']),fmt(r['change_from_inherited']['F202_SD_ratio'])])
display(Markdown(table(['run','主带条件','F202 SD比','条件95%范围','相对原比较变化'],rows)))
display(Image(filename=str(OUT/'figures/noise_selection_audit.png')))''')
    md('''## 2. 原生C++重放：同一信号，只更换下游噪声

新增`include/ReadoutReplica.hh`、`src/ReadoutReplica.cc`，由原`replay_readout`入口调用；保留Geant4的include/src结构和完整输运入口。复用原ROOT中的雪崩时间、电荷和固定读出响应，不重新抽取光子、PDE、串扰或后脉冲。后者已在固定雪崩历史内，因此不能用本轮“母事件内噪声”估计它们的独立物理贡献。

`noise_replica=0`保留原随机种子及ADC；正编号在专用种子域重放，可复现同一编号。`noise_scope`为`all_readout`、`signal_only`或`electronics_only`。分别改变两条流、仅信号相关残差流、仅电子学流；单流对照把另一条流固定为零号实现，因此三者不能直接按方差相加。

主条件为晚段、块128、扩展残差谱：全部噪声16次、信号流8次、电子学流8次、零CV全部噪声16次。另在全部12个条件作零号配对对照。所有R与对照在结果前固定；没有根据结果追加重复或剔除事件。
''')
    code('''tests=read(OUT/'replay_tests.json')
display(tests['native_tests'])
print('C++测试:',tests['ctest_passed'],'合成有限重复检验:',tests['algebra']['trials'])
for source,r in repeated.items():
    print(source,{k:{'R':v['repeat_count'],'负的母事件间对角项':v['negative_between_diagonal'],
                    '最大主带外事件数':max(x['outside_band'] for x in v['band_migrations'])} for k,v in r['cases'].items()})''')
    md(r'''## 3. 有限重复下的母事件内/间方差

在每个原Q箱内，令第i个母事件的R次9维特征为$X_{ir}$，权重$w_i$沿用原Q半箱目标且和为1。$\bar X_i$为重复均值，$S_i$为重复的`ddof=1`协方差：

\[
B_{raw}=\sum_i w_i(\bar X_i-\bar X)(\bar X_i-\bar X)^T,\quad
C_R=\sum_i\frac{w_i(1-w_i)}{R}S_i,\quad B=B_{raw}-C_R.
\]

本项目原统计使用归一权重的经验协方差，因此“一母事件一次读出”的预期方差为

\[
V_{one}=B+W_{one},\qquad W_{one}=\sum_i w_i(1-w_i)S_i.
\]

先逐原Q箱计算，再按原支持箱等权平均。有限R校正后的B可能为负，保留原值；不能把它截零或解释为纯闪烁方差。母事件间项含条件噪声均值、定位和随机分母效应。此公式估计的是固定所选母事件、重复读出条件下的量，不是全源总体的物理方差分解。

512次区间保持原实验索引块和四批完整母事件抽样。一个母事件的所有重复一起重抽；同一ID的重复副本先合并权重，再作有限R校正。区间固定模型、Q参考、响应折，不包括参数拟合或系统误差，也不把198819条记录当作独立样本。每条流的有限重复估计仍有Monte Carlo误差，预设前8/后8敏感性检查另报。
''')
    code('''rows=[]
for source,g in repeated.items():
    for case,r in g['cases'].items():
        for name in ['F202_predicted_SD_ratio','F202_within_fraction','F202_between_fraction']:
            rows.append([source,case,name,fmt(r['point'][name]),ci(r['CI95'][name])])
display(Markdown(table(['源','重复流','指标','点值','条件95%范围'],rows)))
display(Image(filename=str(OUT/'figures/noise_repeat_decomposition.png')))
display(Image(filename=str(OUT/'figures/noise_repeat_convergence.png')))''')
    md('''## 4. 最小参数对照：仅将事件方差CV置零

当前信号相关残差为高斯FIR创新乘以随正信号幅度变化的标准差，再乘一个事件层的均值为1的对数正态方差因子。这里只将该因子的`event_variance_cv`置零，保留平均方差系数、FIR、电子学噪声、原始信号及传感器参数。原CV已为0的条件直接复用相同文件，避免把相同计算虚报为独立记录。

Co原CV约0.19866；Am条件为0或约0.09048。本对照不是重新拟合。用同一批母事件、同一噪声创新及配对区间比较平均波形、F均值/SD、QDC/时间/PCA的SD与中心50%/95%宽度和PID协方差，避免只缩窄F202却破坏其他指标。

保护条件事先冻结：比值指标用`abs(log(ratio))`误差，中心差用绝对值，平均ADC与协方差用非负误差。任何原指标的配对误差增加区间下界>0，均作为保守的描述性退化警报；同时要求两个源晚段F202误差减少。固定分母F只作机制诊断。多指标区间未作多重比较校正，不能解释为统一显著性检验或等效性证明。
''')
    code('''names=['F50_SD_ratio','F102_SD_ratio','F202_SD_ratio','QDC_width50_ratio','QDC_width95_ratio',
       'negative_mean_time_us_width50_ratio','negative_mean_time_us_width95_ratio','mean_ADC_relative_L2','PID_covariance_relative_error']
rows=[]
for run in [3,12]:
    g=points[f'run{run}_last20_B128']
    for name in names:
        rows.append([run,name,fmt(g['variants']['current']['point'][name]),fmt(g['variants']['zero_cv']['point'][name]),
                     ci(g['loss_change_zero_minus_current']['CI95'][name])])
display(Markdown(table(['run','指标','当前','零CV','零CV−当前：误差变化范围'],rows)))
display(Image(filename=str(OUT/'figures/noise_zero_cv_joint_checks.png')))
display(Image(filename=str(OUT/'figures/noise_cv_component_changes.png')))''')
    md('''## 5. 平均波形与全部采集条件

波形图采用原Q目标、原事件权重的基线扣除ADC均值，不逐事件缩放幅度，不通过修改滤波或积分门来对齐。每条重复都重新从ADC定位t10与计算特征；无噪声信号本身逐点不变，有限噪声及定位可改变最终均ADC图。

所有前/中/晚段和块64/128保留。原Q分箱依然来自继承ADC；重放的`qcalc`有保存，但不重新筛选或改箱。因此本轮不是“每次重放后重新触发、重新按当前电荷选择”的端到端验证，电荷条件影响仍需单独检查。
''')
    code('''display(Image(filename=str(OUT/'figures/noise_mean_waveforms.png')))
rows=[]
for context,g in points.items():
    r=g['variants'];loss=g['loss_change_zero_minus_current']
    rows.append([context,fmt(r['current']['point']['F202_SD_ratio']),fmt(r['zero_cv']['point']['F202_SD_ratio']),
                 ci(loss['CI95']['F202_SD_ratio']),sum(x['context']==context for x in decision['clear_loss_regressions'])])
display(Markdown(table(['条件','当前F202 SD比','零CV比','F202误差变化范围','明确退化项数'],rows)))
display(Markdown('**下一步：**'+decision['next_work_zh']))
for limit in decision['limitations_zh']:display(Markdown('边界：'+limit))''')
    md('''## 6. 可追溯记录

`selection_protocol.json`、`replay_protocol.json`、`analysis_protocol.json`分别冻结主带审计、原生重放、有限重复与联合指标；输入、源码和二进制有SHA256。`replay_execution.json`汇总24份执行收据，每份保存ROOT和特征数组哈希、零号一致性与不变信号核查。

`selection_table.csv`有144行，`zero_cv_point_table.csv`有24行，`zero_cv_loss_table.csv`有264行，`repeat_table.csv`有8行。完整逐Q协方差、512组母事件/实验块计数及配对分解位于各`*_evaluation/`目录，原生记录在`replay/`，紧凑特征在`arrays/`与`repeat_arrays/`。大ROOT/NPZ按项目惯例不进入Git，但本地保留并封存哈希。`verification.json`记录独立种子计算、直接卷积窗口、加权协方差及笔记核验；`record.json`封存输出和源码快照。

流程：`prepare_noise_selection.py` → `audit_noise_selection.py` → 构建独立C++目录/`test_noise_replay.py` → `prepare_noise_replay.py` → `run_noise_replay.py` → `analyze_noise_replay.py freeze/points/repeats` → `report_noise_replay.py` → 本笔记生成器 → `verify_noise_replay.py --archive`。本次分析方案在原生重放完成前已冻结。完整重跑应使用新的结果目录，不能覆盖已封存研究。

本轮没有联网、下载文献或修改共享文献库；没有修改默认模拟响应和正式PID。结论只覆盖上述对照，不能据此宣称PID带宽已经全面一致。
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
    nbf.validate(nb);path=PROJECT/'jupyter/codex_0001_csi_noise_replay.ipynb';nbf.write(nb,path);print(path)


if __name__=='__main__':main()
