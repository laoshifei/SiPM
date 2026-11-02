"""Execute the fixed-event QDC/fluctuation residual analysis notebook."""
import os
from pathlib import Path
PROJECT = Path(__file__).resolve().parents[1]
cache = PROJECT/'.cache/codex_0001_qdc_fluctuation_ipython'
cache.mkdir(parents=True, exist_ok=True)
os.environ['IPYTHONDIR'] = str(cache)
os.environ['OPENBLAS_NUM_THREADS'] = '1'; os.environ['OMP_NUM_THREADS'] = '1'
import nbformat as nbf
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output


def main():
    path = PROJECT/'jupyter/codex_0001_csi_qdc_fluctuation_residuals.ipynb'
    assert not path.exists()
    cells = []
    def md(s): cells.append(nbf.v4.new_markdown_cell(s))
    def code(s): cells.append(nbf.v4.new_code_cell(s))
    md('''# CsI–SiPM：QDC尾部与F202离散度的剩余差异

2026-11-01，承接[独立Geant4统计复现](codex_0001_csi_independent_mc_statistics.ipynb)。上一轮新增25.6万个母衰变后，Am平均时间95%分位宽度比从0.872变为0.980；但晚段扩展谱的QDC外侧宽度仍偏大，F202的事件间SD比为Co1.232、Am1.201。

本轮固定已有参数、事件、原Q和统计目标，追查这两类残差的结构。**新增Geant4母事件、原生读出重放、实验事件均为0；没有拟合或替换响应，没有剔除尾部事件，没有改变正式PID。**
先提交上一阶段：`d481527`，英文题目 `Validate CsI PID tails with independent Geant4 statistics`。本笔记属于该提交后的继续分析。
''')
    code('''from pathlib import Path
import json
import numpy as np
from IPython.display import display,Markdown,Image
PROJECT=next(p for p in (Path.cwd(),*Path.cwd().parents) if (p/'simulation/main.cc').exists())
OUT=PROJECT/'simulation/results/qdc_fluctuation_20261101'
def read(p):return json.loads(Path(p).read_text())
def fmt(x,n=3):return '不可估计' if x is None else f'{x:.{n}f}'
def ci(x):return '不可估计' if x['CI95'] is None else '['+', '.join(fmt(v) for v in x['CI95'])+']'
def table(head,rows):return '\\n'.join(['|'+'|'.join(head)+'|','|'+'|'.join(['---']*len(head))+'|']+['|'+'|'.join(map(str,r))+'|' for r in rows])
p=read(OUT/'protocol.json');e=read(OUT/'analysis.json');decision=read(OUT/'decision.json')
variants=p['variants'];labels={'frozen':'原冻结','inherited_CV_fit':'RC+原谱','spectrum_CV_fit':'RC+扩展谱'}
print('方案冻结提交:',p['checkpoint_commit'])
print('输入哈希数:',len(p['inputs_sha256']),'本轮独立母事件:',p['new_independent_mothers'])
display(Markdown(decision['conclusion_zh']))''')
    md(r'''## 1. F的精确定义与分解

对每条原ADC，使用与上一轮相同的25/51/101样点中心滑动均值残差，对应50/102/202 ns。前触发窗200个样点、信号窗1000个样点，边界仍为原 `ceil(t10−300)` 和 `ceil(t10+250)`；两窗的方差均使用 `ddof=1`。

\[
F=\frac{V_{\rm post}-V_{\rm pre}}{\mu}=P+B+D,
\quad P=\frac{V_{\rm post}}{\mu_0},\quad B=-\frac{V_{\rm pre}}{\mu_0},
\quad D=(V_{\rm post}-V_{\rm pre})\left(\frac1\mu-\frac1{\mu_0}\right).
\]

其中μ是该事件信号窗平均幅度；μ₀只用同一原Q箱的**对折实验**、原Q两半目标权重求得，两个源及采集段分别定义。μ₀不是每事件幅度匹配，也没有用于修改模拟波形。

事件间方差保留完整3×3协方差。每项的贡献定义为 `Cov(该项,F)`，三项严格相加得到 `Var(F)`，可以为负；不把它们解释成三种独立物理噪声方差。固定分母的 `P+B` 只是诊断量，原F和PID保持不变。
''')
    code('''names=['F202_SD_ratio','F202_fixed_denominator_SD_ratio','F202_within_cell_SD_ratio','width95_QDC_ratio']
rows=[]
for run in [3,12]:
    for v in variants:
        r=e['groups'][f'run{run}_last20']['128']['variants'][v]
        for n in names:rows.append([run,labels[v],n,fmt(r['point'][n]),ci(r['CI95'][n])])
display(Markdown(table(['晚段run','固定响应','指标','MC/实测比','条件95%区间'],rows)))
display(Image(filename=str(OUT/'figures/fluctuation_term_controls.png')))''')
    code('''rows=[]
for run in [3,12]:
    r=e['groups'][f'run{run}_last20']['128']['variants']['spectrum_CV_fit']
    for term in ['post_window','pre_window','normalization']:
        n=f'F202_{term}_excess_over_data_variance'
        rows.append([run,term,fmt(r['point'][n]),ci(r['CI95'][n])])
    print('run',run,'三项总和:',sum(r['point'][f'F202_{t}_excess_over_data_variance'] for t in ['post_window','pre_window','normalization']),
          '直接方差超额/实测方差:',r['point']['F202_SD_ratio']**2-1)
display(Markdown(table(['晚段run','协方差贡献差/实测F202方差','点值','条件95%区间'],rows)))''')
    md('''## 2. Q条件与响应折的结构

所有方差延续上一轮“原Q箱内方差、再箱间等权”的定义，不混入不同Q箱均值之差。进一步将箱内协方差精确分成：Q两半之间、每半箱内响应折之间、半箱/折内部三项，保留原目标质量和全部事件。

折间均值项含有限样本均值误差，不等于真实电路状态漂移；这里只检查能否用这种分组结构解释当前差异。Co保留6箱，Am保留前3箱，逐箱的样本数和有效样本量另列，不根据结果删除电荷箱。
''')
    code('''rows=[]
for run in [3,12]:
    g=e['groups'][f'run{run}_last20']['128'];r=g['variants']['spectrum_CV_fit']
    for i,b in enumerate(r['per_Q']):
        d=g['data']['per_Q'][i];m=r['summary']['per_Q'][i];x=r['per_Q'][b]
        rows.append([run,b,d['events'],fmt(d['ESS'],1),m['events'],fmt(m['ESS'],1),
                     fmt(x['point']['width95_QDC_ratio']),ci(x['CI95']['width95_QDC_ratio']),
                     fmt(x['point']['F202_SD_ratio']),ci(x['CI95']['F202_SD_ratio'])])
display(Markdown(table(['run','Q箱','实测数','实测ESS','MC数','MC ESS','QDC95%比','区间','F202 SD比','区间'],rows)))
display(Image(filename=str(OUT/'figures/fluctuation_by_charge.png')))''')
    md('''## 3. QDC尾部与F202外侧是否重合

QDC尾部继续使用旧对折实验的稳健标准化 `abs(z)>2`。F202外侧定义为新对折实验5%/95%分位之外，同时报告下侧和上侧；阈值不从MC选择，也不在本轮bootstrap中重新拟合。
5%/95%是训练折的分位，评价折的外侧质量不必等于10%；原计数、加权质量和统计区间应一起阅读。

按“均不在尾部、仅QDC、仅F202、两者同时”四类分配全部事件，统计权重质量和各自对完整箱内方差的贡献。F、QDC及幅度归一化共享ADC样点，尾部重合、相关系数或共同方差不能单独识别因果；这些标签不是坏事件、堆积或粒子类型分类。
''')
    code('''rows=[]
for run in [3,12]:
    g=e['groups'][f'run{run}_last20']['128']
    for label,s in [('实测',g['data'])]+[(labels[v],g['variants'][v]['summary']) for v in variants]:
        rows.append([run,label,' / '.join(str(x) for x in s['category_counts']),
                     ' / '.join(fmt(100*x,2) for x in s['category_mass']),
                     fmt(100*s['F_lower_mass'],2),fmt(100*s['F_upper_mass'],2)])
display(Markdown(table(['run','队列','四类原事件数','四类加权质量/%','F下侧/%','F上侧/%'],rows)))
display(Image(filename=str(OUT/'figures/fluctuation_tail_overlap.png')))''')
    md('''## 4. 模拟内部：原生信号、附加残差与交叉项

对MC保存的原生信号S和 `ADC−原基线−S` 得到的附加残差N，使用同一个线性局部滤波与原窗边界。每个窗的方差满足 `V(S+N)=V(S)+V(N)+2Cov(S,N)`；前后窗作差并除原μ，得到三个逐事件F分量，再计算它们对事件间F方差的协方差贡献。

原生信号包含光子、雪崩及响应涨落；附加残差合并了电子噪声、经验信号相关残差和ADC量化。交叉项不能删掉。这一模型潜变量分解不对实验可观测，不能用“附加残差贡献大”直接宣称已经测得某一种实验噪声。
''')
    code('''rows=[]
for run in [3,12]:
    g=e['groups'][f'run{run}_last20']['128'];dv=np.diag(g['data']['cov'])[1:]
    for v in variants:
        c=np.asarray(g['variants'][v]['summary']['native_terms_cov'])
        for i,w in enumerate([50,102,202]):
            x=c[i].sum(1)/dv[i]
            rows.append([run,labels[v],w,*[fmt(z) for z in x],fmt(x.sum())])
display(Markdown(table(['run','固定响应','尺度/ns','信号贡献/实测方差','附加残差贡献','交叉贡献','总MC/实测方差'],rows)))
display(Image(filename=str(OUT/'figures/fluctuation_native_parts.png')))''')
    md('''## 5. 采集段、块长与统计口径核验

报告前60%、中间20%、最后20%以及块64/128。全部实验和本轮MC已用于诊断，不能称盲验证。

沿用上一轮相同的512次实验原索引块抽样、四批独立母事件的分层完整母事件抽样，保留原随机流中旧母事件抽样的消耗顺序，使配对样本完全相同。QDC的50%/95%宽度、三个F均值及三个F事件SD，**共8项既有指标在所有36种条件/响应下、每次抽样均须复现父记录**。新派生量区间固定全部响应、参考和权重，不包含拟合/模型系统误差，也没有多重比较校正或等效性检验。
''')
    code('''rows=[]
for key,blocks in e['groups'].items():
    for B,g in blocks.items():
        r=g['variants']['spectrum_CV_fit']
        rows.append([key,B,fmt(r['point']['F202_SD_ratio']),ci(r['CI95']['F202_SD_ratio']),
                     fmt(r['point']['F202_within_cell_SD_ratio']),fmt(r['point']['width95_QDC_ratio']),
                     g['parent_metric_draw_closure']])
display(Markdown(table(['组','块长','F202 SD比','区间','半箱/折内SD比','QDC95%比','父抽样逐次闭合'],rows)))
display(Image(filename=str(OUT/'figures/fluctuation_segment_controls.png')))''')
    md('''## 6. 判断与可复现记录
''')
    code('''for finding in decision['findings_zh']:display(Markdown(finding))
display(Markdown(decision['next_work_zh']))
for limit in decision['limitations_zh']:display(Markdown('边界：'+limit))
print('结果目录:',OUT)
print('计算：analyze_qdc_fluctuation.py prepare → analyze --workers 2')
print('报告：report_qdc_fluctuation.py；笔记：jupyter/codex_0001_build_qdc_fluctuation_notebook.py')
print('核验/封存：verify_qdc_fluctuation.py [--archive]')''')
    md('''`protocol.json`在新派生结果出现前冻结方案和输入哈希；`references/`保存对折参考；`arrays/`保存原ID、原Q及逐事件分解；`evaluation/`保存逐条件点值和全部配对区间；`analysis.json`汇总结果。

`comparison_table.csv`、`per_Q_table.csv`、`batch_table.csv`分别保存36行总体、162行逐Q箱和144行逐种子批点值，所有固定响应均保留。`verification.json`、`record.json`和源码快照记录核验及封存。共享文献库保持只读，本轮没有联网、下载或新实验。
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
