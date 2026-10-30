"""Build and execute the continuous RC / noise-distribution research note."""
import os
import argparse
from pathlib import Path
PROJECT=Path(__file__).resolve().parents[1]
cache=PROJECT/'.cache/codex_0001_refined_rc_ipython';cache.mkdir(parents=True,exist_ok=True)
os.environ['IPYTHONDIR']=str(cache);os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import nbformat as nbf
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output


def main(refresh=False):
    path=PROJECT/'jupyter/codex_0001_csi_refined_rc.ipynb';assert refresh or not path.exists()
    cells=[]
    def md(s):cells.append(nbf.v4.new_markdown_cell(s))
    def code(s):cells.append(nbf.v4.new_code_cell(s))
    md('''# CsI(Tl)–SiPM：连续RC与噪声分布联合校准

2026-10-30，承接[采集段条件响应](codex_0001_csi_state_response.ipynb)，按用户“继续”推进。
本轮细化RC并以完整PID协方差、核心IQR、波形起伏均值与事件间起伏共同约束噪声。
**全部实验已经探索，本轮是条件校准与内部检查，没有新增独立实验或Geant4输运母事件。**
''')
    code('''from pathlib import Path
import json,sys
import numpy as np
from IPython.display import display,Markdown,Image
PROJECT=next(p for p in (Path.cwd(),*Path.cwd().parents) if (p/'simulation/main.cc').exists())
OUT=PROJECT/'simulation/results/rc_refinement_20261030'
def read(p):return json.loads(Path(p).read_text())
def table(headers,rows):
    return '\\n'.join(['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|']+['|'+'|'.join(map(str,r))+'|' for r in rows])
def f(x,n=3):return '不可估计' if x is None else f'{x:.{n}f}'
def nums(x):return ' / '.join(f(v) for v in x)
def ci(r):return '不可估计' if r['CI95'] is None else '['+', '.join(f(v) for v in r['CI95'])+']'
p=read(OUT/'protocol.json');cal=read(OUT/'mean_calibration.json');select=read(OUT/'noise_selection.json')
e=read(OUT/'evaluation.json');decision=read(OUT/'decision.json');cache=read(OUT/'cache_verification.json')
variants=e['variants'];labels=['原冻结参数','粗RC+扩展谱','连续RC/继承噪声','连续RC+原谱/原CV','连续RC+原谱/拟合CV','连续RC+扩展谱/原CV','连续RC+扩展谱/拟合CV']
display(Markdown(decision['conclusion_zh']))
print('新研究用原生任务/事件条目:',decision['native_jobs'],decision['native_replay_entries'])
print('不同原母事件470，支持Q箱内456；新增独立母事件0。')
print('缓存工程核验:',cache['paired_event_cases'],'个逐点相同的配对事件对照。')
print('Python / NumPy:',sys.version.split()[0],np.__version__)''')
    md('''## 1. 固定数据和划分

run3为⁶⁰Co源、run12为²⁴¹Am源；Am同时有α、γ和衰变电子，实验主带不作为粒子真值。
保留原ID、原Q、冻结PCA及first60箱内Q两分目标，Co支持6个Q箱，Am支持前3箱。470条固定MC波形来自原8000/24000个输运母事件，支持区内为276/180条。
每个first60/middle20/last20段按`(原事件ID // 块长) % 2`分两折，对折拟合、本折检查；块128为主，64为敏感性分析。MC沿用上一轮母事件折分，每母事件每组比较只选一套响应。

前触发状态拟合继承上一轮。既有光学参数、PCA和Q目标仍共享先前校准信息；交叉划分只隔离新增拟合层，不是全流程独立验证。
本轮C++从保存的Geant4雪崩历史重算读出，几何、光学、传感器物理冻结；完整输运入口和`include/`、`src/`类结构保留。
''')
    code('''rows=[]
for key,g in p['groups'].items():
    fs=g['folds']['128'];rows.append([key,len(g['ids']),len(fs['0']['calibration_ids']),len(fs['0']['evaluation_ids']),g['span']])
display(Markdown(table(['组','原主带数','评价折0训练数','评价折0检查数','原事件区间'],rows)))
print('原Q箱边界:',p['q_edges'])''')
    md('''## 2. 连续RC与含噪声定位

范围保持96–160 ns：原生无噪声网格步长2 ns，再加入原128.21494566 ns点；复用10项旧输出，新算58项。
只对汇总的76个40 ns绝对ADC门均值、平均时间、短/长积分比及2–3 μs尾部比例作PCHIP插值。0.25 ns全范围搜索后局部有界优化，全局传输系数限于0.45–0.55；目标尺度沿用上一轮。
**插值仅提出参数，实际比较的每条ADC都由C++在新RC下生成。** 无逐事件缩放或实测模板注入。

随后执行两轮预设的原生含噪声偏差校正：用当前含噪声汇总响应减去无噪声预测，局部校正提案，再在原范围拟合。按训练均值目标选第0/1/2轮最低损失者。
多项均值约束可能互相权衡，不按时间中心一项挑最优。128次训练块重拟合固定原生偏差和其它参数，仅报告条件稳定性。
''')
    code('''rows=[]
for key,r in cal['groups'].items():
    if r['entry']['block']!=128:continue
    z=read(PROJECT/r['fit']);b=np.array(z['training_block_refits'])
    rows.append([key,f(r['RC_ns']),f(r['transmission'],5),r['selected_round'],nums(np.quantile(b[:,0],[.025,.5,.975])),len(b)])
display(Markdown(table(['条件/评价折','RC/ns','传输','迭代','RC条件重拟合2.5/50/97.5%','有效次数'],rows)))
display(Image(filename=str(OUT/'figures/continuous_RC_calibration.png')))''')
    md('''这些是有效响应参数，不能当作实测电阻/电容或温度。连续RC和原生定位校正共同构成新流程，与上一轮的变化不能全部归于单一原因。

## 3. 噪声协方差、核心和尾部

先保留原谱、原谱+400/1500 ns指数核两个正功率谱族，以完整3×3箱内PID协方差和F50/F102/F202均值拟合非负功率。
F是盒滤波残差在信号区[500,2500) ns的方差减去前触发[−600,−200) ns方差，除以信号均值；相对实际分数t₁₀，定义及整数边界沿用上一轮。

再扫描事件方差乘子的CV：原值和0、0.25、0.5、1、1.5、2。乘子服从均值1的对数正态分布，噪声RMS乘其平方根；改变的是有效残差分布，不是光产额、SiPM增益或逐事件信号幅度。
各谱族只用训练块选CV。损失同时包含原均值目标、完整PID协方差、三项F均值、三种PID的IQR、三项F的箱内标准差；尺度事先固定。协方差约束总方差，IQR约束核心，F标准差限制极端噪声状态。
全部CV候选与原CV对照保留。128次CV选择重采样固定RC、功率、谱、电子学和MC；功率重拟合另存，均不是完整嵌套不确定度。
''')
    code('''rows=[]
for key,r in select['groups'].items():
    if '_B128_' not in key:continue
    fit=read(PROJECT/r['fit'])
    for family,z in fit['families'].items():
        counts=np.bincount(z['conditional_selection_refits'],minlength=len(z['choices']))
        rows.append([key,family,f(z['inherited_CV']),f(z['chosen_CV']),' / '.join(f(v['CV'],2) for v in z['choices']),' / '.join(map(str,counts))])
display(Markdown(table(['条件/评价折','谱族','原CV','选择CV','CV序列','条件重选计数'],rows)))
display(Image(filename=str(OUT/'figures/variance_scatter_selection.png')))''')
    md('''新增`replay_readout_cached`复用最近一套原生基础响应，每个残差候选仍按原事件种子生成。读出/电子学/相位/窗口变化时缓存失效。
缓存只优化计算：216个配对事件覆盖CV、谱形、关闭残差、RC、相位和电子学变化，ADC/无噪声信号/模拟电压与旧程序逐值相同；普通重放入口也完成编译。未新增物理近似。

## 4. 原生联合检查

时间差为MC−实测；IQR/SD/F比值为MC/实测。NRMSE为76个绝对ADC均值的L₂残差除以实测均值L₂范数。
512次完整原母事件和原始实验事件块配对重采样，固定拟合、CV选择、Q目标、PCA及折分。区间是固定拟合的条件区间，未包含完整模型选择误差，也未作等效性检验。
不按新的Q、波形形状或结果重新筛选事件。每条新ADC仍按原定义重新计算基线与t₁₀锚点，使用原Q只做固定目标匹配；比值接近1和区间覆盖1都不等于证明一致。
''')
    code('''rows=[]
for key,blocks in e['groups'].items():
    for v in ['frozen','coarse_spectrum','mean_only','inherited_CV_fit','spectrum_CV_fit']:
        r=blocks['128']['variants'][v]['point']
        rows.append([key,labels[variants.index(v)],f(r['mean_time_delta_ns']),nums([r[n] for n in ['IQR_ratio_ratio','IQR_time_ratio','IQR_PCA_ratio']]),nums([r[n] for n in ['SD_ratio_ratio','SD_time_ratio','SD_PCA_ratio']]),f(100*r['mean_ADC_NRMSE'],2)+'%'])
display(Markdown(table(['组','模型','时间差/ns','QDC/时间/PCA IQR比','QDC/时间/PCA SD比','ADC NRMSE'],rows)))
display(Image(filename=str(OUT/'figures/refined_joint_checks.png')))''')
    code('''rows=[]
for B in ['128','64']:
    for v,label in zip(variants,labels):
        r=e['groups']['run12_last20'][B]['variants'][v]
        rows.append([B,label,f(r['point']['mean_time_delta_ns']),ci(r['CI95']['mean_time_delta_ns']),f(r['point']['IQR_time_ratio']),ci(r['CI95']['IQR_time_ratio'])])
display(Markdown(table(['块长','Am晚段模型','时间差/ns','条件95%区间/ns','时间IQR比','条件95%区间'],rows)))
display(Markdown(decision['block_stability_zh']))''')
    code('''rows=[]
for source,run in [('Co60',3),('Am241',12)]:
    for v,label in zip(variants,labels):
        r=e['groups'][f'run{run}_last20']['128']['variants'][v]['point']
        rows.append([source,label,nums([r[n] for n in ['F50_ratio','F102_ratio','F202_ratio']]),nums([r[n] for n in ['F50_SD_ratio','F102_SD_ratio','F202_SD_ratio']]),f(r['covariance_normalized_RMSE'])])
display(Markdown(table(['晚段源','模型','F均值比50/102/202','F标准差比50/102/202','归一化协方差RMSE'],rows)))
rows=[]
for source,run in [('Co60',3),('Am241',12)]:
    for v in ['inherited_CV_fit','spectrum_CV_fit']:
        r=e['groups'][f'run{run}_last20']['128']['variants'][v]
        rows.append([source,v,r['control_comparator'],ci(r['paired_from_control']['IQR_time_ratio']),ci(r['paired_from_control']['covariance_normalized_RMSE'])])
display(Markdown(table(['源','CV选择','同谱对照','时间IQR比配对变化区间','协方差误差配对变化区间'],rows)))''')
    code('''display(Image(filename=str(OUT/'figures/refined_native_waveforms.png')))
display(Image(filename=str(OUT/'figures/refined_PID_covariance.png')))''')
    md('''40 ns电荷份额经每事件Q归一化，总和1；协方差含归一化影响，不直接等于CsI本征波动。
评价折间/折内的协方差恒等式仍保留，折间项含有限样本均值噪声，不等于物理漂移。原CV和拟合CV的对照使用相同RC/功率/谱，配对区间从同一重采样中计算。

## 5. 结论、下一步和复现
''')
    code('''display(Markdown(decision['conclusion_zh']))
for finding in decision['findings_zh']:display(Markdown(finding))
display(Markdown(decision['next_work_zh']))
print('默认配置已替换:',decision['promoted_to_default'],'；新增独立母事件:',decision['new_independent_mothers'])
print('复现: rc_refinement_common.py → calibrate_refined_rc.py → verify_rc_cache.py → fit_refined_noise.py → evaluate_refined_rc.py')
print('记录: plot_refined_rc.py → summarize_refined_rc.py → jupyter/codex_0001_build_refined_rc_notebook.py')
print('核验与归档: verify_refined_rc.py；随后 --archive')
print('完整结果:',OUT)''')
    md('''`protocol.json`在新数值之前冻结方案，准备代码存于`source_snapshot/rc_refinement_common_frozen.py`。随后采用的原生缓存只优化计算，其采用前核验见`cache_verification.json`。
`fits/`保存训练ID、连续拟合、全部CV和条件重拟合；`native/`是原生ROOT，`arrays/`含ADC及提取；配对抽样保存在`evaluation_bootstrap.npz`。最终`verification.json`和`record.json`记录独立核验及输入/输出/源码哈希。
新研究不得覆盖当前结果。共享文献库未修改；本轮未联网、下载论文或进行新实验。
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
