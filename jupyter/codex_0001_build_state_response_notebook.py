"""Build and execute the state-conditioned native waveform/PID research record."""
import os
import argparse
from pathlib import Path
PROJECT=Path(__file__).resolve().parents[1]
cache=PROJECT/'.cache/codex_0001_state_response_ipython';cache.mkdir(parents=True,exist_ok=True)
os.environ['IPYTHONDIR']=str(cache);os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import nbformat as nbf
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output


def main(refresh=False):
    path=PROJECT/'jupyter/codex_0001_csi_state_response.ipynb';assert refresh or not path.exists()
    cells=[]
    def md(s):cells.append(nbf.v4.new_markdown_cell(s))
    def code(s):cells.append(nbf.v4.new_code_cell(s))
    md('''# CsI(Tl)–SiPM：采集段条件校准与原生波形/PID联合检查

2026-10-30。承接[共享文献库更新核查](codex_0001_csi_library_update_20261030.ipynb)及[采集状态诊断](codex_0001_csi_acquisition_state.ipynb)，按用户确认推进。

本轮实际完成：原生孔径相位对照、仅前沿定位对照、各采集段内部的事件块交叉校准、受限读出时间常数/传输系数拟合，以及以完整PID协方差和三种时间尺度起伏共同约束的原生噪声重放。
**有局部改善，仍不能宣称实验与模拟的PID带宽全面一致。** 所有候选与失败对照均保留，默认参数及原PID定义没有替换。

笔记中的“源”表示采集条件；²⁴¹Am同时产生α、γ和衰变电子，实验主带不是逐事件粒子真值。
本轮没有新增实验数据，也没有新增独立Geant4输运母事件。
''')
    code('''from pathlib import Path
import json,sys
import numpy as np
from IPython.display import display,Markdown,Image
PROJECT=next(p for p in (Path.cwd(),*Path.cwd().parents) if (p/'simulation/main.cc').exists())
OUT=PROJECT/'simulation/results/state_response_20261030'
def read(p):return json.loads(Path(p).read_text())
def table(headers,rows):
    return '\\n'.join(['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|']+['|'+'|'.join(map(str,r))+'|' for r in rows])
def f(v,n=3):return '不可估计' if v is None else f'{v:.{n}f}'
def nums(v):return ' / '.join(f(x) for x in v)
def ci(r):return '不可估计' if r['CI95'] is None else '['+', '.join(f(x) for x in r['CI95'])+']'
p=read(OUT/'protocol.json');a=read(OUT/'alignment.json');e=read(OUT/'evaluation.json')
cal=read(OUT/'mean_calibration.json');decision=read(OUT/'decision.json');fold=read(OUT/'fold_mixture_diagnostic.json')
variants=p['variants'];labels=['原冻结参数','仅更新前触发电子学','RC+原残差谱','RC+扩展残差谱']
display(Markdown(decision['conclusion_zh']))
print('原生读出任务:',decision['native_jobs'],'；固定事件重放条目:',decision['native_selected_replay_entries'])
print('不同的原母事件:',decision['unique_original_mothers_with_fixed_waveforms'],'；支持Q箱内:',decision['n_supported_original_mothers'])
print('新增独立母事件:',decision['new_independent_mothers'],'；新增实验事件:',decision['new_experimental_events'])
print('Python / NumPy:',sys.version.split()[0],np.__version__)''')
    md('''## 1. 数据划分、模型边界与文献采用范围

保留run3（⁶⁰Co）和run12（²⁴¹Am）的原事件ID、原Q、共同Q区间、主带选择和冻结PCA。Co使用6个原Q箱，Am使用前3箱；箱内按first60的Q两分目标加权。
每段按`(原始事件ID // 块长) % 2`分两折，新参数只在对折拟合、在本折检查。块128为主，块64为敏感性检查；两种结果都保留。

first60、middle20和last20都已被探索。后两段本轮明确参与新的**条件校准**，不能继续叫作独立验证集。既有光学参数、PCA和Q匹配目标仍共享先前的校准信息；这次交叉划分仅隔离新增拟合层。
MC使用同一批原Geant4母事件保存的雪崩时刻和电荷，每段每种块长中，每个母事件只分配一套对折参数；不把重复重放视为新增独立统计。

新程序为`include/ReadoutSampling.hh`、`src/ReadoutSampling.cc`及`replay_readout.cc`。它只重算C++原生读出，几何、光学和SiPM雪崩物理冻结，输出树名为`Readout`。
前述470条固定波形来自Co8000、Am24000个原输运母事件；支持Q箱内为Co276、Am180条。本轮未重新运行这32000次输运。

本轮没有再联网或改动共享文献库。方法动机和原文核查范围见前置文献笔记：Hong的条件失配问题支持分段检查；Simpson和Kong支持检查定位/采样敏感性；Georgiev的IRF工作提醒模板与响应可识别性的限制；Calò用于读出电路条件的边界说明。**没有移用其他探测器的性能数字或未经本装置标定的电路参数**。实验均值模板只进入定位分析，未写入模拟发生器。
''')
    code('''rows=[]
for source,run in [('Co60',3),('Am241',12)]:
    for seg in ['first60','middle20','last20']:
        g=p['groups'][f'run{run}_{seg}'];fs=g['folds']['128']
        rows.append([source,seg,len(g['ids']),len(fs['0']['calibration_ids']),len(fs['0']['evaluation_ids']),g['span']])
display(Markdown(table(['源','段','固定实测主带数','评价折0的训练数','评价折0的检查数','原事件区间'],rows)))
print('原Q箱边界/ADC·sample:',p['q_edges'])
print('前置提交:',p['checkpoint'])''')
    md('''## 2. 原生采样相位与仅前沿定位

原生相位为0、0.5、1.0、1.5 ns，ADC周期及孔径均为2 ns，内部积分步长0.5 ns。直接移动原生孔径，保持雪崩历史和随机流配对；没有对含噪ADC进行插值。
零相位在全部470条固定波形上与旧输出逐ADC点一致。非零相位仅向末端延长一采样周期的无噪声积分缓冲，不生成新的雪崩；所有分析门均远离这段截断边界。
新增测试核对了解析孔径积分、零相位完整噪声流程、雪崩不变、配对电子学噪声和非法相位拒绝；本轮相关的5项C++测试通过。

实测中真实采样相位未知。用t₁₀的小数部分作为带噪代理，在Q箱截距/斜率和三项前触发状态回归上增加1、2次谐波，比较对折预测MSE；不把这个关联当作相位因果效应。

前沿模板取训练块中按Q加权、按原分数t₁₀对齐的平均形状。拟合每条原始ADC在0–200 ns中的位置，允许幅度/常数项作为辅助参数；位移只在±10 ns内、步长0.25 ns。模板可插值，实际PID积分继续用原始ADC。
**大量前沿拟合触边，说明这套窗口和自由参数下定位不可靠，未采用它修改PID。** 这是已预设对照的负结果，没有见到结果后放宽边界。
''')
    code('''rows=[]
for source in ['Co60','Am241']:
    for r in a['native_phase'][source]:
        rows.append([source,r['phase_ns'],f(r['mean_time_change_ns'],4),nums(r['IQR_ratio_to_phase0']),ci(r['conditional_whole_mother_intervals'][0])])
display(Markdown(table(['源','原生相位/ns','平均时间变化/ns','QDC/时间/PCA的IQR比','时间变化条件95%区间'],rows)))
rows=[]
for key,groups in a['groups'].items():
    r=groups['128'];rows.append([key,f(r['front_data_boundary_fraction']),f(r['front_MC_boundary_fraction']),nums(r['phase_regression_MSE_ratio'])])
display(Markdown(table(['数据组','实测前沿触边比例','MC前沿触边比例','加相位后QDC/时间/PCA预测MSE比'],rows)))
display(Image(filename=str(OUT/'figures/sampling_and_front_controls.png')))''')
    md('''受控原生相位下，平均时间变化绝对值最大约0.080 ns，远小于Am晚段原7.26 ns的点估计偏差。时间IQR改变约2%以内、全部PID方法最大约3.8%；这是本模型及这4种相位的结果，不能声称排除所有数字采样问题。
前沿方法的触边率约46%–88%，其时间差变化不能用来宣称波形物理得到修复。

## 3. 条件读出和信号噪声如何拟合

前触发噪声使用对折的原合格波形池，**在主带选择之前**取前128/256样点：基线/RMS混合状态由前128点拟合，最多两个有效分量；前256点去趋势、Hann谱估计257抽头FIR。沿用已声明的量化修正和首个正频点以下平坦外推；512 ns记录不能唯一确定微秒尺度低频噪声。

均值响应先比较原生无噪声RC网格96、112、128.215、144、160 ns；每项允许一个全局信号传输系数在0.45–0.55内拟合。目标同时包括76个40 ns绝对ADC均值、平均时间、短/长积分比和2–3 μs尾部电荷比例，权重在结果前固定于`fit_design.json`。
这是有限网格的条件提案，随后必须通过真正含噪声、使用自身t₁₀定位的原生ADC检查；不对输出的每条波形作事后幅度缩放。

残差谱有两类：仅缩放原谱；或原谱加400/1500 ns指数核的非负功率组合。拟合目标为**未经winsorize的完整3×3箱内PID协方差**和F50/F102/F202三项均值，不只拟合三个宽度。
F为中心盒滤波后，信号区[500,2500) ns方差减去前触发[−600,−200) ns方差，再除以信号均值；门相对实际分数t₁₀，含整数边界核验。提案用线性协方差及16次噪声复制估计F，最终比较均来自C++原生重放。
残差仍是有效高斯相关噪声，事件方差CV继承旧值；并未证明它等于CsI固有起伏或SiPM后脉冲。
''')
    code('''rows=[]
for key,r in cal['groups'].items():
    if r['block']!=128:continue
    fit=read(PROJECT/r['fit']);ref=np.array(fit['training_block_refits']);counts=np.bincount(ref[:,0].astype(int),minlength=5)
    rows.append([key,f(r['RC_ns']),f(r['transmission'],4),r['noise_states'],' / '.join(map(str,counts)),len(ref)])
display(Markdown(table(['条件/评价折','RC/ns','全局传输','前触发状态数','128次重拟合各RC被选次数','有效重拟合数'],rows)))
display(Image(filename=str(OUT/'figures/conditional_RC_stability.png')))''')
    md('''Co各段均选择原128.215 ns。Am晚段的主分析两折分别选择128.215和112 ns；其中一折的训练块重采样约一半选112、一半选128.215 ns。
这是粗网格选择和有限样本的敏感性，**不能推断读出电路确有两种时间常数**。块64中晚段两折都选112 ns，所以必须同时列出块长敏感性。

各训练块还有128次功率重拟合，保存在每个`*_signal.json`中；该检查固定RC、前触发拟合和MC，仅说明这一层功率的稳定性。RC稳定性与功率稳定性分开报告，不冒充整个流程重拟合后的置信区间。
''')
    code('''rows=[]
for key,r in cal['groups'].items():
    if r['block']!=128 or r['source']!='Am241':continue
    fit=read(OUT/'fits'/f'{key}_signal.json');z=fit['response_spectrum'];b=np.array(z['training_block_power_refits'])
    q=np.quantile(b,[.025,.975],axis=0)
    rows.append([key,nums(z['component_powers']),f(z['final_strength']),nums(q[0]),nums(q[1])])
display(Markdown(table(['Am条件/评价折','原谱/400/1500ns功率','合成后原生强度','条件重拟合2.5%','条件重拟合97.5%'],rows)))
print('功率单位是有效ADC残差方差/信号ADC，不是器件增益或光产额。')''')
    md('''## 4. 原生波形、中心、带宽和起伏的联合结果

所有比较都固定原ID和Q，不因参数改变重新筛掉不合意波形。以下宽度为支持Q箱内IQR的均方根，MC/实测的理想值为1。
平均时间差定义为MC−实测；波形NRMSE为76个绝对ADC门均值的L₂残差除以实测均值L₂范数。
显示512次原始事件块及完整MC母事件配对重采样的百分位区间。参数、模板、Q匹配目标和母事件折分固定，因此这些是**固定拟合的条件区间**；多项比较未做多重性校正。非线性NRMSE有正偏，百分位区间不一定包含原点估计。
''')
    code('''rows=[]
for key,blocks in e['groups'].items():
    for v,label in zip(variants,labels):
        r=blocks['128']['variants'][v]['point']
        rows.append([key,label,f(r['mean_time_delta_ns']),nums([r[n] for n in ['IQR_ratio_ratio','IQR_time_ratio','IQR_PCA_ratio']]),nums([r[n] for n in ['F50_ratio','F102_ratio','F202_ratio']]),f(100*r['mean_ADC_NRMSE'],2)+'%'])
display(Markdown(table(['数据组','模型','时间差/ns','QDC/时间/PCA的IQR比','F50/102/202比','平均ADC NRMSE'],rows)))
display(Image(filename=str(OUT/'figures/state_native_joint_checks.png')))''')
    code('''rows=[]
for B in ['128','64']:
    for v,label in zip(variants,labels):
        z=e['groups']['run12_last20'][B]['variants'][v]
        rows.append([B,label,f(z['point']['mean_time_delta_ns']),ci(z['CI95']['mean_time_delta_ns']),f(z['point']['IQR_time_ratio']),ci(z['CI95']['IQR_time_ratio']),ci(z['paired_change_from_frozen']['IQR_time_ratio'])])
display(Markdown(table(['块长','Am晚段模型','时间差/ns','条件95%区间/ns','时间IQR比','条件95%区间','相对原模型IQR比变化区间'],rows)))
print('主分析扩展谱的配对平均时间变化95%条件区间/ns:',decision['mean_time_paired_change_conditional_CI95_ns'])''')
    md('''Am晚段（块128）扩展谱方案把平均时间差从7.26降到1.41 ns，时间IQR比从0.917变为0.978，绝对平均波形NRMSE从1.97%变为1.36%；F50/102/202比分别约0.999、0.984、0.989。
但时间IQR比的配对变化区间仍包含0，不能仅凭点估计称带宽显著改善或等效。新中心差的条件区间约[−5.80,8.02] ns，也不是“证明完全一致”。
同一方案下QDC/PCA的IQR比分别约1.134/1.174；块64下平均时间差约−2.99 ns。单纯把时间带宽调到1附近仍未保证全PID分布匹配。

“仅更新前触发电子学”后晚段时间差仍约7.12 ns，说明这项噪声更新本身没有解释中心偏差；RC响应调整才明显改变中心。不能据此把剩余差异全部归于一种噪声。
''')
    code('''display(Image(filename=str(OUT/'figures/state_native_waveforms.png')))
display(Image(filename=str(OUT/'figures/state_PID_covariance.png')))''')
    md('''上图同时检查绝对波形和40 ns电荷份额的波动。后者经每事件Q归一化，总和为1，协方差中的反相关不应直接解释为晶体微观反相关。
完整PID协方差揭示了权衡：扩展残差谱缓解平均时间方差偏小，但晚段QDC/PCA方差随之偏大。平均波形外观接近、某个IQR比接近1，都不能代替这项检查。

## 5. 两折参数混合是否制造了改善

看到主结果后追加一个**只读、描述性的事后诊断**：在每个Q箱按评价折精确拆分总协方差为折间均值项与折内项。没有更改拟合、事件集合或候选排名。
折间均值项既含有限样本噪声，也可能含两套拟合响应的差别，不能把它当作已测物理漂移。
''')
    code('''rows=[]
for B in ['128','64']:
    z=fold['groups']['run12_last20'][B]
    for v,label in zip(variants,labels):
        r=z['variants'][v]
        rows.append([B,label,f(100*z['data_between_fold_fraction'][1],2)+'%',f(100*r['between_fold_fraction'][1],2)+'%',f(e['groups']['run12_last20'][B]['variants'][v]['point']['SD_time_ratio']),f(r['within_fold_SD_ratio'][1])])
display(Markdown(table(['块长','Am晚段模型','实测折间时间方差份额','MC折间份额','总时间SD比','去折间后的时间SD比'],rows)))
print('诊断注册时的评价文件SHA256:',fold['registered_after_evaluation_sha256'])''')
    md('''主分析中，扩展谱的MC折间项约占时间方差3.4%，原参数约3.1%。去掉折间项后，时间SD比从原约0.820改善到0.913，改善并不主要由折间参数混合制造，但点估计仍偏窄。这里没有另行拟合或给这项事后分解加“独立验证”的标签。

## 6. 结论、下一步和复现

本轮完成了文献建议对应的对照和条件校准；没有得到可直接替代默认模型的统一参数。
候选还存在跨块长中心不稳定、三种PID宽度权衡和粗RC网格限制，因此本轮没有进入新Geant4母事件的最终验证阶段。重复使用的26790条原生读出结果始终只对应470个不同的原母事件。
''')
    code('''display(Markdown(decision['next_work_zh']))
display(Markdown('\\n\\n'.join(decision['limitations_zh'])))
print('默认参数已替换:',decision['promoted_to_default'])
print('复现顺序: prepare_state_response.py → state_response_common.py → analyze_state_alignment.py / calibrate_state_response.py → fit_state_signal_noise.py → evaluate_state_response.py')
print('补充诊断和记录: diagnose_state_fold_mixture.py → plot_state_response.py → summarize_state_response.py')
print('笔记: jupyter/codex_0001_build_state_response_notebook.py')
print('核验/归档: simulation/verify_state_response.py；随后 --archive')
print('C++相关测试: ctest --test-dir simulation/build/state_response_20261030 --output-on-failure -E csi_application_workflow')
print('已有输出拒绝覆盖；新完整研究应指定新结果目录或修改统一OUT后冻结新方案。当前不自动commit/push。')''')
    md('''复现工件位于`simulation/results/state_response_20261030/`：`protocol.json`和`fit_design.json`为前置方案；`configs/`、`native/`和`arrays/`为参数、原生ROOT及提取；`fits/`记录各折训练ID和128次条件重拟合；`evaluation_bootstrap.npz`保留配对抽样；`verification.json`与`record.json`记录核验和哈希。

软件读取中曾遇到uproot数组读取停滞，未改变任何数值方案；改用已验证的PyROOT接口读取已完整写出的ROOT，随后完成逐点ADC复现。旧读取脚本快照保留在`source_snapshot/`，不是另一组研究结果。
''')
    nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}})
    shell=InteractiveShell.instance()
    for cell in nb.cells:
        if cell.cell_type!='code':continue
        with capture_output() as capture:result=shell.run_cell(cell.source,store_history=True)
        if result.error_before_exec or result.error_in_exec:raise RuntimeError(result.error_before_exec or result.error_in_exec)
        cell.execution_count=shell.execution_count-1;outputs=[]
        if capture.stdout:outputs.append(nbf.v4.new_output('stream',name='stdout',text=capture.stdout))
        if capture.stderr:outputs.append(nbf.v4.new_output('stream',name='stderr',text=capture.stderr))
        for o in capture.outputs:outputs.append(nbf.v4.new_output('display_data',data=o.data,metadata=o.metadata))
        cell.outputs=outputs
    nbf.validate(nb);nbf.write(nb,path);print(path)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--refresh',action='store_true');args=ap.parse_args();main(args.refresh)
