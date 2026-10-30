"""Build and execute the conditional PID / native sensor response notebook."""
import os
from pathlib import Path
PROJECT=Path(__file__).resolve().parents[1]
cache=PROJECT/'.cache/codex_0001_conditional_response_ipython';cache.mkdir(parents=True,exist_ok=True)
os.environ['IPYTHONDIR']=str(cache)
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import nbformat as nbf
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output


def main():
    path=PROJECT/'jupyter/codex_0001_csi_conditional_response.ipynb'
    assert not path.exists(),'Do not overwrite an existing research notebook'
    cells=[]
    def md(text):cells.append(nbf.v4.new_markdown_cell(text))
    def code(text):cells.append(nbf.v4.new_code_cell(text))
    md('''# CsI(Tl)–SiPM：电荷/基线条件化与后脉冲、恢复时间的原生对照

2026-10-30。承接[文献检索](codex_0001_csi_pid_bandwidth_literature_web.ipynb)与[平均波形/PID联合约束](codex_0001_csi_joint_constraints.ipynb)。本轮把文献建议变成两项可检查的诊断：

1. 在同一电荷区间内，箱内电荷分布和基线状态差异是否改变 PID 带宽结论？
2. 原生 SiPM 后脉冲或像元恢复时间能否提供所缺的波动，同时维持波形中心、均 ADC 和短尺度起伏？

本轮只用已有实测波形和已保存的 Geant4 光子历史。12 组 C++ 响应重放共 192000 条记录，但只有 **32000 个重复使用的母事件，新增独立输运母事件为 0**。以前称为验证的 MC 现在已用于诊断探索，不能继续称为本轮独立验证。
实测 first60 参与过校准；middle20/last20 也已多次探索，只作内部检查。没有本轮新增实验，没有实验粒子真值；²⁴¹Am 包含 α、γ 和衰变电子。
''')
    code('''from pathlib import Path
import json,sys
import numpy as np
from IPython.display import display,Markdown,Image
PROJECT=next(p for p in (Path.cwd(),*Path.cwd().parents) if (p/'simulation/main.cc').exists())
OUT=PROJECT/'simulation/results/conditional_response_20261030'
def read(p):return json.loads(Path(p).read_text())
def table(headers,rows):
    return '\\n'.join(['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|']+['|'+'|'.join(map(str,r))+'|' for r in rows])
def f(x,places=3):return '不可估计' if x is None else f'{x:.{places}f}'
def ci(r):return '不可估计' if r['CI95'] is None else '['+', '.join(f(x) for x in r['CI95'])+']'
def nums(x):return ' / '.join(f(v) for v in x)
p=read(OUT/'protocol.json');c=read(OUT/'conditional.json');m=read(OUT/'mechanisms.json')
s=read(OUT/'supplement.json');decision=read(OUT/'decision.json')
print('Python / NumPy:',sys.version.split()[0],np.__version__)
print('原生重放记录:',p['replay_entries'],'不同原母事件:',p['unique_reused_mothers'],'新增独立母事件:',p['new_independent_mothers'])
display(Markdown(decision['conclusion_zh']))''')
    md('''## 1. 固定分析定义与对照参数

装置假设沿用上轮：10×10×10 mm³ CsI(Tl)、PTFE 包裹、5 μm Mylar、S13361-3050 系列多单元求和、无前放。模拟中的16单元阵列、名义 PDE=0.3、增益1.7×10⁶和源距仍是条件假设；实际求和通道数、电路和温度信息不完整。

PID 用原定义：前128样点估计基线，2 ns采样，整数锚点 floor(t₁₀)+1，长门−40至3000 ns、短门−40至500 ns，以及冻结 PCA。平均时间分数存为负微秒，以下差值已换为正常时间方向的 ns。
共同 Q 区间分为原来的6箱；Co用全部6箱，Am只用前3箱。宽度为各箱 IQR 的平方平均再开方，各原箱等权。实验主带诊断沿用 QDC=0.4 分界，不用它宣称实验 α/γ 真值。

下表中的 AP 概率是代码中每次已接受雪崩的**提议概率参数**，延迟是指数分布均值；还会受恢复程度和次级接受过程影响，因此它不等于最终后脉冲占比。
5%、15%、60/300 ns和恢复60 ns是预设敏感性/压力对照，**不是从文献套来的本器件实测数值**。原名义恢复时间为29 ns。
''')
    code('''rows=[]
for key,e in p['entries'].items():
    cfg=read(PROJECT/e['config']);z=cfg['sipm']['sensor'];r=read(OUT/'extraction'/f'{key}.json')
    v=read((PROJECT/e['path']).with_name(Path(e['path']).stem+'_validation.json'))
    rows.append([e['source'],e['variant'],z['afterpulse_probability'],z['afterpulse_delay_ns'],z['recovery_ns'],
        f(cfg['signal_dependent_noise']['variance_per_signal_adc']),e['events'],r['n_common'],r['gained'],r['lost'],v['passed']])
display(Markdown(table(['源','对照','AP提议概率','AP延迟/ns','恢复/ns','经验信号残差强度','母记录数','重新筛选共同Q数','进入/条','离开/条','物理检查'],rows)))
print('共同Q边界/ADC·sample:',p['q_edges'])
print('支持箱，0起算:',p['support_bins'])''')
    md('''原参数 identity 保留上轮全部设置；其余五组首先关闭经验信号期残差，再单独改变相应原生机制，避免把同一缺失波动重复补两次。
每组保留原母事件/种子及到达光子的时间、位置、波长历史，固定光学、PDE、增益、SPE及被动读出。关闭经验残差只用于消融；不是声称实际器件没有这类起伏。
恢复60 ns这组只改模型中的像元恢复过程，SPE下降时间仍固定29 ns；这是一项局部灵敏度检查，不表示实际器件的恢复、增益和脉冲响应可以彼此独立调整。

空间串扰继续为0：现有阵列展开映射没有保存二维像元邻接关系，代码明确禁止在这种映射下开启空间串扰。暗计数本轮同样固定0，未由本次对照排除其实际存在。
原生后脉冲与恢复变化可能改变后续一次光子雪崩是否被接受；共用光子历史不意味着每组有完全相同的雪崩。
新增 `ResponseDiagnostics` 独立 ROOT 树保存实际一次/AP/暗/串扰计数及恢复程度，不改变 Events 格式或随机数序列。原参数重放要求全部32000个母事件的 ADC 和雪崩时间/电荷逐点完全相同。
''')
    md('''## 2. 箱内分布匹配：固定 first60 目标

在每个原 Q 箱内，用 first60 实验数据冻结中位数/四分位数，比较四种权重：原箱等权、箱内 Q 两分、Q 四分、Q 两分×基线 RMS 两分。
每个组合单元的权重为“first60目标质量/该样本单元事件数”；不删事件、不截权、不对空单元外推。缺少正目标质量的单元时，该项整体不可估计。
因此匹配后的统计量仍是**原 Q 箱内的加权分布**，不是更窄的积分窗口，也不是消除了所有电荷依赖。

加权分位数使用端点归一化的权重中点 CDF，等权时与 NumPy type7 分位数一致。bootstrap 的整数重复次数先展开，再做匹配。
原生 MC 用512次完整原母事件重采样；实验用原始事件索引的连续块重采样，块长64和128都报告。first60切点、目标质量、旧拟合及装置假设固定，区间没有包含这些量的不确定性。
重采样出现空单元则记无效；至少128个有效抽样才列区间，且区间只基于有效抽样。有效次数少于512提示稀疏支持限制，不能解释为无偏的状态校正。
''')
    code('''rows=[]
for run,source in [(3,'Co60'),(12,'Am241')]:
    for mode,a in c['groups'][f'run{run}_last20']['modes'].items():
        u=a['uncertainty']['64'];u128=a['uncertainty']['128']
        rows.append([source,mode,f(a['mean_time_difference_ns']),ci(u['mean_time']),nums(a['IQR_ratio']),
            ci(u['IQR_ratio'][1]),ci(u128['IQR_ratio'][1]),u['IQR_ratio'][1]['valid_draws']])
display(Markdown(table(['源/后段','权重','MC−实测平均时间/ns','块64区间/ns','QDC/时间/PCA宽度比','时间宽度比区间64','区间128','有效抽样/512'],rows)))
display(Image(filename=str(OUT/'figures/conditional_pid.png')))''')
    md('''Co 在不同权重下的中心和宽度差别较小。Am 后段的时间宽度比点估计从0.887变为Q两分匹配的0.917；**匹配前后变化的配对区间包含0，不能声称已确认改善**。
匹配后 Am 时间中心仍约+7.26 ns。时间宽度比区间覆盖1只说明当前精度未排除1，不是等效性证明。

Am 后段 Q×RMS 匹配在第一个原 Q 箱缺少目标单元，结果标为不可估计；没有为了获得一个数字而缩小研究对象或人为填充。
下图和表保留每箱有效样本量 ESS=(Σw)²/Σw²；Co全共同区有276条原模拟波形，Am有194条，但Am进入正式3个支持箱的只有180条。
''')
    code('''rows=[]
for run,source in [(3,'Co60'),(12,'Am241')]:
    for segment in ['first60','last20']:
        for mode in ['q2','q4','q2r2']:
            a=c['groups'][f'run{run}_{segment}']['modes'][mode]
            rows.append([source,segment,mode,f(a['MC_support'].get('minimum_ESS'),1),
                f(a['data_support'].get('minimum_ESS'),1),a['data_support']['valid'],
                ci(a['uncertainty']['64']['paired_change_from_original'][2])])
display(Markdown(table(['源','段','权重','MC最小ESS','实测最小ESS','实测支持充分','时间宽度比相对原权重变化区间'],rows)))
display(Image(filename=str(OUT/'figures/conditional_support.png')))''')
    md('''## 3. 采集段变化是否由箱内电荷分布解释

这里仅比较实测后段相对前段，不含 MC，不将变化归因于某个器件机制。
Q两分匹配后 Am 后段平均时间仍提前约5.02 ns：块64区间略不含0，块128包含0；Q四分的点估计为−5.70 ns，两个块长的区间均不含0。
这些是已探索样本上的多项描述性诊断，显示段差异并未随更细Q匹配消失；不能据此确定温漂、电路漂移、选择效应或闪烁过程哪个占主导。
''')
    code('''rows=[]
for run,source in [(3,'Co60'),(12,'Am241')]:
    for mode in ['original','q2','q4']:
        r=s['data_segment_changes'][f'run{run}_last20__{mode}']
        rows.append([source,mode,f(r['data_later_minus_first60_mean_time_ns']),ci(r['uncertainty']['64']),ci(r['uncertainty']['128'])])
display(Markdown(table(['源','权重','实测后段−前段平均时间/ns','块64区间','块128区间'],rows)))''')
    md('''## 4. 原生后脉冲/恢复机制与事件选择分开检查

所有机制效应都相对于 `no_signal_residual`，并采用同一 first60 Q两分目标。512次完整原母事件重采样在所有组之间配对。

- **固定原ID/Q**：保留原来入选的母事件与原 Q 分箱，但使用新 ADC 自身的t₁₀；这是主要机制对照。
- **重新筛选/自身Q**：重新跑完整质量与电荷选择，揭示选择迁移的共同影响。
- **固定原ID/Q/锚点**：另用原整数锚点算PID，检查对齐变化的贡献。

相同记录不重复增加样本量。固定ID结果即使新Q移出原窗也保留；重新筛选结果则按新Q判断。三组回答不同问题，不能挑最接近实验的一组作为验证成功。
''')
    code('''rows=[]
for source,z in m['sources'].items():
    for variant,r in z['variants'].items():
        a=r['schemes']['fixed'];delta=a['change_from_no_residual'];u=a['paired_effect_CI95']
        rows.append([source,variant,f(delta[0]),ci(u[0]),nums(delta[1:]),ci(u[2])])
display(Markdown(table(['源','对照','平均时间变化/ns','配对区间/ns','QDC/时间/PCA宽度相对残差关闭','时间宽度比配对区间'],rows)))
display(Image(filename=str(OUT/'figures/native_response_effects.png')))''')
    md('''## 5. 实际雪崩计数、平均波形与短尺度起伏

下表后脉冲占比和恢复指标直接来自原生 C++ 雪崩结果。计数以全部固定原共同Q事件汇总（Co276、Am194）；而PID统计只在支持箱内进行（Am180），两者范围不同。
平均恢复值是每事件雪崩平均恢复系数再按事件等权平均；“恢复<0.99”比例与AP比例按雪崩数汇总。还核对了 Σ各类雪崩计数=总雪崩数，以及平均恢复×雪崩数×增益×元电荷=总电荷。

波形图使用固定原ID/Q、Q两分匹配的原生 ADC 均值，没有单事件幅度缩放。F50/F102/F202为中心盒平滑后的信号期减前触发残差方差除以均信号，**不是光子或SiPM增益的Fano因子测量**。
图中的均波形/F对照与first60相比，仅列描述性点估计；不以其进行本轮拟合或模型排名。
''')
    code('''rows=[]
for source,z in m['sources'].items():
    for variant,r in z['variants'].items():
        e=r['extraction'];a=r['schemes']['fixed'];extra=s['waveform_checks'][source][variant]
        rows.append([source,variant,f(100*e['fixed_afterpulse_avalanche_fraction']),f(e['fixed_mean_recovery'],5),
            f(100*e['fixed_fraction_avalanches_recovery_below_099']),f(a['observed_Q_ratio_to_original'],4),
            f(100*extra['mean_ADC_NRMSE_vs_first60']),nums(extra['F_ratio_vs_first60'])])
display(Markdown(table(['源','对照','实际AP占比/%','事件平均恢复','恢复<0.99雪崩/%','新Q/原Q','均ADC NRMSE/%','F50/102/202比'],rows)))
display(Image(filename=str(OUT/'figures/native_response_waveforms.png')))''')
    md('''平均时间带宽只是三个PID指标之一。下面还保留完整3×3协方差变化（顺序QDC、负平均时间、PCA），以first60实测各维标准差乘积归一化。
协方差在原Q箱内按权重计算后等箱平均，采用总体分母，没有把观测协方差直接宣称为独立物理噪声分量。
原Q箱内还存有“子单元均值之间”协方差，可检查总协方差=子单元内协方差+子单元间协方差；这个统计恒等式不构成因果机制分解。
''')
    code('''rows=[]
for source,z in m['sources'].items():
    for variant in ['identity','ap05_60ns','ap05_300ns','ap15_300ns','recovery60ns']:
        x=np.asarray(z['variants'][variant]['schemes']['fixed']['covariance_increment_normalized_data_SD_products'])
        rows.append([source,variant,*[f(x[i,j]) for i,j in [(0,0),(0,1),(0,2),(1,1),(1,2),(2,2)]]])
display(Markdown(table(['源','对照相对残差关闭','QQ','QT','QP','TT','TP','PP'],rows)))''')
    md('''## 6. 本轮判断、下一步和复现

本轮没有寻找“最佳AP概率”，没有为新机制补偿增益或再次拟合材料时间常数。光学、无前放电路和像元模型仍存在退化；对照结果不能单独确定实际SiPM噪声率或CsI固有涨落。
继续保持原默认配置及论文主分析，既有ROC和run18结果未更新。需要补充的是模拟与已有数据的约束，而不是把其它器件或其它实验的论文成绩直接套用。
''')
    code('''display(Markdown(decision['next_work_zh']))
print('默认配置替换:',decision['promoted_to_default'])
print('冻结方案:',OUT/'protocol.json')
print('实际原生命令/日志:',OUT/'execution.json')
print('12行固定事件效应表:',OUT/'mechanism_table.csv')
print('阶段顺序: prepare_conditional_response.py → run_conditional_response.py → analyze_conditional_response.py conditional / mechanisms → plot_conditional_response.py')
print('独立数值核查和归档: simulation/verify_conditional_response.py；随后 --archive')
print('构建目录:',PROJECT/'simulation/build/conditional_response_20261030')
print('四项 C++ 检查: inherited_sensor_charge_recovery_rng, stationary_readout_noise, parallel_readout_response, signal_dependent_readout_noise')''')
    nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.13.5'}})
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


if __name__=='__main__':main()
