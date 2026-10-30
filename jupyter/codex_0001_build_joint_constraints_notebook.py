"""Build and execute the bounded-family joint native waveform/PID notebook."""
import os
from pathlib import Path
PROJECT = Path(__file__).resolve().parents[1]
cache = PROJECT/'.cache/codex_0001_joint_constraints_ipython'; cache.mkdir(parents=True, exist_ok=True)
os.environ['IPYTHONDIR'] = str(cache)
os.environ['OPENBLAS_NUM_THREADS'] = '1'; os.environ['OMP_NUM_THREADS'] = '1'
import nbformat as nbf
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output


def main():
    path = PROJECT/'jupyter/codex_0001_csi_joint_constraints.ipynb'
    assert not path.exists(), 'Frozen notebooks must not be overwritten'
    cells = []
    def md(s): cells.append(nbf.v4.new_markdown_cell(s))
    def code(s): cells.append(nbf.v4.new_code_cell(s))
    md('''# CsI(Tl)–SiPM：平均波形、PID 中心/带宽与短尺度起伏的联合约束

2026-10-29。先提交并推送英文提交 `f7a840c`（[尾部研究](codex_0001_csi_co_tail_refit.ipynb)），再开展本轮。
上一轮改善了 Co 的尾部，却留下 Am 平均时间带宽偏窄、α 慢时间触及15 μs边界的问题。

本轮固定 **4、6、8.4386 μs** 三组α慢时间，在原前60%实验数据中联合约束中心、三种PID宽度/协方差、均ADC与短尺度起伏。
先运行 **96000次独立母衰变** 做三组完整Geant4校准，再对同一批母事件做原生噪声重放；这些重放不增加独立统计量。
按预先冻结的目标选择一组后，再运行 **32000次新种子母衰变** 做验证。所有展示的模拟ADC都来自原生C++输出。

这里的“联合”是分阶段校准与共同评价，不是一个包含所有装置参数的全局似然拟合。未把实测波形注入模拟，也未给PID分数直接加宽。
''')
    code('''from pathlib import Path
import json, sys
import numpy as np
from IPython.display import display, Markdown, Image
PROJECT=next(p for p in (Path.cwd(),*Path.cwd().parents) if (p/'simulation/main.cc').exists())
OUT=PROJECT/'simulation/results/joint_constraints_20261029'
def read(p):return json.loads(Path(p).read_text())
def table(headers,rows):
    return '\\n'.join(['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|']+['|'+'|'.join(map(str,r))+'|' for r in rows])
def nums(v):return ' / '.join(f'{x:.3f}' for x in v)
def ci(m,field,j,block='64'):
    lo,hi=m['uncertainty_by_data_block'][block][field][j]['estimate95']
    return f'[{lo:.3f}, {hi:.3f}]'
d=read(OUT/'design.json');s=read(OUT/'selection.json');r=read(OUT/'assessment.json');decision=read(OUT/'decision.json')
print('Python/NumPy:',sys.version.split()[0],np.__version__)
rows=[]
for stage in ['profile','calibration','validation']:
    p=read(OUT/stage/'protocol.json')
    for key,e in p['entries'].items():
        x=read(OUT/stage/'extraction'/f'{key}.json')
        v=read((PROJECT/e['path']).with_name(Path(e['path']).stem+'_validation.json'))
        rows.append([stage,key,e['events'],e['seed'],x['n_common'],x['common_truth_counts'],v['passed']])
display(Markdown(table(['阶段','场景','母事件/重放数','种子','共同Q事件','α/γ/电子/混合/其他祖先','物理检查'],rows)))
print('独立母事件总数:',r['calibration_independent_mothers']+r['new_independent_mothers'])''')
    md('''## 1. 有限记录窗与α慢分量的条件参数族

积分门为−40至3000 ns，原记录总长约4992 ns。长时间成分的τ、份额、产额可以在此窗口内相互补偿，因此本轮不再把15 μs边界最优值作为材料测量值。
4/6/8.44 μs是预设的敏感性选项，**不是文献确定范围、置信区间或已经测定的CsI衰减常数**。

每组拟合电子快/慢τ、α快τ和共同RC，两个快分量份额用前60%的平均时间约束求解。
叠加旧原生MC与解析卷积的固定平滑差，只用于提出参数；该差不进入原生数字化。
每组3个起点，保留全部结果，选择收敛且解析目标最小的起点。

共同光产额乘子与α相对归一化用于保持解析3 μs门积分；无逐事件幅度缩放。
SiPM PDE=0.3、被动信号传输=0.5、SPE2/29 ns、几何、前触发噪声和事件方差CV均沿用上一阶段的条件假设。
源距、求和电路等信息仍不完整。下表“有效α相对归一化”不能直接称为实验淬灭因子。
继承配置中的 `tail_refit`、`npe_profile` 记录仅表示历史；本轮参数与用途以实际配置、`joint_constraints`、`design.json` 和 `selection.json` 为准。
''')
    code('''rows=[]
for family,slow in d['families_alpha_slow_ns'].items():
    fit=read(OUT/'fit'/f'{family}.json');p=fit['best']['parameters']
    c=read(OUT/'profile'/f'{family}_run12_Am241/config.json')
    rows.append([family,f'{slow/1000:.4f}',nums(p['electron_tau_ns']),f"{p['alpha_tau_ns'][0]:.2f}",
        f"{p['lowpass_ns']:.2f}",f"{c['optical']['alpha_light_yield_ratio']:.4f}",f"{fit['best']['objective']:.4f}"])
display(Markdown(table(['组','α慢τ/μs','电子τ/ns','α快τ/ns','共同RC/ns','有效α相对归一化','解析目标'],rows)))
proposal=dict(np.load(OUT/'proposal_curves.npz'))
for source in ['Co60','Am241']:
    delta=proposal['a4000__'+source]-proposal['a8439__'+source]
    peak=proposal[source+'_data'].max()
    print(source,'4与8.44 μs提议曲线之间的NRMSE/实测峰:',f'{100*np.sqrt(np.mean(delta**2))/peak:.4f}%')
display(Image(filename=str(OUT/'figures/joint_family_profile.png')))''')
    md('''## 2. 在每组原生光学历史上重新拟合噪声

校准输运先关闭信号期经验残差，保留光子统计、SiPM和原前触发电子噪声。
用前60%的三种PID完整3×3协方差、F50/F102/F202三个起伏均值，拟合非负的功率谱混合：继承谱、白噪声、20/60/150/400/1000/2500 ns指数相关核。
合成1025点FIR核，再重估其非负强度；事件方差CV沿用原值。

协方差提议使用固定锚点线性化，F提议使用16组高斯创新的二次矩期望。**拟合预测本身不是验证结果**：随后由C++重新生成噪声、量化ADC、求自己的锚点与积分门，并重做原质量/Q选择。
实测协方差在每Q箱按中位数±3 IQR/1.349截尾，模拟协方差不截尾；IQR另行按原分位数计算。

本轮核查捕获一条 t₁₀ 恰好位于整数样点的事件：用 `anchor−0.5` 代替真实 t₁₀ 会使F统计窗口错开一个样点。
已在本轮拟合中改用实际分数t₁₀，核对全部六组原生ADC，并重算所有噪声拟合；正式PID整数积分门不变。
修正前的中间拟合和失败日志保留在 `diagnostics/anchor_proxy_before_fix`，以前冻结的研究目录保持不变。

带宽包括有限光子数、波形形状、相关残差和采集状态等影响。仅调整前触发RMS不足以解释完整协方差。
这里拟合的是有效的、采集状态相关的随机响应，尚未确定其唯一微观来源。
''')
    code('''rows=[]
for family in d['families_alpha_slow_ns']:
    for run,source in [(3,'Co60'),(12,'Am241')]:
        key=f'{family}_run{run}_{source}';f=read(OUT/'noise_fit'/f'{key}.json')
        active=[f'{n}:{v:.3g}' for n,v in zip(f['components'],f['component_variances']) if v>1e-9]
        rows.append([key,f['n_MC'],f"{f['variance_per_signal_adc']:.4f}",nums(f['predicted_SD_ratio']),nums(f['predicted_F_ratio']),'; '.join(active)])
display(Markdown(table(['组','校准共同Q数','信号方差强度','预测SD比','预测F比','非零谱分量/功率'],rows)))
audit=read(OUT/'diagnostics/anchor_boundary_audit.json')
print('实际t10回归检查事件数:',sum(x['events_checked'] for x in audit['arrays'].values()))
print('半采样近似影响的原事件ID:',{k:v['affected_original_event_ids'] for k,v in audit['arrays'].items() if v['affected_original_event_ids']})
print('原生噪声重放保留同一母事件和雪崩历史；计数不翻倍。')''')
    md('''## 3. 冻结的联合判据与新种子检验

各源等权，以下六组标准化平方残差各取组内均值后相加：

- 三个PID中位数偏差/实验IQR，尺度0.20；
- 平均时间均值差，尺度3 ns；
- 三个IQR比相对1的偏差，尺度0.15；
- 协方差上三角六项偏差/实验SD乘积，尺度0.25；
- 三个F比相对1的偏差，尺度0.10；
- 未缩放均ADC的NRMSE，尺度0.01。

这些是预先冻结的描述性权重，指标彼此相关，**不是独立误差或卡方检验**。
只按前60%校准重放的最小联合目标选组。下面同时展示三组训练表现和选中组的新种子前60%检查；有限MC可能影响排名，不能据此宣称α慢τ被唯一测定。
''')
    code('''rows=[]
for family,case in s['ranking'].items():
    for source,a in case['sources'].items():
        rows.append([family,source,'校准',f"{a['objective']:.3f}",f"{a['mean_time_difference_ns']:+.3f}",nums(a['IQR_ratio']),nums(a['F_ratio']),f"{100*a['mean_ADC_NRMSE']:.3f}%"])
for run,source in [(3,'Co60'),(12,'Am241')]:
    a=r['checks'][f'run{run}_first60']['models']['joint']['joint_criterion_check']
    rows.append([s['selected_family'],source,'新种子',f"{a['objective']:.3f}",f"{a['mean_time_difference_ns']:+.3f}",nums(a['IQR_ratio']),nums(a['F_ratio']),f"{100*a['mean_ADC_NRMSE']:.3f}%"])
display(Markdown(table(['组','源','用途','联合目标','平均时间差/ns','QDC/时间/PCA宽度比','F50/102/202比','均ADC NRMSE'],rows)))
print('在新种子运行前已冻结选组:',s['selected_family'])
print('各组两源平均目标:',{k:round(v['objective'],4) for k,v in s['ranking'].items()})''')
    md('''另做512次**仅针对MC统计量**的排名稳定性检查：固定原实验前60%目标和已经拟合的参数，各组分别对完整母事件重采样，再计算同一联合目标。
此检查不改变已冻结的选择，不读取新验证数据，也没有重新拟合噪声。表中“第一名频率”仅说明有限MC下的排名敏感性，**不是模型后验概率**，未包括实验、拟合或装置不确定性。''')
    code('''stability=read(OUT/'ranking_stability.json')
rows=[]
for family,a in stability['results'].items():
    rows.append([family,f"{a['point_objective']:.3f}",f"{100*a['rank1_frequency']:.1f}%",nums(a['bootstrap_objective_p025_p50_p975']),nums(a['objective_minus_selected_CI95'])])
display(Markdown(table(['组','原目标','重采样第一名频率','目标2.5/50/97.5分位','相对选中组目标差95%范围'],rows)))
print('有效重采样:',stability['valid_resamples'],'/ 512；已冻结选择改变:',stability['saved_selection_changed'])''')
    md('''## 4. 完整Geant4输出的均ADC波形

实际ADC逐事件扣基线、按t₁₀对齐，在原Q支持箱内取均值后箱间等权；模拟不重新缩放峰值。
NRMSE为−40至3000 ns均方根差除以实测均波形峰值。
保留较早光子统计候选和上一轮尾部候选作描述性参照，并列展示两种源与三个实验段，不能只选择吻合的波段。
图额外展示3–3.8 μs阴影区：这部分没有参与本轮拟合或排名，仅作选组后的外推诊断，不据此修改本轮参数。
额外尾电荷为分数t₁₀对齐后3000≤t<3800 ns样点和除以原正式Qlong，不是归一化概率；这里只报告点估计，不附加等效性结论。
''')
    code('''rows=[]
for group,c in r['checks'].items():
    for name,m in c['models'].items():
        extra=m['extended_tail_3000_3800_ns']
        rows.append([group,name,m['n_MC'],f"{100*m['mean_ADC_normalized_RMSE']:.3f}%",f"{m['mean_ADC_peak_ratio']:.4f}",
            f"{100*extra['ADC_NRMSE_over_full_peak']:.3f}%",f"{extra['difference']:+.5f}"])
display(Markdown(table(['实验段','模型','共同Q模拟数','门内均ADC NRMSE','均ADC峰比','额外尾NRMSE/总峰','额外尾电荷/Qlong差'],rows)))
display(Image(filename=str(OUT/'figures/joint_native_waveforms.png')))''')
    md('''## 5. PID中心与宽度必须同时检查

中心均值按正式整数锚点−40至3000 ns积分门计算；时间片的一阶矩共用每个事件的Qlong分母，片间求和严格等于正式平均时间。
宽度先在每个固定Q箱计算IQR，再对支持箱取RMS。run3使用6箱、run12使用前3箱。

区间使用512次整母事件重采样和实验原始序号块64/128重采样；旧MC和数据重采样保持上一轮冻结结果，新验证MC独立采样。
拟合、选组、装置和闭合修正均固定，未计入这些不确定性，未作多重比较校正。
前60%是训练数据检查，中/后段已经在此前研究中看过，**不是新实验盲验证**。
''')
    code('''rows=[]
for group,c in r['checks'].items():
    m=c['models']['joint']
    rows.append([group,f"{m['tail_difference'][0]:+.3f}",ci(m,'tail_difference_CI95',0),nums(m['IQR_ratio']),
        ci(m,'IQR_ratio_CI95',1),ci(m,'IQR_ratio_CI95',1,'128')])
display(Markdown(table(['实验段','平均时间差/ns','95%区间','QDC/时间/PCA宽度比','时间宽度比CI/块64','CI/块128'],rows)))
display(Image(filename=str(OUT/'figures/joint_native_pid.png')))
for c in r['checks'].values():
    for m in c['models'].values():assert abs(sum(m['mean_time_slice_difference_ns'])-m['tail_difference'][0])<1e-8
print('全部18组时间片贡献和通过正式平均时间核查。')''')
    md('''## 6. 短尺度起伏和完整协方差

即使某一种PID宽度接近，也要检查另外两种及其相关性。下图同时展示F与协方差残差，保留未吻合项。
F是波形短尺度统计量，不是α接受率或γ泄漏率。本轮没有把两种源放到完全相同采集响应下重算ROC；不更新实验PID效率。
²⁴¹Am仍包含α、γ及衰变电子，实测主带诊断不是逐事件粒子真值。
''')
    code('''display(Image(filename=str(OUT/'figures/joint_noise_controls.png')))
rows=[]
for run,source in [(3,'Co60'),(12,'Am241')]:
    m=r['checks'][f'run{run}_last20']['models']['joint']
    for j,scale in enumerate([50,102,202]):rows.append([source,scale,f"{m['F_ratio'][j]:.4f}",ci(m,'F_ratio_CI95',j)])
display(Markdown(table(['源/后段','尺度/ns','F比','条件95%区间'],rows)))''')
    md('''逐Q箱再检查PID中位数和IQR，避免汇总比值掩盖电荷依赖。下图的竖线与阴影是分布四分位范围，不是均值或宽度的置信区间。
只画冻结支持箱，未对稀疏箱插值。''')
    code("display(Image(filename=str(OUT/'figures/joint_pid_by_charge.png')))")
    md('''## 7. 本轮结论与证据边界

选择、噪声配置和验证种子都在查看新验证结果前冻结，本轮不因后段表现再调整这份参数。
区间覆盖0或1不等于通过“完全一致”的等效性检验；有限MC、条件假设和参数退化仍需报告。
下表给出新的后段点估计及固定模型区间覆盖情况。默认配置没有自动替换。

若校准与新种子之间差异明显，应先量化有限MC影响，避免把随机宽度起伏当成材料τ信息。
若稳定残差集中在特定Q箱或中心/协方差方向，应检查响应模型的表达能力。
此前run12的噪声状态漂移继续作为单独对照保留；run18及同采集状态下两源PID转移尚未在本轮验证。

还要区分模拟偏差与实验段之间的变化：同一个固定模型对前/后段的中心残差之差，必然等于两段实测平均时间之差的相反数。
下面在相同固定Q箱、相同主带诊断下计算实测中/后段相对前60%的平均时间差，区间仅来自原实验块重采样。
这能提示固定响应跨段比较的限制，但不能单独把变化归因于温度、电路或某一种噪声；箱内电荷分布、选择和未记录状态也需要考虑。
''')
    code('''rows=[]
for run,source in [(3,'Co60'),(12,'Am241')]:
    m=r['checks'][f'run{run}_last20']['models']['joint'];c=decision['last20_conditional_interval_checks'][source]
    rows.append([source,f"{m['tail_difference'][0]:+.3f}",f"{100*m['mean_ADC_normalized_RMSE']:.3f}%",nums(m['IQR_ratio']),
        c['mean_time_CI_contains_zero'],c['PID_width_CI_contains_one'],c['F_CI_contains_one']])
display(Markdown(table(['源','平均时间差/ns','均ADC NRMSE','三个宽度比','中心CI覆盖0','宽度CI覆盖1','FCI覆盖1'],rows)))
boot=dict(np.load(OUT/'bootstrap.npz'));changes=[]
for run,source in [(3,'Co60'),(12,'Am241')]:
    first=r['checks'][f'run{run}_first60']['data_tail_metrics'][0]
    for segment in ['middle20','last20']:
        delta=r['checks'][f'run{run}_{segment}']['data_tail_metrics'][0]-first
        cis=[]
        for block in [64,128]:
            v=boot[f'run{run}_{segment}__block{block}__moments'][:,0]-boot[f'run{run}_first60__block{block}__moments'][:,0]
            cis.append(nums(np.quantile(v[np.isfinite(v)],[.025,.975])))
        changes.append([source,segment,f'{delta:+.3f}',*cis])
display(Markdown(table(['实测源','实验段相对前60%','平均时间差/ns','块64区间/ns','块128区间/ns'],changes)))
print('α慢τ/ns:',decision['alpha_slow_ns'],'有效α相对归一化:',decision['alpha_relative_yield'])
print('默认配置已替换:',decision['promoted_to_default'])
print('复现：prepare_joint_constraints → run_joint_constraints profile → fit_joint_constraints → calibration replay → select_joint_constraints → fresh validation → analyze_joint_constraints → plot_joint_constraints')
print('核验/归档：simulation/verify_joint_constraints.py --archive')
print('表格：',OUT/'validation_table.csv')''')
    nb = nbf.v4.new_notebook(cells=cells, metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.13.5'}})
    shell = InteractiveShell.instance()
    for cell in nb.cells:
        if cell.cell_type != 'code': continue
        with capture_output() as capture: result = shell.run_cell(cell.source, store_history=True)
        if result.error_before_exec or result.error_in_exec: raise RuntimeError(result.error_before_exec or result.error_in_exec)
        cell.execution_count = shell.execution_count-1; outputs = []
        if capture.stdout: outputs.append(nbf.v4.new_output('stream', name='stdout', text=capture.stdout))
        if capture.stderr: outputs.append(nbf.v4.new_output('stream', name='stderr', text=capture.stderr))
        for output in capture.outputs: outputs.append(nbf.v4.new_output('display_data', data=output.data, metadata=output.metadata))
        cell.outputs = outputs
    nbf.validate(nb); nbf.write(nb, path); print(path)


if __name__ == '__main__': main()
