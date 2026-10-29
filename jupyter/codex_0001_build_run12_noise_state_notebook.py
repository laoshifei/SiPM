"""Build and execute the run12 pretrigger-only noise-state experiment notebook."""
import os
from pathlib import Path
PROJECT=Path(__file__).resolve().parents[1]
cache=PROJECT/'.cache/codex_0001_run12_state_ipython';cache.mkdir(parents=True,exist_ok=True)
os.environ['IPYTHONDIR']=str(cache)
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import nbformat as nbf
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output


def main():
    path=PROJECT/'jupyter/codex_0001_csi_run12_noise_state.ipynb'
    assert not path.exists(),'Preserve the executed notebook; use a new output for a new experiment'
    cells=[]
    def md(s):cells.append(nbf.v4.new_markdown_cell(s))
    def code(s):cells.append(nbf.v4.new_code_cell(s))
    md('''# CsI(Tl)–SiPM：run 12 噪声状态能否解释 PID 带宽差异？

2026-10-29，接续英文提交 `c13601b`（已推送）、[下一步检查笔记](codex_0001_csi_next_steps_review.ipynb)与[光电子统计笔记](codex_0001_csi_pid_photon_statistics.ipynb)。

本轮完成 **18 组原生 C++ 响应重放**，只改变电子噪声强度、基线/方差状态及相关谱。源仍为原 Geant4 ²⁴¹Am 完整衰变历史，光子输运、雪崩、无噪声信号、发光/读出参数和信号期残差保持固定。**没有复制实验波形或直接给 PID 分数加宽。**

结果：后段基线 RMS 漂移能够得到近似描述，QDC 与 PCA 的宽度点估计向实测靠近；但三个新对照、三种方法的后段配对变化95%区间均包含0，**统计上还不能确认带宽改善**。平均时间宽度比的点估计仍约0.82，尚未完成全部PID分布的一致性验证。下面同时列出所有预先冻结的对照与统计区间。
''')
    code('''from pathlib import Path
import json, sys, hashlib
import numpy as np
from IPython.display import display, Markdown, Image
PROJECT=next(p for p in (Path.cwd(),*Path.cwd().parents) if (p/'simulation/main.cc').exists())
OUT=PROJECT/'simulation/results/run12_noise_state_20261029'
def read(p):return json.loads(Path(p).read_text())
def table(headers,rows):
    return '\\n'.join(['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|']+['|'+'|'.join(map(str,r))+'|' for r in rows])
def numbers(values,precision=3):return ' / '.join(f'{x:.{precision}f}' for x in values)
def ci_text(row,method,block='64',field='IQR_ratio_CI95'):
    lo,hi=row['uncertainty_by_data_block'][block][field][method]['estimate95']
    return f'[{lo:.3f}, {hi:.3f}]'
r=read(OUT/'assessment.json');p=read(OUT/'protocol.json')
variants=p['noise_variants'];segments=['first60','middle20','last20']
names={'baseline':'原噪声','scale':'只改RMS','states':'基线/方差状态','spectrum_states':'状态+相关谱'}
print('Python / NumPy:',sys.version.split()[0],np.__version__)
print('唯一母事件：',r['unique_parent_mothers'],'；新增独立母事件：',r['new_native_mothers'])
print('原共同Q内MC：',r['old_common_MC'],'；实际原生重放数：',len(p['entries'])-1)
assert r['prior_baseline_reproduced'] and r['no_PID_based_candidate_refit']''')
    md('''## 1. 预先冻结的比较与交叉估计

原始 run 12 有7516条记录。沿用已冻结的波形队列和前60%/中20%/后20%分段。
各段按原始事件号 `floor(id/64) % 2` 分两组：只用一组的前触发样点估计状态，在另一组检查，然后交换。
状态估计用前128点；相关谱用前256点（512 ns，线性去趋势）。这一步**不用主脉冲、PID分数或主带门切**。
基线均值/对数方差混合模型在一/两状态间由校准组 BIC 选择，且每个成分至少20条有效观测。

四组响应为：原模型；保留原谱/原状态分布只改 RMS；保留原谱重估基线/方差状态；重估状态和相关谱。
所有配置在重放前写入 `protocol.json` 并保存 SHA-256。没有根据留出 PID 结果再调参或隐藏不理想的候选。

每个模拟母事件用固定随机数分配一个评估组，与该段实验组比例相匹配；每段每个母事件只保留一份响应。
18组重放共用24000个原母事件，不计成432000个独立样本。共同Q区间与支持箱0/1/2保持原样。

**解释边界：**只有这次新增噪声估计进行了交叉估计。旧的脉冲/信号参数已经用过前60%；后段前触发样点也参与相应条件估计。
因此这是以采集状态为条件的内部验证，不是从前段预测后段的盲检。相邻块仍可能相关；本轮增加块长128的区间敏感性，但不把ID不重合称为完全独立。
''')
    code('''rows=[]
for seg in segments:
    for fold in [0,1]:
        f=read(PROJECT/p['groups'][seg]['folds'][str(fold)]['fit'])
        a=f['state_fit_original_spectrum']['chosen'];b=f['state_fit_new_spectrum']['chosen']
        rows.append([seg,fold,f['calibration_n'],f['evaluation_n'],f"{f['RMS_scale']:.3f}",len(a['states']),len(b['states'])])
display(Markdown(table(['段','评估组','校准数','留出数','RMS尺度','原谱状态数','新谱状态数'],rows)))
print('共同Q区间:',p['q_edges'][0],p['q_edges'][-1],'；等权支持箱:',p['support_bins'])''')
    md('''## 2. 前触发噪声：先检查噪声本身

下表和图对实验使用整段冻结波形队列，不加 PID 主带门切；模拟使用原来的206个共同Q母事件，每个只分配一个状态响应。
图中竖线是事件分布的10%–90%分位，**不是中位数的置信区间**。分组留出结果另存 `assessment.json → noise → folds`。
图中的原谱对照即使调大RMS，仍未重现后段低频增强和相关结构；重估频谱的对照更接近这些特征，这是描述性比较。
RMS 可接近并不等于完整噪声过程已重现：有限512 ns窗口、去趋势、短 FIR 和高斯混合假设，不能唯一识别微秒尺度相关、瞬态或基线与主脉冲的耦合。
''')
    code('''rows=[]
for seg in segments:
    n=r['noise'][seg]
    rows.append([seg,f"{n['data']['RMS_p10_p50_p90'][1]:.3f}",*[f"{n[v]['pooled']['RMS_p10_p50_p90'][1]:.3f}" for v in variants]])
display(Markdown(table(['段','实验RMS中位数']+[names[v] for v in variants],rows)))
display(Image(filename=str(OUT/'figures/noise_state_closure.png')))''')
    md('''## 3. 三种 PID 带宽及成对区间

每个Q箱独立求 IQR，再对原支持箱等权取平方均值的平方根；表中的值是该宽度的 MC/实验比。
实验只在评估时沿用 `Qshort/Qlong < 0.4` 主带诊断，不代表已知纯α样本。²⁴¹Am 同时有γ与衰变电子，源名不作事件真标签。

512次重采样中，同一个母事件计数向量同时用于所有模拟候选和分段；实验按原始序号块重采样。
这样新旧差值保留配对，重放副本不增加有效样本数。区间固定噪声拟合、频谱、母事件分组和旧模型参数，**未包含完整拟合/模型不确定性**，也未作多重比较调整。
原模型点估计逐值复现上一版；本轮为所有候选建立统一的配对重采样，使用新固定种子，512次区间端点会有Monte Carlo变化，未改写旧记录。
本轮后段绝对宽度比区间包含1、配对变化区间包含0，既不能确认改善，也不能据此证明模型等效。
''')
    code('''rows=[]
for seg in segments:
    for v in variants:
        a=r['checks'][seg]['variants'][v]['reselected']
        rows.append([seg,names[v],a['n_MC'],numbers(a['IQR_ratio']),ci_text(a,1)])
display(Markdown(table(['段','对照','MC数','QDC / 平均时间 / PCA带宽比','平均时间95%区间'],rows)))
display(Image(filename=str(OUT/'figures/noise_pid_widths.png')))
rows=[]
for v in variants:
    a=r['checks']['last20']['variants'][v]['reselected']
    rows.append([names[v],f"{a['IQR_ratio'][1]:.3f}",ci_text(a,1,'64'),ci_text(a,1,'128'),f"{a['paired_delta_ratio'][1]:+.3f}",ci_text(a,1,field='paired_delta_ratio_CI95')])
display(Markdown(table(['后段对照','平均时间比','块64区间','块128区间','相对原模型变化','配对变化95%区间'],rows)))''')
    md('''## 4. 固定事件与固定锚点：检查筛选迁移

主结果重新执行原正式质量与Q筛选。同时保留原206个母事件及原Q分箱，检查相同事件的宽度；再把积分锚点固定到原 ADC 的锚点。
后一个检查依然使用候选波形自己的前128点基线，不是重用原基线值。固定锚点只用于诊断，未改正式算法。
下面画的是相对原噪声的宽度比变化及成对95%区间，0表示没有变化。完整三段和三种定义都在CSV/JSON中。
''')
    code('''rows=[]
for seg in segments:
    for v in variants:
        a=r['pairing'][f'{seg}__{v}']
        rows.append([seg,names[v],a['selected_n'],len(a['gained']),len(a['lost']),'/'.join(map(str,a['assigned_fixed_mothers_per_fold']))])
display(Markdown(table(['段','对照','重筛数','新增数','流失数','固定母事件两组数'],rows)))
display(Image(filename=str(OUT/'figures/noise_pid_paired_controls.png')))''')
    md('''## 5. 波形均值、带中心和短尺度起伏是否保持

平均 ADC 先扣本事件前128点基线并按 t₁₀ 对齐，然后在原Q支持箱内等权平均；没有按实验幅度缩放模拟波形。
NRMSE为−40至3000 ns的均方根差除以实测均波形峰值。短尺度 F 沿用此前50/102/202 ns定义。
这些是点估计，与上节带宽统计区间用途不同。
''')
    code('''rows=[]
for seg in segments:
    for v in variants:
        a=r['checks'][seg]['variants'][v]['reselected']
        rows.append([seg,names[v],f"{100*a['mean_ADC_normalized_RMSE']:.3f}%",f"{a['mean_ADC_peak_ratio']:.4f}",numbers(a['F_ratio']),numbers(a['median_shift_in_data_IQR'])])
display(Markdown(table(['段','对照','均波形NRMSE/峰','均波形峰比','三个F比','三个中心差/实验IQR'],rows)))
display(Image(filename=str(OUT/'figures/noise_waveform_controls.png')))''')
    md('''## 6. 这一轮能得出什么，接下来怎么做

- 基线噪声状态漂移是应纳入的采集条件。后段QDC/PCA宽度点估计有所接近，但配对区间尚不能确认改善；固定事件后平均时间宽度比仍约0.81–0.83，完整解释尚缺证据。
- 本轮不选一个“最好看”的后段候选替换默认配置，也不声称已测得唯一的电子噪声机制。尤其要保留原谱/新谱两种状态对照及其残差。
- 下一项按既定顺序处理 **Co 平均时间中心与2–3 µs尾部**：在前60%训练中联合两源，加入平均时间/分段尾积分约束，检查旧700 ns慢分量边界与拟合权重；同时保护平均ADC、F和带宽结果。
- run12剩余带宽需要区分信号期残差、慢相关项、物理事件差异和参数族的不确定性。不能继续只增大基线RMS，也不能用留出PID结果反复选参数后仍称独立验证。
- 最终参数冻结后再补新种子母衰变，改善统计精度。当前18次重放没有增加独立α/γ样本；本轮也没有为变更噪声状态补齐共同状态的Co对照，不能据此更新α接受率/γ泄漏率。

这轮是有价值的约束：把“带宽来自噪声吧”拆成了可检验的噪声来源。平均波形接近、基线RMS接近、PID分布全部接近是三个不同的要求。
''')
    code('''decision=read(OUT/'decision.json')
display(Markdown(decision['conclusion_zh']))
print('结果目录：',OUT.relative_to(PROJECT))
print('108行带宽数值：validation_table.csv；完整区间、筛选与噪声结果：assessment.json')
print('复现顺序：prepare_run12_noise_state.py → run_run12_noise_state.py → analyze_run12_noise_state.py → plot_run12_noise_state.py')
print('归档核查：simulation/verify_run12_noise_state.py；已完成目录拒绝覆盖，重算请另用新目录。')''')
    nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
        'language_info':{'name':'python','version':'3.13.5'}})
    shell=InteractiveShell.instance()
    for cell in nb.cells:
        if cell.cell_type!='code':continue
        with capture_output() as capture:result=shell.run_cell(cell.source,store_history=True)
        if result.error_before_exec or result.error_in_exec:raise RuntimeError(result.error_before_exec or result.error_in_exec)
        cell.execution_count=shell.execution_count-1;outputs=[]
        if capture.stdout:outputs.append(nbf.v4.new_output('stream',name='stdout',text=capture.stdout))
        if capture.stderr:outputs.append(nbf.v4.new_output('stream',name='stderr',text=capture.stderr))
        for output in capture.outputs:outputs.append(nbf.v4.new_output('display_data',data=output.data,metadata=output.metadata))
        cell.outputs=outputs
    nbf.validate(nb);nbf.write(nb,path);print(path)


if __name__=='__main__':main()
