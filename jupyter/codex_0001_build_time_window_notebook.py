"""Build and execute the exact temporal fluctuation diagnosis notebook."""
import os
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
cache = PROJECT / '.cache/codex_0001_time_window_ipython'
cache.mkdir(parents=True, exist_ok=True)
os.environ['IPYTHONDIR'] = str(cache)
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import nbformat as nbf
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output


def main():
    out = PROJECT / 'simulation/results/time_windows_20261102'
    assert (out / 'decision.json').exists() and not (out / 'record.json').exists()
    cells = []

    def md(text):
        cells.append(nbf.v4.new_markdown_cell(text))

    def code(text):
        cells.append(nbf.v4.new_code_cell(text))

    md('''# CsI–SiPM：残余波动差异的时间分段诊断

2026-11-02。在英文提交 `6f49b781b2f636ee8b6bde81792677b6b2a2819a`（`Evaluate low-frequency noise tradeoffs in CsI SiPM readout`）push并核对远端后继续。[上一轮低频衰减](codex_0001_csi_low_frequency_noise.ipynb)虽然能收窄Am241的F202，却同时破坏PID宽度，两个校准折均保留原参数。本轮从已有实测BIN和原生读出ROOT重新提取，定位差异在脉冲中的时间分布。

**主要发现：** Am241的F202方差差额中，最大的带符号分段贡献落在t10后500–1000 ns；随机分母也有正的统计贡献。用固定分母作诊断时，MC/实测SD比从1.162/1.176变为1.078/1.102，仍有残差。Co60整体F202宽度接近，但2000–2500 ns局部F202均值比只有约0.84–0.85。下一轮需要同时约束早期波动和尾部时间结构。

这是一项统计定位，**尚未修正模拟参数或证明具体物理噪声来源**。当前ADC、正式PID和默认配置不变；没有新实验、新独立Geant4母事件或新增原生读出。模拟波形仍来自已有C++/Geant4链路，不由Python生成或修改。Am241包含α与γ/电子，实测事件没有逐事件粒子真值。
''')
    code('''from pathlib import Path
import json
import numpy as np
from IPython.display import display, Markdown, Image
PROJECT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p/'simulation/main.cc').exists())
OUT = PROJECT/'simulation/results/time_windows_20261102'
def read(path): return json.loads(Path(path).read_text())
def fmt(x): return '不可估计' if x is None else f'{x:.4g}'
def ci(x): return '不可估计' if x['CI95'] is None else '['+', '.join(fmt(y) for y in x['CI95'])+f"] ({x['valid_draws']}/512)"
def table(head, rows): return '\\n'.join(['|'+'|'.join(head)+'|', '|'+'|'.join(['---']*len(head))+'|']+['|'+'|'.join(map(str, r))+'|' for r in rows])
p = read(OUT/'protocol.json'); a = read(OUT/'analysis.json'); d = read(OUT/'decision.json')
term_names = [name.removeprefix('excess_') for name in p['metrics'][11:18]]
rows = []
for source in a:
    r = read(OUT/'evaluation'/f'{source}_extraction.json')
    for variant, record in r.items():
        rows.append([source, variant, record['active'], record['nonpositive_local_mu']])
display(Markdown(table(['源', '数据/读出', '原选择内事件记录数', '局部均值非正的记录数'], rows)))
display(Image(filename=str(OUT/'figures/time_window_pulse_means.png')))''')
    md(r'''## 1. 固定窗口、事件与电荷权重

Co60使用run 3，Am241使用run 12；沿用53.2 V数据的last20段、B128基线、原质量条件及完整母事件池下的当前Q选择。Co支持6个Q箱，Am支持3个Q箱；每箱的上下半箱目标质量、折分配和均值参考都保持原样。不因某个局部量不理想而增加切选。表中的MC两条读出共享母事件，不能把事件记录数相加后称作独立样本量。

采样间隔2 ns。原前触发区间从$lceil t_{10}-300ceil$开始，共200个样点；原信号区间从$lceil t_{10}+250ceil$开始，共1000个样点。后者固定切成4个相邻250样点窗口，名义范围为t10后500–1000、1000–1500、1500–2000、2000–2500 ns。窗口边界包含向上取整造成的小于2 ns离散偏移。

对原ADC减基线后，沿用25/51/101样点中心滑动均值残差$r=y-mathrm{MA}(y)$，对应F50/F102/F202。各段局部诊断定义为

\[
F_k=\frac{V_k-V_{\rm pre}}{\mu_k},
\]

其中$V_k$和$V_{\rm pre}$是残差样本方差（ddof=1），$\mu_k$是该段原脉冲的平均ADC。局部F不能直接相加得到整体F。所有选中事件的局部$\mu_k$均为正；本轮没有删除或截断事件。下图和表展示F202，全部三尺度的48行局部结果保存于`local_window_table.csv`，其余指标和区间见`analysis.json`。
''')
    code('''rows = []
for source, group in a.items():
    for variant, scales in group['variants'].items():
        r = scales['202']
        for w in range(4):
            rows.append([source, variant, f'{500+500*w}–{1000+500*w}', fmt(r['point'][f'local_mean_ratio_w{w}']), ci(r['CI95'][f'local_mean_ratio_w{w}']), fmt(r['point'][f'local_SD_ratio_w{w}']), ci(r['CI95'][f'local_SD_ratio_w{w}'])])
display(Markdown(table(['源', '读出', '时间/ns', '局部F202均值比', '均值比条件95%范围', '局部F202 SD比', 'SD比条件95%范围'], rows)))
display(Image(filename=str(OUT/'figures/time_window_local_F202.png')))''')
    md(r'''## 2. 精确还原整体F的七项分解

记四段残差均值为$\bar r_k$，整体残差均值为$\bar r$。样本方差的恒等式为

\[
V_{\rm post}=\sum_{k=1}^{4}\frac{249}{999}V_k
 +\frac{250}{999}\sum_{k=1}^{4}(\bar r_k-\bar r)^2.
\]

使用既有对折、每Q箱的固定脉冲均值$\mu_0$，依次定义四个窗口项、段间均值项、负前触发项，共六个$T_j$；它们均以上式相应项除以$\mu_0$。第七项为

\[
T_7=(V_{\rm post}-V_{\rm pre})\left(\frac1\mu-\frac1{\mu_0}\right),\qquad
\sum_{j=1}^{7}T_j=\frac{V_{\rm post}-V_{\rm pre}}{\mu}=F.
\]

事件内方差使用ddof=1；事件之间的协方差沿用原加权总体矩定义。在每个当前Q箱内保持既有半箱质量，先计算七项协方差$C$，然后对支持Q箱等权平均。因此$\sum_{ij}C_{ij}$必须准确还原原整体F的条件方差。

每项的带符号贡献取$\mathrm{Cov}(T_j,F)=\sum_i C_{ji}$。图表给出MC与实测之差，再除以实测整体F方差。七项之和等于整体方差相对差额。**这些量包括交叉协方差，可以为负，不能当作七种独立噪声功率或因果份额。** 最大贡献是分解后的描述，不表示事先指定了该窗口为唯一检验对象；全部窗口和尺度均保留。
''')
    code('''rows = []
for source, group in a.items():
    for variant, scales in group['variants'].items():
        r = scales['202']; point = r['point']
        rows.append([source, variant, fmt(point['global_variance_excess']), *[fmt(point['excess_'+term]) for term in term_names]])
display(Markdown(table(['源', '读出', '整体方差差额', '500–1000', '1000–1500', '1500–2000', '2000–2500', '段间均值', '负前触发', '分母项'], rows)))
display(Image(filename=str(OUT/'figures/time_window_excess_terms.png')))
rows = []
for variant in ['split_r0', 'split_r1']:
    r = a['Am241']['variants'][variant]['202']
    for name in ['excess_window_500_1000', 'excess_normalization']:
        rows.append([variant, name, fmt(r['point'][name]), ci(r['CI95'][name])])
display(Markdown(table(['Am读出', '带符号项', '差额/实测整体方差', '条件95%范围'], rows)))''')
    md(r'''## 3. 段间协方差与局部宽度需要一起看

另将整体方差拆为三组：四个窗口项的对角方差之和、这些窗口之间全部非对角协方差之和、其余所有项。对称矩阵的非对角元素计入两次，三组严格相加为整体方差。该分组与前面的七项行和分解是同一矩阵的不同汇总，不能再次相加。

Am241的第一个窗口对角项偏大，段间交叉协方差也有正差额。Co60的四个窗口对角方差差额之和略负，但段间交叉项为正，说明整体宽度接近可能同时包含方向不同的局部差异。图中数值是固定分母窗口项的协方差差额，不是原ADC样点间的协方差，也不是局部$F_k$协方差。
''')
    code('''rows = []
for source, group in a.items():
    for variant, scales in group['variants'].items():
        r = scales['202']
        for name in ['quarter_diagonal_excess', 'quarter_cross_covariance_excess', 'other_covariance_excess']:
            rows.append([source, variant, name, fmt(r['point'][name]), ci(r['CI95'][name])])
display(Markdown(table(['源', '读出', '分组', '差额/实测整体方差', '条件95%范围'], rows)))
display(Image(filename=str(OUT/'figures/time_window_cross_covariance.png')))''')
    md(r'''## 4. 固定分母对照、分折和不确定度

固定分母诊断使用前六项之和，即$(V_{\rm post}-V_{\rm pre})/\mu_0$，同时作用于MC与实测。其宽度比更接近1，只表示随机分母与分子耦合影响了原F的统计宽度，不能据此改写原ADC或宣布PID已改善。七项中的分母行和含一份与其他项的交叉协方差；删除第七项时总方差还会去掉另一份交叉项，所以不能把该行和直接当作固定分母造成的总变化。

直接沿用前轮512组实测事件块和MC母事件的配对计数，不重新抽样窗口，不重拟合，不把相同母事件的两个读出当作独立实验。各尺度整体SD比的点估计及全部512次结果均须与前轮闭合。区间固定拟合、Q参考、选择、已实现噪声，未含模型选择、重拟合或系统误差，也未作多重比较校正。两折数值只作敏感性检查；当前样本已被反复探索，不是新盲验证。
''')
    code('''rows = []
for source, group in a.items():
    for variant, scales in group['variants'].items():
        for scale, r in scales.items():
            rows.append([source, variant, scale, fmt(r['point']['global_SD_ratio']), ci(r['CI95']['global_SD_ratio']), fmt(r['point']['fixed_denominator_SD_ratio']), ci(r['CI95']['fixed_denominator_SD_ratio'])])
display(Markdown(table(['源', '读出', '尺度/ns', '原F SD比', '原F条件95%范围', '固定分母SD比', '固定分母条件95%范围'], rows)))
display(Image(filename=str(OUT/'figures/time_window_denominator.png')))
rows = []
for source, group in a.items():
    for fold, variants in group['fold_checks'].items():
        for variant, values in variants.items():
            r = dict(zip(p['metrics'], values[2]))
            rows.append([source, fold, variant, fmt(r['global_SD_ratio']), fmt(r['fixed_denominator_SD_ratio']), fmt(r['excess_window_500_1000']), fmt(r['excess_normalization'])])
display(Markdown(table(['源', '评价折', '读出', 'F202 SD比', '固定分母SD比', '首窗口贡献', '分母贡献'], rows)))
display(Markdown('**下一步：**'+d['next_work_zh']))
for limit in d['limitations_zh']:
    display(Markdown('边界：'+limit))''')
    md('''## 5. 复现和核验范围

实际流程为 `prepare_time_windows.py` → `extract_time_windows.py` → `analyze_time_windows.py` → `report_time_windows.py` → 本笔记生成器 → `verify_time_windows.py`及`--archive`。输入为已有实测BIN、split模型完整池原生ROOT、既有Q参考和512组配对计数；源码、输入哈希和窗口在提取前冻结。完整重跑应使用新结果目录。

`test_time_windows.py`使用独立直接卷积验证合成波形的方差恒等式，并用显式重复事件检查加权协方差。`verify_time_windows.py`再从原始BIN/ROOT抽取真实ADC，以直接卷积重算各项，独立验证点估计和第0/127/511组重采样的中心化加权矩；全部事件的七项和及全部512组整体SD比必须复现旧结果。所有选中事件的局部均值有效性、表格行数、执行单元、嵌入图片及来源哈希均有检查。

`verification.json`记录实际检查数；`record.json`封存结果和源码快照。五幅图均作可视检查。未新增原生读出，未修改默认配置、正式PID或共享文献库。前一轮低频检查已经commit和push；本轮时间分段分析为随后新增的本地工作。
''')
    nb = nbf.v4.new_notebook(cells=cells, metadata={
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python'},
    })
    shell = InteractiveShell.instance()
    for cell in nb.cells:
        if cell.cell_type != 'code':
            continue
        with capture_output() as captured:
            result = shell.run_cell(cell.source, store_history=True)
        if result.error_before_exec or result.error_in_exec:
            raise RuntimeError(result.error_before_exec or result.error_in_exec)
        cell.execution_count = shell.execution_count - 1
        outputs = []
        if captured.stdout:
            outputs.append(nbf.v4.new_output('stream', name='stdout', text=captured.stdout))
        if captured.stderr:
            outputs.append(nbf.v4.new_output('stream', name='stderr', text=captured.stderr))
        for output in captured.outputs:
            outputs.append(nbf.v4.new_output('display_data', data=output.data, metadata=output.metadata))
        cell.outputs = outputs
    nbf.validate(nb)
    path = PROJECT / 'jupyter/codex_0001_csi_time_resolved_fluctuations.ipynb'
    nbf.write(nb, path)
    print(path)


if __name__ == '__main__':
    main()
