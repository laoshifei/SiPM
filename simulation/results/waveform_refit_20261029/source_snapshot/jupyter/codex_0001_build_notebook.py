"""Build the user-requested executed notebook from frozen numerical outputs."""
import os
from pathlib import Path
import json

PROJECT = Path(__file__).resolve().parents[1]
CACHE = PROJECT / '.cache/codex_0001_jupyter'
for key, sub in [('JUPYTER_CONFIG_DIR', 'config'), ('JUPYTER_DATA_DIR', 'data'),
                 ('JUPYTER_RUNTIME_DIR', 'runtime'), ('IPYTHONDIR', 'ipython'),
                 ('MPLCONFIGDIR', 'matplotlib')]:
    path = CACHE / sub
    path.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(path)
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
kernel = CACHE / 'data/kernels/codex-sipm'
kernel.mkdir(parents=True, exist_ok=True)
(kernel / 'kernel.json').write_text(json.dumps({
    'argv': ['/home/xc/software/jupyter/.venv/bin/python', '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
    'display_name': 'Python (SiPM notebook)', 'language': 'python',
}, indent=2) + '\n')

import nbformat as nbf
from nbclient import NotebookClient


def build():
    cells = []
    def md(text): cells.append(nbf.v4.new_markdown_cell(text))
    def code(text): cells.append(nbf.v4.new_code_cell(text))
    md('''# CsI(Tl)–SiPM：Geant4 波形参数再校准

记录日期：2026-10-29。目标是在真实 Geant4 核衰变、光学输运和原生 SiPM 数字化基础上，
使模拟的平均形状、峰值时间、积分比、基线与起伏统计接近实测。随机单条波形的细峰不要求逐点相同。

装置已知为 10×10×10 mm³ CsI(Tl)、PTFE 侧面包裹、前面 5 µm Mylar、S13361-3050 阵列多个单元求和、无前放。
精确求和电路、源距、历史输入范围和温度仍不全。²⁴¹Am 包含 α 和 γ，源名不能代替实测逐事件粒子标签。

本笔记保留拟合方法、参数、实际执行输出、图与未通过的项目。旧阶段结果不覆盖。''')
    code('''from pathlib import Path
import json, sys
import numpy as np
from IPython.display import display, Markdown, Image
PROJECT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / 'simulation/main.cc').exists())
OLD = PROJECT / 'simulation/results/signal_fluctuations_20261028'
NEW = PROJECT / 'simulation/results/waveform_refit_20261029'
def read(path):
    return json.loads(Path(path).read_text())
print('Project:', PROJECT)
print('Notebook Python:', sys.version.split()[0])
print('NumPy:', np.__version__)
print('Numerical G4/PyROOT scripts use /usr/bin/python3; this notebook uses the existing Jupyter environment.')''')
    md('''## 1. 上一版的起点与剩余差距

上一版按 run 训练信号相关随机残差；RC 为 105 ns。训练仅用 run 3 / 12 前 60% 原始序号区间内
的 2450 / 1547 条冻结合格波形。之后用新随机种子完成 4000 / 24000 个 Co / Am 衰变。
下表按旧结果重新读取，不能把归一化平均形状误差小直接等同于全部响应一致。''')
    code('''old = read(OLD / 'independent/validation.json')
lines = ['|源 / 检查组|平均形状 NRMSE|F50 偏差|F102 偏差|F202 偏差|', '|---|---:|---:|---:|---:|']
for source, run in [('Co60', 3), ('Am241', 12)]:
    row = old['sources'][source]['comparisons'][f'run{run}_last20']
    lines.append(f"|{source} / 后20%|{100*row['shape_nrmse']:.2f}%|" + '|'.join(f'{100*x:+.2f}%' for x in row['relative']) + '|')
display(Markdown('\\n'.join(lines)))
extra = read(OLD / 'waveform_feature_checks.json')['variants']['independent']['sources']
for source, run in [('Co60', 3), ('Am241', 12)]:
    print(source)
    for key in ('tpeak_minus_t10_ns', 'short_long_ratio', 'baseline_RMS_ADC'):
        row = extra[source][f'run{run}_last20'][key]
        print(' ', key, 'MC/data medians:', round(row['MC_q25_median_q75'][1], 4), round(row['data_q25_median_q75'][1], 4), 'KS:', round(row['KS'], 3))''')
    md('''## 2. 本轮参数搜索与验证规则

1. 先用冻结训练波形重估发光时间、快慢份额和共同读出 RC；加入上升段与积分比约束。
2. 快速期望模型只提出参数；候选必须送回完整 Geant4 产生光子，再经原生数字化验证。
3. 信号内残差只在训练段重估；基线校准仅使用前触发样本，单独标注采集条件。
4. run 3/12 中段、后段排除在本轮信号拟合之外；另设 run 18 前段训练分支，检查其中段/后段。各检查段以前已探索，不称为盲验证。
5. 展示固定 Q 排位的未缩放 ADC 示例，同时保留峰值/积分比/起伏分布、电荷谱以及失败项。

这些参数是当前装置假设下的条件估计，不能由拟合吻合反推唯一的本征材料常数或真实电路。''')
    md('''### 两类平均形状模型的搜索

先检验共享发光时间的模型，再允许电子与 α 沉积分开使用时间常数；读出参数仍共用。
第二类模型还允许工程已有的快路径份额变化。目标包含平均曲线、峰值时间、上升到 50% / 90% 的时间和积分比。
目标函数的尺度是描述性拟合权重，不是独立测量误差或 χ² 概率。

快速模型使用上一轮原生 MC 与解析均值的平滑差异来提出候选；这项差异不会写入原生波形。
完整 Geant4 只接收时间常数、份额、RC、光产额等数值参数。边界附近的参数及有限窗口内未约束的长时间常数均需保留说明。''')
    code('''for relative in ['shape_fit.json', 'parallel_shape/shape_fit.json']:
    path = NEW / relative
    if not path.exists():
        print(relative, '尚未完成')
        continue
    fit = read(path)
    best = fit['selected']['best']
    display(Markdown('**' + relative + '**'))
    print('Selected parameters:', json.dumps(best['parameters'], indent=2))
    print('Descriptive objective:', best['objective'])
    for source, row in best['metrics'].items():
        print(source, 'proposal mean NRMSE:', round(100*row['mean_nrmse'], 3), '%')
    print('这些是候选拟合值，实际吻合程度以下面的原生输出为准。')''')
    md('''### 基线与信号内起伏分开处理

基线状态只使用每条波形最前面的 128 个采样点。保留原平稳 FIR，允许事件间噪声方差具有均值为 1 的
对数正态乘子，以及基线电平变化。先检验单一状态；由于 Am 训练段包含少量基线偏移较大的记录，
再比较 1 / 2 个联合高斯状态，并用 BIC 选择，要求每个状态至少有 20 个有效样本。
状态同时描述基线电平与对数方差的相关性；有限 128 点估计噪声通过 FIR 协方差矩近似扣除。
这是一种采集状态描述，不是已确认的电路原因。关闭新增参数时，旧随机数和输出保持不变。

训练配置只用训练前触发区。若后续给出后段“前触发条件化”的配置，它明确使用了该段的前触发统计，
属于给定采集状态的比较，不能写成仅从训练段预测出了后段漂移；脉冲部分仍不用于这项基线估计。''')
    code('''path = NEW / 'baseline_states.json'
if path.exists():
    states = read(path)['states']
    rows = ['|数据段|基线 mV|噪声 RMS mV|事件方差 CV|实测基线 RMS 中位 ADC|', '|---|---:|---:|---:|---:|']
    for group, row in states.items():
        rows.append(f"|{group}|{row['baseline_mV']:.4f}|{row['noise_rms_mV']:.5f}|{row['event_variance_cv']:.3f}|{row['measured_RMS_ADC_quantiles'][1]:.3f}|")
    display(Markdown('\\n'.join(rows)))
else:
    print('基线统计尚未完成。')''')
    code('''path = NEW / 'baseline_mixture_fit.json'
if path.exists():
    mixtures = read(path)
    rows = ['|数据段|选中状态数|状态权重|说明|', '|---|---:|---|---|']
    for group, row in mixtures['groups'].items():
        weights = ', '.join(f'{100*w:.1f}%' for w in row['selected']['weights'])
        role = '训练前触发区' if group.endswith('first60') else '本段前触发条件，非预测'
        rows.append(f"|{group}|{row['selected']['components']}|{weights}|{role}|")
    display(Markdown('\\n'.join(rows)))
else:
    print('联合状态检查尚未完成。')''')
    code('''if (NEW / 'final_summary.json').exists():
    result = read(NEW / 'final_summary.json')
    display(Markdown(result['notebook_summary_markdown']))
    for relative in result.get('figures', []):
        display(Image(filename=str(PROJECT / relative)))
else:
    print('完整 Geant4 验证尚未完成，不预填拟合成功结论。')''')
    md('''## 6. 复现与来源

上一阶段：`simulation/results/signal_fluctuations_20261028/record.json`。
本轮数值：`simulation/results/waveform_refit_20261029/`。
笔记生成器：`jupyter/codex_0001_build_notebook.py`，使用现有 Jupyter Python 环境执行。
耗时输运在独立命令中执行；重新运行本笔记只读取保存的数值与图，不自动覆盖冻结模拟。
本轮不修改原始 BIN，也不修改共享文献库。

本轮修改仍在 `simulation/include/`、`simulation/src/` 所组织的 Geant4 工程内执行。
`ReadoutNoise.cc` 增加前触发状态混合、方差分散及相关性；`SignalDependentNoise.cc` 对全零信号跳过无效卷积。
原生完整模拟与重放均调用这些 C++ 模块。Python 负责参数搜索与统计，不替代核衰变、光学输运或 SiPM 雪崩。
最终构建的 5 个 CTest 目标、2 个原生/重放集成测试通过；测试包含逐元素 ADC 一致性、光子与雪崩保留、
状态混合的矩、异常参数、关闭/零信号退化。日志和源码快照存入本轮 `record.json` 对应归档。

当前配置：
- [run 3 / Co](../simulation/configs/csi_tl_run3_waveform_20261029.json)
- [run 12 / Am](../simulation/configs/csi_tl_run12_waveform_20261029.json)
- [run 18 / Am 单独采集校准](../simulation/configs/csi_tl_run18_waveform_20261029.json)

同名前缀的 `_last20_pretrigger.json` 使用后段前触发区校准，不能与前段预测结果混写。
上述配置分别与已验证配置逐字节相同。需要重新产生完整模拟时，在项目根目录使用**新的输出目录**：

```bash
./simulation/build/csi_sources simulation/configs/csi_tl_run3_waveform_20261029.json \\
  Co60 4000 291029503 simulation/results/reproduce_20261029_co60/events.root
./simulation/build/csi_sources simulation/configs/csi_tl_run12_waveform_20261029.json \\
  Am241 24000 291029512 simulation/results/reproduce_20261029_am241/events.root
python3 simulation/validate.py simulation/results/reproduce_20261029_co60/events.root \\
  simulation/results/reproduce_20261029_am241/events.root
```

数值与审计入口：
- [最终参数训练记录](../simulation/results/waveform_refit_20261029/final_parameters/fit.json)
- [新种子完整模拟验证](../simulation/results/waveform_refit_20261029/independent/analysis/validation.json)
- [run 18 单独校准检查](../simulation/results/waveform_refit_20261029/run18/validation.json)
- [各阶段起伏与形状数值 CSV](../simulation/results/waveform_refit_20261029/waveform_validation.csv)
- [特征中位数、条件区间及 KS 数值 CSV](../simulation/results/waveform_refit_20261029/feature_validation.csv)
- [本阶段源码、输入和输出哈希记录](../simulation/results/waveform_refit_20261029/record.json)

没有自动提交或推送 GitHub；本次交付是本地代码、可运行配置、冻结结果与已执行 notebook。''')
    notebook = nbf.v4.new_notebook(cells=cells, metadata={
        'kernelspec': {'name': 'codex-sipm', 'display_name': 'Python (SiPM notebook)', 'language': 'python'},
        'language_info': {'name': 'python', 'version': '3.13.5'},
    })
    path = PROJECT / 'jupyter/codex_0001_csi_geant4_waveform_calibration.ipynb'
    nbf.write(notebook, path)
    client = NotebookClient(notebook, timeout=180, kernel_name='codex-sipm',
                            resources={'metadata': {'path': str(PROJECT)}})
    client.execute()
    # Portable notebook metadata; execution above uses the task-local kernel.
    notebook.metadata['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3 (ipykernel)', 'language': 'python'}
    nbf.validate(notebook)
    assert not any(o.get('output_type') == 'error' for c in notebook.cells for o in c.get('outputs', []))
    nbf.write(notebook, path)
    print(path)
    print('Executed code cells:', sum(c.cell_type == 'code' for c in notebook.cells))


if __name__ == '__main__':
    build()
