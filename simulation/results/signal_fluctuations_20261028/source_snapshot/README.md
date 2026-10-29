# SiPM

基于 C++ / CERN ROOT 的 SiPM 实验数据分析程序，用于处理数字化采集卡记录的波形，
提取能量和长短门信号，并研究 CsI(Tl) 中的 α/γ 脉冲形状甄别。

## 目录与数据流程

| 目录 | 用途 |
| --- | --- |
| `DT5730_DPP_PHA/` | PHA 数据转换、排序、波形分析、积分门扫描 |
| `DT5720_DPP_PSD/` | PSD 数据转换、排序和波形分析 |
| `XIA/` | 已解码 XIA ROOT 数据的波形查看、平均波形与粒子甄别 |
| [`simulation/`](simulation/README.md) | CsI(Tl) 中 Am-241 / Co-60 衰变、光学输运、S13361 阵列读出与实测波形比较 |
| `PLAN.md` | CsI(Tl) α/γ 甄别、能量阈值与稳定性的研究计划 |
| `EXPERIMENT_INFO.md` | 探测器、放射源、run 对照及测量条件的可填写信息表 |

主要流程：

```text
UNFILTERED/Data_run_N.BIN
  -> raw2root/runNNNN.root（按通道保存 tr_ch00、tr_ch01 等）
  -> sort/runNNNN_sort.root（时间排序后的 tr）
  -> 平均波形、能谱、长短门分布和 PID 分离度
```

原始数据、ROOT 结果、图表、逐事件数组、实验刻度文件和编译产物不纳入 Git。代码、配置、数值汇总和结果说明保存在仓库；文档中的图像及数组链接指向本地生成结果，需按各目录说明重新生成。

当前 [论文工作稿](analysis/paper_baseline/MANUSCRIPT_DRAFT.md)已统一 40795 条冻结波形，补齐重新训练 PCA 的配对方法区间、跨 run 等样本量和质量检查、重估均值的局部协方差区间，以及按共享父衰变配对的原生 MC 误差条，形成 6 张主图。3 μs 下 PCA 与负平均时间的差值区间包含零；run 12/18 的局部起伏差异约 57%，但不能据此确定唯一物理原因。模拟平均波形接近仍不足以同时复现实测起伏和电荷谱。[复现入口及结果记录](analysis/paper_baseline/README.md)；[模拟结果与历史阶段](simulation/RESULTS.md)。
最新补充已去掉 run 18 前段 3000 条上限，复核 **65034 条**合格波形：相近形状下的局部差异为 **58.0% [54.6%, 59.4%]**。MC 与实测使用相同的共同支持 Q 箱权重后，Co/Am 短时起伏仍分别偏高/偏低。新增 2 张补充图及 [12 份数据表](analysis/paper_baseline/results/robustness_20261028/supplementary_tables.zip)，验证与原主分析分开[归档](analysis/paper_baseline/results/robustness_20261028/record.json)。

最新模拟已加入按 run 训练的原生信号相关残差，冻结参数后另运行 **4000 个 Co / 24000 个 Am 完整 Geant4 衰变**。后段实测对照的归一化平均形状误差为 **2.17% / 1.08%**，三个尺度的起伏均值偏差约在 5% 内；电荷谱、Co 峰值时间、Am 后段基线与 run 18 迁移仍未吻合。它是明确标注的经验校准，不是唯一物理解释。[单条 ADC 对照](simulation/results/signal_fluctuations_20261028/independent/individual_native_calibration.png)、[可运行配置与复现](simulation/README.md)、[完整结果和限制](simulation/RESULTS.md)。

本地可能另有 `PKUXIADAQ/`，它是单独分发的 XIA Pixie-16 采集软件；本仓库的
`XIA/` 宏读取其解码结果。实验 run、通道选择和门参数需要与对应数据核对。

## 构建

需要 64 位 Linux、GNU make、g++ 和 CERN ROOT 6，且 `root`、`root-config` 在 PATH 中。
编译使用当前 ROOT 提供的编译及链接参数。

```bash
make -C DT5730_DPP_PHA/raw2root
make -C DT5730_DPP_PHA/sort
make -C DT5730_DPP_PHA/optimize_gate_par
make -C DT5720_DPP_PSD/raw2root
make -C DT5720_DPP_PSD/sort
```

## 转换与分析

程序使用相对于工作目录的输入输出路径，请在相应子目录运行。以下以 PHA run 3 为例。

1. 将数据放在 `data/DT5730_DPP_PHA/DAQ/run_3/UNFILTERED/Data_run_3.BIN`。
2. 在 `DT5730_DPP_PHA/raw2root/` 中执行 `./raw2root 3`，得到 `run0003.root`。
   也可用 `./raw2root 3 /absolute/path/input.BIN` 显式指定输入文件。
3. 在 `DT5730_DPP_PHA/sort/` 准备 `SiPM_cali.dat` 和 `SiPM_ts_offset.dat`，然后执行
   `./sort 3`，得到 `run0003_sort.root`。

PSD 的对应默认数据目录为 `data/DT5720_DPP_PSD/DAQ/`，也支持显式输入路径。
某些本地数据目录名为 `DT5730_DPP_PSD`，应先核对采集配置中的实际板卡型号和数据格式，
再显式指定文件；程序不会把这两个硬件目录自动混用。

转换器读取带 2 字节头部标志的 CoMPASS 二进制事件，按标志读取能量、短门能量及波形。
文件必须符合源码使用的小端字段布局。排序还要求输入包含 `energy_ch`、波形和时间戳。

转换、排序和参数扫描使用 ROOT `CREATE` 模式，已有同名结果时会报错退出。
重新分析前请先移走或重命名旧结果。任何非零退出码都表示本次处理失败；失败过程中
可能留下未完成的输出文件，不能作为有效分析结果使用。

### 刻度和时间修正

两个参数文件均需包含通道 0–7，每通道一行，支持空行及 `#` 注释：

```text
# SiPM_cali.dat: channel p0 p1 p2 chi2
0 0 1 0 0
1 0 1 0 0
2 0 1 0 0
3 0 1 0 0
4 0 1 0 0
5 0 1 0 0
6 0 1 0 0
7 0 1 0 0
```

```text
# SiPM_ts_offset.dat: channel offset_ps
0 0
1 0
2 0
3 0
4 0
5 0
6 0
7 0
```

能量计算为 `p0 + p1 * energy_ch + p2 * energy_ch²`；时间修正为
`timestamp + offset_ps`，修正量须为整数皮秒。
缺失文件、缺失通道、重复通道或不完整参数行会报错。

以上示例采用恒等能量变换和零时间修正，**不代表实际实验刻度，也不会把 ADC 道址转换为 keV**。
物理分析必须使用该批实验的参数。

### 波形与 PID

`sort/set.h` 指定基线采样数及长短门起止点，坐标单位为采样点，区间为 `[start, stop)`。
排序输出保留板卡号、通道号、刻度能量、修正时间戳、基线、峰值和波形特征：

- `energy_qdc`：扣基线后的全波形求和。
- `qdc_long`、`qdc_short`：各门内求和后除以门宽，即门内平均幅度。
- `v_data`、`v_dt`：扣基线波形与采样点索引；积分门扫描版本不保存这两个向量。

在 `analysis/`、`get_data/` 中用 ROOT 加载对应宏进行拟合、绘图或导出。
各宏中的 run、筛选阈值和输入文件是实验设置，使用前需核对。
在 `sort/` 下计算两个 run 的 PID 分离度：

```text
root -l
.L get_fom.cpp
get_fom_single(3, 11)
```

该宏拟合 `(qdc_long-qdc_short)/qdc_long` 的分布，并输出
`abs(mean2-mean1)/(abs(sigma1)+abs(sigma2))`；这里分母用的是高斯标准差之和，
不是半高全宽之和。空分布或拟合失败时不输出分离度。

在 `DT5730_DPP_PHA/optimize_gate_par/` 中准备同样的两个参数文件后，可执行：

```bash
./sort 3 500 2000 400 900
```

参数依次为 run、长门起止点、短门起止点，结果写入自动创建的 `rootfile/`。
这些数字仅示范命令格式；`get_batch.cpp` 可以生成扫描脚本，但大范围扫描会产生大量结果文件。

`XIA/analysis/ana.C` 和 `XIA/pid/pid.C` 在 ROOT 中加载后可创建 `ana`、`pid` 对象。
默认数据路径由各自头文件位置定位到项目的 `data/XIA/rootfile/`，也可向构造函数传入已有 `TTree`。

## 分析前核对

先选一个小 run，确认原始数据和 ROOT 中的事件数、板卡号、通道、时间戳及波形一致，
再检查基线、峰值、积分和刻度能量。所有检查使用单独的输出目录。
后续工作及验收条件见 [PLAN.md](PLAN.md)。
