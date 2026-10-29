# 已有波形的论文分析基线

更新日期：2026-10-28。当前以已确认来源的 PHA CH0 数据开展波形方法研究。²⁴¹Am 按 **α/γ 混合源**保存，源名称不充当逐事件粒子标签。现有数据支持比较共同 ADC 电荷区间内的波形形状；α 接受率、γ 泄漏率和最低甄别能量尚不能由这些源标签直接计算。

## 新增模拟校准（2026-10-28）

模拟目录已增加原生信号相关随机残差，并用冻结 run 3/12 前段训练。参数冻结后的新完整输运为 Co 4000 / Am 24000 个衰变；后段平均形状 NRMSE 为 2.17% / 1.08%，三尺度起伏均值偏差约在 5% 内。电荷谱、Co 峰值时间、Am 后段基线及跨 run 迁移仍未通过完整一致性检查。该阶段是采集条件相关的经验校准，和以下冻结的共同快响应分析分开保存。[结果与模型边界](../../simulation/RESULTS.md)、[ADC 示例](../../simulation/results/signal_fluctuations_20261028/independent/individual_native_calibration.png)、[数值表](../../simulation/results/signal_fluctuations_20261028/waveform_validation.csv)、[归档](../../simulation/results/signal_fluctuations_20261028/record.json)。原主分析与补充阶段的记录、图表及源码快照不覆盖。

## 全量波形与相同 Q 权重补充（2026-10-28）

取消每段 3000 条上限，复核 run 12/18 原始前 60% 的全部合格波形，分别为 **1547/65034 条**。在 50 ns 滤波、500–900 ns 局部窗和 0.30–0.35 形状条件下，六个 Q 箱等权的相对差异为 **+58.0% [54.6%, 59.4%]**；细分至 24 箱仍为 +58.24%。这是既有数据的敏感性检查，正式的 40795 条队列及原约 57% 的结果保留。

模拟与实测在共同支持 Q 箱内使用相同权重后，固定 f=0.10 候选的 F₅₀ 偏差仍为 **Co +81.6% [72.6%, 90.1%]、Am −25.2% [−27.9%, −22.4%]**。主支持要求每箱各候选至少 10 条，Co 仅余 2 箱 24 条 MC、Am 3 箱 79 条；箱内连续电荷仍可能不匹配，不向缺失箱外推。

- [更新工作稿及两张补充图](MANUSCRIPT_DRAFT.md)
- [全量、细分箱、序号区间和配对抽样](results/robustness_20261028/full_cohort_covariance.json)
- [MC 共同支持、箱权重及有效抽样数](results/robustness_20261028/mc_charge_matched.json)
- [12 份补充 CSV 的字段、单位和来源](results/robustness_20261028/tables/manifest.json)、[压缩包](results/robustness_20261028/supplementary_tables.zip)
- [独立归档与验证记录](results/robustness_20261028/record.json)

本补充阶段只读使用 `publication_20261028` 的数值与原记录。复现本阶段时使用下列命令；不要重新运行旧 `verify_publication.py` 来归档本次改稿，以免覆盖其历史源码快照。

```bash
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
python3 analysis/paper_baseline/robustness_full_cohort.py prepare
python3 analysis/paper_baseline/robustness_full_cohort.py run
python3 analysis/paper_baseline/robustness_mc_charge.py
python3 analysis/paper_baseline/test_robustness.py
python3 analysis/paper_baseline/plot_robustness.py
python3 analysis/paper_baseline/export_publication_tables.py
python3 analysis/paper_baseline/verify_robustness.py
```

## 主分析：统一队列与不确定度

正式分析沿用此前冻结的 **40795 条事件编号**，训练仅含 run 3/12 前段的 **3997 条**。所有方法和窗口使用同一训练队列，PCA 与拟合背景参数在每次配对块重采样中重新估计。3 μs 下，负平均时间−QDC 的中位 FOM 差为 **0.234 [0.096, 0.305]**，PCA−QDC 为 **0.165 [0.124, 0.203]**，PCA−负平均时间为 **−0.069 [−0.146, 0.070]**；均为条件 95% 区间，不能宣布 PCA 或负平均时间普遍最优。

跨 run 检查已补等样本量背景、训练不确定度、相同支持下的选择敏感性。局部协方差每次重估分组均值，run 12/18 在相同 Q 和形状条件下相差约 **57% [53%, 59%]**；不能唯一归因于增益或温度。原生 MC 按共享父衰变配对重采样后，响应不吻合仍在。FOM 的绝对值也依赖峰形假设，Student-t 检查保留在补充结果中。

- [完整工作稿、6 张主图及图注](MANUSCRIPT_DRAFT.md)
- [队列流量表](results/publication_20261028/cohort_flow.csv)、[冻结协议](results/publication_20261028/protocol.json)
- [方法/窗口配对区间](results/publication_20261028/paired_uncertainty.json)、[峰形模型检查](results/publication_20261028/core_model_sensitivity.json)
- [跨 run 区间](results/publication_20261028/transfer_uncertainty.json)、[等样本量](results/publication_20261028/transfer_matched.json)、[共同支持敏感性](results/publication_20261028/transfer_selection_sensitivity.json)
- [局部协方差区间](results/publication_20261028/covariance_uncertainty.json)、[MC 误差条](results/publication_20261028/mc_uncertainty.json)
- [数值验证、输入和输出哈希、源码归档](results/publication_20261028/record.json)

依赖现有 NumPy、PyROOT、g++ 及此前生成的特征和原生 MC 文件，不需联网安装。从仓库根目录复现：

```bash
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
python3 analysis/paper_baseline/publication_stats.py prepare
python3 analysis/paper_baseline/publication_stats.py point
python3 analysis/paper_baseline/test_publication_stats.py
python3 analysis/paper_baseline/publication_compute.py verify
python3 analysis/paper_baseline/publication_compute.py bootstrap --block-size 64 --draws 256
python3 analysis/paper_baseline/publication_compute.py bootstrap --block-size 32 --draws 128
python3 analysis/paper_baseline/publication_compute.py bootstrap --block-size 128 --draws 128
python3 analysis/paper_baseline/publication_stats.py summarize
python3 analysis/paper_baseline/publication_model_check.py
python3 analysis/paper_baseline/publication_covariance.py
python3 analysis/paper_baseline/publication_transfer.py matched
python3 analysis/paper_baseline/publication_transfer.py bootstrap
python3 analysis/paper_baseline/publication_transfer_sensitivity.py
python3 analysis/paper_baseline/publication_mc_uncertainty.py prepare
python3 analysis/paper_baseline/publication_mc_uncertainty.py bootstrap
python3 analysis/paper_baseline/test_publication_diagnostics.py
python3 analysis/paper_baseline/plot_publication.py
python3 analysis/paper_baseline/verify_publication.py
```

FOM、局部协方差和跨 run 重采样有逐次检查点；只有完整 256 次训练结果才进入跨 run 区间。编译加速保留与 Python 相同的三组初始化、EM 方程和收敛条件，验证记录见 [computation.json](results/publication_20261028/computation.json)。结果不含盲测或候选选择后的统计校正，不把相同 G4 光子历史的重放计作独立输运。

## 历史分析的范围

第二轮已完成质量条件敏感性、共享形状混合模型、三种方法与四种记录窗口、时间分段及 run 18 复核，见 [第二轮结果](PHASE2_RESULTS.md)。已形成 [方法与结果草稿](MANUSCRIPT_DRAFT.md)。下文保留第一轮数据建档与初始诊断的范围。

## 已确认的样本

以下对应关系由用户确认，逐事件特征均来自各目录的 `UNFILTERED/Data_run_N.BIN`。

| PHA run | 通道 | 放射源 | 偏压 V | 保存的波形数 | 当前 settings 中的触发阈值 LSB |
| --- | --- | --- | --- | ---: | ---: |
| 3 | CH0 | ⁶⁰Co | 53.2 | 4,776 | 120 |
| 7 | CH0 | ⁶⁰Co | 53.4 | 4,747 | 160 |
| 10 | CH0 | ²⁴¹Am，α/γ 混合 | 52.4 | 4,961 | 200 |
| 11 | CH0 | ²⁴¹Am，α/γ 混合 | 52.8 | 96,632 | 90 |
| 12 | CH0 | ²⁴¹Am，α/γ 混合 | 53.2 | 7,516 | 160 |
| 13 | CH0 | ²⁴¹Am，α/γ 混合 | 53.6 | 80,649 | 160 |
| 14 | CH0 | ²⁴¹Am，α/γ 混合 | 54.0 | 112,249 | 160 |
| 18 | CH0 | ²⁴¹Am，α/γ 混合 | 53.2 | 156,966 | 200 |

合计 **468,496 条波形**。用户明确排除 run **9、21、22、24、25**；当前图与共同区间分析均不使用它们。其余未确认来源的数据仅列入清单。

run 3、12、18 使用同一晶体、SiPM 和接线，均为名义 53.2 V。放大器增益和温度未单独确认；当前设置中的触发阈值也不同，因此 run 12 与 18 先作为同一探测器的两次源测量，不认定为严格同条件重复。

[labels.csv](labels.csv) 保存每个 run—通道的源、偏压、采用状态及用户备注；[user_provenance.json](user_provenance.json) 保存本次确认的范围。脚本重跑会保留已填字段和排除状态。

## 原始文件与采集记录的对应

PHA 共有 24 个非空 BIN、786,517 条波形，逐条确认均为 board 0、CH0、2496 点。run 23 只有文件头；run 0 未发现可供本分析使用的波形。波形时间戳未发现倒退，这不能证明目录中所有采集记录都属于同一轮。

从波形中的硬件能量重新填谱，与 37 份保存的能谱逐 bin 比较，包含上下溢出，共找到 24 个完全匹配的快照，每个非空 BIN 各一个。多快照目录中关键的匹配如下：

| run | 与 BIN 完全匹配的能谱文件时间标签 | 当前 run.info 的采集开始时间 |
| --- | --- | --- |
| 10 | 2022-12-06 23:38:59 | 2022-12-07 15:39:29 |
| 11 | 2022-12-07 15:51:00 | 2022-12-07 15:57:50 |
| 12 | 2022-12-07 17:28:32 | 2022-12-07 17:32:46 |

用户进一步确认 run 10 的 **²⁴¹Am、52.4 V 对应上述 12 月 6 日的 BIN**。能谱文件时间标签不是波形的绝对起始时间；当前 `settings.xml` 与早先 BIN 的逐项关联也不能只凭同目录保证。表中的阈值因此保留为“当前设置值”，不据较新的 `run.info` 推算旧波形的计数率、活时间或触发效率。完整证据见 [histogram_association.json](results/histogram_association.json)。

其他数据的实际非空通道为：

| 数据目录 | run 或文件编号 | 非空通道 |
| --- | --- | --- |
| `data/DT5730_DPP_PSD/DAQ` | 1–5 | board 0，CH0 |
| `DT5720_DPP_PSD/raw2root` | 6、21 | board 0，CH0 |
| `DT5720_DPP_PSD/raw2root` | 28、30–33、35–39 | board 0，CH2 |
| `data/XIA/rootfile/data_C1_0002_wave.root` | 2 | crate 0，slot 2，CH0、2、3、4、5、6 |
| `data/XIA/rootfile/data_C1_0003_wave.root` | 3 | crate 0，slot 2，CH0、2、3、4、5、6 |

这些系统的相同 run 编号不能互相替代。全清单共 53 个非空 run—通道组合，见 [run_channels.csv](results/run_channels.csv)。PSD 原生 ROOT 的 `RAW` 与 `UNFILTERED` 计数不同，不能当作两批独立事件或默认同样的筛选条件。旧 PSD ROOT 的原始采集配置尚未关联。

XIA `data_all.root` 的 1,164,024 条记录，与上述两个文件按顺序拼接的全部事件键完全一致；六条抽查波形也一致。因此合并文件不作为额外样本。该核验覆盖全部标量事件键和六条波形，未逐字节比较全部波形，见 [xia_merge_audit.json](results/xia_merge_audit.json)。ROOT 文件中的旧 TTree key cycle 不重复计数。

## 第一轮波形结果

源标签确认前已固定本轮诊断窗口，没有按源间分离效果调参。所有 PHA 数据均做逐事件检查，当前采用的 8 个 run 中有 **468,329 条**未触发下述基础诊断标志，其余 167 条仍完整保存在特征文件中。这个比例只描述诊断条件，不是探测效率或粒子选择效率。

全体 786,517 条波形中，触及 ADC 边界 1 条、幅度不超过 5 倍基线 RMS 的 62 条、长积分窗不完整的 228 条、长积分非正或缺失的 265 条；条件之间有重叠。尚未完成可靠的堆积识别，硬件 flags 保留原值而未猜测位含义，也未因 flags 非零直接删除事件。

- [基线和波形诊断图](results/pha_qc_overview.png)：仅显示已确认采用的 8 个 run。点为中位数，误差棒为事件分布的 5%–95% 分位数，不是中位数的统计误差。
- [53.2 V 光输出覆盖与波形形状图](results/pha_charge_coverage.png)：run 3、12、18；二维图颜色表示各自的事件数，色标分别缩放。
- [相同积分区间内的形状分布](results/pha_matched_projections.png)：相同六个积分分箱、按各 run 的分箱内事件总数归一化；图例保留源名称。
- [两条形状带的原始波形抽查](results/pha_waveform_band_examples.png)：共同积分范围内每组最多四条，按候选事件序号等间隔选择；保留完整记录并作基线扣除、幅度归一和 t10 对齐，没有添加随机扰动。

run 3、12、18 各自积分分布中央 90% 范围的交集为 **201,824–418,733 ADC·sample**。这一界限仅根据数据覆盖确定，未优化 PID。三批在交集内分别有 **4,257、2,741、108,352** 条基础诊断可用事件；六个分箱中每批均不少于 152 条。逐箱计数和分位数见 [matched_charge_bins.csv](results/matched_charge_bins.csv)。

六个分箱中，run 3 的短长积分比中位数约为 0.460–0.467，run 12 为 0.321–0.325，run 18 为 0.323–0.325。两份 ²⁴¹Am 源数据的主要形状分布在相同积分下较接近；这仍是源测数据的分布比较，不能把 ²⁴¹Am 中的全部事件解释为 α。⁶⁰Co 数据本身也有位于较低比值带的少量事件，应检查其波形、硬件标志和本底来源，不能预先删除后再宣称 γ 泄漏很小。

run 10 的积分比分布主体更接近 ⁶⁰Co 源数据，与其他已列 ²⁴¹Am run 不同。其源与偏压按用户确认保留；这种差异应进入混合成分和采集条件核查，不能直接把 run 10–14 拼成“纯 α 的偏压扫描”。

针对共同积分范围，另以 `0.28≤R<0.38` 和 `0.43≤R<0.60` 两个探索性区间抽查了 20 条原始波形。run 3 的低比值区有 42 条，hardware flags 全为零，四条样例呈连续长尾、单个主脉冲；没有据此把整带视为坏事件的依据。run 18 的高比值区有 13 条，flags 也全为零，但抽查四条可见前触发扰动和负尾部，提示基线估计或邻近脉冲影响，不能直接把它们标成 γ。run 12 在所选高比值区没有事件，这不等于其源不发射 γ。区间仅用于诊断抽查，不是粒子真值或已验证的筛选门。计数、flags 和事件号见 [waveform_band_audit.json](results/waveform_band_audit.json) 与 [waveform_inspection_events.csv](results/waveform_inspection_events.csv)。

## 特征定义和数值验证

输入保持只读。PHA 使用 CoMPASS BIN 的逐事件字节布局直接解析，检查文件尾、每条记录长度、board/channel 和时间戳顺序。时间戳保存为 uint64，硬件 flags 为 uint32，不经过浮点转换。变长或截断文件会明确报错，不按固定长度悄悄继续。

当前 settings 标记记录长 4992 ns，实际波形均为 2496 点，对应 2 ns/点，与 `sampleTime=2000` 一致。正极性和 14 bit ADC 取自本地配置。

| 特征 | 定义 |
| --- | --- |
| 基线 | 前 128 点均值；另存前 64 点均值、前后各 64 点均值差 |
| 基线噪声 | 前 128 点的总体标准差 |
| 幅度 | 第 128 点以后最大值减基线 |
| 前沿 t10 | 最大值之前最后一个 10% 幅度上升交点，线性插值；它是初始对齐方法，不是堆积判据 |
| 积分锚点 a | 上述交点的上侧整数采样点 |
| 短积分 | 基线扣除后，半开区间 `[a−20, a+250)` 的和，约 −40 至 +500 ns |
| 长积分 | 同一起点，半开区间 `[a−20, a+1500)` 的和，约 −40 至 +3000 ns |
| 其他积分 | `charge_1us`、`charge_2us` 的终点分别为 a+500、a+1000，起点仍为 a−20；名字表示终点而非总窗宽 |
| 末端水平 | 最后 128 点均值减基线；未要求脉冲在记录末端完全回到基线 |

积分单位为 ADC·sample；乘 2 可写为 ADC·ns，均未作能量或电荷标定。此处光输出是有限窗口积分代理量，不等于已经证明的 keVee，也不等于 α 的真实沉积能量。

`diagnostic_flags` 各位依次表示触及数值边界、幅度非正、幅度≤5 RMS、完整积分窗不可用、长积分非正或缺失。形状图使用这些位全为零的事件。窗口越界返回 NaN，不补零或缩短窗口冒充完整积分。

已通过 6 项数值/数据完整性测试，覆盖解析精度、截断和变长拒绝、解析矩形脉冲积分、时间平移、窗口越界、数值边界及来源记录保留。另与已有 run 3 ROOT 对照总事件数和首、中、末三条完整波形及标量，均一致，见 [verification.json](results/verification.json)。

现有 C++ 分析脚本另有需要保持区分的定义：`qdc_long/short` 是窗口均值，旧 `get_fom.cpp` 分母是两个 σ 之和，旧归一波形绘图还加入随机扰动。本轮使用独立脚本的积分和无随机扰动数据；没有覆盖既有实验参数或把旧 FOM 当作 FWHM 定义的 FOM。

## 后续写作重点

第二轮已执行前触发稳定性检查、两核心混合模型、积分比／平均时间／PCA 比较及跨 run 复核。结果同时揭示 run 12 内的基线噪声变化和 run 18 的峰宽差异；固定峰形不能被视为已经普适验证。后续写作需补全实际装置描述，并解释统计成分与粒子来源的关系，保留源混合、刻度和独立验证范围的限制。

文章现阶段可围绕“混合源条件下的波形形状比较与有限记录窗口选择”组织方法和结果。经验 α 接受率、γ 泄漏率、低能甄别阈值以及温度稳定性仍需要各自证据；现有有限积分与源标签不会自动提供这些结论。[本地文献依据](LOCAL_EVIDENCE.md) 给出方法出处及适用边界。

## 复现与文件

在项目根目录使用现有 Python、NumPy、PyROOT 环境运行；无需联网或安装依赖：

```bash
python3 analysis/paper_baseline/prepare.py all
python3 analysis/paper_baseline/audit_provenance.py
python3 analysis/paper_baseline/verify.py
python3 analysis/paper_baseline/make_figures.py
python3 analysis/paper_baseline/audit_waveforms.py
```

也可单独运行 `prepare.py pha`、`root` 或 `labels`。执行顺序为先产生特征，再核对快照和绘图；`labels.csv` 中仅 `analysis_status=include` 的 PHA run 进入总览图，共同区间图明确使用已确认的 run 3、12、18。

- `results/pha_inventory.json`：每个 PHA BIN 的计数、配置快照、硬件 flags 和长度检查。
- `results/root_inventory.json`：原生 PSD、旧 PSD 和 XIA 的非空通道及计数；尚未对这些系统做完整波形质量分析。
- `results/pha_runNNNN_features.npz`：全量逐事件特征、原始事件序号及标量。
- `results/pha_runNNNN_wave_sample.npz`：按事件序号均匀抽取的最多 32 条完整波形，仅供诊断，非带标签模板。
- `results/pha_qc.csv`：所有 PHA run 的诊断汇总，包括后续被用户排除的历史核查行。
- `results/confirmed_cohort.json`：当前采用范围、共同积分区间及计数。
- `results/execution_environment.json`：软件版本及原始文件路径、大小、mtime。它不是全数据文件的密码学哈希证明。

PNG 供快速检查，同名 SVG 可用于后续编辑。图为探索性分析图，文件不会改动共享文献库或原始实验数据。
