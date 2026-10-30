# CsI(Tl) — Am-241 / Co-60 — SiPM 波形模拟

已实现并运行 **核衰变 → 能量沉积 → 闪烁光输运 → SiPM 雪崩 → 被动读出响应 → ADC → 实测对照**。只支持 `Am241` 和 `Co60`。本目录参考 GAGG_pileup 的 SiPM 数值内核，新建 CsI 几何、源、材料参数和本项目波形对照；未修改参考项目。

最新的[电荷/基线条件化与原生响应对照笔记](../jupyter/codex_0001_csi_conditional_response.ipynb)固定first60目标，检查原Q箱内两分、四分及Q×基线RMS匹配。Am后段平均时间宽度比从0.887变为Q两分匹配的 **0.917**（固定模型、块64条件95%区间[0.705,1.117]），匹配引起的变化区间包含0；中心仍为 **+7.26 ns**（[0.78,14.20] ns）。Q×RMS匹配缺少目标单元，不能可靠估计。Am实测后段相对前段的时间变化在Q两分匹配后仍约−5.02 ns，块64/128结论存在敏感性，不能直接归因于一种噪声。

同一批8000 Co / 24000 Am原光子历史完成 **12组、192000条有效原生响应重放**，新增独立母事件为0。原参数对照的全部32000条ADC与雪崩时间/电荷逐点不变；新增独立ROOT诊断树保存实际后脉冲计数和恢复程度。关闭经验信号残差后，单独测试AP概率0.05、延迟60/300 ns，AP概率0.15/300 ns压力对照，以及恢复60 ns；均固定光学、PDE、增益和读出。固定原事件/Q时，Am原经验残差的时间宽度为关闭残差后的2.36倍，而0.15/300 ns后脉冲仅为1.036倍，却把中心推迟34.96 ns。**这些预设机制对照尚不能替代经验残差，不是拟合出的器件参数，也不替换默认配置。** 空间串扰仍因阵列映射不保留邻接关系而禁用；重放未更新实验ROC。

见[条件化结果](results/conditional_response_20261030/figures/conditional_pid.png)、[匹配支持与ESS](results/conditional_response_20261030/figures/conditional_support.png)、[固定事件/重新筛选/固定锚点](results/conditional_response_20261030/figures/native_response_effects.png)、[原生均ADC与F](results/conditional_response_20261030/figures/native_response_waveforms.png)、[配对效应表](results/conditional_response_20261030/mechanism_table.csv)、[冻结方案](results/conditional_response_20261030/protocol.json)和[结论与下一步](results/conditional_response_20261030/decision.json)。中断产生的不完整文件归档在该轮`diagnostics`，未纳入分析。现有支持箱只有Co276、Am180条MC，所有条件区间均未包含拟合、装置和匹配目标的不确定性；原验证母事件在本轮已转作诊断探索。

此前的 [平均波形、PID与起伏联合约束笔记](../jupyter/codex_0001_csi_joint_constraints.ipynb)比较固定α慢时间 **4 / 6 / 8.4386 μs** 三组，只用run3/run12前60%校准。完成96000次完整Geant4校准母衰变及同历史噪声重放，再对选中的8.44 μs组运行 **8000 Co / 24000 Am独立种子完整输运**，合计128000个新独立母事件；96000次重放不重复计数。每组重新拟合非负信号相关噪声谱，并以实际原生ADC的中心、IQR、完整协方差、F和均波形联合排名。MC重采样中8.44 μs组排名第一的频率仅53.5%，4 μs组36.5%；这不是模型后验概率，α慢时间和有效归一化仍不能唯一确定。

新候选后段Co平均时间偏差由上一轮+6.32 ns降至 **−0.16 ns**（固定模型95%区间[−2.64,1.90] ns），三个PID带宽比为1.030/0.927/1.037；Am平均时间带宽比点估计由0.698变为 **0.887**（区间[0.689,1.072]），但中心偏差仍为 **+7.22 ns**（区间[0.73,13.79] ns）。两源均ADC NRMSE为1.34%/1.03%；Am的F50仍偏高约5.1%，3–3.8 μs额外尾部也保留残差。**本轮改善了Co中心与Am带宽点估计，尚未达到全面一致，继续保留为候选。** 区间不包括拟合、选组和装置不确定性；实测run12后段相对前60%的平均时间差约−5.06 ns，对实验块长度的敏感性也已报告，不能直接归因于某一种噪声。

见[原生均波形与额外尾部](results/joint_constraints_20261029/figures/joint_native_waveforms.png)、[中心/带宽](results/joint_constraints_20261029/figures/joint_native_pid.png)、[逐电荷箱PID](results/joint_constraints_20261029/figures/joint_pid_by_charge.png)、[起伏/协方差](results/joint_constraints_20261029/figures/joint_noise_controls.png)、[数值表](results/joint_constraints_20261029/validation_table.csv)及[冻结验证配置与命令](results/joint_constraints_20261029/validation/protocol.json)。本轮还修正了噪声提议中整数t₁₀边界的一个样点偏移：使用实际分数t₁₀后重算全部六组，五组未受影响的配置保持相同；原中间结果和[1398条校准波形的边界检查](results/joint_constraints_20261029/diagnostics/anchor_boundary_audit.json)保留。没有向原生波形注入实测模板，也没有直接给PID分数加宽；未在本轮更新同读出状态ROC或run18验证。

上一轮的 [Co 尾部与平均时间笔记](../jupyter/codex_0001_csi_co_tail_refit.ipynb)只用两源前60%联合拟合，保留四组目标/边界对照，随后完成 **8000 Co / 24000 Am 新种子完整 Geant4 母衰变**。Co后段平均时间差由−16.21 ns变为+6.32 ns（条件95%区间约[3.91,9.01] ns），2–3 µs电荷份额差由−0.957减至−0.084个百分点；中心仍未完全一致。新Co/Am均ADC波形NRMSE为1.43%/0.63%，但Am平均时间带宽比点估计从0.822降到0.698。**尾部改善尚未同时解决PID宽度，因此不替换默认配置。** 见[原生中心/尾部检查](results/co_tail_20261029/figures/tail_native_moments.png)、[均波形](results/co_tail_20261029/figures/tail_native_waveforms.png)、[PID与起伏](results/co_tail_20261029/figures/tail_native_pid_controls.png)、[数值表](results/co_tail_20261029/validation_table.csv)与[结论边界](results/co_tail_20261029/decision.json)。

带尾部约束时保留或放宽电子慢分量700 ns下界，得到的约780 ns最优值几乎相同；该轮效果主要对应目标权重/约束组合。α慢分量碰到15 µs上界，积分补偿使有效α归一化达到1.197，均不可解释为唯一的实测材料参数。旧MC参与提议校准，新旧输运不作同母事件配对；实验后段是已有内部检查数据。该轮提出的中心、PID协方差/宽度、均ADC和短尺度F联合约束，以及α长分量与归一化参数族，已在上方新笔记执行；两轮分别冻结。

此前的 [run 12 噪声状态笔记](../jupyter/codex_0001_csi_run12_noise_state.ipynb)完成 **18 组原生响应重放**：前触发样点按原始事件块交叉估计，比较原噪声、只改 RMS、基线/方差状态、状态与相关谱四组。后段实测基线 RMS 中位数为5.677 ADC，原模型3.174，新对照5.516–5.588；QDC/PCA带宽点估计向实测靠近，但**后段各配对变化95%区间均包含0，尚不能确认改善**。平均时间带宽比点估计仍为0.817–0.846，固定原事件/Q后为0.807–0.830；平均ADC波形NRMSE约1.17%–1.22%。本轮共用24000个原母事件，每段每个母事件仅选一份响应，新增独立母事件数为0。

这项检查只改变电子噪声，未改光子/雪崩、无噪声信号和信号期残差，也未替换默认配置。后段前触发状态参与条件化，不能称为盲预测；统计区间未包含拟合和模型不确定性。见[带宽与区间](results/run12_noise_state_20261029/figures/noise_pid_widths.png)、[固定事件/锚点对照](results/run12_noise_state_20261029/figures/noise_pid_paired_controls.png)、[108行数值表](results/run12_noise_state_20261029/validation_table.csv)、[结论边界](results/run12_noise_state_20261029/decision.json)与[冻结配置/命令](results/run12_noise_state_20261029/protocol.json)。该阶段提出的联合平均时间/尾部约束与慢分量边界检查，已在上方 Co 尾部笔记中执行；两轮结果分别冻结。

此前的 [PID 光电子统计笔记](../jupyter/codex_0001_csi_pid_photon_statistics.ipynb)在共同有效光电子规模与被动电压传输系数上扫描，只用实验前60%拟合。选出的“2倍光产额、0.5电压传输”候选以名义 PDE=0.3 重新运行 **8000 Co / 24000 Am 完整 Geant4 母衰变**。最后20%的 QDC 带宽比（MC/实测）从 run3 的1.468降到1.168，run12为0.897，run18为0.948；run12平均时间带仍偏窄（0.822），Co平均时间中心也有残差。未按单条实测波形缩放，平均 ADC 波形峰值归一化 RMSE 为 **0.96% / 1.17% / 0.96%（run3/12/18）**。这套候选尚未替换默认配置；参数不唯一，不能作为实测光产额/PDE。见[带宽与区间](results/pid_npe_20261029/figures/npe_bandwidth_validation.png)、[原始 ADC 均波形](results/pid_npe_20261029/figures/npe_waveform_validation.png)、[数值表](results/pid_npe_20261029/validation_table.csv)、[结论边界](results/pid_npe_20261029/decision.json)和[实际配置/命令](results/pid_npe_20261029/validation/protocol.json)。本阶段使用独立构建目录 `simulation/build/npe_20261029`，原构建及冻结结果保留。

此前平均波形校准的参数、笔记、图和复现命令见 [codex_0001 notebook](../jupyter/codex_0001_csi_geant4_waveform_calibration.ipynb)。2026-10-29 新种子完整输运的后段平均形状 NRMSE 为 **1.03% / 1.18%（Co / Am）**；另有前触发状态条件化和 run 18 单独读出校准。默认配置为 [run 3](configs/csi_tl_run3_waveform_20261029.json)、[run 12](configs/csi_tl_run12_waveform_20261029.json)、[run 18](configs/csi_tl_run18_waveform_20261029.json)；[未缩放 ADC 对照](results/waveform_refit_20261029/figures/unscaled_individual_waveforms.png)。峰值时间、电荷谱及装置参数退化仍限制解释，详细数值和适用边界写在 notebook。

[PID notebook](../jupyter/codex_0001_csi_geant4_pid.ipynb)在同一读出状态下对两源应用长短门、平均时间和冻结 PCA，使用沉积祖先真值检查测试母事件。提供 [PID 图](results/pid_20261029/figures/pid_truth_and_measurement.png)、[固定分类分区配置](results/pid_20261029/pid_classifier.json)和 [完整数值](results/pid_20261029/pid.json)。主场景长短门为92/92 α保留、0/62 γ误判；γ泄漏率单侧95%上限仍为4.72%。实验真值缺失、带宽不完全吻合，不能称为已测实验效率。重放没有增加独立输运历史。

[PID 带宽与噪声笔记](../jupyter/codex_0001_csi_pid_bandwidth.ipynb)新增逐分量协方差分解和信号期相关噪声联合校准，并用 **8000 Co / 48000 Am 新母衰变**检验。实验最后20%的 QDC 带宽比（MC/实测）在 run12 从0.726改善到0.881，run18 从0.811改善到0.882；Co 仍偏宽，run12平均时间带仍偏窄。平均波形和短尺度起伏保持接近，但**尚未全面解决 PID 带宽差异**，候选频谱未替换默认配置。见[带宽图](results/pid_bandwidth_20261029/figures/pid_bandwidth_validation.png)、[验证数值与区间](results/pid_bandwidth_20261029/validation/assessment.json)和[结论边界](results/pid_bandwidth_20261029/decision.json)。

以下保留 2026-10-28 阶段及工程通用说明，历史配置与数值不代表最新校准。该阶段增加**按采集 run 校准的信号相关随机残差**，在原生数字化中执行。参数冻结后重新运行 4000 个 Co、24000 个 Am 完整衰变；对未用于该次拟合的后段波形，归一化平均形状 NRMSE 为 **2.17% / 1.08%**，50/102/202 ns 起伏均值偏差均在约 5% 内。

[未缩放幅度的单条 ADC 对照](results/signal_fluctuations_20261028/independent/individual_native_calibration.png)、[平均形状、起伏和 Q 谱](results/signal_fluctuations_20261028/independent/native_calibration_summary.png)、[事件起伏分布](results/signal_fluctuations_20261028/independent/native_fluctuation_distributions.png)。新结果使用 run 3/12 前段 2450/1547 条训练，后段及 run 18 是此前已探索的内部检查；新随机种子验证的是模拟统计可重复性，并不产生新的实验验证数据。

历史阶段的 [联合筛选](results/response_candidates_20261028/response_candidates.png)仅重放原 14000 个光子历史，未找到同时满足两源、三个起伏尺度和电荷谱的候选。对应的 [Q 箱权重检查](../analysis/paper_baseline/results/robustness_20261028/mc_charge_matched.json)中，f=0.10 的 F₅₀ 偏差仍为 Co **+81.6% [72.6%, 90.1%]**、Am **−25.2% [−27.9%, −22.4%]**。这些冻结结果与新校准分开保存；[论文主分析](../analysis/paper_baseline/MANUSCRIPT_DRAFT.md)的 40795 条波形和 [原记录](../analysis/paper_baseline/results/publication_20261028/record.json)未覆盖。

此前的 [Gaussian FIR 噪声](results/colored_noise_20261028/noise_comparison/noise_and_signal.png)、[响应参数退化](results/bias_response_profile_20261028_final/response_profiles.png)和[单条波形示例](results/response_factorization_20261028/report/individual_waveforms.png)保留；[Co 几何](results/joint_optical_20261028/geometry_co/geometry.png)与 [Am 几何](results/joint_optical_20261028/geometry_am/geometry.png)沿用明确标注的候选假设。

## 2026-10-28 历史波形校准

配置为 [run 3](configs/csi_tl_run3_response_calibrated.json) 和 [run 12](configs/csi_tl_run12_response_calibrated.json)，与新完整输运使用的文件逐字节相同。它们是相应采集条件下的**经验校准**；原 `csi_tl_initial.json` 起点保持原样。配置继承的历史假设文字不能当成器件实测参数；本阶段的数值和状态以 `status`、`signal_dependent_noise` 及下述记录为准。

```bash
./simulation/build/csi_sources simulation/configs/csi_tl_run3_response_calibrated.json \
  Co60 4000 281028403 simulation/results/new_calibrated_co60/events.root
./simulation/build/csi_sources simulation/configs/csi_tl_run12_response_calibrated.json \
  Am241 24000 281028412 simulation/results/new_calibrated_am241/events.root
python3 simulation/validate.py simulation/results/new_calibrated_co60/events.root \
  simulation/results/new_calibrated_am241/events.root
```

每次使用新输出目录。完整流程仍是 Geant4 核衰变、光学输运和原生 SiPM 雪崩；没有把实测波形当模板，也没有按单条实测波形缩放模拟幅度。两个配置共用 105 ns 被动 RC，信号内额外协方差和事件间方差变化分别由各 run 训练数据约束。原生模块不读取源名或实验 run 标签；不同校准文件不代表已经确定“不同核素有不同 SiPM 增益”。

计算入口依次为 `calibrate_signal_fluctuations.py`、`fit_signal_fluctuations.py`、`fit_event_variance.py`；不要覆盖本次冻结结果重新拟合。既有原生输出可用下列命令重新核查、绘图和导表：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 simulation/validate_signal_fluctuations.py \
  --out simulation/results/signal_fluctuations_20261028/independent
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 simulation/summarize_signal_calibration.py
```

[42 行数值表](results/signal_fluctuations_20261028/waveform_validation.csv)、[额外形状与基线检查](results/signal_fluctuations_20261028/waveform_feature_checks.json)、[拟合](results/signal_fluctuations_20261028/fit.json)、[事件间变化拟合](results/signal_fluctuations_20261028/variance_scatter/fit.json)、[完整输运验证](results/signal_fluctuations_20261028/independent/validation.json)、[本阶段验证与源码归档](results/signal_fluctuations_20261028/record.json)。

## Geant4 工程结构

按参考 GAGG 工程的类命名和目录组织，每个实现类都有独立的 `.hh` / `.cc`。`main.cc` 只负责装配 RunManager 和选择宏、交互或命令行入口。

```text
simulation/
├── main.cc
├── include/       # 类声明
├── src/           # 类实现
├── macros/        # run_am241.mac / run_co60.mac / init_vis.mac
├── configs/       # 几何、光学、SiPM 和读出参数
├── vendor/        # 参考项目的原始 SiPM 数值内核及来源记录
├── tests/
└── CMakeLists.txt
```

| 类 | Geant4 基类 / 职责 |
| --- | --- |
| [DetectorConstruction](src/DetectorConstruction.cc) | `G4VUserDetectorConstruction`；材料、光学属性、几何、表面、步长和显示属性 |
| [PhysicsList](src/PhysicsList.cc) | `G4VModularPhysicsList`；Livermore 电磁、衰变、光学和步长限制 |
| [ActionInitialization](src/ActionInitialization.cc) | `G4VUserActionInitialization`；统一创建并注册各 Action 与数字化模块 |
| [PrimaryGeneratorAction](src/PrimaryGeneratorAction.cc) | `G4VUserPrimaryGeneratorAction`；Am/Co 母核与源几何 |
| [RunAction](src/RunAction.cc) | `G4UserRunAction`；开始/结束运行、输出生命周期和种子一致性检查 |
| [EventAction](src/EventAction.cc) | `G4UserEventAction`；清空事件、调用数字化、读取 DigiCollection 并保存 |
| [SteppingAction](src/SteppingAction.cc) | `G4UserSteppingAction`；各材料能损、向 SiPM 敏感探测器转交入口边界步骤 |
| [StackingAction](src/StackingAction.cc) | `G4UserStackingAction`；光子生成统计、提示衰变产物、记录窗外轨迹处理 |
| [TrackingAction](src/TrackingAction.cc)、[TrackInformation](include/TrackInformation.hh) | `G4UserTrackingAction` / `G4VUserTrackInformation`；保存 α、γ、衰变电子及其后代的来源，记录实际发光粒子 |
| [SiPMSD](src/SiPMSD.cc)、[SiPMHit](include/SiPMHit.hh) | `G4VSensitiveDetector` / `G4VHit`；只记录透入 SiPM 的光子，生成当前事件的 `SiPM/OpticalArrivals` 命中集合 |
| [SiPMDigitizer](src/SiPMDigitizer.cc) | `G4VDigitizerModule`；生成 `SiPMDigiCollection`，所有权交给当前 `G4Event` |
| [ReadoutResponse](src/ReadoutResponse.cc) | 可选快慢双路径及 `signal_transmission` 被动电压传输；相同雪崩电荷驱动原生 `BuildReadout`，传输系数只作用于输出信号电压，先于电子噪声执行；保存的 SiPM 雪崩电荷、物理电流及有限窗尾部电荷不衰减 |
| [ReadoutNoise](src/ReadoutNoise.cc) | 可选平稳 Gaussian FIR 电子噪声；由数字化模块调用，带随机前历史并重新量化 ADC，保持光子、雪崩与无噪声信号不变 |
| [SignalDependentNoise](src/SignalDependentNoise.cc) | 可选信号相关 Gaussian FIR 残差及均值为 1 的事件间方差乘子；使用独立随机流，在原生数字化及重放中同样执行，不改变光子、雪崩电荷或无噪声信号 |
| [RootIO](src/RootIO.cc) | ROOT 事件树、运行元数据和完成标记 |
| [RunConfig](src/RunConfig.cc) | JSON 校验和 `/csi/` 命令，初始化后禁止更改配置 |
| [DataStructure](include/DataStructure.hh)、[Constants](include/Constants.hh) | 事件记录、物理光子命中、器件常量 |

使用串行 RunManager。`SteppingAction` 把光学入口步骤转交 `SiPMSD`；后者检查 `G4OpBoundaryProcess`，仅接受真正透入的光子并终止，反射光子继续输运。`EventAction` 从 `G4HCofThisEvent` 读取命中集合，再调用数字化模块。实际命中位置保留毫米单位；微单元编号到参考内核内部坐标的转换集中在 `SiPMDigitizer`，不会作为 Geant4 几何尺寸使用。

schema 4 修正了旧模型“仅凭 post-volume 名称接收光子”的错误。短程核查发现旧模型也接收了菲涅耳反射光子，故本次物理修正后不再要求与旧 ADC 逐事件相同；保留修正前记录和源码，禁止混合两种模型的 MC 文件。详见[边界核查](results/sd_boundary_audit_20261027/finding.json)。

schema 5 增加来源 × 实际沉积粒子的能量矩阵、可选源出口片，以及明确关闭数字化的输运模式。schema 6 记录可选的 Geant4 有限闪烁上升时间及启用状态，默认关闭；schema 7 记录电子噪声模型，schema 8 增加 `readout_response` 元数据；当前 **schema 9** 另记录 `signal_dependent_noise`。省略新配置时原输出路径不变。可选功能保持在现有 Geant4 工程中实现；Python 扫描程序不替代核衰变或光学输运。

## 装置与参数状态

| 项目 | 当前处理 | 依据 / 限制 |
| --- | --- | --- |
| 晶体 | CsI(Tl)，10 × 10 × 10 mm³ | 用户确认；物质用 `G4_CESIUM_IODIDE`，Tl 发光由光学模型表示 |
| 入射面 | 5 μm Mylar | 用户确认；包含 α 能损 |
| 侧面 | PTFE 生胶带 | 材料已确认；0.1 mm 厚、0.95 有效反射率是假设 |
| SiPM | S13361-3050 系列，多个单元并联或求和 | 用户确认；具体型号后缀、接入数、位置和电路未知 |
| 初始阵列 | 4 × 4，16 个单元全部求和 | **假设**，不可当成实验记录；在 JSON 修改 `array_nx/ny` 和 `summed_channels` |
| 单元参数 | 3 × 3 mm²，3.2 mm 节距，3584 像元/单元 | 厂商系列资料；不是 GAGG 项目的 S13360-6050PE |
| 前放 | 无 | 用户确认；`transimpedance_ohm` 在这里表示有效电流—电压转换/负载，不表示新增前放 |
| 耦合层 | 0.1 mm 有效光学耦合层 | 材料和厚度未确认 |
| 源距 / 出口片 | 本轮候选 Co 1 mm + 0.25 mm Al；Am 30 mm，无出口片 | 用户允许扫描；不能当作实际源距、源封装已确认 |
| 采样 | 2496 点 × 2 ns，14 bit，正脉冲 | 实测 BIN/配置依据 |
| ADC 输入范围 | 本轮候选 0.5 Vpp，旧模型假设 2 V | 当前 XML 的 X4 设置支持 0.5 Vpp，但与历史 BIN 的完整关联未确认 |
| 光产额 / PDE / α 相对产额 | 初始 54000 / 0.30 / 0.50；本轮候选 18195.9 / 0.30 / 0.65 | 有效输出归一化与 PDE、增益、收光退化；不是本征材料参数测量 |
| 噪声 | 可选用 run 3 / 12 训练前触发区约束的 257 tap Gaussian FIR；省略配置时沿用独立高斯样本 | 改善平均协方差与频谱；固定平稳模型尚未复现弱脉冲、非高斯尾部及后段状态变化 |

厂商 [S13361-3050 系列资料](https://www.hamamatsu.com/content/dam/hamamatsu-photonics/sites/documents/99_SALES_LIBRARY/ssd/s13361-3050_series_kapd1054e.pdf) 第 2 页给出每通道典型 320 pF、3 V 过电压下典型增益 1.7×10⁶；这些是规格条件，**不能用典型击穿电压把 53.2 V 实验偏压直接换成真实过电压**。PDE 已含填充因子的影响，不再次乘填充因子。

`lowpass_ns` 与有效电流—电压转换系数是现阶段独立的等效参数，尚未由真实求和电路约束。初始 256 ns 来自“16 × 320 pF × 50 Ω”这一直接并联假设；旧拟合的 94.47 ns 和本次训练选择的 105 ns 均不可反推已经确认的单元数量或负载。

[CAEN DT5730 规格](https://caen.it/product/dt5730-dt5730s/)给出 X4 对应 0.5 Vpp 和 50 Ω 输入；它不证明实际 SiPM 求和电路。现存配置审计见 [settings_audit.json](results/settings_audit_20261028/settings_audit.json)。

## 构建与运行

依赖已安装的 Geant4 ≥11.2、ROOT、C++17，以及 Python 的 NumPy/PyROOT。当前环境为 Geant4 11.4.2、ROOT 6.34.10；没有新增安装依赖。

在项目根目录执行：

```bash
cmake -S simulation -B simulation/build \
  -DGeant4_DIR=/opt/geant4/11.4.2/lib/cmake/Geant4 \
  -DROOT_DIR=/opt/root/6.34.10/cmake -DCMAKE_BUILD_TYPE=Release
cmake --build simulation/build -j4
ctest --test-dir simulation/build --output-on-failure
```

`ctest` 的 5 个目标覆盖继承 SiPM 内核、平稳电子噪声、双路径响应、信号相关残差及 Python/PyROOT 工作流。新增 5 项残差检查验证关闭/零信号退化、方差与信号成比例、FIR 协方差、事件间方差乘子的矩、种子复现与异常配置。另运行 `python3 simulation/tests/test_response_replay.py` 的 2 项双源集成检查，验证相同种子下光子/雪崩/无噪声信号保留，原生数字化与重放的 ADC 逐元素一致。本阶段全部通过；[测试日志](results/signal_fluctuations_20261028/logs/sipm_signal_final_ctest.log)。

2026-10-29 光电子统计阶段在独立构建上通过5个 CTest 目标和3个双源重放集成测试。新增测试检查 `signal_transmission=1` 的逐元素旧路径一致性、0.5传输的信号电压减半、下游噪声保持，以及原生数字化与重放逐元素一致；系数必须在 `(0,1]`，省略时为1。测试使用 `CSI_TEST_BUILD_DIR=simulation/build/npe_20261029`，日志在 [本轮 logs](results/pid_npe_20261029/logs)。实际 SiPM 电荷守恒对应传输前的电流，不能把衰减后的 ADC 积分直接当成全部雪崩电荷。

和 GAGG 项目一样，可以直接交给 Geant4 执行宏。在 **SiPM 项目根目录**执行：

```bash
./simulation/build/csi_sources simulation/macros/run_am241.mac
./simulation/build/csi_sources simulation/macros/run_co60.mac
```

示例宏各运行 100 个衰变，输出到新的 `simulation/results/macro_am241/` 或 `macro_co60/`。再次运行前须在宏中指定新的输出路径。可用命令为：

```text
/csi/config simulation/configs/csi_tl_initial.json
/csi/source Am241
/csi/seed 280241
/csi/output simulation/results/my_am241/events.root
/run/initialize
/run/beamOn 100
```

`/csi/` 配置命令只允许在 `/run/initialize` 之前使用；只读的 `/csi/exportGeometry 新文件.json` 则在初始化之后使用。种子用 `/csi/seed` 同时配置输运和 SiPM 随机流，范围为 1–2147483646；避免只修改 `/random/` 而留下不一致的电子学种子。每个进程写一个文件、执行一次 `/run/beamOn`。宏中的相对路径相对于运行目录。

带图形界面的环境中，无参数执行 `./simulation/build/csi_sources` 会加载 `macros/init_vis.mac`，显示几何并进入 Geant4 交互会话；该可视化宏不自动生成事件。无显示器时可执行：

```bash
python3 simulation/inspect_geometry.py --out simulation/results/new_geometry
```

它执行 Geant4 几何重叠检查，导出实际 placements 的 JSON、Geant4 原生 `geometry.wrl` 三维文件、二维剖面/通道图和日志。可加 `--config 候选.json --source Co60` 检查指定场景。基础几何包含世界体积和 8 个子体积，可选出口片再增加 1 个；重叠检查每体积采样 10000 个表面点，不能替代未知装置参数的确认。图中通道是入口计分掩模，不是额外构造的半导体体积；Mylar 示意线加粗，JSON 保留真实 0.005 mm 厚度。源点属于生成器，显示于二维图和 JSON，不是 VRML 中的材料体积。

原来的六参数入口和下面的 Python 运行器继续可用。`run.py` 在冻结源码哈希前先执行增量构建；构建失败会停止，避免误运行遗留的旧可执行文件。

Geant4 数据库须可用。当前 `GEANT4_DATA_DIR` 为 `/opt/geant4/11.4.2/share/Geant4/data`。运行器把配置、源码/可执行文件哈希、种子、耗时、日志和 ROOT 保存在同一新目录；拒绝覆盖结果。

```bash
python3 simulation/run.py --source Co60 --events 100 --seed 280001 \
  --config simulation/results/calibration/csi_tl_effective.json \
  --out simulation/results/new_co60
python3 simulation/run.py --source Am241 --events 100 --seed 280002 \
  --config simulation/results/calibration/csi_tl_effective.json \
  --gap-mm 1 --out simulation/results/new_am241
python3 simulation/validate.py \
  simulation/results/new_co60/events.root simulation/results/new_am241/events.root
```

`configs/csi_tl_initial.json` 是文献/规格/假设起点；`results/calibration/csi_tl_effective.json` 是本批数据训练得到的**有效平均形状**配置。复制 JSON 到新文件修改假设，再使用新输出目录；不要覆盖已完成运行的冻结配置。完整光学输运较慢，首轮本机 400 个 Co 衰变约 82 秒，240 个 Am 衰变约 495 秒，实际速度取决于光子数和计算负载。

本轮通过完整光学复核的配置在 `results/joint_screen_20261028/Co60_candidate.json` 和 `Am241_candidate.json`。它们仍为条件候选，谱形一致性未通过。使用新目录复现，例如：

```bash
python3 simulation/run.py --source Co60 --events 1500 --seed 282103 \
  --config simulation/results/joint_screen_20261028/Co60_candidate.json \
  --out simulation/results/replay_joint/co60
python3 simulation/run.py --source Am241 --events 12000 --seed 282112 \
  --config simulation/results/joint_screen_20261028/Am241_candidate.json \
  --out simulation/results/replay_joint/am241
python3 simulation/validate.py \
  simulation/results/replay_joint/co60/events.root \
  simulation/results/replay_joint/am241/events.root
```

## 实测基准、标定与比较

使用已有 PHA CH0 分析和原始 BIN。训练限定 run 3 / run 12 的采集顺序前 60%，中间 20%、最后 20% 以及 run 18 用作检查。各段已经在前期分析中看过，不能称为盲验证。按冻结的质量筛选与共同电荷范围取最多 1200 条确定性样本，保留原始事件索引和源文件哈希。

```bash
python3 simulation/reference.py
python3 simulation/calibrate.py
python3 simulation/compare.py \
  --co simulation/results/new_co60/events.root \
  --am simulation/results/new_am241/events.root \
  --distance simulation/results/distance_scan/summary.json \
  --out simulation/results/new_comparison
```

`reference.py` / `calibrate.py` 会重建各自的派生参考/标定产物，重建前如需保留旧版本应另行归档；`run.py` / `compare.py` 的结果目录必须全新。比较支持同一源、**同一配置**的多个 ROOT 文件，以空格依次放在 `--co` 或 `--am` 后；不要混合参数扫描的不同配置。

比较沿用现有 `prepare.wave_features`：前 128 点扣基线、主峰前最后一次 10% 上升沿、长门约 [−40,3000) ns 和短门约 [−40,500) ns。形状按 t₁₀ 对齐并逐事件除以长门面积；该归一化只用于形状图，电荷谱保留原始 ADC·sample，没有按源另乘幅度因子。

实测参考额外要求原阶段四种门宽都完整。MC 在 1000 ns 注入衰变，未模拟硬件触发对记录起点的定位；MC 检查完整 3 μs 门和绘图范围。这个选择差别已写入比较 JSON，不能用当前结果声称复现了硬件触发时序。

标定模型为双指数发光与固定 2/29 ns SPE 响应、共享 RC 的卷积。两种源共享时间常数和读出参数，只允许主导分量份额不同；Am 主导分量暂配给 α 是**待检验假设**。241Am 中的 γ 始终保留，源标签及波形带不当成实测逐事件真值。保存多初值拟合和边界解，未报告无依据的材料寿命误差或拟合显著性。

对 schema 4–8 的完整光学输出可以继续按来源诊断电荷谱：

```bash
python3 simulation/diagnose_origin.py \
  --co simulation/results/new_co60/events.root \
  --am simulation/results/new_am241/events.root \
  --out simulation/results/new_origin_diagnosis
```

该工具按雪崩电荷占比 ≥90% 标出模拟中的 α 后代、γ 后代、衰变电子后代或混合事件；“衰变电子”含 β、转换电子和原子退激电子，不等同于纯 β。另统计 α 后代中由次级电子发光产生的电荷，检查 80/120/160/200 ADC 峰值标记的敏感性。电荷谱不作按源幅度缩放；峰值标记不是 PHA 硬件触发模型，实测事件不获得粒子真值标签。

保留原始 ADC 幅度的平均波形另用：

```bash
python3 simulation/compare_absolute.py \
  --diagnosis simulation/results/new_origin_diagnosis \
  --out simulation/results/new_absolute_comparison
```

只扣基线并对齐，不作共同电荷选择或面积归一化；使用实测训练段和来源诊断的 MC 质量筛选，额外检查完整绘图时间支持。上一轮的[未缩放波形图](results/corrected_optics_20261027/absolute_verified/absolute_waveforms.png)保留了幅度差异；本轮采用下述相同 200 ADC 分析条件重新比较。

### 联合筛选与相同分析阈值的完整 MC 对照

新增工具可重现本轮流程；以下各输出目录须为新目录：

```bash
python3 simulation/audit_acquisition.py --out simulation/results/new_settings_audit
python3 simulation/check_trigger_proxy.py --out simulation/results/new_trigger_proxy
python3 simulation/scan_transport.py --events 10000 \
  --out simulation/results/new_transport_candidates
python3 simulation/screen_joint.py \
  --transport simulation/results/new_transport_candidates \
  --out simulation/results/new_joint_screen
python3 simulation/validate_joint.py \
  --new simulation/results/replay_joint \
  --screen simulation/results/joint_screen_20261028 \
  --out simulation/results/replay_joint_validation
```

`scan_transport.py` 完成 9 个纯能量输运场景，保存来源×沉积粒子的能量库及源码快照。`screen_joint.py` 用该库和上一轮完整 MC 的响应系数筛选共享输出尺度、α 相对光产额、快份额及几何。代理误差先与已有 MC 核查；它只用于减少候选数量，选中参数仍须用 `run.py` 做完整光学复核。训练仅用 run 3 / 12 前 60%，未删除 Am γ 或按实测波形带赋真值。

`validate_joint.py` 在实测与 MC 上同时施加 amplitude ≥ 200 ADC 的分析条件，输出电荷谱、门比分布、未缩放平均 ADC 波形和后段检查。该阈值不模拟硬件触发。分位数损失和 KS 距离用于描述差异，不输出拟合显著性；完整光学结果仍未复现实测电荷谱宽度和尾部。`compare.py` 则继续使用原冻结共同 Q 选择，图中绿色线标为原平均形状拟合，红色线才是输入的新 MC。

`check_trigger_proxy.py` 的简单 FIR 只用于核查假设，没有成为选择模型：当前 XML 和简化 FIR 不能完整重放已记录波形的触发，详见 [RESULTS.md](RESULTS.md)。

### 电荷尾部、单条波形起伏与有限上升时间

`diagnose_charge.py` 保留原有训练样本，另按 `flags & 0x8000` 分组，比较电荷分箱的平均形状、硬件能量、原始波形和 MC 来源。其他非零状态位不会被一起删除；该位也不当作粒子真值。`diagnose_fluctuations.py` 使用固定共同 Q、200 ADC 分析阈值及实测该位未置位的样本，比较 50 / 102 / 202 ns 居中滑动平均残差的起伏。指标为脉冲区方差减前触发方差，再除平均信号；它不是增益、光电子数或噪声标准差。

```bash
python3 simulation/diagnose_charge.py --out simulation/results/new_charge_diagnostics
python3 simulation/diagnose_fluctuations.py --out simulation/results/new_fluctuations
```

`optical.rise_time_ns` 是可选的两元素数组，单位 ns，缺省 `[0,0]`。正值启用 Geant4 的 `SCINTILLATIONRISETIME1/2` 和 `SetScintFiniteRiseTime`，由 `PhysicsList::ConstructProcess` 在读取宏配置之后设置。两个上升参数对粒子类型共用，α/e 衰减参数仍可分别配置。按 [Geant4 11.4.2 的实现](https://github.com/Geant4/geant4/blob/v11.4.2/source/processes/electromagnetic/xrays/src/G4Scintillation.cc#L557-L570)，发射时间密度正比于 `(1-exp(-t/rise))*exp(-t/decay)`，不能把 RC 数值直接当成这里的 rise。

受控检验将旧 RC 的 94.4717 ns 极点从电子滤波移到随机发光时间，换算为两个 rise 参数 117.696 / 101.455 ns，`sipm.readout.lowpass_ns=0`，其他数值参数不变。连续线性平均响应相同，但单条波形的光子涨落会改变；有限记录窗、像元恢复和事件选择仍可能改变完整 MC。这个控制不是材料上升时间测量，也不宣称实际读出电路没有低通。

下面生成新目录、冻结设计和源码，然后通过 Geant4 宏运行与比较；默认 5000 次 Co 衰变、30000 次 Am 衰变：

```bash
python3 simulation/prepare_response_test.py --out simulation/results/new_response_control
simulation/build/csi_sources simulation/results/new_response_control/co60/run.mac
simulation/build/csi_sources simulation/results/new_response_control/am241/run.mac
python3 simulation/validate.py simulation/results/new_response_control/co60/events.root
python3 simulation/validate.py simulation/results/new_response_control/am241/events.root
python3 simulation/validate_joint.py \
  --new simulation/results/new_response_control \
  --old simulation/results/joint_optical_20261028 \
  --screen simulation/results/joint_screen_20261028 \
  --design simulation/results/new_response_control/design.json \
  --out simulation/results/new_response_control/joint_validation_final
python3 simulation/compare.py \
  --co simulation/results/new_response_control/co60/events.root \
  --am simulation/results/new_response_control/am241/events.root \
  --out simulation/results/new_response_control/common_charge
python3 simulation/diagnose_fluctuations.py \
  --mc simulation/results/new_response_control \
  --out simulation/results/new_response_control/fluctuations
python3 simulation/report_response_test.py \
  --control simulation/results/new_response_control \
  --out simulation/results/new_response_control/report
```

## 前触发噪声校准与可选相关噪声

`characterize_readout.py` 在 t₁₀ ≥768 ns 的已有波形中取前 512 ns，分别检查原基本质量样本与额外前触发筛选后的样本。其正负弱脉冲搜索规则相同，候选不当作已知单光电子。`prepare_noise_model.py` 仅用 run 3 / 12 前 60% 且满足原前触发质量条件的基线，生成 257 tap FIR 核与 RMS；后段和 run 18 不参与噪声校准。

FIR 系数来自经验噪声功率谱的平方根，经过窗口截断并归一化为平方和 1。电子噪声从独立高斯随机数生成，包含平稳随机前历史。`noise_rms_mV` 控制未去趋势时的 RMS；它不等于有限 512 ns 窗内去趋势后的 RMS。该参数化描述电子噪声，不是单雪崩信号响应。

顶层 `electronics_noise` 为可选字段，生成配置含完整系数和来源。以下只演示两点滤波，**不是实验校准系数**：

```json
"electronics_noise": {
  "model": "gaussian_fir",
  "sample_period_ns": 2.0,
  "coefficients": [0.7071067811865476, 0.7071067811865476]
}
```

系数必须有限、非空、平方和为 1，采样周期须与 ADC 一致。模块复用独立读出随机流，保留雪崩电荷和无噪声输出；模拟电压、截断计数和 ADC 量化由加入噪声后的结果重新计算。缺省配置沿用原独立高斯噪声。

```bash
python3 simulation/characterize_readout.py --out simulation/results/new_readout_noise
python3 simulation/prepare_noise_model.py \
  --noise simulation/results/new_readout_noise --out simulation/results/new_colored_noise
simulation/build/csi_sources simulation/results/new_colored_noise/co60/run.mac
simulation/build/csi_sources simulation/results/new_colored_noise/am241/run.mac
python3 simulation/validate.py simulation/results/new_colored_noise/co60/events.root \
  simulation/results/new_colored_noise/am241/events.root
python3 simulation/validate_joint.py \
  --new simulation/results/new_colored_noise --old simulation/results/joint_optical_20261028 \
  --screen simulation/results/joint_screen_20261028 \
  --design simulation/results/new_colored_noise/design.json \
  --out simulation/results/new_colored_noise/joint_validation_final
python3 simulation/diagnose_fluctuations.py --mc simulation/results/new_colored_noise \
  --out simulation/results/new_colored_noise/fluctuations
python3 simulation/compare_noise.py --new simulation/results/new_colored_noise \
  --noise simulation/results/new_readout_noise --out simulation/results/new_colored_noise/noise_comparison
python3 simulation/profile_signal_response.py --out simulation/results/new_signal_constraints
```

最后一个工具是固定单雪崩 / 增益假设下的平稳散粒噪声诊断，不代替 Geant4，也不输出“已测得的 RC”。源码与数值假设见工具说明和 [RESULTS.md](RESULTS.md)。

## 已有偏压序列和响应退化检查

`analyze_bias_response.py` 使用已确认的 8 个 run，按原始序号前 60% / 中间 20% / 最后 20% 分组；原 feature 文件和训练标记不变。脉冲样本采用原 primary、幅度 ≥200 ADC、`0x8000` 未置位、冻结共同 Q 和 t₁₀ ≥768 ns，每组最多 3000 条均匀抽样。输出波形均值、三个起伏尺度、固定电荷分箱、前后频谱和正负弱候选。

`profile_bias_response.py` 在平稳独立雪崩近似下，扫描两个响应极点和有效 ADC 面积。默认 `--statistic mean` 与期望矩的解释相符；`--statistic median` 保留为描述性敏感性比较。它明确保留两极点互换的退化、有限网格边界和同偏压共同响应的残差。5% / 10% / 20% 容差范围不是参数置信区间，有效面积也不是单光电子增益；不自动改写 Geant4 配置。

```bash
python3 simulation/analyze_bias_response.py --out simulation/results/new_bias_response
python3 simulation/profile_bias_response.py \
  --bias simulation/results/new_bias_response --out simulation/results/new_bias_profile
python3 simulation/tests/test_bias_response.py
```

如只更新已有图形，可运行 `analyze_bias_response.py --out 原结果目录 --replot`，它保持数值结果不变并另存绘图代码与输入哈希。数值提取和参数扫描拒绝覆盖已有输出目录。完整执行来源、与旧指标的逐事件回归及源码快照见[本轮记录](results/bias_response_20261028/record.json)。

## 原生读出记录与有限窗矩核查

`diagnose_mc_components.py` 在冻结 MC 事件中，将 ADC 起伏精确分解为保存的信号、电子噪声、量化与交叉协方差，区分事件均值和中位数。`diagnose_nonstationary_moments.py` 通过 PyROOT 调用原 `vendor/SiPMResponse.cc` 的 C++ `BuildReadout`，重放真实雪崩记录，并用同一内核计算实际到达时间下的有限窗单雪崩自项。它们是完整 MC 输出的诊断，不执行新的核衰变或更换 Geant4 输运。

```bash
python3 simulation/diagnose_mc_components.py \
  --mc simulation/results/colored_noise_20261028 --out simulation/results/new_mc_components
python3 simulation/diagnose_nonstationary_moments.py \
  --mc simulation/results/colored_noise_20261028 \
  --components simulation/results/new_mc_components --out simulation/results/new_native_moments
python3 simulation/tests/test_mc_components.py
python3 simulation/tests/test_nonstationary_moments.py
```

已核查的 158 条无噪声信号重放差异为零；实际时间自项与 MC 组均值相差约 0.2%–2.6%，其配对统计区间包含零。这是内部一致性检查，不是实测响应精度，见[原生矩结果](results/nonstationary_moments_20261028/moments.json)。

## 局部协方差、原生响应候选与固定 PSD 检查

`analyze_residual_covariance.py` 直接沿用冻结偏压样本的事件编号，在六个 Q 箱及统计门比范围内，计算等长基线/脉冲窗的局部协方差。分组均值结构与跨事件项分开报告；后者仍含定时、混合成分与非平稳效应，不等于单独测得的光电子噪声。`--resume` 只接受相同分析代码的分组检查点，`--replot` 不改变数值。

```bash
python3 simulation/analyze_residual_covariance.py \
  --out simulation/results/new_local_covariance
python3 simulation/estimate_voltage_feedback.py \
  --covariance simulation/results/new_local_covariance/covariance.json \
  --out simulation/results/new_local_covariance/voltage_feedback_bounds.json
python3 simulation/tests/test_residual_covariance.py
```

电压工具只将已有事件峰值按两种 ADC 输入范围转换，再给出若干假设过电压下的比例；不能将 ADC 端电压当作已知 SiPM 偏压节点变化。实际负载和求和拓扑仍未知。

可选顶层配置如下，数值是本轮候选，**不是校准结果**：

```json
"readout_response": {
  "model": "parallel_paths",
  "prompt_charge_fraction": 0.10,
  "prompt_decay_ns": 5.0,
  "prompt_lowpass_ns": 0.0
}
```

快支路沿用原 `rise_ns`、电流—电压转换和采样孔径；只替换衰减与额外低通。两支的权重作用于积分电荷，不是峰值；快衰减必须大于上升时间，低通 ≥0，快份额在 [0,1]。省略配置或份额为零保持旧响应。Gaussian FIR 噪声在混合后只施加一次。此模块是有效线性响应族，不是已确定元件的 S13361 阵列电路。

重新运行完整输运可照常用 `run.py`；仅改变数字化时，可用新 C++ 程序重放完成的 ROOT 光子历史：

```bash
simulation/build/replay_response \
  simulation/results/colored_noise_20261028/co60/events.root \
  simulation/results/response_candidates_20261028/confirmation_final/native_best/co60/config.json \
  simulation/results/new_response_replay.root
python3 simulation/validate.py simulation/results/new_response_replay.root
```

输出文件必须不存在。重放拒绝改变几何、光学和采样，沿用父文件事件种子，保留全部事件并重算雪崩来源、电荷、信号和 ADC。它没有重新执行 Geant4 输运，不能作为新增独立 MC 计数；若要改变光学或源距，必须重新运行完整输运。平铺微胞映射下的空间串扰仍禁止开启。

本轮候选筛选的默认实测输入为 `bias_response_20261028` 和 `residual_covariance_20261028` 的冻结结果。用新目录重现：

```bash
python3 simulation/screen_response_candidates.py prepare --out simulation/results/new_response_candidates
python3 simulation/screen_response_candidates.py replay --out simulation/results/new_response_candidates
python3 simulation/screen_response_candidates.py screen --out simulation/results/new_response_candidates
python3 simulation/confirm_response_candidate.py \
  --screen simulation/results/new_response_candidates \
  --out simulation/results/new_response_candidates/confirmation
python3 simulation/plot_response_confirmation.py \
  --screen simulation/results/new_response_candidates \
  --confirmation simulation/results/new_response_candidates/confirmation
python3 simulation/tests/test_response_replay.py
```

前两步保存冻结设计及原生基础输出，第三步组合相同噪声下的两支信号，重新量化并重算选择。确认步骤对描述性最优快份额作全事件原生重放，逐点核对 ADC，另作与 AP=0.15 的探索性组合。后者是在首轮筛选之后追加的条件检查，不是预先独立验证。完整性检查逐项保存，恢复运行仍核对输出哈希；原始中断目录未作为最终证据。

`check_frozen_psd_transfer.py` 保持 phase2 的 3 μs 门宽、PCA 中心/向量/尺度不变；仅在已有偏压样本中计算相对于同源训练段的分数分布变化。不存在额外 PCA 训练，也不把 KS 距离转换为粒子效率。它与原 FOM 样本有明确的幅度/状态位/窗口支持区别。

```bash
python3 simulation/check_frozen_psd_transfer.py --out simulation/results/new_frozen_psd_transfer
python3 simulation/tests/test_frozen_psd_transfer.py
```

本轮 BIN 文件哈希沿用冻结偏压记录，没有再次遍历原始大文件；当前代码读取选中波形并检查布局/特征支持，具体逐文件核验范围见 JSON。当前数值、源码快照和各阶段日志见[本轮记录](results/response_candidates_20261028/record.json)。

## 源距扫描

```bash
python3 simulation/scan_distance.py --events 1000 \
  --out simulation/results/new_distance_scan
```

默认六个空气间隙 0.5、1、2、5、10、20 mm，每点 1000 个 Am-241 衰变。为减少耗时，新扫描关闭闪烁和数字化，仅研究材料输运；**ADC 数组为空，不能用于波形比较**。旧输运文件可能保留纯基线噪声 ADC，也不是波形模拟结果。`--resume` 只接续配置、种子和事件数相同、已完成并通过检查的目录；不把不完整文件视为结果。

源到前 Mylar 外表面的距离是 `source_gap_mm`。初始点源各向发射，未加未知的源活性层、基底、封装或准直器。特别是 Co-60 的 β 也会被产生；真实封装若阻挡 β，会改变电荷谱。没有把模拟父核数量除以实验时长来冒充活动度或探测效率。

可以在 `geometry` 中加入有限平面出口片；它只表示一个明确假设，不等价于完整源外壳：

```json
"source_exit_foil": {
  "material": "G4_Al",
  "thickness_mm": 0.25,
  "distance_from_source_mm": 0.01,
  "width_mm": 20.0,
  "status": "Hypothetical exit foil; source encapsulation unknown"
}
```

片的源侧表面距源为 `distance_from_source_mm`，朝晶体方向延伸 `thickness_mm`；两者之和须小于源距，正方形宽度须小于世界体积宽度。省略该对象即不放置出口片。`run.py --transport-only` 同时设光产额为 0 和 `acquisition.digitize=false`；配置校验拒绝在正光产额下关闭数字化。

## 输出与物理实现

- `Events`：每个父核衰变一行，保留零沉积和未过近似阈值事件。含 CsI / Mylar / 空气能损、按沉积粒子分的 CsI 能损、光子入射时间/位置/能量、雪崩时间/电荷、无噪声信号、含噪模拟电压及 ADC。
- `decay_*`：所有记录窗内由 `RadioactiveDecay` 产生的粒子，包括提示子核退激。只看 `ParentID==1` 会漏掉 Am γ 和 Co γ 级联，因此保存 `decay_parent_id`。模拟真值只用于模拟诊断。
- `*_by_origin_*`、`photon_origin`、`avalanche_origin`：来源顺序为 unknown、α 后代、γ 后代、衰变电子后代、反冲核后代、other；`*_emitter_pdg` 另存实际发光粒子。`charge_by_origin_C` 是记录窗内雪崩电荷，不能当作有限积分门的 ADC 电荷。schema 4 的 `photon_boundary_status` 必须为 1/2/12，即透入状态。
- schema 5 的 `edep_origin_species_MeV` 是按来源优先展开的 6×3 矩阵，第二维为 α、e±、其他沉积粒子；用于保留 α 后代次级电子的不同光产额和时间响应。`RunInfo.digitization_enabled` 标记是否有真实 ADC，关闭时波形和雪崩数组为空。
- `RunInfo`：Geant4 版本、完整配置、种子和模型边界；新运行增加 `completed` / `events_completed`，由 `RunAction` 正常结束时更新。父核寿命设零表示“条件于本次衰变发生”，避免数百年绝对时钟丢失纳秒精度；不修改核衰变分支，不把子核长寿命链压到当前记录窗。
- 光学输运：两发光分量、粒子相关相对产额、有效侧面 PTFE 反射、Mylar/耦合界面、真实阵列单元间隙。到达入口的光子再经过 SiPM PDE 抽样，不把所有入射光子直接计为光电子。
- SiPM：每通道 3584 个有效像元、像元恢复、有限占据、电荷守恒的 SPE 与读出卷积、ADC 孔径/量化/饱和。实际空间到有效像元均匀近似；平铺索引不保留真实邻接，空间串扰被禁止开启。暗计数/后脉冲默认关闭是未标定的简化，不代表器件没有这些效应。
- 当前不模拟 SiPM 内的直接电离信号、电路元件级求和、连续源计数率、堆积、死时间或完整 PHA 触发。`approximate_trigger` 只是信号幅度阈值标记，未据它删掉事件。

ROOT 树分批写入，避免把长模拟的全部波形都积在内存；最终完成情况以退出码、事件数及 `validate.py` 为准。中断产生的文件不得直接投入比较。

## 来源与核验边界

[vendor/provenance.json](vendor/provenance.json) 保存从 GAGG_pileup 复制的源码及 SHA-256。数值内核和其 14 项测试保持原样；测试夹具中的 S13360 型号仅测试通用计算，生产模拟使用本项目 S13361 配置。新几何、核衰变和 ADC/实测流程另有 [集成测试](tests/test_workflow.py)。

发光初值参考 Valentine 等，*NIM A* 325 (1993) 147–157 的 Horiba +23°C 样品 679 ns / 3.34 μs 及积分份额。依据为[共享库既有笔记](</media/sf_works/papers/sources/1993_Valentine_NIMA_Temperature-dependence-of-CsI(Tl)-gamma-ray-excited-scintillation-characteristics.md>)所定位的原文表 2（PDF 第 5 页）；笔记为待人工审核 AI 草稿，本文使用的是初值建议，不把它迁移为本晶体的已验证参数。文献使用限制同 [本地证据汇总](../analysis/paper_baseline/LOCAL_EVIDENCE.md)。共享文献库按只读使用。

目前软件和首轮物理数值检查通过，实验幅度/谱形一致性尚未通过。论文可先把这套实现作为可复现的方法框架，不能据此宣称已测得 α 接受率、γ 泄漏率、实际源距或本征发光时间常数。
