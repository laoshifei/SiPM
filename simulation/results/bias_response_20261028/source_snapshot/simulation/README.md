# CsI(Tl) — Am-241 / Co-60 — SiPM 波形模拟

已实现并运行 **核衰变 → 能量沉积 → 闪烁光输运 → SiPM 雪崩 → 被动读出响应 → ADC → 实测对照**。只支持 `Am241` 和 `Co60`。本目录参考 GAGG_pileup 的 SiPM 数值内核，新建 CsI 几何、源、材料参数和本项目波形对照；未修改参考项目。

先看 [结果与适用边界](RESULTS.md)、[偏压序列](results/bias_response_20261028/bias_overview.png)和[响应参数退化](results/bias_response_profile_20261028_final/response_profiles.png)。最新检查覆盖 8 个有效 run 的 40795 条脉冲波形，同偏压的平均形状相近也不能保证起伏一致；条件扫描未被当作新校准参数。最近一次 14000 次完整衰变模拟验证了可选 Gaussian FIR 噪声模块，[基线相关性明显改善](results/colored_noise_20261028/noise_comparison/noise_and_signal.png)，但单条脉冲起伏和电荷谱仍不吻合。此前的有限上升时间控制检验与[单条波形示例](results/response_factorization_20261028/report/individual_waveforms.png)保留；[Co 几何](results/joint_optical_20261028/geometry_co/geometry.png)与 [Am 几何](results/joint_optical_20261028/geometry_am/geometry.png)沿用明确标注的候选假设。

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
| [ReadoutNoise](src/ReadoutNoise.cc) | 可选平稳 Gaussian FIR 电子噪声；由数字化模块调用，带随机前历史并重新量化 ADC，保持光子、雪崩与无噪声信号不变 |
| [RootIO](src/RootIO.cc) | ROOT 事件树、运行元数据和完成标记 |
| [RunConfig](src/RunConfig.cc) | JSON 校验和 `/csi/` 命令，初始化后禁止更改配置 |
| [DataStructure](include/DataStructure.hh)、[Constants](include/Constants.hh) | 事件记录、物理光子命中、器件常量 |

使用串行 RunManager。`SteppingAction` 把光学入口步骤转交 `SiPMSD`；后者检查 `G4OpBoundaryProcess`，仅接受真正透入的光子并终止，反射光子继续输运。`EventAction` 从 `G4HCofThisEvent` 读取命中集合，再调用数字化模块。实际命中位置保留毫米单位；微单元编号到参考内核内部坐标的转换集中在 `SiPMDigitizer`，不会作为 Geant4 几何尺寸使用。

schema 4 修正了旧模型“仅凭 post-volume 名称接收光子”的错误。短程核查发现旧模型也接收了菲涅耳反射光子，故本次物理修正后不再要求与旧 ADC 逐事件相同；保留修正前记录和源码，禁止混合两种模型的 MC 文件。详见[边界核查](results/sd_boundary_audit_20261027/finding.json)。

schema 5 增加来源 × 实际沉积粒子的能量矩阵、可选源出口片，以及明确关闭数字化的输运模式。schema 6 记录可选的 Geant4 有限闪烁上升时间及启用状态，默认关闭；当前 schema 7 另记录电子噪声模型。可选功能保持在现有 Geant4 工程中实现；Python 扫描程序不替代核衰变或光学输运。

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

`lowpass_ns` 与有效电流—电压转换系数是现阶段独立的等效参数，尚未由真实求和电路约束。初始 256 ns 来自“16 × 320 pF × 50 Ω”这一直接并联假设；拟合后的 94.47 ns 不可反推已经确认的单元数量或负载。

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

`ctest` 同时运行 14 项继承 SiPM 内核检查、3 项电子噪声检查及 10 项 Python/PyROOT 集成检查，包括宏/命令行等价、种子重放、来源守恒、反射排除、几何导出、输运模式、有限上升时间，以及改变噪声时光子 / 雪崩 / 无噪声信号逐元素不变。另运行 `python3 simulation/tests/test_charge_diagnostics.py` 和 `python3 simulation/tests/test_readout_characterization.py`，分别检查状态位 / 波形诊断，以及频谱单位、正负搜索对称性、FIR 统计和频域积分收敛。

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

对 schema 4–7 的完整光学输出可以继续按来源诊断电荷谱：

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
