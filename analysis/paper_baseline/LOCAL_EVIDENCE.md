# 现有波形研究的本地文献依据

下列证据支持方法选择和论文论证边界。资料来自只读共享文献库；本轮定向核对过原 PDF 的相关位置，范围记录在 [检索清单](../../.cache/paper-search/2026-10-27-article/manifest.json)。这不代表重新精读了每篇全文，也不把既有 AI 笔记视为用户已认可的结论。

Skulski 2001 的 DOI 另据 [Bendel 2013 原文](</media/sf_works/papers/raw/2013_Bendel_EPJA_RPID-A-new-digital-particle-identification-algorithm-for-CsI(Tl)-scintillators.pdf>)第 7 页参考文献 11 核对；这里只用它核对书目信息，不把引用关系当作额外性能证据。

## 可直接用于方法设计的依据

| 论文 | 核验位置与可用依据 | 对本项目的限制及已有意见 |
| --- | --- | --- |
| W. Skulski、M. Momayezi，2001，*Particle identification in CsI(Tl) using digital pulse shape analysis*。DOI 10.1016/S0168-9002(00)00938-4 | PDF 第 6 页图 3：在相同光输出区间比较 α/γ 波形；第 10 页图 7–8：加权方法与带宽的作用。支持先匹配光输出、再比较积分与加权指标。 | 光电二极管读出，与本地 SiPM 链不同；同数据调参不能当独立验证。DOI 在本地另一文献书目中核对，原笔记仅列 PII。笔记为 AI 草稿，未发现针对本篇的明确用户采用意见。 |
| Tao Liu 等，2023，*Toward real-time digital pulse process algorithms for CsI(Tl) detector array at external target facility in HIRFL-CSR*。DOI 10.1007/s41365-023-01272-6 | PDF 第 8 页表 3、图 8 与第 9 页图 10：同起点长短积分、上升时间和 RPID 的比较。支持明确定义窗口、定时和计算步骤。 | FOM 比较具有能区和 CSA 读出条件，不能直接移植最佳参数或换算为纯度；不能用标题中的 real-time 代替本地吞吐测量。笔记仍为 AI 草稿。 |
| T. Teranishi 等，2021，*Pulse shape analysis of signals from SiPM-based CsI(Tl) detectors for low-energy protons: Saturation correction and particle identification*。DOI 10.1016/j.nima.2020.164967 | PDF 第 4 页式 2、表 2、图 5：SiPM 恢复和饱和会影响观测波形。支持将幅度依赖和读出响应纳入检查。 | 研究 p/d 等带电粒子及其能量范围，不是本项目 α/γ 数据；有效恢复参数不等于直接测得本征时间常数。笔记的 AI 解释不等于用户采用。 |
| Xing-Chi Han 等，2023，*CsI-bowl: an ancillary detector for exit channel selection in γ-ray spectroscopy experiments*。DOI 10.1007/s41365-023-01289-x | PDF 第 3 页式 1、图 3：FOM 的分母为两峰 FWHM 之和；第 4 页式 4、图 4–5：波形拟合包含前放响应。 | PD/APD 源测和在束结果条件不同。用户第 646 项仅限定采用在束选道和有条件的源测 FOM；93% 是已探测样本的可分辨比例，不能写成正确率或探测效率。旧状态文字不能覆盖较新的局部审核。 |
| F. Amorini 等，2012，*Investigation of the Dependence of CsI(Tl) Scintillation Time Constants and Intensities on Particle's Energy, Charge and Mass Through Direct Fitting of Digitized Waveforms*。DOI 10.1109/TNS.2012.2201499 | PDF 第 3 页式 2–3：含前放卷积的拟合；结合独立 ΔE–E 粒子标记。支持区分观测波形、读出响应和粒子真值。 | 拟合排除比例不是识别准确率；本项目没有对应的独立 ΔE–E 标签。笔记采用边界仍以其现有审核记录为准。 |

## 原文和笔记

- Skulski 2001：[原文 PDF](</media/sf_works/papers/raw/2001_Skulski_NIMA_Particle-identification-in-CsI(Tl)-using-digital-pulse-shape-analysis.pdf>)；[完整笔记](</media/sf_works/papers/sources/2001_Skulski_NIMA_Particle-identification-in-CsI(Tl)-using-digital-pulse-shape-analysis.md>)。
- Liu 2023：[原文 PDF](</media/sf_works/papers/raw/2023_Liu_NST_Toward-real-time-digital-pulse-process-algorithms-for-CsI(Tl)-detector-array-at-external-target-facility-in-HIRFL-CSR.pdf>)；[完整笔记](</media/sf_works/papers/sources/2023_Liu_NST_Toward-real-time-digital-pulse-process-algorithms-for-CsI(Tl)-detector-array-at-external-target-facility-in-HIRFL-CSR.md>)。
- Teranishi 2021：[原文 PDF](</media/sf_works/papers/raw/2021_Teranishi_NIMA_Pulse-shape-analysis-of-signals-from-SiPM-based-CsI(Tl)-detectors-for-low-energy-protons-saturation-correction-and-particle-identification.pdf>)；[完整笔记](</media/sf_works/papers/sources/2021_Teranishi_NIMA_Pulse-shape-analysis-of-signals-from-SiPM-based-CsI(Tl)-detectors-for-low-energy-protons-saturation-correction-and-particle-identification.md>)。
- Han 2023：[原文 PDF](</media/sf_works/papers/raw/2023_Han_NST_CsI-bowl-an-ancillary-detector-for-exit-channel-selection-in-γ-ray-spectroscopy-experiments.pdf>)；[完整笔记](</media/sf_works/papers/sources/2023_Han_NST_CsI-bowl-an-ancillary-detector-for-exit-channel-selection-in-γ-ray-spectroscopy-experiments.md>)；[用户采用意见](</media/sf_works/papers/global-review.md:13032>)（第 646 项）。
- Amorini 2012：[原文 PDF](</media/sf_works/papers/raw/2012_Amorini_TNS_Investigation-of-the-Dependence-of-CsI(Tl)-Scintillation-Time-Constants-and-Intensities-on-Particle's-Energy-Charge-and-Mass-Through-Direct-Fitting-of-Digitized-Waveforms.pdf>)；[完整笔记](</media/sf_works/papers/sources/2012_Amorini_TNS_Investigation-of-the-Dependence-of-CsI(Tl)-Scintillation-Time-Constants-and-Intensities-on-Particle's-Energy-Charge-and-Mass-Through-Direct-Fitting-of-Digitized-Waveforms.md>)。

源身份与事件身份的区分、混合模型的验证要求和本项目数据划分是针对当前数据的分析设计，不能借上述论文声称本地混合成分或分类性能已经得到验证。

## 2026-10-28：响应模型与稳健性检查的补充依据

按项目 `paper-search` 技能只读检索共享库，复用完整既有笔记并定向核对原文。新的文本与页面图均在本项目缓存，详见[本轮核验清单](../../.cache/paper-search/2026-10-28-next-steps/extraction_manifest.json)。以下页码从本地 PDF 第 1 页计；这是关键段落核验，不宣称本轮重新精读全文或运行论文软件。四篇现有笔记均为 AI 草稿，本轮未发现这些具体模型的用户科学采用确认；Skulski / Teranishi 的 AI 归并记录不算人工认可。

| 论文 | 本轮实际核验及可用论点 | 本项目采用与边界 |
| --- | --- | --- |
| F. Corsi 等，2007，*Modelling a silicon photomultiplier (SiPM) as a signal source for optimum front-end design*。DOI 10.1016/j.nima.2006.10.219 | PDF 1–3 页相关正文；第 2 页图 1–2 已看原页。单胞信号受到其余微胞、寄生电容、输入负载与带宽影响；波形比较需要包含读出。 | 用于提出具有快慢通路的候选及电路约束需求，不据此确认 S13361 元件值。本地笔记提示原文反演式的量纲/符号疑点，本轮没有直接实现这些反演式；当前双路径模块不是论文全电路的独立复现。 |
| E. Garutti、R. Klanner、J. Rolph、J. Schwandt，2021，*Simulation of the response of SiPMs; Part I: Without saturation effects*。DOI 10.1016/j.nima.2021.165853 | 定向核对 PDF 1、2、4 页文本，特别是第 4 页式 1–3。将放电序列、相关雪崩、单位面积单胞响应、门积分及电子学卷积分开。未在本轮视觉重核所有公式/图。 | 支持明确快电荷份额及保留积分电荷，候选 AP 用已有原生内核执行；不移植默认参数。论文适用低占据，不提供绝对 PDE、增益或真实电路推导，不能用其名称替本项目完成饱和或阵列验证。 |
| T. Teranishi 等，2021，*Pulse shape analysis of signals from SiPM-based CsI(Tl) detectors for low-energy protons: Saturation correction and particle identification*。DOI 10.1016/j.nima.2020.164967 | PDF 1、2、4、6 页相关文本；第 6 页图 8–9 已看原页。无放大器测量中的电压下降、式 2 / 表 2 的有效恢复参数提示读出负载和恢复不能混为一谈。 | 先以已有 ADC 幅度给出条件电压量级，不擅自拟合反馈。其 5.5 mm 晶体、S13360 单器件及 p/d 能区不同于本项目；图 9 缩放方向的文字疑点保留，不据此换算绝对增益或确认反馈幅度。 |
| W. Skulski、M. Momayezi，2001，*Particle identification in CsI(Tl) using digital pulse shape analysis*。DOI 10.1016/S0168-9002(00)00938-4 | PDF 1、6、10 页相关文本；第 10 页图 7–8 已看原页。相同光输出下比较数字积分/权重，并检查滤波、带宽和时间条件。 | 固定原 3 μs 方法再检查跨 run 分布；未移植 PD/前放条件下的最优权重。其同数据优化不能作为本项目盲测依据；DOI 仍仅由 Bendel 2013 的 PDF 第 7 页参考 11 核对。 |

补充文件链接：

- Corsi 2007：[原文 PDF](</media/sf_works/papers/raw/2007_Corsi_NIMA_Modelling-a-silicon-photomultiplier-(SiPM)-as-a-signal-source-for-optimum-front-end-design.pdf>)；[完整笔记](</media/sf_works/papers/sources/2007_Corsi_NIMA_Modelling-a-silicon-photomultiplier-(SiPM)-as-a-signal-source-for-optimum-front-end-design.md>)。
- Garutti 2021：[原文 PDF](</media/sf_works/papers/raw/2021_Garutti_NIMA_Simulation-of-the-response-of-SiPMs;-Part-I-Without-saturation-effects.pdf>)；[完整笔记](</media/sf_works/papers/sources/2021_Garutti_NIMA_Simulation-of-the-response-of-SiPMs;-Part-I-Without-saturation-effects.md>)。
- Teranishi 和 Skulski 的原文 / 完整笔记链接见上节。

局部协方差的具体分解、共同 Q / 形状条件、双路径候选扫描和描述性容差是本项目分析设计，不声称为上述作者直接给出的结论。本轮[实测与原生模拟结果](../../simulation/RESULTS.md)显示候选尚不能同时解释两源及三个起伏尺度，保留这一不吻合，未将“有文献动机”写成“已获实测验证”。

## 论文贡献定位与本轮落实

本次执行复用此前 `paper-search` 已核验的[投稿准备评估](../../.cache/paper-search/2026-10-28-publication-readiness/assessment.json)，没有重新宣称精读全文、联网补检或改写共享库。该评估另外核验了 **M. Bondí 等，2017，Large-size CsI(Tl) crystal read-out by SiPM for low-energy charged-particles detection，DOI 10.1016/j.nima.2017.06.024** 的 PDF 第 1–3 页文本和第 4 页图 5–6。其 SPE 参考、已知质子能量、31 cm 晶体和放大器与本项目不同；其低能模拟差距也不能证明本项目已校准或一定可发表。

Bondí 原文与笔记：[原文 PDF](</media/sf_works/papers/raw/2017_Bondi_NIMA_Large-size-CsI(Tl)-crystal-read-out-by-SiPM-for-low-energy-charged-particles-detection.pdf>)；[既有完整笔记](</media/sf_works/papers/sources/2017_Bondi_NIMA_Large-size-CsI(Tl)-crystal-read-out-by-SiPM-for-low-energy-charged-particles-detection.md>)。笔记仍为 AI 草稿；前次检索未发现针对本篇具体科学论点的明确用户采用确认。原文哈希及核验范围保存在上述评估中。

| 最近相关工作 | 本项目可据已有数据提出的具体比较 | 本轮落实 |
| --- | --- | --- |
| Skulski 2001 已比较积分、权重和带宽 | 同一队列、四个窗口的配对方法差值，并传播学习误差 | 统一 40795 条队列；3997 条训练；PCA 和 Cauchy 参数逐次重估。3 μs 的 PCA−负平均时间区间含零，不宣称唯一最佳算法 |
| Bondí 2017 已开展 CsI(Tl)–SiPM 与 Geant4 对照 | 检查平均形状之外的起伏、电荷谱，并正确处理重用光子历史 | 512 次共享父衰变重采样；同一 14000 条历史不累计为独立 MC。平均形状接近但短时起伏仍不吻合 |
| Teranishi 2021 说明读出/恢复的实际条件重要 | 在已知电路不足时报告可观测差异及不能唯一反演的参数 | 等 Q、形状与均值扣除后局部差异约 57%；保留器件后缀、拓扑、源距、实际过电压未知，不把条件扫描写成测量 |

具体结果、6 张主图和引用位置已进入[工作稿](MANUSCRIPT_DRAFT.md)。这是局部文献证据支持的研究定位，尚不能替代全面新颖性论证。方法差值的块重采样、Student-t 核心敏感性和局部协方差估计是本项目分析选择，不声称这些论文直接给出本项目的统计结论。
