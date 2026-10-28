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
