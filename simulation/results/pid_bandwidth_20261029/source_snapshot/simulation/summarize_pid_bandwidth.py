"""Record the partial-closure decision without retuning on validation results."""
from pathlib import Path
from prepare_pid_bandwidth import OUT
from prepare_pid import read
from publication_stats import save,sha


def main():
    assessment=read(OUT/'validation/assessment.json')
    r3=assessment['checks']['run3_last20'];r12=assessment['checks']['run12_last20'];r18=assessment['checks']['run18_last20']
    rows=[]
    for run,r in [(3,r3),(12,r12),(18,r18)]:
        rows.append(f"| {run} | {r['before']['IQR_ratio'][0]:.3f} | {r['after']['IQR_ratio'][0]:.3f} | {r['after']['IQR_ratio'][1]:.3f} | {r['after']['IQR_ratio'][2]:.3f} |")
    summary='''**已改善 Am 的带宽差异；尚未实现所有 PID 方法、所有采集段一致。基线噪声不是唯一来源。**

下表为实验最后20%对应的、按固定Q箱汇总的模拟/实测IQR比值，1为相同宽度：

| run | 旧QDC | 新QDC | 新负平均时间 | 新PCA |
|---|---|---|---|---|
'''+ '\n'.join(rows)+'''

Co 的偏宽仍在；Am run12 的平均时间带仍偏窄。统计区间与逐箱结果见后文。
本轮保留噪声修正为条件候选，未将其宣称为全面验证的探测器响应，也未覆盖默认配置或旧结果。'''
    text='''**结论边界与后续处理：**

- 噪声关闭对照确认，在当前MC的QDC方差分解中，基线电子学项远小于原生信号与信号期残差；这不是直接测得实验中的因果占比。
- 改善来自信号期低频相关功率及积分协方差的约束，不是简单增加基线白噪声。模型能解释Am带宽缺口的一部分，不能据此唯一识别其电子学或材料来源。
- 新旧响应使用相同新光学历史。固定旧ID的QDC宽度变化：Co约0.992倍、run12 Am约1.185倍、run18 Am约1.088倍；因此Co没有实质改善，Am改善也不全由选择迁移造成。
- 平均波形归一化RMSE仍约峰值的0.8%–1.0%；最后20%三个短尺度涨落均值的新MC/实验比为约0.99–1.06。保持平均波形与短时起伏并不自动保证PID带宽完全匹配。
- 同一run12响应下两种源仍可分离，但有限样本零γ误判的单侧95%上限为1.15%，不是零泄漏；这些均为模拟条件性能。
- 不继续用本次验证段修改参数。下一轮应在原first60上联合约束有效光电子统计、读出传递函数和事件形状离散度；统一放大/缩小噪声不能同时处理Co偏宽与Am偏窄。只靠当前波形不能唯一确定光产额、PDE、实际负载和并联单元数。'''
    save(OUT/'decision.json',{'scope':__doc__,'code_sha256':sha(Path(__file__)),
      'assessment_sha256':sha(OUT/'validation/assessment.json'),
      'status':'partial_closure_noise_only_insufficient','promoted_to_default':False,
      'validation_used_for_retuning':False,'summary_zh':summary,'interpretation_zh':text,
      'results':{f'run{run}_last20':{'before_IQR_ratio':r['before']['IQR_ratio'],'after_IQR_ratio':r['after']['IQR_ratio'],
             'after_CI95':r['after']['IQR_ratio_CI95'],'paired_fixed_ID_change':r['fixed_old_ids']} for run,r in [(3,r3),(12,r12),(18,r18)]}})
    print(summary)


if __name__=='__main__':main()
