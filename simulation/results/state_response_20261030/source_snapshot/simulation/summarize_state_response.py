"""Record the joint decision without changing production defaults."""
from state_response_common import *


def main():
    assert not (OUT/'decision.json').exists()
    e=read(OUT/'evaluation.json');a=read(OUT/'alignment.json');fold=read(OUT/'fold_mixture_diagnostic.json')
    last=e['groups']['run12_last20'];base=last['128']['variants']['frozen'];candidate=last['128']['variants']['response_spectrum']
    p0=base['point'];p1=candidate['point'];native_jobs=[]
    for stage in ['initial','state_base','signal_candidates']:
        for source,run in SOURCES:native_jobs.extend(read(OUT/'logs'/f'{stage}_{source}.json')['jobs'].values())
    decision={'scope':__doc__,'code_sha256':sha(__file__),
        'conclusion_zh':f"本轮完成采集段条件校准、原生采样相位和仅前沿定位对照。Am晚段的RC+扩展谱方案（块128）把平均时间差从{p0['mean_time_delta_ns']:.2f} ns降到{p1['mean_time_delta_ns']:.2f} ns，时间IQR比从{p0['IQR_time_ratio']:.3f}改善到{p1['IQR_time_ratio']:.3f}，绝对平均波形NRMSE从{100*p0['mean_ADC_NRMSE']:.2f}%降到{100*p1['mean_ADC_NRMSE']:.2f}%。但QDC/PCA的IQR比变为{p1['IQR_ratio_ratio']:.3f}/{p1['IQR_PCA_ratio']:.3f}，且块64下时间差为{last['64']['variants']['response_spectrum']['point']['mean_time_delta_ns']:.2f} ns。因此是条件改善，尚未达到全PID分布和波形的一致性。",
        'promoted_to_default':False,'front_estimator_adopted':False,
        'new_independent_mothers':0,'unique_original_mothers_with_fixed_waveforms':470,
        'n_supported_original_mothers':456,'new_experimental_events':0,
        'native_jobs':len(native_jobs),'native_selected_replay_entries':sum(j['n_selected_original_mothers'] for j in native_jobs),
        'phase_max_abs_mean_time_change_ns':max(abs(z['mean_time_change_ns']) for ss in a['native_phase'].values() for z in ss),
        'mean_time_conditional_CI95_ns':candidate['CI95']['mean_time_delta_ns']['CI95'],
        'mean_time_paired_change_conditional_CI95_ns':candidate['paired_change_from_frozen']['mean_time_delta_ns']['CI95'],
        'time_IQR_conditional_CI95':candidate['CI95']['IQR_time_ratio']['CI95'],
        'time_IQR_paired_change_conditional_CI95':candidate['paired_change_from_frozen']['IQR_time_ratio']['CI95'],
        'time_SD_ratio_within_response_fold':fold['groups']['run12_last20']['128']['variants']['response_spectrum']['within_fold_SD_ratio'][1],
        'native_fresh_check_deferred_reason_zh':'当前候选在不同块长下的中心仍不稳定，扩展残差谱使QDC/PCA核心宽度偏大；本轮没有确定可推广的最终参数。因此没有把候选送入新母事件的验证阶段，已有重放只计为内部校准证据。',
        'limitations_zh':['后段现在参与新的条件校准，全部实验数据已经探索过，不称盲测。既有光学参数、PCA和Q匹配目标仍共享先前校准；本轮交叉检查仅隔离新增拟合层。',
            '512次配对区间固定本轮参数、模板和母事件折分；训练块重拟合另报RC选择/残差功率稳定性，不代表完整模型选择不确定度。区间覆盖零或一不构成等效性证明。',
            '粗RC网格只作局部响应族探查；112与128 ns之间的切换可能含网格离散误差，不能推断物理电路存在两种时间常数或温漂。',
            '前沿拟合有大量边界命中，原PID定义保持不变。相位结果只约束本模型及已测试的4种原生孔径偏移。',
            'Am源含α、γ和衰变电子；实验主带不作为粒子真值。有效噪声谱不是已确定的CsI本征或SiPM微观机制。'],
        'next_work_zh':'下一轮先在既定96–160 ns范围内细化或连续化RC校准，并检查每折的中心和波形残差，分清粗网格误差与真实段间差异；同时以QDC、平均时间、PCA完整协方差和IQR/SD差异约束噪声分布，不能只按时间带宽增加噪声。只有稳定的联合候选冻结后，再用新Geant4母事件检查MC统计；实验部分仍属于已有数据的条件验证。',
        'inputs_sha256':{relative(OUT/n):sha(OUT/n) for n in ['protocol.json','fit_design.json','alignment.json','mean_calibration.json','signal_calibration.json','evaluation.json','fold_mixture_diagnostic.json']}}
    save(OUT/'decision.json',decision);print(decision['conclusion_zh']);print('Native jobs / selected replays / new mothers:',decision['native_jobs'],decision['native_selected_replay_entries'],0)


if __name__=='__main__':main()
