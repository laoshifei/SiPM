"""Write the bounded conclusion and numeric table without another fit."""
import csv
from pathlib import Path
from prepare_npe_profile import OUT
from prepare_pid import read
from publication_stats import save,sha


def main():
    assert not (OUT/'decision.json').exists(),'Refuse overwrite'
    r=read(OUT/'validation/assessment.json')
    summary='提高有效光电子统计并加入共同的被动电压传输系数后，Co 过宽的问题明显减轻，run18 的三种 PID 带宽比接近1；run12 后段平均时间带仍偏窄。平均 ADC 波形保持约1%的峰值归一化均方根误差。本轮属于有条件改善，尚未实现所有采集段、所有判别量的中心和带宽同时一致。'
    interpretation='带宽包含光子/雪崩统计与信号期相关起伏，不能只靠增加基线白噪声解释。有效光电子规模2倍、共同电压传输系数0.5是训练选出的候选；3倍方案训练分数接近，光产额、PDE、收光和电路传输仍无法单独确定。完整验证使用名义PDE=0.3及2倍光产额；扫描中较高PDE只是数值提案，不代表器件实测值。'
    rows={k:v for k,v in r['checks'].items() if k.endswith('last20')}
    decision={'scope':__doc__,'code_sha256':sha(Path(__file__)),'assessment_sha256':sha(OUT/'validation/assessment.json'),
        'summary_zh':summary,'interpretation_zh':interpretation,
        'outcome':'Partial improvement with independent fresh optical histories; no unique physical calibration or full bandwidth closure.',
        'promoted_to_default':False,'effective_photon_scale':2.,'common_signal_transmission':.5,
        'final_constant_PDE':.3,'final_yield_photons_per_MeV':35180.391309925,
        'independent_new_mothers':32000,'last20':rows,
        'unresolved_zh':[
          'run12 最后20%的平均时间带宽比为0.822，固定参数自举95%区间约[0.662,0.994]；这不是多重比较校正后的显著性结论。',
          'Co 平均时间带中心仍有约0.79个实验IQR的偏移；平均波形NRMSE约1%并不保证所有积分矩都吻合。',
          'run12 实验基线RMS中位数从训练段3.17 ADC上升到最后段5.68 ADC，当前训练态模拟约3.17 ADC。状态漂移已被观测到，但对PID残差的因果贡献尚未单独确定。',
          '新MC样本的逐Q箱区间仍较宽；幅度和波形一致性只在当前共同Q区间及预先支持的箱内评价。',
          '信号相关残差仍是经验采集状态，不能唯一认定为SiPM噪声、发光本征涨落或事件形状变化。'],
        'next_steps_zh':[
          '保留这套候选作为后续基线，用已有触发前样本建立按时间变化的噪声状态模型，并把用于估计噪声与用于检查PID的事件分开。',
          '对Co平均时间中心偏移检查训练段波形尾部和对齐误差；只在训练段修改共同响应或发光形状，再冻结新的检查。',
          '论文同时报告条件参数退化、带宽区间与未闭合残差；不能把MC的高AUC或零误判计数写成实测效率。'],
        'notes':'All fitting used frozen first60; later experimental checks had been explored previously. Fixed-parameter intervals omit calibration/model uncertainty. Previous noise-only and new transport use different native seeds.'}
    save(OUT/'decision.json',decision)
    with (OUT/'validation_table.csv').open('w',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['experimental_group','method','previous_noise_only_IQR_ratio','new_IQR_ratio','CI95_low','CI95_high',
            'mean_ADC_NRMSE','mean_ADC_peak_ratio','mean_center_shift_data_IQR'])
        for key,v in r['checks'].items():
            for j,m in enumerate(r['methods']):
                writer.writerow([key,m,v['previous_noise_only_IQR_ratio'][j],v['IQR_ratio'][j],
                    *v['IQR_ratio_CI95'][j]['estimate95'],v['mean_ADC_normalized_RMSE'],v['mean_ADC_peak_ratio'],v['median_shift_in_data_IQR'][j]])
    print(summary)


if __name__=='__main__':main()
