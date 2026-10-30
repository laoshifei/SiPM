"""Record continuous-RC and noise-distribution findings without promoting defaults."""
import csv
from collections import Counter
from rc_refinement_common import *


def main():
    assert not (OUT/'decision.json').exists()
    evaluation=read(OUT/'evaluation.json');selection=read(OUT/'noise_selection.json')
    last=evaluation['groups']['run12_last20'];co=evaluation['groups']['run3_last20']['128']['variants']
    old=last['128']['variants']['coarse_spectrum'];new=last['128']['variants']['spectrum_CV_fit']
    a=old['point'];b=new['point'];c=co['spectrum_CV_fit']['point'];c0=co['coarse_spectrum']['point']
    gaps={v:abs(last['128']['variants'][v]['point']['mean_time_delta_ns']-
        last['64']['variants'][v]['point']['mean_time_delta_ns']) for v in evaluation['variants']}
    cv_counts={source:Counter() for source,run in SOURCES}
    for source,run in SOURCES:
        for key,r in selection['groups'].items():
            if not key.startswith(f'run{run}_'):continue
            for family,z in r['families'].items():
                cv_counts[source][str(z['chosen_CV'])]+=1
                assert z['chosen_CV'] in [0.,z['inherited_CV']]
    logs=[]
    for stage in ['quiet_grid','mean_round0','mean_round1','mean_round2','base','noise_grid']:
        for source,run in SOURCES:logs.extend(read(OUT/'logs'/f'{stage}_{source}.json')['jobs'].values())
    metrics=evaluation['metrics'];flat=[]
    for group,blocks in evaluation['groups'].items():
        for B,row in blocks.items():
            for variant,r in row['variants'].items():
                item={'group':group,'raw_block_size':B,'variant':variant,**r['point']}
                for name in metrics:
                    ci=r['CI95'][name]['CI95'];item[name+'_low']=ci[0];item[name+'_high']=ci[1]
                flat.append(item)
    with (OUT/'validation_table.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    conclusion=(f"本轮完成连续RC、原生含噪声响应校正和噪声分布联合约束。Am晚段扩展谱方案（块128）的MC−实测平均时间差由上一轮{a['mean_time_delta_ns']:.2f} ns变为{b['mean_time_delta_ns']:.2f} ns；"
        f"时间IQR比为{b['IQR_time_ratio']:.3f}，QDC/PCA IQR比为{b['IQR_ratio_ratio']:.3f}/{b['IQR_PCA_ratio']:.3f}，绝对平均ADC波形NRMSE为{100*b['mean_ADC_NRMSE']:.2f}%。"
        "主要进展是时间中心对分块的敏感性减小；三个PID的联合分布仍未全面吻合，不能称为已解决带宽差异。")
    block=(f"Am晚段块64/128的时间中心差距由粗RC扩展谱的{gaps['coarse_spectrum']:.2f} ns缩小为连续RC扩展谱的{gaps['spectrum_CV_fit']:.2f} ns；"
        f"仅更新均值响应时为{gaps['mean_only']:.2f} ns。新扩展谱的块64/128中心分别为{last['64']['variants']['spectrum_CV_fit']['point']['mean_time_delta_ns']:.2f}/{b['mean_time_delta_ns']:.2f} ns。"
        "这些是内部点估计的稳定性比较，不能把全部变化唯一归因于网格离散，也不能据此识别电路真实RC或温漂。")
    time_ci=new['CI95']['mean_time_delta_ns']['CI95'];width_ci=new['paired_from_coarse']['IQR_time_ratio']['CI95']
    findings=[
        f"Am晚段平均时间差的固定拟合条件95%区间为[{time_ci[0]:.2f}, {time_ci[1]:.2f}] ns。时间IQR比相对粗RC的配对变化区间为[{width_ci[0]:.3f}, {width_ci[1]:.3f}]，包含0；不能声称带宽有统计上明确的改善或达到等效。",
        f"Am晚段完整PID协方差归一化误差点估计从{a['covariance_normalized_RMSE']:.3f}变为{b['covariance_normalized_RMSE']:.3f}，但配对变化区间包含0。QDC/PCA的核心宽度仍偏大；三项F的事件间SD比为{b['F50_SD_ratio']:.2f}/{b['F102_SD_ratio']:.2f}/{b['F202_SD_ratio']:.2f}。",
        "在本轮两个谱族、固定功率拟合和48个条件/谱族组合中，Co全部24项保留原CV；Am19项选择CV=0、5项保留原小CV。较大的事件方差CV均未被训练目标选中。因此本谱族内继续增加这类噪声波动缺乏支持；这不排除其它波形相关机制。",
        f"Co晚段扩展谱的时间差从{c0['mean_time_delta_ns']:.2f}变为{c['mean_time_delta_ns']:.2f} ns，绝对均ADC NRMSE从{100*c0['mean_ADC_NRMSE']:.2f}%变为{100*c['mean_ADC_NRMSE']:.2f}%。中心更接近零这一点没有改善，多目标拟合存在权衡，不能只报告Am的改善。",
        "原生缓存是计算优化，216个配对事件对照中ADC、无噪声信号和模拟电压逐点等于旧程序。研究重放反复使用470个原母事件（支持Q箱内456个），重复条目不增加独立MC统计。"
    ]
    next_work=("下一步固定本轮校准，不再按单一带宽追加全局噪声。先在原Q箱内比较三个PID的中心核心、尾部概率和相关方向，关联F50/102/202、前触发状态与40 ns波形残差；"
        "实测分组规则只从训练块确定，保留原事件全集，尾部检查仅用于诊断。利用现有Geant4祖先标签核查源混合/能损拓扑对模拟尾部的贡献，但不把Am实验主带标成纯α。"
        "只有形成同时约束均波形、IQR、完整协方差和起伏分布的稳定候选后，才冻结参数并运行新种子Geant4母事件；已有实验始终作为内部条件检查。")
    decision={'scope':__doc__,'code_sha256':sha(__file__),'conclusion_zh':conclusion,
        'block_stability_zh':block,'findings_zh':findings,'next_work_zh':next_work,
        'native_jobs':len(logs),'native_replay_entries':sum(r['n_original_mothers'] for r in logs),
        'unique_original_mothers':470,'supported_original_mothers':456,'new_independent_mothers':0,'new_experimental_events':0,
        'engineering_paired_event_cases':read(OUT/'cache_verification.json')['paired_event_cases'],
        'promoted_to_default':False,'final_candidate_selected':False,'selected_CV_counts':{s:dict(v) for s,v in cv_counts.items()},
        'Am_last20_time_block64_vs128_gap_ns':gaps,
        'Am_last20_extended_spectrum':{k:new[k] for k in ['point','CI95','paired_from_coarse','control_comparator','paired_from_control']},
        'native_fresh_check_deferred_reason_zh':'当前不同噪声谱族在时间宽度与QDC/PCA核心之间仍有权衡，且起伏分布尚未同时吻合；未形成最终联合候选，因此本轮没有启动新母事件验证或替换默认参数。',
        'limitations_zh':['所有实验段已经探索；对折只隔离新增拟合层，既有光学、PCA与Q目标共享先前校准信息。',
            '512次配对区间固定拟合、谱族/CV选择、Q目标及母事件折分；各128次训练稳定性检查不是完整嵌套参数或模型不确定度。',
            '重放使用相同原MC母事件进行拟合和检查，不是独立MC验证；本轮事件CV只表示经验残差方差乘子的离散度。',
            '源分组不是实验粒子真值，Am含α、γ和衰变电子；有效RC与噪声谱不能唯一归因于某种物理机制。',
            '中心/宽度点估计接近目标及区间覆盖零或一，都不构成等效性证明；本轮未测量实验PID效率。'],
        'inputs_sha256':{relative(OUT/n):sha(OUT/n) for n in ['protocol.json','mean_calibration.json','noise_selection.json','evaluation.json','cache_verification.json']},
        'validation_table_sha256':sha(OUT/'validation_table.csv')}
    assert decision['native_jobs']==490 and decision['native_replay_entries']==115150
    save(OUT/'decision.json',decision);print(conclusion);print(block)
    print('Native jobs / entries / unique mothers:',len(logs),decision['native_replay_entries'],470)


if __name__=='__main__':main()
