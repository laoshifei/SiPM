"""Record original-ADC diagnostics without changing simulation or official PID."""
import csv
from waveform_tail_common import *


def main():
    assert not (OUT/'decision.json').exists()
    e=read(OUT/'analysis.json');last=e['groups']['run12_last20']['128'];v=last['variants']['spectrum_CV_fit'];c=v['controls']
    name='width95_negative_mean_time_ratio';x0=c['original']['point'][name];x1=c['recent_and_smooth']['point'][name]
    delta=c['recent_and_smooth']['paired_from_original'][name]['CI95'];mode=v['mode_point']
    tables=[]
    for key,blocks in e['groups'].items():
        for B,r in blocks.items():
            for variant,z in r['variants'].items():
                for control,stat in z['controls'].items():
                    item={'group':key,'raw_block_size':B,'variant':variant,'control':control,**stat['point']}
                    for n in CONTROL_NAMES:
                        ci=stat['CI95'][n]['CI95'];item[n+'_low']=ci[0] if ci else None;item[n+'_high']=ci[1] if ci else None
                    tables.append(item)
    with (OUT/'control_table.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(tables[0]),lineterminator='\n');w.writeheader();w.writerows(tables)
    rows0=[]
    for key,examples in read(OUT/'examples.json')['groups'].items():
        for r in examples:rows0.append({'cohort':key,**{k:r[k] for k in ['event_id','method','quantile','Q_bin','original_Q','actual_z','anchor']},**r['observables']})
    with (OUT/'example_events.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows0[0]),lineterminator='\n');w.writeheader();w.writerows(rows0)
    conclusion=("本轮完成原始ADC极端分位示例、四种基线/取时对照及固定三维波形模式诊断。"
        f"Am晚段扩展谱（块128）的平均时间95%分位宽度比，原定义为{x0:.3f}，近期基线与平滑取时同时应用后为{x1:.3f}；"
        f"配对变化95%区间为[{delta[0]:.3f}, {delta[1]:.3f}]，包含0。该统一修正没有解决晚段外侧宽度差异，正式PID和默认模拟参数保持原样。")
    a=last['feature_audits']['data'];m=last['feature_audits']['spectrum_CV_fit']
    findings=[
        "按预定中间Q箱和2.5/50/97.5%得分分位选出两源共36条示例记录，实验与MC独立选取、原ID及Q全部保存，展示未平滑ADC且无逐事件幅度缩放。部分记录重复使用同一事件，不能计为36个不同母事件。图中相近的波形不证明总体分布一致；未以肉眼将事件判成堆积或坏事件。",
        f"Am晚段联合PID尾部共{a['joint_PID_tail_events']}个原实验事件；大基线差、大锚点差、较大后段正向台阶及极端晚电荷份额分别标记{a['flag_events_in_PID_tail']}个，其中可重叠。{a['PID_tail_events_without_any_flag']}个未触发任一标记；MC联合尾部{m['joint_PID_tail_events']}个中{m['PID_tail_events_without_any_flag']}个未触发。阈值来自对折训练块的99%或1%–99%分位，这仅约束所测试的观察量，不能排除较弱效应。",
        f"Am晚段前三模式捕获的检查块波形方差为实验{100*last['data_modes']['captured_fraction']:.1f}%、MC{100*v['native_modes']['captured_fraction']:.1f}%。三模式总方差、正交剩余方差、全形状方差的MC/实验比分别为{mode['rank3_variance_ratio']:.3f}/{mode['orthogonal_variance_ratio']:.3f}/{mode['full_shape_variance_ratio']:.3f}，各条件区间均包含1。整体二阶量接近，仍不等于高阶分布及PID外侧分位已经一致。",
        f"Am晚段两折三维子空间的最小奇异值为{last['fold_subspace_singular_values'][-1]:.3f}，提示第三个方向较不稳定；前三模式也只覆盖约三分之一方差，不能把它们直接当作三个已识别的物理噪声分量或合成波形模板。",
        "各段对近期基线和平滑定位的响应方向不同；所有模型、两种块长和前/中/后段均保留。数据共享原事件，区间固定响应、模式和诊断阈值，不能据条件检查宣称跨状态转移有效。"
    ]
    next_work=("下一步应冻结现有候选和评价定义，先增加独立种子的Geant4母事件，检查当前仅180个受支持Am母事件导致的分位宽度和模式估计波动。"
        "该轮应明确标成模拟统计复现/诊断，保留原模型与两种固定响应，不把新增模拟当成新实验或最终验证；新批次结果不得再用于调整同一批次的参数。"
        "若独立批次重复出现同方向的核心/外侧差异，再针对稳定波形方向提出受约束的非高斯事件间变化，并同时检验均ADC、分位宽度、完整协方差和F。")
    decision={'scope':__doc__,'code_sha256':sha(__file__),'conclusion_zh':conclusion,'findings_zh':findings,'next_work_zh':next_work,
        'new_independent_mothers':0,'new_native_replays':0,'default_changed':False,'official_PID_changed':False,
        'example_records':len(rows0),'unique_example_events':len({(x['cohort'],x['event_id']) for x in rows0}),
        'Am_last20_time_width95_original':x0,'Am_last20_time_width95_both':x1,'paired_change_CI95':delta,
        'Am_last20_mode_points':mode,'Am_last20_mode_CI95':v['mode_CI95'],
        'limitations_zh':['全部实验已探索；对折仅隔离本轮模式与标记阈值，既有响应、PCA与Q目标含先前校准信息。',
            '512次区间固定响应参数、模式、诊断边界和目标权重，未含这些量的拟合不确定度或多重比较校正；未作等效性检验。',
            '前触发和信号观察量不是因果分类器，后段正台阶不等于堆积，未触发标记不等于无任何异常。',
            '波形模式以每事件Q归一化并含噪声/定位影响，二阶量不能唯一识别闪烁、SiPM或读出机制；模式符号与序号不保证跨折相同物理方向。',
            '所示示例只是固定规则下的有限事件，不代表粒子真值、粒子效率或整体一致性；Am实验主带仍不能当作纯α。'],
        'inputs_sha256':{relative(OUT/n):sha(OUT/n) for n in ['protocol.json','extraction.json','examples.json','analysis.json']},
        'tables_sha256':{relative(OUT/n):sha(OUT/n) for n in ['control_table.csv','example_events.csv']}}
    save(OUT/'decision.json',decision);print(conclusion);print('Example records / distinct within-cohort events:',len(rows0),decision['unique_example_events'])


if __name__=='__main__':main()
