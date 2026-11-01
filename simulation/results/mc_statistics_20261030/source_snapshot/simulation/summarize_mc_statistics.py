"""Record independent-batch results and limits without promoting or refitting a model."""
import csv
from mc_statistics_common import *


def contains(row, value):
    ci = row['CI95']
    return ci is not None and ci[0] <= value <= ci[1]


def write_table(path, rows0):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows0[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows0)


def main():
    assert not (OUT/'decision.json').exists()
    p = read(OUT/'protocol.json'); e = read(OUT/'analysis.json')
    rows0 = []; batches = []
    for key, blocks in e['groups'].items():
        for B, g in blocks.items():
            for v, r in g['variants'].items():
                row = {'group': key, 'block': B, 'variant': v, 'new_supported_mothers': r['support']['supported_events']}
                for name in METRICS:
                    row[name+'_old'] = r['old_point'][name]; row[name+'_fresh'] = r['point'][name]
                    ci = r['CI95'][name]['CI95']
                    row[name+'_low'] = ci[0] if ci else None; row[name+'_high'] = ci[1] if ci else None
                rows0.append(row)
                for b in g['batches'][v]:
                    batches.append({'group': key, 'block': B, 'variant': v, 'batch': b['batch'],
                                    'estimable': b['support']['estimable'],
                                    **(b['point'] or {name: None for name in METRICS})})
    write_table(OUT/'comparison_table.csv', rows0); write_table(OUT/'batch_table.csv', batches)
    primary = {source: e['groups'][f'run{run}_last20']['128']['variants']['spectrum_CV_fit'] for source, run in SOURCES}
    am = primary['Am241']; name = 'width95_negative_mean_time_ratio'
    ci = am['CI95'][name]['CI95']; oldci = am['fresh_minus_old_CI95'][name]['CI95']
    outside = []
    for source, r in primary.items():
        for coverage in [50, 95]:
            for method in METHODS:
                metric = f'width{coverage}_{method}_ratio'
                if not contains(r['CI95'][metric], 1): outside.append(source+':'+metric)
    conclusion = (f"完成256000个独立Geant4母衰变，固定Q支持内获得Co {primary['Co60']['support']['supported_events']}、"
        f"Am {am['support']['supported_events']}个新母事件。冻结扩展谱在Am晚段（块128）的平均时间95%宽度比，"
        f"从旧样本{am['old_point'][name]:.3f}变为新合并样本{am['point'][name]:.3f}，条件95%区间[{ci[0]:.3f},{ci[1]:.3f}]；"
        f"新减旧的配对区间为[{oldci[0]:.3f},{oldci[1]:.3f}]。"
        f"两源核心及95%宽度的12项检查中，{len(outside)}项条件区间不含1。"
        "这些区间未包含拟合和模型不确定性，也未作多重比较校正或等效性检验；本轮不宣称全面一致，默认配置和正式PID保持原样。")
    findings = []
    for source, run in SOURCES:
        r = primary[source]; point = r['point']; old = r['old_point']
        newcore = '/'.join(f"{point[f'width50_{m}_ratio']:.3f}" for m in METHODS)
        newouter = '/'.join(f"{point[f'width95_{m}_ratio']:.3f}" for m in METHODS)
        findings.append(f"{source}晚段扩展谱：新QDC/平均时间/PCA核心宽度比为{newcore}，95%宽度比为{newouter}；"
            f"均ADC相对L2误差从{100*old['mean_ADC_NRMSE']:.2f}%变为{100*point['mean_ADC_NRMSE']:.2f}%，"
            f"时间中心MC−实测从{old['mean_time_delta_ns']:.2f}变为{point['mean_time_delta_ns']:.2f} ns。所有固定模型、四批结果和两种块长均保留。")
    small = e['groups']['run12_last20']['128']['old_size_diagnostic']['variants']['spectrum_CV_fit'][name]
    span = small['conditional_range95']['CI95']
    findings.append(f"将新Am完整母事件池抽回24000个母事件时，平均时间95%宽度比的512次条件描述范围为[{span[0]:.3f},{span[1]:.3f}]，"
        f"旧点的经验位置为{small['old_point_empirical_rank']:.3f}。抽样共享同一新池且固定实测，不是512次独立输运，也不是校准后的p值或单一因果证明。")
    findings.append(f"Am晚段同一指标的仅MC抽样SD为{am['MC_only_SD'][name]:.4f}，仅实验块抽样SD为{am['data_only_SD'][name]:.4f}。"
        "两项用于判断有限统计来源；比值指标非线性，不能把两项方差机械相加当完整误差预算。")
    mode = am['point']
    findings.append(f"Am晚段三模式内、正交剩余及完整形状方差比分别为{mode['rank3_variance_ratio']:.3f}/{mode['orthogonal_variance_ratio']:.3f}/{mode['full_shape_variance_ratio']:.3f}。"
        "模式和边界沿用上一轮对折实验；即使部分二阶量接近，也不能据此认定尾部分布或微观噪声机制已一致。")
    next_work = ("后续以这轮独立模拟复现结果为固定基准，按原Q箱和采集段列清仍存在的中心、核心/外侧宽度、协方差与F残差。"
        "只针对跨种子仍稳定的残差提出新的、受约束的波形变化机制，同时检查均ADC和完整PID分布；不再单独给某一个PID分数加宽。"
        "本轮新MC已用于诊断，后续若据此改模型，应另外冻结方案并用新种子检查，不能把同批再拟合结果称为独立验证。"
        "若误差主要受已有实验块统计限制，应保留这一边界，不靠无限增加MC或反复调参宣称一致。")
    decision = {'scope': __doc__, 'code_sha256': sha(__file__), 'conclusion_zh': conclusion,
        'findings_zh': findings, 'next_work_zh': next_work,
        'primary_width_intervals_excluding_one': outside,
        'new_independent_mothers': 256000, 'replay_entries': e['replay_entries'],
        'new_supported_mothers': {s: r['support']['supported_events'] for s, r in primary.items()},
        'new_experimental_events': 0, 'default_changed': False, 'official_PID_changed': False,
        'limitations_zh': [
            '实验均已探索，响应按采集段条件化；新种子只提供模拟统计复现，不产生新的实验或跨装置验证。',
            '旧MC曾参与拟合与模型选择；固定参数区间未包含拟合、选择、装置、源距、有效收光/PDE/增益退化等不确定性。',
            '51项指标、多个段/块长/候选并列，无多重比较校正或预设等效性检验；区间包含1不证明等效，不包含1也不能单独识别噪声原因。',
            '本轮固定基准ADC的选择及Q，评价候选不重新筛选；结论限于既定支持区，不是硬件触发效率、全谱性能或实验粒子鉴别效率。',
            'Am源含α/γ/电子，实验主带没有逐事件真值；仿真祖先标签只用于仿真描述，不能转为实验纯度。',
            '小样本诊断重复使用新模拟池，不能计成新增母事件或独立验证批；四批固定规模的条件bootstrap不代表全部模型不确定性。'],
        'inputs_sha256': {relative(OUT/n): sha(OUT/n) for n in ['protocol.json', 'analysis.json', 'transport_execution.json', 'replay_execution.json']},
        'tables_sha256': {relative(OUT/n): sha(OUT/n) for n in ['comparison_table.csv', 'batch_table.csv']}}
    save(OUT/'decision.json', decision)
    print(conclusion)


if __name__ == '__main__': main()
