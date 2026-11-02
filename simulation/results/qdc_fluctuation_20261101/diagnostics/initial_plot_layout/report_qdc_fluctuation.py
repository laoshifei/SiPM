"""Plot and summarize fixed-event QDC/fluctuation diagnostics without model promotion."""
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from qdc_fluctuation_common import *

COLORS = ['#777777', '#25858b', '#dc7627']
LABELS = ['Frozen', 'RC + inherited', 'RC + extended']
FIGURES = ['fluctuation_term_controls', 'fluctuation_by_charge', 'fluctuation_tail_overlap',
           'fluctuation_native_parts', 'fluctuation_segment_controls']


def finish(fig, name, note, top=.88, bottom=.16):
    fig.tight_layout(rect=[0, bottom, 1, top])
    fig.text(.5, .02, note, ha='center', va='bottom', fontsize=9)
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(OUT/'figures'/f'{name}.{ext}', dpi=180)
    plt.close(fig)


def point(ax, x, row, name, color, label=None, marker='o'):
    y = row['point'][name]; ci = row['CI95'][name]['CI95']
    ax.plot(x, y, marker=marker, color=color, ls='none', label=label, ms=5)
    if ci is not None:
        ax.plot([x, x], ci, color=color, lw=1)
        for endpoint in ci: ax.plot([x-.025, x+.025], [endpoint, endpoint], color=color, lw=1)


def plots(e):
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.grid': True, 'grid.alpha': .14})
    groups = [e['groups'][f'run{run}_last20']['128'] for _, run in SOURCES]
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    controls = ['F202_SD_ratio', 'F202_fixed_denominator_SD_ratio', 'F202_within_cell_SD_ratio']
    for j, ((source, _), group) in enumerate(zip(SOURCES, groups)):
        ax[0, j].set_title(source)
        for iv, v in enumerate(VARIANTS):
            row = group['variants'][v]
            for k, metric in enumerate(controls): point(ax[0, j], k+(iv-1)*.17, row, metric, COLORS[iv], LABELS[iv] if k == 0 else None)
            for k, term in enumerate(TERMS):
                point(ax[1, j], k+(iv-1)*.17, row, f'F202_{term}_excess_over_data_variance', COLORS[iv])
        ax[0, j].set_xticks(range(3), ['Original F', 'Fixed denominator', 'Within Q-half/fold'])
        ax[0, j].axhline(1, color='black', ls='--', lw=1); ax[0, j].set_ylabel('MC / measured SD')
        ax[1, j].set_xticks(range(3), ['Post-window', 'Pre-window', 'Normalization'])
        ax[1, j].axhline(0, color='black', ls='--', lw=1); ax[1, j].set_ylabel('Excess covariance contribution / data Var(F202)')
    handles, labels = ax[0, 0].get_legend_handles_labels(); fig.legend(handles, labels, loc='upper center', ncol=3)
    finish(fig, FIGURES[0], 'Last20 / block128. Algebraic controls; original ADC and PID retained. Contributions can be negative and sum to the excess F202 variance.')
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    for j, ((source, _), group) in enumerate(zip(SOURCES, groups)):
        bins = list(group['variants']['spectrum_CV_fit']['per_Q'])
        for rown, name in enumerate(['width95_QDC_ratio', 'F202_SD_ratio']):
            for iv, v in enumerate(VARIANTS):
                for k, b in enumerate(bins):
                    point(ax[rown, j], k+(iv-1)*.17, group['variants'][v]['per_Q'][b], name, COLORS[iv], LABELS[iv] if k == 0 else None)
            ax[rown, j].set_xticks(range(len(bins)), bins); ax[rown, j].set_xlabel('Original Q bin')
            ax[rown, j].axhline(1, color='black', ls='--', lw=1)
            ax[rown, j].set_ylabel('QDC central95% width ratio' if rown == 0 else 'F202 event SD ratio')
        ax[0, j].set_title(source)
    handles, labels = ax[0, 0].get_legend_handles_labels(); fig.legend(handles, labels, loc='upper center', ncol=3)
    finish(fig, FIGURES[1], 'Last20 / block128. Same original Q-half targets in every bin. Conditional intervals; no bin removed and no multiplicity correction.')
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    cat_colors = ['#b9c1cb', '#4478ad', '#e7ac54', '#a94a50']
    cat_labels = ['Neither', 'QDC only', 'F202 only', 'Both']
    for j, ((source, _), group) in enumerate(zip(SOURCES, groups)):
        summaries = [group['data']]+[group['variants'][v]['summary'] for v in VARIANTS]
        mass = np.array([s['category_mass'] for s in summaries]); low = np.zeros(4)
        for k in range(4):
            ax[0, j].bar(range(4), 100*mass[:, k], bottom=low, color=cat_colors[k], label=cat_labels[k])
            low += 100*mass[:, k]
        ax[0, j].set_title(source); ax[0, j].set_ylabel('Q-weighted event mass (%)')
        ax[0, j].set_xticks(range(4), ['Measured', *LABELS], rotation=18)
        for iv, v in enumerate(VARIANTS):
            for k, metric in enumerate(['F202_variance_in_QDC_tail_delta_pp', 'QDC_variance_in_F202_outer_delta_pp']):
                point(ax[1, j], k+(iv-1)*.17, group['variants'][v], metric, COLORS[iv], LABELS[iv] if k == 0 else None)
        ax[1, j].set_xticks(range(2), ['F variance in QDC tail', 'QDC variance in F outer'])
        ax[1, j].axhline(0, color='black', ls='--', lw=1); ax[1, j].set_ylabel('MC minus measured share (percentage points)')
    handles, labels = ax[0, 0].get_legend_handles_labels(); fig.legend(handles, labels, loc='upper center', ncol=4)
    ax[1, 0].legend(fontsize=8)
    finish(fig, FIGURES[2], 'Last20 / block128. QDC abs(z)>2 uses prior references; F202 outer uses opposite-fold measured5/95% thresholds. Shared samples imply association, not causation.')
    fig, ax = plt.subplots(2, 3, figsize=(13, 8))
    part_colors = ['#4a79a8', '#d58c36', '#7b5a9c']
    for rown, ((source, _), group) in enumerate(zip(SOURCES, groups)):
        dvar = np.diag(group['data']['cov'])[1:]
        for k, scale in enumerate(SCALES):
            parts = np.array([np.asarray(group['variants'][v]['summary']['native_terms_cov'])[k].sum(1)/dvar[k] for v in VARIANTS])
            pos = np.zeros(3); neg = np.zeros(3)
            for t, term in enumerate(NATIVE_TERMS):
                value = parts[:, t]; base = np.where(value >= 0, pos, neg)
                ax[rown, k].bar(range(3), value, bottom=base, color=part_colors[t], label=term.replace('_', ' '))
                pos += np.maximum(value, 0); neg += np.minimum(value, 0)
            ax[rown, k].plot(range(3), parts.sum(1), 'k_', ms=13)
            ax[rown, k].axhline(1, color='black', ls='--', lw=1)
            ax[rown, k].set_xticks(range(3), LABELS, rotation=18); ax[rown, k].set_title(f'{source}: F{scale}')
            ax[rown, k].set_ylabel('MC covariance contribution / data variance')
    handles, labels = ax[0, 0].get_legend_handles_labels(); fig.legend(handles, labels, loc='upper center', ncol=3)
    finish(fig, FIGURES[3], 'MC-only latent decomposition. Native signal includes photon/avalanche fluctuations; additive residual combines noise and quantization. Not an experimental noise measurement.')
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    for j, (source, run) in enumerate(SOURCES):
        for rown, metric in enumerate(['F202_SD_ratio', 'width95_QDC_ratio']):
            for ib, B in enumerate([64, 128]):
                color = ['#4478ad', '#dc7627'][ib]
                for k, seg in enumerate(SEGMENTS):
                    r = e['groups'][f'run{run}_{seg}'][str(B)]['variants']['spectrum_CV_fit']
                    point(ax[rown, j], k+(ib-.5)*.18, r, metric, color, f'Block{B}' if k == 0 else None)
            ax[rown, j].set_xticks(range(3), SEGMENTS); ax[rown, j].axhline(1, color='black', ls='--', lw=1)
            ax[rown, j].set_ylabel('F202 event SD ratio' if rown == 0 else 'QDC central95% width ratio')
        ax[0, j].set_title(source)
    handles, labels = ax[0, 0].get_legend_handles_labels(); fig.legend(handles, labels, loc='upper center', ncol=2)
    finish(fig, FIGURES[4], 'Frozen RC + extended-spectrum response in each acquisition condition. Existing segments were explored; these are internal comparisons, not blind validation.')


def table(path, rows0):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows0[0]), lineterminator='\n'); w.writeheader(); w.writerows(rows0)


def numeric(row):
    result = {}
    for n in METRICS:
        result[n] = row['point'][n]; ci = row['CI95'][n]['CI95']
        result[n+'_low'] = ci[0] if ci else None; result[n+'_high'] = ci[1] if ci else None
    return result


def report(e):
    comparisons = []; charge = []; batches = []
    for group, blocks in e['groups'].items():
        for B, g in blocks.items():
            for v, r in g['variants'].items():
                base = {'group': group, 'block': B, 'variant': v}
                comparisons.append({**base, **numeric(r)})
                for b, q in r['per_Q'].items(): charge.append({**base, 'Q_bin': b, **numeric(q)})
                for b in g['batches'][v]: batches.append({**base, 'batch': b['batch'], **b['point']})
    table(OUT/'comparison_table.csv', comparisons); table(OUT/'per_Q_table.csv', charge); table(OUT/'batch_table.csv', batches)
    findings = []
    for source, run in SOURCES:
        g = e['groups'][f'run{run}_last20']['128']; r = g['variants']['spectrum_CV_fit']; pt = r['point']
        contribution = [pt[f'F202_{t}_excess_over_data_variance'] for t in TERMS]
        findings.append(f"{source}晚段块128：F202事件SD比{pt['F202_SD_ratio']:.3f}，固定分母诊断为{pt['F202_fixed_denominator_SD_ratio']:.3f}，"
            f"只看Q半箱/响应折内部的SD比为{pt['F202_within_cell_SD_ratio']:.3f}。这两种计算不是对原波形或PID的校正。")
        findings.append(f"{source}的F202方差超额/实验方差为{pt['F202_SD_ratio']**2-1:.3f}；信号窗、前触发窗、归一化三项的协方差贡献差分别为"
            +'/'.join(f'{x:.3f}' for x in contribution)+"，严格相加得到该超额。贡献可以为负，不能解释为三种独立噪声强度。")
        d = g['data']; m = r['summary']; dv = vectors({k: np.asarray(x) if isinstance(x, list) else x for k, x in d.items() if k != 'per_Q'})
        mv = vectors({k: np.asarray(x) if isinstance(x, list) else x for k, x in m.items() if k != 'per_Q'})
        findings.append(f"{source}的QDC尾部承载实验/模拟F202方差{100*dv['F202_variance_in_QDC_tail']:.1f}%/{100*mv['F202_variance_in_QDC_tail']:.1f}%；"
            f"F202外侧承载实验/模拟QDC方差{100*dv['QDC_variance_in_F202_outer']:.1f}%/{100*mv['QDC_variance_in_F202_outer']:.1f}%。"
            "边界只由对折实验定义，所有事件保留；关联不是因果解释。")
        nc = np.asarray(m['native_terms_cov'])[2].sum(1)/m['cov'][3][3]
        findings.append(f"{source}模型内部F202事件方差的信号/附加残差/交叉协方差贡献为"
            +'/'.join(f'{100*x:.1f}%' for x in nc)+"。附加残差合并多种模型项；这些比例不是实验噪声测量。")
    decision = {'scope': __doc__, 'code_sha256': sha(__file__), 'findings_zh': findings,
        'conclusion_zh': '完成固定事件的F方差分解、QDC尾部关联、逐Q箱和采集段检查；未调参数、未删事件，也未增加独立模拟或实测样本。具体点值和条件区间见全部表格。',
        'next_work_zh': '优先处理在分母固定、Q半箱/响应折分解及多个种子后仍存在的信号窗起伏分布残差。任何新响应机制需同时约束均ADC、PID中心/协方差/分位宽度及F均值/离散度，并重新冻结新种子检查。波形高阶统计及噪声时间相关结构是待检验方向，不把本轮代数贡献当作已识别物理机制。',
        'limitations_zh': ['全部实验和本轮MC已用于诊断；不存在新增盲实验验证。',
            '区间固定响应参数、Q目标、分折、尾部和分母参考，没有包含拟合/模型不确定性，也未作多重比较校正或等效性检验。',
            'F、QDC和归一化量使用同一波形，相关、尾部重合和带符号协方差贡献不能单独识别噪声因果。',
            '5%/95%是训练折分位，评价折外侧质量不必等于10%；尤其联合尾部实测事件很少，需保留原计数与条件区间。',
            'MC信号/残差分解仅限已保存的模型输出，无法把相同潜变量标签直接赋给实验；Am实验主带不作为纯α真值。',
            'Q半箱与折间均值项含有限样本均值噪声；去除该统计项的诊断不等于已校正真实漂移。'],
        'new_independent_mothers': 0, 'new_native_replays': 0, 'default_changed': False, 'official_PID_changed': False,
        'inputs_sha256': {relative(OUT/n): sha(OUT/n) for n in ['protocol.json', 'analysis.json']},
        'tables_sha256': {relative(OUT/n): sha(OUT/n) for n in ['comparison_table.csv', 'per_Q_table.csv', 'batch_table.csv']}}
    save(OUT/'decision.json', decision)


def main():
    assert not (OUT/'decision.json').exists() and not (OUT/'figure_record.json').exists()
    e = read(OUT/'analysis.json'); report(e); plots(e)
    save(OUT/'figure_record.json', {'code_sha256': sha(__file__), 'figures': FIGURES,
        'inputs_sha256': {relative(OUT/n): sha(OUT/n) for n in ['protocol.json', 'analysis.json']}})
    print('Saved5 figures and36/162/144-row comparison tables.', flush=True)


if __name__ == '__main__': main()
