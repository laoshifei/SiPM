"""Report covariance closure and its remainder, retaining sparse and negative estimates."""
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from gaussian_closure_common import *

FIGURES = ['gaussian_excess_decomposition', 'gaussian_normalization_controls',
           'gaussian_charge_controls', 'gaussian_temporal_covariance', 'gaussian_segment_controls']
COLORS = ['#6f7780', '#25858b', '#d9762b']
LABELS = ['Frozen', 'RC + inherited', 'RC + extended']


def record(group, v='spectrum_CV_fit', cond='Q_half', norm='event_mu', scale=202, b=None):
    c = group['cohorts'][v]
    r = c['contrasts'] if b is None else c['per_Q_contrasts'][str(b)]
    return r[cond][norm][str(scale)]


def point(ax, x, r, name, color, label=None, marker='o'):
    value = r['point'][name]; ci = r['CI95'][name]['CI95']
    if value is None: return
    ax.plot(x, value, marker=marker, color=color, ms=5, ls='none', label=label)
    if ci is not None:
        ax.plot([x, x], ci, color=color, lw=1)
        for y in ci: ax.plot([x-.025, x+.025], [y, y], color=color, lw=1)


def finish(fig, name, note):
    fig.tight_layout(rect=[0, .14, 1, .91])
    fig.text(.5, .025, note, ha='center', va='bottom', fontsize=9)
    for ext in ['png', 'pdf', 'svg']: fig.savefig(OUT/'figures'/f'{name}.{ext}', dpi=180)
    plt.close(fig)


def plots(e):
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.grid': True, 'grid.alpha': .14})
    groups = [e['groups'][f'run{run}_last20']['128'] for _, run in SOURCES]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    metrics = ['observed_excess_over_data', 'gaussian_excess_over_data', 'remainder_excess_over_data']
    ml = ['Observed excess', 'Gaussian closure excess', 'Remainder excess']
    mc = ['#303840', '#2a7a99', '#b56335']
    for row, ((source, _), group) in enumerate(zip(SOURCES, groups)):
        for k, scale in enumerate(SCALES):
            ax = axes[row, k]
            for iv, v in enumerate(VARIANTS):
                r = record(group, v, scale=scale)
                for j, name in enumerate(metrics): point(ax, iv+(j-1)*.18, r, name, mc[j], ml[j] if iv == 0 else None)
            ax.set_title(f'{source}: F{scale}'); ax.set_xticks(range(3), LABELS, rotation=18)
            ax.axhline(0, color='black', ls='--', lw=1); ax.set_ylabel('MC - data contribution / data Var(F)')
    h, l = axes[0, 0].get_legend_handles_labels(); fig.legend(h, l, loc='upper center', ncol=3)
    finish(fig, FIGURES[0], 'Last20 / block128 / Q-half cells / original denominator. Observed = Gaussian closure + remainder.\nConditional distinct-ID weighted-U ranges; no response parameters changed.')
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for j, ((source, _), group) in enumerate(zip(SOURCES, groups)):
        for im, norm in enumerate(NORMS):
            r = record(group, norm=norm)
            point(axes[0, j], im-.08, r, 'observed_SD_ratio', '#303840', 'Observed' if im == 0 else None)
            point(axes[0, j], im+.08, r, 'gaussian_SD_ratio', '#2a7a99', 'Gaussian closure' if im == 0 else None)
            for ic, cond in enumerate(CONDITIONS):
                r = record(group, cond=cond, norm=norm)
                point(axes[1, j], im+(ic-.5)*.18, r, 'remainder_excess_over_data', ['#b56335','#6b63a5'][ic],
                      ['Q half', 'Q half + fold'][ic] if im == 0 else None)
                if r['point']['remainder_excess_over_data'] is None:
                    axes[1, j].annotate('fold split:\nnot estimable', (im+.09, 0), xytext=(im+.09, .08), ha='center', fontsize=8, color='#6b63a5')
        axes[0, j].set_title(source); axes[0, j].axhline(1, color='black', ls='--', lw=1)
        axes[0, j].set_ylabel('MC / measured SD'); axes[1, j].axhline(0, color='black', ls='--', lw=1)
        axes[1, j].set_ylabel('Remainder excess / data Var(F202)')
        for ax in axes[:, j]:
            ax.set_xticks([0, 1], ['Event denominator', 'Opposite-fold fixed denominator'])
            ax.set_xlim(-.3, 1.3)
    h, l = axes[0, 0].get_legend_handles_labels(); fig.legend(h, l, loc='upper center', ncol=2)
    axes[1, 1].legend(fontsize=9)
    finish(fig, FIGURES[1], 'F202 / last20 / block128 / RC + extended. Original PID retained; fixed denominator is diagnostic only.\nCo has a 3-event Q-half/fold cell: its full fold-conditioned closure is not estimable, with no bin dropped.')
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8))
    for j, ((source, _), group) in enumerate(zip(SOURCES, groups)):
        bins = list(group['cohorts']['data']['per_Q'])
        for i, b in enumerate(bins):
            r = record(group, b=b)
            for k, name in enumerate(metrics): point(axes[j], i+(k-1)*.18, r, name, mc[k], ml[k] if i == 0 else None)
        axes[j].set_title(source); axes[j].set_xticks(range(len(bins)), bins); axes[j].set_xlabel('Original Q bin')
        axes[j].axhline(0, color='black', ls='--', lw=1); axes[j].set_ylabel('Contribution / data within-bin Var(F202)')
    h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, loc='upper center', ncol=3)
    finish(fig, FIGURES[2], 'Last20 / block128 / RC + extended / original denominator / Q-half cells. Every supported Q bin retained.\nRanges use available distinct-ID draws; small-cell availability is recorded in per_Q_table.csv.')
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for j, ((source, _), group) in enumerate(zip(SOURCES, groups)):
        for im, norm in enumerate(NORMS):
            ax = axes[im, j]
            for v, color, label in zip(['data', *VARIANTS], ['#202b35', *COLORS], ['Measured', *LABELS]):
                curve = np.asarray(group['cohorts'][v]['lag_correlation'])[im, 2]
                ax.plot(2*np.array(LAGS), curve, 'o-', ms=4, color=color, label=label)
            ax.set_xscale('symlog', linthresh=2); ax.set_xticks([0, 2, 10, 50, 200, 400], ['0', '2', '10', '50', '200', '400'])
            ax.set_xlabel('Sample lag (ns)'); ax.set_ylabel('Post-window covariance / zero lag')
            ax.set_title(f'{source}: '+['event denominator','fixed denominator'][im]); ax.axhline(0, color='black', lw=.8)
    h, l = axes[0, 0].get_legend_handles_labels(); fig.legend(h, l, loc='upper center', ncol=4)
    finish(fig, FIGURES[3], 'F202 / last20 / block128. Across-event post-window covariance within Q halves; fixed half/Q masses.\nDescriptive lag averages, not a stationary noise spectrum or an intrinsic SiPM correlation time.')
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for j, (source, run) in enumerate(SOURCES):
        for k, seg in enumerate(SEGMENTS):
            for ib, B in enumerate([64, 128]):
                r = record(e['groups'][f'run{run}_{seg}'][str(B)])
                for row, metric in enumerate(['gaussian_excess_over_data', 'remainder_excess_over_data']):
                    point(axes[row, j], k+(ib-.5)*.2, r, metric, ['#2a7a99','#b56335'][ib], f'Block{B}' if k == 0 else None)
        axes[0, j].set_title(source)
        for row, label in enumerate(['Gaussian closure excess / data variance', 'Remainder excess / data variance']):
            axes[row, j].set_ylabel(label); axes[row, j].set_xticks(range(3), SEGMENTS); axes[row, j].axhline(0, color='black', ls='--', lw=1)
    h, l = axes[0, 0].get_legend_handles_labels(); fig.legend(h, l, loc='upper center', ncol=2)
    finish(fig, FIGURES[4], 'F202 / RC + extended / Q-half cells / original denominator. All segments are previously explored.\nDistinct-ID block-bootstrap ranges are conditional diagnostics; no simultaneous confidence or equivalence claim.')


def table(path, values):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(values[0]), lineterminator='\n'); w.writeheader(); w.writerows(values)


def numeric(r):
    result = dict(r['point'])
    for n in METRICS:
        ci = r['CI95'][n]
        result.update({n+'_lo': None if ci['CI95'] is None else ci['CI95'][0],
                       n+'_hi': None if ci['CI95'] is None else ci['CI95'][1], n+'_valid': ci['valid_draws']})
    return result


def main():
    assert not (OUT/'record.json').exists() and not (OUT/'verification.json').exists()
    if (OUT/'figure_record.json').exists():
        for name, value in read(OUT/'figure_record.json')['inputs_sha256'].items(): assert sha(PROJECT/name) == value
    e = read(OUT/'analysis.json'); p = read(OUT/'protocol.json'); summary = []; perq = []; cells = []
    for group, blocks in e['groups'].items():
        for B, g in blocks.items():
            for v, cohort in g['cohorts'].items():
                for c in cohort['cells']:
                    cells.append({'group': group, 'block': B, 'variant': v, 'Q_bin': c['Q_bin'], 'half': c['half'],
                        'events': c['events'], 'fold0': c['fold_counts'][0], 'fold1': c['fold_counts'][1], 'target_mass': c['target']})
                if v == 'data': continue
                for cond in CONDITIONS:
                    for norm in NORMS:
                        for scale in SCALES:
                            base = {'group': group, 'block': B, 'variant': v, 'conditioning': cond, 'normalization': norm, 'scale_ns': scale}
                            summary.append({**base, **numeric(record(g, v, cond, norm, scale))})
                            for b in p['support_bins'][g['source']]: perq.append({**base, 'Q_bin': b, **numeric(record(g, v, cond, norm, scale, b))})
    for name, tab in [('comparison_table.csv', summary), ('per_Q_table.csv', perq), ('cell_counts.csv', cells)]: table(OUT/name, tab)
    plots(e)
    def fmt(r, name):
        x = r['point'][name]; ci = r['CI95'][name]
        return '不可估计' if x is None else f"{x:.3f}，区间"+('不可估计' if ci['CI95'] is None else f"[{ci['CI95'][0]:.3f}, {ci['CI95'][1]:.3f}]")+f"（{ci['valid_draws']}/512次有效）"
    findings = []
    for source, run in SOURCES:
        group = e['groups'][f'run{run}_last20']['128']; r = record(group); fixed = record(group, norm=NORMS[1])
        findings.append(f"**{source} 晚段、块128、扩展谱、原分母：**直接F202 SD比为"+fmt(r,'observed_SD_ratio')+
            '；Gaussian闭合的SD比为'+fmt(r,'gaussian_SD_ratio')+'。以实测F202方差为单位，二阶闭合超额为'+fmt(r,'gaussian_excess_over_data')+
            '，余项超额为'+fmt(r,'remainder_excess_over_data')+'；二者相加严格等于直接方差超额。')
        findings.append(f"**{source} 固定分母对照：**余项超额为"+fmt(fixed,'remainder_excess_over_data')+
            '；这里的分母来自既有对折实测参考，不用于修改ADC或正式PID。')
    primary = [x for x in summary if x['variant']=='spectrum_CV_fit' and x['conditioning']=='Q_half' and x['normalization']=='event_mu' and x['scale_ns']==202]
    positive = sum(x['remainder_excess_over_data_lo'] is not None and x['remainder_excess_over_data_lo']>0 for x in primary)
    findings.append(f'扩展谱F202的全部12个采集段/块长条件中，余项超额的条件区间有{positive}个完全大于0。它们共享事件并有多重比较，不能把该计数当作独立重复检验或显著性合并。')
    conclusion = '二阶时间协方差的Gaussian闭合与实际F202离散度给出了不同的MC/实测关系。应将二阶结构与事件间高阶/混合结构分别约束，不能仅靠一个噪声幅度倍率认定PID带宽已经匹配。'
    next_work = '下一步在原生MC的信号/附加残差分解内，检验信号相关方差包络及事件间异质性，区分条件Gaussian噪声的混合效应与创新分布高阶差异；候选须同时保留平均ADC、F50/102/202均值与SD、QDC/时间PID宽度约束。当前结果不足以直接改用某种非Gaussian噪声或确定FIR新参数。若用本池调参，随后需另冻结新母事件种子验证。'
    limitations = [
        'Gaussian闭合是波形向量一、二阶矩所决定的二次型方差。余项同时可能包含高阶累积量、信号/电荷/响应折混合、随机分母、非平稳或事件相关性，不唯一指认电子学或SiPM噪声机制。',
        '主分析保留响应折混合；细分Q半箱/折作为敏感性检查。Co中段和晚段部分格不足4个原事件，完整细分结果明确不可估计；不删除电荷箱、不将负U估计截成零。',
        '配对512次沿用父抽样计数；U项只使用不同原ID。有效次数不足512时，区间仅基于可估计抽样；没有严格覆盖率保证，稀疏Co低Q格尤其需要谨慎。',
        '所有均值、权重、对折参考和响应参数固定；区间不含重新拟合、装置系统误差或多重比较修正。已有实验和MC已经反复探索，不能称盲验证或宣称等效。',
        'Am数据包含α、γ与电子贡献；选择带及F诊断不是粒子真值或实验纯度。时间协方差曲线是描述性统计，不是直接测得的器件固有噪声谱。']
    inputs = {relative(OUT/name): sha(OUT/name) for name in ['protocol.json', 'analysis.json', 'algebra_tests.json']}
    decision = {'conclusion_zh': conclusion, 'findings_zh': findings, 'next_work_zh': next_work, 'limitations_zh': limitations,
        'inputs_sha256': inputs, 'code_sha256': sha(__file__), 'positive_remainder_conditional_intervals': positive,
        'tables_sha256': {relative(OUT/name): sha(OUT/name) for name in ['comparison_table.csv','per_Q_table.csv','cell_counts.csv']},
        'new_independent_mothers': 0, 'new_native_replays': 0, 'default_changed': False, 'official_PID_changed': False}
    save(OUT/'decision.json', decision)
    save(OUT/'figure_record.json', {'figures': FIGURES, 'inputs_sha256': inputs, 'code_sha256': sha(__file__)})
    print('Saved tables, five figure groups and conditional interpretation.', flush=True)


if __name__ == '__main__': main()
