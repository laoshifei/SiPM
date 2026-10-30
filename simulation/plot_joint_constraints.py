"""Export bounded-family calibration and independent native waveform checks."""
import csv
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/sipm-joint-constraints-mpl')
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from prepare_joint_constraints import OUT, T, SOURCES, FAMILIES, PROJECT
from prepare_pid import read
from extract_pid import load
from publication_stats import save, sha

COLORS = {'npe': '#82909e', 'tail': '#bd7435', 'joint': '#176e99', 'data': '#222222',
          'a4000': '#9563a8', 'a6000': '#287a68', 'a8439': '#3c73ac'}
LABELS = {'npe': 'Photon-statistics reference', 'tail': 'Previous tail candidate', 'joint': 'Joint candidate (fresh MC)'}


def finish(fig, name, caption, legend=None, ncol=3):
    top = 1.
    if legend is not None:
        h, l = legend.get_legend_handles_labels()
        fig.legend(h, l, loc='upper center', ncol=ncol, fontsize=9, frameon=False); top = .94
    fig.tight_layout(rect=(0, .055, 1, top)); fig.text(.5, .012, caption, ha='center', fontsize=8)
    for ext in ['png', 'svg', 'pdf']: fig.savefig(OUT/'figures'/f'{name}.{ext}', dpi=180, bbox_inches='tight')
    plt.close(fig)


def bar(ax, x, y, ci, color):
    lo, hi = ci; ax.vlines(x, lo, hi, color=color, lw=1.1)
    ax.hlines([lo, hi], x-.025, x+.025, color=color, lw=1.1); ax.plot(x, y, 'o', color=color, ms=4.5)


def main():
    report = read(OUT/'assessment.json'); selection = read(OUT/'selection.json')
    proposals = load(OUT/'proposal_curves.npz'); means = load(OUT/'means.npz')
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for col, (source, run, kind) in enumerate(SOURCES):
        data = proposals[source+'_data']
        for family in FAMILIES:
            axes[0, col].plot(T, 100*(proposals[family+'__'+source]-data)/data.max(), color=COLORS[family], label=f'{FAMILIES[family]/1000:.2f} us')
        axes[0, col].axhline(0, color='k', lw=.7)
        axes[0, col].set(title=f'{source}: analytic proposal residual', xlabel='Integer-anchor time (ns)', ylabel='Difference / data peak (%)')
    group_names = ['median', 'mean time', 'width', 'covariance', 'F', 'ADC mean']
    fields = ['median_in_data_IQR', 'mean_time_ns', 'IQR_ratio', 'covariance_normalized', 'F_ratio', 'mean_ADC_NRMSE']
    for col, (source, run, kind) in enumerate(SOURCES):
        for i, family in enumerate(FAMILIES):
            r = selection['ranking'][family]['sources'][source]
            axes[1, col].bar(np.arange(6)+(i-1)*.24, [r['objective_terms'][x] for x in fields], width=.23, color=COLORS[family])
        axes[1, col].set_xticks(range(6), group_names, rotation=20)
        axes[1, col].set(title=f'{source}: actual native calibration criterion', ylabel='Squared standardized group loss')
    finish(fig, 'joint_family_profile', 'Top: proposals only. Bottom: native ADC replays on calibration mothers; first60 only. Families are conditional, not measured decay constants.', axes[0, 0])

    time = means['time_ns']; region = (time >= -40)&(time <= 3800)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for col, (source, run, kind) in enumerate(SOURCES):
        key = f'run{run}_{source}'; data = means[f'run{run}_last20__data_ADC']
        axes[0, col].plot(time[region], data[region], color=COLORS['data'], lw=2, label='Measured last20')
        for name in ['npe', 'tail', 'joint']:
            y = means[f'{key}__{name}__ADC']
            axes[0, col].plot(time[region], y[region], color=COLORS[name], lw=1.4, label=LABELS[name])
            axes[1, col].plot(time[region], 100*(y[region]-data[region])/data.max(), color=COLORS[name])
        axes[0, col].set(title=f'{source}: unscaled mean ADC', xlabel='Time from t10 (ns)', ylabel='Baseline-subtracted ADC')
        axes[1, col].set(title=f'{source}: mean waveform residual', xlabel='Time from t10 (ns)', ylabel='Difference / data peak (%)')
        axes[1, col].axhline(0, color='k', lw=.7)
        for ax in axes[:, col]: ax.axvspan(3000, 3800, color='#d9e4e8', alpha=.35)
    finish(fig, 'joint_native_waveforms', 'Unscaled equal-Q mean ADC; last20 internal check. Shaded3-3.8 us is outside the fitted/formal3 us gate; no retuning.', axes[0, 0], 2)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for col, (source, run, kind) in enumerate(SOURCES):
        for k, name in enumerate(['npe', 'tail', 'joint']):
            for i, segment in enumerate(['first60', 'middle20', 'last20']):
                r = report['checks'][f'run{run}_{segment}']['models'][name]
                bar(axes[0, col], i+(k-1)*.16, r['tail_difference'][0], r['uncertainty_by_data_block']['64']['tail_difference_CI95'][0]['estimate95'], COLORS[name])
            r = report['checks'][f'run{run}_last20']['models'][name]
            for j in range(3): bar(axes[1, col], j+(k-1)*.16, r['IQR_ratio'][j], r['uncertainty_by_data_block']['64']['IQR_ratio_CI95'][j]['estimate95'], COLORS[name])
            axes[0, col].plot([], [], 'o', color=COLORS[name], label=LABELS[name])
        axes[0, col].set(title=f'{source}: mean-time center', ylabel='MC - data mean time (ns)')
        axes[0, col].set_xticks(range(3), ['First60', 'Middle20', 'Last20']); axes[0, col].axhline(0, color='k', lw=.7, ls='--')
        axes[1, col].set(title=f'{source}: last20 PID width', ylabel='Equal-Q RMS(IQR): MC / data')
        axes[1, col].set_xticks(range(3), ['QDC', 'Mean time', 'PCA']); axes[1, col].axhline(1, color='k', lw=.7, ls='--')
        for ax in axes[:, col]: ax.grid(axis='y', alpha=.15)
    finish(fig, 'joint_native_pid', 'Pointwise 95% intervals:512 whole-mother / data block64 draws. Fit, family selection and apparatus fixed; no experimental truth claim.', axes[0, 0])

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for col, (source, run, kind) in enumerate(SOURCES):
        for k, name in enumerate(['npe', 'tail', 'joint']):
            r = report['checks'][f'run{run}_last20']['models'][name]
            for j in range(3): bar(axes[0, col], j+(k-1)*.16, r['F_ratio'][j], r['uncertainty_by_data_block']['64']['F_ratio_CI95'][j]['estimate95'], COLORS[name])
            axes[0, col].plot([], [], 'o', color=COLORS[name], label=LABELS[name])
            axes[1, col].plot(range(6), r['joint_criterion_check']['covariance_difference_normalized'], 'o-', color=COLORS[name], ms=4)
        axes[0, col].set(title=f'{source}: last20 short-scale fluctuations', ylabel='Mean F: MC / data', xlabel='Scale (ns)')
        axes[0, col].set_xticks(range(3), ['50', '102', '202']); axes[0, col].axhline(1, color='k', lw=.7, ls='--')
        axes[1, col].set(title=f'{source}: full PID covariance residual', ylabel='Covariance difference / data SD products')
        axes[1, col].set_xticks(range(6), ['QQ', 'QT', 'QP', 'TT', 'TP', 'PP']); axes[1, col].axhline(0, color='k', lw=.7, ls='--')
    finish(fig, 'joint_noise_controls', 'F intervals use fixed-model bootstrap; covariance panels show point estimates. Data covariance winsorized; native covariance untrimmed.', axes[0, 0])

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    edges = np.asarray(read(OUT/'design.json')['q_edges']); centers = .5*(edges[1:]+edges[:-1])/1000
    offset = np.mean(np.diff(centers))*.10
    for i, (source, run, kind) in enumerate(SOURCES):
        row = report['checks'][f'run{run}_last20']; support = row['support_bins']
        data = np.asarray(row['data_quartiles'], dtype=float)
        for j, method in enumerate(['QDC', 'Negative mean time (us)', 'PCA projection']):
            ax = axes[i, j]; x = centers[support]
            ax.fill_between(x, data[support, 0, j], data[support, 2, j], color='#ddd', label='Measured IQR')
            ax.plot(x, data[support, 1, j], 'o-', color=COLORS['data'], ms=4, label='Measured median')
            for k, name in enumerate(['npe', 'tail', 'joint']):
                q = np.asarray(row['models'][name]['MC_quartiles'], dtype=float)
                ax.errorbar(x+(k-1)*offset, q[support, 1, j],
                    yerr=np.array([q[support, 1, j]-q[support, 0, j], q[support, 2, j]-q[support, 1, j]]),
                    fmt='o--', lw=1, ms=3, capsize=2, color=COLORS[name], label=LABELS[name])
            ax.set(title=f'{source}: {method}', xlabel='Qlong (1000 ADC samples)', ylabel='PID score')
    finish(fig, 'joint_pid_by_charge', 'Last20 fixed supported Q bins. Bars/shading show interquartile ranges, not confidence intervals; small horizontal offsets aid visibility.', axes[0, 0], 3)

    rows = []
    for group, case in report['checks'].items():
        for name, r in case['models'].items():
            lo, hi = r['uncertainty_by_data_block']['64']['tail_difference_CI95'][0]['estimate95']
            rows.append({'group': group, 'model': name, 'n_MC': r['n_MC'], 'mean_time_difference_ns': r['tail_difference'][0],
                'mean_time_CI95_low': lo, 'mean_time_CI95_high': hi, 'ADC_NRMSE_percent': 100*r['mean_ADC_normalized_RMSE'],
                'extended_tail_charge_over_Qlong_difference': r['extended_tail_3000_3800_ns']['difference'],
                'extended_tail_ADC_NRMSE_percent': 100*r['extended_tail_3000_3800_ns']['ADC_NRMSE_over_full_peak'],
                **{f'{k}_IQR_ratio': v for k, v in zip(['QDC', 'mean_time', 'PCA'], r['IQR_ratio'])},
                **{f'F{k}_ratio': v for k, v in zip([50, 102, 202], r['F_ratio'])}, 'joint_criterion': r['joint_criterion_check']['objective']})
    with (OUT/'validation_table.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n'); writer.writeheader(); writer.writerows(rows)
    status = {}
    for source, run, kind in SOURCES:
        r = report['checks'][f'run{run}_last20']['models']['joint']; ci = r['uncertainty_by_data_block']['64']
        contains = lambda field, j, target: ci[field][j]['estimate95'][0] <= target <= ci[field][j]['estimate95'][1]
        status[source] = {'mean_time_CI_contains_zero': contains('tail_difference_CI95', 0, 0.),
            'PID_width_CI_contains_one': [contains('IQR_ratio_CI95', j, 1.) for j in range(3)],
            'F_CI_contains_one': [contains('F_ratio_CI95', j, 1.) for j in range(3)]}
    c = read(OUT/'validation/run12_Am241/config.json')
    save(OUT/'decision.json', {'selected_family': selection['selected_family'], 'promoted_to_default': False,
        'last20_conditional_interval_checks': status, 'alpha_relative_yield': c['optical']['alpha_light_yield_ratio'],
        'alpha_slow_ns': c['optical']['alpha_tau_ns'][1],
        'limits': 'No validation-driven retuning. Interval overlap is not an equivalence test. Conditional family and noise fit remain nonunique; apparatus and selection uncertainty excluded. No source-independent electronics or matched-state ROC validation in this phase.',
        'next_work': 'Use calibration-vs-validation and per-Q residuals to decide whether more independent MC or response-shape model flexibility is justified. Keep later noise-state drift as a separate controlled study; do not conflate it with particle truth.'})
    save(OUT/'figure_record.json', {'code_sha256': sha(Path(__file__)), 'assessment_sha256': sha(OUT/'assessment.json'),
        'selection_sha256': sha(OUT/'selection.json'), 'figures': ['joint_family_profile', 'joint_native_waveforms', 'joint_native_pid', 'joint_noise_controls', 'joint_pid_by_charge'], 'table_rows': len(rows)})
    print('Last20 conditional checks:', status, flush=True)


if __name__ == '__main__': main()
