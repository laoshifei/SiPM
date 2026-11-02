"""Plot fresh Geant4 batch statistics against fixed existing experimental references."""
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mc_statistics_common import *

FIGURES = ['independent_MC_support', 'independent_PID_quantiles', 'independent_batch_stability',
           'independent_native_mean_ADC', 'independent_modes_and_F', 'old_size_MC_subsamples']
COLORS = ['#777777', '#24848b', '#cf702d']
LABELS = ['Frozen', 'RC + inherited spectrum', 'RC + extended spectrum']
METHOD_LABELS = ['QDC', 'Mean-time score (-t)', 'PCA']
plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False,
                     'svg.fonttype': 'none', 'figure.dpi': 135, 'savefig.dpi': 180})


def finish(fig, name, note):
    fig.text(.5, .012, note, ha='center', va='bottom', fontsize=9)
    fig.tight_layout(rect=(0, .055, 1, .93))
    for ext in ['png', 'pdf', 'svg']: fig.savefig(OUT/'figures'/f'{name}.{ext}')
    plt.close(fig)


def error(ax, x, y, ci, color, marker='o', hollow=False):
    if ci is not None:
        ax.plot([x, x], ci, color=color, lw=1.2)
        ax.plot([x-.025, x+.025], [ci[0]]*2, color=color, lw=1)
        ax.plot([x-.025, x+.025], [ci[1]]*2, color=color, lw=1)
    ax.plot(x, y, marker, color=color, mfc='white' if hollow else color, ms=5)


def main(refresh=False):
    assert refresh or not (OUT/'figure_record.json').exists()
    e = read(OUT/'analysis.json'); p = read(OUT/'protocol.json')
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, (source, run) in zip(axes, SOURCES):
        group = e['groups'][f'run{run}_last20']['128']
        old = load(RC/'arrays'/f'run{run}_last20_B128_frozen_evaluation.npz')
        counts = [np.histogram(old['q'], p['q_edges'])[0]]
        for batch in BATCHES:
            a = load(OUT/'arrays'/f'{batch_key(batch,run)}_frozen.npz')
            counts.append(np.histogram(a['q'], p['q_edges'])[0])
        counts.append(np.sum(counts[1:], axis=0)); counts = np.array(counts)
        bottom = np.zeros(6)
        for b in range(6):
            if b not in p['support_bins'][source]: continue
            ax.bar(np.arange(6), counts[:, b], bottom=bottom, label=f'Q bin {b}')
            bottom += counts[:, b]
        for x, n in enumerate(bottom): ax.text(x, n, str(int(n)), ha='center', va='bottom', fontsize=9)
        ax.set_xticks(range(6), ['Old', 'Batch 1', 'Batch 2', 'Batch 3', 'Batch 4', 'Fresh pool'], rotation=20)
        ax.set_ylabel('Distinct mothers in fixed Q support'); ax.set_title(source)
        ax.set_ylim(0, max(bottom)*1.18); ax.grid(axis='y', alpha=.15)
    axes[0].legend(ncol=3, fontsize=9)
    finish(fig, FIGURES[0], 'Full transport: 64,000 fresh Co + 192,000 fresh Am mothers. Response replays do not increase independent statistics.')

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    styles = [(50, 'old', '#2166ac', -.15, True), (50, 'fresh', '#2166ac', -.05, False),
              (95, 'old', '#d97622', .05, True), (95, 'fresh', '#d97622', .15, False)]
    for i, (source, run) in enumerate(SOURCES):
        g = e['groups'][f'run{run}_last20']['128']
        for j, (method, label) in enumerate(zip(METHODS, METHOD_LABELS)):
            ax = axes[i, j]
            for x, v in enumerate(VARIANTS):
                r = g['variants'][v]
                for cov, oldfresh, color, offset, hollow in styles:
                    name = f'width{cov}_{method}_ratio'; old = oldfresh == 'old'
                    value = r['old_point' if old else 'point'][name]
                    ci = r['old_CI95_recomputed' if old else 'CI95'][name]['CI95']
                    error(ax, x+offset, value, ci, color, hollow=hollow)
            ax.axhline(1, color='black', ls='--', lw=.8); ax.grid(axis='y', alpha=.15)
            ax.set_xticks(range(3), ['Frozen', 'RC+inherited', 'RC+extended'], rotation=15)
            ax.set_title(f'{source}: {label}'); ax.set_ylabel('MC / data within-Q width')
    handles = [plt.Line2D([], [], color=c, marker='o', mfc='white' if hollow else c, ls='-',
                         label=f'{oldfresh}: central {cov}%') for cov, oldfresh, c, _, hollow in styles]
    fig.legend(handles=handles, loc='upper center', ncol=4, frameon=False)
    finish(fig, FIGURES[1], 'Last20, block128. Conditional 95% intervals; old MC and fresh MC independent, existing measured blocks shared. No equivalence claim.')

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for i, (source, run) in enumerate(SOURCES):
        g = e['groups'][f'run{run}_last20']['128']; r = g['variants']['spectrum_CV_fit']
        for j, (method, label) in enumerate(zip(METHODS, METHOD_LABELS)):
            ax = axes[i, j]
            for cov, color in [(50, '#2166ac'), (95, '#d97622')]:
                name = f'width{cov}_{method}_ratio'
                values = [r['old_point'][name]]+[x['point'][name] if x['point'] else np.nan for x in g['batches']['spectrum_CV_fit']]+[r['point'][name]]
                ax.plot(range(6), values, 'o-', color=color, ms=4, label=f'Central {cov}%')
                for x, ci in [(0, r['old_CI95_recomputed'][name]['CI95']), (5, r['CI95'][name]['CI95'])]:
                    error(ax, x, values[x], ci, color)
            ax.axhline(1, color='black', ls='--', lw=.8); ax.grid(alpha=.15)
            ax.set_xticks(range(6), ['Old', '1', '2', '3', '4', 'Pool'])
            ax.set_xlabel('Independent seed batch'); ax.set_ylabel('MC / data width'); ax.set_title(f'{source}: {label}')
    axes[0, 0].legend(frameon=False)
    finish(fig, FIGURES[2], 'Frozen extended-spectrum response, last20/block128. Individual batches show point estimates; old and fresh-pool bars are conditional 95% intervals.')

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8)); time = -20+40*np.arange(76)
    for col, (source, run) in enumerate(SOURCES):
        g = e['groups'][f'run{run}_last20']['128']; data = np.array(g['data_summary']['joint']['wave'])
        axes[0, col].plot(time, data, color='#153c65', lw=2, label='Measured')
        old = np.array(g['variants']['spectrum_CV_fit']['old_summary']['joint']['wave'])
        curves = [('Old extended', old, '#9d3b6d', '--')]
        for v, color, label in zip(VARIANTS, COLORS, LABELS):
            curves.append(('Fresh '+label, np.array(g['variants'][v]['fresh_summary']['joint']['wave']), color, '-'))
        for label, wave, color, style in curves:
            axes[0, col].plot(time, wave, color=color, ls=style, label=label)
            axes[1, col].plot(time, 100*(wave-data)/data.max(), color=color, ls=style)
        axes[0, col].set_title(source); axes[0, col].set_ylabel('Mean baseline-subtracted ADC')
        axes[1, col].set_ylabel('(MC - data) / data peak (%)'); axes[1, col].axhline(0, color='black', lw=.8)
        for ax in axes[:, col]: ax.set_xlabel('Time from own integer t10 anchor (ns)'); ax.grid(alpha=.15)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, fontsize=9, frameon=False)
    finish(fig, FIGURES[3], 'Last20, block128. Original Q-half targets and unscaled native ADC; no event amplitude matching or measured-template injection.')

    fig, axes = plt.subplots(3, 2, figsize=(12.5, 10))
    metric_groups = [(['rank3_variance_ratio', 'orthogonal_variance_ratio', 'full_shape_variance_ratio'], ['Rank 3', 'Orthogonal', 'Full shape']),
                     (['F50_ratio', 'F102_ratio', 'F202_ratio'], ['F50 mean', 'F102 mean', 'F202 mean']),
                     (['F50_SD_ratio', 'F102_SD_ratio', 'F202_SD_ratio'], ['F50 SD', 'F102 SD', 'F202 SD'])]
    for col, (source, run) in enumerate(SOURCES):
        g = e['groups'][f'run{run}_last20']['128']
        for row, (names, labels) in enumerate(metric_groups):
            ax = axes[row, col]
            for iv, v in enumerate(VARIANTS):
                r = g['variants'][v]
                for j, name in enumerate(names):
                    error(ax, j+(iv-1)*.13, r['point'][name], r['CI95'][name]['CI95'], COLORS[iv])
            r = g['variants']['spectrum_CV_fit']
            for j, name in enumerate(names):
                error(ax, j+.26, r['old_point'][name], r['old_CI95_recomputed'][name]['CI95'], '#9d3b6d', hollow=True)
            ax.axhline(1, color='black', ls='--', lw=.8); ax.grid(axis='y', alpha=.15)
            ax.set_title(source); ax.set_xticks(range(3), labels); ax.set_ylabel('MC / measured')
    handles = [plt.Line2D([], [], color=c, marker='o', ls='', label='Fresh '+label) for c, label in zip(COLORS, LABELS)]
    handles += [plt.Line2D([], [], color='#9d3b6d', marker='o', mfc='white', ls='', label='Old extended')]
    fig.legend(handles=handles, loc='upper center', ncol=2, frameon=False)
    finish(fig, FIGURES[4], 'Last20/block128; modes and responses frozen. F and PID share ADC samples. Conditional intervals exclude fitted-model uncertainty.')

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    names = ['width50_QDC_ratio', 'width95_negative_mean_time_ratio', 'width50_PCA_ratio']
    labels = ['QDC central 50% width', 'Mean-time central 95% width', 'PCA central 50% width']
    for i, (source, run) in enumerate(SOURCES):
        g = e['groups'][f'run{run}_last20']['128']; r = g['variants']['spectrum_CV_fit']
        boot = load(OUT/'evaluation'/f'run{run}_last20_B128_bootstrap.npz')
        for j, (name, label) in enumerate(zip(names, labels)):
            x = boot['old_size_subsamples'][:, 2, METRICS.index(name)]; x = x[np.isfinite(x)]
            ax = axes[i, j]; ax.hist(x, bins=24, density=True, color='#98b9ca', alpha=.85)
            ax.axvline(r['old_point'][name], color='black', ls='--', label='Old point')
            ax.axvline(r['point'][name], color='#b44c2f', lw=2, label='Fresh pool')
            ax.set_title(f'{source}: {label}'); ax.set_xlabel('MC / fixed data width'); ax.set_ylabel('Conditional diagnostic density')
            ax.grid(axis='y', alpha=.15)
    axes[0, 0].legend(frameon=False)
    finish(fig, FIGURES[5], '512 old-size subsets drawn from fresh full-parent IDs, without replacement. Repeated subsets are not new histories; ranges are not calibrated p-values.')
    inputs = [OUT/'protocol.json', OUT/'analysis.json']
    for _, run in SOURCES:
        inputs += [RC/'arrays'/f'run{run}_last20_B128_frozen_evaluation.npz',
                   OUT/'evaluation'/f'run{run}_last20_B128_bootstrap.npz']
        inputs += [OUT/'arrays'/f'{batch_key(batch,run)}_frozen.npz' for batch in BATCHES]
    save(OUT/'figure_record.json', {'code_sha256': sha(__file__), 'figures': FIGURES,
         'inputs_sha256': {relative(path): sha(path) for path in inputs}})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--refresh', action='store_true')
    main(parser.parse_args().refresh)
