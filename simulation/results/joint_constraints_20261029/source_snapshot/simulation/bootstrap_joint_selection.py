"""Fixed-fit MC-only stability of the frozen calibration-family ranking.

This diagnostic does not refit, reselect the saved candidate, or use validation
data. Rank frequencies are not posterior model probabilities.
"""
from pathlib import Path
import numpy as np
from prepare_joint_constraints import OUT, PROJECT, DATA, WIDTH, FAMILIES, SOURCES, relative
from prepare_pid import read
from extract_pid import load
from analyze_npe_validation import raw_shapes
from analyze_bandwidth_validation import per_bin, widths, rms_width
from fit_pid_bandwidth import average_bins, conditional_cov
from reference import TIME_NS
from publication_stats import save, sha

NBOOT = 512


def main():
    assert not (OUT/'ranking_stability.json').exists()
    selection = read(OUT/'selection.json'); protocol = read(OUT/'calibration/protocol.json'); design = read(OUT/'design.json')
    edges = np.asarray(design['q_edges']); scale = design['ranking_scales']; region = (TIME_NS >= -40)&(TIME_NS <= 3000)
    records = {}; inputs = {}; rng = np.random.default_rng(401029880); families = list(FAMILIES)
    values = np.full((NBOOT, len(families), 2), np.nan); i, j = np.triu_indices(3)
    for si, (source, run, kind) in enumerate(SOURCES):
        support = design['support_bins'][source]
        dp = DATA/f'run{run}_first60.npz'; sp = WIDTH/'arrays'/f'measured_run{run}_first60.npz'
        data = load(dp); d = load(sp); u = d['main_band']; q = d['q'][u]; scores = d['scores'][u]
        dq = per_bin(q, scores, edges); dc = conditional_cov(scores, q, edges, support, robust=True)
        df = average_bins(data['metrics'][u, :, 3], q, edges, support)
        da = average_bins(raw_shapes(data['waves'][u]), q, edges, support)[region]
        dmu = -1000*average_bins(scores[:, 1], q, edges, support)
        normal = np.sqrt(np.diag(dc)[i]*np.diag(dc)[j]); den = rms_width(widths(dq), support)
        for p in [dp, sp]: inputs[relative(p)] = sha(p)
        for fi, family in enumerate(families):
            key = f'{family}_run{run}_{source}'; mp = OUT/'calibration/arrays'/f'{key}.npz'; m = load(mp)
            raw = raw_shapes(m['waves'])[:, region]; bins = np.clip(np.searchsorted(edges, m['q'], 'right')-1, 0, 5)
            def objective(counts):
                covs = []; quarts = np.full((6, 3, 3), np.nan); ff = []; aa = []; mu = []
                for b in support:
                    mask = bins == b; c = counts[mask]
                    if c.sum() < 5: return np.nan
                    x = np.repeat(m['scores'][mask], c, axis=0)
                    quarts[b] = np.quantile(x, [.25, .5, .75], axis=0); covs.append(np.cov(x, rowvar=False))
                    w = c/c.sum(); aa.append(w@raw[mask]); ff.append(w@m['metrics'][mask]); mu.append(-1000*x[:, 1].mean())
                residuals = {'median_in_data_IQR': np.mean(((quarts[:, 1, :]-dq[:, 1, :])/widths(dq))[support], axis=0),
                    'mean_time_ns': np.array([np.mean(mu)-dmu]),
                    'IQR_ratio': rms_width(widths(quarts), support)/den-1,
                    'covariance_normalized': (np.mean(covs, axis=0)-dc)[i, j]/normal,
                    'F_ratio': np.mean(ff, axis=0)/df-1,
                    'mean_ADC_NRMSE': np.array([np.sqrt(np.mean((np.mean(aa, axis=0)-da)**2))/da.max()])}
                return sum(float(np.mean((v/scale[k])**2)) for k, v in residuals.items())
            point = objective(np.ones(len(m['q']), dtype=int))
            np.testing.assert_allclose(point, selection['ranking'][family]['sources'][source]['objective'], rtol=1e-10, atol=1e-10)
            n = int(m['n_parent'])
            for b in range(NBOOT):
                counts = np.bincount(rng.integers(n, size=n), minlength=n)[m['ids']]
                values[b, fi, si] = objective(counts)
            inputs[relative(mp)] = sha(mp)
            print('MC ranking resamples:', key, flush=True)
    total = values.mean(axis=2); valid = np.isfinite(total).all(axis=1); assert valid.sum() >= 480
    winners = np.argmin(total[valid], axis=1); selected = families.index(selection['selected_family'])
    results = {}
    for i, family in enumerate(families):
        delta = total[valid, i]-total[valid, selected]
        results[family] = {'point_objective': selection['ranking'][family]['objective'],
            'bootstrap_objective_p025_p50_p975': np.quantile(total[valid, i], [.025, .5, .975]).tolist(),
            'rank1_frequency': float(np.mean(winners == i)),
            'objective_minus_selected_CI95': np.quantile(delta, [.025, .975]).tolist()}
    np.savez_compressed(OUT/'ranking_bootstrap.npz', source_objectives=values, total_objectives=total, valid=valid)
    save(OUT/'ranking_stability.json', {'scope': __doc__, 'code_sha256': sha(Path(__file__)),
        'selection_sha256': sha(OUT/'selection.json'), 'inputs_sha256': inputs, 'families': families,
        'resamples': NBOOT, 'valid_resamples': int(valid.sum()), 'seed': 401029880, 'results': results,
        'conditional_scope': 'Original experimental first60 targets fixed; independently resample all calibration mothers within each source/family. Fitted parameters and family selection rule fixed. No refit or validation data. This isolates finite-MC sensitivity, not complete model uncertainty or a confidence interval for tau.',
        'saved_selection_changed': False})
    print(results, flush=True)


if __name__ == '__main__': main()
