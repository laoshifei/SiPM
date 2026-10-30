"""Fit positive signal-noise spectra for each fixed native timing family."""
import argparse
import copy
from pathlib import Path
import numpy as np
from scipy.optimize import nnls
from prepare_joint_constraints import OUT, PROJECT, BUILD, DATA, WIDTH, relative
from prepare_pid import read
from extract_pid import load
from reference import wave_features
from fit_pid_bandwidth import conditional_cov, average_bins, projected_cov
from fit_signal_fluctuations import colored
from diagnose_fluctuations import fluctuation_metrics
from refit_final_parameters import combine_kernels
from publication_stats import save, sha


def fit_one(entry, protocol, pca):
    key = entry['key']; run = entry['run']; dest = OUT/'noise_fit'
    assert not (dest/f'{key}.json').exists(), 'Noise fit already saved'
    cfg = read(PROJECT/entry['config']); assert cfg['signal_dependent_noise']['variance_per_signal_adc'] == 0.
    mp = OUT/'profile/arrays'/f'{key}.npz'; m = load(mp)
    dp = DATA/f'run{run}_first60.npz'; sp = WIDTH/'arrays'/f'measured_run{run}_first60.npz'
    data = load(dp); d = load(sp); use = d['main_band']; np.testing.assert_array_equal(data['ids'], d['ids'])
    edges = np.asarray(protocol['q_edges']); support = protocol['support_bins'][entry['source']]
    bins = np.clip(np.searchsorted(edges, m['q'], 'right')-1, 0, 5)
    assert all(np.sum(bins == b) >= 10 for b in support)
    target_cov = conditional_cov(d['scores'][use], d['q'][use], edges, support, robust=True)
    target_F = average_bins(data['metrics'][use, :, 3], d['q'][use], edges, support)
    base_cov = conditional_cov(m['scores'], m['q'], edges, support)
    base_F = average_bins(m['metrics'], m['q'], edges, support)
    mf = wave_features(m['waves']); t10 = mf['t10_sample']
    # F windows are relative to the actual fractional t10. The midpoint proxy
    # anchor-0.5 shifts a boundary sample when t10 is exactly an integer.
    metrics = fluctuation_metrics(m['waves'], mf['baseline'], t10)
    np.testing.assert_allclose(metrics[:, :, 3], m['metrics'], atol=1e-12, rtol=1e-12)
    kernels = [np.asarray(cfg['signal_dependent_noise']['coefficients']), np.array([1.])]
    names = ['inherited_spectrum', 'white']
    for tau in [20., 60., 150., 400., 1000., 2500.]:
        h = np.exp(-np.arange(1025)*2/tau); h /= np.linalg.norm(h)
        kernels.append(h); names.append(f'exp_{tau:g}ns')
    rng = np.random.default_rng(4010291100+run)
    white = rng.normal(size=(16, len(m['q']), 2496+1024)); sqrt_signal = np.sqrt(np.maximum(m['signal'], 0))
    def contributions(h):
        cov = average_bins(projected_cov(m['signal'], m['anchor'], m['scores'], m['q'], h, pca), m['q'], edges, support)
        fs = []
        for z in white:
            noise = colored(h, z[:, :2496+len(h)-1])*sqrt_signal
            fm = fluctuation_metrics(noise, np.zeros(len(m['q'])), t10)
            fs.append(average_bins((fm[:, :, 2]-fm[:, :, 1])/metrics[:, :, 0], m['q'], edges, support))
        return cov, np.mean(fs, axis=0)
    C, F = map(np.asarray, zip(*(contributions(h) for h in kernels)))
    i, j = np.triu_indices(3); normal = np.sqrt(np.diag(target_cov)[i]*np.diag(target_cov)[j])
    def vector(c, f): return np.r_[c[i, j]/normal/.25, f/target_F/.10]
    A = np.column_stack([vector(c, f) for c, f in zip(C, F)])
    target = vector(target_cov-base_cov, target_F-base_F)
    variances, linear_loss = nnls(A, target)
    h, initial_strength = combine_kernels(kernels, variances)
    c, f = contributions(h); v = vector(c, f)
    strength = max(0., float(v@target/(v@v)))
    pcov = base_cov+strength*c; pF = base_F+strength*f
    candidate = copy.deepcopy(cfg)
    candidate['signal_dependent_noise'].update(coefficients=h.tolist(), variance_per_signal_adc=strength,
        provenance=relative(dest/f'{key}.json'), spectral_components='Positive full PID covariance and F joint fit on first60, fixed bounded timing family')
    candidate['joint_constraints']['stage'] = 'noise_fitted_calibration'
    candidate['status'] = 'Joint timing/noise calibration candidate; native ADC replay and independent validation required'
    cp = dest/f'{key}_config.json'; save(cp, candidate)
    report = {'scope': __doc__, 'key': key, 'family': entry['family'], 'run': run, 'source': entry['source'],
        'code_sha256': sha(Path(__file__)), 'protocol_sha256': sha(OUT/'profile/protocol.json'),
        'inputs_sha256': {relative(p): sha(p) for p in [mp, dp, sp]}, 'support_bins': support,
        'n_MC': len(m['q']), 'bin_counts': np.bincount(bins, minlength=6).tolist(),
        'target_cov': target_cov.tolist(), 'base_cov': base_cov.tolist(), 'predicted_cov': pcov.tolist(),
        'target_F': target_F.tolist(), 'base_F': base_F.tolist(), 'predicted_F': pF.tolist(),
        'predicted_SD_ratio': np.sqrt(np.diag(pcov)/np.diag(target_cov)).tolist(), 'predicted_F_ratio': (pF/target_F).tolist(),
        'loss': float(np.sum(vector(pcov-target_cov, pF-target_F)**2)), 'linear_loss_squared': float(linear_loss**2),
        'components': names, 'component_variances': variances.tolist(), 'component_covariances': C.tolist(), 'component_F': F.tolist(),
        'combined_initial_strength': initial_strength, 'variance_per_signal_adc': strength,
        'inherited_event_variance_cv': cfg['signal_dependent_noise']['event_variance_cv'],
        'F_window_anchor': 'Actual original fractional t10; exact-integer boundaries retained, no midpoint proxy',
        'config': relative(cp), 'config_sha256': sha(cp),
        'limits': 'Linearized covariance and fixed-anchor quadratic F proposal; actual own-anchor/reselected native ADC replay is the ranking input. Fixed inherited event variance CV; no fitted PID score smearing.'}
    save(dest/f'{key}.json', report)
    print(key, 'strength', strength, 'SD', report['predicted_SD_ratio'], 'F', report['predicted_F_ratio'], flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--keys', nargs='+'); p.add_argument('--protocol', action='store_true')
    args = p.parse_args(); protocol = read(OUT/'profile/protocol.json')
    pca = read(PROJECT/'analysis/paper_baseline/results/publication_20261028/point.json')['windows']['3000']['pca']
    if not args.protocol:
        for key in args.keys or protocol['entries']: fit_one(protocol['entries'][key], protocol, pca)
        return
    entries = {}
    for key, e in protocol['entries'].items():
        fit = read(OUT/'noise_fit'/f'{key}.json'); cp = PROJECT/fit['config']
        folder = OUT/'calibration'/key; folder.mkdir(); rp = folder/'events.root'
        entries[key] = {**e, 'native': False, 'parent': e['path'], 'parent_sha256': sha(PROJECT/e['path']),
            'config': relative(cp), 'config_sha256': sha(cp), 'path': relative(rp),
            'command': [str(BUILD/'replay_response'), str(PROJECT/e['path']), str(cp), str(rp)]}
    save(OUT/'calibration/protocol.json', {**protocol, 'entries': entries, 'independent_mothers': 0,
        'stage': 'calibration', 'noise_fit_code_sha256': sha(Path(__file__)),
        'fit_sha256': {k: sha(OUT/'noise_fit'/f'{k}.json') for k in entries},
        'note': 'Replays reuse all 96000 calibration mothers; never count them again as independent events.'})


if __name__ == '__main__': main()
