"""Compare short-scale waveform fluctuations in the frozen common-charge range.

This is an empirical filter-dependent variance metric, not a photoelectron count,
SiPM gain measurement, or electronics deconvolution. No detector parameters fit.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import ROOT
from reference import PROJECT, bin_layout, wave_features
from compare import BLUE, RED, GREEN, KEEP, panel, graph, label, save
from diagnose_charge import load_measured, sha

HERE = Path(__file__).resolve().parent
WINDOWS = (25, 51, 101)


def fluctuation_metrics(waves, baselines, t10):
    """Centered box-filter residual variance, with all filter samples supported."""
    results = []
    for wave, baseline, start in zip(waves, baselines, t10):
        y = np.asarray(wave, dtype=float)-baseline
        cumulative = np.r_[0., np.cumsum(y)]; row = []
        for width in WINDOWS:
            half = width//2
            smooth = (cumulative[width:]-cumulative[:-width])/width
            signal = y[half:-half]; residual = signal-smooth
            time = (np.arange(half, len(y)-half)-start)*2
            pre = (time >= -600) & (time < -200)
            post = (time >= 500) & (time < 2500)
            if pre.sum() != 200 or post.sum() != 1000: raise ValueError('Incomplete diagnostic time window')
            mu = float(signal[post].mean()); vp = float(residual[pre].var(ddof=1)); vs = float(residual[post].var(ddof=1))
            row.append([mu, vp, vs, (vs-vp)/mu if mu > 0 else np.nan])
        results.append(row)
    return np.array(results)


def interval(x):
    x = np.asarray(x); x = x[np.isfinite(x)]
    if not len(x): return {'n': 0}
    rng = np.random.default_rng(281019)
    medians = np.median(x[rng.integers(0, len(x), size=(1000, len(x)))], axis=1)
    return {'n': len(x), 'p10_p50_p90': np.quantile(x, [.1, .5, .9]).tolist(),
            'median_event_bootstrap_95': np.quantile(medians, [.025, .975]).tolist()}


def load_mc(path, selected_indices):
    file = ROOT.TFile.Open(str(path)); tree = file.Get('Events')
    info = json.loads(str(file.Get('RunInfo').GetString().Data()))
    if not info.get('completed') or not info.get('digitization_enabled', True): raise ValueError('Invalid MC input')
    tree.SetBranchStatus('*', 0); tree.SetBranchStatus('adc', 1)
    waves = []
    for i in selected_indices:
        if tree.GetEntry(int(i)) <= 0: raise ValueError('MC event missing')
        waves.append(np.array(tree.adc, dtype=np.uint16))
    file.Close(); waves = np.array(waves)
    features = wave_features(waves)
    return waves, features


def plot(out, arrays, report):
    ROOT.gROOT.SetBatch(True); ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetTitleFontSize(.044); ROOT.gStyle.SetLabelSize(.035, 'XYZ'); ROOT.gStyle.SetTitleSize(.04, 'XYZ')
    canvas = ROOT.TCanvas('fluctuations', 'Short-scale fluctuation check', 1350, 1050); canvas.Divide(2, 2, .003, .003)
    for column, source in enumerate(('Co60', 'Am241')):
        datasets = [(arrays[source+'_measured_metrics'], BLUE, 'Measured / 0x8000 clear'),
                    (arrays[source+'_mc_metrics'], RED, 'Frozen Geant4 + SiPM')]
        ymax = max(np.quantile(v[:, :, 3], .95) for v, _, _ in datasets)*1.3
        ymin = min(-.01, min(np.quantile(v[:, :, 3], .05) for v, _, _ in datasets)*1.2)
        panel(canvas, column+1, source+';Box-filter width (ns);(Var_{signal} - Var_{pre}) / mean signal (ADC)', 30, float(ymin), 225, float(ymax))
        for i, (values, color, text) in enumerate(datasets):
            estimates = [interval(values[:, j, 3]) for j in range(len(WINDOWS))]
            x = np.array(WINDOWS, dtype=float)*2; y = np.array([e['p10_p50_p90'][1] for e in estimates])
            lo = np.array([e['median_event_bootstrap_95'][0] for e in estimates]); hi = np.array([e['median_event_bootstrap_95'][1] for e in estimates])
            g = ROOT.TGraphAsymmErrors(len(x), x, y, np.zeros(len(x)), np.zeros(len(x)), y-lo, hi-y)
            g.SetMarkerStyle(20+i); g.SetMarkerColor(color); g.SetLineColor(color); g.SetLineWidth(2); g.Draw('LP SAME'); KEEP.append(g)
            label(text+f', n={len(values)}', .23, .84-i*.055, color, .031)
        label('Median + event-bootstrap interval', .23, .72, size=.029)
        label('Same frozen Q range; no refit', .23, .66, size=.029)
        edges = np.linspace(-.1, .65, 61)
        hist = [np.histogram(values[:, 0, 3], edges)[0]/len(values) for values, _, _ in datasets]
        panel(canvas, column+3, '50 ns filter / event distribution;Excess variance / mean (ADC);Fraction per bin', -.1, 0, .65, float(max(v.max() for v in hist)*1.4))
        for values, (_, color, _) in zip(hist, datasets): graph((edges[1:]+edges[:-1])/2, values, color)
        label('Pre: -600 to -200 ns; signal: 500 to 2500 ns', .18, .86, size=.028)
        label('Filter metric; not a gain or photoelectron estimate', .18, .80, size=.027)
    save(canvas, out, 'fluctuation_comparison')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mc', type=Path, default=HERE/'results/joint_optical_20261028')
    parser.add_argument('--out', type=Path, required=True); args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    cohort = PROJECT/'analysis/paper_baseline/results/confirmed_cohort.json'
    qlo, qhi = json.loads(cohort.read_text())['common_charge_range_adc_samples']
    prior_path = args.mc/'joint_validation_final/validation.json'; prior = json.loads(prior_path.read_text())
    report = {'scope': __doc__, 'selection': 'Original training split, primary quality, amplitude >=200 ADC, fixed common Q; measured bit 0x8000 clear. Frozen MC event IDs from previous validation, then same common Q.',
        'common_charge_range_adc_samples': [qlo, qhi], 'filters_samples': WINDOWS,
        'metric_columns': ['mean_signal_adc', 'pretrigger_highpass_variance_adc2', 'signal_highpass_variance_adc2', 'excess_variance_over_mean_adc'],
        'filter': 'Subtract centered odd-width box average, retain only complete filter support; 2ns sampling. Unbiased sample variances in [-600,-200)ns and [500,2500)ns relative to t10. Negative excess variances retained.',
        'uncertainty': '1000 event-bootstrap medians, seed281019; finite recorded-event uncertainty only, not total model uncertainty. Q spectra within the common interval are not reweighted; per-Q-bin diagnostics retained.',
        'limitations': 'Signal-dependent fluctuations, readout filtering, photon statistics and correlated noise are not separately identified. Do not assign the difference to a unique material or circuit parameter.',
        'inputs_sha256': {str(cohort.relative_to(PROJECT)): sha(cohort), str(prior_path.relative_to(PROJECT)): sha(prior_path)},
        'code_sha256': {str(p.relative_to(PROJECT)): sha(p) for p in [Path(__file__), HERE/'diagnose_charge.py', HERE/'reference.py', PROJECT/'analysis/paper_baseline/prepare.py']},
        'sources': {}}
    arrays = {}; edges = np.linspace(qlo, qhi, 6)
    for source, run, name in [('Co60', 3, 'co60'), ('Am241', 12, 'am241')]:
        observed = load_measured(run, 0)
        mask = observed['selected'] & ~observed['flag_0x8000'] & (observed['q'] >= qlo) & (observed['q'] <= qhi)
        ids = np.flatnonzero(mask); raw_path = PROJECT/f'data/DT5730_DPP_PHA/DAQ/run_{run}/UNFILTERED/Data_run_{run}.BIN'
        _, dtype, n = bin_layout(raw_path); raw = np.memmap(raw_path, dtype=dtype, offset=2, mode='r', shape=(n,))
        metrics = fluctuation_metrics(raw['data'][ids], observed['f']['baseline'][ids], observed['f']['t10_sample'][ids])
        arrays[source+'_measured_metrics'] = metrics; arrays[source+'_measured_ids'] = ids; arrays[source+'_measured_q'] = observed['q'][ids]
        path = args.mc/name/'events.root'; all_ids = np.array(prior['sources'][source]['new']['event_indices'])
        assert sha(path) == prior['sources'][source]['input_sha256']['new']
        waves, f = load_mc(path, all_ids)
        mask = (f['charge_long'] >= qlo) & (f['charge_long'] <= qhi)
        metrics = fluctuation_metrics(waves[mask], f['baseline'][mask], f['t10_sample'][mask])
        arrays[source+'_mc_metrics'] = metrics; arrays[source+'_mc_ids'] = all_ids[mask]; arrays[source+'_mc_q'] = f['charge_long'][mask]
        report['inputs_sha256'].update(observed['input_sha256'])
        report['inputs_sha256'][str(path.relative_to(PROJECT))] = sha(path); report['inputs_sha256'][str(raw_path.relative_to(PROJECT))] = sha(raw_path)
        row = {'run': run, 'groups': {}, 'charge_bin_edges': edges.tolist(), 'charge_bins': []}
        for group in ('measured', 'mc'):
            values = arrays[source+'_'+group+'_metrics']; q = arrays[source+'_'+group+'_q']
            row['groups'][group] = {'n': len(values), 'charge_p10_p50_p90': np.quantile(q, [.1, .5, .9]).tolist(),
                'metrics': {str(width*2)+'ns': {'excess_variance_over_mean': interval(values[:, j, 3]),
                                             'pretrigger_variance': interval(values[:, j, 1]),
                                             'signal_variance': interval(values[:, j, 2])} for j, width in enumerate(WINDOWS)}}
        for low, high in zip(edges[:-1], edges[1:]):
            entry = {'q_low': low, 'q_high': high}
            for group in ('measured', 'mc'):
                q = arrays[source+'_'+group+'_q']; mask = (q >= low) & ((q < high) | ((high == qhi) & (q == high)))
                entry[group] = interval(arrays[source+'_'+group+'_metrics'][mask, 0, 3])
            row['charge_bins'].append(entry)
        report['sources'][source] = row
        print(source, {g: row['groups'][g]['metrics']['50ns']['excess_variance_over_mean'] for g in ('measured', 'mc')}, flush=True)
    np.savez_compressed(args.out/'metrics.npz', **arrays)
    (args.out/'diagnostics.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    plot(args.out, arrays, report)


if __name__ == '__main__': main()
