"""Characterize saved pretrigger noise and sign-symmetric weak-pulse candidates.

The main pulse is not used to select a candidate's shape. Candidates are NOT
certified single-photoelectron events, and saved intervals are NOT a dark rate.
Training is run3/run12 first60%; later data/run18 are previously explored checks.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import ROOT
from reference import PROJECT, bin_layout
from diagnose_charge import sha
from compare import BLUE, RED, GREEN, KEEP, panel, graph, label, save

HERE = Path(__file__).resolve().parent
LENGTH = 256
LAGS = np.arange(65)
WIDTH = 5
THRESHOLDS = (4., 5., 6.)


def detrend(waves):
    y = np.asarray(waves, dtype=float)
    y = y-y.mean(axis=1, keepdims=True)
    x = np.arange(y.shape[1], dtype=float); x -= x.mean()
    return y-(y@x/(x@x))[:, None]*x


def covariance(waves):
    y = detrend(waves)
    return np.array([np.mean(y[:, :LENGTH-l]*y[:, l:], axis=1) for l in LAGS]).T


def periodogram(waves):
    y = detrend(waves); window = np.hanning(y.shape[1])
    power = abs(np.fft.rfft(y*window, axis=1))**2*(.002/np.sum(window**2))
    power[:, 1:-1] *= 2  # dt=.002 us -> ADC^2/MHz, frequency in MHz
    return np.fft.rfftfreq(y.shape[1], .002), power


def smooth(waves):
    cumulative = np.pad(np.cumsum(waves, axis=1), ((0, 0), (1, 0)))
    return (cumulative[:, WIDTH:]-cumulative[:, :-WIDTH])/WIDTH


def candidate_indices(values, scale, threshold, sign):
    """One strongest excursion per trace/sign, with complete [-100,+200]ns support.

    A global per-training-run robust scale is frozen. Positive and negative
    searches are otherwise identical; a peak-selected average is selection biased.
    """
    z = smooth(detrend(values))*sign
    center = WIDTH//2
    search = z[:, 50-center:LENGTH-100-center]
    peak = search.argmax(axis=1)+50
    amp = z[np.arange(len(z)), peak-center]
    ids = np.flatnonzero(amp >= threshold*scale)
    return ids, peak[ids], amp[ids]


def summary(values):
    values = np.asarray(values)
    return {'n': len(values), 'p10_p50_p90': np.quantile(values, [.1, .5, .9]).tolist()} if len(values) else {'n': 0}


def read_run(run):
    fpath = PROJECT/f'analysis/paper_baseline/results/pha_run{run:04d}_features.npz'
    apath = PROJECT/f'analysis/paper_baseline/results/phase2/run{run:04d}_features.npz'
    with np.load(fpath) as z:
        f = {k: z[k] for k in ('diagnostic_flags', 't10_sample', 'amplitude', 'charge_long', 'flags')}
    with np.load(apath) as z:
        a = {k: z[k] for k in ('split', 'eligible_primary')}
    # First 512 ns are at least 256 ns before t10. No primary pretrigger veto.
    eligible = (f['diagnostic_flags'] == 0) & (f['t10_sample'] >= 384)
    path = PROJECT/f'data/DT5730_DPP_PHA/DAQ/run_{run}/UNFILTERED/Data_run_{run}.BIN'
    _, dtype, count = bin_layout(path); raw = np.memmap(path, dtype=dtype, offset=2, mode='r', shape=(count,))
    return f, a, eligible, raw, (fpath, apath, path)


def plot(out, curves, report):
    ROOT.gROOT.SetBatch(True); ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetTitleFontSize(.043); ROOT.gStyle.SetLabelSize(.034, 'XYZ'); ROOT.gStyle.SetTitleSize(.04, 'XYZ')
    colors = [('run3_train', BLUE, 'Run 3 training'), ('run12_train', RED, 'Run 12 training'), ('run18_check', GREEN, 'Run 18 check')]
    c = ROOT.TCanvas('pretrigger_noise', 'Pretrigger covariance and spectrum', 1350, 1050); c.Divide(2, 2, .003, .003)
    panel(c, 1, 'First 512 ns / linear detrend;Lag (ns);Normalized covariance', 0, -.18, 128, 1.2)
    for i, (key, color, text) in enumerate(colors):
        cov = curves[key+'_covariance']; graph(LAGS*2., cov/cov[0], color); label(text, .50, .84-i*.055, color, .032)
    label('Primary pretrigger veto NOT applied', .38, .62, size=.029)
    maximum = max(curves[key+'_psd'].max() for key, _, _ in colors)
    c.cd(2).SetLogy(); panel(c, 2, 'Hann periodogram / event average;Frequency (MHz);PSD (ADC^{2}/MHz)', 0, .0001, 250, float(maximum*5))
    for i, (key, color, text) in enumerate(colors):
        graph(curves['frequency_mhz'][1:], curves[key+'_psd'][1:], color); label(text, .52, .84-i*.055, color, .032)
    label('Finite 512 ns window; no rate inference', .22, .27, size=.028)
    for col, run in enumerate((3, 12)):
        key = f'run{run}_train'; row = report['groups'][key]; groups = row['weak_candidates']
        counts = np.array([[groups[str(t)][sign]['n'] for t in THRESHOLDS] for sign in ('positive', 'negative')])
        panel(c, col+3, f'Run {run} / matched sign controls;Threshold / training robust scale;Selected pretrigger traces', 3.5, 0, 6.5, float(max(2, counts.max()*1.6)))
        for values, color, text, y in [(counts[0], BLUE, 'Positive excursion', .84), (counts[1], RED, 'Negative control', .78)]:
            g = ROOT.TGraph(3, np.array(THRESHOLDS), values.astype(float)); g.SetMarkerStyle(20); g.SetLineColor(color); g.SetMarkerColor(color); g.Draw('LP SAME'); KEEP.append(g); label(text, .49, y, color, .032)
        label('Same search rule; not a single-PE spectrum', .22, .66, size=.028)
    save(c, out, 'pretrigger_noise')
    c = ROOT.TCanvas('weak_pulses', 'Selection-biased weak pulse averages', 1350, 600); c.Divide(2, 1, .003, .003)
    for col, run in enumerate((3, 12)):
        key = f'run{run}_train'
        styles = [('positive', BLUE, 'Positive selection'), ('negative', RED, 'Sign-inverted negative control')]
        available = [(s, co, text) for s, co, text in styles if key+'_'+s+'_candidate_mean' in curves]
        maximum = max([curves[key+'_'+s+'_candidate_mean'].max() for s, _, _ in available]+[1.])
        panel(c, col+1, f'Run {run} / threshold 5;Time from selected peak (ns);Detrended ADC (sign corrected)', -100, -maximum*.4, 200, maximum*1.7)
        for i, (sign, color, text) in enumerate(available):
            graph(np.arange(-50, 101)*2., curves[key+'_'+sign+'_candidate_mean'], color)
            n = report['groups'][key]['weak_candidates']['5.0'][sign]['n']; label(text+f', n={n}', .39, .84-i*.055, color, .030)
        label('Peak selection creates a pulse-shaped noise average', .19, .66, size=.027)
    save(c, out, 'weak_candidates')


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--out', type=Path, required=True); args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    report = {'scope': __doc__, 'selection': 'Basic diagnostic_flags==0 and t10>=384 samples. First256 samples/512ns, at least256ns before t10. No pretrigger veto or hardware-bit cut. Run18 capped at12000 evenly spaced eligible indices.',
        'noise_method': 'Per-trace ordinary least-squares removal of level and linear trend, then event-averaged unbiased-lag covariance and Hann periodogram; finite-window detrending bias retained. PSD units ADC^2/MHz.',
        'weak_method': '5-sample/10ns box smooth after linear detrend; strongest excursion in samples50..155 with complete [-100,+200]ns support. Fixed training pooled MAD/0.67448975 defines thresholds4/5/6. Positive/negative rules identical. Candidates may be noise or multiphoton/electronic features, not calibrated single avalanches.',
        'limits': 'Triggered waveforms only; no source activity, complete trigger count, dark rate, single-PE gain or physical RC measurement inferred.',
        'inputs_sha256': {}, 'code_sha256': {str(Path(__file__).relative_to(PROJECT)): sha(Path(__file__))}, 'groups': {}}
    arrays = {}; cohort = json.loads((PROJECT/'analysis/paper_baseline/results/confirmed_cohort.json').read_text()); qlo, qhi = cohort['common_charge_range_adc_samples']
    for run in (3, 12, 18):
        f, a, eligible, raw, paths = read_run(run)
        groups = [('train', 0), ('middle', 1), ('late', 2)] if run != 18 else [('check', 3)]
        scale = None
        for name, split in groups:
            key = f'run{run}_{name}'; ids = np.flatnonzero(eligible & (a['split'] == split)); available = len(ids)
            if len(ids) > 12000: ids = ids[np.linspace(0, len(ids)-1, 12000, dtype=int)]
            waves = raw['data'][ids, :LENGTH].astype(float); residual = detrend(waves)
            cov = covariance(waves); frequency, power = periodogram(waves)
            if scale is None:
                z = smooth(residual); scale = float(np.median(abs(z-np.median(z)))/.6744897501960817)
            primary = a['eligible_primary'][ids]; common = primary & (f['amplitude'][ids] >= 200) & ((f['flags'][ids] & 0x8000) == 0) & (f['charge_long'][ids] >= qlo) & (f['charge_long'][ids] <= qhi)
            row = {'role': 'training' if name == 'train' else 'previously explored check', 'n_available': available, 'n_used': len(ids),
                'noise_rms_detrended': summary(residual.std(axis=1)), 'normalized_covariance_lag1': float(cov[:, 1].mean()/cov[:, 0].mean()),
                'candidate_scale_adc': scale, 'primary_subset_n': int(primary.sum()), 'primary_covariance_lag1': float(cov[primary, 1].mean()/cov[primary, 0].mean()),
                'common_bit_clear_n': int(common.sum()), 'weak_candidates': {}}
            arrays[key+'_ids'] = ids; arrays[key+'_pretrigger_adc'] = waves.astype(np.uint16)
            arrays[key+'_covariance'] = cov.mean(axis=0); arrays[key+'_psd'] = power.mean(axis=0)
            arrays[key+'_primary_covariance'] = cov[primary].mean(axis=0); arrays[key+'_common_mask'] = common
            arrays['frequency_mhz'] = frequency
            for threshold in THRESHOLDS:
                entry = {}
                for sign, label_ in [(1, 'positive'), (-1, 'negative')]:
                    ii, peak, amplitude = candidate_indices(waves, scale, threshold, sign)
                    snippets = sign*residual[ii[:, None], peak[:, None]+np.arange(-50, 101)]
                    # Adjacent windows flank the peak and do not select by a presumed SPE shape.
                    areas = snippets[:, 40:100].sum(axis=1)*2 if len(ii) else np.array([])
                    entry[label_] = {'n': len(ii), 'amplitude_adc': summary(amplitude), 'area_minus20_plus100ns_adc_ns': summary(areas), 'in_primary_subset': int(primary[ii].sum())}
                    if threshold == 5.:
                        arrays[key+'_'+label_+'_candidate_ids'] = ids[ii]; arrays[key+'_'+label_+'_candidate_peak_sample'] = peak
                        arrays[key+'_'+label_+'_candidate_area'] = areas
                        if len(ii): arrays[key+'_'+label_+'_candidate_mean'] = snippets.mean(axis=0)
                row['weak_candidates'][str(threshold)] = entry
            report['groups'][key] = row
            print(key, 'n', len(ids), 'lag1', row['normalized_covariance_lag1'], 'candidates5', {s:v['n'] for s,v in row['weak_candidates']['5.0'].items()}, flush=True)
            (args.out/'progress.json').write_text(json.dumps(report, indent=2)+'\n')
        for path in paths: report['inputs_sha256'][str(path.relative_to(PROJECT))] = sha(path)
    np.savez_compressed(args.out/'samples.npz', **arrays)
    (args.out/'readout.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    plot(args.out, arrays, report)


if __name__ == '__main__': main()
