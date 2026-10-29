"""Join frozen control results; compare mean shapes and event fluctuations."""
import argparse
import json
from pathlib import Path
import numpy as np
import ROOT
from compare import BLUE, RED, KEEP, panel, graph, label, save
from prepare_response_test import sha, PROJECT, HERE
from reference import bin_layout, wave_features
from diagnose_fluctuations import load_mc


def event_examples(args, gray):
    """Choose by Q rank/distance only; never rank by waveform similarity."""
    old = np.load(args.base_fluctuations/'metrics.npz')
    new = np.load(args.control/'fluctuations/metrics.npz')
    record = {'selection': 'Measured Q25/Q75 rank within the fluctuation cohort; nearest-Q old/new MC (ties by saved order). No waveform-distance selection, amplitude scaling or denoising.', 'examples': []}
    canvas = ROOT.TCanvas('response_events', 'Individual waveform illustrations', 1400, 1100); canvas.Divide(2, 2, .003, .003)
    for col, (source, run, name) in enumerate([('Co60', 3, 'co60'), ('Am241', 12, 'am241')]):
        path = PROJECT/f'data/DT5730_DPP_PHA/DAQ/run_{run}/UNFILTERED/Data_run_{run}.BIN'
        _, dtype, count = bin_layout(path); raw = np.memmap(path, dtype=dtype, offset=2, mode='r', shape=(count,))
        measured_q = new[source+'_measured_q']; order = np.argsort(measured_q, kind='stable')
        for row, quantile in enumerate((.25, .75)):
            rank = order[int(quantile*(len(order)-1))]; event_id = int(new[source+'_measured_ids'][rank])
            q = float(measured_q[rank]); w = np.array([raw['data'][event_id]]); f = wave_features(w)
            data = [('Measured', w[0], f, BLUE, event_id, q)]; detail = {'source': source, 'run': run, 'q_rank': quantile, 'measured_event_id': event_id, 'measured_q': q, 'mc': {}}
            for text, values, folder, color in [('Previous MC', old, args.base, gray), ('Finite-rise MC', new, args.control, RED)]:
                j = int(np.argmin(np.abs(values[source+'_mc_q']-q))); index = int(values[source+'_mc_ids'][j])
                waves, features = load_mc(folder/name/'events.root', [index]); mq = float(features['charge_long'][0])
                data.append((text, waves[0], features, color, index, mq))
                detail['mc'][text] = {'event_id': index, 'q': mq, 'q_relative_difference': mq/q-1}
            peak = max(float((wave-f['baseline'][0]).max()) for _, wave, f, _, _, _ in data)
            panel(canvas, 1+col+2*row, f'{source} / measured Q{int(100*quantile)} rank;Time from t_{{10}} (#mus);Baseline-subtracted ADC', -.4, -20, 3.4, peak*1.45)
            for i, (text, wave, f, color, index, charge) in enumerate(data):
                x = (np.arange(len(wave))-f['t10_sample'][0])*.002
                graph(x, wave-f['baseline'][0], color, 2 if color == gray else 1)
                label(f'{text}: event {index}, Q={charge/1000:.1f}k', .34, .85-i*.055, color, .028)
            label('Q-matched illustration; no waveform matching', .43, .61, size=.026)
            record['examples'].append(detail)
    save(canvas, args.out, 'individual_waveforms')
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control', type=Path, required=True)
    parser.add_argument('--base', type=Path, default=HERE/'results/joint_optical_20261028')
    parser.add_argument('--base-fluctuations', type=Path, default=HERE/'results/fluctuations_20261028')
    parser.add_argument('--out', type=Path, required=True); args = parser.parse_args()
    args.control = args.control.resolve(); args.base = args.base.resolve(); args.base_fluctuations = args.base_fluctuations.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    oldpath = args.base_fluctuations/'diagnostics.json'; newpath = args.control/'fluctuations/diagnostics.json'
    old = json.loads(oldpath.read_text()); new = json.loads(newpath.read_text())
    joint_path = args.control/'joint_validation_final/validation.json'; joint = json.loads(joint_path.read_text())
    mean_path = args.control/'common_charge/comparison.json'; means = json.loads(mean_path.read_text())
    report = {'scope': 'Controlled factorization, no refit of geometry, yield, gain, fractions, or decay constants.',
        'selection': 'Mean curves use original frozen common-Q reference (no flag cut). Fluctuations use training primary-QC + amplitude>=200 + same common Q, measured bit0x8000 clear. Do not conflate these populations.',
        'uncertainties': 'Intervals are conditional event-bootstrap intervals; do not include parameter, selection, hardware-history or geometry uncertainty. Old and new MC are independent, not event-paired.',
        'inputs_sha256': {str(p.relative_to(PROJECT)): sha(p) for p in [oldpath, newpath, joint_path, mean_path, args.control/'design.json']},
        'code_sha256': {str(Path(__file__).relative_to(PROJECT)): sha(Path(__file__))},
        'sources': {}, 'mean_checks': means['comparisons']}
    for source in ('Co60', 'Am241'):
        # A model change cannot silently change the measured diagnostic cohort.
        if old['sources'][source]['groups']['measured'] != new['sources'][source]['groups']['measured']:
            raise ValueError('Measured fluctuation cohort changed')
        entry = {'fluctuations': {}, 'absolute_mean_and_spectra': {k: v for k, v in joint['sources'][source].items() if k not in ('mc_run_info',)}}
        # Event identities are already archived in the detailed validation file.
        for key in ('measured', 'old', 'new'): entry['absolute_mean_and_spectra'][key].pop('event_indices', None)
        for width in ('50ns', '102ns', '202ns'):
            entry['fluctuations'][width] = {
                'measured': new['sources'][source]['groups']['measured']['metrics'][width]['excess_variance_over_mean'],
                'old_mc': old['sources'][source]['groups']['mc']['metrics'][width]['excess_variance_over_mean'],
                'finite_rise_mc': new['sources'][source]['groups']['mc']['metrics'][width]['excess_variance_over_mean']}
        report['sources'][source] = entry
    ROOT.gROOT.SetBatch(True); ROOT.gStyle.SetOptStat(0); ROOT.gStyle.SetTitleFontSize(.043)
    ROOT.gStyle.SetLabelSize(.034, 'XYZ'); ROOT.gStyle.SetTitleSize(.039, 'XYZ')
    gray = ROOT.TColor.GetColor('#888888')
    refs = np.load(HERE/'results/reference/references.npz')
    base = np.load(args.base/'common_charge_final/curves.npz')
    candidate = np.load(args.control/'common_charge/curves.npz')
    canvas = ROOT.TCanvas('response_control', 'Light timing and readout control', 1350, 1050); canvas.Divide(2, 2, .003, .003)
    for col, (source, run) in enumerate([('Co60', 3), ('Am241', 12)]):
        curves = [(refs[f'run{run}_train_mean']*1000, BLUE, 1, 'Measured training'),
                  (base[source+'_common_mean']*1000, gray, 2, 'Previous: electronic RC'),
                  (candidate[source+'_common_mean']*1000, RED, 1, 'Control: emission rise')]
        panel(canvas, col+1, source+';Time from t_{10} (#mus);Area-normalized signal (#mus^{-1})', -.2, -.02, 3.8, float(max(y.max() for y, _, _, _ in curves)*1.4))
        for i, (y, color, style, text) in enumerate(curves):
            graph(candidate['time_ns']/1000, y, color, style); label(text, .48, .84-i*.055, color, .031)
        label('Original common Q; no flag cut', .48, .62, size=.027)
        entry = report['sources'][source]['fluctuations']
        maximum = max(entry[w][g]['p10_p50_p90'][1] for w in entry for g in entry[w])*9.
        canvas.cd(col+3).SetLogy()
        panel(canvas, col+3, source+';Box-filter width (ns);Excess variance / mean signal (ADC)', 30., .005, 225., maximum)
        for i, (key, color, text) in enumerate([('measured', BLUE, 'Measured / 0x8000 clear'), ('old_mc', gray, 'Previous MC'), ('finite_rise_mc', RED, 'Finite-rise MC')]):
            records = [entry[w][key] for w in ('50ns', '102ns', '202ns')]
            x = np.array([50., 102., 202.]); y = np.array([r['p10_p50_p90'][1] for r in records])
            lo = np.array([r['median_event_bootstrap_95'][0] for r in records]); hi = np.array([r['median_event_bootstrap_95'][1] for r in records])
            g = ROOT.TGraphAsymmErrors(3, x, y, np.zeros(3), np.zeros(3), y-lo, hi-y)
            g.SetMarkerStyle(20+i); g.SetMarkerColor(color); g.SetLineColor(color); g.SetLineWidth(2); g.Draw('LP SAME'); KEEP.append(g)
            label(text+f", n={records[0]['n']}", .23, .84-i*.055, color, .031)
        label('Log y; fixed Q; bootstrap intervals', .45, .21, size=.025)
    save(canvas, args.out, 'response_control')
    report['individual_waveforms'] = event_examples(args, gray)
    (args.out/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({s: row['fluctuations']['50ns'] for s, row in report['sources'].items()}, indent=2))


if __name__ == '__main__': main()
