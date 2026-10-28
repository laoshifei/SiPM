"""Compare aligned mean waveforms in ADC units, without area/amplitude scaling."""
import argparse
import json
from pathlib import Path

import numpy as np
import ROOT
from reference import PROJECT, TIME_NS, bin_layout, wave_features
from diagnose_origin import sha
from compare import BLUE, RED, GREEN, KEEP, graph, label, panel, save


def aligned_mean(waves, baseline, t10, indices):
    total = np.zeros(len(TIME_NS)); used = []
    sample = np.arange(waves.shape[1])
    for i in indices:
        time = (sample-t10[i])*2.
        if not np.isfinite(t10[i]) or time[0] > TIME_NS[0] or time[-1] < TIME_NS[-1]: continue
        total += np.interp(TIME_NS, time, np.asarray(waves[i], dtype=float)-baseline[i]); used.append(int(i))
    if not used: raise ValueError('No fully supported waveforms')
    return total/len(used), used


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnosis', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True); args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    diagnosis = json.loads((args.diagnosis/'diagnosis.json').read_text())
    features = dict(np.load(args.diagnosis/'events.npz'))
    report = {'scope': 'Training-source mean signals in ADC units, not a common-charge comparison',
              'normalization': 'Subtract per-event baseline and align t10 only; no area or amplitude scaling',
              'selection': diagnosis['selection']+' Also require full [-200,3800] ns aligned support.',
              'warning': 'Different source, capsule and trigger assumptions affect the whole populations. Curves are not paired events. 120 ADC peak flag is not a PHA hardware trigger.',
              'diagnosis_sha256': sha(args.diagnosis/'diagnosis.json'),
              'diagnostic_arrays_sha256': sha(args.diagnosis/'events.npz'),
              'script_sha256': sha(Path(__file__)), 'sources': {}}
    arrays = {'time_ns': TIME_NS}
    ROOT.gROOT.SetBatch(True); ROOT.gStyle.SetOptStat(0); ROOT.gStyle.SetTitleFontSize(.04)
    canvas = ROOT.TCanvas('absolute_waveforms', 'Unscaled mean ADC waveform', 1370, 620)
    canvas.Divide(2, 1, .003, .003)
    for column, (source, run) in enumerate((('Co60', 3), ('Am241', 12))):
        path = Path(diagnosis['simulation'][source]['input'])
        if sha(path) != diagnosis['simulation'][source]['input_sha256']: raise ValueError('MC file changed')
        file = ROOT.TFile.Open(str(path)); tree = file.Get('Events'); tree.SetBranchStatus('*', 0)
        tree.SetBranchStatus('adc', 1); waves = np.array([np.array(event.adc, dtype=float) for event in tree]); file.Close()
        f = wave_features(waves); quality = features[source+'_quality']; peak = features[source+'_peak_adc_noiseless']
        curves = {}; indices = {}
        for name, mask in [('mc_quality', quality), ('mc_peak120', quality & (peak >= 120))]:
            curves[name], indices[name] = aligned_mean(waves, f['baseline'], f['t10_sample'], np.flatnonzero(mask))
        raw_path = PROJECT/f'data/DT5730_DPP_PHA/DAQ/run_{run}/UNFILTERED/Data_run_{run}.BIN'
        _, dtype, count = bin_layout(raw_path)
        raw = np.memmap(raw_path, dtype=dtype, offset=2, mode='r', shape=(count,))
        with np.load(PROJECT/f'analysis/paper_baseline/results/pha_run{run:04d}_features.npz') as z:
            f = {key: z[key] for key in ('baseline', 't10_sample')}
        with np.load(PROJECT/f'analysis/paper_baseline/results/phase2/run{run:04d}_features.npz') as z:
            selected = np.flatnonzero(z['eligible_primary'] & (z['split'] == 0))
        curves['measured'], indices['measured'] = aligned_mean(raw['data'], f['baseline'], f['t10_sample'], selected)
        del raw
        report['sources'][source] = {
            name: {'n': len(indices[name]), 'event_indices': indices[name], 'mean_waveform_peak_adc': float(curve.max())}
            for name, curve in curves.items()}
        for name, curve in curves.items(): arrays[source+'_'+name] = curve
        ymax = max(v.max() for v in curves.values())*1.5
        panel(canvas, column+1, f'{source} / run {run} training;Time from t_{{10}} (#mus);Baseline-subtracted signal (ADC)', -.2, -20, 3.8, ymax)
        for name, color, style in [('measured', BLUE, 1), ('mc_quality', GREEN, 2), ('mc_peak120', RED, 1)]:
            graph(TIME_NS/1000, curves[name], color, style)
        label(f"Measured training, n={len(indices['measured'])}", .45, .83, BLUE, .03)
        label(f"MC quality, n={len(indices['mc_quality'])}", .45, .78, GREEN, .03)
        label(f"MC + 120 ADC peak flag, n={len(indices['mc_peak120'])}", .45, .73, RED, .03)
        label('No common-Q cut; no amplitude scaling', .23, .88, size=.028)
        label('Peak flag #neq PHA trigger', .48, .66, size=.028)
    np.savez_compressed(args.out/'curves.npz', **arrays)
    (args.out/'absolute.json').write_text(json.dumps(report, indent=2)+'\n')
    save(canvas, args.out, 'absolute_waveforms')
    print(json.dumps({s: {k: {'n': v['n'], 'mean_peak_adc': v['mean_waveform_peak_adc']}
                         for k, v in r.items()} for s, r in report['sources'].items()}))


if __name__ == '__main__': main()
