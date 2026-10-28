"""Diagnose charge spectra by simulated decay ancestry; never label measured particles."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import ROOT
from reference import PROJECT, wave_features
from refine_waveforms import advanced_features

HERE = Path(__file__).resolve().parent
NAMES = ('alpha_family', 'gamma_family', 'decay_electron_family', 'mixed_or_other')
LABELS = ('#alpha family', '#gamma family', 'Decay e^{-} family', 'Mixed / other')
COLORS = [ROOT.TColor.GetColor(s) for s in ('#D3876C', '#7EB5D4', '#B5BE7C', '#C6BBD4')]
KEEP = []


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def statistics(q, ratio, mask):
    n = int(mask.sum())
    return {'n': n, 'charge_p10_p50_p90_adc_samples': np.quantile(q[mask], [.1, .5, .9]).tolist() if n else [],
            'short_long_ratio_p10_p50_p90': np.quantile(ratio[mask], [.1, .5, .9]).tolist() if n else []}


def load_simulation(path):
    file = ROOT.TFile.Open(str(path))
    if not file or file.IsZombie(): raise ValueError(f'Cannot read {path}')
    info = json.loads(str(file.Get('RunInfo').GetString().Data())); tree = file.Get('Events')
    if not info.get('completed') or tree.GetEntries() != info['events_requested']:
        raise ValueError('Simulation is incomplete')
    if info['schema_version'] < 4: raise ValueError('Use corrected optical-boundary model (schema >=4)')
    tree.SetBranchStatus('*', 0)
    for name in ('adc', 'event_id', 'charge_by_origin_C', 'signal_mV', 'avalanche_origin',
                 'avalanche_emitter_pdg', 'avalanche_q_C', 'photon_boundary_status'):
        tree.SetBranchStatus(name, 1)
    waves = []; charges = []; alpha_e = []; peak = []; ids = []; boundary = np.zeros(15, dtype=np.int64)
    scale = (info['config']['sipm']['readout']['adc_max_mV']-info['config']['sipm']['readout']['adc_min_mV'])/16384
    for event in tree:
        waves.append(np.array(event.adc, dtype=float)); charges.append(np.array(event.charge_by_origin_C))
        origin = np.array(event.avalanche_origin); emitter = np.array(event.avalanche_emitter_pdg)
        q = np.array(event.avalanche_q_C)
        alpha_e.append(float(q[(origin == 1) & (np.abs(emitter) == 11)].sum()))
        peak.append(float(np.max(np.array(event.signal_mV))/scale)); ids.append(int(event.event_id))
        status = np.array(event.photon_boundary_status, dtype=int)
        if not np.all(np.isin(status, [1, 2, 12])): raise ValueError('Non-transmitted optical hit')
        boundary += np.bincount(status, minlength=15)
    file.Close()
    waves = np.array(waves); charges = np.array(charges); alpha_e = np.array(alpha_e); peak = np.array(peak)
    features = wave_features(waves)
    advanced = advanced_features(waves, features['t10_sample'], features['peak_sample'], features['amplitude'],
                                 features['baseline'], float(np.median(features['baseline_rms'])))
    good = (features['diagnostic_flags'] == 0) & ~advanced['pretrigger_flag_primary']
    total = charges.sum(axis=1); fractions = charges/np.maximum(total[:, None], 1e-300)
    category = np.full(len(waves), 3, dtype=int)
    for i, origin in enumerate((1, 2, 3)): category[fractions[:, origin] >= .9] = i
    result = {'source': info['source'], 'input': str(path.resolve()), 'input_sha256': sha(path),
              'run_info': info, 'events': len(waves), 'transmitted_boundary_counts': boundary.tolist(),
              'quality_subsets': {}, 'peak_flag_sensitivity': {}}
    for name, mask in [('quality', good), ('quality_and_peak120', good & (peak >= 120))]:
        result['quality_subsets'][name] = statistics(features['charge_long'], features['short_long_ratio'], mask)
        result['quality_subsets'][name]['by_dominant_charge_ancestry'] = {
            name: statistics(features['charge_long'], features['short_long_ratio'], mask & (category == i))
            for i, name in enumerate(NAMES)}
    for threshold in (80, 120, 160, 200):
        result['peak_flag_sensitivity'][str(threshold)] = statistics(
            features['charge_long'], features['short_long_ratio'], good & (peak >= threshold))
    result['all_event_avalanche_charge_fractions'] = (charges.sum(axis=0)/max(total.sum(), 1e-300)).tolist()
    result['alpha_family_charge_from_secondary_electron_fraction'] = (
        float(alpha_e.sum()/charges[:, 1].sum()) if charges[:, 1].sum() else None)
    arrays = {'event_id': np.array(ids), 'charge': features['charge_long'], 'ratio': features['short_long_ratio'],
              'quality': good, 'peak_adc_noiseless': peak, 'dominant_category': category,
              'charge_by_origin_C': charges, 'alpha_family_electron_charge_C': alpha_e}
    return result, arrays


def load_measured(run):
    feature_path = PROJECT/f'analysis/paper_baseline/results/pha_run{run:04d}_features.npz'
    phase_path = PROJECT/f'analysis/paper_baseline/results/phase2/run{run:04d}_features.npz'
    with np.load(feature_path) as z: f = {k: z[k] for k in z.files}
    with np.load(phase_path) as z: a = {k: z[k] for k in z.files}
    mask = a['eligible_primary'] & (a['split'] == 0)
    result = statistics(f['charge_long'], f['short_long_ratio'], mask)
    result.update({'run': run, 'source': 'Co60' if run == 3 else 'Am241',
                   'role': 'Previously explored training portion (first 60%); no particle truth assigned',
                   'features_sha256': sha(feature_path), 'phase2_sha256': sha(phase_path)})
    return result, f['charge_long'][mask]


def plot_charge(out, simulations, measured, reports):
    ROOT.gROOT.SetBatch(True); ROOT.gStyle.SetOptStat(0); ROOT.gStyle.SetTitleFontSize(.043)
    c = ROOT.TCanvas('origin_charge', 'Simulated charge ancestry diagnostic', 1420, 1050)
    c.Divide(2, 2, .003, .003)
    edges = np.linspace(0., 1200., 49)
    for col, source in enumerate(('Co60', 'Am241')):
        data = simulations[source]; observed = measured[source]
        for row in range(2):
            pad = c.cd(1+col+2*row)
            pad.SetLeftMargin(.14); pad.SetBottomMargin(.14); pad.SetTopMargin(.09); pad.SetRightMargin(.04)
            mask = data['quality'] & ((data['peak_adc_noiseless'] >= 120) if row else True)
            n = int(mask.sum()); title = source + (' / quality' if row == 0 else ' / quality + 120 ADC peak flag')
            stack = ROOT.THStack(f'stack{row}{col}', title+';Charge (10^{3} ADC samples);Fraction per bin')
            legend = ROOT.TLegend(.58, .56, .94, .87); legend.SetBorderSize(0); legend.SetFillStyle(0); legend.SetTextSize(.031)
            total = np.zeros(48)
            for category, (label, color) in enumerate(zip(LABELS, COLORS)):
                hist = ROOT.TH1D(f'origin{row}{col}{category}', '', 48, edges)
                values = data['charge'][mask & (data['dominant_category'] == category)]/1000
                count = np.histogram(values, edges)[0]/max(1, n); total += count
                for i, value in enumerate(count): hist.SetBinContent(i+1, value)
                hist.SetFillColor(color); hist.SetLineColor(ROOT.kWhite); stack.Add(hist)
                legend.AddEntry(hist, label, 'f'); KEEP.append(hist)
            obs = ROOT.TH1D(f'observed{row}{col}', '', 48, edges)
            counts = np.histogram(observed/1000, edges)[0]/max(1, len(observed))
            for i, value in enumerate(counts): obs.SetBinContent(i+1, value)
            obs.SetLineColor(ROOT.TColor.GetColor('#173E63')); obs.SetLineWidth(3); obs.SetFillStyle(0)
            stack.Draw('HIST'); stack.SetMaximum(float(max(total.max(), counts.max())*1.45)); stack.SetMinimum(0.)
            stack.GetYaxis().SetTitleOffset(1.45); stack.GetXaxis().SetTitleOffset(1.05)
            obs.Draw('HIST SAME'); legend.AddEntry(obs, f'Measured run {3 if col == 0 else 12} training', 'l'); legend.Draw()
            t = ROOT.TLatex(.18, .845, f'MC n={n}; measured n={len(observed)}')
            t.SetNDC(); t.SetTextSize(.031); t.SetTextFont(42); t.Draw()
            note = ROOT.TLatex(.18, .785, 'No amplitude rescaling')
            note.SetNDC(); note.SetTextSize(.029); note.SetTextFont(42); note.Draw()
            if row:
                note2 = ROOT.TLatex(.18, .725, 'Peak flag #neq PHA trigger')
                note2.SetNDC(); note2.SetTextSize(.029); note2.SetTextFont(42); note2.Draw(); KEEP.append(note2)
            KEEP.extend([stack, legend, obs, t, note])
    c.Modified(); c.Update()
    for extension in ('svg', 'png'): c.SaveAs(str(out/f'charge_by_origin.{extension}'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--co', type=Path, required=True); parser.add_argument('--am', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True); args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    report = {'scope': 'Source/response diagnostics, not a fitted spectrum or experimental particle identification',
              'selection': 'MC basic quality + primary pretrigger veto, complete 3 us integral. Measured phase2 eligible_primary also requires 4 us support. No common-charge or ratio cut in these spectra.',
              'origin_definition': 'At least 90% of realized in-window avalanche charge from one decay ancestry; otherwise mixed/other. This is not a classification of measured events.',
              'electron_definition': 'Decay electrons include beta, conversion and atomic relaxation; secondary electrons descending from alpha/gamma stay in the respective family.',
              'trigger_definition': '80/120/160/200 ADC noiseless peak flags are sensitivity scenarios, not an emulation of the hardware trigger or confirmed run-specific thresholds.',
              'normalization': 'Counts divided by total selected events, including out-of-plot-range events. Raw ADC charge, no source-dependent scale.',
              'scripts_sha256': {str(p.relative_to(PROJECT)): sha(p) for p in
                                 (Path(__file__), HERE/'reference.py', PROJECT/'analysis/paper_baseline/prepare.py',
                                  PROJECT/'analysis/paper_baseline/refine_waveforms.py')},
              'simulation': {}, 'measured': {}}
    simulations = {}; observed = {}; arrays = {}
    for source, run, path in [('Co60', 3, args.co), ('Am241', 12, args.am)]:
        report['simulation'][source], simulations[source] = load_simulation(path)
        if report['simulation'][source]['source'] != source: raise ValueError('Wrong source file')
        report['measured'][source], observed[source] = load_measured(run)
        for key, value in simulations[source].items(): arrays[source+'_'+key] = value
    (args.out/'diagnosis.json').write_text(json.dumps(report, indent=2)+'\n')
    np.savez_compressed(args.out/'events.npz', **arrays)
    plot_charge(args.out, simulations, observed, report)
    print(json.dumps({source: {'quality': value['quality_subsets']['quality']['n'],
                              'quality_and_peak120': value['quality_subsets']['quality_and_peak120']['n'],
                              'alpha_secondary_electron_charge_fraction': value['alpha_family_charge_from_secondary_electron_fraction']}
                      for source, value in report['simulation'].items()}))


if __name__ == '__main__': main()
