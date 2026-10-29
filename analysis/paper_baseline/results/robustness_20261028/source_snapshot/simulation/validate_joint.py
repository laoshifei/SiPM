"""Evaluate a frozen joint candidate with independent full optical simulations."""
import argparse
import json
from pathlib import Path
import numpy as np
import ROOT
from reference import PROJECT, TIME_NS, bin_layout
from compare import mc_wave_features, BLUE, RED, GREEN, KEEP, panel, graph, label, save
from compare_absolute import aligned_mean
from screen_joint import ANALYSIS_AMPLITUDE_ADC, PROBABILITIES, sha

HERE = Path(__file__).resolve().parent
GRAY = ROOT.TColor.GetColor('#888888')


def read_simulation(path):
    file = ROOT.TFile.Open(str(path)); tree = file.Get('Events')
    info = json.loads(str(file.Get('RunInfo').GetString().Data()))
    if not info.get('completed') or tree.GetEntries() != info['events_requested']:
        raise ValueError('Refuse incomplete simulation')
    if not info.get('digitization_enabled', True): raise ValueError('Transport-only file has no ADC')
    tree.SetBranchStatus('*', 0); tree.SetBranchStatus('adc', 1)
    waves = np.array([np.array(event.adc, dtype=np.uint16) for event in tree]); file.Close()
    features, good = mc_wave_features(waves)
    mask = good & (features['amplitude'] >= ANALYSIS_AMPLITUDE_ADC)
    mean, used = aligned_mean(waves, features['baseline'], features['t10_sample'], np.flatnonzero(mask))
    del waves
    return {'q': features['charge_long'][mask], 'r': features['short_long_ratio'][mask],
            'mean_adc': mean, 'indices': np.flatnonzero(mask), 'mean_indices': used,
            'info': info, 'input': str(path.resolve()), 'sha256': sha(path)}


def measured(run, split):
    fpath = PROJECT/f'analysis/paper_baseline/results/pha_run{run:04d}_features.npz'
    apath = PROJECT/f'analysis/paper_baseline/results/phase2/run{run:04d}_features.npz'
    with np.load(fpath) as z: f = {k: z[k] for k in z.files}
    with np.load(apath) as z: a = {k: z[k] for k in z.files}
    mask = a['eligible_primary'] & (a['split'] == split) & (f['amplitude'] >= ANALYSIS_AMPLITUDE_ADC)
    result = {'q': f['charge_long'][mask], 'r': f['short_long_ratio'][mask], 'indices': np.flatnonzero(mask),
              'feature_sha256': sha(fpath), 'phase_sha256': sha(apath)}
    if split == 0:
        raw_path = PROJECT/f'data/DT5730_DPP_PHA/DAQ/run_{run}/UNFILTERED/Data_run_{run}.BIN'
        _, dtype, count = bin_layout(raw_path); raw = np.memmap(raw_path, dtype=dtype, offset=2, mode='r', shape=(count,))
        result['mean_adc'], result['mean_indices'] = aligned_mean(raw['data'], f['baseline'], f['t10_sample'], np.flatnonzero(mask))
        result['raw_sha256'] = sha(raw_path)
    return result


def summary(data):
    return {'n': len(data['q']), 'charge_quantiles': np.quantile(data['q'], PROBABILITIES).tolist(),
            'ratio_quantiles': np.quantile(data['r'], PROBABILITIES).tolist(), 'event_indices': np.asarray(data['indices']).tolist(),
            **({'mean_waveform_peak_adc': float(data['mean_adc'].max()), 'mean_event_count': len(data['mean_indices'])} if 'mean_adc' in data else {})}


def discrepancy(simulated, observed):
    q_sim = np.quantile(simulated['q'], PROBABILITIES); q_obs = np.quantile(observed['q'], PROBABILITIES)
    r_sim = np.quantile(simulated['r'], PROBABILITIES); r_obs = np.quantile(observed['r'], PROBABILITIES)
    def ks(x, y):
        points = np.sort(np.r_[x, y]); x = np.sort(x); y = np.sort(y)
        return float(np.max(np.abs(np.searchsorted(x, points, side='right')/len(x)-np.searchsorted(y, points, side='right')/len(y))))
    q_loss = float(np.mean((np.log(q_sim/q_obs)/.15)**2)); r_loss = float(np.mean(((r_sim-r_obs)/.025)**2))
    return {'charge_loss': q_loss, 'ratio_loss': r_loss, 'total_loss': q_loss+r_loss,
            'charge_ks_distance': ks(simulated['q'], observed['q']), 'ratio_ks_distance': ks(simulated['r'], observed['r']),
            'charge_quantile_relative_residual': (q_sim/q_obs-1).tolist(), 'ratio_quantile_residual': (r_sim-r_obs).tolist()}


def figures(out, data):
    ROOT.gROOT.SetBatch(True); ROOT.gStyle.SetOptStat(0); ROOT.gStyle.SetTitleFontSize(.043)
    ROOT.gStyle.SetLabelSize(.035, 'XYZ'); ROOT.gStyle.SetTitleSize(.039, 'XYZ')
    canvas = ROOT.TCanvas('joint_comparison', 'Joint candidate vs measured training data', 1400, 1400)
    canvas.Divide(2, 3, .003, .003)
    styles = [('measured', BLUE, 1, 'Measured training'), ('new', RED, 1, 'New full Geant4 + SiPM'),
              ('old', GRAY, 2, 'Previous full Geant4 + SiPM')]
    for col, source in enumerate(('Co60', 'Am241')):
        for row, key in enumerate(('q', 'r')):
            edges = np.linspace(0, 1200, 49) if key == 'q' else np.linspace(.15, .65, 51)
            curves = {}
            for name, _, _, _ in styles:
                values = data[source][name][key]/1000 if key == 'q' else data[source][name][key]
                curves[name] = np.histogram(values, edges)[0]/len(values)
            title = source+(';Charge (10^{3} ADC samples);Fraction per bin' if key == 'q' else ';Short / long charge;Fraction per bin')
            panel(canvas, 1+col+2*row, title, float(edges[0]), 0, float(edges[-1]), float(max(x.max() for x in curves.values())*1.5))
            for name, color, style, text in styles:
                graph((edges[1:]+edges[:-1])/2, curves[name], color, style)
                label(text, .48, .79-.055*[n for n, _, _, _ in styles].index(name), color, .031)
            label('Same analysis cut: amplitude #geq 200 ADC', .20, .855, size=.030)
        maximum = max(data[source][name]['mean_adc'].max() for name, _, _, _ in styles)
        panel(canvas, 5+col, source+';Time from t_{10} (#mus);Baseline-subtracted signal (ADC)', -.2, -20, 3.8, float(maximum*1.4))
        for i, (name, color, style, text) in enumerate(styles):
            graph(TIME_NS/1000, data[source][name]['mean_adc'], color, style)
            label(text+f", n={len(data[source][name]['mean_indices'])}", .40, .79-i*.055, color, .031)
        label('No area / amplitude rescaling', .20, .855, size=.030)
    save(canvas, out, 'joint_comparison')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--new', type=Path, required=True)
    parser.add_argument('--old', type=Path, default=HERE/'results/corrected_optics_20261027')
    parser.add_argument('--screen', type=Path, required=True)
    parser.add_argument('--design', type=Path, help='Optional frozen controlled-test design; screening parameters are then inherited, not refitted')
    parser.add_argument('--out', type=Path, required=True); args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    screen = json.loads((args.screen/'screen.json').read_text()); data = {}; report = {
        'scope': 'Full optical test of one frozen joint-screening candidate using new MC random seeds',
        'selection': 'Frozen experimental phase2 quality and split, plus amplitude >=200 ADC. MC basic quality, primary pretrigger veto and same amplitude cut. MC requires 3 us gate; experimental quality also requires 4 us support. Mean signals require full plotted support.',
        'trigger': 'The 200 ADC cut is an analysis condition, not the PHA trigger or a detection efficiency.',
        'normalization': 'No source-specific gain or post-hoc amplitude normalization. Histogram denominators include events outside plotted axis limits.',
        'metrics': 'Quantile discrepancy uses fixed practical scales from screening, not a likelihood; KS distances are descriptive, with no p-values.',
        'check_scope': 'Late groups and run18 were explored previously; not blind. They did not enter this joint fit.',
        'screen_sha256': sha(args.screen/'screen.json'), 'parameters': screen['best'], 'sources': {}, 'checks': {},
        'code_sha256': {str(p.relative_to(PROJECT)): sha(p) for p in (Path(__file__), HERE/'compare.py', HERE/'compare_absolute.py', HERE/'reference.py')}}
    design = json.loads(args.design.read_text()) if args.design else None
    if design:
        report['scope'] = design.get('comparison_scope', 'Controlled response-factorization test with new MC seeds; inherited geometry/normalization, no refit')
        report['frozen_design_sha256'] = sha(args.design)
        report['controlled_changes'] = design['sources']
        report['inherited_screen_parameters'] = report.pop('parameters')
    arrays = {'time_ns': TIME_NS}
    for source, name, run in [('Co60', 'co60', 3), ('Am241', 'am241', 12)]:
        data[source] = {'measured': measured(run, 0), 'old': read_simulation(args.old/name/'events.root'),
                        'new': read_simulation(args.new/name/'events.root')}
        observed = data[source]['measured']
        if design:
            if sha(args.new/name/'config.json') != design['sources'][source]['config_sha256']:
                raise ValueError('Candidate config changed after the control design was frozen')
        report['sources'][source] = {'measured': summary(observed), 'old': summary(data[source]['old']), 'new': summary(data[source]['new']),
                                    'old_discrepancy': discrepancy(data[source]['old'], observed),
                                    'new_discrepancy': discrepancy(data[source]['new'], observed),
                                    'input_sha256': {key: data[source][key]['sha256'] for key in ('old', 'new')},
                                    'raw_training_sha256': observed['raw_sha256']}
        report['sources'][source]['mc_run_info'] = {key: data[source][key]['info'] for key in ('old', 'new')}
        for key in ('old', 'new', 'measured'):
            for field in ('q', 'r', 'mean_adc'): arrays[source+'_'+key+'_'+field] = data[source][key][field]
        for check_run, split in ([(3, 2)] if source == 'Co60' else [(12, 2), (18, 3)]):
            check_data = measured(check_run, split)
            report['checks'][f'run{check_run}'] = {'observed': summary(check_data),
                'old_discrepancy': discrepancy(data[source]['old'], check_data), 'new_discrepancy': discrepancy(data[source]['new'], check_data),
                'feature_sha256': check_data['feature_sha256'], 'phase_sha256': check_data['phase_sha256']}
    report['training_joint_loss'] = {name: sum(report['sources'][s][name+'_discrepancy']['total_loss'] for s in data) for name in ('old', 'new')}
    np.savez_compressed(args.out/'curves.npz', **arrays)
    (args.out/'validation.json').write_text(json.dumps(report, indent=2)+'\n'); figures(args.out, data)
    print(json.dumps({'joint_loss': report['training_joint_loss'], 'sources': {s: {k: v for k, v in r.items() if k.endswith('discrepancy')} for s, r in report['sources'].items()}}, indent=2))


if __name__ == '__main__': main()
