"""Audit hardware flags and charge-dependent shapes without refitting the detector.

The frozen primary cohort remains intact. Bit-0x8000-clear and stricter subsets
are sensitivity comparisons, not experimentally certified single-particle truth.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import ROOT
from reference import PROJECT, TIME_NS, bin_layout, aligned_shapes
from compare import mc_wave_features, BLUE, RED, GREEN, KEEP, panel, graph, label, save
from compare_absolute import aligned_mean
from validate_joint import discrepancy

HERE = Path(__file__).resolve().parent
BIT = np.uint32(0x8000)
ORANGE = ROOT.TColor.GetColor('#B56B22')
GRAY = ROOT.TColor.GetColor('#777777')
QUANTILES = [.1, .25, .5, .75, .9]
NAMES = ('low', 'core', 'upper', 'high')
MAX_SHAPES = 600


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''): digest.update(block)
    return digest.hexdigest()


def hardware_flag(flags): return (np.asarray(flags, dtype=np.uint32) & BIT) != 0


def charge_bands(q, cuts):
    q = np.asarray(q)
    if len(cuts) != 3 or not np.all(np.diff(cuts) > 0): raise ValueError('Three ordered charge boundaries required')
    bands = np.searchsorted(cuts, q, side='right')
    return np.where(np.isfinite(q) & (q > 0), bands, -1)


def neighboring_intervals_us(timestamps):
    t = np.asarray(timestamps, dtype=np.uint64)
    if np.any(t[1:] < t[:-1]): raise ValueError('Timestamp backstep; do not infer an inter-event interval')
    delta = (t[1:] - t[:-1]).astype(float)/1e6
    return np.r_[np.nan, delta], np.r_[delta, np.nan]


def quantiles(x):
    x = np.asarray(x); x = x[np.isfinite(x)]
    return np.quantile(x, QUANTILES).tolist() if len(x) else []


def statistics(data, mask):
    indices = np.flatnonzero(mask).astype('<i8')
    result = {'n': int(mask.sum()), 'event_indices_sha256_le_int64': hashlib.sha256(indices.tobytes()).hexdigest()}
    for name in ('q', 'r', 'amplitude', 'baseline_rms', 'peak_delay_ns', 'q2_over_q3',
                 'q4_over_q3', 'recent_baseline_delta', 'before_us', 'after_us'):
        if name in data: result[name+'_quantiles'] = quantiles(data[name][mask])
    if mask.any():
        for name in ('flag_0x8000', 'energy_zero', 'secondary_rise'):
            if name in data: result[name+'_fraction'] = float(np.mean(data[name][mask]))
        if 'flags' in data:
            codes, counts = np.unique(data['flags'][mask], return_counts=True)
            result['hardware_flag_counts'] = {hex(int(k)): int(v) for k, v in zip(codes, counts)}
    if 'origin' in data:
        result['simulated_dominant_charge_origin_counts'] = {str(k): int(np.sum(mask & (data['origin'] == k))) for k in range(6)}
        result['edep_MeV_quantiles'] = quantiles(data['edep'][mask])
    return result


def load_measured(run, split):
    path = PROJECT/f'analysis/paper_baseline/results/pha_run{run:04d}_features.npz'
    phase = PROJECT/f'analysis/paper_baseline/results/phase2/run{run:04d}_features.npz'
    with np.load(path) as z: f = dict(z)
    with np.load(phase) as z: a = dict(z)
    before, after = neighboring_intervals_us(f['timestamp'])
    data = {'q': f['charge_long'], 'r': f['short_long_ratio'], 'amplitude': f['amplitude'],
            'baseline_rms': f['baseline_rms'], 'peak_delay_ns': 2*(f['peak_sample']-f['t10_sample']),
            'q2_over_q3': f['charge_2us']/f['charge_long'], 'q4_over_q3': a['long_charge'][:, 3]/f['charge_long'],
            'recent_baseline_delta': a['recent_baseline_delta'], 'flag_0x8000': hardware_flag(f['flags']),
            'energy_zero': f['energy_ch'] == 0, 'flags': f['flags'], 'secondary_rise': a['secondary_rise_flag'],
            'before_us': before, 'after_us': after, 'f': f,
            'selected': a['eligible_primary'] & (a['split'] == split) & (f['amplitude'] >= 200),
            'strict': a['eligible_strict'] & (a['split'] == split) & (f['amplitude'] >= 200) & ~a['secondary_rise_flag'],
            'input_sha256': {str(p.relative_to(PROJECT)): sha(p) for p in (path, phase)}}
    return data


def load_simulation(path):
    file = ROOT.TFile.Open(str(path))
    if not file or file.IsZombie(): raise ValueError(f'Cannot open {path}')
    info = json.loads(str(file.Get('RunInfo').GetString().Data())); tree = file.Get('Events')
    if not info.get('completed') or tree.GetEntries() != info['events_requested']: raise ValueError('Incomplete MC')
    if not info.get('digitization_enabled', True): raise ValueError('Transport-only MC has no waveform')
    tree.SetBranchStatus('*', 0)
    for name in ('adc', 'edep_MeV', 'charge_by_origin_C', 'edep_origin_species_MeV'): tree.SetBranchStatus(name, 1)
    waves, energy, charge, deposits = [], [], [], []
    for event in tree:
        waves.append(np.array(event.adc, dtype=np.uint16)); energy.append(float(event.edep_MeV))
        charge.append(np.array(event.charge_by_origin_C)); deposits.append(np.array(event.edep_origin_species_MeV))
    file.Close(); waves = np.array(waves); charge = np.array(charge)
    f, good = mc_wave_features(waves)
    total = charge.sum(axis=1); fractions = charge/np.maximum(total[:, None], 1e-300)
    origin = np.where(fractions.max(axis=1) >= .9, fractions.argmax(axis=1), 0)
    data = {'q': f['charge_long'], 'r': f['short_long_ratio'], 'amplitude': f['amplitude'],
            'baseline_rms': f['baseline_rms'], 'peak_delay_ns': 2*(f['peak_sample']-f['t10_sample']),
            'q2_over_q3': f['charge_2us']/f['charge_long'], 'selected': good & (f['amplitude'] >= 200),
            'origin': origin, 'edep': np.array(energy), 'deposits': np.array(deposits).reshape(-1, 6, 3),
            'waves': waves, 'f': f, 'info': info, 'input_sha256': {str(path.relative_to(PROJECT)): sha(path)}}
    return data


def subset_masks(data):
    primary = data['selected']; bit = data['flag_0x8000']
    return {'primary': primary, 'bit_clear': primary & ~bit, 'bit_set': primary & bit,
            'strict_no_secondary': data['strict'], 'strict_bit_clear': data['strict'] & ~bit,
            'flags_zero': primary & (data['flags'] == 0)}


def shape_summary(waves, features, mask):
    ids = np.flatnonzero(mask)
    if len(ids) > MAX_SHAPES: ids = ids[np.linspace(0, len(ids)-1, MAX_SHAPES, dtype=int)]
    if not len(ids): return {'n_available': 0, 'n_used': 0, 'indices': []}, None
    shapes, valid = aligned_shapes(waves[ids], {k: v[ids] for k, v in features.items()})
    ids = ids[valid]
    if not len(ids): return {'n_available': int(mask.sum()), 'n_used': 0, 'indices': []}, None
    return {'n_available': int(mask.sum()), 'n_used': len(ids), 'indices': ids.tolist()}, shapes.mean(axis=0)


def inspect_raw(run, data, cuts, arrays):
    raw_path = PROJECT/f'data/DT5730_DPP_PHA/DAQ/run_{run}/UNFILTERED/Data_run_{run}.BIN'
    _, dtype, n = bin_layout(raw_path)
    raw = np.memmap(raw_path, dtype=dtype, offset=2, mode='r', shape=(n,)); f = data['f']
    if n != len(data['q']): raise ValueError('Raw/feature event-count mismatch')
    masks = subset_masks(data); bands = charge_bands(data['q'], cuts)
    result = {'raw_sha256': sha(raw_path), 'means': {}, 'examples': []}
    # Full training means remain exact; run18 is used for sampled shapes only.
    if run != 18:
        for name in ('primary', 'bit_clear', 'bit_set'):
            mean, used = aligned_mean(raw['data'], f['baseline'], f['t10_sample'], np.flatnonzero(masks[name]))
            arrays[f'run{run}_{name}_absolute_mean'] = mean
            result['means'][name] = {'n': len(used), 'peak_adc': float(mean.max()), 'indices': [int(i) for i in used]}
    for band, name in enumerate(NAMES):
        for category in ('primary', 'bit_clear', 'bit_set'):
            mask = masks[category] & (bands == band)
            record, mean = shape_summary(raw['data'], f, mask)
            result[f'{name}_{category}'] = record
            if mean is not None: arrays[f'run{run}_{name}_{category}_shape'] = mean
    # Deterministic examples: medians in charge and waveform-shape outliers,
    # retaining both categories rather than hand-picking smooth pulses.
    for band, category in [(1, 'bit_clear'), (3, 'bit_clear'), (3, 'bit_set')]:
        ids = np.flatnonzero(masks[category] & (bands == band))
        if not len(ids): continue
        order = ids[np.argsort(data['q'][ids], kind='stable')]
        for fraction in (.25, .75):
            i = int(order[int((len(order)-1)*fraction)])
            y = raw['data'][i].astype(float)-f['baseline'][i]
            arrays[f'run{run}_example_{i}_adc'] = y
            arrays[f'run{run}_example_{i}_time_ns'] = (np.arange(len(y))-f['t10_sample'][i])*2
            result['examples'].append({'index': i, 'group': NAMES[band]+'_'+category, 'charge_order_fraction': fraction,
                'flags': hex(int(f['flags'][i])), 'energy_ch': int(f['energy_ch'][i]),
                'q': float(data['q'][i]), 'r': float(data['r'][i]), 'secondary_rise': bool(data['secondary_rise'][i])})
    return result


def plots(out, report, data, simulations, arrays):
    ROOT.gROOT.SetBatch(True); ROOT.gStyle.SetOptStat(0); ROOT.gStyle.SetTitleFontSize(.043)
    ROOT.gStyle.SetLabelSize(.034, 'XYZ'); ROOT.gStyle.SetTitleSize(.039, 'XYZ')
    canvas = ROOT.TCanvas('charge_flags', 'Hardware-flag sensitivity without refitting', 1550, 1000); canvas.Divide(3, 2, .003, .003)
    edges = np.linspace(0, 1600, 65)
    for column, (run, source) in enumerate(((3, 'Co60'), (12, 'Am241'), (18, 'Am241'))):
        observed = data[run]; mc = simulations[source]; masks = subset_masks(observed)
        curves = [(np.histogram(observed['q'][masks[k]]/1000, edges)[0]/max(1, masks[k].sum()), color, style, text)
                  for k, color, style, text in [('primary', BLUE, 1, 'Original cohort'), ('bit_clear', GREEN, 1, '0x8000 clear'), ('bit_set', ORANGE, 2, '0x8000 set')]]
        curves.append((np.histogram(mc['q'][mc['selected']]/1000, edges)[0]/mc['selected'].sum(), RED, 1, 'Frozen full MC'))
        panel(canvas, column+1, f'Run {run} / {source};Charge (10^{{3}} ADC samples);Fraction per bin', 0, 0, 1600, max(v.max() for v, _, _, _ in curves)*1.45)
        for j, (values, color, style, text) in enumerate(curves):
            graph((edges[1:]+edges[:-1])/2, values, color, style); label(text, .53, .83-j*.05, color, .032)
        label('Each subset normalized; no amplitude scaling', .17, .88, size=.026)
        selected = observed['selected']; bit = observed['flag_0x8000']
        total = np.histogram(observed['q'][selected]/1000, edges)[0]
        flagged = np.histogram(observed['q'][selected & bit]/1000, edges)[0]
        use = total >= 10
        fraction = flagged[use]/total[use]; n = total[use]; z = 1.96
        center = (fraction+z*z/(2*n))/(1+z*z/n)
        half = z*np.sqrt(fraction*(1-fraction)/n+z*z/(4*n*n))/(1+z*z/n)
        panel(canvas, column+4, 'Hardware flag association;Charge (10^{3} ADC samples);Fraction with 0x8000', 0, 0, 1600, 1.)
        x = ((edges[1:]+edges[:-1])/2)[use]
        g = ROOT.TGraphAsymmErrors(len(x), x.astype(float), fraction.astype(float), np.zeros(len(x)), np.zeros(len(x)),
                                  np.maximum(0, fraction-(center-half)), np.maximum(0, center+half-fraction))
        g.SetMarkerStyle(20); g.SetMarkerSize(.6); g.SetMarkerColor(ORANGE); g.SetLineColor(ORANGE); g.Draw('P SAME'); KEEP.append(g)
        label('95% Wilson intervals; bins with n #geq 10', .18, .86, size=.028)
        label('Flag state is not particle truth', .18, .80, size=.028)
    save(canvas, out, 'charge_flag_comparison')

    canvas = ROOT.TCanvas('charge_shapes', 'Charge-conditioned shapes', 1500, 1000); canvas.Divide(3, 2, .003, .003)
    for column, (run, source) in enumerate(((3, 'Co60'), (12, 'Am241'), (18, 'Am241'))):
        groups = [('core_bit_clear', BLUE, 1, 'Core Q, bit clear'), ('high_bit_clear', GREEN, 1, 'High Q, bit clear'), ('high_bit_set', ORANGE, 2, 'High Q, bit set')]
        curves = [(arrays[f'run{run}_{key}_shape'], color, style, text) for key, color, style, text in groups if f'run{run}_{key}_shape' in arrays]
        mc_key = f'{source}_core_shape'
        if mc_key in arrays: curves.append((arrays[mc_key], RED, 1, 'Frozen MC, core Q'))
        panel(canvas, column+1, f'Run {run} / {source};Time from t_{{10}} (#mus);Area-normalized signal (#mus^{{-1}})', -.2, 0, 3.8, max(v.max() for v, _, _, _ in curves)*1400)
        for j, (values, color, style, text) in enumerate(curves):
            graph(TIME_NS/1000, values*1000, color, style); label(text, .49, .84-j*.055, color, .029)
        core = arrays[f'run{run}_core_bit_clear_shape']
        panel(canvas, column+4, 'Difference from measured core;Time from t_{10} (#mus);Residual / core peak (%)', -.2, -15, 3.8, 15)
        for values, color, style, text in curves[1:]: graph(TIME_NS/1000, (values-core)/core.max()*100, color, style)
        label('Core: training Q25-Q75; high: above Q90', .18, .30, size=.027)
        label('Q boundaries fixed from original training cohort', .18, .24, size=.024)
    save(canvas, out, 'charge_conditioned_shapes')

    canvas = ROOT.TCanvas('examples', 'Deterministic run12 examples', 1400, 1100); canvas.Divide(2, 3, .003, .003)
    for slot, example in enumerate(report['raw']['run12']['examples']):
        i = example['index']; y = arrays[f'run12_example_{i}_adc']; t = arrays[f'run12_example_{i}_time_ns']
        panel(canvas, slot+1, f"Run12 event {i} / {example['group']};Time from t_{{10}} (#mus);Signal (ADC)", -.75, -30, 4., float(y.max()*1.35))
        graph(t/1000, y, ORANGE if example['flags'] != '0x0' else BLUE)
        label(f"flag={example['flags']}; PHA E={example['energy_ch']}; Q={example['q']/1000:.1f}k", .39, .84, size=.029)
        label(f"R={example['r']:.3f}; secondary rise={example['secondary_rise']}", .39, .77, size=.029)
    save(canvas, out, 'run12_examples')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mc', type=Path, default=HERE/'results/joint_optical_20261028')
    parser.add_argument('--out', type=Path, required=True); args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    data = {run: load_measured(run, 3 if run == 18 else 0) for run in (3, 12, 18)}
    cuts = {source: np.quantile(data[run]['q'][data[run]['selected']], [.25, .75, .9]) for source, run in [('Co60', 3), ('Am241', 12)]}
    report = {'scope': __doc__, 'analysis_amplitude_adc': 200,
        'selection': 'Original primary quality and first60% of run3/run12; run18 already-explored check. MC frozen candidate quality + 200 ADC, no refit.',
        'flag_definition_evidence': {'bit': '0x8000', 'manual': 'CAEN UM5960 CoMPASS rev18, p68',
            'url': 'https://seltokphotonics.com/upload/iblock/28b/28bd53047bcb1dc23f8fe78548ec7021.pdf',
            'indexed_table_text': '0x8000 Piled-up event', 'checked': '2026-10-28',
            'verification_scope': 'Primary-manual indexed text only. Direct mirror PDF returns HTTP404; official legacy download redirects to HTML. Not a full-PDF verification.',
            'interpretation': 'Recorded hardware tag; no proof of two physical pulses or a perfect single-event selector. Other bits are retained, not rejected collectively.'},
        'shape_sampling': f'Deterministic evenly spaced original indices, at most {MAX_SHAPES} per group; normalized in [-40,3000)ns. Full training absolute means use all supported selected events.',
        'charge_bands': {source: {'Q25_Q75_Q90': x.tolist(), 'names': NAMES, 'rule': 'Positive finite Q; half-open intervals, boundary goes to higher bin'} for source, x in cuts.items()},
        'timestamp_scope': 'Intervals among saved waveforms only, not all hardware triggers or the source rate.',
        'origin_definition': 'MC origin has >=90% of in-window avalanche charge; 0 includes mixed/unknown. 1 alpha family, 2 gamma family, 3 decay electrons, 4 recoil, 5 other.',
        'quantile_probabilities': QUANTILES, 'measured': {}, 'simulation': {}, 'raw': {}, 'comparisons': {},
        'code_sha256': {str(p.relative_to(PROJECT)): sha(p) for p in [Path(__file__), HERE/'compare.py', HERE/'reference.py', HERE/'compare_absolute.py', HERE/'validate_joint.py', PROJECT/'analysis/paper_baseline/prepare.py', PROJECT/'analysis/paper_baseline/refine_waveforms.py']}}
    arrays = {'time_ns': TIME_NS}; simulations = {}; rows = []
    for source, name in [('Co60', 'co60'), ('Am241', 'am241')]:
        mc = simulations[source] = load_simulation(args.mc/name/'events.root')
        if mc['info']['source'] != source: raise ValueError('Wrong source file')
        bands = charge_bands(mc['q'], cuts[source]); stats = {'selected': statistics(mc, mc['selected']), 'input_sha256': mc['input_sha256'], 'bands': {}}
        for i, band in enumerate(NAMES):
            mask = mc['selected'] & (bands == i); stats['bands'][band] = statistics(mc, mask)
            record, mean = shape_summary(mc['waves'], mc['f'], mask); stats['bands'][band]['shape'] = record
            if mean is not None: arrays[source+'_'+band+'_shape'] = mean
        report['simulation'][source] = stats
        for key in ('q', 'r', 'selected', 'origin', 'edep'): arrays[source+'_'+key] = mc[key]
        del mc['waves']
        print(source, 'MC selected', stats['selected']['n'], flush=True)
    for run, source in [(3, 'Co60'), (12, 'Am241'), (18, 'Am241')]:
        measured = data[run]; masks = subset_masks(measured); bands = charge_bands(measured['q'], cuts[source]); mc = simulations[source]
        stats = {'input_sha256': measured['input_sha256'], 'subsets': {}, 'bands': {}}
        for category, mask in masks.items():
            stats['subsets'][category] = statistics(measured, mask)
            obs = {key: measured[key][mask] for key in ('q', 'r')}
            sim = {key: mc[key][mc['selected']] for key in ('q', 'r')}
            report['comparisons'][f'run{run}_{category}'] = discrepancy(sim, obs)
        for i, band in enumerate(NAMES):
            stats['bands'][band] = {}
            for category in ('primary', 'bit_clear', 'bit_set'):
                mask = masks[category] & (bands == i); record = statistics(measured, mask)
                stats['bands'][band][category] = record
                rows.append({'run': run, 'source': source, 'band': band, 'subset': category, 'n': record['n'],
                             'q_median': record['q_quantiles'][2] if record['n'] else '',
                             'ratio_median': record['r_quantiles'][2] if record['n'] else '',
                             'flag_fraction': record.get('flag_0x8000_fraction', ''), 'secondary_rise_fraction': record.get('secondary_rise_fraction', '')})
        stats['energy_zero_bit_disagreement_primary'] = int(np.sum(measured['selected'] & (measured['energy_zero'] != measured['flag_0x8000'])))
        report['measured'][f'run{run}'] = stats
        report['raw'][f'run{run}'] = inspect_raw(run, measured, cuts[source], arrays)
        for key in ('q', 'r', 'selected', 'flag_0x8000'): arrays[f'run{run}_{key}'] = measured[key]
        for name, mask in masks.items(): arrays[f'run{run}_{name}_mask'] = mask
        print('run', run, 'primary/clear/set', *(stats['subsets'][k]['n'] for k in ('primary', 'bit_clear', 'bit_set')), flush=True)
    np.savez_compressed(args.out/'curves.npz', **arrays)
    (args.out/'diagnostics.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    with (args.out/'charge_bins.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    plots(args.out, report, data, simulations, arrays)
    print('Completed', args.out, flush=True)


if __name__ == '__main__': main()
