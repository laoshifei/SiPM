"""Charge/shape-conditioned local pulse covariance in recorded and native MC data.

The across-event covariance removes reproducible filtered mean structure using
an unbiased U-statistic. It is not an identification of photon shot noise:
shape mixing, timing, nonstationarity and selection can remain in the covariance.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import ROOT
from reference import PROJECT, bin_layout, wave_features
from diagnose_charge import sha
from diagnose_fluctuations import WINDOWS, load_mc
from compare import BLUE, RED, GREEN, KEEP, panel, graph, label, save

HERE = Path(__file__).resolve().parent
RUNS = (3, 7, 10, 11, 12, 13, 14, 18)
GROUPS = ('first60', 'middle20', 'last20')
OFFSETS = np.array([-300, 0, 250, 600, 1100])
LAGS = np.array([0, 1, 2, 3, 5, 8, 12, 16, 24, 32, 48])
N = 200


def covariance(x, lags=LAGS):
    """Window-centred, pair-normalized covariance; identical window conventions."""
    x = np.asarray(x, float)
    x = x - x.mean(axis=-1, keepdims=True)
    return np.stack([(x[..., :x.shape[-1]-k] * x[..., k:]).mean(axis=-1) for k in lags], axis=-1)


def covariance_parts(z, lags=LAGS):
    """raw = reproducible cross-event mean + unbiased across-event covariance.

    All rows have the same temporal support. Algebra is exact; unbiasedness as
    a population estimate additionally requires independent exchangeable events.
    """
    z = np.asarray(z, float)
    if len(z) < 2:
        raise ValueError('At least two events are required')
    z = z-z.mean(axis=-1, keepdims=True)
    raw = covariance(z, lags).mean(axis=0)
    centered = z-z.mean(axis=0)
    contribution = covariance(centered, lags)*len(z)/(len(z)-1)
    stochastic = contribution.mean(axis=0)
    return raw, raw-stochastic, stochastic, contribution


def extract_windows(waves, baseline, t10):
    waves = np.asarray(waves, float)-np.asarray(baseline)[:, None]
    anchors = np.ceil(t10).astype(int)
    positions = anchors[:, None, None]+OFFSETS[None, :, None]+np.arange(N)
    margin = max(WINDOWS)//2
    if positions.min() < margin or positions.max() >= waves.shape[1]-margin:
        raise ValueError('Incomplete filtered window support')
    local = np.take_along_axis(waves[:, None, :], positions, axis=2)
    means = local[:, 1:].mean(axis=2)
    if np.any(means <= 0):
        raise ValueError('Non-positive local pulse mean')
    cumulative = np.c_[np.zeros(len(waves)), np.cumsum(waves, axis=1)]
    filtered = []
    for width in WINDOWS:
        h = width//2
        average = (np.take_along_axis(cumulative[:, None, :], positions+h+1, axis=2)
                   - np.take_along_axis(cumulative[:, None, :], positions-h, axis=2))/width
        residual = local-average
        filtered.append(residual-residual.mean(axis=2, keepdims=True))
    return means, np.stack(filtered, axis=1)


def charge_bins(q, edges):
    q = np.asarray(q)
    if np.any((q < edges[0]) | (q > edges[-1])):
        raise ValueError('Charge outside frozen interval')
    return np.minimum(np.searchsorted(edges, q, side='right')-1, len(edges)-2)


def analyze(waves, baseline, t10, q, ratio, ids, edges):
    means, residual = extract_windows(waves, baseline, t10)
    cells = 2*charge_bins(q, edges)+(np.asarray(ratio) >= .4)
    shape_control = (ratio >= .30) & (ratio < .35)
    output = {'n': len(waves), 'shape_control_n': int(shape_control.sum()), 'views': {}}
    arrays = {'ids': np.asarray(ids), 'q': q, 'ratio': ratio, 'mean_adc': means,
              't10_phase': np.asarray(t10) % 1, 'baseline_covariance': covariance(residual[:, :, 0])}
    for name, view in [('all', np.ones(len(q), bool)), ('ratio_030_035', shape_control)]:
        contributions = np.full((len(q), len(WINDOWS), 4, len(LAGS)), np.nan)
        raw_events = np.full_like(contributions, np.nan)
        bins = []
        for cell in np.unique(cells[view]):
            ix = np.flatnonzero(view & (cells == cell))
            if len(ix) < 10:
                bins.append({'cell': int(cell), 'n': len(ix), 'used': False})
                continue
            root_mu = np.sqrt(means[ix])[:, None, :, None]
            signal = residual[ix, :, 1:]/root_mu
            pre = np.broadcast_to(residual[ix, :, :1], signal.shape)/root_mu
            rs, cs, vs, es = covariance_parts(signal)
            rb, cb, vb, eb = covariance_parts(pre)
            contributions[ix] = es-eb
            raw_events[ix] = covariance(signal)-covariance(pre)
            bins.append({'cell': int(cell), 'q_bin': int(cell//2), 'ratio_below_040': bool(cell%2 == 0),
                         'n': len(ix), 'used': True, 'raw': (rs-rb).tolist(),
                         'coherent': (cs-cb).tolist(), 'across_event': (vs-vb).tolist()})
        used = np.isfinite(contributions[:, 0, 0, 0])
        row = {'n_requested': int(view.sum()), 'n_used': int(used.sum()), 'cells': bins}
        if used.any():
            raw = raw_events[used].mean(axis=0)
            stochastic = contributions[used].mean(axis=0)
            row.update(raw=raw.tolist(), across_event=stochastic.tolist(), coherent=(raw-stochastic).tolist(),
                       mean_adc=means[used].mean(axis=0).tolist())
            # Conditional block uncertainty: group means are held fixed, not refit.
            block_id = np.asarray(ids)[used]//64
            blocks, inv = np.unique(block_id, return_inverse=True)
            sums = np.zeros((len(blocks), len(WINDOWS), 4))
            np.add.at(sums, inv, contributions[used, :, :, 0])
            counts = np.bincount(inv)
            rng = np.random.default_rng(281104)
            draws = rng.integers(len(blocks), size=(300, len(blocks)))
            boot = sums[draws].sum(axis=1)/counts[draws].sum(axis=1)[:, None, None]
            row['lag0_conditional_block_bootstrap95'] = np.quantile(boot, [.025, .975], axis=0).tolist()
            row['original_index_blocks'] = len(blocks)
            row['phase_quartile_lag0'] = []
            for lo, hi in zip([0, .25, .5, .75], [.25, .5, .75, 1]):
                sel = used & (arrays['t10_phase'] >= lo) & (arrays['t10_phase'] < hi)
                row['phase_quartile_lag0'].append({'interval': [lo, hi], 'n': int(sel.sum()),
                    'mean': contributions[sel, :, :, 0].mean(axis=0).tolist() if sel.any() else None})
        output['views'][name] = row
        arrays[name+'_lag0_across_event'] = contributions[..., 0]
    # Conditional scaling is descriptive; no fit coefficient is called a gain.
    output['adc_amplitude_p10_p50_p90'] = np.quantile(np.max(waves, axis=1)-baseline, [.1, .5, .9]).tolist()
    return output, arrays


def plot(out, report):
    ROOT.gROOT.SetBatch(True); ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetTitleFontSize(.038); ROOT.gStyle.SetTitleSize(.036, 'XYZ'); ROOT.gStyle.SetLabelSize(.031, 'XYZ')
    c = ROOT.TCanvas('local_covariance', 'Local residual covariance', 1500, 1100); c.Divide(2, 2, .003, .003)
    x = OFFSETS[1:]*2+N
    colors = [BLUE, RED, GREEN, ROOT.kMagenta+1, ROOT.kOrange+7]
    for col, width in enumerate((0, 2)):
        datasets = [('run3_first60', 'Co run 3'), ('run12_first60', 'Am run 12'), ('run18_first60', 'Am run 18'),
                    ('Co60_mc', 'Co Geant4'), ('Am241_mc', 'Am Geant4')]
        vals = [np.array(report['groups'][key]['views']['all']['across_event'])[width, :, 0] for key, _ in datasets]
        ymax = max(v.max() for v in vals)*1.3
        panel(c, col+1, f'{2*WINDOWS[width]} ns filter / mean-structure removed;Window centre from t10 (ns);Excess covariance / local mean (ADC)', 0, -.05, 2800, float(ymax))
        for (key, title), color, values in zip(datasets, colors, vals):
            graph(x.astype(float), values, color, 2 if 'mc' in key else 1)
            label(title, .49, .86-datasets.index((key, title))*.048, color, .028)
        label('400 ns windows; charge/shape cells; no particle truth', .18, .19, size=.024)
    for col, wi in enumerate((1, 3)):
        keys = ['run3_first60', 'run12_first60', 'run18_first60', 'Co60_mc', 'Am241_mc']
        curves = [np.array(report['groups'][key]['views']['all']['across_event'])[0, wi] for key in keys]
        low = min(v.min() for v in curves)*1.25; high = max(v.max() for v in curves)*1.25
        panel(c, col+3, f'50 ns filter / window {OFFSETS[wi+1]*2}-{OFFSETS[wi+1]*2+400} ns;Lag (ns);Excess covariance / local mean (ADC)', 0, float(low), 100, float(high))
        for key, color, values in zip(keys, colors, curves):
            graph(LAGS.astype(float)*2, values, color, 2 if 'mc' in key else 1)
        label('Includes filter-induced negative covariance', .19, .86, size=.027)
    save(c, out, 'local_covariance')
    c = ROOT.TCanvas('covariance_controls', 'State and shape controls', 1500, 1100); c.Divide(2, 2, .003, .003)
    for col, view in enumerate(('all', 'ratio_030_035')):
        datasets = [(f'run{run}_{group}', f'Run {run} {group}') for run, group in [(12,'first60'), (12,'last20'), (18,'first60')]]
        values = [np.array(report['groups'][key]['views'][view]['across_event'])[0, :, 0] for key, _ in datasets]
        panel(c, col+1, ('All shapes' if view == 'all' else 'Fixed 0.30 <= Qshort/Qlong < 0.35')+';Window centre from t10 (ns);50 ns excess covariance / mean (ADC)', 0, -.02, 2800, float(max(v.max() for v in values)*1.35))
        for i, ((key, title), v) in enumerate(zip(datasets, values)):
            graph(x.astype(float), v, colors[i]); label(title, .48, .86-i*.05, colors[i], .029)
    for col, wi in enumerate((1, 3)):
        runs = [10,11,12,13,14,18]
        panel(c, col+3, f'50 ns / shape control / {OFFSETS[wi+1]*2}-{OFFSETS[wi+1]*2+400} ns;Q (10^3 ADC sample);Excess covariance / local mean (ADC)', report['q_edges'][0]/1000, 0, report['q_edges'][-1]/1000, .5)
        for i, run in enumerate(runs):
            rows = report['groups'][f'run{run}_first60']['views']['ratio_030_035']['cells']
            rows = [r for r in rows if r['used']]
            if rows:
                xx = np.array([(report['q_edges'][r['q_bin']]+report['q_edges'][r['q_bin']+1])/2000 for r in rows])
                yy = np.array([r['across_event'][0][wi][0] for r in rows])
                color = [BLUE, GREEN, RED, ROOT.kViolet+1, ROOT.kOrange+7, ROOT.kMagenta+1][i]
                graph(xx, yy, color); label(f'Run {run}', .20+(i%3)*.23, .86-(i//3)*.05, color, .029)
        label('At least 10 events/cell; shapes are not particle labels', .19, .70, size=.026)
    save(c, out, 'covariance_controls')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--bias', type=Path, default=HERE/'results/bias_response_20261028')
    p.add_argument('--mc', type=Path, default=HERE/'results/colored_noise_20261028')
    p.add_argument('--replot', action='store_true')
    p.add_argument('--resume', action='store_true', help='Reuse per-group checkpoints from the identical analysis code')
    args = p.parse_args()
    if args.replot:
        plot(args.out, json.loads((args.out/'covariance.json').read_text())); return
    args.out.mkdir(parents=True, exist_ok=args.resume)
    with np.load(args.bias/'samples.npz') as z: previous = dict(z)
    prior = json.loads((args.bias/'bias_response.json').read_text())
    edges = np.array(prior['q_edges'])
    report = {'scope': __doc__, 'q_edges': edges.tolist(), 'window_start_ns': (OFFSETS*2).tolist(),
        'window_length_ns': N*2, 'filter_width_ns': (np.array(WINDOWS)*2).tolist(), 'lag_ns': (LAGS*2).tolist(),
        'selection': 'Exactly the event IDs frozen in bias_response_20261028; no new selection on the residual statistic. Same original first60/middle20/last20 roles. MC IDs from existing common-charge fluctuation check.',
        'conditioning': 'Six fixed charge bins crossed with ratio<0.40 or >=0.40; descriptive split in the previously observed gap, not particle labels. Additional fixed 0.30<=ratio<0.35 view for overlapping Am shapes. Cells with n<10 omitted and counted.',
        'covariance': 'Integer ceil(t10) anchoring; no interpolation. Subtract centered box average then the temporal mean in each 200-sample window. Pair count is N-lag. Each signal covariance minus its equal-window pre covariance is divided by local signal mean. Across-event covariance is n/(n-1) times covariance of conditionally event-centred residuals/sqrt(local mean); raw minus this is the unbiased cross-event coherent term. Negative terms retained.',
        'uncertainty': '300 bootstrap resamples of original event-index blocks of64; conditional on fixed cell means. This omits mean-estimation, mixing, setup and model uncertainty. Deterministic subsampling may leave sparse blocks.',
        'limits': ['Across-event covariance includes shape variations, event alignment, nonstationarity and selection effects; it is not uniquely avalanche noise.', 'Finite filtered windows impose correlations, including negative lobes; delay peaks alone are not afterpulse identification.', 'Known mean-response differences and selected-charge/shape conditioning prevent interpreting simple slopes as measured gain.', 'All groups previously explored; no blind validation claim.'],
        'inputs_sha256': {str(p.relative_to(PROJECT)): sha(p) for p in [args.bias/'samples.npz', args.bias/'bias_response.json']},
        'raw_identity_policy': 'Raw BIN hashes are inherited from the frozen bias analysis, not recomputed in this pass. Layout/count and frozen feature charges are checked; features are hashed here. No raw files are modified.',
        'code_sha256': {str(p.relative_to(PROJECT)): sha(p) for p in [Path(__file__),HERE/'reference.py',HERE/'diagnose_fluctuations.py']}, 'groups': {}}
    arrays = {}
    for run in RUNS:
        path = PROJECT/f'data/DT5730_DPP_PHA/DAQ/run_{run}/UNFILTERED/Data_run_{run}.BIN'
        fp = PROJECT/f'analysis/paper_baseline/results/pha_run{run:04d}_features.npz'
        with np.load(fp) as z: f = dict(z)
        _, dtype, count = bin_layout(path); raw = np.memmap(path, dtype=dtype, offset=2, mode='r', shape=(count,))
        for group in GROUPS:
            key = f'run{run}_{group}'; ids = previous[key+'_ids']
            np.testing.assert_array_equal(previous[key+'_q'], f['charge_long'][ids])
            checkpoint = args.out/(key+'.json')
            if args.resume and checkpoint.exists():
                record = json.loads(checkpoint.read_text())
                if record['code_sha256'] != sha(Path(__file__)): raise ValueError('Checkpoint code mismatch')
                row = record['result']
                with np.load(args.out/(key+'.npz')) as z: values = dict(z)
            else:
                row, values = analyze(raw['data'][ids].astype(float), f['baseline'][ids], f['t10_sample'][ids], f['charge_long'][ids], f['short_long_ratio'][ids], ids, edges)
                np.savez_compressed(args.out/(key+'.npz'), **values)
                checkpoint.write_text(json.dumps({'code_sha256':sha(Path(__file__)), 'result':row},allow_nan=False)+'\n')
            report['groups'][key] = row
            arrays.update({key+'_'+k:v for k,v in values.items()})
            print(key, row['n'], '50ns local', np.round(np.array(row['views']['all']['across_event'])[0,:,0], 5).tolist(), flush=True)
        report['inputs_sha256'][str(path.relative_to(PROJECT))] = prior['inputs_sha256'][str(path.relative_to(PROJECT))]
        report['inputs_sha256'][str(fp.relative_to(PROJECT))] = sha(fp)
    with np.load(args.mc/'fluctuations/metrics.npz') as z: mc = dict(z)
    for source, name in [('Co60','co60'),('Am241','am241')]:
        path = args.mc/name/'events.root'; ids = mc[source+'_mc_ids']
        waves, f = load_mc(path, ids)
        row, values = analyze(waves.astype(float), f['baseline'], f['t10_sample'], f['charge_long'], f['short_long_ratio'], ids, edges)
        key = source+'_mc'; report['groups'][key] = row; arrays.update({key+'_'+k:v for k,v in values.items()})
        report['inputs_sha256'][str(path.relative_to(PROJECT))] = sha(path)
        print(key, row['n'], '50ns local', np.round(np.array(row['views']['all']['across_event'])[0,:,0],5).tolist(), flush=True)
    np.savez_compressed(args.out/'samples.npz', **arrays)
    (args.out/'covariance.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    plot(args.out, report)


if __name__ == '__main__': main()
