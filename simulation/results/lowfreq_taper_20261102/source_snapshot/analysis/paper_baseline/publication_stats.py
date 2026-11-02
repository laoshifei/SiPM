"""Paired, training-refitted uncertainty for the frozen publication cohort.

Source-constrained statistical shape cores are not particle truth labels.
The original phase2 analysis and later diagnostic results remain unchanged.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from shape_mixture import fit_mixture
from prepare import bin_layout

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
SIM = PROJECT/'simulation'
OUT = HERE/'results/publication_20261028'
RUNS = (3, 7, 10, 11, 12, 13, 14, 18)
GROUPS = ('first60', 'middle20', 'last20')
ENDS = (500, 1000, 1500, 2000)
METHODS = ('qdc', 'negative_mean_time', 'pca_projection')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def load(path):
    with np.load(path) as z:
        return dict(z)


def charge_bin(q, edges):
    if np.any((q < edges[0]) | (q > edges[-1])):
        raise ValueError('Outside frozen Q interval')
    return np.minimum(np.searchsorted(edges, q, side='right')-1, 5)


def blocks_from_raw(waves, baseline, anchor):
    y = np.asarray(waves, float)-baseline[:, None]
    c = np.c_[np.zeros(len(y)), np.cumsum(y, axis=1)]
    boundaries = anchor[:, None]-20+20*np.arange(102)
    if boundaries.min() < 128 or boundaries.max() > waves.shape[1]:
        raise ValueError('Unsupported 4 us block window')
    return np.diff(np.take_along_axis(c, boundaries, axis=1), axis=1).astype(np.float32)


def split_of(ids, count):
    return np.where(ids < int(.6*count), 0, np.where(ids < int(.8*count), 1, 2)).astype(np.int8)


def prepare(out):
    out.mkdir(parents=True, exist_ok=True)
    bias_path = SIM/'results/bias_response_20261028'
    bias = load(bias_path/'samples.npz')
    prior = json.loads((bias_path/'bias_response.json').read_text())
    edges = np.array(prior['q_edges'])
    inputs = {str((bias_path/'samples.npz').relative_to(PROJECT)): sha(bias_path/'samples.npz')}
    flow = []
    for run in RUNS:
        fp = HERE/f'results/pha_run{run:04d}_features.npz'
        pp = HERE/f'results/phase2/run{run:04d}_features.npz'
        f, p = load(fp), load(pp)
        n = len(f['amplitude']); ids = np.arange(n)
        split = split_of(ids, n)
        common = (f['charge_long'] >= edges[0]) & (f['charge_long'] <= edges[-1])
        masks = [np.ones(n, bool), p['basic_eligible'], p['basic_eligible'] & p['comparison_support'],
                 p['eligible_primary'], p['eligible_primary'] & common,
                 p['eligible_primary'] & common & (f['amplitude'] >= 200),
                 p['eligible_primary'] & common & (f['amplitude'] >= 200) & ((f['flags'] & 0x8000) == 0)]
        masks.append(masks[-1] & (f['t10_sample'] >= 384))
        names = ('saved', 'basic', 'four_windows', 'pretrigger_primary', 'common_Q', 'amplitude200', 'bit_clear', 'time_support')
        selected = []
        for s, group in enumerate(GROUPS):
            candidate = np.flatnonzero(masks[-1] & (split == s))
            kept = candidate if len(candidate) <= 3000 else candidate[np.linspace(0, len(candidate)-1, 3000, dtype=int)]
            np.testing.assert_array_equal(kept, bias[f'run{run}_{group}_ids'])
            selected.extend(kept)
            flow.append({'run': run, 'segment': group,
                         **{name: int((m & (split == s)).sum()) for name, m in zip(names, masks)},
                         'frozen_primary': len(kept)})
        selected = np.asarray(selected)
        # A fixed broad pool is retained only for predeclared selection sensitivity.
        pool = np.flatnonzero(p['basic_eligible'] & p['comparison_support'] & common)
        if run not in (3, 12, 18):
            pool = selected
        blocks = []
        if 'block_charge' in p:
            blocks = p['block_charge'][pool]
        else:
            raw_path = PROJECT/f'data/DT5730_DPP_PHA/DAQ/run_{run}/UNFILTERED/Data_run_{run}.BIN'
            _, dtype, count = bin_layout(raw_path)
            if count != n: raise ValueError('Raw/feature count mismatch')
            raw = np.memmap(raw_path, dtype=dtype, offset=2, mode='r', shape=(count,))
            for start in range(0, len(pool), 512):
                ix = pool[start:start+512]
                blocks.append(blocks_from_raw(raw['data'][ix], f['baseline'][ix], p['anchor_sample'][ix]))
            blocks = np.concatenate(blocks)
            inputs[str(raw_path.relative_to(PROJECT))] = prior['inputs_sha256'][str(raw_path.relative_to(PROJECT))]
        values = {k: p[k][pool] for k in ('long_charge', 'short_charge', 'mean_time_us', 'recent_baseline_delta',
                  'eligible_primary', 'eligible_strict', 'secondary_rise_flag')}
        values.update({k: f[k][pool] for k in ('amplitude', 'flags', 't10_sample', 'baseline', 'short_long_ratio')})
        values.update(ids=pool, run=np.full(len(pool), run), split=split[pool], charge_bin=charge_bin(f['charge_long'][pool], edges),
                      block_charge=blocks, primary=np.isin(pool, selected), original_count=np.array(n))
        target = out/f'pool_run{run}.npz'
        if target.exists():
            old = load(target)
            for key in values: np.testing.assert_equal(old[key], values[key])
        else:
            np.savez_compressed(target, **values)
        inputs[str(fp.relative_to(PROJECT))] = sha(fp)
        inputs[str(pp.relative_to(PROJECT))] = sha(pp)
        print('prepared', run, 'pool', len(pool), 'frozen', len(selected), flush=True)
    protocol = {
        'scope': __doc__, 'date': '2026-10-28', 'runs': RUNS, 'q_edges': edges.tolist(),
        'selection': 'Exactly the previously frozen40795 pulse IDs; at most3000 per original first60/middle20/last20 segment, deterministic evenly spaced eligible positions. No selection based on the new results.',
        'analysis': 'All PSD methods/windows use the same event IDs. Training only run3/12 first60; checks already explored, never blind. Primary endpoint3us, window1/2/4us comparisons secondary.',
        'resampling': 'Run-specific original acquisition-index blocks; all methods, windows and Q bins share a draw. Empty blocks in the full training span included. PCA and Cauchy location/scale are re-estimated inside each draw. Conditional on chosen model, cuts, fixed Q edges and deterministic subsampling.',
        'main_bootstrap': {'draws': 256, 'block_size': 64, 'seed': 28102831},
        'block_sensitivity': {'draws': 128, 'sizes': [32, 128], 'window_ns': 3000},
        'selection_sensitivity': ['include_hardware_bit', 'strict_pretrigger', 'exclude_secondary_rise', 'recent_baseline', 'background_half', 'background_double'],
        'covariance_focus': 'Run12/18 first60; equal fixed common Q-cell weighting; recompute group means in each block draw, baseline and shape gates paired. No unique gain/noise attribution.',
        'raw_identity_policy': 'Existing raw hashes inherited where raw data are used; not rehashed. Current feature files hashed, frozen IDs reconstructed and checked. Earlier phase2 results not overwritten.',
        'inputs_sha256': inputs, 'flow': flow,
        'code_sha256': {str(p.relative_to(PROJECT)): sha(p) for p in [Path(__file__), HERE/'shape_mixture.py']}}
    if (out/'protocol.json').exists() and json.loads((out/'protocol.json').read_text()) != json.loads(json.dumps(protocol)):
        raise ValueError('Frozen protocol changed')
    save(out/'protocol.json', protocol)
    with (out/'cohort_flow.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flow[0])); writer.writeheader(); writer.writerows(flow)
    assert sum(row['frozen_primary'] for row in flow) == 40795


def subset(d, ix):
    return {k: v[ix] for k, v in d.items() if v.ndim and len(v) == len(d['ids'])}


def training(out, variant='primary'):
    result = []
    for run in (3, 12):
        d = load(out/f'pool_run{run}.npz')
        mask = d['primary'] & (d['split'] == 0)
        if variant == 'include_hardware_bit':
            mask = d['eligible_primary'] & (d['amplitude'] >= 200) & (d['t10_sample'] >= 384) & (d['split'] == 0)
        if variant == 'strict_pretrigger': mask &= d['eligible_strict']
        if variant == 'exclude_secondary_rise': mask &= ~d['secondary_rise_flag']
        a = subset(d, np.flatnonzero(mask))
        if variant == 'recent_baseline':
            delta = a['recent_baseline_delta']
            old_q = a['long_charge'].copy()
            for j, end in enumerate(ENDS):
                length = end+20
                q = old_q[:, j]-delta*length
                moment = a['mean_time_us'][:, j]*old_q[:, j]-delta*length*(end-21)/2*.002
                a['mean_time_us'][:, j] = moment/q
                a['long_charge'][:, j] = q
            a['short_charge'] = a['short_charge']-delta[:, None]*np.array([120,170,270,420])
            a['block_charge'] = a['block_charge'].astype(float)-20*delta[:, None]
            if np.any(a['long_charge'] <= 0): raise ValueError('Non-positive alternative charge')
        result.append(a)
    return {k: np.concatenate([d[k] for d in result]) for k in result[0]}


def block_counts(ids, span, size, rng):
    """Integer event multiplicities from non-overlapping original-index blocks."""
    lo, hi = span
    blocks = (hi-lo+size-1)//size
    if ids.min() < lo or ids.max() >= hi: raise ValueError('Event outside acquisition span')
    selected = rng.integers(0, blocks, size=blocks)
    return np.bincount(selected, minlength=blocks)[(ids-lo)//size]


def pca_scores(d, window, counts):
    nblocks = (ENDS[window]+20)//20
    matrix = d['block_charge'][:, :nblocks].astype(float)/d['long_charge'][:, window, None]
    weights = np.zeros(len(counts), float)
    for run in (3, 12):
        for b in range(6):
            ix = (d['run'] == run) & (d['charge_bin'] == b)
            n = counts[ix].sum()
            if n < 20: raise ValueError('Sparse PCA resample cell')
            weights[ix] = counts[ix]/(12*n)
    center = weights @ matrix
    z = matrix-center
    cov = z.T @ (weights[:, None]*z)
    eigen, vectors = np.linalg.eigh(cov)
    vector = vectors[:, -1]
    projection = z @ vector
    reference = d['short_long_ratio']
    if np.sum(weights*projection*(reference-weights @ reference)) < 0:
        vector = -vector; projection = -projection
    scale = np.sqrt(weights @ (projection*projection))
    if scale <= 0: raise ValueError('Degenerate PCA')
    scores = np.column_stack((d['short_charge'][:, 2]/d['long_charge'][:, window],
                             -d['mean_time_us'][:, window], projection/scale))
    model = {'blocks': nblocks, 'center': center.tolist(), 'vector': vector.tolist(), 'score_scale': float(scale),
             'explained_variance_fraction': float(eigen[-1]/eigen.sum())}
    return scores, model


def fitted_foms(d, window, counts, background_scale=1.):
    scores, pca = pca_scores(d, window, counts)
    foms = np.full((3, 6), np.nan); models = []
    for method in range(3):
        for b in range(6):
            ix = np.flatnonzero(d['charge_bin'] == b)
            repeated = np.repeat(ix, counts[ix])
            fit = fit_mixture(scores[repeated, method], d['run'][repeated],
                              background_scale_factor=background_scale, max_iter=600, tolerance=1e-7)
            if fit['converged'] and not fit['at_sigma_floor']:
                foms[method, b] = fit['shape_fom_fwhm']
            models.append({'method': METHODS[method], 'bin': b, **fit})
    return foms, pca, models


def point(out):
    d = training(out); report = {'windows': {}, 'sensitivity': {}}; scores = {}
    for w, end in enumerate(ENDS):
        foms, pca, fits = fitted_foms(d, w, np.ones(len(d['ids']), int))
        if not np.isfinite(foms).all(): raise ValueError('Point fits failed')
        report['windows'][str(end*2)] = {'fom_by_method_bin': foms.tolist(), 'pca': pca, 'fits': fits}
        scores[str(end*2)], _ = pca_scores(d, w, np.ones(len(d['ids']), int))
    for variant in ('include_hardware_bit', 'strict_pretrigger', 'exclude_secondary_rise', 'recent_baseline', 'background_half', 'background_double'):
        x = training(out, variant)
        factor = .5 if variant == 'background_half' else 2. if variant == 'background_double' else 1.
        foms, _, fits = fitted_foms(x, 2, np.ones(len(x['ids']), int), factor)
        report['sensitivity'][variant] = {'events': len(x['ids']), 'fom_by_method_bin': foms.tolist(),
            'converged': all(f['converged'] and not f['at_sigma_floor'] for f in fits)}
    save(out/'point.json', report)
    np.savez_compressed(out/'training_scores.npz', **scores, ids=d['ids'], runs=d['run'])
    print('point median FOM', {k: np.median(v['fom_by_method_bin'], axis=1).tolist() for k, v in report['windows'].items()}, flush=True)


def bootstrap(out, size, draws):
    d = training(out); windows = range(4) if size == 64 else [2]
    protocol = json.loads((out/'protocol.json').read_text())
    for path, digest in protocol['code_sha256'].items():
        if sha(PROJECT/path) != digest: raise ValueError('Analysis code changed after protocol freeze')
    folder = out/f'bootstrap_b{size}'; folder.mkdir(exist_ok=True)
    spans = {r: (0, int(.6*int(load(out/f'pool_run{r}.npz')['original_count']))) for r in (3, 12)}
    start = time.monotonic()
    for draw in range(draws):
        target = folder/f'draw{draw:04d}.npz'
        if target.exists(): continue
        rng = np.random.default_rng(np.random.SeedSequence([28102831, size, draw]))
        counts = np.zeros(len(d['ids']), int)
        for r in (3, 12):
            ix = d['run'] == r
            counts[ix] = block_counts(d['ids'][ix], spans[r], size, rng)
        values = {}; failures = []
        for w in windows:
            try:
                foms, pca, fits = fitted_foms(d, w, counts)
                values[f'fom_{ENDS[w]*2}'] = foms
                values[f'pca_center_{ENDS[w]*2}'] = np.array(pca['center'])
                values[f'pca_vector_{ENDS[w]*2}'] = np.array(pca['vector'])
                values[f'pca_scale_{ENDS[w]*2}'] = np.array(pca['score_scale'])
                failures.extend([{'window': ENDS[w]*2, 'method': f['method'], 'bin': f['bin']} for f in fits if not f['converged'] or f['at_sigma_floor']])
            except ValueError as e:
                values[f'fom_{ENDS[w]*2}'] = np.full((3,6), np.nan)
                failures.append({'window': ENDS[w]*2, 'reason': str(e)})
        np.savez_compressed(target, **values, counts=counts)
        if failures: save(target.with_suffix('.json'), {'failures': failures})
        if (draw+1) % 8 == 0 or draw == 0:
            print('block', size, 'draw', draw+1, 'seconds', round(time.monotonic()-start,1), flush=True)


def intervals(values):
    values = np.asarray(values)
    good = np.isfinite(values)
    return {'n': int(good.sum()), 'total': len(values),
            'interval95': np.quantile(values[good], [.025,.975]).tolist() if good.sum() >= max(20, .8*len(values)) else None}


def summarize(out):
    point_report = json.loads((out/'point.json').read_text())
    report = {'scope': __doc__, 'uncertainty': 'Percentile block bootstrap, PCA and Cauchy nuisance re-estimated; paired draws. Exploratory intervals conditional on model/cuts. Window and bin intervals are not multiplicity-adjusted.', 'block_sizes': {}}
    for size in (64,32,128):
        files = sorted((out/f'bootstrap_b{size}').glob('draw*.npz'))
        if not files: continue
        windows = ('1000','2000','3000','4000') if size == 64 else ('3000',)
        data = {w: np.array([load(p)['fom_'+w] for p in files]) for w in windows}
        result = {'draws': len(files), 'windows': {}, 'window_differences': {}}
        for w, foms in data.items():
            median = np.median(foms, axis=2)
            baseline = np.median(point_report['windows'][w]['fom_by_method_bin'], axis=1)
            result['windows'][w] = {'point_median_fom': baseline.tolist(),
                'median_fom': {m: intervals(median[:,i]) for i,m in enumerate(METHODS)},
                'paired_differences': {}, 'per_bin_differences': {}}
            for a,b in ((1,0),(2,0),(2,1)):
                key = METHODS[a]+' minus '+METHODS[b]
                row = intervals(median[:,a]-median[:,b]); row['point'] = float(baseline[a]-baseline[b])
                result['windows'][w]['paired_differences'][key] = row
                result['windows'][w]['per_bin_differences'][key] = [intervals(foms[:,a,j]-foms[:,b,j]) for j in range(6)]
        if size == 64:
            for lo, hi in (('1000','2000'),('2000','3000'),('3000','4000')):
                values = np.median(data[hi],axis=2)-np.median(data[lo],axis=2)
                result['window_differences'][hi+' minus '+lo] = {m: intervals(values[:,i]) for i,m in enumerate(METHODS)}
        report['block_sizes'][str(size)] = result
    report['selection_sensitivity'] = point_report['sensitivity']
    save(out/'paired_uncertainty.json', report)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['prepare','point','bootstrap','summarize'])
    p.add_argument('--out', type=Path, default=OUT); p.add_argument('--block-size', type=int, default=64)
    p.add_argument('--draws', type=int, default=256); args = p.parse_args()
    if args.stage == 'bootstrap': bootstrap(args.out, args.block_size, args.draws)
    else: {'prepare':prepare, 'point':point, 'summarize':summarize}[args.stage](args.out)


if __name__ == '__main__': main()
