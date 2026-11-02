"""Compare fixed responses on four fresh seed batches without fitting or selection."""
import argparse
import concurrent.futures
import subprocess
from mc_statistics_common import *
from publication_stats import block_counts
from analyze_acquisition_state import interval


def compact(a):
    return {k: v for k, v in a.items() if k not in ['waves', 'signal', 'state']}


def support(a, p, source):
    rr = rows(a, p, source)
    if rr is None:
        return {'estimable': False, 'selected_events': len(a['ids'])}
    w = event_weights(a, p, source)
    result = {'estimable': True, 'selected_events': len(a['ids']),
              'supported_events': int(np.count_nonzero(w)), 'ESS': float(1/(w@w)), 'Q_bins': {}}
    for b, ix, weight, cell in rr:
        result['Q_bins'][str(b)] = {'count': len(ix), 'Q_half_counts': np.bincount(cell, minlength=2),
            'ESS': float(1/(weight@weight)), 'fold_counts': np.bincount(a['fold'][ix], minlength=2),
            'fold0_weight': float(weight@(a['fold'][ix] == 0))}
    return result


def prepared(a, p, source, ref, design):
    z = standardized(a, p, source, ref, a['fold'])
    modes = projection(a, p, source, design, a['fold'])
    return z, modes


def one(group, B):
    path = OUT/'evaluation'/f'{group}_B{B}.json'
    if path.exists():
        r = read(path)
        assert r['code_sha256'] == sha(__file__)
        assert r['bootstrap_sha256'] == sha(OUT/'evaluation'/f'{group}_B{B}_bootstrap.npz')
        return
    p = read(OUT/'protocol.json')
    assert p['common_code_sha256'] == sha(PROJECT/'simulation/mc_statistics_common.py')
    run = int(group.split('_')[0][3:]); seg = group.split('_')[1]
    source = dict((r, s) for s, r in SOURCES)[run]; si = SEGMENTS.index(seg)
    data, _ = measured(run, seg); data = compact(data); data['fold'] = (data['ids']//B)%2
    original = load(WAVE/'arrays'/f'{group}_data.npz')
    for name in ['ids', 'q']: np.testing.assert_array_equal(data[name], original[name])
    np.testing.assert_allclose(data['scores'], original['scores'][:, 0], rtol=1e-12, atol=1e-12)
    ref = read(CORE/'references'/f'{group}_B{B}.json')
    design = read(WAVE/'designs'/f'{group}_B{B}.json')
    zd, md = prepared(data, p, source, ref, design); ds = stats(data, p, source, zd, md)
    old = {}; fresh = {}; batches = {}; contexts = {}; points = {}; old_points = {}; supports = {}
    batch_results = {}
    prevcore = read(CORE/'diagnostics.json')['groups'][group][str(B)]['variants']
    prevjoint = read(RC/'evaluation.json')['groups'][group][str(B)]['variants']
    prevwave = read(WAVE/'analysis.json')['groups'][group][str(B)]['variants']
    for v in VARIANTS:
        a = compact(load(RC/'arrays'/f'{group}_B{B}_{v}_evaluation.npz'))
        a['fold'] = load(CORE/'arrays'/f'{group}_B{B}_{v}_z.npz')['fold']; old[v] = a
        oz, om = prepared(a, p, source, ref, design); os = stats(a, p, source, oz, om)
        old_points[v] = contrast(os, ds)
        inherited = [prevcore[v]['point'][n] for n in CORE_NAMES]
        inherited += [prevjoint[v]['point'][JOINT_NAMES[i]] for i in EXTRA]
        inherited += [prevwave[v]['mode_point'][n] for n in MODE_NAMES]
        np.testing.assert_allclose(old_points[v], inherited, rtol=1e-12, atol=1e-12)
        batches[v] = [compact(assemble(batch, run, seg, B, v, data, p, source)) for batch in BATCHES]
        fresh[v] = pool(batches[v]); fz, fm = prepared(fresh[v], p, source, ref, design)
        fs = stats(fresh[v], p, source, fz, fm)
        if fs is None: raise RuntimeError('Fresh pooled sample has missing original target cells; do not refit support.')
        points[v] = contrast(fs, ds); contexts[v] = {'old': (oz, om), 'fresh': (fz, fm), 'point': fs, 'old_summary': os}
        supports[v] = support(fresh[v], p, source)
        br = []
        for batch, a in zip(BATCHES, batches[v]):
            bz, bm = prepared(a, p, source, ref, design); bs = stats(a, p, source, bz, bm)
            br.append({'batch': batch, 'support': support(a, p, source),
                       'point': dict(zip(METRICS, contrast(bs, ds))) if bs is not None else None})
        batch_results[v] = br
    for v in VARIANTS[1:]:
        for name in ['ids', 'q', 'fold']: np.testing.assert_array_equal(fresh[v][name], fresh['frozen'][name])
    rng = np.random.default_rng(731031900+run+100*si+B)
    shape = (512, len(VARIANTS), len(METRICS))
    draws = {name: np.full(shape, np.nan) for name in ['fresh_joint', 'old_joint', 'fresh_MC_only', 'fresh_data_only']}
    for ib in range(512):
        dc = block_counts(data['ids'], tuple(data['span']), B, rng)
        db = stats(data, p, source, zd, md, dc)
        count = mother_counts(fresh['frozen'], rng, [EVENTS[source]]*4)
        oldcount = mother_counts(old['frozen'], rng, [int(old['frozen']['n_parent'])])
        for iv, v in enumerate(VARIANTS):
            fz, fm = contexts[v]['fresh']; oz, om = contexts[v]['old']
            fb = stats(fresh[v], p, source, fz, fm, count)
            ob = stats(old[v], p, source, oz, om, oldcount)
            if fb is not None:
                draws['fresh_MC_only'][ib, iv] = contrast(fb, ds)
                if db is not None: draws['fresh_joint'][ib, iv] = contrast(fb, db)
            if db is not None:
                draws['fresh_data_only'][ib, iv] = contrast(contexts[v]['point'], db)
                if ob is not None: draws['old_joint'][ib, iv] = contrast(ob, db)
    record = {'scope': __doc__, 'code_sha256': sha(__file__), 'protocol_sha256': sha(OUT/'protocol.json'),
              'group': group, 'block': B, 'data_summary': ds, 'variants': {}, 'batches': batch_results,
              'old_point_closure': True, 'new_independent_mothers_source': 4*EVENTS[source]}
    for iv, v in enumerate(VARIANTS):
        stat = {'point': dict(zip(METRICS, points[v])), 'old_point': dict(zip(METRICS, old_points[v])),
                'fresh_summary': contexts[v]['point'], 'old_summary': contexts[v]['old_summary'],
                'support': supports[v], 'CI95': {}, 'old_CI95_recomputed': {},
                'fresh_minus_old_CI95': {}, 'paired_from_frozen_CI95': {}, 'MC_only_CI95': {},
                'data_only_CI95': {}, 'MC_only_SD': {}, 'data_only_SD': {}}
        for j, name in enumerate(METRICS):
            new = draws['fresh_joint'][:, iv, j]; oldb = draws['old_joint'][:, iv, j]
            mb = draws['fresh_MC_only'][:, iv, j]; db = draws['fresh_data_only'][:, iv, j]
            stat['CI95'][name] = interval(new)
            stat['old_CI95_recomputed'][name] = interval(oldb)
            stat['fresh_minus_old_CI95'][name] = interval(new-oldb)
            stat['paired_from_frozen_CI95'][name] = interval(new-draws['fresh_joint'][:, 0, j])
            stat['MC_only_CI95'][name] = interval(mb); stat['data_only_CI95'][name] = interval(db)
            stat['MC_only_SD'][name] = float(np.nanstd(mb, ddof=1))
            stat['data_only_SD'][name] = float(np.nanstd(db, ddof=1))
        record['variants'][v] = stat
    if seg == 'last20' and B == 128:
        rng = np.random.default_rng(731032500+run)
        small = np.full(shape, np.nan); sizes = np.zeros(512, int)
        n = int(fresh['frozen']['n_parent']); oldn = p['old_independent_mothers'][source]
        for ib in range(512):
            ids = rng.choice(n, oldn, replace=False)
            mask = np.isin(fresh['frozen']['ids'], ids).astype(int); sizes[ib] = mask.sum()
            for iv, v in enumerate(VARIANTS):
                fz, fm = contexts[v]['fresh']; ms = stats(fresh[v], p, source, fz, fm, mask)
                if ms is not None: small[ib, iv] = contrast(ms, ds)
        draws['old_size_subsamples'] = small; draws['old_size_selected_counts'] = sizes
        record['old_size_diagnostic'] = {'sampled_full_parent_count': oldn, 'selected_count_quantiles': np.quantile(sizes, [.025, .5, .975]), 'variants': {}}
        for iv, v in enumerate(VARIANTS):
            d = {}
            for j, name in enumerate(METRICS):
                x = small[:, iv, j]; x = x[np.isfinite(x)]
                d[name] = {'conditional_range95': interval(x),
                          'old_point_empirical_rank': float(np.mean(x <= old_points[v][j])) if len(x) else None}
            record['old_size_diagnostic']['variants'][v] = d
    bp = OUT/'evaluation'/f'{group}_B{B}_bootstrap.npz'
    np.savez_compressed(bp, **draws)
    record['bootstrap_sha256'] = sha(bp)
    save(path, serialize(record))
    v = record['variants']['spectrum_CV_fit']; name = 'width95_negative_mean_time_ratio'
    print(group, B, 'new time95', round(v['point'][name], 4), 'old', round(v['old_point'][name], 4),
          'new CI', v['CI95'][name]['CI95'], flush=True)


def subprocess_one(group, B):
    command = [PYTHON, '-u', __file__, '--group', group, '--block', str(B)]
    with (OUT/'logs'/f'{group}_B{B}_evaluation.log').open('w') as log:
        r = subprocess.run(command, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT)
    if r.returncode: raise RuntimeError(f'{group} B{B} failed; see evaluation log')
    print(group, B, 'evaluated', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--group'); parser.add_argument('--block', type=int)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    assert read(OUT/'transport_execution.json')['new_independent_mothers'] == 256000
    assert read(OUT/'replay_execution.json')['new_independent_mothers'] == 0
    (OUT/'evaluation').mkdir(exist_ok=True)
    if args.group: one(args.group, args.block); return
    assert not (OUT/'analysis.json').exists()
    p = read(OUT/'protocol.json')
    tasks = [(f'run{run}_{seg}', B) for _, run in SOURCES for seg in SEGMENTS for B in p['block_sizes']]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool0:
        futures = [pool0.submit(subprocess_one, *task) for task in tasks]
        for f in concurrent.futures.as_completed(futures): f.result()
    groups = {}
    for group, B in tasks: groups.setdefault(group, {})[str(B)] = read(OUT/'evaluation'/f'{group}_B{B}.json')
    save(OUT/'analysis.json', {'scope': __doc__, 'code_sha256': sha(__file__), 'metrics': METRICS,
         'protocol_sha256': sha(OUT/'protocol.json'), 'groups': groups,
         'new_independent_mothers': 256000, 'new_experimental_events': 0,
         'replay_entries': read(OUT/'replay_execution.json')['replay_entries'],
         'default_changed': False, 'official_PID_changed': False})
    print('All independent-batch and pooled evaluations complete.', flush=True)


if __name__ == '__main__': main()
