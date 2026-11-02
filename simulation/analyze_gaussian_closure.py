"""Freeze and evaluate second-moment closure without changing the simulated ADC."""
import argparse
import concurrent.futures
import subprocess
from gaussian_closure_common import *
from publication_stats import block_counts
from analyze_acquisition_state import interval
from test_gaussian_closure import run as algebra_tests


def prepare():
    assert not OUT.exists()
    tests = algebra_tests()
    for name in ['kernels', 'evaluation', 'figures', 'source_snapshot']:
        (OUT/name).mkdir(parents=True, exist_ok=True)
    qp = read(QDC/'protocol.json'); qr = read(QDC/'record.json')
    needed = dict(qp['inputs_sha256'])
    for key, value in qr['outputs_sha256'].items():
        path = PROJECT/key
        if path.parent.name in ['arrays', 'evaluation', 'references'] or path.name in ['protocol.json', 'analysis.json', 'verification.json']:
            needed[key] = value
    needed[relative(QDC/'record.json')] = sha(QDC/'record.json')
    for key, value in needed.items(): assert sha(PROJECT/key) == value, key
    inherited = {k: v for k, v in qr['sources_sha256'].items() if k.endswith(('.py', '.cc', '.hh'))}
    for key, value in inherited.items(): assert sha(PROJECT/key) == value, key
    code = {relative(PROJECT/'simulation'/name): sha(PROJECT/'simulation'/name) for name in
            ['gaussian_closure_common.py', 'analyze_gaussian_closure.py', 'test_gaussian_closure.py']}
    p = {k: qp[k] for k in ['q_edges', 'support_bins', 'conditional_cuts', 'block_sizes', 'variants']}
    p.update(scope=__doc__, date='2026-11-02', checkpoint_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECT).decode().strip(),
        inputs_sha256=needed, code_sha256=code, inherited_sources_sha256=inherited,
        scales_ns=SCALES, normalizations=NORMS, conditioning=CONDITIONS, stats=STATS, metrics=METRICS,
        design='All old IDs, ADC, Q support and Q-half targets, three acquisition segments, blocks128 primary/64 sensitivity, all three frozen responses. No refit, event rejection, noise replacement or PID change. The four-seed Geant4 pool is previously explored diagnostic data.',
        quadratic_form='Separately center original high-pass post1000 and pre200 samples; divide by sqrt(999*mu) and sqrt(199*mu), concatenate X, A=diag(+1_post,-1_pre). F=X^T A X exactly. Repeat with opposite-fold mu0 from the parent fixed-denominator diagnostic.',
        gaussian_closure='For independent X,Y with second moment M and mean m: V_G=2 tr(AMAM)-2(m^T A m)^2 = 2 tr(ACAC)+4m^T A C A m. Estimate the first term with distinct-event pairs Kij^2 and the second with distinct-event quadruples Kij Kkl; Kij=Xi^T A Xj. U may be negative: never clip it.',
        weighting='Within each original Q half, uniform event weights; original half target masses and equal Q-bin weights. Primary cells are Q halves, retaining response-fold mixture. Sensitivity splits additionally by original response fold; metadata already shows Co middle/late cells with2/3 events. Fewer than4 distinct positive-weight original IDs is unestimable; retain every Q bin and propagate NaN, never merge a cell after inspecting its result.',
        aggregation='Use (N-1)/N times population Gaussian-U variance per cell to compare with original biased within-Q variance. Keep identical observed between-cell mean terms on both sides. This is a diagnostic closure, not a prediction of between-cell means or a correction for drift. Point and bootstrap observed variances reproduce parent F SDs exactly.',
        bootstrap='Use all512 parent paired original-index block and four-batch full-mother multinomial count vectors; consume old-mother counts to preserve RNG. For U moments, products of counts are assigned only to distinct ORIGINAL IDs; replicas never turn a self-pair into an independent pair. Denominators are weighted ordered distinct-pair/quadruple masses. This weighted-U resampling avoids high-dimensional self-pair contamination; intervals are conditional diagnostic percentile ranges, not a proven coverage guarantee. Parent observed F variance uses usual replica multiplicities. No refit/systematics/multiplicity or equivalence guarantee.',
        weighted_U='With diagonal-zero K, S=c^T K c, T=c^T(K*K)c, R=sum_i c_i^2(Kc)_i^2, H=(c*c)^T(K*K)(c*c). Pair=T/(N^2-p2); quad=(S^2-4R+2H)/(N^4-6N^2*p2+3*p2^2+8N*p3-6p4), pj=sum c^j. Verify by brute-force distinct tuples.',
        lag_diagnostic={'lags_samples': LAGS, 'lags_ns': [2*l for l in LAGS],
            'definition': 'Across-event centered post-window vectors, within original Q halves; average Cov(t,t+lag) divided by average Var(t), then fixed half/Q masses. Descriptive point curves; no stationarity, intrinsic device time constant or causal identification implied.'},
        interpretation='Remainder=observed variance minus Gaussian closure can include higher cumulants, heterogeneous event means/covariances, normalization, source mixture and finite/intra-cell dependent sampling. It does not uniquely identify non-Gaussian electronics or SiPM noise. Am source is mixed alpha/gamma/e, not event truth.',
        new_independent_mothers=0, new_native_replays=0, new_experimental_events=0, default_changed=False, official_PID_changed=False)
    save(OUT/'protocol.json', p); save(OUT/'algebra_tests.json', serialize(tests))
    print('Frozen closure protocol:', len(needed), 'verified inputs; independent algebra tests passed.', flush=True)


def contrasts_record(values, draws):
    out = {}
    for ic, cond in enumerate(CONDITIONS):
        out[cond] = {}
        for im, norm in enumerate(NORMS):
            out[cond][norm] = {}
            for k, scale in enumerate(SCALES):
                out[cond][norm][str(scale)] = {
                    'point': dict(zip(METRICS, values[ic, im, k])),
                    'CI95': {name: interval(draws[ic, im, k, :, j]) for j, name in enumerate(METRICS)}}
    return out


def counts_for_context(data, mc, group, B, source):
    run = int(group.split('_')[0][3:]); seg = group.split('_')[1]
    old = load(RC/'arrays'/f'{group}_B{B}_frozen_evaluation.npz')
    rng = np.random.default_rng(731031900+run+100*SEGMENTS.index(seg)+B)
    dc = [np.ones(len(data['ids']), int)]; mc_counts = [np.ones(len(mc['ids']), int)]
    for _ in range(512):
        dc.append(block_counts(data['ids'], tuple(data['span']), B, rng))
        mc_counts.append(mother_counts(mc, rng, [EVENTS[source]]*4))
        mother_counts(old, rng, [int(old['n_parent'])])
    return np.stack(dc, 1), np.stack(mc_counts, 1)


def one(group, B):
    p = read(OUT/'protocol.json'); dest = OUT/'evaluation'/f'{group}_B{B}.json'
    assert not dest.exists()
    run = int(group.split('_')[0][3:]); seg = group.split('_')[1]
    source = dict((run, source) for source, run in SOURCES)[run]
    key = f'{group}_B{B}'
    compact = {v: load(QDC/'arrays'/f'{key}_{v}.npz') for v in ['data', *VARIANTS]}
    dc, mc = counts_for_context(compact['data'], compact['frozen'], group, B, source)
    raw_data = data_waves(run, seg); raw_data['fold'] = (raw_data['ids']//B)%2
    previous = read(QDC/'evaluation'/f'{key}.json')
    boots = load(QDC/'evaluation'/f'{key}_bootstrap.npz')['joint']
    stats = {}; perq = {}; records = {}; kernel_hashes = {}; lag = {}
    for v in ['data', *VARIANTS]:
        raw = raw_data if v == 'data' else pool([assemble(b, run, seg, B, v, raw_data, p, source) for b in BATCHES])
        kernels, cells = build_kernels(raw, compact[v], p, source)
        path = OUT/'kernels'/f'{key}_{v}.npz'; np.savez_compressed(path, **kernels)
        kernel_hashes[relative(path)] = sha(path)
        s, q = aggregate(kernels, cells, dc if v == 'data' else mc, p, source)
        stats[v] = s; perq[v] = q
        old = previous['data'] if v == 'data' else previous['variants'][v]['summary']
        np.testing.assert_allclose(s[:, 0, :, 0, 0], np.diag(old['cov'])[None, 1:].repeat(2, 0), rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(s[:, 1, 2, 0, 0], np.sum(np.array(old['terms_cov'])[2, :2, :2]), rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(s[0, :, :, :, 0], s[1, :, :, :, 0], rtol=1e-9, atol=1e-10)
        curve = sum(c['target']*kernels[c['key']+'_lag'] for c in cells)/len(p['support_bins'][source]); lag[v] = curve
        records[v] = {'summary': point_rows(s[:, :, :, 0]), 'cells': cells,
            'per_Q': {str(b): point_rows(q[:, :, :, ib, 0]) for ib, b in enumerate(p['support_bins'][source])},
            'lag_correlation': curve, 'negative_U_point_entries': int(np.sum(s[:, :, :, 0, 1] < 0)),
            'unestimable_U_point_entries': int(np.sum(~np.isfinite(s[:, :, :, 0, 1])))}
        if v != 'data': del raw
        del kernels
    contrasts = {}; qcontrasts = {}
    for iv, v in enumerate(VARIANTS):
        z = contrast(stats[v], stats['data']); qz = contrast(perq[v], perq['data'])
        for k in range(3):
            np.testing.assert_allclose(z[:, 0, k, 1:, 0], np.tile(boots[:, iv, 5+k], (2, 1)), rtol=1e-8, atol=1e-10)
        contrasts[v] = z; qcontrasts[v] = qz
        records[v]['contrasts'] = contrasts_record(z[:, :, :, 0], z[:, :, :, 1:])
        records[v]['per_Q_contrasts'] = {str(b): contrasts_record(qz[:, :, :, ib, 0], qz[:, :, :, ib, 1:])
                                       for ib, b in enumerate(p['support_bins'][source])}
    bp = OUT/'evaluation'/f'{key}_bootstrap.npz'
    np.savez_compressed(bp, data_counts=dc, MC_counts=mc,
        **{f'{v}_stats': a for v, a in stats.items()}, **{f'{v}_per_Q': a for v, a in perq.items()},
        **{f'{v}_contrast': a for v, a in contrasts.items()}, **{f'{v}_per_Q_contrast': a for v, a in qcontrasts.items()})
    save(dest, serialize({'group': group, 'block': B, 'source': source, 'protocol_sha256': sha(OUT/'protocol.json'),
        'cohorts': records, 'kernel_sha256': kernel_hashes, 'bootstrap_sha256': sha(bp),
        'inherited_F_SD_scalar_checks': 512*3*3*2, 'new_independent_mothers': 0}))
    print('Completed', key, 'Gaussian U closure, fold sensitivity, all512 paired draws.', flush=True)


def analyze(workers):
    p = read(OUT/'protocol.json'); assert not (OUT/'analysis.json').exists()
    for key, value in p['code_sha256'].items(): assert sha(PROJECT/key) == value
    contexts = [(f'run{run}_{seg}', B) for _, run in SOURCES for seg in SEGMENTS for B in p['block_sizes']]
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(one, *c) for c in contexts]
        for f in concurrent.futures.as_completed(futures): f.result()
    save(OUT/'analysis.json', {'scope': __doc__, 'protocol_sha256': sha(OUT/'protocol.json'),
        'groups': {f'run{run}_{seg}': {str(B): read(OUT/'evaluation'/f'run{run}_{seg}_B{B}.json')
                    for B in p['block_sizes']} for _, run in SOURCES for seg in SEGMENTS}})
    print('All closure diagnostics completed.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['prepare', 'analyze'])
    parser.add_argument('--workers', type=int, default=2); args = parser.parse_args()
    prepare() if args.stage == 'prepare' else analyze(args.workers)
