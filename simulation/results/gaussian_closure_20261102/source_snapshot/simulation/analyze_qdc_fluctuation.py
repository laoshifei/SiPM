"""Freeze and evaluate fluctuation terms on the existing independent Geant4 pool."""
import argparse
import concurrent.futures
import subprocess
from qdc_fluctuation_common import *
from publication_stats import block_counts


def prepare():
    assert not OUT.exists()
    for name in ['arrays', 'references', 'evaluation', 'figures', 'source_snapshot']:
        (OUT/name).mkdir(parents=True, exist_ok=True)
    pp = read(PARENT/'protocol.json'); pr = read(PARENT/'record.json')
    wr = read(WAVE/'record.json'); wp = read(WAVE/'protocol.json'); cr = read(CORE/'record.json')
    rc = read(RC/'record.json')
    needed = [PARENT/n for n in ['protocol.json', 'record.json', 'analysis.json', 'verification.json']]
    needed += [WAVE/'record.json', CORE/'record.json', RC/'record.json']
    needed += sorted((PARENT/'arrays').glob('*.npz'))
    needed = [f for f in needed if not f.name.endswith('_identity.npz')]
    for _, run in SOURCES:
        needed.append(PROJECT/f'data/DT5730_DPP_PHA/DAQ/run_{run}/UNFILTERED/Data_run_{run}.BIN')
        for seg in SEGMENTS:
            group = f'run{run}_{seg}'; needed.append(WAVE/'arrays'/f'{group}_data.npz')
            for B in pp['block_sizes']:
                needed += [CORE/'references'/f'{group}_B{B}.json',
                           PARENT/'evaluation'/f'{group}_B{B}_bootstrap.npz',
                           RC/'arrays'/f'{group}_B{B}_frozen_evaluation.npz']
    expected = {}
    for rec in [pr, wr, cr, rc]: expected.update(rec['outputs_sha256'])
    expected.update(wp['inputs_sha256'])
    inputs = {}
    for f in sorted(set(needed)):
        value = sha(f); key = relative(f)
        if key in expected: assert value == expected[key], key
        inputs[key] = value
    protocol = {k: pp[k] for k in ['q_edges', 'support_bins', 'conditional_cuts', 'block_sizes', 'variants']}
    inherited_sources = {k: v for k, v in pr['sources_sha256'].items() if k.endswith(('.py', '.cc', '.hh'))}
    for path, value in inherited_sources.items(): assert sha(PROJECT/path) == value, path
    protocol.update(scope=__doc__, checkpoint_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECT).decode().strip(),
        inputs_sha256=inputs, code_sha256=sha(__file__), common_code_sha256=sha(PROJECT/'simulation/qdc_fluctuation_common.py'),
        inherited_sources_sha256=inherited_sources,
        parent_record_sha256=sha(PARENT/'record.json'), metrics=METRICS,
        design='All original segments, block128 primary and64 sensitivity; frozen/inherited/extended responses, old Q/IDs/targets and one response per mother retained. No fitting of a response, event rejection or changed PID. Existing independent pool is now diagnostic.',
        F_definition='F=(Vpost-Vpre)/mu. Original centered high-pass windows25/51/101 samples; pre200 samples and post1000 samples at original ceil(t10-300)/ceil(t10+250). Exact original ddof1 and original ADC.',
        reference='Only opposite measured raw-index fold, per original Q bin with original Q-half weights: pulse-window mean mu0 and F2025/50/95% quantiles. Fixed for all bootstrap draws; not a new response calibration or blind experimental validation.',
        decomposition='P=Vpost/mu0, B=-Vpre/mu0, D=(Vpost-Vpre)*(1/mu-1/mu0), so F=P+B+D exactly. Report full term covariance; contribution Cov(term,F) sums exactly to Var(F), can be negative and is not an independent physical noise variance.',
        native_decomposition='MC only: filter saved native signal and ADC-minus-baseline-minus-signal. Signal variance difference, additive-residual variance difference and twice their covariance difference divided by original mu sum to F. Native signal retains photon/avalanche fluctuations; additive residual groups electronics, empirical signal noise and quantization. Latent decomposition is not available for experiment and is not unique causal identification.',
        hierarchy='Exact within-Q covariance split: between Q-halves, between frozen evaluation folds within each half, and within half/fold. No finite-mean correction or removal of these components; between means includes sampling noise.',
        tail_rule='Existing opposite-fold QDC robust abs(z)>2; F202 outside opposite-fold measured5/95% thresholds. Four exclusive categories neither/QDC-only/F-only/both retain all events. Report counts, weighted masses, variance shares and correlations; shared samples/normalization prevent causal interpretation.',
        uncertainty='Regenerate the parent512 paired draws: original-index measured blocks, full-parent multinomial within four seeds, and consume the same old-parent counts to keep RNG sequence identical. Verify eight inherited metrics for every draw against parent. New references and fitted model fixed. No refit, selection, systematic or multiplicity/equivalence allowance.',
        per_Q='Report all original support bins; conditional paired intervals for every new contrast, with no bin removed based on its result.',
        new_independent_mothers=0, new_native_replays=0, new_experimental_events=0, default_changed=False, official_PID_changed=False)
    save(OUT/'protocol.json', protocol)
    print('Frozen diagnostic design and', len(inputs), 'input hashes.', flush=True)


def compare_Q(m, d):
    return np.stack([compare(x, y) for x, y in zip(m['per_Q'], d['per_Q'])])


def one(group, B):
    dest = OUT/'evaluation'/f'{group}_B{B}.json'
    if dest.exists():
        r = read(dest); assert r['code_sha256'] == sha(__file__)
        assert r['bootstrap_sha256'] == sha(OUT/'evaluation'/f'{group}_B{B}_bootstrap.npz')
        return
    p = read(OUT/'protocol.json'); assert p['common_code_sha256'] == sha(PROJECT/'simulation/qdc_fluctuation_common.py')
    run = int(group.split('_')[0][3:]); seg = group.split('_')[1]
    source = dict((r, s) for s, r in SOURCES)[run]
    data0 = data_waves(run, seg); data0['fold'] = (data0['ids']//B)%2
    data = extract(data0); ref = reference(data, p, source)
    save(OUT/'references'/f'{group}_B{B}.json', ref)
    pidref = read(CORE/'references'/f'{group}_B{B}.json')
    data = attach(data, p, source, ref, pidref); ds = summarize(data, p, source); closure(ds)
    np.savez_compressed(OUT/'arrays'/f'{group}_B{B}_data.npz', **data)
    fresh = {}; point = {}; per_batch = {}; cohort_hashes = {}
    parent = read(PARENT/'analysis.json')['groups'][group][str(B)]
    for v in VARIANTS:
        batches = [extract(assemble(batch, run, seg, B, v, data0, p, source)) for batch in BATCHES]
        a = attach(pool(batches), p, source, ref, pidref); fresh[v] = a
        point[v] = summarize(a, p, source); closure(point[v])
        per_batch[v] = []
        for batch, ba in zip(BATCHES, batches):
            ba = attach(ba, p, source, ref, pidref); bs = summarize(ba, p, source)
            per_batch[v].append({'batch': batch, 'summary': bs, 'point': dict(zip(METRICS, compare(bs, ds))) if bs else None})
        path = OUT/'arrays'/f'{group}_B{B}_{v}.npz'; np.savez_compressed(path, **a); cohort_hashes[relative(path)] = sha(path)
        values = compare(point[v], ds)
        for name, val in zip(METRICS[:8], values[:8]):
            np.testing.assert_allclose(val, parent['variants'][v]['point'][name], rtol=1e-10, atol=1e-12)
    del data0
    nb = len(p['support_bins'][source]); draws = np.full((512, 3, len(METRICS)), np.nan)
    perq = np.full((512, 3, nb, len(METRICS)), np.nan)
    old = load(RC/'arrays'/f'{group}_B{B}_frozen_evaluation.npz')
    rng = np.random.default_rng(731031900+run+100*SEGMENTS.index(seg)+B)
    for ib in range(512):
        dc = block_counts(data['ids'], tuple(data['span']), B, rng)
        count = mother_counts(fresh['frozen'], rng, [EVENTS[source]]*4)
        mother_counts(old, rng, [int(old['n_parent'])])
        db = summarize(data, p, source, dc)
        for iv, v in enumerate(VARIANTS):
            mb = summarize(fresh[v], p, source, count)
            if mb is not None and db is not None:
                draws[ib, iv] = compare(mb, db); perq[ib, iv] = compare_Q(mb, db)
    previous = load(PARENT/'evaluation'/f'{group}_B{B}_bootstrap.npz')['fresh_joint']
    parent_names = read(PARENT/'protocol.json')['metrics']
    for i, name in enumerate(METRICS[:8]):
        np.testing.assert_allclose(draws[:, :, i], previous[:, :, parent_names.index(name)], rtol=1e-10, atol=1e-12)
    bp = OUT/'evaluation'/f'{group}_B{B}_bootstrap.npz'; np.savez_compressed(bp, joint=draws, per_Q=perq)
    result = {'scope': __doc__, 'code_sha256': sha(__file__), 'protocol_sha256': sha(OUT/'protocol.json'),
              'group': group, 'block': B, 'source': source, 'data': ds, 'variants': {}, 'batches': per_batch,
              'reference_sha256': sha(OUT/'references'/f'{group}_B{B}.json'), 'bootstrap_sha256': sha(bp),
              'cohort_sha256': cohort_hashes, 'parent_metric_draw_closure': {'metrics': 8, 'draws': 512, 'variants': 3}}
    for iv, v in enumerate(VARIANTS):
        s = point[v]; per_point = compare_Q(s, ds)
        r = {'summary': s, 'point': dict(zip(METRICS, compare(s, ds))),
             'CI95': {n: interval(draws[:, iv, j]) for j, n in enumerate(METRICS)},
             'per_Q': {str(b): {'point': dict(zip(METRICS, per_point[k])),
                               'CI95': {n: interval(perq[:, iv, k, j]) for j, n in enumerate(METRICS)}}
                       for k, b in enumerate(p['support_bins'][source])}}
        result['variants'][v] = r
    save(dest, serialize(result)); print('Completed', group, B, 'exact terms, tails, per-Q intervals and parent-draw closure.', flush=True)


def analyze(workers):
    assert not (OUT/'analysis.json').exists()
    p = read(OUT/'protocol.json'); assert p['code_sha256'] == sha(__file__)
    contexts = [(f'run{run}_{seg}', B) for _, run in SOURCES for seg in SEGMENTS for B in p['block_sizes']]
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(one, *c) for c in contexts]
        for f in concurrent.futures.as_completed(futures): f.result()
    save(OUT/'analysis.json', {'scope': __doc__, 'code_sha256': sha(__file__),
        'protocol_sha256': sha(OUT/'protocol.json'), 'metrics': METRICS,
        'groups': {f'run{run}_{seg}': {str(B): read(OUT/'evaluation'/f'run{run}_{seg}_B{B}.json')
                   for B in p['block_sizes']} for _, run in SOURCES for seg in SEGMENTS},
        'new_independent_mothers': 0, 'new_native_replays': 0, 'default_changed': False, 'official_PID_changed': False})
    print('All fixed-event residual diagnostics complete.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['prepare', 'analyze'])
    parser.add_argument('--workers', type=int, default=2); args = parser.parse_args()
    prepare() if args.stage == 'prepare' else analyze(args.workers)
