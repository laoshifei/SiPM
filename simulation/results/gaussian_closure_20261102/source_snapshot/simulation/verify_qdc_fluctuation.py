"""Verify exact fluctuation algebra, independent windows, paired draws and notebook."""
import argparse
import base64
import csv
import shutil
import platform
import nbformat
from qdc_fluctuation_common import *
from analyze_qdc_fluctuation import compare_Q
from publication_stats import block_counts


def sources():
    p = read(OUT/'protocol.json')
    names = list(p['inherited_sources_sha256']) + [
        'simulation/qdc_fluctuation_common.py', 'simulation/analyze_qdc_fluctuation.py',
        'simulation/recover_qdc_fluctuation_report.py', 'simulation/report_qdc_fluctuation.py',
        'simulation/verify_qdc_fluctuation.py', 'simulation/README.md',
        'jupyter/codex_0001_build_qdc_fluctuation_notebook.py']
    return [PROJECT/n for n in sorted(set(names))]


def hashes(items):
    for path, value in items.items(): assert sha(PROJECT/path) == value, path


def windows(raw, compact):
    """Independent full convolution and direct slices, rather than prefix sums."""
    take = np.unique(np.linspace(0, len(raw['ids'])-1, min(24, len(raw['ids']))).astype(int))
    for i in take:
        y = raw['waves'][i].astype(float)-raw['baseline'][i]
        pre = int(np.ceil(raw['t10'][i]-300)); post = int(np.ceil(raw['t10'][i]+250))
        mu = np.mean(y[post:post+1000])
        for k, width in enumerate([25, 51, 101]):
            kernel = np.ones(width)/width
            r = y-np.convolve(y, kernel, 'same')
            vp = np.var(r[pre:pre+200], ddof=1); vs = np.var(r[post:post+1000], ddof=1)
            np.testing.assert_allclose([mu, vp, vs, (vs-vp)/mu], compact['detail'][i, k], rtol=1e-9, atol=1e-8)
            if 'signal' in raw:
                s = raw['signal'][i]; sr = s-np.convolve(s, kernel, 'same'); nr = r-sr
                parts = []
                for start, n in [(pre, 200), (post, 1000)]:
                    c = np.cov(sr[start:start+n], nr[start:start+n], ddof=1)
                    parts.append(np.array([c[0, 0], c[1, 1], 2*c[0, 1]]))
                np.testing.assert_allclose((parts[1]-parts[0])/mu, compact['native_terms'][i, k], rtol=1e-8, atol=1e-8)
    return len(take)


def independent_covariance(a, p, source, s):
    bins = np.searchsorted(p['q_edges'], a['q'], side='right')-1; bins = np.clip(bins, 0, 5)
    covs = []; terms = []; native = []
    for b in p['support_bins'][source]:
        ix = np.flatnonzero(bins == b); cut = p['conditional_cuts'][source][str(b)]
        cell = (a['q'][ix] >= cut['q_quartiles'][1]).astype(int)
        w = np.array(cut['q2_target'])[cell]/np.bincount(cell, minlength=2)[cell]
        x = np.c_[a['scores'][ix, 0], a['F'][ix]]
        covs.append(np.cov(x.T, aweights=w, ddof=0))
        terms.append(np.stack([np.cov(a['terms'][ix, k].T, aweights=w, ddof=0) for k in range(3)]))
        if 'native_terms' in a:
            native.append(np.stack([np.cov(a['native_terms'][ix, k].T, aweights=w, ddof=0) for k in range(3)]))
    np.testing.assert_allclose(np.mean(covs, axis=0), s['cov'], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(np.mean(terms, axis=0), s['terms_cov'], rtol=1e-10, atol=1e-10)
    if native: np.testing.assert_allclose(np.mean(native, axis=0), s['native_terms_cov'], rtol=1e-10, atol=1e-10)


def check_ci(x, row):
    a = x[np.isfinite(x)]; assert len(a) == row['valid_draws']
    if len(a) >= 128: np.testing.assert_array_equal(np.quantile(a, [.025, .975]), row['CI95'])
    else: assert row['CI95'] is None


def verify():
    assert not (OUT/'verification.json').exists()
    for f in sources(): assert f.is_file(), f
    p = read(OUT/'protocol.json'); e = read(OUT/'analysis.json'); repair = read(OUT/'diagnostics/serialization_repair.json')
    hashes(p['inputs_sha256']); hashes(p['inherited_sources_sha256'])
    hashes(repair['saved_numerical_files_sha256'])
    original = PROJECT/repair['old_source_snapshot']
    assert sha(original) == p['code_sha256'] == e['numerical_code_sha256']
    current = PROJECT/'simulation/analyze_qdc_fluctuation.py'
    assert current.read_text() == original.read_text().replace('save(dest, result);', 'save(dest, serialize(result));')
    assert sha(current) == e['code_sha256'] == repair['new_analysis_sha256']
    assert sha(PROJECT/'simulation/qdc_fluctuation_common.py') == p['common_code_sha256']
    assert sha(PROJECT/'simulation/recover_qdc_fluctuation_report.py') == repair['recovery_code_sha256'] == e['recovery_code_sha256']
    assert e['protocol_sha256'] == sha(OUT/'protocol.json') and e['metrics'] == METRICS
    cohorts = 0; slices = 0; draw_vectors = 0; inherited_scalars = 0; minimum = 512
    parent_metrics = read(PARENT/'protocol.json')['metrics']
    for source, run in SOURCES:
        for seg in SEGMENTS:
            group = f'run{run}_{seg}'; raw_data = data_waves(run, seg)
            for B in p['block_sizes']:
                key = f'{group}_B{B}'; row = e['groups'][group][str(B)]
                assert row == read(OUT/'evaluation'/f'{key}.json')
                hashes(row['cohort_sha256'])
                ref = read(OUT/'references'/f'{key}.json'); assert sha(OUT/'references'/f'{key}.json') == row['reference_sha256']
                arrays = {'data': load(OUT/'arrays'/f'{key}_data.npz')}
                data = arrays['data']; raw_data['fold'] = (raw_data['ids']//B)%2
                np.testing.assert_array_equal(data['ids'], raw_data['ids']); np.testing.assert_array_equal(data['q'], raw_data['q'])
                np.testing.assert_array_equal(data['fold'], raw_data['fold'])
                for ef in [0, 1]: assert ref[str(ef)]['training_ids'] == data['ids'][data['fold'] != ef].tolist()
                assert reference(data, p, source) == ref
                slices += windows(raw_data, data)
                for v in VARIANTS:
                    a = load(OUT/'arrays'/f'{key}_{v}.npz'); arrays[v] = a
                    raw = pool([assemble(b, run, seg, B, v, raw_data, p, source) for b in BATCHES])
                    for name in ['ids', 'q', 'fold', 'scores', 'F']: np.testing.assert_allclose(a[name], raw[name], rtol=1e-10, atol=1e-10)
                    slices += windows(raw, a); del raw
                points = {}
                for v, a in arrays.items():
                    np.testing.assert_allclose(a['terms'].sum(2), a['F'], rtol=1e-12, atol=1e-12)
                    if v != 'data': np.testing.assert_allclose(a['native_terms'].sum(2), a['F'], rtol=1e-9, atol=1e-9)
                    s = summarize(a, p, source); points[v] = s; closure(s); independent_covariance(a, p, source, s); cohorts += 1
                    saved = row['data'] if v == 'data' else row['variants'][v]['summary']
                    assert serialize(s) == saved
                boot_path = OUT/'evaluation'/f'{key}_bootstrap.npz'; assert sha(boot_path) == row['bootstrap_sha256']
                boot = load(boot_path); olddraws = load(PARENT/'evaluation'/f'{key}_bootstrap.npz')['fresh_joint']
                for j, name in enumerate(METRICS[:8]):
                    np.testing.assert_allclose(boot['joint'][:, :, j], olddraws[:, :, parent_metrics.index(name)], rtol=1e-10, atol=1e-12)
                    inherited_scalars += 512*3
                for iv, v in enumerate(VARIANTS):
                    expected = compare(points[v], points['data'])
                    np.testing.assert_allclose(expected, [row['variants'][v]['point'][n] for n in METRICS], rtol=1e-12, atol=1e-12)
                    for j, n in enumerate(METRICS):
                        r = row['variants'][v]['CI95'][n]; check_ci(boot['joint'][:, iv, j], r); minimum = min(minimum, r['valid_draws'])
                        for ib, b in enumerate(p['support_bins'][source]):
                            check_ci(boot['per_Q'][:, iv, ib, j], row['variants'][v]['per_Q'][str(b)]['CI95'][n])
                    qp = compare_Q(points[v], points['data'])
                    for ib, b in enumerate(p['support_bins'][source]):
                        np.testing.assert_allclose(qp[ib], [row['variants'][v]['per_Q'][str(b)]['point'][n] for n in METRICS])
                rng = np.random.default_rng(731031900+run+100*SEGMENTS.index(seg)+B)
                old = load(RC/'arrays'/f'{key}_frozen_evaluation.npz')
                for ib in range(512):
                    dc = block_counts(data['ids'], tuple(data['span']), B, rng)
                    counts = mother_counts(arrays['frozen'], rng, [EVENTS[source]]*4)
                    mother_counts(old, rng, [int(old['n_parent'])])
                    if ib not in [0, 127, 511]: continue
                    db = summarize(data, p, source, dc)
                    for iv, v in enumerate(VARIANTS):
                        mb = summarize(arrays[v], p, source, counts)
                        np.testing.assert_allclose(compare(mb, db), boot['joint'][ib, iv], rtol=1e-12, atol=1e-12)
                        np.testing.assert_allclose(compare_Q(mb, db), boot['per_Q'][ib, iv], rtol=1e-12, atol=1e-12)
                        draw_vectors += 1
                print('Verified', key, 'independent windows, exact covariance/tails and paired draws.', flush=True)
    nbpath = PROJECT/'jupyter/codex_0001_csi_qdc_fluctuation_residuals.ipynb'
    nb = nbformat.read(nbpath, as_version=4); nbformat.validate(nb)
    code = [c for c in nb.cells if c.cell_type == 'code']
    assert all(c.execution_count for c in code) and not any(o.output_type == 'error' for c in code for o in c.outputs)
    blobs = [base64.b64decode(o.data['image/png']) for c in code for o in c.outputs if 'data' in o and 'image/png' in o.data]
    figures = read(OUT/'figure_record.json'); hashes(figures['inputs_sha256'])
    assert figures['code_sha256'] == sha(PROJECT/'simulation/report_qdc_fluctuation.py')
    assert len(blobs) == len(figures['figures']) == 5
    for blob, name in zip(blobs, figures['figures']): assert blob == (OUT/'figures'/f'{name}.png').read_bytes()
    decision = read(OUT/'decision.json'); hashes(decision['inputs_sha256']); hashes(decision['tables_sha256'])
    assert decision['code_sha256'] == figures['code_sha256']
    table_counts = {}
    for name, size in [('comparison_table.csv', 36), ('per_Q_table.csv', 162), ('batch_table.csv', 144)]:
        with (OUT/name).open(newline='') as f: tab = list(csv.DictReader(f))
        assert len(tab) == size; table_counts[name] = size
        for r in tab:
            group = e['groups'][r['group']][r['block']]
            target = group['variants'][r['variant']]
            if name == 'per_Q_table.csv': target = target['per_Q'][r['Q_bin']]
            if name == 'batch_table.csv': target = group['batches'][r['variant']][int(r['batch'])]
            for n in METRICS: assert float(r[n]) == target['point'][n]
    assert minimum >= 128
    report = {'scope': __doc__, 'inputs_checked': len(p['inputs_sha256']), 'cohorts_checked': cohorts,
        'independent_ADC_examples': slices, 'independent_highpass_window_pairs': 3*slices,
        'inherited_bootstrap_scalar_checks': inherited_scalars,
        'regenerated_new_metric_vectors': draw_vectors, 'minimum_valid_joint_draws': minimum,
        'notebook': {'cells': len(nb.cells), 'executed_cells': len(code), 'exact_embedded_pngs': len(blobs)},
        'table_rows': table_counts, 'new_independent_mothers': 0, 'new_native_replays': 0,
        'default_changed': False, 'official_PID_changed': False,
        'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'nbformat': nbformat.__version__}}
    report['outputs_sha256'] = {relative(f): sha(f) for f in sorted(OUT.rglob('*')) if f.is_file() and 'source_snapshot' not in f.parts}
    report['outputs_sha256'][relative(nbpath)] = sha(nbpath)
    report['sources_sha256'] = {relative(f): sha(f) for f in sources()}
    save(OUT/'verification.json', report); print('All fluctuation, tail, provenance and notebook checks passed.', flush=True)


def archive():
    assert not (OUT/'record.json').exists()
    v = read(OUT/'verification.json'); hashes(v['outputs_sha256']); hashes(v['sources_sha256'])
    snapshots = {}
    for f in sources():
        dest = OUT/'source_snapshot'/f.relative_to(PROJECT); dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dest); snapshots[relative(f)] = sha(dest)
    outputs = {relative(f): sha(f) for f in sorted(OUT.rglob('*')) if f.is_file() and 'source_snapshot' not in f.parts}
    nb = PROJECT/'jupyter/codex_0001_csi_qdc_fluctuation_residuals.ipynb'; outputs[relative(nb)] = sha(nb)
    save(OUT/'record.json', {'scope': __doc__, 'outputs_sha256': outputs, 'sources_sha256': snapshots,
        'parent_record_sha256': sha(PARENT/'record.json'), 'new_independent_mothers': 0, 'new_native_replays': 0,
        'default_changed': False, 'official_PID_changed': False})
    print('Archived', len(outputs), 'outputs and', len(snapshots), 'sources.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--archive', action='store_true')
    args = parser.parse_args(); archive() if args.archive else verify()
