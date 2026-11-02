"""Verify signed ADC kernels, inherited variances, weighted U algebra and notebook."""
import argparse
import base64
import csv
import platform
import shutil
import nbformat
from gaussian_closure_common import *
from analyze_gaussian_closure import counts_for_context
from test_gaussian_closure import run as algebra_tests
from analyze_acquisition_state import rows, interval


def hashes(items):
    for name, value in items.items(): assert sha(PROJECT/name) == value, name


def source_paths():
    p = read(OUT/'protocol.json')
    names = set(p['inherited_sources_sha256']) | set(p['code_sha256']) | {
        'simulation/report_gaussian_closure.py', 'simulation/verify_gaussian_closure.py',
        'simulation/README.md', 'jupyter/codex_0001_build_gaussian_closure_notebook.py'}
    return [PROJECT/n for n in sorted(names)]


def direct_windows(raw, compact, kernels, cells):
    checked = 0
    for cell in cells:
        key = cell['key']; indices = kernels[key+'_indices']
        take = np.unique(np.linspace(0, len(indices)-1, min(3, len(indices))).astype(int))
        direct = []
        for k in take:
            i = indices[k]; y = raw['waves'][i].astype(float)-raw['baseline'][i]
            pre = int(np.ceil(raw['t10'][i]-300)); post = int(np.ceil(raw['t10'][i]+250))
            mu = y[post:post+1000].mean(); vec = []
            for width in [25, 51, 101]:
                r = y-np.convolve(y, np.ones(width)/width, mode='same')
                a = r[post:post+1000]; b = r[pre:pre+200]
                vec.append(np.r_[(a-a.mean())/np.sqrt(999), (b-b.mean())/np.sqrt(199)])
            direct.append(vec)
            np.testing.assert_allclose(mu, compact['detail'][i, :, 0], rtol=1e-10, atol=1e-9)
            checked += 1
        z = np.array(direct)
        for scale in range(3):
            x = z[:, scale]; expected = x[:, :1000]@x[:, :1000].T-x[:, 1000:]@x[:, 1000:].T
            np.testing.assert_allclose(expected, kernels[key+'_gram'][scale][np.ix_(take, take)], rtol=1e-8, atol=1e-8)
    return checked


def independent_variance(a, p, source, counts):
    rr = rows(a, p, source, counts)
    assert rr is not None
    f = a['F']; fixed = (a['detail'][:, :, 2]-a['detail'][:, :, 1])/a['mu_reference'][:, None]
    return np.stack([np.mean([np.diag(np.cov(x[ix].T, aweights=w, ddof=0))
                        for _, ix, w, _ in rr], 0) for x in [f, fixed]])


def verify():
    assert not (OUT/'verification.json').exists()
    p = read(OUT/'protocol.json'); e = read(OUT/'analysis.json')
    hashes(p['inputs_sha256']); hashes(p['code_sha256']); hashes(p['inherited_sources_sha256'])
    assert serialize(algebra_tests()) == read(OUT/'algebra_tests.json')
    assert e['protocol_sha256'] == sha(OUT/'protocol.json')
    ncohort = 0; nwindows = 0; inherited = 0; regenerated = 0
    unestimable = 0; negative = 0; min_valid_primary = 512
    for source, run in SOURCES:
        for seg in SEGMENTS:
            group = f'run{run}_{seg}'; raw_data = data_waves(run, seg)
            for B in p['block_sizes']:
                key = f'{group}_B{B}'; row = e['groups'][group][str(B)]
                assert row == read(OUT/'evaluation'/f'{key}.json')
                hashes(row['kernel_sha256']); bp = OUT/'evaluation'/f'{key}_bootstrap.npz'
                assert sha(bp) == row['bootstrap_sha256']; boot = load(bp)
                compact = {v: load(QDC/'arrays'/f'{key}_{v}.npz') for v in ['data', *VARIANTS]}
                dc, mc = counts_for_context(compact['data'], compact['frozen'], group, B, source)
                np.testing.assert_array_equal(dc, boot['data_counts']); np.testing.assert_array_equal(mc, boot['MC_counts'])
                raw_data['fold'] = (raw_data['ids']//B)%2
                old = load(QDC/'evaluation'/f'{key}_bootstrap.npz')['joint']
                for v in ['data', *VARIANTS]:
                    a = compact[v]; saved = row['cohorts'][v]; count = dc if v == 'data' else mc
                    s = boot[v+'_stats']; q = boot[v+'_per_Q']; kernel = load(OUT/'kernels'/f'{key}_{v}.npz')
                    raw = raw_data if v == 'data' else pool([assemble(b, run, seg, B, v, raw_data, p, source) for b in BATCHES])
                    for name in ['ids', 'q', 'fold']: np.testing.assert_array_equal(a[name], raw[name])
                    nwindows += direct_windows(raw, a, kernel, saved['cells'])
                    if v != 'data': del raw
                    select = [0, 1, 128, 512]
                    sr, qr = aggregate(kernel, saved['cells'], count[:, select], p, source)
                    np.testing.assert_allclose(sr, s[:, :, :, select], rtol=1e-10, atol=1e-10, equal_nan=True)
                    np.testing.assert_allclose(qr, q[:, :, :, :, select], rtol=1e-10, atol=1e-10, equal_nan=True)
                    for draw in select:
                        observed = independent_variance(a, p, source, count[:, draw])
                        np.testing.assert_allclose(s[:, :, :, draw, 0], np.stack([observed]*2), rtol=1e-8, atol=1e-10)
                        regenerated += 1
                    np.testing.assert_allclose(s[..., 3]+s[..., 4], s[..., 0], rtol=1e-9, atol=1e-10)
                    np.testing.assert_allclose(s[..., 0]-s[..., 1], s[..., 6], rtol=1e-9, atol=1e-10, equal_nan=True)
                    assert serialize(point_rows(s[:, :, :, 0])) == saved['summary']
                    assert saved['unestimable_U_point_entries'] == int(np.sum(~np.isfinite(s[:, :, :, 0, 1])))
                    assert saved['negative_U_point_entries'] == int(np.sum(s[:, :, :, 0, 1] < 0))
                    unestimable += saved['unestimable_U_point_entries']; negative += saved['negative_U_point_entries']
                    ncohort += 1
                    if v == 'data': continue
                    z = boot[v+'_contrast']; qz = boot[v+'_per_Q_contrast']
                    np.testing.assert_allclose(z, contrast(s, boot['data_stats']), equal_nan=True)
                    for k in range(3):
                        np.testing.assert_allclose(z[:, 0, k, 1:, 0], np.tile(old[:, VARIANTS.index(v), 5+k], (2, 1)), rtol=1e-8, atol=1e-10)
                        inherited += 512*2
                    for ic, cond in enumerate(CONDITIONS):
                        for im, norm in enumerate(NORMS):
                            for k, scale in enumerate(SCALES):
                                r = saved['contrasts'][cond][norm][str(scale)]
                                np.testing.assert_allclose([r['point'][n] if r['point'][n] is not None else np.nan for n in METRICS], z[ic, im, k, 0], equal_nan=True)
                                for j, name in enumerate(METRICS):
                                    assert interval(z[ic, im, k, 1:, j]) == r['CI95'][name]
                                    if ic == 0: min_valid_primary = min(min_valid_primary, r['CI95'][name]['valid_draws'])
                                    for ib, b in enumerate(p['support_bins'][source]):
                                        target = saved['per_Q_contrasts'][str(b)][cond][norm][str(scale)]
                                        assert interval(qz[ic, im, k, ib, 1:, j]) == target['CI95'][name]
                print('Verified', key, 'ADC kernels, all parent draws, sparse cells and U contrasts.', flush=True)
    nbpath = PROJECT/'jupyter/codex_0001_csi_gaussian_closure.ipynb'
    nb = nbformat.read(nbpath, as_version=4); nbformat.validate(nb)
    code = [c for c in nb.cells if c.cell_type == 'code']
    assert all(c.execution_count for c in code)
    assert not any(o.output_type == 'error' for c in code for o in c.outputs)
    blobs = [base64.b64decode(o.data['image/png']) for c in code for o in c.outputs if 'data' in o and 'image/png' in o.data]
    figs = read(OUT/'figure_record.json'); assert len(blobs) == len(figs['figures'])
    for blob, name in zip(blobs, figs['figures']): assert blob == (OUT/'figures'/f'{name}.png').read_bytes()
    hashes(figs['inputs_sha256']); decision = read(OUT/'decision.json'); hashes(decision['inputs_sha256'])
    assert figs['code_sha256'] == decision['code_sha256'] == sha(PROJECT/'simulation/report_gaussian_closure.py')
    table_counts = {}
    for name, expected in [('comparison_table.csv', 432), ('per_Q_table.csv', 1944), ('cell_counts.csv', 432)]:
        with (OUT/name).open(newline='') as f: tab = list(csv.DictReader(f))
        assert len(tab) == expected, (name, len(tab)); table_counts[name] = len(tab)
        if name == 'cell_counts.csv': continue
        for item in tab:
            r = e['groups'][item['group']][item['block']]['cohorts'][item['variant']]
            target = r['contrasts'] if name == 'comparison_table.csv' else r['per_Q_contrasts'][item['Q_bin']]
            target = target[item['conditioning']][item['normalization']][item['scale_ns']]
            for n in METRICS:
                assert item[n] == '' if target['point'][n] is None else float(item[n]) == target['point'][n]
    report = {'scope': __doc__, 'inputs_checked': len(p['inputs_sha256']), 'cohorts_checked': ncohort,
        'independent_ADC_examples': nwindows, 'independent_highpass_window_pairs': 3*nwindows,
        'inherited_F_SD_scalar_checks': inherited, 'regenerated_cohort_draws': regenerated,
        'unestimable_U_point_entries': unestimable, 'negative_U_point_entries': negative,
        'minimum_primary_valid_draws': min_valid_primary, 'table_rows': table_counts,
        'notebook': {'cells': len(nb.cells), 'executed_cells': len(code), 'exact_embedded_pngs': len(blobs)},
        'new_independent_mothers': 0, 'new_native_replays': 0, 'default_changed': False, 'official_PID_changed': False,
        'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'nbformat': nbformat.__version__},
        'outputs_sha256': {relative(f): sha(f) for f in sorted(OUT.rglob('*')) if f.is_file() and 'source_snapshot' not in f.parts},
        'sources_sha256': {relative(f): sha(f) for f in source_paths()}}
    report['outputs_sha256'][relative(nbpath)] = sha(nbpath)
    save(OUT/'verification.json', report); print('All Gaussian closure checks passed.', flush=True)


def archive():
    assert not (OUT/'record.json').exists()
    v = read(OUT/'verification.json'); hashes(v['outputs_sha256']); hashes(v['sources_sha256'])
    for f in source_paths():
        dest = OUT/'source_snapshot'/f.relative_to(PROJECT); dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dest); assert sha(dest) == sha(f)
    outputs = dict(v['outputs_sha256']); outputs[relative(OUT/'verification.json')] = sha(OUT/'verification.json')
    save(OUT/'record.json', {'scope': __doc__, 'outputs_sha256': outputs, 'sources_sha256': v['sources_sha256'],
        'parent_record_sha256': sha(QDC/'record.json'), 'new_independent_mothers': 0,
        'new_native_replays': 0, 'default_changed': False, 'official_PID_changed': False})
    print('Archived', len(outputs), 'outputs and', len(v['sources_sha256']), 'source files.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--archive', action='store_true'); args = parser.parse_args()
    archive() if args.archive else verify()
