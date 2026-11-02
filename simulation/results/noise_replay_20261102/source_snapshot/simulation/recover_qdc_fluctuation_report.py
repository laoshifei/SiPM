"""Recover JSON summaries from unchanged completed arrays after serialization repair."""
from qdc_fluctuation_common import *
from analyze_qdc_fluctuation import compare_Q


def main():
    assert not (OUT/'analysis.json').exists()
    p = read(OUT/'protocol.json'); repair = read(OUT/'diagnostics/serialization_repair.json')
    assert p['code_sha256'] == repair['old_analysis_sha256']
    old = (PROJECT/repair['old_source_snapshot']).read_text()
    current = (PROJECT/'simulation/analyze_qdc_fluctuation.py').read_text()
    assert current == old.replace('save(dest, result);', 'save(dest, serialize(result));')
    for path, value in repair['saved_numerical_files_sha256'].items(): assert sha(PROJECT/path) == value, path
    inherited = read(PARENT/'analysis.json'); parent_metrics = read(PARENT/'protocol.json')['metrics']
    groups = {}
    for source, run in SOURCES:
        for seg in SEGMENTS:
            group = f'run{run}_{seg}'; groups[group] = {}
            for B in p['block_sizes']:
                key = f'{group}_B{B}'; dest = OUT/'evaluation'/f'{key}.json'; assert not dest.exists()
                data = load(OUT/'arrays'/f'{key}_data.npz'); ds = summarize(data, p, source); closure(ds)
                bs = load(OUT/'evaluation'/f'{key}_bootstrap.npz'); draws = bs['joint']; perq = bs['per_Q']
                parent = inherited['groups'][group][str(B)]
                olddraws = load(PARENT/'evaluation'/f'{key}_bootstrap.npz')['fresh_joint']
                for i, name in enumerate(METRICS[:8]):
                    np.testing.assert_allclose(draws[:, :, i], olddraws[:, :, parent_metrics.index(name)], rtol=1e-10, atol=1e-12)
                result = {'scope': __doc__, 'code_sha256': sha(PROJECT/'simulation/analyze_qdc_fluctuation.py'),
                    'numerical_code_sha256': p['code_sha256'], 'recovery_code_sha256': sha(__file__),
                    'protocol_sha256': sha(OUT/'protocol.json'), 'group': group, 'block': B, 'source': source,
                    'data': ds, 'variants': {}, 'batches': {}, 'cohort_sha256': {},
                    'reference_sha256': sha(OUT/'references'/f'{key}.json'),
                    'bootstrap_sha256': sha(OUT/'evaluation'/f'{key}_bootstrap.npz'),
                    'parent_metric_draw_closure': {'metrics': 8, 'draws': 512, 'variants': 3}}
                for iv, v in enumerate(VARIANTS):
                    path = OUT/'arrays'/f'{key}_{v}.npz'; a = load(path)
                    s = summarize(a, p, source); closure(s); values = compare(s, ds); per_point = compare_Q(s, ds)
                    for i, name in enumerate(METRICS[:8]):
                        np.testing.assert_allclose(values[i], parent['variants'][v]['point'][name], rtol=1e-10, atol=1e-12)
                    result['cohort_sha256'][relative(path)] = sha(path)
                    result['variants'][v] = {'summary': s, 'point': dict(zip(METRICS, values)),
                        'CI95': {n: interval(draws[:, iv, j]) for j, n in enumerate(METRICS)},
                        'per_Q': {str(b): {'point': dict(zip(METRICS, per_point[k])),
                            'CI95': {n: interval(perq[:, iv, k, j]) for j, n in enumerate(METRICS)}}
                            for k, b in enumerate(p['support_bins'][source])}}
                    result['batches'][v] = []
                    for batch in BATCHES:
                        sel = a['batch'] == batch
                        ba = {k: x[sel] for k, x in a.items() if x.ndim}
                        ba['ids'] = ba['ids']-batch*EVENTS[source]; ba['n_parent'] = np.array(EVENTS[source])
                        sb = summarize(ba, p, source)
                        result['batches'][v].append({'batch': batch, 'summary': sb,
                            'point': dict(zip(METRICS, compare(sb, ds))) if sb else None})
                save(dest, serialize(result)); groups[group][str(B)] = read(dest)
                print('Recovered', key, 'from completed numerical arrays; parent512-draw closure passed.', flush=True)
    repair['new_analysis_sha256'] = sha(PROJECT/'simulation/analyze_qdc_fluctuation.py')
    repair['recovery_code_sha256'] = sha(__file__)
    repair['completed_without_bootstrap_recalculation'] = True
    save(OUT/'diagnostics/serialization_repair.json', repair)
    save(OUT/'analysis.json', {'scope': __doc__, 'code_sha256': repair['new_analysis_sha256'],
        'numerical_code_sha256': p['code_sha256'], 'recovery_code_sha256': sha(__file__),
        'protocol_sha256': sha(OUT/'protocol.json'), 'metrics': METRICS, 'groups': groups,
        'new_independent_mothers': 0, 'new_native_replays': 0, 'default_changed': False, 'official_PID_changed': False})


if __name__ == '__main__': main()
