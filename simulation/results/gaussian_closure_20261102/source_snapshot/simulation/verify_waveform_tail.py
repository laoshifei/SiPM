"""Audit raw ADC, independent gate algebra, fold isolation and shape variance closure."""
import argparse
import base64
import csv
import shutil
import nbformat
from scipy.ndimage import uniform_filter1d
from waveform_tail_common import *
from evaluate_refined_rc import selected_arrays
from reference import bin_layout
from verify_acquisition_state import independent_prefix_scores, check_interval


def sources():
    names = ['waveform_tail_common.py', 'prepare_waveform_tail.py', 'analyze_waveform_tail.py',
             'plot_waveform_tail.py', 'summarize_waveform_tail.py', 'verify_waveform_tail.py',
             'pid_core_tail_common.py', 'rc_refinement_common.py', 'evaluate_refined_rc.py',
             'state_response_common.py', 'analyze_acquisition_state.py', 'prepare_acquisition_state.py',
             'verify_acquisition_state.py', 'prepare_state_response.py', 'extract_pid.py',
             'reference.py', 'README.md']
    return [PROJECT/'simulation'/n for n in names] + [
        PROJECT/'analysis/paper_baseline/publication_stats.py',
        PROJECT/'jupyter/codex_0001_build_waveform_tail_notebook.py']


def check_hashes(mapping):
    for path, value in mapping.items():
        assert sha(PROJECT/path) == value, path
    return len(mapping)


def check_adc(a, waves):
    """Prefix-sum gate integrals are independent of the production gate matrix."""
    f = wave_features(waves)
    sf = wave_features(uniform_filter1d(waves.astype(float), size=11, axis=1, mode='nearest'))
    anchor = np.floor(f['t10_sample']).astype(int) + 1
    smooth_anchor = np.floor(sf['t10_sample']).astype(int) + 1
    recent = waves[np.arange(len(waves))[:, None], anchor[:, None]+np.arange(-192, -64)].mean(1)
    for name, value in [('anchor', anchor), ('smooth_anchor', smooth_anchor), ('recent_baseline', recent),
                        ('baseline', f['baseline']), ('baseline_rms', f['baseline_rms'])]:
        np.testing.assert_array_equal(a[name], value)
    for c, name in enumerate(CONTROLS):
        an = smooth_anchor if name in ['smooth_anchor', 'recent_and_smooth'] else anchor
        base = recent if name in ['recent_baseline', 'recent_and_smooth'] else f['baseline']
        scores, q, shape = independent_prefix_scores(waves, an, base, PCA)
        np.testing.assert_allclose(scores, a['scores'][:, c], rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(q, a['control_q'][:, c], rtol=1e-12, atol=1e-8)
        if c == 0:
            np.testing.assert_allclose(shape, a['shape'], rtol=1e-11, atol=1e-13)
            mean = shape*q[:, None]/20
            np.testing.assert_allclose(mean, a['mean_wave'], rtol=1e-11, atol=1e-11)
    delta = recent-f['baseline']
    t = (np.arange(256)-127.5)*.002
    slope = waves[:, :256].astype(float)@t/(t@t)
    obs = np.c_[delta, delta/f['baseline_rms'], slope, 2*(smooth_anchor-anchor),
                2*(sf['t10_sample']-f['t10_sample']), np.zeros(len(waves)), np.zeros(len(waves)),
                np.maximum(0, np.diff(mean[:, 16:71], axis=1).max(1))/f['amplitude']]
    # The 500 ns boundary falls halfway through a 40 ns bin, so use raw sums here.
    for j, lo, hi in [(5, 250, 750), (6, 1000, 1500)]:
        signal = waves[np.arange(len(waves))[:, None], anchor[:, None]+np.arange(lo, hi)]-f['baseline'][:, None]
        obs[:, j] = signal.sum(1)/a['control_q'][:, 0]
    np.testing.assert_allclose(obs, a['observables'], rtol=1e-10, atol=1e-11)


def direct_mode_statistics(a, p, source, design, fold):
    """Project full direct 76-by-76 covariance, without saved event projections."""
    rr = rows(a, p, source)
    pc = np.zeros((3, 3)); whole = 0.; bias = 0.
    for b, ix, w, _ in rr:
        for ef in [0, 1]:
            use = fold[ix] == ef
            mass = w[use].sum()
            if not mass: continue
            u = ix[use]; weight = w[use]; shape = a['shape'][u]
            mu = np.average(shape, axis=0, weights=weight)
            centered = shape-mu
            cov = centered.T@(weight[:, None]*centered)
            basis = np.array(design[str(ef)]['vectors'])
            pc += basis.T@cov@basis
            whole += np.trace(cov)
            delta = mu-np.array(design[str(ef)]['templates'][str(b)])
            bias += mass*np.dot(delta, delta)
    pc /= len(rr); whole /= len(rr); bias /= len(rr)
    parallel = np.trace(pc)
    return {'mode_covariance': pc, 'mode_variance': np.diag(pc), 'parallel': parallel,
            'orthogonal': whole-parallel, 'whole': whole, 'captured_fraction': parallel/whole,
            'template_mean_squared_bias': bias}


def archive():
    assert not (OUT/'record.json').exists()
    v = read(OUT/'verification.json')
    check_hashes(v['outputs_sha256']); check_hashes(v['sources_sha256'])
    snap = {}
    for f in sources():
        target = OUT/'source_snapshot'/f.relative_to(PROJECT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, target); snap[relative(f)] = sha(target)
    files = {relative(f): sha(f) for f in sorted(OUT.rglob('*')) if f.is_file() and 'source_snapshot' not in f.parts}
    nb = PROJECT/'jupyter/codex_0001_csi_waveform_tail.ipynb'
    files[relative(nb)] = sha(nb)
    save(OUT/'record.json', {'scope': __doc__, 'outputs_sha256': files, 'sources_sha256': snap,
         'checkpoint_commit': read(OUT/'protocol.json')['checkpoint_commit'],
         'parent_record_sha256': sha(PARENT/'record.json'), 'unique_original_mothers': 470,
         'supported_original_mothers': 456, 'new_independent_mothers': 0, 'new_native_replays': 0,
         'default_changed': False, 'official_PID_changed': False})
    print('Archived', len(files), 'outputs and', len(snap), 'sources.', flush=True)


def main():
    assert not (OUT/'verification.json').exists()
    p = read(OUT/'protocol.json'); e = read(OUT/'analysis.json')
    previous = read(PARENT/'diagnostics.json'); boot = load(OUT/'bootstrap.npz')
    examples = read(OUT/'examples.json')['groups']
    mean = read(RC/'mean_calibration.json'); noise = read(RC/'noise_selection.json')
    report = {'scope': __doc__, 'inputs_checked': check_hashes(p['inputs_sha256'])}
    assert p['code_sha256'] == sha(PROJECT/'simulation/prepare_waveform_tail.py')
    assert p['common_code_sha256'] == e['common_code_sha256'] == sha(PROJECT/'simulation/waveform_tail_common.py')
    assert e['code_sha256'] == sha(PROJECT/'simulation/analyze_waveform_tail.py')
    assert e['protocol_sha256'] == sha(OUT/'protocol.json')
    assert read(OUT/'extraction.json')['code_sha256'] == p['code_sha256']
    raw_cohorts = 0; mode_partitions = 0; fences = 0; example_records = 0; minimum = 512
    for source, run in SOURCES:
        path = PROJECT/f'data/DT5730_DPP_PHA/DAQ/run_{run}/UNFILTERED/Data_run_{run}.BIN'
        _, dtype, n = bin_layout(path)
        raw = np.memmap(path, dtype=dtype, offset=2, mode='r', shape=(n,))
        for seg in SEGMENTS:
            key = f'run{run}_{seg}'; data = load(OUT/'arrays'/f'{key}_data.npz')
            old = load(PARENT/'arrays'/f'{key}_data.npz')
            for name in ['ids', 'q']: np.testing.assert_array_equal(data[name], old[name])
            np.testing.assert_allclose(data['scores'][:, 0], old['scores'], rtol=1e-12, atol=1e-12)
            waves = raw['data'][data['ids']]
            check_adc(data, waves); raw_cohorts += 1
            for B in p['block_sizes']:
                design = read(OUT/'designs'/f'{key}_B{B}.json')
                assert design == training_design(data, p, source, B)
                for ef in [0, 1]:
                    d = design[str(ef)]; train = np.array(d['training_ids']); test = np.array(d['evaluation_ids'])
                    assert not set(train)&set(test)
                    assert np.all((train//B)%2 != ef) and np.all((test//B)%2 == ef)
                    u = np.array(d['vectors']); cov = np.array(d['covariance']); values = np.array(d['eigenvalues'])
                    np.testing.assert_allclose(u.T@u, np.eye(3), atol=1e-12)
                    np.testing.assert_allclose(cov@u, u*values[:3], rtol=1e-10, atol=1e-15)
                    np.testing.assert_allclose(u.sum(0), 0., atol=1e-10)
                    assert np.all(np.diff(values) <= 1e-15)
                    fences += 1
                row = e['groups'][key][str(B)]
                for variant in ['data', *VARIANTS]:
                    if variant == 'data':
                        a = data; adc = waves; fold = (a['ids']//B)%2
                    else:
                        a = load(OUT/'arrays'/f'{key}_B{B}_{variant}.npz')
                        native = selected_arrays(run, seg, B, variant, mean, noise)
                        adc = native['waves']; fold = a['fold']
                        for name in ['ids', 'q']: np.testing.assert_array_equal(a[name], native[name])
                        np.testing.assert_array_equal(fold, load(STATE/'arrays'/f'{key}_alignment_B{B}.npz')['mother_assignment'])
                        np.testing.assert_allclose(a['scores'][:, 0], native['scores'], atol=1e-12)
                        check_adc(a, adc); raw_cohorts += 1
                    saved = load(OUT/'arrays'/f'{key}_B{B}_{variant}_modes.npz')
                    applied = apply_design(a, p, source, design, fold)
                    np.testing.assert_array_equal(saved['ids'], a['ids'])
                    for name, value in applied.items(): np.testing.assert_array_equal(saved[name], value)
                    direct = direct_mode_statistics(a, p, source, design, fold)
                    recorded = row['data_modes'] if variant == 'data' else row['variants'][variant]['native_modes']
                    for name, value in direct.items():
                        np.testing.assert_allclose(value, recorded[name], rtol=1e-10, atol=1e-14)
                    controls = control_statistics(a, p, source)
                    recorded_controls = row['data_controls'] if variant == 'data' else row['variants'][variant]['native_controls']
                    for name, value in controls.items():
                        np.testing.assert_allclose(value, recorded_controls[name], rtol=1e-12, atol=1e-12)
                    mode_partitions += 1
                    z = load(PARENT/'arrays'/f'{key}_B{B}_{variant}_z.npz')
                    np.testing.assert_array_equal(z['ids'], a['ids'])
                    np.testing.assert_array_equal(z['fold'], fold)
                    assert serialize(feature_audit(a, p, source, applied, z['z'])) == row['feature_audits'][variant]
                    if seg == 'last20' and B == 128 and variant in ['data', 'spectrum_CV_fit']:
                        name = f'{key}_{variant}'; selected = select_examples(a, p, source, z['z'])
                        assert selected == examples[name]
                        ex = load(OUT/'arrays'/f'{name}_examples.npz')
                        np.testing.assert_array_equal(ex['ids'], [r['event_id'] for r in selected])
                        np.testing.assert_array_equal(ex['waves'], adc[[r['index'] for r in selected]])
                        example_records += len(selected)
                    if variant == 'data': continue
                    iv = VARIANTS.index(variant); r = row['variants'][variant]
                    prior = previous['groups'][key][str(B)]['variants'][variant]
                    point = control_contrasts(controls, {k: np.array(v) for k, v in row['data_controls'].items()})
                    mode_point = mode_contrasts(direct, {k: np.array(v) for k, v in row['data_modes'].items()})
                    for c, control in enumerate(CONTROLS):
                        stat = r['controls'][control]
                        for j, name in enumerate(CONTROL_NAMES):
                            np.testing.assert_allclose(point[c, j], stat['point'][name], rtol=1e-12, atol=1e-12)
                            bs = boot[f'{key}_B{B}_controls'][:, iv, c, j]
                            base = boot[f'{key}_B{B}_controls'][:, iv, 0, j]
                            check_interval(bs, stat['CI95'][name])
                            check_interval(bs-base, stat['paired_from_original'][name])
                            minimum = min(minimum, stat['CI95'][name]['valid_draws'])
                            if c == 0 and name != 'mean_time_delta_ns':
                                np.testing.assert_allclose(stat['point'][name], prior['point'][name], atol=1e-12, rtol=1e-12)
                                np.testing.assert_allclose(stat['CI95'][name]['CI95'], prior['CI95'][name]['CI95'], atol=1e-12, rtol=1e-12)
                    for j, name in enumerate(MODE_NAMES):
                        np.testing.assert_allclose(mode_point[j], r['mode_point'][name], rtol=1e-10, atol=1e-12)
                        check_interval(boot[f'{key}_B{B}_modes'][:, iv, j], r['mode_CI95'][name])
                        minimum = min(minimum, r['mode_CI95'][name]['valid_draws'])
                print('Verified', key, B, 'original ADC, gates, folds, modes and intervals', flush=True)
        del raw
    nbpath = PROJECT/'jupyter/codex_0001_csi_waveform_tail.ipynb'
    nb = nbformat.read(nbpath, as_version=4); nbformat.validate(nb)
    code = [c for c in nb.cells if c.cell_type == 'code']
    assert all(c.execution_count for c in code)
    assert not any(o.output_type == 'error' for c in code for o in c.outputs)
    images = [base64.b64decode(o.data['image/png']) for c in code for o in c.outputs if 'data' in o and 'image/png' in o.data]
    figs = read(OUT/'figure_record.json')
    assert len(images) == len(figs['figures']) == 5
    for blob, name in zip(images, figs['figures']): assert blob == (OUT/'figures'/f'{name}.png').read_bytes()
    check_hashes(figs['inputs_sha256'])
    assert figs['code_sha256'] == sha(PROJECT/'simulation/plot_waveform_tail.py')
    decision = read(OUT/'decision.json')
    check_hashes(decision['inputs_sha256']); check_hashes(decision['tables_sha256'])
    assert decision['code_sha256'] == sha(PROJECT/'simulation/summarize_waveform_tail.py')
    with (OUT/'control_table.csv').open(newline='') as f:
        table = list(csv.DictReader(f))
    assert len(table) == 144
    for r in table:
        stat = e['groups'][r['group']][r['raw_block_size']]['variants'][r['variant']]['controls'][r['control']]
        for name in CONTROL_NAMES:
            assert float(r[name]) == stat['point'][name]
            assert [float(r[name+'_low']), float(r[name+'_high'])] == stat['CI95'][name]['CI95']
    assert example_records == decision['example_records'] == 36
    assert decision['unique_example_events'] == len({(key, r['event_id']) for key, rr in examples.items() for r in rr}) == 32
    for name in ['new_independent_mothers', 'new_native_replays', 'default_changed']:
        assert not p[name] and not e[name] and not decision[name]
    assert not decision['official_PID_changed'] and minimum >= 128
    report.update(raw_ADC_cohorts=raw_cohorts, independent_gate_controls=4*raw_cohorts,
                  opposite_block_designs=fences, direct_covariance_partitions=mode_partitions,
                  exact_example_records=example_records, inherited_width_SD_points_intervals_exact=True,
                  minimum_valid_bootstrap_draws=minimum,
                  notebook={'cells': len(nb.cells), 'executed_cells': len(code), 'exact_embedded_pngs': len(images)},
                  new_independent_mothers=0, new_native_replays=0, default_changed=False, official_PID_changed=False)
    report['outputs_sha256'] = {relative(f): sha(f) for f in sorted(OUT.rglob('*')) if f.is_file() and 'source_snapshot' not in f.parts}
    report['outputs_sha256'][relative(nbpath)] = sha(nbpath)
    report['sources_sha256'] = {relative(f): sha(f) for f in sources()}
    save(OUT/'verification.json', report)
    print('All original-ADC, independent gate, shape variance and notebook checks passed.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', action='store_true')
    args = parser.parse_args()
    archive() if args.archive else main()
