"""Verify new transport provenance, native ADC identity, fixed metrics and paired intervals."""
import argparse
import base64
import csv
import json
import platform
import shutil
import ROOT
import nbformat
from mc_statistics_common import *
from analyze_mc_statistics import compact, prepared
from verify_acquisition_state import independent_prefix_scores, check_interval
from verify_waveform_tail import direct_mode_statistics
from publication_stats import block_counts


def sources():
    names = ['mc_statistics_common.py', 'prepare_mc_statistics.py', 'run_mc_statistics.py', 'continue_mc_statistics.py',
             'analyze_mc_statistics.py', 'plot_mc_statistics.py', 'summarize_mc_statistics.py',
             'verify_mc_statistics.py', 'extract_npe_profile.py', 'compare.py', 'reference.py',
             'extract_pid.py', 'refine_waveforms.py', 'diagnose_fluctuations.py', 'validate.py',
             'state_response_common.py', 'prepare_state_response.py', 'prepare_acquisition_state.py',
             'analyze_acquisition_state.py', 'pid_core_tail_common.py', 'rc_refinement_common.py',
             'evaluate_refined_rc.py', 'evaluate_state_response.py', 'waveform_tail_common.py',
             'verify_acquisition_state.py', 'verify_waveform_tail.py', 'replay_readout.cc',
             'src/ReadoutSampling.cc', 'src/ReadoutResponse.cc', 'src/ReadoutNoise.cc',
             'src/SignalDependentNoise.cc', 'vendor/SiPMResponse.cc', 'README.md']
    return [PROJECT/'simulation'/n for n in names]+[
        PROJECT/'analysis/paper_baseline/publication_stats.py',
        PROJECT/'jupyter/codex_0001_build_mc_statistics_notebook.py']


def hashes(mapping):
    for path, value in mapping.items(): assert sha(PROJECT/path) == value, path
    return len(mapping)


def check_array(a):
    s, q, shape = independent_prefix_scores(a['waves'], a['anchor'], a['baseline'], PCA)
    np.testing.assert_allclose(s, a['scores'], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(q, a['qcalc'], rtol=1e-12, atol=1e-8)
    np.testing.assert_allclose(shape, a['shape'], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(shape.sum(1), 1., atol=1e-12)
    np.testing.assert_allclose(shape*q[:, None]/20, a['mean_wave'], atol=1e-10)


def archive():
    assert not (OUT/'record.json').exists()
    v = read(OUT/'verification.json')
    hashes(v['outputs_sha256']); hashes(v['sources_sha256']); snapshots = {}
    for f in sources():
        target = OUT/'source_snapshot'/f.relative_to(PROJECT)
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(f, target)
        snapshots[relative(f)] = sha(target)
    outputs = {relative(f): sha(f) for f in sorted(OUT.rglob('*')) if f.is_file() and 'source_snapshot' not in f.parts}
    nb = PROJECT/'jupyter/codex_0001_csi_independent_mc_statistics.ipynb'; outputs[relative(nb)] = sha(nb)
    save(OUT/'record.json', {'scope': __doc__, 'outputs_sha256': outputs, 'sources_sha256': snapshots,
        'parent_record_sha256': sha(WAVE/'record.json'), 'native_parent_record_sha256': sha(JOINT/'record.json'),
        'binary_sha256': read(OUT/'protocol.json')['binary_sha256'], 'new_independent_mothers': 256000,
        'replay_entries': read(OUT/'replay_execution.json')['replay_entries'],
        'new_experimental_events': 0, 'default_changed': False, 'official_PID_changed': False,
        'source_boundary': 'Native full-transport binary is the unchanged prior joint-candidate binary. Its original source provenance is linked by native_parent_record_sha256; current downstream sources are snapshotted separately.'})
    print('Archived', len(outputs), 'outputs and', len(snapshots), 'sources.', flush=True)


def main():
    assert not (OUT/'verification.json').exists()
    p = read(OUT/'protocol.json'); e = read(OUT/'analysis.json')
    report = {'scope': __doc__, 'inputs_checked': hashes(p['inputs_sha256']),
              'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                              'ROOT': ROOT.gROOT.GetVersion(), 'nbformat': nbformat.__version__}}
    hashes(p['binary_sha256'])
    assert p['code_sha256'] == sha(PROJECT/'simulation/prepare_mc_statistics.py')
    assert p['common_code_sha256'] == sha(PROJECT/'simulation/mc_statistics_common.py')
    assert e['code_sha256'] == sha(PROJECT/'simulation/analyze_mc_statistics.py')
    assert e['protocol_sha256'] == sha(OUT/'protocol.json') and e['metrics'] == METRICS
    definition = read(OUT/'definition_check.json')
    assert definition['passed'] and definition['cohorts'] == 36 and definition['metrics_per_cohort'] == 51
    assert definition['common_code_sha256'] == p['common_code_sha256']
    assert definition['analysis_code_sha256'] == e['code_sha256']
    repair = read(OUT/'diagnostics/import_repair.json')
    assert repair['old_code_sha256'] == sha(PROJECT/repair['old_code_snapshot'])
    assert repair['new_code_sha256'] == sha(PROJECT/'simulation/run_mc_statistics.py')
    assert repair['failed_attempt_new_mothers'] == repair['failed_attempt_replay_entries'] == 0
    assert repair['transport_function_AST_unchanged']
    transport = read(OUT/'transport_execution.json'); replay = read(OUT/'replay_execution.json')
    for r in [transport, replay]:
        assert r['protocol_sha256'] == sha(OUT/'protocol.json')
        assert r['code_sha256'] == sha(PROJECT/'simulation/run_mc_statistics.py')
    assert transport['new_independent_mothers'] == e['new_independent_mothers'] == 256000
    assert replay['new_independent_mothers'] == 0
    assert len(transport['jobs']) == len(replay['jobs']) == 8
    assert len({x['seed'] for x in p['entries'].values()}) == 8
    native_identity = 0; gates = 0; entries = 0; native_versions = set()
    for key, entry in p['entries'].items():
        assert sha(PROJECT/entry['config']) == entry['config_sha256']
        tr = read(OUT/'logs'/f'{key}_transport.json'); hashes(tr['outputs_sha256'])
        validation = read(OUT/'native'/f'{key}_validation.json')
        assert validation['passed'] and validation['ancestry_checks'] and validation['events'] == entry['events']
        if entry['source'] == 'Am241':
            assert validation['prompt_gamma_lines_counts']['0.05954'] > 0
            assert validation['decay_product_counts']['1000020040'] > 0
        else:
            for line in ['1.17323', '1.33249']: assert validation['prompt_gamma_lines_counts'][line] > 0
        base = load(OUT/'arrays'/f'{key}_frozen.npz'); initial = load(OUT/'transport/arrays'/f'{key}.npz')
        np.testing.assert_array_equal(base['ids'], np.flatnonzero(initial['all_common']))
        np.testing.assert_array_equal(base['q'], initial['all_q'][base['ids']])
        assert len(np.unique(base['ids'])) == len(base['ids']) and int(base['n_parent']) == entry['events']
        check_array(base); gates += 1
        f = ROOT.TFile.Open(str(PROJECT/entry['path'])); tree = f.Get('Events')
        info = json.loads(str(f.Get('RunInfo').GetString().Data()))
        native_versions.add((info['geant4'], info['schema_version'], info['sensor_entry_policy']))
        assert info['completed'] and info['seed'] == entry['seed'] and info['config'] == read(PROJECT/entry['config'])
        assert tree.GetEntries() == entry['events']
        tree.SetBranchStatus('*', 0)
        for name in ['event_id', 'adc']: tree.SetBranchStatus(name, 1)
        for i, idx in enumerate(base['ids']):
            tree.GetEntry(int(idx)); assert tree.event_id == idx
            np.testing.assert_array_equal(np.asarray(tree.adc), base['waves'][i])
        f.Close(); native_identity += len(base['ids'])
        plan = read(OUT/'configs'/f'{key}_jobs.json'); assert plan['ids'] == base['ids'].tolist()
        rr = read(OUT/'logs'/f'{key}_replay_extraction.json')
        assert rr['code_sha256'] == sha(PROJECT/'simulation/run_mc_statistics.py')
        assert rr['parent_root_sha256'] == tr['outputs_sha256'][entry['path']]
        assert rr['plan_sha256'] == sha(OUT/'configs'/f'{key}_jobs.json')
        for job in plan['jobs']:
            rec = rr['jobs'][job['key']]
            assert rec['config_sha256'] == sha(job['config'])
            assert rec['root_sha256'] == sha(job['output'])
            ap = OUT/'arrays'/f"{job['key']}.npz"; assert rec['array_sha256'] == sha(ap)
            a = load(ap)
            for name in ['ids', 'q']: np.testing.assert_array_equal(a[name], base[name])
            check_array(a); gates += 1; entries += len(a['ids'])
            if job['require_identity']:
                assert rec['equal_parent_ADC'] == len(a['ids'])
                np.testing.assert_array_equal(a['waves'], base['waves'])
        print('Verified', key, 'full-parent provenance, native identity and', len(plan['jobs']), 'readout arrays', flush=True)
    assert entries == replay['replay_entries'] == e['replay_entries']
    assert len(native_versions) == 1
    report['native_transport_version_schema_entry_policy'] = list(next(iter(native_versions)))
    minimum = 512; partitions = 0; selected_draws = 0
    for source, run in SOURCES:
        for si, seg in enumerate(SEGMENTS):
            group = f'run{run}_{seg}'; data, _ = measured(run, seg); data = compact(data)
            for B in p['block_sizes']:
                r = e['groups'][group][str(B)]; data['fold'] = (data['ids']//B)%2
                assert r == read(OUT/'evaluation'/f'{group}_B{B}.json')
                assert r['bootstrap_sha256'] == sha(OUT/'evaluation'/f'{group}_B{B}_bootstrap.npz')
                boot = load(OUT/'evaluation'/f'{group}_B{B}_bootstrap.npz')
                ref = read(CORE/'references'/f'{group}_B{B}.json'); design = read(WAVE/'designs'/f'{group}_B{B}.json')
                zd, md = prepared(data, p, source, ref, design); ds = stats(data, p, source, zd, md)
                fresh = {}; old = {}; context = {}; points = {}
                for iv, v in enumerate(VARIANTS):
                    fresh[v] = pool([compact(assemble(batch, run, seg, B, v, data, p, source)) for batch in BATCHES])
                    old[v] = compact(load(RC/'arrays'/f'{group}_B{B}_{v}_evaluation.npz'))
                    old[v]['fold'] = load(CORE/'arrays'/f'{group}_B{B}_{v}_z.npz')['fold']
                    fz, fm = prepared(fresh[v], p, source, ref, design)
                    oz, om = prepared(old[v], p, source, ref, design)
                    fs = stats(fresh[v], p, source, fz, fm); os = stats(old[v], p, source, oz, om)
                    context[v] = (fz, fm, oz, om); points[v] = fs; row = r['variants'][v]
                    np.testing.assert_allclose(contrast(fs, ds), [row['point'][n] for n in METRICS], atol=1e-12)
                    np.testing.assert_allclose(contrast(os, ds), [row['old_point'][n] for n in METRICS], atol=1e-12)
                    direct = direct_mode_statistics(fresh[v], p, source, design, fresh[v]['fold'])
                    for name, value in direct.items(): np.testing.assert_allclose(value, fs['mode'][name], rtol=1e-10, atol=1e-14)
                    partitions += 1
                    for j, name in enumerate(METRICS):
                        nb = boot['fresh_joint'][:, iv, j]; ob = boot['old_joint'][:, iv, j]
                        for values, field in [(nb, 'CI95'), (ob, 'old_CI95_recomputed'), (nb-ob, 'fresh_minus_old_CI95'),
                                              (nb-boot['fresh_joint'][:, 0, j], 'paired_from_frozen_CI95'),
                                              (boot['fresh_MC_only'][:, iv, j], 'MC_only_CI95'),
                                              (boot['fresh_data_only'][:, iv, j], 'data_only_CI95')]:
                            check_interval(values, row[field][name])
                            minimum = min(minimum, row[field][name]['valid_draws'])
                rng = np.random.default_rng(731031900+run+100*si+B)
                for ib in range(512):
                    dc = block_counts(data['ids'], tuple(data['span']), B, rng)
                    counts = mother_counts(fresh['frozen'], rng, [EVENTS[source]]*4)
                    oldcounts = mother_counts(old['frozen'], rng, [int(old['frozen']['n_parent'])])
                    if ib not in [0, 127, 511]: continue
                    db = stats(data, p, source, zd, md, dc)
                    for iv, v in enumerate(VARIANTS):
                        fz, fm, oz, om = context[v]
                        fb = stats(fresh[v], p, source, fz, fm, counts)
                        ob = stats(old[v], p, source, oz, om, oldcounts)
                        pairs = [('fresh_joint', fb, db), ('old_joint', ob, db),
                                 ('fresh_MC_only', fb, ds), ('fresh_data_only', points[v], db)]
                        for name, a, b in pairs:
                            expected = contrast(a, b) if a is not None and b is not None else np.full(len(METRICS), np.nan)
                            np.testing.assert_allclose(boot[name][ib, iv], expected, rtol=1e-12, atol=1e-12)
                            selected_draws += 1
                if seg == 'last20' and B == 128:
                    rng = np.random.default_rng(731032500+run); n = int(fresh['frozen']['n_parent'])
                    for ib in range(512):
                        take = rng.choice(n, p['old_independent_mothers'][source], replace=False)
                        counts = np.isin(fresh['frozen']['ids'], take).astype(int)
                        assert counts.sum() == boot['old_size_selected_counts'][ib]
                        if ib not in [0, 127, 511]: continue
                        for iv, v in enumerate(VARIANTS):
                            fz, fm, _, _ = context[v]; ss = stats(fresh[v], p, source, fz, fm, counts)
                            expected = contrast(ss, ds) if ss is not None else np.full(len(METRICS), np.nan)
                            np.testing.assert_allclose(boot['old_size_subsamples'][ib, iv], expected, atol=1e-12)
                    for iv, v in enumerate(VARIANTS):
                        for j, name in enumerate(METRICS):
                            check_interval(boot['old_size_subsamples'][:, iv, j], r['old_size_diagnostic']['variants'][v][name]['conditional_range95'])
                print('Verified', group, B, 'point definitions, direct variance and paired uncertainty', flush=True)
    nbpath = PROJECT/'jupyter/codex_0001_csi_independent_mc_statistics.ipynb'
    nb = nbformat.read(nbpath, as_version=4); nbformat.validate(nb)
    code = [c for c in nb.cells if c.cell_type == 'code']
    assert all(c.execution_count for c in code) and not any(o.output_type == 'error' for c in code for o in c.outputs)
    blobs = [base64.b64decode(o.data['image/png']) for c in code for o in c.outputs if 'data' in o and 'image/png' in o.data]
    figs = read(OUT/'figure_record.json'); assert len(blobs) == len(figs['figures']) == 6
    for blob, name in zip(blobs, figs['figures']): assert blob == (OUT/'figures'/f'{name}.png').read_bytes()
    hashes(figs['inputs_sha256']); assert figs['code_sha256'] == sha(PROJECT/'simulation/plot_mc_statistics.py')
    decision = read(OUT/'decision.json'); hashes(decision['inputs_sha256']); hashes(decision['tables_sha256'])
    assert decision['code_sha256'] == sha(PROJECT/'simulation/summarize_mc_statistics.py')
    assert not decision['default_changed'] and not decision['official_PID_changed'] and minimum >= 128
    with (OUT/'comparison_table.csv').open(newline='') as f: comparison = list(csv.DictReader(f))
    with (OUT/'batch_table.csv').open(newline='') as f: batch_rows = list(csv.DictReader(f))
    assert len(comparison) == 36 and len(batch_rows) == 144
    for row in comparison:
        expected = e['groups'][row['group']][row['block']]['variants'][row['variant']]
        assert int(row['new_supported_mothers']) == expected['support']['supported_events']
        for name in METRICS:
            assert float(row[name+'_old']) == expected['old_point'][name]
            assert float(row[name+'_fresh']) == expected['point'][name]
            limits = expected['CI95'][name]['CI95']
            if limits is None: assert row[name+'_low'] == row[name+'_high'] == ''
            else: assert [float(row[name+'_low']), float(row[name+'_high'])] == limits
    for row in batch_rows:
        expected = e['groups'][row['group']][row['block']]['batches'][row['variant']][int(row['batch'])]
        assert row['estimable'] == str(expected['support']['estimable'])
        for name in METRICS:
            if expected['point'] is None: assert row[name] == ''
            else: assert float(row[name]) == expected['point'][name]
    report.update(new_independent_mothers=256000, distinct_selected_native_mothers=native_identity,
                  replay_entries=entries, independently_integrated_ADC_cohorts=gates,
                  direct_shape_covariance_partitions=partitions, regenerated_paired_metric_vectors=selected_draws,
                  minimum_valid_joint_bootstrap_draws=minimum, default_changed=False, official_PID_changed=False,
                  notebook={'cells': len(nb.cells), 'executed_cells': len(code), 'exact_embedded_pngs': len(blobs)})
    report['outputs_sha256'] = {relative(f): sha(f) for f in sorted(OUT.rglob('*')) if f.is_file() and 'source_snapshot' not in f.parts}
    report['outputs_sha256'][relative(nbpath)] = sha(nbpath)
    report['sources_sha256'] = {relative(f): sha(f) for f in sources()}
    save(OUT/'verification.json', report)
    print('All fresh-transport, native ADC, frozen-metric, interval and notebook checks passed.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--archive', action='store_true')
    args = parser.parse_args(); archive() if args.archive else main()
