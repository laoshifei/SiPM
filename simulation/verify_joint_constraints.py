"""Verify native joint calibration, independent validation and frozen provenance."""
import argparse
import base64
import copy
import functools
import json
import platform
import shutil
from pathlib import Path
import numpy as np
import scipy
import ROOT
import nbformat
from prepare_joint_constraints import OUT, TAIL, NPE, PROJECT, BUILD, DATA, WIDTH, SOURCES, FAMILIES, relative
from prepare_pid import read
from extract_pid import load, label_truth
from verify_pid import independent_scores
from verify_co_tail import independent_moments, equal_bins
from analyze_bandwidth_validation import widths, rms_width, interval
from publication_stats import sha, save


@functools.lru_cache(None)
def digest(path): return sha(Path(path))


def hashes(mapping, base=PROJECT):
    for path, value in mapping.items(): assert digest(base/path) == value, path
    return len(mapping)


def source_files():
    sources = set((PROJECT/'simulation').glob('*.py'))
    for folder in ['include', 'src', 'vendor', 'tests']:
        sources.update(p for p in (PROJECT/'simulation'/folder).rglob('*') if p.suffix in ['.py', '.cc', '.hh', '.h', '.json'])
    sources.update([PROJECT/'simulation/main.cc', PROJECT/'simulation/replay_response.cc', PROJECT/'simulation/CMakeLists.txt',
                    PROJECT/'simulation/README.md', PROJECT/'jupyter/codex_0001_build_joint_constraints_notebook.py'])
    sources.update((PROJECT/'analysis/paper_baseline').glob('*.py'))
    return sources


def freeze_archive():
    checks = read(OUT/'verification.json')
    # Rehash verified artifacts before freezing, without repeating ADC/statistical
    # tests that already passed. Only the explanatory README may be updated.
    hashes(checks['verified_outputs_sha256']); hashes(checks['verified_analysis_sources_sha256'])
    for rel, row in checks['parents'].items(): assert digest(PROJECT/rel/'record.json') == row['record_sha256']
    source_hashes = {}
    for p in sorted(source_files()):
        target = OUT/'source_snapshot'/p.relative_to(PROJECT); target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target); source_hashes[relative(p)] = sha(target)
    outputs = {relative(p): digest(p) for p in sorted(OUT.rglob('*')) if p.is_file() and 'source_snapshot' not in p.parts}
    nbpath = PROJECT/'jupyter/codex_0001_csi_joint_constraints.ipynb'; outputs[relative(nbpath)] = digest(nbpath)
    save(OUT/'record.json', {'scope': __doc__, 'outputs_sha256': outputs, 'source_sha256': source_hashes,
        'limits': 'Conditional apparatus and family-specific effective noise fit; no uniqueness, equivalence, microscopic noise mechanism or experimental PID efficiency claim.'})
    print('Archived', len(outputs), 'outputs and', len(source_hashes), 'source files.', flush=True)


def main(archive=False):
    if (OUT/'record.json').exists():
        assert not archive, 'Already archived'
        record = read(OUT/'record.json')
        print('Frozen outputs:', hashes(record['outputs_sha256']), flush=True)
        print('Source snapshots:', hashes(record['source_sha256'], OUT/'source_snapshot'), flush=True); return
    if archive and (OUT/'verification.json').exists():
        freeze_archive(); return
    design = read(OUT/'design.json'); selection = read(OUT/'selection.json'); assessment = read(OUT/'assessment.json')
    checks = {'scope': __doc__, 'parents': {}}
    for rel, value in design['parent_record_sha256'].items():
        folder = PROJECT/rel; assert digest(folder/'record.json') == value
        old = read(folder/'record.json')
        checks['parents'][rel] = {'record_sha256': value, 'outputs': hashes(old['outputs_sha256']),
            'sources': hashes(old['source_sha256'], folder/'source_snapshot')}
        print('Preserved archive:', folder.name, flush=True)
    checks['inputs'] = {'design': hashes(design['inputs_sha256']), 'selection': hashes(selection['inputs_sha256']),
                        'assessment': hashes(assessment['inputs_sha256'])}
    assert design['code_sha256'] == digest(PROJECT/'simulation/prepare_joint_constraints.py')
    assert selection['code_sha256'] == digest(PROJECT/'simulation/select_joint_constraints.py')
    assert assessment['code_sha256'] == digest(PROJECT/'simulation/analyze_joint_constraints.py')
    assert selection['design_sha256'] == digest(OUT/'design.json')
    assert selection['calibration_protocol_sha256'] == digest(OUT/'calibration/protocol.json')
    assert assessment['selection_sha256'] == digest(OUT/'selection.json')
    assert assessment['protocol_sha256'] == digest(OUT/'validation/protocol.json')
    assert selection['selected_family'] == min(selection['ranking'], key=lambda k: selection['ranking'][k]['objective'])
    stability = read(OUT/'ranking_stability.json'); rb = load(OUT/'ranking_bootstrap.npz')
    assert stability['code_sha256'] == digest(PROJECT/'simulation/bootstrap_joint_selection.py')
    assert stability['selection_sha256'] == digest(OUT/'selection.json') and not stability['saved_selection_changed']
    checks['ranking_inputs'] = hashes(stability['inputs_sha256'])
    np.testing.assert_array_equal(rb['total_objectives'], rb['source_objectives'].mean(axis=2))
    valid = np.isfinite(rb['total_objectives']).all(axis=1); np.testing.assert_array_equal(valid, rb['valid'])
    assert valid.sum() == stability['valid_resamples'] >= 480
    total = rb['total_objectives'][valid]; winners = np.argmin(total, axis=1)
    chosen_index = stability['families'].index(selection['selected_family'])
    for i, family in enumerate(stability['families']):
        row = stability['results'][family]
        assert row['rank1_frequency'] == float(np.mean(winners == i))
        np.testing.assert_array_equal(row['bootstrap_objective_p025_p50_p975'], np.quantile(total[:, i], [.025, .5, .975]))
        np.testing.assert_array_equal(row['objective_minus_selected_CI95'], np.quantile(total[:, i]-total[:, chosen_index], [.025, .975]))
    checks['ranking_valid_draws'] = int(valid.sum())
    audit = read(OUT/'diagnostics/anchor_boundary_audit.json')
    assert audit['fit_code_sha256'] == digest(PROJECT/'simulation/fit_joint_constraints.py')
    corrected = {}
    for key, row in audit['arrays'].items():
        path = OUT/'profile/arrays'/f'{key}.npz'; assert row['array_sha256'] == digest(path)
        a = load(path); assert len(a['ids']) == row['events_checked']
        use = np.isin(a['ids'], row['affected_original_event_ids'])
        assert int(use.sum()) == len(row['affected_original_event_ids'])
        if use.any():
            from reference import wave_features
            f = wave_features(a['waves'][use]); np.testing.assert_array_equal(f['t10_sample'], row['actual_t10_samples'])
            np.testing.assert_array_equal(a['metrics'][use], row['original_extracted_F'])
            corrected[key] = row['affected_original_event_ids']
    assert corrected == {'a8439_run3_Co60': [3302]}
    unchanged_fits = 0
    for oldcfg in (OUT/'diagnostics/anchor_proxy_before_fix').glob('*_config.json'):
        assert read(oldcfg) == read(OUT/'noise_fit'/oldcfg.name)
        unchanged_fits += 1
    assert unchanged_fits == 5
    checks['anchor_boundary_regression'] = {'checked_events': sum(x['events_checked'] for x in audit['arrays'].values()),
        'corrected_exact_integer_t10_events': corrected, 'unchanged_unaffected_noise_configs': unchanged_fits}
    for family, row in selection['ranking'].items():
        np.testing.assert_allclose(row['objective'], np.mean([r['objective'] for r in row['sources'].values()]), rtol=1e-14)
        fit = read(OUT/'fit'/f'{family}.json')
        assert fit['best'] == min([r for r in fit['attempts'] if r['success']], key=lambda r: r['objective'])
        assert len(fit['attempts']) == 3
        assert fit['best']['parameters']['alpha_tau_ns'][1] == FAMILIES[family]
        assert max(abs(v) for v in fit['best']['mean_time_difference_ns'].values()) < 1e-6
    pub = PROJECT/'analysis/paper_baseline/results/publication_20261028'
    assert digest(pub/'record.json') == '69d1858dafa473c58d795cbfbb40deb1aca5ad6a9a20e1d5920eb67de0bf18e1'
    assert digest(pub/'point.json') == read(pub/'record.json')['outputs_sha256'][relative(pub/'point.json')]
    pca = read(pub/'point.json')['windows']['3000']['pca']
    oldsource = read(NPE/'protocol.json')['source_sha256']
    native_source = {k: v for k, v in oldsource.items() if Path(k).suffix in ['.cc', '.hh', '.h']}
    checks['unchanged_native_sources'] = hashes(native_source)
    checks['native'] = {}; independent = 0; replays = 0; seeds = set(); validation_arrays = {}
    for stage in ['profile', 'calibration', 'validation']:
        dest = OUT/stage; protocol = read(dest/'protocol.json'); execution = read(dest/'execution.json')
        assert execution['protocol_sha256'] == digest(dest/'protocol.json')
        assert execution['code_sha256'] == digest(PROJECT/'simulation/run_joint_constraints.py')
        assert len(execution['jobs']) == len(protocol['entries'])
        assert protocol['design_sha256'] == digest(OUT/'design.json')
        for name, value in protocol['binary_sha256'].items():
            assert digest(BUILD/name) == value == read(NPE/'validation/protocol.json')['binary_sha256'][name]
        for key, entry in protocol['entries'].items():
            cp = PROJECT/entry['config']; rp = PROJECT/entry['path']; cfg = read(cp)
            assert digest(cp) == entry['config_sha256']
            assert cfg['sipm']['sensor']['constant_pde'] == .3 and cfg['readout_response']['signal_transmission'] == .5
            family = entry['family']; param = read(OUT/'fit'/f'{family}.json')['best']['parameters']
            for field in ['electron_tau_ns', 'alpha_tau_ns', 'electron_fast_fraction', 'alpha_fast_fraction']:
                assert cfg['optical'][field] == param[field]
            assert cfg['sipm']['readout']['lowpass_ns'] == param['lowpass_ns']
            oldcfg = read(TAIL/'configs'/f"run{entry['run']}_{entry['source']}.json")
            factors = read(OUT/'fit'/f'{family}.json')['analytic_gate_area_multipliers']
            assert cfg['optical']['yield_photons_per_MeV'] == oldcfg['optical']['yield_photons_per_MeV']*factors['electron']
            assert cfg['optical']['alpha_light_yield_ratio'] == oldcfg['optical']['alpha_light_yield_ratio']*(factors['alpha']/factors['electron'])
            normalized = copy.deepcopy(cfg); normalized.pop('joint_constraints'); normalized['status'] = oldcfg['status']
            for field in ['electron_tau_ns', 'alpha_tau_ns', 'electron_fast_fraction', 'alpha_fast_fraction',
                          'yield_photons_per_MeV', 'alpha_light_yield_ratio']:
                normalized['optical'][field] = oldcfg['optical'][field]
            for field in ['coefficients', 'variance_per_signal_adc', 'provenance', 'spectral_components']:
                if field in oldcfg['signal_dependent_noise']:
                    normalized['signal_dependent_noise'][field] = oldcfg['signal_dependent_noise'][field]
                else: normalized['signal_dependent_noise'].pop(field, None)
            normalized['sipm']['readout']['lowpass_ns'] = oldcfg['sipm']['readout']['lowpass_ns']
            assert normalized == oldcfg, 'Unplanned apparatus/readout change'
            noise = cfg['signal_dependent_noise']
            assert noise['event_variance_cv'] == oldcfg['signal_dependent_noise']['event_variance_cv']
            assert noise['variance_per_signal_adc'] >= 0.
            np.testing.assert_allclose(np.dot(noise['coefficients'], noise['coefficients']), 1., atol=1e-8)
            if stage == 'profile': assert noise['variance_per_signal_adc'] == 0.
            else:
                fitkey = key if stage == 'calibration' else f'{family}_{key}'
                nf = read(OUT/'noise_fit'/f'{fitkey}.json')
                assert nf['code_sha256'] == digest(PROJECT/'simulation/fit_joint_constraints.py')
                assert nf['protocol_sha256'] == digest(OUT/'profile/protocol.json')
                hashes(nf['inputs_sha256']); assert digest(PROJECT/nf['config']) == nf['config_sha256']
                assert noise == read(PROJECT/nf['config'])['signal_dependent_noise']
            checked = read(rp.with_name(rp.stem+'_validation.json'))
            assert checked['passed'] and checked['ancestry_checks'] and checked['events'] == entry['events']
            assert checked['max_relative_charge_balance_error'] < 1e-9
            meta = read(dest/'extraction'/f'{key}.json')
            assert meta['root_sha256'] == digest(rp)
            assert meta['code_sha256'] == digest(PROJECT/'simulation/extract_npe_profile.py')
            f = ROOT.TFile.Open(str(rp)); info = json.loads(str(f.Get('RunInfo').GetString().Data()))
            tree = f.Get('Events'); assert tree.GetEntries() == entry['events']
            assert info['completed'] and info['config'] == cfg and info['seed'] == entry['seed']
            parentfile = None
            if entry['native']:
                assert 'response_replay' not in info and entry['seed'] not in seeds
                seeds.add(entry['seed']); independent += entry['events']
            else:
                replays += entry['events']; replay = info['response_replay']
                assert replay['equal_original_avalanche_events'] == entry['events']
                assert replay['max_relative_charge_balance'] < 1e-9
                assert digest(PROJECT/entry['parent']) == entry['parent_sha256']
                parentfile = ROOT.TFile.Open(str(PROJECT/entry['parent'])); pt = parentfile.Get('Events')
                pt.SetBranchStatus('*', 0); pt.SetBranchStatus('signal_mV', 1)
            a = load(dest/'arrays'/f'{key}.npz')
            assert int(a['n_parent']) == entry['events'] and len(a['ids']) == meta['n_common']
            np.testing.assert_array_equal(a['ids'], np.flatnonzero(a['all_common']))
            np.testing.assert_array_equal(a['energy'], a['all_energy'][a['ids']])
            np.testing.assert_array_equal(a['truth'], label_truth(a['energy'])[0])
            np.testing.assert_allclose(independent_scores(a['waves'], pca), a['scores'], rtol=1e-10, atol=1e-10)
            tree.SetBranchStatus('*', 0); tree.SetBranchStatus('adc', 1); tree.SetBranchStatus('signal_mV', 1)
            for i, eventid in enumerate(a['ids']):
                tree.GetEntry(int(eventid)); np.testing.assert_array_equal(np.asarray(tree.adc), a['waves'][i])
                if parentfile:
                    pt.GetEntry(int(eventid)); np.testing.assert_array_equal(np.asarray(tree.signal_mV), np.asarray(pt.signal_mV))
            if parentfile: parentfile.Close()
            f.Close()
            checks['native'][f'{stage}/{key}'] = {'mothers': entry['events'], 'common_events': len(a['ids']),
                'native': entry['native'], 'all_mother_physics_charge_checks': True, 'independent_ADC_scores': len(a['ids']),
                'replay_common_signal_bit_identical': not entry['native']}
            if stage == 'validation': validation_arrays[entry['run']] = a
            print('Verified', stage, key, flush=True)
    expected_seeds = set(design['native']['calibration_Co_seeds']+design['native']['calibration_Am_seeds']+
                         list(design['native']['validation_seeds'].values()))
    assert independent == 128000 and replays == 96000 and seeds == expected_seeds
    checks['independent_mothers'] = independent; checks['replayed_mothers_not_recounted'] = replays
    boot = load(OUT/'bootstrap.npz'); edges = np.asarray(design['q_edges']); minimum = 512; aggregates = 0
    for source, run, kind in SOURCES:
        support = design['support_bins'][source]; key = f'run{run}_{source}'; a = validation_arrays[run]
        native_mu = equal_bins(independent_moments(a['waves']), a['q'], edges, support)
        for segment in ['first60', 'middle20', 'last20']:
            group = f'run{run}_{segment}'; data = load(DATA/f'{group}.npz'); d = load(WIDTH/'arrays'/f'measured_{group}.npz'); u = d['main_band']
            data_mu = equal_bins(independent_moments(data['waves'][u]), d['q'][u], edges, support)
            row = assessment['checks'][group]
            np.testing.assert_allclose(data_mu, row['data_tail_metrics'], atol=1e-9)
            np.testing.assert_allclose(native_mu-data_mu, row['models']['joint']['tail_difference'], atol=1e-9)
            for name in ['npe', 'tail', 'joint']:
                item = row['models'][name]
                np.testing.assert_allclose(sum(item['mean_time_slice_difference_ns']), item['tail_difference'][0], atol=1e-9)
                am = boot[f'{key}__{name}__moments']; aq = boot[f'{key}__{name}__quartiles']
                assert am.shape == (512, 6) and aq.shape == (512, 6, 3, 3)
                for size in [64, 128]:
                    bm = boot[f'{group}__block{size}__moments']; bq = boot[f'{group}__block{size}__quartiles']
                    vals = {'tail_difference_CI95': am[:, :3]-bm[:, :3], 'F_ratio_CI95': am[:, 3:]/bm[:, 3:],
                            'IQR_ratio_CI95': rms_width(widths(aq), support)/rms_width(widths(bq), support)}
                    for field, values in vals.items():
                        for j in range(3):
                            saved = item['uncertainty_by_data_block'][str(size)][field][j]
                            assert interval(values[:, j]) == saved; minimum = min(minimum, saved['valid_draws'])
                            assert saved['valid_draws'] >= 480
                aggregates += 1
    checks['moment_aggregates'] = aggregates; checks['minimum_valid_bootstrap_draws'] = minimum
    fig = read(OUT/'figure_record.json')
    assert fig['code_sha256'] == digest(PROJECT/'simulation/plot_joint_constraints.py')
    assert fig['assessment_sha256'] == digest(OUT/'assessment.json') and fig['selection_sha256'] == digest(OUT/'selection.json')
    nbpath = PROJECT/'jupyter/codex_0001_csi_joint_constraints.ipynb'; nb = nbformat.read(nbpath, as_version=4); nbformat.validate(nb)
    images = []; codes = [c for c in nb.cells if c.cell_type == 'code']
    for c in codes:
        assert c.execution_count is not None and all(o.output_type != 'error' for o in c.outputs)
        images.extend(base64.b64decode(o.data['image/png']) for o in c.outputs if 'data' in o and 'image/png' in o.data)
    assert len(images) == len(fig['figures']) == 5
    for image, name in zip(images, fig['figures']):
        assert image == (OUT/'figures'/f'{name}.png').read_bytes()
        assert (OUT/'figures'/f'{name}.pdf').read_bytes().startswith(b'%PDF-')
        assert '<svg' in (OUT/'figures'/f'{name}.svg').read_text()[:3000]
    checks['notebook'] = {'executed_code_cells': len(codes), 'embedded_images': 5, 'byte_identical_PNGs': 5}
    prior = read(NPE/'verification.json')['tests']; assert prior['passed']
    checks['native_tests'] = {'unchanged_binary_and_sources': True, 'parent_passed': prior,
        'new_complete_native_mother_checks': 128000, 'new_replay_mother_checks': 96000,
        'note': 'No native source changes; existing unit/integration evidence retained, new configurations fully exercised.'}
    checks['versions'] = {'Python': platform.python_version(), 'NumPy': np.__version__, 'SciPy': scipy.__version__, 'ROOT': ROOT.gROOT.GetVersion()}
    checks['verified_outputs_sha256'] = {relative(p): digest(p) for p in sorted(OUT.rglob('*'))
        if p.is_file() and 'source_snapshot' not in p.parts and p.name != 'verification.json'}
    checks['verified_outputs_sha256'][relative(nbpath)] = digest(nbpath)
    checks['verified_analysis_sources_sha256'] = {relative(p): digest(p) for p in sorted(source_files()) if p.name != 'README.md'}
    save(OUT/'verification.json', checks)
    if archive: freeze_archive()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--archive', action='store_true')
    main(p.parse_args().archive)
