"""Verify and freeze the signal-residual calibration, preserving earlier archives."""
import json
import platform
import shutil
from pathlib import Path
import numpy as np
import ROOT
from calibrate_signal_fluctuations import OUT, save
from diagnose_charge import sha
from reference import PROJECT
from screen_response_candidates import info_of
from validate_signal_fluctuations import arrays, error


def read(path):
    return json.loads(path.read_text())


def check_hashes(mapping, base=PROJECT):
    for name, digest in mapping.items():
        assert sha(base / name) == digest, name
    return len(mapping)


def verify_variant(folder, new_transport):
    frozen = read(folder / 'native_run.json')
    result = read(folder / 'validation.json')
    assert frozen['new_transport'] == result['independent_new_transport'] == new_transport
    assert frozen['fit_sha256'] == sha(folder / 'fit.json') == result['fit_sha256']
    assert result['code_sha256'] == sha(PROJECT / 'simulation/validate_signal_fluctuations.py')
    assert result['native_run_sha256'] == sha(folder / 'native_run.json')
    binary = 'csi_sources' if new_transport else 'replay_response'
    assert frozen['binary_sha256'] == sha(PROJECT / 'simulation/build' / binary)
    checked = {}
    for source, name, run in [('Co60', 'co60', 3), ('Am241', 'am241', 12)]:
        r = result['sources'][source]
        root = folder / name / 'events.root'
        info = info_of(root)
        config = folder / f'{source}_calibrated.json'
        assert info['completed'] and info['schema_version'] == 9
        assert info['config'] == read(config)
        assert frozen['config_sha256'][source] == sha(config) == r['config_sha256']
        assert r['root_sha256'] == sha(root)
        assert r['charge_balance']['passed'] and r['charge_balance']['events'] == r['nparent']
        if new_transport:
            assert 'response_replay' not in info
            assert info['events_requested'] == frozen['sources'][source]['events'] == r['nparent']
            assert info['seed'] == frozen['sources'][source]['seed'] == r['seed']
            assert info['seed'] != info_of(OUT / name / 'events.root')['seed']
            assert sha(config) == sha(OUT / 'variance_scatter' / config.name)
            assert sha(config) == sha(PROJECT / f'simulation/configs/csi_tl_run{run}_response_calibrated.json')
        else:
            assert info['response_replay']['equal_original_avalanche_events'] == r['nparent']
            assert sha(PROJECT / frozen['parents'][source]['path']) == frozen['parents'][source]['sha256']
        model = arrays(folder / f'{source}_native.npz')
        np.testing.assert_allclose(model['metrics'][:, :, 3].mean(0), r['F'])
        assert len(model['ids']) == r['common_n']
        for group, c in r['comparisons'].items():
            data = arrays(OUT / f'{group}.npz')
            assert len(data['ids']) == c['n']
            np.testing.assert_allclose(data['metrics'][:, :, 3].mean(0), c['measured_F'])
            np.testing.assert_allclose(np.array(r['F']) / c['measured_F'] - 1, c['relative'])
            assert error(model['shapes'].mean(0), data['shapes'].mean(0)) == c['shape_nrmse']
            draws = np.load(folder / f'{source}_{group}_relative_draws.npy')
            assert draws.shape == (512, 3) and np.isfinite(draws).all()
            np.testing.assert_array_equal(np.quantile(draws, [.025, .975], axis=0), c['relative_interval95'])
        checked[source] = {'events': r['nparent'], 'common_Q_events': r['common_n'],
                           'seed': r['seed'], 'config_sha256': sha(config), 'root_sha256': sha(root)}
    fig = read(folder / 'figure_record.json')
    assert fig['code_sha256'] == result['code_sha256']
    assert fig['validation_sha256'] == sha(folder / 'validation.json')
    for stem in fig['stems']:
        assert (folder / (stem + '.png')).read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
        assert (folder / (stem + '.pdf')).read_bytes().startswith(b'%PDF')
        assert '<svg' in (folder / (stem + '.svg')).read_text()
    for e in fig['examples']:
        data = arrays(OUT / (e['measured_group'] + '.npz'))
        model = arrays(folder / (e['source'] + '_native.npz'))
        i = np.argsort(data['q'])[int(e['Q_rank'] * (len(data['q']) - 1))]
        j = int(np.argmin(abs(model['q'] - data['q'][i])))
        assert int(data['ids'][i]) == e['measured_id'] and int(model['ids'][j]) == e['MC_parent_id']
        assert float(model['q'][j] / data['q'][i] - 1) == e['relative_Q_difference']
    return checked


def main():
    design = read(OUT / 'design.json')
    checks = {'input_hashes': check_hashes(design['inputs_sha256'])}
    checks['parent_archives'] = {}
    for stage, expected in [('publication_20261028', design['parent_publication_record_sha256']),
                            ('robustness_20261028', design['parent_robustness_record_sha256'])]:
        path = PROJECT / 'analysis/paper_baseline/results' / stage / 'record.json'
        assert sha(path) == expected
        parent = read(path)
        checks['parent_archives'][stage] = {
            'record_sha256': expected,
            'unchanged_outputs': check_hashes(parent['outputs_sha256']),
            'unchanged_source_snapshots': check_hashes(parent['source_sha256'], path.parent / 'source_snapshot'),
        }
    fit = read(OUT / 'fit.json')
    assert fit['code_sha256'] == sha(PROJECT / 'simulation/fit_signal_fluctuations.py')
    assert fit['prepare_design_sha256'] == sha(OUT / 'design.json')
    scatter = read(OUT / 'variance_scatter/fit.json')
    assert scatter['code_sha256'] == sha(PROJECT / 'simulation/fit_event_variance.py')
    assert scatter['parent_fit_sha256'] == sha(OUT / 'fit.json')
    assert scatter['parent_validation_sha256'] == sha(OUT / 'validation.json')
    # The constant-variance stage used its own preserved source version.
    old = read(OUT / 'validation.json')
    assert old['code_sha256'] == sha(OUT / 'source_snapshot_constant_variance/simulation/validate_signal_fluctuations.py')
    for source, name in [('Co60', 'co60'), ('Am241', 'am241')]:
        r = old['sources'][source]
        info = info_of(OUT / name / 'events.root')
        assert info['completed'] and r['root_sha256'] == sha(OUT / name / 'events.root')
        assert info['response_replay']['equal_original_avalanche_events'] == r['nparent']
    checks['replayed_calibration'] = verify_variant(OUT / 'variance_scatter', False)
    checks['new_full_transport'] = verify_variant(OUT / 'independent', True)
    fresh = read(OUT / 'independent/native_run.json')
    assert fresh['calibration_record_sha256'] == sha(OUT / 'variance_scatter/native_run.json')
    logs = OUT / 'logs'
    logs.mkdir(exist_ok=True)
    for source in sorted(Path('/tmp').glob('sipm_signal*.log')):
        if 'archive' not in source.name:
            shutil.copy2(source, logs / source.name)
    assert '100% tests passed, 0 tests failed out of 5' in (logs / 'sipm_signal_final_ctest.log').read_text()
    assert 'Ran 2 tests' in (logs / 'sipm_signal_final_replay_tests.log').read_text()
    assert (logs / 'sipm_signal_final_replay_tests.log').read_text().rstrip().endswith('OK')
    checks['tests'] = {'ctest_targets': 5, 'native_replay_integration_tests': 2, 'all_passed': True}
    checks['figures'] = 'PNG/PDF/SVG headers verified; three independent-transport PNG figures visually reviewed. PDFs not separately rendered.'
    checks['interruption'] = read(OUT / 'independent/co60/interrupted_attempt_1/status.json')
    sources = list((PROJECT / 'simulation').glob('*.py'))
    sources += list((PROJECT / 'simulation').glob('*.cc'))
    for sub in ('include', 'src', 'tests', 'vendor'):
        sources += [x for x in (PROJECT / 'simulation' / sub).rglob('*')
                    if x.is_file() and x.suffix in ('.py', '.cc', '.hh', '.h', '.json')]
    for path in ('simulation/CMakeLists.txt', 'simulation/README.md', 'simulation/RESULTS.md', 'README.md',
                 'analysis/paper_baseline/MANUSCRIPT_DRAFT.md', 'analysis/paper_baseline/README.md',
                 'analysis/paper_baseline/prepare.py', 'analysis/paper_baseline/publication_stats.py',
                 'analysis/paper_baseline/publication_mc_uncertainty.py'):
        sources.append(PROJECT / path)
    sources += list((PROJECT / 'simulation/configs').glob('csi_tl_run*_response_calibrated.json'))
    snapshots = OUT / 'source_snapshot'
    source_hashes = {}
    for source in sorted(set(sources)):
        rel = source.relative_to(PROJECT)
        target = snapshots / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        source_hashes[str(rel)] = sha(source)
        assert sha(target) == source_hashes[str(rel)]
    outputs = {str(path.relative_to(PROJECT)): sha(path) for path in sorted(OUT.rglob('*'))
               if path.is_file() and path != OUT / 'record.json' and 'source_snapshot' not in path.parts}
    save(OUT / 'record.json', {
        'date': '2026-10-28', 'scope': __doc__, 'checks': checks,
        'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'ROOT': ROOT.gROOT.GetVersion()},
        'training': 'Frozen run3/12 first60 only, 2450+1547 measured waveforms. Later segments/run18 were previously explored, not blind.',
        'protocol_note': 'Initial design wording said fixed average response. Actual fit explicitly scans a shared RC grid and fits per-acquisition residual spectra; second-stage event variance scatter was added adaptively using training F50 variance. Both stages were frozen before new full transport.',
        'MC_accounting': 'Two calibration replay stages reuse the original 2000+12000 histories. Independent validation uses 4000+24000 new complete histories. Interrupted Co output is excluded; retry repeats the same frozen seed/config/count.',
        'raw_BIN_hashes': 'Inherited from parent provenance; not recomputed this phase. Frozen feature and parent ROOT inputs rehashed.',
        'limits': ['Empirical acquisition-specific residual, not a unique physical mechanism or isotope-dependent SiPM gain.',
                   'Intervals condition on selected calibration; parameter fitting and unknown apparatus are not propagated.',
                   'Charge distributions, Co timing/shape-score distributions, Am late baseline and run18 transfer remain mismatched.',
                   'Am source contains alpha and gamma; no measured eventwise truth or alpha/gamma classification efficiency inferred.'],
        'source_sha256': source_hashes, 'outputs_sha256': outputs,
        'repository_actions': 'No commit or push in this phase.',
    })
    print(json.dumps({'source_files': len(source_hashes), 'output_files': len(outputs), 'checks': checks}, indent=2))


if __name__ == '__main__':
    main()
