"""Archive the checked 2026-10-28 candidate without rerunning or changing physics."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import numpy as np

OUT = Path(__file__).resolve().parent
SIM = OUT.parents[1]
PROJECT = SIM.parent


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''): digest.update(chunk)
    return digest.hexdigest()


def read(path): return json.loads(path.read_text())
def write(path, value): path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def main():
    transport = SIM/'results/transport_candidates_20261028'
    scan = read(transport/'summary.json')
    assert scan['completed'] and len(scan['scenarios']) == 9
    executable_sha = sha(SIM/'build/csi_sources')
    assert executable_sha == scan['executable_sha256']
    for name, digest in scan['source_sha256'].items():
        assert sha(SIM/name) == digest
        assert sha(transport/'source_snapshot'/name) == digest
    assert all(row['validation']['passed'] for row in scan['scenarios'])
    validation = read(OUT/'joint_validation_final/validation.json')
    common = read(OUT/'common_charge_final/comparison.json')
    for code in (validation['code_sha256'], common['analysis_code_sha256']):
        for name, digest in code.items(): assert sha(PROJECT/name) == digest
    runs = {}
    for name, source in [('co60', 'Co60'), ('am241', 'Am241')]:
        directory = OUT/name
        run = read(directory/'run.json')
        numeric = read(directory/'events_validation.json')
        assert numeric['passed'] and numeric['events'] == run['events'] and run['exit_code'] == 0
        assert run['executable_sha256'] == executable_sha
        assert run['source_sha256'] == scan['source_sha256']
        assert sha(directory/'config.json') == run['config_sha256']
        root_sha = sha(directory/'events.root')
        assert root_sha == validation['sources'][source]['input_sha256']['new']
        assert root_sha in common['mc'][source]['file_sha256'].values()
        run.update(status='completed and numerically validated',
                   validation_sha256=sha(directory/'events_validation.json'),
                   events_root_sha256=root_sha,
                   validation_scope='Numerical/source checks passed; experimental distribution agreement remains incomplete')
        write(directory/'run.json', run)
        runs[source] = {'directory': name, 'events': run['events'], 'seed': run['seed'],
                        'config_sha256': run['config_sha256'], 'root_sha256': root_sha,
                        'run_record_sha256': sha(directory/'run.json'), 'numerical_validation': numeric}
    numerical_equality = {}
    for before, after in [('common_charge', 'common_charge_final'), ('joint_validation', 'joint_validation_final')]:
        with np.load(OUT/before/'curves.npz') as old, np.load(OUT/after/'curves.npz') as new:
            assert set(old.files) == set(new.files)
            assert all(np.array_equal(old[key], new[key], equal_nan=True) for key in old.files)
            numerical_equality[after] = {'intermediate_directory': before, 'equal_array_count': len(old.files),
                                        'all_values_exactly_equal': True}
    geometry = {}
    for name in ('geometry_co', 'geometry_am'):
        inspection = read(OUT/name/'inspection.json')
        assert inspection['passed'] and inspection['executable_sha256'] == executable_sha
        for artifact, digest in inspection['artifacts_sha256'].items(): assert sha(OUT/name/artifact) == digest
        geometry[name] = inspection
    log = (transport/'ctest.log').read_text()
    assert '14 tests passed' in log and 'Ran 8 tests' in log and '\nOK\n' in log

    snapshot = OUT/'analysis_snapshot'
    paths = sorted(SIM.glob('*.py')) + sorted((SIM/'macros').glob('*.mac'))
    paths += [SIM/'tests/test_workflow.py', SIM/'tests/vendor_fixture.json', SIM/'vendor/provenance.json',
              PROJECT/'analysis/paper_baseline/prepare.py', PROJECT/'analysis/paper_baseline/refine_waveforms.py',
              SIM/'README.md', SIM/'RESULTS.md', PROJECT/'README.md', PROJECT/'EXPERIMENT_INFO.md', Path(__file__)]
    manifest = {}
    for path in paths:
        relative = path.relative_to(PROJECT)
        target = snapshot/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if path == Path(__file__): shutil.copyfile(path, target)
        elif target.exists(): assert sha(target) == sha(path), 'Do not overwrite a different archived file'
        else: shutil.copyfile(path, target)
        manifest[str(relative)] = sha(path)
    write(snapshot/'record.json', {
        'scope': 'Final analysis, drivers, tests, macros and documentation snapshot; paths relative to project root',
        'sha256': manifest,
        'compiled_source_snapshot': '../../transport_candidates_20261028/source_snapshot',
        'compiled_source_manifest': '../../transport_candidates_20261028/summary.json',
        'intermediate_analysis_note': 'Intermediate plots were rerendered after label/guard-only changes; final arrays are exactly equal. batch_regression.json records the earlier feature refactor check.'})

    dependencies = [SIM/'results/settings_audit_20261028/settings_audit.json',
                    SIM/'results/trigger_proxy_20261028/trigger_proxy.json',
                    transport/'summary.json', transport/'ctest.log',
                    SIM/'results/joint_screen_20261028/screen.json',
                    SIM/'results/joint_screen_20261028/grid.csv',
                    SIM/'results/joint_screen_20261028/geometry_profile.json',
                    SIM/'results/joint_screen_20261028/Co60_candidate.json',
                    SIM/'results/joint_screen_20261028/Am241_candidate.json',
                    SIM/'results/reference/references.json', SIM/'results/reference/references.npz',
                    SIM/'results/calibration/fit.json', SIM/'results/calibration/expectations.npz',
                    PROJECT/'analysis/paper_baseline/results/confirmed_cohort.json']
    for run in (3, 12, 18):
        dependencies += [PROJECT/f'analysis/paper_baseline/results/pha_run{run:04d}_features.npz',
                         PROJECT/f'analysis/paper_baseline/results/phase2/run{run:04d}_features.npz']
    figures = ['joint_validation_final/joint_comparison.png', 'common_charge_final/waveform_comparison.png',
               'geometry_co/geometry.png', 'geometry_am/geometry.png']
    summary = {
        'date': '2026-10-28', 'status': 'completed conditional-candidate evaluation',
        'scope': 'Geant4 schema 5 energy transport, training-only joint screening, independent full optical simulation and measured-waveform comparison',
        'experimental_response_validated': False,
        'finding': 'Absolute mean amplitudes improved. Co charge distribution remains too broad; Am distribution is too narrow and lacks the measured high-charge tail.',
        'selection': validation['selection'], 'normalization': validation['normalization'],
        'limits': [
            '500 mV input range is supported by current X4 settings but their complete historical BIN association is unconfirmed.',
            'Source foil, gap, array size/summing and effective light yield remain conditional hypotheses with parameter degeneracy.',
            '200 ADC is a shared analysis cut, not a hardware trigger or efficiency model.',
            'Am gamma and secondary electrons are retained; measured source labels are not per-event particle truth.',
            'Late/check groups were previously explored. Common-charge area-normalized errors do not establish complete response precision.',
            'Quantile loss and KS distances are descriptive; no likelihood, p-value or parameter confidence interval is claimed.'
        ],
        'transport': {'scenarios': 9, 'parent_decays': 90000, 'digitization_enabled': False,
                      'record': '../transport_candidates_20261028/summary.json'},
        'screen': {'record': '../joint_screen_20261028/screen.json', 'frozen_candidate': validation['parameters']},
        'runs': runs, 'executable_sha256': executable_sha,
        'compiled_source_sha256': scan['source_sha256'],
        'compiled_source_snapshot': '../transport_candidates_20261028/source_snapshot',
        'build_note': 'Clean rebuilt after C++ changes because shared-folder timestamps ran ahead of guest clock; all scans and full optical runs used the same verified binary.',
        'tests': {'inherited_kernel': 14, 'integration': 8, 'passed': True,
                  'log': '../transport_candidates_20261028/ctest.log', 'log_sha256': sha(transport/'ctest.log')},
        'analysis_snapshot': 'analysis_snapshot/record.json', 'analysis_snapshot_record_sha256': sha(snapshot/'record.json'),
        'final_analysis_arrays_equal_to_intermediate': numerical_equality,
        'feature_batch_regression': read(OUT/'batch_regression.json'),
        'geometry': geometry,
        'absolute_training': {source: {key: {k: v for k, v in values.items() if k != 'event_indices'}
                              for key, values in row.items() if key in ('measured', 'old', 'new', 'old_discrepancy', 'new_discrepancy')}
                              for source, row in validation['sources'].items()},
        'joint_diagnostic_loss': validation['training_joint_loss'],
        'common_charge_checks': {key: common['comparisons'][key] for key in ('run3_late', 'run12_late', 'run18_check')},
        'common_charge_mc_count': {source: row['groups']['common']['n'] for source, row in common['mc'].items()},
        'figures_visually_inspected': figures,
        'artifacts_sha256': {str(path.relative_to(OUT)): sha(path) for directory in ('joint_validation_final', 'common_charge_final', 'geometry_co', 'geometry_am')
                             for path in sorted((OUT/directory).iterdir()) if path.is_file()},
        'dependency_sha256': {str(path.relative_to(PROJECT)): sha(path) for path in dependencies},
        'next_priority': 'Explain charge-distribution widths/tails using existing apparatus/trigger records and charge-binned waveform diagnostics before further amplitude fitting.'
    }
    write(OUT/'summary.json', summary)
    missing = []
    for path in [PROJECT/'README.md', SIM/'README.md', SIM/'RESULTS.md', PROJECT/'EXPERIMENT_INFO.md']:
        for target in re.findall(r'\]\((<[^>]+>|[^\s)]+)\)', path.read_text()):
            target = target.strip('<>').split('#')[0]
            if not target or '://' in target: continue
            if not (path.parent/target).exists(): missing.append((str(path), target))
    assert not missing, missing
    print(json.dumps({'summary': str(OUT/'summary.json'), 'source_files_verified': len(scan['source_sha256']),
                      'analysis_files_archived': len(manifest), 'tests': summary['tests'],
                      'documentation_links_valid': True}, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
