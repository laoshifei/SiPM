"""Archive this completed analysis stage without changing any source data."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from datetime import datetime, timezone
import numpy as np

PROJECT = Path(__file__).resolve().parents[3]
SIM = PROJECT/'simulation'
OUT = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(4*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def rel(path):
    return str(path.relative_to(PROJECT))


def read(path):
    return json.loads(path.read_text())


def main():
    if (OUT/'record.json').exists():
        raise ValueError('Completed archive already exists')
    cov_dir = SIM/'results/residual_covariance_20261028'
    psd_dir = SIM/'results/frozen_psd_transfer_20261028'
    cov = read(cov_dir/'covariance.json')
    psd = read(psd_dir/'transfer.json')
    design = read(OUT/'design.json')
    confirmed = read(OUT/'confirmation_final/confirmation.json')
    screen = read(OUT/'screen.json')
    for record in (cov, design, confirmed['design']):
        for path, expected in record['code_sha256'].items():
            assert sha(PROJECT/path) == expected, f'Executed code changed: {path}'
    assert sha(SIM/'check_frozen_psd_transfer.py') == psd['code_sha256']
    assert sha(OUT/'screen.json') == confirmed['design']['parent_screen_sha256']
    with np.load(SIM/'results/bias_response_20261028/samples.npz') as bias, \
         np.load(cov_dir/'samples.npz') as local, np.load(psd_dir/'scores.npz') as scores:
        identities = {}
        for key, group in psd['groups'].items():
            ids = bias[key+'_ids']
            np.testing.assert_array_equal(ids, local[key+'_ids'])
            assert len(ids) == cov['groups'][key]['n'] == group['n']
            for method in group['methods']:
                assert len(scores[key+'_'+method]) == len(ids)
                assert np.isfinite(scores[key+'_'+method]).all()
            identities[key] = len(ids)
    assert sum(identities.values()) == 40795
    for variant in confirmed['results'].values():
        for row in variant['sources'].values():
            assert row['integrity']['passed']
    for row in confirmed['results']['native_best']['sources'].values():
        assert row['linear_screen_max_adc_difference'] == 0
        assert row['linear_screen_differing_samples'] == 0
    assert not screen['any_within_screening_tolerances']
    assert not any(r['within_screening_tolerances'] for r in confirmed['results'].values())
    integrity = read(OUT/'integrity.json')
    assert len(integrity) == 8 and all(r['passed'] for r in integrity.values())

    source_paths = set()
    for pattern in ('*.py', '*.cc', 'CMakeLists.txt', 'include/*.hh', 'src/*.cc',
                    'vendor/*', 'tests/*.py', 'tests/*.cc', 'tests/*.json', 'configs/*.json'):
        source_paths.update(p for p in SIM.glob(pattern) if p.is_file())
    source_paths.update((PROJECT/'analysis/paper_baseline').glob('*.py'))
    source_paths.add(Path(__file__).resolve())
    source_hashes = {}
    for path in sorted(source_paths):
        target = OUT/'source_snapshot'/path.relative_to(PROJECT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        source_hashes[rel(path)] = sha(path)
        assert sha(target) == source_hashes[rel(path)]
    logs = [SIM/'results'/name for name in (
        'response_build_20261028.log', 'replay_build_final_20261028.log',
        'response_ctest_20261028.log', 'response_replay_tests_20261028.log',
        'residual_covariance_tests_20261028.log', 'frozen_psd_tests_20261028.log',
        'residual_covariance_20261028.log', 'frozen_psd_transfer_final_20261028.log',
        'response_candidates_prepare_20261028.log', 'response_candidates_replay_20261028.log',
        'response_candidates_screen_20261028.log', 'response_confirmation_final_20261028.log',
        'response_confirmation_plot_20261028.log')]
    for path in logs:
        assert path.exists(), path
    artifacts = set(logs)
    for folder in (cov_dir, psd_dir, OUT):
        for pattern in ('*.json', '*.npz', '*.png', '*.svg'):
            artifacts.update(folder.glob(pattern))
    for variant in ('control', 'prompt_basis', 'ap15', 'ap30'):
        for source in ('co60', 'am241'):
            artifacts.update((OUT/variant/source).glob('*'))
    for path in (OUT/'confirmation_final').rglob('*'):
        if path.is_file():
            artifacts.add(path)
    artifact_hashes = {}
    for path in sorted(artifacts):
        if path.is_file():
            artifact_hashes[rel(path)] = {'sha256': sha(path), 'bytes': path.stat().st_size}
    docs = [PROJECT/name for name in ('README.md', 'EXPERIMENT_INFO.md',
        'simulation/README.md', 'simulation/RESULTS.md',
        'analysis/paper_baseline/MANUSCRIPT_DRAFT.md',
        'analysis/paper_baseline/LOCAL_EVIDENCE.md', 'analysis/paper_baseline/PHASE2_RESULTS.md')]
    record = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'Completed literature-guided local covariance, native response candidates, conditional voltage bounds and frozen PSD transfer. No new measured data, source labels are not particle truth.',
        'git_head_at_archive': subprocess.check_output(['git','rev-parse','HEAD'], cwd=PROJECT, text=True).strip(),
        'working_tree': 'Existing uncommitted work retained; this record does not imply a commit or push.',
        'independent_transport_histories': {'Co60':2000,'Am241':12000},
        'new_independent_transport_in_this_stage': 0,
        'completed_replay_files': 12,
        'replay_dependence': 'All candidates reuse the same source-specific parent photons. Eight base files plus four final confirmation files; do not sum their events as independent transport statistics.',
        'sample_identity_checks': identities,
        'raw_identity_policy': cov['raw_identity_policy'],
        'verification': {'ctest_targets_passed':4, 'inherited_sensor_checks':14,
            'noise_checks':3, 'parallel_response_checks':5, 'workflow_checks':10,
            'covariance_checks':5, 'frozen_psd_checks':3, 'replay_integration_test_cases':1,
            'all_eight_base_files_passed_full_validation':True,
            'all_four_confirmation_files_passed_full_validation':True,
            'native_best_all_adc_samples_equal_linear_screen':True},
        'candidate_outcome': 'None passes the shared two-source descriptive tolerances; no parameters promoted to calibrated status.',
        'additional_combination': 'Prompt 0.10 + AP 0.15 was exploratory after inspecting the initial screen; it also fails.',
        'unfinished_attempts_excluded': ['confirmation/ is an interrupted first confirmation attempt, superseded by confirmation_final/; its ROOT output is not an additional independent sample.'],
        'literature_scope': 'Shared library read-only; targeted local PDF verification and existing AI draft notes, not new full reviews or user endorsement.',
        'literature_manifest_sha256': sha(PROJECT/'.cache/paper-search/2026-10-28-next-steps/extraction_manifest.json'),
        'source_snapshot_root': rel(OUT/'source_snapshot'), 'source_sha256':source_hashes,
        'artifacts':artifact_hashes, 'documentation_sha256':{rel(p):sha(p) for p in docs}
    }
    (OUT/'record.json').write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
    print(f"Archived {len(source_hashes)} source files and {len(artifact_hashes)} artifacts; 40795 frozen waveform IDs verified.")


if __name__ == '__main__':
    main()
