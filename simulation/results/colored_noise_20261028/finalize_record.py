"""Archive this completed local study; does not rerun or change simulation data."""
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parent
PROJECT = BASE.parents[2]
SIM = PROJECT / 'simulation'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def relative(path):
    return str(path.relative_to(PROJECT))


build = read(BASE / 'build_record.json')
for path, expected in build['source_sha256'].items():
    assert sha(PROJECT / path) == expected, path
    assert sha(BASE / 'source_snapshot' / path) == expected, path
assert sha(SIM / 'build/csi_sources') == build['binary_sha256']
assert sha(BASE / 'design.json') == build['design_sha256']
metadata = read(BASE / 'run_metadata_check.json')
for source, name in [('Co60', 'co60'), ('Am241', 'am241')]:
    assert all(metadata['sources'][source]['checks'].values()), source
    assert sha(BASE / name / 'events.root') == metadata['sources'][source]['root_sha256']

readout = SIM / 'results/readout_noise_20261028'
constraints = SIM / 'results/signal_constraints_20261028'
executed_readout = readout / 'characterize_readout.py'
assert sha(executed_readout) == read(readout / 'readout.json')['code_sha256']['simulation/characterize_readout.py']

# Preserve the current analysis and its local dependencies, including tests.
sources = [PROJECT / 'README.md', SIM / 'README.md', SIM / 'RESULTS.md']
sources += sorted(SIM.glob('*.py'))
sources += sorted(p for p in (SIM / 'tests').iterdir() if p.is_file())
sources += [PROJECT / 'analysis/paper_baseline' / name for name in ['prepare.py', 'refine_waveforms.py']]
snapshot = BASE / 'analysis_snapshot'
for path in sources:
    target = snapshot / path.relative_to(PROJECT)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, target)
    assert sha(path) == sha(target), path

reports = [readout / 'readout.json', BASE / 'design.json',
           BASE / 'common_charge/comparison.json', BASE / 'fluctuations/diagnostics.json',
           BASE / 'joint_validation_final/validation.json',
           BASE / 'noise_comparison/comparison.json', constraints / 'constraints.json']
code_checks = {}
for report in reports:
    data = read(report)
    for field in ['code_sha256', 'source_code_sha256', 'analysis_code_sha256']:
        for path, expected in data.get(field, {}).items():
            actual = executed_readout if report == readout / 'readout.json' and path == 'simulation/characterize_readout.py' else PROJECT / path
            assert sha(actual) == expected, (str(report), path)
            code_checks[relative(report) + ':' + path] = relative(actual)

ctest = (BASE / 'ctest.log').read_text()
analysis_tests = (BASE / 'analysis_tests.log').read_text()
assert all(s in ctest for s in ['14 tests passed', '3 tests passed', 'Ran 10 tests', '100% tests passed'])
assert 'Ran 6 tests' in analysis_tests and '\nOK\n' in analysis_tests
validations = {source: read(BASE / name / 'events_validation.json') for source, name in [('Co60', 'co60'), ('Am241', 'am241')]}
assert all(item['passed'] for item in validations.values())

outputs = []
for folder in [BASE, readout, constraints]:
    for path in folder.rglob('*'):
        if not path.is_file():
            continue
        parts = path.relative_to(folder).parts
        if any(part in ['source_snapshot', 'analysis_snapshot', 'noise_comparison_initial_plot', '__pycache__'] for part in parts):
            continue
        if path.name in ['record.json', 'finalize.log', 'noise_comparison_initial_plot.log']:
            continue
        outputs.append(path)
outputs += [SIM / 'results' / name for name in ['readout_noise_20261028.log', 'colored_noise_20261028_prepare.log', 'signal_constraints_20261028.log']]

comparison = read(BASE / 'noise_comparison/comparison.json')
record = {
    'stage': 'Pretrigger noise characterization, native Gaussian FIR readout, full optical MC and conditional signal-response constraints',
    'archived_utc': datetime.now(timezone.utc).isoformat(),
    'result': 'Baseline covariance and spectrum improve; pulse fluctuations and charge spectra remain inconsistent. No single-PE gain calibration or full detector-response validation is claimed.',
    'verification': {
        'inherited_kernel_tests': 14, 'noise_kernel_tests': 3,
        'geant4_workflow_tests': 10, 'analysis_checks': 6,
        'frozen_transport_source_and_binary_unchanged': True,
        'root_config_source_seed_counts_and_schema_checked': True,
        'full_mc_events': {'Co60': 2000, 'Am241': 12000},
        'source_and_numerical_validations': validations,
        'scoped_git_diff_check': 'Passed for README.md and simulation; preexisting whitespace in two legacy set.h files left unchanged.',
    },
    'baseline_comparison': {key: value['baseline'] for key, value in comparison['sources'].items()},
    'interpretation_limits': [
        'Only existing experimental data used. Run 3 and 12 first 60% train the noise model; later groups and run 18 were already explored and are not blind validation.',
        'The baseline mean detrended variance and PSD enter calibration; their agreement is not independent validation.',
        'A stationary Gaussian FIR does not model transient weak pulses, non-Gaussian tails or run 12 temporal noise changes.',
        'Positive and negative weak-pulse candidates are not identified single photoelectrons or a dark-count-rate measurement.',
        'Signal yield, avalanche gain, geometry and time constants were inherited from the joint candidate without refitting in this stage.',
        'Assumed 16-channel array summation, PDE/gain/load, source gaps and optional Co foil remain conditional apparatus hypotheses.',
        'Am-241 includes alpha, gamma and electron contributions. Source labels and acquisition bits are not experimental particle truth.',
        'Frozen-Q pulse statistics use the existing amplitude and quality selections; baseline and mean-shape cohorts differ as documented.',
        'Signal-response frequency integration is a stationary Poisson diagnostic, not a substitute for Geant4 or a measurement of circuit RC.',
        'New and previous MC have different random seeds; small differences cannot be attributed solely to the noise model.',
    ],
    'analysis_notes': [
        'readout.json records the exact executed characterize_readout.py preserved beside it. Current source additionally saves primary-mask and primary-PSD arrays; analysis formulas are unchanged.',
        'A PyROOT noncontiguous plotting-array issue was fixed before the final comparison. The initial plot directory and log are intermediate artifacts, excluded from the authoritative output manifest.',
        'Final baseline/signal comparison, temporal-stability and conditional RC figures were visually inspected.',
        'No Git commit or push performed in this stage.',
    ],
    'build_record': relative(BASE / 'build_record.json'),
    'run_metadata_check': relative(BASE / 'run_metadata_check.json'),
    'exact_executed_characterization_source': relative(executed_readout),
    'recorded_analysis_code_verified': code_checks,
    'analysis_snapshot_sha256': {relative(path): sha(path) for path in sources},
    'output_sha256': {relative(path): sha(path) for path in sorted(set(outputs))},
}
(BASE / 'record.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'archived_sources': len(sources), 'archived_outputs': len(record['output_sha256']),
                  'record': relative(BASE / 'record.json'), 'verification': 'passed'}, indent=2))
