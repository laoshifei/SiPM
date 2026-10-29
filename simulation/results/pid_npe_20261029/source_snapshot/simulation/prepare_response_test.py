"""Freeze a controlled light-timing / readout-filter factorization experiment.

No fit. Preserve the continuous linear mean kernel by moving one exponential
pole from the deterministic readout filter into stochastic photon emission.
Recovery, finite windows, discretization and event selection may break exact
equality of the full nonlinear Monte Carlo response and are checked separately.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def native_rise(decay_ns, pole_ns):
    decay = np.asarray(decay_ns, dtype=float)
    if not np.isfinite(decay).all() or not np.isfinite(pole_ns) or pole_ns <= 0 or np.any(decay <= pole_ns):
        raise ValueError('The moved positive pole must be shorter than each decay constant')
    return pole_ns*decay/(decay-pole_ns)


def finite_rise_pdf(time_ns, decay_ns, rise_ns):
    t = np.asarray(time_ns, dtype=float)
    if rise_ns == 0: return np.where(t >= 0, np.exp(-np.maximum(t, 0)/decay_ns)/decay_ns, 0.)
    positive = np.maximum(t, 0)
    return -np.expm1(-positive/rise_ns)*np.exp(-positive/decay_ns)*(rise_ns+decay_ns)/decay_ns**2


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, default=HERE/'results/joint_optical_20261028')
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args(); args.out.mkdir(parents=True, exist_ok=False)
    record = {'scope': __doc__, 'status': 'Frozen before full optical MC; no parameter optimization',
        'native_formula': 'p(t)=(rise+decay)/decay^2 * (1-exp(-t/rise))*exp(-t/decay), t>=0. Equivalent poles: decay and rise*decay/(rise+decay).',
        'conversion': 'rise_i=RC*decay_i/(decay_i-RC); readout.lowpass_ns=0. Requires the same alpha/electron decay constants for this shared-rise control.',
        'primary_source': 'https://github.com/Geant4/geant4/blob/v11.4.2/source/processes/electromagnetic/xrays/src/G4Scintillation.cc#L557-L570',
        'interpretation': 'An identifiability test, not measured material rise constants or recovered circuit parameters. Conditional geometry and common normalization inherited without refitting.',
        'sources': {}, 'source_sha256': {}}
    for name, source, events, seed in [('co60', 'Co60', 5000, 282203), ('am241', 'Am241', 30000, 282212)]:
        folder = args.out/name; folder.mkdir()
        old_path = args.base/name/'config.json'; config = json.loads(old_path.read_text())
        o = config['optical']; r = config['sipm']['readout']
        if o['electron_tau_ns'] != o['alpha_tau_ns']: raise ValueError('This control requires shared decay constants')
        if any(o.get('rise_time_ns', [0., 0.])): raise ValueError('Base must have zero rise')
        old_rc = r['lowpass_ns']; o['rise_time_ns'] = native_rise(o['electron_tau_ns'], old_rc).tolist()
        r['lowpass_ns'] = 0.
        config['evidence']['response_factorization'] = {'moved_pole_ns': old_rc,
            'policy': record['interpretation'], 'mean_kernel_scope': record['scope'],
            'inherited_evidence': 'Earlier evidence fields describe previous stages; numerical optical/readout values define this run.'}
        config_path = folder/'config.json'; config_path.write_text(json.dumps(config, indent=2)+'\n')
        output = folder/'events.root'; macro = folder/'run.mac'
        macro.write_text(f'/csi/config {config_path.resolve()}\n/csi/source {source}\n/csi/seed {seed}\n/csi/output {output.resolve()}\n/run/initialize\n/run/beamOn {events}\n')
        record['sources'][source] = {'events_requested': events, 'seed': seed, 'base_config_sha256': sha(old_path),
            'config_sha256': sha(config_path), 'moved_pole_ns': old_rc, 'native_rise_time_ns': o['rise_time_ns'],
            'changes_to_numeric_config': ['optical.rise_time_ns', 'sipm.readout.lowpass_ns']}
    paths = [HERE/'main.cc', HERE/'CMakeLists.txt']
    for directory in ('src', 'include', 'vendor'):
        paths.extend(p for p in (HERE/directory).iterdir() if p.suffix in ('.cc', '.hh'))
    paths.extend([Path(__file__), HERE/'validate_joint.py', HERE/'diagnose_fluctuations.py', HERE/'tests/test_workflow.py', HERE/'tests/test_charge_diagnostics.py'])
    for path in paths:
        relative = path.relative_to(PROJECT); target = args.out/'source_snapshot'/relative
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(path, target)
        record['source_sha256'][str(relative)] = sha(path)
    record['binary_sha256'] = sha(HERE/'build/csi_sources')
    (args.out/'design.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record['sources'], indent=2))


if __name__ == '__main__': main()
