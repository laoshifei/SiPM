"""Geant4 energy-deposition library for source-distance and exit-foil scenarios.

Scintillation and digitization are disabled. No waveform matching or actual
source encapsulation/distance is inferred by this scan alone.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np
import ROOT
from validate import check

HERE = Path(__file__).resolve().parent


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=HERE/'results/calibration/csi_tl_effective.json')
    parser.add_argument('--events', type=int, default=10000)
    parser.add_argument('--out', type=Path, required=True); args = parser.parse_args()
    if args.out.exists(): parser.error('Use a new output directory')
    build = subprocess.run(['cmake', '--build', str(HERE/'build'), '-j4'], capture_output=True, text=True)
    if build.returncode: raise RuntimeError(build.stdout+build.stderr)
    args.out.mkdir(parents=True); (args.out/'build.log').write_text(build.stdout+build.stderr)
    paths = [HERE/'main.cc', HERE/'CMakeLists.txt', HERE/'vendor/SiPMResponse.cc', HERE/'vendor/SiPMResponse.hh']
    paths += sorted((HERE/'include').glob('*.hh'))+sorted((HERE/'src').glob('*.cc'))
    manifest = {str(path.relative_to(HERE)): sha(path) for path in paths}
    snapshot = args.out/'source_snapshot'
    for path in paths:
        target = snapshot/path.relative_to(HERE); target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(path, target)
    executable = HERE/'build/csi_sources'; executable_sha = sha(executable)
    config = json.loads(args.config.read_text())
    config['optical']['yield_photons_per_MeV'] = 0.; config['acquisition']['digitize'] = False
    scenarios = [('Co60', 1., foil) for foil in (0., .25, .5)]
    scenarios += [('Am241', gap, 0.) for gap in (1., 10., 20., 25., 30., 35.)]
    report = {'scope': __doc__, 'events_per_scenario': args.events, 'executable_sha256': executable_sha,
              'source_sha256': manifest, 'scenarios': []}
    for i, (source, gap, thickness) in enumerate(scenarios):
        name = f'{source}_gap{gap:g}_Al{thickness:g}'; out = args.out/name; out.mkdir()
        local = copy.deepcopy(config); local['geometry']['source_gap_mm'] = gap
        local['geometry'].pop('source_exit_foil', None)
        if thickness:
            local['geometry']['source_exit_foil'] = {'material': 'G4_Al', 'thickness_mm': thickness,
                'distance_from_source_mm': .01, 'width_mm': 20., 'status': 'Hypothetical planar exit foil; not measured source capsule'}
        local['status'] = 'Transport-only candidate scenario; no scintillation, avalanches or ADC'
        path = out/'config.json'; path.write_text(json.dumps(local, indent=2)+'\n')
        seed = 281000+i; started = time.monotonic()
        record = {'source': source, 'events': args.events, 'seed': seed, 'status': 'running',
                  'config_sha256': sha(path), 'executable_sha256': executable_sha, 'source_manifest': '../summary.json'}
        (out/'run.json').write_text(json.dumps(record, indent=2)+'\n')
        with (out/'run.log').open('w') as log:
            result = subprocess.run([str(executable), str(path.resolve()), source, str(args.events), str(seed), str((out/'events.root').resolve())], stdout=log, stderr=subprocess.STDOUT)
        record.update(exit_code=result.returncode, elapsed_seconds=time.monotonic()-started,
                      status='completed; validation pending' if result.returncode == 0 else 'failed')
        (out/'run.json').write_text(json.dumps(record, indent=2)+'\n'); result.check_returncode()
        validation = check(out/'events.root')
        (out/'events_validation.json').write_text(json.dumps(validation, indent=2)+'\n')
        file = ROOT.TFile.Open(str(out/'events.root')); tree = file.Get('Events')
        tree.SetBranchStatus('*', 0); tree.SetBranchStatus('edep_origin_species_MeV', 1)
        deposits = np.array([np.array(event.edep_origin_species_MeV) for event in tree]).reshape(-1, 6, 3); file.Close()
        np.savez_compressed(out/'deposits.npz', edep_origin_species_MeV=deposits)
        record['status'] = 'completed and numerically validated'; (out/'run.json').write_text(json.dumps(record, indent=2)+'\n')
        row = {'name': name, 'source': source, 'gap_mm': gap, 'foil_Al_mm': thickness, 'seed': seed,
               'root_sha256': sha(out/'events.root'), 'deposits_sha256': sha(out/'deposits.npz'),
               'validation': validation, 'elapsed_seconds': record['elapsed_seconds']}
        report['scenarios'].append(row)
        (args.out/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps({'scenario': name, 'deposits': validation['deposited_events'], 'alpha_deposits': validation['alpha_deposit_events']}), flush=True)
    assert sha(executable) == executable_sha, 'Executable changed during scan'
    for name, expected in manifest.items(): assert sha(HERE/name) == expected, 'Source changed during scan'
    report['completed'] = True; (args.out/'summary.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__': main()
