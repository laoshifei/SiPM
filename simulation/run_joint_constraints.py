"""Run full native transport or calibrated response replays, validate and extract."""
import argparse
import concurrent.futures
import os
import subprocess
import time
from pathlib import Path
from prepare_joint_constraints import OUT, PROJECT, BUILD
from prepare_pid import read
from publication_stats import save, sha

PYTHON = '/home/xc/software/jupyter/.venv/bin/python'


def one(entry, stage):
    dest = OUT/stage; key = entry['key']; start = time.time(); steps = []
    assert sha(PROJECT/entry['config']) == entry['config_sha256']
    root = PROJECT/entry['path']
    jobs = [('native' if entry['native'] else 'replay', entry['command'], root),
            ('validate', [PYTHON, '-u', 'simulation/validate.py', str(root)], root.with_name(root.stem+'_validation.json')),
            ('extract', [PYTHON, '-u', 'simulation/run_joint_constraints.py', stage, '--extract', key], dest/'extraction'/f'{key}.json')]
    for label, command, target in jobs:
        if target.exists():
            steps.append({'stage': label, 'existing': True}); continue
        log = dest/'logs'/f'{key}_{label}.log'
        with log.open('w') as f:
            result = subprocess.run(command, cwd=PROJECT, stdout=f, stderr=subprocess.STDOUT)
        if result.returncode: raise RuntimeError(f'{key} {label} exit {result.returncode}; see {log}')
        steps.append({'stage': label, 'command': command, 'exit': result.returncode, 'log': str(log.relative_to(PROJECT))})
        print(stage, key, label, 'completed', flush=True)
    return {'key': key, 'seconds': time.time()-start, 'steps': steps}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['profile', 'calibration', 'validation']); p.add_argument('--extract')
    p.add_argument('--workers', type=int, default=3); args = p.parse_args()
    dest = OUT/args.stage; protocol = read(dest/'protocol.json')
    for folder in ['arrays', 'extraction', 'logs']: (dest/folder).mkdir(exist_ok=True)
    if args.extract:
        from extract_npe_profile import extract
        assert not (dest/'extraction'/f'{args.extract}.json').exists()
        pca = read(PROJECT/'analysis/paper_baseline/results/publication_20261028/point.json')['windows']['3000']['pca']
        extract(protocol['entries'][args.extract], dest, protocol['q_edges'], pca); return
    assert not (dest/'execution.json').exists(), 'Stage already complete'
    for name, value in protocol['binary_sha256'].items(): assert sha(BUILD/name) == value
    os.environ['OPENBLAS_NUM_THREADS'] = '1'; os.environ['OMP_NUM_THREADS'] = '1'
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(one, e, args.stage) for e in protocol['entries'].values()]
        rows = [f.result() for f in concurrent.futures.as_completed(futures)]
    save(dest/'execution.json', {'scope': __doc__, 'code_sha256': sha(Path(__file__)),
        'protocol_sha256': sha(dest/'protocol.json'), 'jobs': rows,
        'new_independent_mothers': sum(e['events'] for e in protocol['entries'].values() if e['native'])})
    print(args.stage, 'completed with full native checks.', flush=True)


if __name__ == '__main__': main()
