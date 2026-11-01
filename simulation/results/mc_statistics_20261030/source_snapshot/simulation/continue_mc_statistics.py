"""Finish pending batches in dependency order, then generate the analysis and notebook."""
import concurrent.futures
import subprocess
import time
from run_mc_statistics import replay_one
from mc_statistics_common import *


def wait_for(path):
    count = 0
    while not path.exists():
        time.sleep(10)
        count += 1
        if count % 6 == 0: print('Waiting for', path.name, flush=True)


def pending(entry):
    wait_for(OUT/'logs'/f"{entry['key']}_transport.json")
    return replay_one(entry)


def main():
    assert not (OUT/'pipeline_execution.json').exists()
    p = read(OUT/'protocol.json')
    for path, value in p['binary_sha256'].items(): assert sha(PROJECT/path) == value
    wait_for(OUT/'early_replay_execution.json')
    entries = sorted([e for e in p['entries'].values() if e['batch'] in [2, 3]], key=lambda x: (x['batch'], x['run']))
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool0:
        futures = [pool0.submit(pending, e) for e in entries]
        records = [f.result() for f in concurrent.futures.as_completed(futures)]
    save(OUT/'remaining_replay_execution.json', {'scope': 'Readout jobs on completed fresh parent files, overlapping remaining transport.',
         'jobs': records, 'new_independent_mothers': 0, 'protocol_sha256': sha(OUT/'protocol.json')})
    wait_for(OUT/'transport_execution.json')
    commands = [
        ('replay_finalize', [PYTHON, '-u', 'simulation/run_mc_statistics.py', 'replay', '--workers', '2'], OUT/'replay_execution.json'),
        ('analysis', [PYTHON, '-u', 'simulation/analyze_mc_statistics.py', '--workers', '2'], OUT/'analysis.json'),
        ('figures', [PYTHON, '-u', 'simulation/plot_mc_statistics.py'], OUT/'figure_record.json'),
        ('summary', [PYTHON, '-u', 'simulation/summarize_mc_statistics.py'], OUT/'decision.json'),
        ('notebook', [PYTHON, '-u', 'jupyter/codex_0001_build_mc_statistics_notebook.py'], PROJECT/'jupyter/codex_0001_csi_independent_mc_statistics.ipynb')]
    steps = []
    for label, command, target in commands:
        if target.exists():
            steps.append({'stage': label, 'existing': True, 'output_sha256': sha(target)}); continue
        start = time.monotonic(); log = OUT/'logs'/f'pipeline_{label}.log'
        save(OUT/'pipeline_status.json', {'stage': label, 'command': command, 'completed': False})
        print('Starting', label, flush=True)
        with log.open('w') as f: result = subprocess.run(command, cwd=PROJECT, stdout=f, stderr=subprocess.STDOUT)
        if result.returncode:
            save(OUT/'pipeline_status.json', {'stage': label, 'command': command, 'completed': False, 'exit': result.returncode})
            raise RuntimeError(f'{label} failed; see {log}')
        steps.append({'stage': label, 'command': command, 'monotonic_seconds': time.monotonic()-start,
                      'log': relative(log), 'output_sha256': sha(target)})
        print('Completed', label, flush=True)
    save(OUT/'pipeline_execution.json', {'code_sha256': sha(__file__), 'steps': steps, 'new_independent_mothers': 0,
         'scope': 'Coordinator only. Parent histories counted by transport_execution.json; no additional transport or fitting.'})
    save(OUT/'pipeline_status.json', {'stage': 'ready_for_review_and_verification', 'completed': True})


if __name__ == '__main__': main()
