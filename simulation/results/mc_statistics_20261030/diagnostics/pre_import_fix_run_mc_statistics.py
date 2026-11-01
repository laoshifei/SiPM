"""Run independent full transport and unchanged C++ readout responses with resumable records."""
import argparse
import concurrent.futures
import json
import os
import subprocess
import time
import ROOT
from mc_statistics_common import *

ROOT.EnableThreadSafety()


def root_info(path):
    f = ROOT.TFile.Open(str(path)); assert f and not f.IsZombie(), path
    obj = f.Get('RunInfo'); assert obj, path
    info = json.loads(str(obj.GetString().Data())); f.Close()
    assert info['completed'], path
    return info


def extract_native(key):
    from extract_npe_profile import extract
    p = read(OUT/'protocol.json'); entry = p['entries'][key]
    record = OUT/'transport/extraction'/f'{key}.json'
    if not record.exists(): extract(entry, OUT/'transport', p['q_edges'], PCA)
    a = load(OUT/'transport/arrays'/f'{key}.npz')
    assert read(record)['root_sha256'] == sha(PROJECT/entry['path'])
    result = features(a['waves'], a['ids'], a['q'], a['signal'])
    result['n_parent'] = a['n_parent']
    np.testing.assert_allclose(result['scores'], a['scores'], rtol=1e-12, atol=1e-12)
    target = OUT/'arrays'/f'{key}_frozen.npz'
    if target.exists():
        old = load(target)
        for name, value in result.items(): np.testing.assert_array_equal(old[name], value)
    else: np.savez_compressed(target, **result)
    print(key, 'baseline selection:', len(result['ids']), flush=True)


def extract_replay(key):
    p = read(OUT/'protocol.json'); entry = p['entries'][key]
    plan = read(OUT/'configs'/f'{key}_jobs.json')
    base = load(OUT/'arrays'/f'{key}_frozen.npz'); records = {}
    for job in plan['jobs']:
        path = Path(job['output']); info = root_info(path)
        assert info['config'] == read(job['config']) and info['seed'] == entry['seed']
        assert info['phase_ns'] == 0 and info['clipped_samples'] == 0 and info['new_independent_mothers'] == 0
        assert info['ids'] == base['ids'].tolist()
        f = ROOT.TFile.Open(str(path)); tree = f.Get('Readout')
        tree.SetBranchStatus('*', 0)
        for name in ['event_id', 'adc', 'signal_mV']: tree.SetBranchStatus(name, 1)
        ids = []; waves = []; signal = []
        for row in tree:
            ids.append(int(row.event_id)); waves.append(np.array(row.adc)); signal.append(np.array(row.signal_mV))
        f.Close(); np.testing.assert_array_equal(ids, base['ids'])
        waves = np.asarray(waves, np.uint16)
        if job['require_identity']:
            np.testing.assert_array_equal(waves, base['waves'])
            assert info['equal_parent_ADC'] == len(ids)
        ro = info['config']['sipm']['readout']
        lsb = (ro['adc_max_mV']-ro['adc_min_mV'])/2**ro['adc_bits']
        a = features(waves, base['ids'], base['q'], np.asarray(signal)/lsb)
        a['n_parent'] = base['n_parent']
        target = OUT/'arrays'/f"{job['key']}.npz"
        if target.exists():
            old = load(target)
            for name, value in a.items(): np.testing.assert_array_equal(old[name], value)
        else: np.savez_compressed(target, **a)
        records[job['key']] = {'config_sha256': sha(job['config']), 'root_sha256': sha(path),
            'array_sha256': sha(target), 'entries': len(ids), 'equal_parent_ADC': info['equal_parent_ADC'],
            'new_independent_mothers': 0}
    save(OUT/'logs'/f'{key}_replay_extraction.json', {'code_sha256': sha(__file__), 'jobs': records,
         'parent_root_sha256': sha(PROJECT/entry['path']), 'plan_sha256': sha(OUT/'configs'/f'{key}_jobs.json')})
    print(key, 'extracted', len(records), 'native responses per selected mother', flush=True)


def run_command(key, label, cmd):
    log = OUT/'logs'/f'{key}_{label}.log'; start = time.time()
    with log.open('w') as f:
        result = subprocess.run(cmd, cwd=PROJECT, stdout=f, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'{key} {label}: exit {result.returncode}; see {log}')
    print(key, label, 'completed in', round(time.time()-start, 1), 's', flush=True)
    return {'label': label, 'command': cmd, 'seconds': time.time()-start, 'log': relative(log), 'exit': 0}


def transport_one(entry):
    key = entry['key']; record = OUT/'logs'/f'{key}_transport.json'
    if record.exists():
        r = read(record)
        for path, value in r['outputs_sha256'].items(): assert sha(PROJECT/path) == value
        return r
    assert sha(PROJECT/entry['config']) == entry['config_sha256']
    root = PROJECT/entry['path']; steps = []
    if not root.exists(): steps.append(run_command(key, 'native', entry['command']))
    info = root_info(root)
    assert info['config'] == read(PROJECT/entry['config']) and info['seed'] == entry['seed']
    assert info['events_requested'] == entry['events'] and info['source'] == entry['source']
    validation = root.with_name(root.stem+'_validation.json')
    if not validation.exists():
        steps.append(run_command(key, 'validate', [PYTHON, '-u', 'simulation/validate.py', str(root)]))
    assert read(validation)['passed'] and read(validation)['events'] == entry['events']
    steps.append(run_command(key, 'extract_native', [PYTHON, '-u', __file__, 'extract-native', '--key', key]))
    files = [root, validation, OUT/'transport/arrays'/f'{key}.npz', OUT/'transport/extraction'/f'{key}.json',
             OUT/'arrays'/f'{key}_frozen.npz']
    r = {'key': key, 'code_sha256': sha(__file__), 'steps': steps, 'new_independent_mothers': entry['events'],
         'outputs_sha256': {relative(f): sha(f) for f in files}}
    save(record, r); return r


def replay_one(entry):
    key = entry['key']; record = OUT/'logs'/f'{key}_replay.json'
    if record.exists():
        r = read(record)
        assert sha(OUT/'logs'/f'{key}_replay_extraction.json') == r['extraction_sha256']
        return r
    p = read(OUT/'protocol.json'); base = load(OUT/'arrays'/f'{key}_frozen.npz')
    jobs = [{'key': key+'_identity', 'config': str(PROJECT/entry['config']),
             'output': str(OUT/'replay'/f'{key}_identity.root'), 'phase_ns': 0., 'require_identity': True}]
    for name, c in p['responses'].items():
        if not name.startswith(f"run{entry['run']}_"): continue
        assert sha(PROJECT/c['config']) == c['config_sha256']
        suffix = name.split('_', 1)[1]
        jobs.append({'key': f'{key}_{suffix}', 'config': str(PROJECT/c['config']),
                     'output': str(OUT/'replay'/f'{key}_{suffix}.root'), 'phase_ns': 0., 'require_identity': False})
    plan = {'ids': base['ids'].tolist(), 'jobs': jobs}; pp = OUT/'configs'/f'{key}_jobs.json'
    if pp.exists(): assert read(pp) == plan
    else: save(pp, plan)
    # Completed prior jobs may be resumed, while incomplete ROOT files require audit.
    missing = []
    for job in jobs:
        if not Path(job['output']).exists(): missing.append(job)
        else: root_info(job['output'])
    steps = []
    if missing:
        pending = OUT/'configs'/f'{key}_pending_jobs.json'
        save(pending, {'ids': plan['ids'], 'jobs': missing})
        steps.append(run_command(key, 'replay', [str(REPLAY), str(PROJECT/entry['path']), str(pending)]))
    steps.append(run_command(key, 'extract_replay', [PYTHON, '-u', __file__, 'extract-replay', '--key', key]))
    r = {'key': key, 'code_sha256': sha(__file__), 'steps': steps, 'new_independent_mothers': 0,
         'replay_entries': len(base['ids'])*len(jobs), 'distinct_selected_mothers': len(base['ids']),
         'extraction_sha256': sha(OUT/'logs'/f'{key}_replay_extraction.json'), 'plan_sha256': sha(pp)}
    save(record, r); return r


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['transport', 'replay', 'extract-native', 'extract-replay'])
    parser.add_argument('--key'); parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    p = read(OUT/'protocol.json')
    assert p['common_code_sha256'] == sha(PROJECT/'simulation/mc_statistics_common.py')
    for path, value in p['binary_sha256'].items(): assert sha(PROJECT/path) == value
    if args.stage == 'extract-native': extract_native(args.key); return
    if args.stage == 'extract-replay': extract_replay(args.key); return
    dest = OUT/f'{args.stage}_execution.json'
    assert not dest.exists(), 'Stage already complete'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'; os.environ['OMP_NUM_THREADS'] = '1'
    entries = sorted(p['entries'].values(), key=lambda e: (e['batch'], e['run']))
    function = transport_one if args.stage == 'transport' else replay_one
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool0:
        futures = [pool0.submit(function, entry) for entry in entries]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
    record = {'scope': __doc__, 'code_sha256': sha(__file__), 'protocol_sha256': sha(OUT/'protocol.json'),
              'jobs': results, 'new_independent_mothers': sum(r['new_independent_mothers'] for r in results)}
    if args.stage == 'replay': record['replay_entries'] = sum(r['replay_entries'] for r in results)
    save(dest, record)
    print(args.stage, 'completed.', flush=True)


if __name__ == '__main__': main()
