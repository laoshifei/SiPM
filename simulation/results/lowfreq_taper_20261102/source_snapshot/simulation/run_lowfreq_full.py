"""Full-mother validation of the frozen native low-frequency candidate."""
import argparse
import concurrent.futures
import subprocess
import time
import json
from lowfreq_common import *


def prepare():
    p=read(OUT/'protocol.json');fit=read(OUT/'fit.json');entries={}
    for key,e in p['entries'].items():
        current=[str(PARENT/'configs'/f'run12_last20_B128_eval{ef}_spectrum.json') for ef in [0,1]]
        zero=[str(SPLIT/'configs'/f'run12_last20_B128_eval{ef}_zero.json') for ef in [0,1]]
        candidate=[str(PROJECT/fit['folds'][str(ef)]['config']) for ef in [0,1]]
        path=OUT/'configs'/f'{key}_full_jobs.json';reference=FULL/'native'/f'{key}.root'
        plan={'key':key,'events':e['events'],'seed':e['seed'],'folds':read(FULL/'configs'/f'{key}.json')['folds'],
            'reference':str(reference),'output':str(OUT/'native'/f'{key}.root'),
            'cases':[{'name':n,'configs':c,'noise_replica':r,'noise_scope':'all_readout'} for n,c,r in [
                ('current_r0',current,0),('zero_cv_r0',zero,0),('split_r0',candidate,0),('split_r1',candidate,1)]]}
        save(path,plan);entries[key]={**e,'reference':relative(reference),'reference_sha256':sha(reference),
            'plan':relative(path),'plan_sha256':sha(path),'output':relative(OUT/'native'/f'{key}.root')}
    save(OUT/'native_protocol.json',{'entries':entries,'fit_sha256':sha(OUT/'fit.json'),'protocol_sha256':sha(OUT/'protocol.json'),
        'binary_sha256':sha(BUILD/'replay_spectral_pool'),'code_sha256':sha(__file__),
        'branch_names':'The inherited native driver calls the selected candidate split_r0/r1; analysis labels these taper_r0/r1.'})


def one(key):
    p=read(OUT/'native_protocol.json');e=p['entries'][key];assert not (PROJECT/e['output']).exists()
    for path,h in [(e['path'],e['full_parent_sha256']),(e['reference'],e['reference_sha256']),(e['plan'],e['plan_sha256'])]:assert sha(PROJECT/path)==h,path
    cmd=[str(BUILD/'replay_spectral_pool'),str(PROJECT/e['path']),str(PROJECT/e['plan'])];start=time.time()
    with (OUT/'logs'/f'{key}_native.log').open('w') as stream:subprocess.run(cmd,cwd=PROJECT,stdout=stream,stderr=subprocess.STDOUT,check=True)
    import ROOT
    root=ROOT.TFile.Open(str(PROJECT/e['output']));info=json.loads(str(root.Get('RunInfo').GetString().Data()))
    assert info['completed'] and info['events']==EVENTS and root.Get('Readout').GetEntries()==EVENTS
    assert info['legacy_control_readouts_bit_identical']==2*EVENTS and info['plan']==read(PROJECT/e['plan'])
    assert not any(info['clipped_samples']);root.Close()
    save(OUT/'logs'/f'{key}_native.json',{'output_sha256':sha(PROJECT/e['output']),'binary_sha256':p['binary_sha256'],
        'plan_sha256':e['plan_sha256'],'seconds':time.time()-start,'command':cmd,
        'legacy_control_readouts_bit_identical':2*EVENTS,'clipped_samples':info['clipped_samples']})
    print(key,'full native low-frequency candidate complete',round(time.time()-start,1),'seconds',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=4);args=parser.parse_args();p=read(OUT/'protocol.json')
    for name,h in {**p['code_sha256'],**p['inherited_code_sha256'],**p['binary_sha256']}.items():assert sha(PROJECT/name)==h,name
    if not (OUT/'native_protocol.json').exists():prepare()
    np_=read(OUT/'native_protocol.json');assert np_['fit_sha256']==sha(OUT/'fit.json') and np_['code_sha256']==sha(__file__)
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        for result in executor.map(one,list(np_['entries'])):pass
