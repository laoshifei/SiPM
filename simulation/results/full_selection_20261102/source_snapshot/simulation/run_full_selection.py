"""Run immutable full-pool plans with bounded native event memory."""
import argparse
import concurrent.futures
import subprocess
import time
from full_selection_common import *


def one(key):
    p=read(OUT/'protocol.json');entry=p['entries'][key]
    output=PROJECT/entry['output'];assert not output.exists(),'Refuse existing/partial native output'
    assert sha(PROJECT/entry['plan'])==entry['plan_sha256']
    assert sha(PROJECT/entry['path'])==entry['full_parent_sha256']
    command=[str(BUILD/'replay_full_pool'),str(PROJECT/entry['path']),str(PROJECT/entry['plan'])]
    start=time.time();log=OUT/'logs'/f'{key}_native.log'
    with log.open('w') as stream:subprocess.run(command,cwd=PROJECT,stdout=stream,stderr=subprocess.STDOUT,check=True)
    import ROOT,json
    f=ROOT.TFile.Open(str(output));meta=json.loads(str(f.Get('RunInfo').GetString().Data()))
    assert meta['completed'] and meta['events']==entry['events'] and f.Get('Readout').GetEntries()==entry['events']
    assert meta['plan']==read(PROJECT/entry['plan']) and meta['seed']==entry['seed']
    assert meta['new_independent_mothers']==0
    for j in [0,1]:assert meta['configs'][j]==read(meta['plan']['configs'][j])
    f.Close()
    save(OUT/'logs'/f'{key}_native.json',{'command':command,'seconds':time.time()-start,
        'output_sha256':sha(output),'plan_sha256':entry['plan_sha256'],'binary_sha256':sha(BUILD/'replay_full_pool'),
        'events':entry['events'],'clipped_samples':meta['clipped_samples'],'equal_original_ADC':meta['equal_original_ADC']})
    print(key,'full-pool readouts complete',round(time.time()-start,1),'seconds',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--keys',nargs='+');args=parser.parse_args();p=read(OUT/'protocol.json')
    for name,h in {**p['code_sha256'],**p['inherited_code_sha256']}.items():assert sha(PROJECT/name)==h,name
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        for result in executor.map(one,args.keys or list(p['entries'])):pass
