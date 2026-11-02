"""Freeze chosen candidate plans, then replay all available original mothers."""
import argparse
import concurrent.futures
import copy
import subprocess
import time
import json
from spectral_split_common import *


def prepare_plans():
    p=read(OUT/'protocol.json');fit=read(OUT/'fit.json');parent=read(PARENT/'protocol.json');entries={}
    assert not (OUT/'native_protocol.json').exists()
    for key,e in p['entries'].items():
        context=f"run{e['run']}_last20_B128";current=[];zero=[];candidate=[]
        for ef in [0,1]:
            cp=PROJECT/parent['responses'][f'{context}_eval{ef}_spectrum']['config'];current.append(str(cp))
            config=copy.deepcopy(read(cp));config['signal_dependent_noise']['event_variance_cv']=0.
            zp=OUT/'configs'/f'{context}_eval{ef}_zero.json'
            if zp.exists():assert read(zp)==config
            else:save(zp,config)
            zero.append(str(zp));choice=fit[e['source']]['folds'][str(ef)];candidate.append(str(PROJECT/choice['config']))
            assert sha(candidate[-1])==choice['config_sha256']
        original=read(PROJECT/e['plan']);path=OUT/'configs'/f'{key}_jobs.json'
        plan={'key':key,'events':e['events'],'seed':e['seed'],'folds':original['folds'],
            'reference':str(PROJECT/e['reference']),'output':str(OUT/'native'/f'{key}.root'),
            'cases':[{'name':n,'configs':c,'noise_replica':r,'noise_scope':'all_readout'} for n,c,r in [
                ('current_r0',current,0),('zero_cv_r0',zero,0),('split_r0',candidate,0),('split_r1',candidate,1)]]}
        save(path,plan);entries[key]={**e,'plan':relative(path),'plan_sha256':sha(path),'output':relative(OUT/'native'/f'{key}.root')}
    save(OUT/'native_protocol.json',{'entries':entries,'fit_sha256':sha(OUT/'fit.json'),
        'protocol_sha256':sha(OUT/'protocol.json'),'code_sha256':sha(__file__),'binary_sha256':sha(BUILD/'replay_spectral_pool')})


def one(key):
    p=read(OUT/'native_protocol.json');e=p['entries'][key];output=PROJECT/e['output']
    assert not output.exists(),'Refuse existing/partial native output'
    assert sha(PROJECT/e['path'])==e['full_parent_sha256'] and sha(PROJECT/e['reference'])==e['reference_sha256']
    assert sha(PROJECT/e['plan'])==e['plan_sha256']
    command=[str(BUILD/'replay_spectral_pool'),str(PROJECT/e['path']),str(PROJECT/e['plan'])]
    start=time.time()
    with (OUT/'logs'/f'{key}_native.log').open('w') as stream:subprocess.run(command,cwd=PROJECT,stdout=stream,stderr=subprocess.STDOUT,check=True)
    import ROOT
    f=ROOT.TFile.Open(str(output));info=json.loads(str(f.Get('RunInfo').GetString().Data()))
    assert info['completed'] and info['events']==e['events'] and f.Get('Readout').GetEntries()==e['events']
    assert info['legacy_control_readouts_bit_identical']==2*e['events'] and info['plan']==read(PROJECT/e['plan'])
    f.Close();save(OUT/'logs'/f'{key}_native.json',{'output_sha256':sha(output),'binary_sha256':p['binary_sha256'],
        'plan_sha256':e['plan_sha256'],'command':command,'seconds':time.time()-start,
        'legacy_control_readouts_bit_identical':info['legacy_control_readouts_bit_identical'],'clipped_samples':info['clipped_samples']})
    print(key,'spectral native complete',round(time.time()-start,1),'seconds',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=4);parser.add_argument('--keys',nargs='+');args=parser.parse_args()
    p=read(OUT/'protocol.json')
    for n,h in {**p['code_sha256'],**p['inherited_code_sha256']}.items():assert sha(PROJECT/n)==h,n
    if not (OUT/'native_protocol.json').exists():prepare_plans()
    n=read(OUT/'native_protocol.json');assert sha(OUT/'fit.json')==n['fit_sha256'] and sha(BUILD/'replay_spectral_pool')==n['binary_sha256']
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        for result in executor.map(one,args.keys or list(n['entries'])):pass
