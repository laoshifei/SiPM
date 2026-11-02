"""Run frozen noise-replica plans and extract compact features without reselection."""
import argparse
import concurrent.futures
import hashlib
import json
import subprocess
import time
from pathlib import Path
import ROOT
from noise_replay_common import *

ROOT.EnableThreadSafety()


def info(path):
    f=ROOT.TFile.Open(str(path));assert f and not f.IsZombie(),path
    meta=f.Get('RunInfo');assert meta,path
    result=json.loads(str(meta.GetString().Data()));f.Close();assert result['completed'];return result


def one(name):
    p=read(OUT/'replay_protocol.json');plan=p['plans'][name];receipt=OUT/'logs'/f'{name}_receipt.json'
    if receipt.exists():
        r=read(receipt)
        for path,value in r['outputs_sha256'].items():assert sha(PROJECT/path)==value,path
        return
    assert sha(PROJECT/plan['path'])==plan['sha256']
    pending=[]
    for job in plan['jobs']:
        if Path(job['output']).exists():info(job['output'])
        else:pending.append(job)
    start=time.time();binary=BUILD/'replay_readout_cached'
    if pending:
        path=OUT/'configs'/f'{name}_pending.json';save(path,{'ids':plan['ids'],'jobs':pending})
        with (OUT/'logs'/f'{name}_native.log').open('w') as log:
            subprocess.run([str(binary),str(PROJECT/plan['parent']),str(path)],cwd=PROJECT,stdout=log,stderr=subprocess.STDOUT,check=True)
    print('Native replicas ready',name,'jobs',len(plan['jobs']),'seconds',round(time.time()-start,1),flush=True)
    run=int(name.split('_')[1][3:]);source=dict((r,s) for s,r in SOURCES)[run]
    batch=int(name.split('_')[0][1:]);base=load(PARENT/'arrays'/f'b{batch:02d}_run{run}_frozen.npz')
    index=np.searchsorted(base['ids'],plan['ids']);np.testing.assert_array_equal(base['ids'][index],plan['ids'])
    records={};reference_path=None;old=None
    for job in plan['jobs']:
        path=Path(job['output']);meta=info(path)
        assert meta['ids']==plan['ids'] and meta['config']==read(job['config'])
        assert meta['noise_replica']==job['noise_replica'] and meta['noise_scope']==job['noise_scope']
        assert meta['new_independent_mothers']==0 and meta['phase_ns']==0.
        assert meta['clipped_samples']==0,'Clipping must be audited before interpreting this fixed cohort'
        if reference_path!=job['reference_array']:
            old=load(PROJECT/job['reference_array']);reference_path=job['reference_array']
        old_ix=np.searchsorted(old['ids'],plan['ids']);np.testing.assert_array_equal(old['ids'][old_ix],plan['ids'])
        f=ROOT.TFile.Open(str(path));tree=f.Get('Readout');tree.SetBranchStatus('*',0)
        for field in ['event_id','adc','signal_mV','baseline_readout_seed','signal_readout_seed']:tree.SetBranchStatus(field,1)
        ids=[];waves=[];signal=[];bs=[];ss=[]
        for row in tree:
            ids.append(int(row.event_id));waves.append(np.array(row.adc,dtype=np.uint16));signal.append(np.array(row.signal_mV))
            bs.append(int(row.baseline_readout_seed));ss.append(int(row.signal_readout_seed))
        f.Close();np.testing.assert_array_equal(ids,plan['ids']);waves=np.asarray(waves);signal=np.asarray(signal)
        ro=meta['config']['sipm']['readout'];lsb=(ro['adc_max_mV']-ro['adc_min_mV'])/2**ro['adc_bits']
        np.testing.assert_array_equal(signal/lsb,old['signal'][old_ix])
        if job['require_identity']:
            np.testing.assert_array_equal(waves,base['waves'][index]);assert meta['equal_parent_ADC']==len(ids)
        if job['noise_replica']==0 and meta['config']==read(PROJECT/read(PARENT/'protocol.json')['responses'].get(
            f"{job['context']}_eval{job['fold']}_spectrum",{'config':relative(PROJECT/job['config'])})['config']):
            np.testing.assert_array_equal(waves,old['waves'][old_ix])
        ap=OUT/'arrays'/f"{job['key']}.npz"
        if not job['native_identity']:
            raw=features(waves,np.asarray(ids),base['q'][index]);raw['fold']=np.full(len(ids),job['fold'])
            ref=read(QDC/'references'/f"{job['context']}.json");a=compact(raw,p,source,ref)
            a['baseline_seed']=np.array(bs,dtype=np.uint64);a['signal_seed']=np.array(ss,dtype=np.uint64)
            assert np.isfinite(a['X']).all(),'Undefined repeated-readout features require explicit audit, not event removal'
            if ap.exists():
                saved=load(ap)
                for k,value in a.items():np.testing.assert_array_equal(saved[k],value)
            else:np.savez_compressed(ap,**a)
        records[job['key']]={'root_sha256':sha(path),'array_sha256':None if job['native_identity'] else sha(ap),
            'config_sha256':sha(job['config']),'entries':len(ids),'noise_replica':job['noise_replica'],
            'noise_scope':job['noise_scope'],'unchanged_noiseless_signal':True,
            'signal_sha256':hashlib.sha256(signal.tobytes()).hexdigest(),'clipped_samples':meta['clipped_samples'],
            'new_independent_mothers':0,'native_identity':job['native_identity']}
    files=[OUT/'arrays'/f'{k}.npz' for k,v in records.items() if not v['native_identity']]
    result={'plan':name,'plan_sha256':plan['sha256'],'protocol_sha256':sha(OUT/'replay_protocol.json'),
        'jobs':records,'seconds':time.time()-start,'new_independent_mothers':0,
        'outputs_sha256':{relative(f):sha(f) for f in files}}
    save(receipt,result);print('Extracted and verified',name,'unchanged histories/seed records; no event reselection.',flush=True)


def main(workers):
    p=read(OUT/'replay_protocol.json')
    for k,v in p['code_sha256'].items():assert sha(PROJECT/k)==v,k
    for k,v in p['binary_sha256'].items():assert sha(PROJECT/k)==v,k
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures=[executor.submit(one,k) for k in p['plans']]
        for f in concurrent.futures.as_completed(futures):f.result()
    save(OUT/'replay_execution.json',{'protocol_sha256':sha(OUT/'replay_protocol.json'),
        'plans':{name:read(OUT/'logs'/f'{name}_receipt.json') for name in p['plans']},
        'physical_replay_entries':p['physical_replay_entries'],'new_independent_mothers':0})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=4);args=parser.parse_args();main(args.workers)
