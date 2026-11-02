"""Native calibration grid; reuse Geant4 histories without changing physics."""
import argparse
import concurrent.futures
import json
import subprocess
import time
import ROOT
from lowfreq_common import *
from reference import wave_features
from refine_waveforms import advanced_features
from charge_scale_common import charge_bin


def one(name):
    p=read(OUT/'protocol.json');e=p['grid_jobs'][name];plan=read(PROJECT/e['path']);dest=OUT/'logs'/f'{name}.json'
    assert not dest.exists() and sha(PROJECT/e['path'])==e['sha256'];assert sha(PROJECT/e['parent'])==e['parent_sha256']
    assert not any(Path(j['output']).exists() for j in plan['jobs'])
    command=[str(BUILD/'replay_readout_cached'),str(PROJECT/e['parent']),str(PROJECT/e['path'])];start=time.time()
    with (OUT/'logs'/f'{name}.log').open('w') as log:subprocess.run(command,cwd=PROJECT,stdout=log,stderr=subprocess.STDOUT,check=True)
    old=load(FULL/'arrays'/f"{e['key']}_original.npz");records={};ids=np.asarray(plan['ids']);identity=0
    reference=ROOT.TFile.Open(str(SPLIT/'native'/f"{e['key']}.root"));ref_tree=reference.Get('Readout')
    for job in plan['jobs']:
        root=ROOT.TFile.Open(job['output']);info=json.loads(str(root.Get('RunInfo').GetString().Data()));tree=root.Get('Readout')
        assert info['completed'] and info['ids']==plan['ids'] and info['config']==read(job['config'])
        assert info['clipped_samples']==0 and info['noise_replica']==job['noise_replica'] and info['new_independent_mothers']==0
        tree.SetBranchStatus('*',0);tree.SetBranchStatus('adc',1);tree.SetBranchStatus('event_id',1)
        waves=np.empty((len(ids),2496),np.uint16)
        for i,row in enumerate(tree):assert row.event_id==ids[i];waves[i]=np.asarray(row.adc)
        root.Close();r=job['noise_replica'];label=job['candidate']
        if label=='identity':
            ref_tree.SetBranchStatus('*',0);ref_tree.SetBranchStatus(f'adc_split_r{r}',1)
            for i,id_ in enumerate(ids):
                assert ref_tree.GetEntry(int(id_))>0;np.testing.assert_array_equal(waves[i],np.asarray(getattr(ref_tree,f'adc_split_r{r}')))
            identity+=len(ids)
        baseline=read(SPLIT/'logs'/f"{e['key']}_extraction.json")['variants'][f'split_r{r}']['global_baseline_RMS']
        f=wave_features(waves);advanced=advanced_features(waves,f['t10_sample'],f['peak_sample'],f['amplitude'],f['baseline'],baseline)
        selected=(f['diagnostic_flags']==0)&~advanced['pretrigger_flag_primary']&(f['amplitude']>=200)&(f['t10_sample']>=384)
        q=f['charge_long'];common=selected&(q>=p['q_edges'][0])&(q<=p['q_edges'][-1]);ix=np.flatnonzero(common)
        X=np.full((len(ids),6),np.nan);profile=np.full((len(ids),76),np.nan)
        a=features(waves[ix],ids[ix],q[ix]);np.testing.assert_array_equal(a['qcalc'],q[ix]);X[ix]=np.c_[a['scores'],a['F']];profile[ix]=a['mean_wave']
        path=OUT/'grid_arrays'/f"{job['key']}.npz"
        np.savez_compressed(path,ids=ids+EVENTS*e['batch'],fold=np.full(len(ids),e['fold']),q=old['q'][ids],qcalc=q,
            X=X,mean_wave=profile,selected=selected,common=common,old_common=old['old_common'][ids],truth=old['truth'][ids],
            supported=common&np.isin(charge_bin(q,p),np.asarray(p['support_bins'][SOURCE])+1))
        records[job['key']]={'array_sha256':sha(path),'root_sha256':sha(job['output']),'config_sha256':sha(job['config']),
            'mothers':len(ids),'current_common':int(common.sum()),'clipped_samples':0}
    reference.Close();save(dest,{'jobs':records,'identity_ADC_readouts':identity,'seconds':time.time()-start,'command':command,'protocol_sha256':sha(OUT/'protocol.json')})
    print(name,'native grid and identity checks complete',round(time.time()-start,1),'seconds',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=4);args=parser.parse_args();p=read(OUT/'protocol.json')
    for n,h in {**p['code_sha256'],**p['inherited_code_sha256'],**p['binary_sha256']}.items():assert sha(PROJECT/n)==h,n
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        for result in executor.map(one,list(p['grid_jobs'])):pass
