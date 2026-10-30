"""Replay frozen Geant4 histories and extract paired response diagnostics."""
import argparse
import concurrent.futures
import json
import os
import subprocess
import time
import numpy as np
import ROOT
from prepare_conditional_response import OUT, PARENT, PROJECT, BUILD, relative
from prepare_pid import read
from extract_pid import load, score_waves, label_truth
from compare import mc_wave_features
from reference import aligned_shapes
from diagnose_pid_bandwidth import linear_scores
from diagnose_fluctuations import fluctuation_metrics
from publication_stats import save, sha

PYTHON='/home/xc/software/jupyter/.venv/bin/python'


def extract(key):
    e=read(OUT/'protocol.json')['entries'][key]; root=PROJECT/e['path']; assert not (OUT/'extraction'/f'{key}.json').exists()
    p=read(OUT/'protocol.json'); edges=np.asarray(p['q_edges'])
    original=load(PARENT/'validation/arrays'/f'run{e["run"]}_{e["source"]}.npz')
    parent_ids=original['ids']; n=e['events']; pca=read(PROJECT/'analysis/paper_baseline/results/publication_20261028/point.json')['windows']['3000']['pca']
    file=ROOT.TFile.Open(str(root)); tree=file.Get('Events'); diag=file.Get('ResponseDiagnostics')
    info=json.loads(str(file.Get('RunInfo').GetString().Data()))
    assert info['completed'] and tree.GetEntries()==diag.GetEntries()==n and info['seed']==e['seed']
    assert info['config']==read(PROJECT/e['config'])
    tree.SetBranchStatus('*',0)
    for b in ['event_id','adc','edep_by_origin_MeV','avalanche_q_C','charge_C']:tree.SetBranchStatus(b,1)
    waves=np.empty((n,2496),np.uint16); energy=np.empty((n,6)); nav=np.empty(n,int); charge=np.empty(n)
    for i,event in enumerate(tree):
        assert event.event_id==i; waves[i]=np.asarray(event.adc); energy[i]=np.asarray(event.edep_by_origin_MeV)
        nav[i]=len(event.avalanche_q_C);charge[i]=float(event.charge_C)
    np.testing.assert_array_equal(energy,original['all_energy'])
    diagnostics={name:np.empty(n) for name in ['n_input_photons','n_primary','n_afterpulse','n_dark','n_crosstalk','n_recovery_below_099','mean_recovery']}
    for i,event in enumerate(diag):
        assert event.event_id==i
        for name in diagnostics:diagnostics[name][i]=getattr(event,name)
    counts=sum(diagnostics[k] for k in ['n_primary','n_afterpulse','n_dark','n_crosstalk'])
    np.testing.assert_array_equal(counts,nav)
    gain=info['config']['sipm']['sensor']['gain_electrons']
    np.testing.assert_allclose(diagnostics['mean_recovery']*nav*gain*1.602176634e-19,charge,rtol=1e-11,atol=1e-24)
    assert np.all(diagnostics['n_recovery_below_099']<=nav)
    assert np.all((diagnostics['mean_recovery']>0)&(diagnostics['mean_recovery']<=1+1e-14))
    f,quality=mc_wave_features(waves); selected=quality&(f['amplitude']>=200)&(f['t10_sample']>=384)
    common=selected&(f['charge_long']>=edges[0])&(f['charge_long']<=edges[-1])
    fixed=np.zeros(n,bool);fixed[parent_ids]=True;ids=np.flatnonzero(common|fixed)
    cf={k:v[ids] for k,v in f.items()}; anchor=np.floor(cf['t10_sample']).astype(int)+1
    assert np.all((anchor>=20)&(anchor+1500<=2496))
    scores=score_waves(waves[ids],pca,cf); fixed_anchor=linear_scores(waves[parent_ids],original['anchor'],f['baseline'][parent_ids],pca)[0]
    shapes,keep=aligned_shapes(waves[ids],cf);assert len(keep)==len(ids)
    metrics=fluctuation_metrics(waves[ids],cf['baseline'],cf['t10_sample'])[:,:,3]
    tree.SetBranchStatus('*',0);tree.SetBranchStatus('signal_mV',1)
    signal=[]
    ro=info['config']['sipm']['readout']; lsb=(ro['adc_max_mV']-ro['adc_min_mV'])/2**ro['adc_bits']
    for i in ids:
        tree.GetEntry(int(i));signal.append(np.array(tree.signal_mV)/lsb)
    file.Close()
    if e['variant']=='identity':
        assert info['response_replay']['equal_original_adc_events']==n
        assert info['response_replay']['equal_original_avalanche_events']==n
        np.testing.assert_array_equal(ids,original['ids']);np.testing.assert_array_equal(waves[ids],original['waves'])
        np.testing.assert_allclose(scores,original['scores'],rtol=1e-12,atol=1e-12)
    elif e['variant']=='no_signal_residual':
        assert info['response_replay']['equal_original_avalanche_events']==n
        np.testing.assert_array_equal(np.asarray(signal)[fixed[ids]],original['signal'])
    np.savez_compressed(OUT/'arrays'/f'{key}.npz',ids=ids,q=cf['charge_long'],scores=scores,waves=waves[ids],
        signal=np.asarray(signal),baseline=cf['baseline'],baseline_rms=cf['baseline_rms'],anchor=anchor,
        common=common[ids],fixed=fixed[ids],original_q=original['all_q'][ids],metrics=metrics,shapes=shapes,
        fixed_ids=parent_ids,fixed_anchor_scores=fixed_anchor,truth=label_truth(energy[ids])[0],
        all_q=f['charge_long'],all_selected=selected,all_common=common,all_energy=energy,n_parent=n,
        **{'diag_'+k:v for k,v in diagnostics.items()})
    record={'key':key,'n_parent':n,'n_common':int(common.sum()),'n_fixed':len(parent_ids),
        'gained':int(np.sum(common&~fixed)),'lost':int(np.sum(fixed&~common)),
        'code_sha256':sha(__file__),'root_sha256':sha(root),'config_sha256':sha(PROJECT/e['config']),
        'replay':info['response_replay'],'energy_identity':True,'diagnostic_charge_count_closure':True,
        'fixed_mean_recovery':float(np.mean(diagnostics['mean_recovery'][parent_ids])),
        'fixed_fraction_avalanches_recovery_below_099':float(diagnostics['n_recovery_below_099'][parent_ids].sum()/nav[parent_ids].sum()),
        'fixed_afterpulse_avalanche_fraction':float(diagnostics['n_afterpulse'][parent_ids].sum()/nav[parent_ids].sum())}
    save(OUT/'extraction'/f'{key}.json',record)
    print(key,'selected',record['n_common'],'AP fraction',record['fixed_afterpulse_avalanche_fraction'],flush=True)


def one(e):
    start=time.time();key=e['key'];steps=[]
    assert sha(PROJECT/e['config'])==e['config_sha256']
    root=PROJECT/e['path']
    jobs=[('replay',e['command'],root),('validate',[PYTHON,'-u','simulation/validate.py',str(root)],root.with_name(root.stem+'_validation.json')),
          ('extract',[PYTHON,'-u',__file__,'--extract',key],OUT/'extraction'/f'{key}.json')]
    for name,cmd,path in jobs:
        if path.exists():
            steps.append({'step':name,'existing':True});continue
        log=OUT/'logs'/f'{key}_{name}.log'
        with log.open('w') as f:r=subprocess.run(cmd,cwd=PROJECT,stdout=f,stderr=subprocess.STDOUT)
        if r.returncode:raise RuntimeError(f'{key} {name} failed: {log}')
        steps.append({'step':name,'command':cmd,'exit':r.returncode,'log':relative(log)})
        print(key,name,'completed',flush=True)
    return {'key':key,'steps':steps,'seconds':time.time()-start}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--extract');ap.add_argument('--workers',type=int,default=2);args=ap.parse_args()
    if args.extract:extract(args.extract);return
    assert not (OUT/'execution.json').exists()
    p=read(OUT/'protocol.json');assert sha(BUILD/'replay_response')==p['binary_sha256']
    for path in set(e['parent'] for e in p['entries'].values()):
        expected=next(e['parent_sha256'] for e in p['entries'].values() if e['parent']==path)
        assert sha(PROJECT/path)==expected
    os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(one,e) for e in p['entries'].values()]
        rows=[f.result() for f in concurrent.futures.as_completed(futures)]
    save(OUT/'execution.json',{'code_sha256':sha(__file__),'protocol_sha256':sha(OUT/'protocol.json'),
        'jobs':rows,'new_independent_mothers':0,'replay_entries':192000,'unique_reused_mothers':32000})


if __name__=='__main__':main()
