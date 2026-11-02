"""Shared frozen-cohort utilities for state-conditioned native response checks."""
import copy
import json
import subprocess
import time
from pathlib import Path
import numpy as np
import ROOT
from prepare_state_response import OUT, ACQ, BUILD, DATA, WIDTH, PARENT, PROJECT, SOURCES, SEGMENTS, read, load, relative
from prepare_acquisition_state import score_gate, finite
from analyze_acquisition_state import event_weights, rows, weighted_quantile
from reference import wave_features
from publication_stats import save, sha

PCA=read(PROJECT/'analysis/paper_baseline/results/publication_20261028/point.json')['windows']['3000']['pca']


def F_metrics(waves, baseline, t10):
    """Vectorized existing 50/102/202 ns diagnostic, including exact boundaries."""
    y=waves.astype(float)-baseline[:,None];cum=np.c_[np.zeros(len(y)),y.cumsum(1)]
    pre=np.ceil(t10-300).astype(int)[:,None]+np.arange(200)
    post=np.ceil(t10+250).astype(int)[:,None]+np.arange(1000)
    ix=np.arange(len(y))[:,None];result=[]
    for width in [25,51,101]:
        half=width//2
        assert pre.min()>=half and post.max()<y.shape[1]-half
        residual=y[:,half:-half]-(cum[:,width:]-cum[:,:-width])/width
        mu=y[ix,post].mean(1);vp=residual[ix,pre-half].var(1,ddof=1);vs=residual[ix,post-half].var(1,ddof=1)
        result.append(np.c_[mu,vp,vs,np.divide(vs-vp,mu,out=np.full(len(mu),np.nan),where=mu>0)])
    return np.stack(result,axis=1)


def features(waves,ids,q,signal=None):
    f=wave_features(waves);anchor=np.floor(f['t10_sample']).astype(int)+1
    assert np.all(np.isfinite(f['t10_sample'])) and np.all(anchor>=384)
    scores,qcalc,gate=score_gate(waves,anchor,f['baseline'],PCA)
    a=dict(ids=ids,q=q,waves=waves,anchor=anchor,t10=f['t10_sample'],baseline=f['baseline'],
        baseline_rms=f['baseline_rms'],scores=scores,qcalc=qcalc,F=F_metrics(waves,f['baseline'],f['t10_sample'])[:,:,3],
        mean_wave=gate.reshape(len(waves),76,20).mean(2),shape=gate.reshape(len(waves),76,20).sum(2)/qcalc[:,None])
    if signal is not None:a['signal']=signal
    return a


def measured(run,seg):
    raw=load(DATA/f'run{run}_{seg}.npz');old=load(ACQ/'arrays'/f'run{run}_{seg}.npz')
    use=load(WIDTH/'arrays'/f'measured_run{run}_{seg}.npz')['main_band']
    np.testing.assert_array_equal(raw['ids'][use],old['ids'])
    a=features(raw['waves'][use],old['ids'],old['q'])
    np.testing.assert_allclose(a['scores'],old['scores'][:,0],rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(a['F'],old['F'],rtol=1e-12,atol=1e-12)
    a.update(state=old['state'],span=old['span'],n_raw=old['n_raw'])
    return a,raw


def moments(a,p,source,counts=None):
    rr=rows(a,p,source,counts)
    if rr is None:return None
    means=[];covs=[];iqrs=[];Fs=[];waves=[];tails=[]
    for _,ix,w,_ in rr:
        x=a['scores'][ix];mu=w@x;r=x-mu
        means.append(mu);covs.append(r.T@(w[:,None]*r))
        iqrs.append([np.diff(weighted_quantile(x[:,j],w))[0] for j in range(3)])
        Fs.append(w@a['F'][ix]);waves.append(w@a['mean_wave'][ix]);tails.append(w@a['shape'][ix,51:].sum(1))
    return {'mean':np.mean(means,0),'cov':np.mean(covs,0),'iqr':np.sqrt(np.mean(np.array(iqrs)**2,0)),
        'F':np.mean(Fs,0),'wave':np.mean(waves,0),'tail':float(np.mean(tails))}


def serialize(v):
    if isinstance(v,np.ndarray):return serialize(v.tolist())
    if isinstance(v,dict):return {k:serialize(x) for k,x in v.items()}
    if isinstance(v,(tuple,list)):return [serialize(x) for x in v]
    return finite(v)


def create_job(key,cfg,phase=0.,identity=False):
    cp=OUT/'configs'/f'{key}.json'
    if cp.exists():assert read(cp)==cfg, 'Existing job configuration differs'
    else:save(cp,cfg)
    return {'key':key,'config':str(cp),'output':str(OUT/'native'/f'{key}.root'),
        'phase_ns':phase,'require_identity':identity,'config_sha256':sha(cp)}


def execute_jobs(source,jobs,stage):
    p=read(OUT/'protocol.json');entry=p['native'][source]
    plan=OUT/'configs'/f'jobs_{stage}_{source}.json'
    content={'ids':entry['selected_ids'],'jobs':jobs}
    if plan.exists():assert read(plan)==content
    else:save(plan,content)
    start=time.time();log=OUT/'logs'/f'{stage}_{source}.log'
    command=[str(BUILD/'replay_readout'),str(PROJECT/entry['path']),str(plan)]
    existing=[Path(j['output']).exists() for j in jobs]
    if not all(existing):
        assert not any(existing), 'Partially written native batch requires explicit audit'
        with log.open('w') as f:r=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,cwd=PROJECT)
        if r.returncode:raise RuntimeError(f'Native replay failed: {log}')
    original=load(PARENT/'arrays'/f"run{3 if source=='Co60' else 12}_{source}__identity.npz")
    q=original['original_q'][original['fixed']];ids=original['ids'][original['fixed']]
    records={}
    for job in jobs:
        f=ROOT.TFile.Open(job['output']);assert f and not f.IsZombie()
        info=json.loads(str(f.Get('RunInfo').GetString().Data()));tree=f.Get('Readout')
        got=[];waves=[];signal=[]
        tree.SetBranchStatus('*',0)
        for name in ['event_id','adc','signal_mV']:tree.SetBranchStatus(name,1)
        for row in tree:
            got.append(int(row.event_id));waves.append(np.array(row.adc));signal.append(np.array(row.signal_mV))
        np.testing.assert_array_equal(got,ids);waves=np.asarray(waves,dtype=np.uint16)
        ro=info['config']['sipm']['readout'];lsb=(ro['adc_max_mV']-ro['adc_min_mV'])/2**ro['adc_bits']
        signal=np.asarray(signal)/lsb;f.Close()
        assert info['config']==read(job['config']) and info['phase_ns']==job['phase_ns']
        assert info['completed'] and info['clipped_samples']==0 and info['new_independent_mothers']==0
        a=features(waves,ids,q,signal);a['n_parent']=np.array(entry['events'])
        if job['require_identity']:
            np.testing.assert_array_equal(waves,original['waves'][original['fixed']])
            assert info['equal_parent_ADC']==len(ids)
        np.savez_compressed(OUT/'arrays'/f"{job['key']}.npz",**a)
        records[job['key']]={'root':relative(Path(job['output'])),'root_sha256':sha(job['output']),
            'array_sha256':sha(OUT/'arrays'/f"{job['key']}.npz"),'config_sha256':job['config_sha256'],
            'phase_ns':job['phase_ns'],'equal_parent_ADC':info['equal_parent_ADC'],
            'n_selected_original_mothers':len(ids),'new_independent_mothers':0}
    save(OUT/'logs'/f'{stage}_{source}.json',{'command':command,'seconds':time.time()-start,'jobs':records,
        'binary_sha256':sha(BUILD/'replay_readout'),'plan_sha256':sha(plan)})
    print(stage,source,'completed',len(jobs),'jobs on',len(ids),'fixed original mothers',flush=True)


def initial():
    p=read(OUT/'protocol.json')
    # Freeze numeric loss scales before any response-grid result is examined.
    design=OUT/'fit_design.json'
    design_content={'protocol_sha256':sha(OUT/'protocol.json'),'code_sha256':sha(__file__),
        'mean_loss':'MSE of76 absolute ADC gate means /(0.03*training mean peak)^2 + mean-time error^2/(3ns)^2 + short/long mean error^2/(0.003)^2 + charge fraction[2000,3000) error^2/(0.005)^2. Quiet RC-grid proposal with bounded analytic global transmission; validate noisy native own-anchor ADC.',
        'noise_loss':'Full upper triangular untrimmed Q2-weighted within-Q PID covariance normalized by target diagonal SD products and0.25; three F means normalized by target F and0.10. Positive powers by NNLS. No winsorization or removal of covariance terms.',
        'F_proposal_replicas':16,'fit_stability_draws':128,'fixed_fit_interval_draws':512,
        'front_lag_step_ns':.25,'front_fit_bounds_ns':[-10.,10.],
        'selection':'No final candidate promotion if joint waveform/center/covariance/F discrepancies remain; retain every tested variant. No expansion of parameter bounds after outcomes.'}
    if design.exists():
        old=read(design);assert {k:v for k,v in old.items() if k!='code_sha256'}=={k:v for k,v in design_content.items() if k!='code_sha256'}
    else:save(design,design_content)
    for source,run in SOURCES:
        base=read(PROJECT/p['native'][source]['config']);jobs=[]
        for phase in p['phase_offsets_ns']:
            jobs.append(create_job(f'run{run}_phase{int(phase*10):02d}',base,phase,phase==0))
        for j,rc in enumerate(p['RC_grid_ns']):
            c=copy.deepcopy(base);c.pop('electronics_noise',None);c.pop('signal_dependent_noise',None)
            c['sipm']['readout'].update(lowpass_ns=rc,noise_rms_mV=0.)
            jobs.append(create_job(f'run{run}_quietRC{j}',c))
        execute_jobs(source,jobs,'initial')


if __name__=='__main__':initial()
