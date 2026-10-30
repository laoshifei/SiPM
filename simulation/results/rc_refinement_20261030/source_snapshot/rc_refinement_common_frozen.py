"""Fixed-cohort utilities for continuous RC and joint noise-distribution checks."""
import copy
import json
import time
import subprocess
from pathlib import Path
import numpy as np
import ROOT
from state_response_common import (PROJECT,BUILD,SOURCES,SEGMENTS,PCA,features,measured,moments,F_metrics,
    read,load,relative,serialize,event_weights,rows,weighted_quantile,save,sha)

PARENT=PROJECT/'simulation/results/state_response_20261030'
OUT=PROJECT/'simulation/results/rc_refinement_20261030'
FAMILIES=['inherited','spectrum']
CV_GRID=[0.,.25,.5,1.,1.5,2.]


def summary(a,p,source,counts=None):
    m=moments(a,p,source,counts)
    if m is None:return None
    vs=[]
    for _,ix,w,_ in rows(a,p,source,counts):
        x=a['F'][ix];mu=w@x;vs.append(w@((x-mu)**2))
    return {**m,'F_sd':np.sqrt(np.mean(vs,axis=0))}


def mean_loss(m,d):
    return float(np.mean((m['wave']-d['wave'])**2)/(.03*max(d['wave']))**2+
        ((m['mean'][1]-d['mean'][1])/.003)**2+((m['mean'][0]-d['mean'][0])/.003)**2+
        ((m['tail']-d['tail'])/.005)**2)


def joint_loss(m,d):
    i,j=np.triu_indices(3);normal=np.sqrt(np.diag(d['cov'])[i]*np.diag(d['cov'])[j])
    parts={'mean':mean_loss(m,d),'covariance':float(np.sum(((m['cov']-d['cov'])[i,j]/normal/.25)**2)),
        'F_mean':float(np.sum(((m['F']/d['F']-1)/.10)**2)),
        'PID_IQR':float(np.sum(((m['iqr']/d['iqr']-1)/.10)**2)),
        'F_scatter':float(np.sum(((m['F_sd']/d['F_sd']-1)/.20)**2))}
    return sum(parts.values()),parts


def job(key,cfg):
    cp=OUT/'configs'/f'{key}.json'
    if cp.exists():assert read(cp)==cfg
    else:save(cp,cfg)
    return {'key':key,'config':str(cp),'output':str(OUT/'native'/f'{key}.root'),
        'phase_ns':0.,'require_identity':False,'config_sha256':sha(cp)}


def execute(source,jobs,stage):
    p=read(OUT/'protocol.json');entry=p['native'][source]
    plan=OUT/'configs'/f'jobs_{stage}_{source}.json';content={'ids':entry['selected_ids'],'jobs':jobs}
    if plan.exists():assert read(plan)==content
    else:save(plan,content)
    logpath=OUT/'logs'/f'{stage}_{source}.json'
    if logpath.exists():
        log=read(logpath)
        for j in jobs:
            r=log['jobs'][j['key']]
            assert sha(j['config'])==r['config_sha256'] and sha(j['output'])==r['root_sha256']
            assert sha(OUT/'arrays'/f"{j['key']}.npz")==r['array_sha256']
        return
    start=time.time();cmd=[str(BUILD/'replay_readout'),str(PROJECT/entry['path']),str(plan)]
    existing=[Path(j['output']).exists() for j in jobs]
    if not all(existing):
        assert not any(existing),'Partial native batch requires explicit audit'
        with (OUT/'logs'/f'{stage}_{source}.log').open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,cwd=PROJECT)
        if r.returncode:raise RuntimeError(f'Native batch failed: {stage} {source}')
    original=load(PARENT/'arrays'/f"run{3 if source=='Co60' else 12}_phase00.npz");records={}
    for j in jobs:
        f=ROOT.TFile.Open(j['output']);assert f and not f.IsZombie()
        info=json.loads(str(f.Get('RunInfo').GetString().Data()));tree=f.Get('Readout')
        assert info['completed'] and info['clipped_samples']==0 and info['new_independent_mothers']==0
        assert info['config']==read(j['config']) and info['phase_ns']==0.
        tree.SetBranchStatus('*',0)
        for name in ['event_id','adc','signal_mV']:tree.SetBranchStatus(name,1)
        ids=[];waves=[];signal=[]
        for row in tree:ids.append(int(row.event_id));waves.append(np.array(row.adc));signal.append(np.array(row.signal_mV))
        f.Close();np.testing.assert_array_equal(ids,original['ids'])
        ro=info['config']['sipm']['readout'];lsb=(ro['adc_max_mV']-ro['adc_min_mV'])/2**ro['adc_bits']
        z=features(np.asarray(waves,np.uint16),original['ids'],original['q'],np.asarray(signal)/lsb);z['n_parent']=original['n_parent']
        ap=OUT/'arrays'/f"{j['key']}.npz";np.savez_compressed(ap,**z)
        records[j['key']]={'root':relative(Path(j['output'])),'root_sha256':sha(j['output']),
            'array_sha256':sha(ap),'config_sha256':j['config_sha256'],'n_original_mothers':len(ids),'new_independent_mothers':0}
    save(logpath,{'command':cmd,'seconds':time.time()-start,'binary_sha256':sha(BUILD/'replay_readout'),
        'plan_sha256':sha(plan),'jobs':records})
    print(stage,source,len(jobs),'native jobs completed',flush=True)


def as_arrays(m):return {k:np.asarray(v) if isinstance(v,list) else v for k,v in m.items()}


def prepare():
    OUT.mkdir(exist_ok=False)
    for name in ['arrays','configs','native','logs','fits','figures','source_snapshot']:(OUT/name).mkdir()
    p=read(PARENT/'protocol.json');parent=read(PARENT/'record.json');cal=read(PARENT/'mean_calibration.json')
    needed=[PARENT/n for n in ['record.json','protocol.json','decision.json','evaluation.json','mean_calibration.json','signal_calibration.json']]
    entries={};grid={}
    for key,e in cal['groups'].items():
        for name in [f'fits/{key}.json',f'fits/{key}_signal.json',f'configs/{key}_electronics.json',f'configs/{key}_response_spectrum.json',f'configs/{key}_response_inherited.json']:
            needed.append(PARENT/name)
        entries[key]=e
    for source,run in SOURCES:
        for seg in SEGMENTS:
            for B in p['block_sizes']:needed.append(PARENT/'arrays'/f'run{run}_{seg}_alignment_B{B}.npz')
        needed.append(PARENT/'arrays'/f'run{run}_phase00.npz')
        grid[source]=[]
        for rc in sorted(set(np.arange(96.,160.01,2.).tolist()+p['RC_grid_ns'])):
            if rc in p['RC_grid_ns']:
                path=PARENT/'arrays'/f"run{run}_quietRC{p['RC_grid_ns'].index(rc)}.npz";needed.append(path);reuse=True
            else:path=OUT/'arrays'/f'run{run}_grid{int(rc):03d}.npz';reuse=False
            grid[source].append({'RC_ns':rc,'array':relative(path),'reused':reuse})
    for f in needed:
        if relative(f) in parent['outputs_sha256']:assert sha(f)==parent['outputs_sha256'][relative(f)],f
    for file,value in p['inputs_sha256'].items():assert sha(PROJECT/file)==value,file
    protocol={'scope':__doc__,'code_sha256':sha(__file__),'parent_record_sha256':sha(PARENT/'record.json'),
        'inputs_sha256':{relative(f):sha(f) for f in needed},'inherited_inputs_sha256':p['inputs_sha256'],
        'native':p['native'],'groups':p['groups'],'entries':entries,'q_edges':p['q_edges'],'support_bins':p['support_bins'],
        'conditional_cuts':p['conditional_cuts'],'block_sizes':p['block_sizes'],'grid':grid,
        'RC_bounds_ns':[96.,160.],'transmission_bounds':[.45,.55],
        'mean_fit':'2ns native quiet grid plus original128.21494566 point; PCHIP interpolation of aggregate summaries only. Deterministic0.25ns global search then bounded local minimization. Same mean-ADC/time/ratio/tail objective as parent. Two fixed native noisy-bias correction rounds; choose lowest training loss among rounds0,1,2. Never interpolate generated ADC.',
        'native_bias_correction':'At each chosen RC/transmission, noisy native minus quiet-summary prediction provides a local additive summary offset. Refit RC/transmission inside unchanged bounds on opposite experimental blocks. Parent pretrigger fit, optical physics and initial signal-noise coefficients/CV fixed during this stage.',
        'noise_fit':'Inherited spectrum or positive inherited+400/1500ns exponential spectra; NNLS full untrimmed within-Q covariance and F means. Then native CV grid includes old value and0,.25,.5,1,1.5,2. CV changes mean-one lognormal signal-variance multiplier only, not physical gain or scintillation.',
        'selection_loss':'Mean-response loss + six upper-triangle covariance residuals/(data SD products*.25)^2 + three F-mean relative residuals/.10 squared + three PID-IQR relative residuals/.10 squared + three within-Q F-SD relative residuals/.20 squared. Select each family CV using training blocks only. Retain inherited-CV and selected-CV controls.',
        'CV_grid':CV_GRID,'F_proposal_replicas':16,'training_stability_draws':128,'fixed_fit_interval_draws':512,
        'selection_scope':'Only new calibration layer cross-fitted; all experimental segments explored. All original MC mothers reused for calibration/internal checks; no independent simulation validation claimed.',
        'uncertainty':'512 paired original-index blocks and whole-mother draws conditional on fitted parameters. Separate128 training-block refits of continuous RC, NNLS powers and CV-grid selection; not full nested fit/model uncertainty.',
        'fixed_choices':'Original experimental/main-band IDs/Q/PCA, first60 Q2 targets, original mother fold assignments, physics, pretrigger fits, block128 primary and64 sensitivity. No post-fit event selection.',
        'candidate_boundary':'Do not promote a candidate based on one width. A stable joint candidate would require frozen parameters and fresh independent Geant4 mothers; existing experimental checks stay internal.',
        'new_independent_mothers':0,'new_experimental_events':0,'default_changed':False}
    save(OUT/'protocol.json',serialize(protocol))
    for source,run in SOURCES:
        base=read(PROJECT/p['native'][source]['config']);jobs=[]
        for r in grid[source]:
            if r['reused']:continue
            c=copy.deepcopy(base);c.pop('electronics_noise',None);c.pop('signal_dependent_noise',None)
            c['sipm']['readout'].update(lowpass_ns=r['RC_ns'],noise_rms_mV=0.)
            jobs.append(job(f"run{run}_grid{int(r['RC_ns']):03d}",c))
        execute(source,jobs,'quiet_grid')


if __name__=='__main__':prepare()
