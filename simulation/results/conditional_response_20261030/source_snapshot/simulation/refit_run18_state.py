"""Separate run18 acquisition-state calibration with frozen internal checks.

Physical optical/SiPM parameters are shared with final Am241 configuration.
Only run18 first60 pulse data train its residual; other segments are checks.
"""
import copy
import json
from pathlib import Path
import shutil
import numpy as np
from reference import PROJECT, bin_layout, aligned_shapes, wave_features
from calibrate_signal_fluctuations import diagnostic, save
from diagnose_charge import sha
from diagnose_fluctuations import fluctuation_metrics
from fit_signal_fluctuations import fir_from_power, colored
from refit_final_parameters import arrays, positive_fit, combine_kernels
from refit_baseline_mixture import fit_mixture, derive_states, apply_states

OLD=PROJECT/'simulation/results/signal_fluctuations_20261028'
BASE=PROJECT/'simulation/results/waveform_refit_20261029'
OUT=BASE/'run18'


def main():
    OUT.mkdir(exist_ok=False)
    poolpath=PROJECT/'analysis/paper_baseline/results/publication_20261028/pool_run18.npz';pool=arrays(poolpath)
    fp=PROJECT/'analysis/paper_baseline/results/pha_run0018_features.npz';features=arrays(fp)
    rawpath=PROJECT/'data/DT5730_DPP_PHA/DAQ/run_18/UNFILTERED/Data_run_18.BIN'
    _,dtype,n=bin_layout(rawpath);raw=np.memmap(rawpath,dtype=dtype,offset=2,mode='r',shape=(n,))
    edges=np.array(json.loads((OLD/'design.json').read_text())['q_edges'])
    cfg=json.loads((BASE/'final_parameters/Am241_calibrated.json').read_text())
    h=np.array(cfg['electronics_noise']['coefficients']);r=cfg['sipm']['readout'];lsb=(r['adc_max_mV']-r['adc_min_mV'])/2**r['adc_bits']
    record={'scope':__doc__,'code_sha256':sha(Path(__file__)),'groups':{},'input_sha256':{str(poolpath.relative_to(PROJECT)):sha(poolpath),str(fp.relative_to(PROJECT)):sha(fp)},
      'training':'Only existing frozen3000 run18 first60 events. Middle/last frozen3000 each are excluded from pulse fitting; these segments were previously explored.',
      'baseline_conditioning':'Each segment state uses its own first128 pretrigger samples only. Later conditioned variants are not forecasts.',
      'raw_provenance':'Original BIN read-only; raw-file hashes inherited from earlier provenance, not recomputed.',
      'physical_config_parent_sha256':sha(BASE/'final_parameters/Am241_calibrated.json')}
    groups={};states={}
    for split,segment in enumerate(['first60','middle20','last20']):
        ids=pool['ids'][pool['primary']&(pool['split']==split)]
        waves=np.array(raw['data'][ids]);f={key:value[ids] for key,value in features.items() if value.ndim and len(value)==n}
        shapes,keep=aligned_shapes(waves,f);np.testing.assert_array_equal(keep,np.arange(len(ids)))
        metrics=fluctuation_metrics(waves,f['baseline'],f['t10_sample']);d=diagnostic(waves,f,edges)
        data={'ids':ids,'waves':waves,'q':f['charge_long'],'ratio':f['short_long_ratio'],'baseline':f['baseline'],
          't10':f['t10_sample'],'shapes':shapes,'metrics':metrics,'power':d['power'],'frequency_MHz':d['frequency_MHz']}
        np.savez_compressed(OUT/f'run18_{segment}.npz',**data);groups[segment]=data
        pre=waves[:,:128].astype(float);x=np.c_[pre.mean(1),np.log(np.maximum(pre.var(1)-127/(128*12),1e-6))]
        attempts=[fit_mixture(x,1)]+[fit_mixture(x,2,v) for v in [.5,.9,.97]]
        chosen=min([a for a in attempts if min(a['component_n'])>=20],key=lambda a:a['BIC'])
        state,extra=derive_states(chosen,h,lsb);states[segment]=state
        valid=(pool['split']==split)&pool['eligible_primary']&(pool['amplitude']>=200)&((pool['flags']&0x8000)==0)&(pool['t10_sample']>=384)
        record['groups'][segment]={'frozen_n':len(ids),'full_eligible_n':int(valid.sum()),'mixture_attempts':attempts,
          'mixture_selected':chosen,'states':state,**extra,'F':metrics[:,:,3].mean(0).tolist()}
    model=arrays(BASE/'parallel_shape/analysis/Am241_native.npz');data=groups['first60']
    adc=model['waves'].astype(float);baseline=model['baseline'];t10=model['t10'];signal=np.maximum(model['signal_mV']/lsb,0.)
    m=model['metrics'][:,:,3];target=data['metrics'][:,:,3].mean(0)
    excess=data['power'].mean(0)-model['power'].mean(0);h0=fir_from_power(data['frequency_MHz'],excess);h0/=np.sqrt(h0@h0)
    hl=np.exp(-np.arange(1025)*2/100);hl/=np.sqrt(hl@hl);kernels=[h0,hl,np.array([1.])]
    rng=np.random.default_rng(291029618);white=rng.normal(size=(8,len(adc),adc.shape[1]+1024));cols=[]
    for kernel in kernels:
        rows=[]
        for innovations in white:
            z=colored(kernel,innovations[:,:adc.shape[1]+len(kernel)-1])*np.sqrt(signal)
            pair=[fluctuation_metrics(adc+sign*z,baseline,t10)[:,:,3] for sign in [-1,1]]
            rows.append((pair[0]+pair[1])/2-m)
        cols.append(np.array(rows).mean(axis=(0,1)))
    A=np.array(cols).T;v=positive_fit(A/target[:,None],(target-m.mean(0))/target);kernel,strength=combine_kernels(kernels,v)
    perturb=np.array([colored(kernel,x)*np.sqrt(signal) for x in white]);added=[]
    for z in perturb:
        pair=[fluctuation_metrics(adc+sign*z,baseline,t10)[:,:,3] for sign in [-1,1]];added.append((pair[0]+pair[1])/2-m)
    delta=np.array(added).mean(axis=(0,1));strength=max(0.,float(np.dot(delta/target,(target-m.mean(0))/target)/np.dot(delta/target,delta/target)))
    values=[];squares=[]
    for z in perturb*np.sqrt(strength):
        pair=[fluctuation_metrics(adc+sign*z,baseline,t10)[:,:,3] for sign in [-1,1]];values.extend(pair);squares.append(((pair[0]+pair[1])/2-m)**2)
    values=np.array(values);var=values.reshape(-1,3).var(0,ddof=1);dv=data['metrics'][:,:,3].var(0,ddof=1)
    cv2=max(0.,float((dv[0]-var[0])/np.array(squares).mean(axis=(0,1))[0]));assert cv2<4
    noise=cfg['signal_dependent_noise'];noise.update(coefficients=kernel.tolist(),variance_per_signal_adc=strength,event_variance_cv=float(np.sqrt(cv2)),calibration_run=18,provenance=str((OUT/'fit.json').relative_to(PROJECT)))
    for segment in ['first60','middle20','last20']:
        candidate=apply_states(cfg,states[segment],f'run18_{segment}')
        candidate['electronics_noise']['state_provenance']['record']=str((OUT/'fit.json').relative_to(PROJECT))
        candidate['status']='Separate run18 acquisition calibration; shared physical Am241 optical/SiPM parameters; not run12-to18 transfer'
        save(OUT/f'Am241_run18_{segment}.json',candidate)
    record['residual']={'base_F':m.mean(0).tolist(),'target_F':target.tolist(),'component_variances':v.tolist(),
        'predicted_F':values.mean(axis=(0,1)).tolist(),'predicted_relative':(values.mean(axis=(0,1))/target-1).tolist(),
        'variance_per_signal_adc':strength,'event_variance_cv':float(np.sqrt(cv2)),
        'model_input_sha256':sha(BASE/'parallel_shape/analysis/Am241_native.npz')}
    save(OUT/'fit.json',record);print(json.dumps(record['residual'],indent=2),flush=True)


if __name__=='__main__':main()
