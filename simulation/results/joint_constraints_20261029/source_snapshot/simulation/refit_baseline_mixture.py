"""Pretrigger-only joint baseline/logvariance calibration with 1-2 Gaussian states.

The mixture describes acquisition-state heterogeneity, not scintillation or
SiPM gain. Later-segment states are measured conditions, never blind forecasts.
"""
import copy
import json
from pathlib import Path
import numpy as np
from scipy.special import logsumexp, digamma, polygamma
from refit_waveform_shape import PROJECT, OLD, OUT, arrays, sha, save


def fit_mixture(x,k,split=.5):
    n,d=x.shape;center=x.mean(0);scale=x.std(0);z=(x-center)/scale
    if k==1:responsibility=np.ones((n,1))
    else:
        threshold=np.quantile(z[:,0],split)
        responsibility=np.c_[z[:,0]<=threshold,z[:,0]>threshold].astype(float)*.98+.01
    last=-np.inf
    for iteration in range(500):
        count=responsibility.sum(0);w=count/n;means=responsibility.T@z/count[:,None]
        covariances=[]
        for j in range(k):
            delta=z-means[j]
            covariances.append((delta*responsibility[:,j,None]).T@delta/count[j]+np.eye(d)*1e-5)
        covariances=np.array(covariances);lp=[]
        for j in range(k):
            delta=z-means[j];inv=np.linalg.inv(covariances[j]);_,det=np.linalg.slogdet(covariances[j])
            lp.append(np.log(w[j])-.5*(d*np.log(2*np.pi)+det+np.sum((delta@inv)*delta,axis=1)))
        lp=np.array(lp).T;total=logsumexp(lp,axis=1);ll=float(total.sum())
        responsibility=np.exp(lp-total[:,None])
        if abs(ll-last)<1e-7*n:break
        last=ll
    return {'weights':w.tolist(),'means':(means*scale+center).tolist(),
       'covariances':(covariances*scale[None,:,None]*scale[None,None,:]).tolist(),
       'loglikelihood_standardized':ll,'BIC':float(-2*ll+(6*k-1)*np.log(n)),
       'component_n':count.tolist(),'iterations':iteration+1,'components':k,'initial_split':split}


def derive_states(fit,h,lsb):
    n=128;ac=np.correlate(h,h,'full')[len(h)-1:]
    C=ac[abs(np.arange(n)[:,None]-np.arange(n)[None,:])]
    P=np.eye(n)-np.ones((n,n))/n;PCP=P@C@P
    a=np.trace(PCP)/n;varW=2*np.sum(PCP**2)/n**2
    nu=2*a*a/varW
    logWmean=float(np.log(a)+digamma(nu/2)-np.log(nu/2));logWvar=float(polygamma(1,nu/2))
    states=[];diagnostics=[]
    for w,mean,covariance in zip(fit['weights'],fit['means'],fit['covariances']):
        covariance=np.array(covariance)
        logvariance=max(float(covariance[1,1]-logWvar),1e-6)
        mean_log_sigma2=mean[1]-logWmean
        sigma2=float(np.exp(mean_log_sigma2+logvariance/2))
        baseline_var=max(float(covariance[0,0]-sigma2*C.mean()-1/(12*n)),1e-6)
        rawrho=float(covariance[0,1]/np.sqrt(baseline_var*logvariance));rho=float(np.clip(rawrho,-.99,.99))
        cv=float(np.sqrt(np.expm1(logvariance)));assert cv<=2
        states.append({'weight':w,'baseline_mV':float((mean[0]+.5)*lsb),
          'noise_rms_mV':float(np.sqrt(sigma2)*lsb),'baseline_jitter_mV':float(np.sqrt(baseline_var)*lsb),
          'event_variance_cv':cv,'baseline_logvariance_correlation':rho})
        diagnostics.append({'unclipped_correlation':rawrho,'latent_logvariance':logvariance,'latent_baseline_variance_ADC2':baseline_var})
    return states,{'effective_sample_variance_dof':float(nu),'log_sample_variance_mean':logWmean,
                   'log_sample_variance_variance':logWvar,'state_diagnostics':diagnostics}


def apply_states(config,states,label):
    config=copy.deepcopy(config)
    config['electronics_noise']['states']=states
    config['electronics_noise']['state_provenance']={'record':str((OUT/'baseline_mixture_fit.json').relative_to(PROJECT)),
        'group':label,'input_region':'first128 pretrigger samples only',
        'role':'training calibration' if label.endswith('first60') else 'measured pretrigger condition; not a prediction'}
    config['sipm']['readout']['baseline_mV']=sum(s['weight']*s['baseline_mV'] for s in states)
    config['sipm']['readout']['noise_rms_mV']=float(np.sqrt(sum(s['weight']*s['noise_rms_mV']**2 for s in states)))
    config['status']='2026-10-29 optical-time and stochastic readout calibration; pretrigger-state mixture explicitly conditioned by acquisition segment'
    return config


def main():
    destination=OUT/'calibrated_mixture';destination.mkdir(exist_ok=False)
    report={'scope':__doc__,'code_sha256':sha(Path(__file__)),'groups':{},'inputs_sha256':{},
      'reason':'Single Gaussian baseline offset over-broadened the run12 core because sparse shifted-state observations dominate variance. Joint baseline/RMS dependence is measured in pretrigger data.',
      'selection':'Compare1 and2 Gaussian components by BIC in (pretrigger mean,log pretrigger variance). Require>=20 effective observations per component.',
      'deconvolution':'Subtract1/12 quantization variance; approximate finite-window Gaussian sample-variance log moments by effective chi-square degrees of freedom from exact FIR covariance.',
      'limits':['Phenomenological pretrigger-state calibration, not an identified circuit mechanism.',
        'Approximate log-variance sampling correction and Gaussian-mixture family; pulse-like baseline transients not synthesized.',
        'Later states use their own pretrigger samples only and are not forecasts. Pulse residual fit remains run3/12 first60.']}
    for source,run in [('Co60',3),('Am241',12)]:
        path=OUT/'calibrated'/f'{source}_calibrated.json';cfg=json.loads(path.read_text())
        r=cfg['sipm']['readout'];lsb=(r['adc_max_mV']-r['adc_min_mV'])/2**r['adc_bits']
        h=np.array(cfg['electronics_noise']['coefficients'])
        for segment in ('first60','middle20','last20'):
            data_path=OLD/f'run{run}_{segment}.npz';data=arrays(data_path)
            pre=data['waves'][:,:128].astype(float)
            x=np.c_[pre.mean(1),np.log(np.maximum(pre.var(1)-(1/12)*127/128,1e-6))]
            attempts=[fit_mixture(x,1)]+[fit_mixture(x,2,split) for split in (.5,.9,.97)]
            valid=[a for a in attempts if min(a['component_n'])>=20]
            best=min(valid,key=lambda a:a['BIC']);states,diagnostic=derive_states(best,h,lsb)
            label=f'run{run}_{segment}'
            report['groups'][label]={'attempts':attempts,'selected':best,'native_states':states,**diagnostic}
            report['inputs_sha256'][str(data_path.relative_to(PROJECT))]=sha(data_path)
            candidate=apply_states(cfg,states,label)
            target=destination if segment=='first60' else destination/segment
            target.mkdir(exist_ok=True)
            save(target/f'{source}_calibrated.json',candidate)
            print(label,'components',best['components'],'weights',best['weights'],'states',states,flush=True)
    save(OUT/'baseline_mixture_fit.json',report)


if __name__=='__main__':main()
