"""Estimate acquisition-state noise using pretrigger samples only.

Later-segment states are measured covariates, not predictions from training.
No pulse samples or measured noise traces are injected into native simulation.
"""
import copy
import json
from pathlib import Path
import numpy as np
from refit_waveform_shape import PROJECT, OLD, OUT, arrays, sha, save


def state(waves, h, lsb):
    n=128
    pre=waves[:,:n].astype(float)
    means=pre.mean(1);variances=pre.var(1)
    # Unit-FIR Gaussian covariance, including arbitrary available lag length.
    ac=np.correlate(h,h,'full')[len(h)-1:]
    lag=abs(np.arange(n)[:,None]-np.arange(n)[None,:])
    covariance=ac[lag]
    centering=np.eye(n)-np.ones((n,n))/n
    centered=centering@covariance@centering
    a=np.trace(centered)/n
    b=(np.trace(centered)**2+2*np.sum(centered**2))/n**2
    quantization_var=(1/12)*(n-1)/n
    mean_signal_var=max(float(variances.mean()-quantization_var),1e-8)
    sigma2=mean_signal_var/a
    # Quantization contributes <1% at these noise levels; retain this limitation.
    second=float(np.mean((variances-quantization_var)**2))
    cv2=max(0.,second/(sigma2*sigma2*b)-1.)
    jitter2=max(0.,float(means.var())-sigma2*covariance.mean()-1/(12*n))
    return {'baseline_mV':float((means.mean()+.5)*lsb),
            'noise_rms_mV':float(np.sqrt(sigma2)*lsb),
            'event_variance_cv':float(np.sqrt(cv2)),
            'baseline_jitter_mV':float(np.sqrt(jitter2)*lsb),
            'measured_baseline_ADC_quantiles':np.quantile(means,[.1,.5,.9]).tolist(),
            'measured_RMS_ADC_quantiles':np.quantile(np.sqrt(variances),[.1,.5,.9]).tolist(),
            'sample_n':len(pre),'sample_variance_factor':float(a),'sample_variance_second_moment':float(b)}


def apply_state(cfg,s):
    cfg=copy.deepcopy(cfg)
    for key in ('baseline_mV','noise_rms_mV'):cfg['sipm']['readout'][key]=s[key]
    for key in ('event_variance_cv','baseline_jitter_mV'):cfg['electronics_noise'][key]=s[key]
    return cfg


def main():
    root=OUT/'parallel_shape'
    report={'scope':__doc__,'code_sha256':sha(Path(__file__)),'states':{},'input_sha256':{},
      'model':'Stationary FIR with a mean-one lognormal event variance multiplier and independent Gaussian baseline offset.',
      'calibration':'First128 pretrigger samples only. Exact finite-window Gaussian covariance moments; approximate independent ADC quantization correction1/12.',
      'limitations':['No exact model of skewed baseline offsets, weak pulses, or baseline/RMS dependence.',
        'Later-segment states explicitly use their measured pretrigger samples and are conditional comparisons, not a forecast.']}
    for source,run in [('Co60',3),('Am241',12)]:
        cp=root/f'{source}_shape_candidate.json';cfg=json.loads(cp.read_text())
        h=np.array(cfg['electronics_noise']['coefficients']);r=cfg['sipm']['readout'];lsb=(r['adc_max_mV']-r['adc_min_mV'])/2**r['adc_bits']
        for segment in ('first60','middle20','last20'):
            path=OLD/f'run{run}_{segment}.npz';a=arrays(path)
            s=state(a['waves'],h,lsb);key=f'run{run}_{segment}'
            s['role']='training' if segment=='first60' else 'measured pretrigger covariate; not blind/predictive'
            report['states'][key]=s;report['input_sha256'][str(path.relative_to(PROJECT))]=sha(path)
            print(key,s,flush=True)
        cfg=apply_state(cfg,report['states'][f'run{run}_first60'])
        cfg['evidence']['baseline_state_calibration']={'record':str((OUT/'baseline_states.json').relative_to(PROJECT)),
              'state':f'run{run}_first60','samples':'first128 pretrigger samples only'}
        # Preserve expected 3us integral when changing timing parameters. These
        # are shared emission-yield parameters, never per-event ADC rescaling.
        from refit_parallel_shape import convolved_exponentials
        def long_fraction(config,kind):
            o=config['optical'];r=config['sipm']['readout'];pr=config.get('readout_response',{}).get('prompt_charge_fraction',0.)
            curves=[]
            for tau in o[kind+'_tau_ns']:
                slow=convolved_exponentials([tau,2.,29.,r['lowpass_ns']])
                fast=convolved_exponentials([tau,2.,5.])
                curves.append((1-pr)*slow+pr*fast)
            fraction=o[kind+'_fast_fraction'];y=fraction*curves[0]+(1-fraction)*curves[1]
            peak=int(y.argmax());lo=int(np.flatnonzero(y[:peak]<=.1*y[peak])[-1]);anchor=lo+1
            return y[max(0,anchor-40):anchor+3000].sum()
        old=json.loads((OLD/'independent'/f'{source}_calibrated.json').read_text())
        se=long_fraction(old,'electron')/long_fraction(cfg,'electron')
        sa=long_fraction(old,'alpha')/long_fraction(cfg,'alpha')
        cfg['optical']['yield_photons_per_MeV']=old['optical']['yield_photons_per_MeV']*se
        cfg['optical']['alpha_light_yield_ratio']=old['optical']['alpha_light_yield_ratio']*sa/se
        cfg['evidence']['yield_window_adjustment']={'electron_scale':float(se),'alpha_scale':float(sa),
            'scope':'Preserve earlier expected3us charge when emission time parameters change; not a new spectral fit or intrinsic light-yield measurement'}
        save(root/f'{source}_native_candidate.json',cfg)
    save(OUT/'baseline_states.json',report)


if __name__=='__main__':main()
