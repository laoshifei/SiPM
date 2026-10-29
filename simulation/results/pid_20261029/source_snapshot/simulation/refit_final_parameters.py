"""One final training-only closure correction before new-seed full transport.

Adjust particle fast fractions for training median integration ratios and
recalibrate positive residual spectral components for the three F means.
No later measured pulse or new-seed validation output enters this operation.
"""
import copy
import itertools
import json
from pathlib import Path
import numpy as np
from reference import PROJECT
from diagnose_charge import sha
from diagnose_fluctuations import fluctuation_metrics
from fit_signal_fluctuations import colored
from calibrate_signal_fluctuations import save

OLD=PROJECT/'simulation/results/signal_fluctuations_20261028'
OUT=PROJECT/'simulation/results/waveform_refit_20261029'


def arrays(path):
    with np.load(path) as z:return dict(z)


def kernel_masses(taus):
    edges=np.arange(0.,10001.)
    taus=np.array(taus)
    c=np.array([v**(len(taus)-1)/np.prod(v-np.delete(taus,i)) for i,v in enumerate(taus)])
    return c@(-np.diff(np.exp(-edges[None,:]/taus[:,None]),axis=1))


def pulse_summary(config,kind,fraction=None):
    o=config['optical'];r=config['sipm']['readout'];prompt=config['readout_response']['prompt_charge_fraction']
    if fraction is None:fraction=o[kind+'_fast_fraction']
    ys=[]
    for tau in o[kind+'_tau_ns']:
        ys.append((1-prompt)*kernel_masses([tau,2.,29.,r['lowpass_ns']])+prompt*kernel_masses([tau,2.,5.]))
    y=fraction*ys[0]+(1-fraction)*ys[1]
    peak=int(y.argmax());i=int(np.flatnonzero(y[:peak]<=.1*y[peak])[-1]);t10=i+(.1*y[peak]-y[i])/(y[i+1]-y[i])
    # Interpolated integrals avoid one-sample anchor jumps during scalar solve.
    cumulative=np.r_[0.,np.cumsum(y)];x=np.arange(len(cumulative))
    lo=max(0.,t10-40);q=np.interp(t10+3000,x,cumulative)-np.interp(lo,x,cumulative)
    short=np.interp(t10+500,x,cumulative)-np.interp(lo,x,cumulative)
    return short/q,q


def positive_fit(matrix,target):
    best=None
    for count in range(1,matrix.shape[1]+1):
        for subset in itertools.combinations(range(matrix.shape[1]),count):
            values=np.linalg.lstsq(matrix[:,subset],target,rcond=None)[0]
            if (values<0).any():continue
            x=np.zeros(matrix.shape[1]);x[list(subset)]=values
            loss=float(np.sum((matrix@x-target)**2))
            if best is None or loss<best[0]:best=(loss,x)
    assert best is not None
    return best[1]


def combine_kernels(kernels,variances):
    fft_n=4096
    power=sum(v*np.abs(np.fft.rfft(h,n=fft_n))**2 for h,v in zip(kernels,variances))
    impulse=np.fft.fftshift(np.fft.irfft(np.sqrt(power),n=fft_n))
    # Longer support retains the low-frequency correction instead of tapering
    # it away. The native module already supports up to1025 taps.
    impulse=impulse[fft_n//2-512:fft_n//2+513]*np.hanning(1025)
    variance=float(np.dot(impulse,impulse));return impulse/np.sqrt(variance),variance


def main():
    output=OUT/'final_parameters';output.mkdir(exist_ok=False)
    native=json.loads((OUT/'calibrated_mixture/analysis/validation.json').read_text())
    base=json.loads((OUT/'parallel_shape/analysis/validation.json').read_text())
    configs={s:json.loads((OUT/'calibrated_mixture'/f'{s}_calibrated.json').read_text()) for s in ['Co60','Am241']}
    fractions={};corrections={}
    for source,run,kind in [('Co60',3,'electron'),('Am241',12,'alpha')]:
        data=arrays(OLD/f'run{run}_first60.npz');model=arrays(OUT/'calibrated_mixture/analysis'/f'{source}_native.npz')
        cfg=configs[source];old_fraction=cfg['optical'][kind+'_fast_fraction']
        delta=float(np.median(data['ratio'])-np.median(model['ratio']))
        old_ratio,_=pulse_summary(cfg,kind);target=old_ratio+delta
        low=max(.005,old_fraction-.08);high=min(.995,old_fraction+.08)
        for _ in range(60):
            mid=(low+high)/2
            if pulse_summary(cfg,kind,mid)[0]<target:low=mid
            else:high=mid
        fractions[kind]=(low+high)/2
        corrections[source]={'training_median_ratio_difference':delta,'old_fraction':old_fraction,
          'new_fraction':fractions[kind],'old_analytic_ratio':old_ratio,'target_analytic_ratio':target}
    report={'scope':__doc__,'code_sha256':sha(Path(__file__)),
      'training_validation_sha256':sha(OUT/'calibrated_mixture/analysis/validation.json'),
      'base_validation_sha256':sha(OUT/'parallel_shape/analysis/validation.json'),
      'ratio_corrections':corrections,'sources':{},
      'method':'One-step median-ratio closure using analytic emission/readout dependence. Positive spectral components calibrated to3 F means on training optical histories; their combined1025-tap Gaussian FIR is regenerated in native C++. Event variance scatter calibrated from training F50 variance.',
      'limits':['Optical fraction correction is a proposal; fresh full optical transport must check it.',
        'Low-frequency correction and acquisition-state mixtures are empirical; not a uniquely identified detector mechanism.',
        'No charge-spectrum matching by per-event weights/amplitude scaling. Shared effective yield is adjusted only to preserve expected3us integral after changing timing fractions.']}
    for source,run in [('Co60',3),('Am241',12)]:
        data=arrays(OLD/f'run{run}_first60.npz');model=arrays(OUT/'parallel_shape/analysis'/f'{source}_native.npz')
        cfg=copy.deepcopy(configs[source]);before=copy.deepcopy(cfg)
        for kind in ['electron','alpha']:cfg['optical'][kind+'_fast_fraction']=fractions[kind]
        se=pulse_summary(before,'electron')[1]/pulse_summary(cfg,'electron')[1]
        sa=pulse_summary(before,'alpha')[1]/pulse_summary(cfg,'alpha')[1]
        cfg['optical']['yield_photons_per_MeV']*=se
        cfg['optical']['alpha_light_yield_ratio']*=sa/se
        adc=model['waves'].astype(float);baseline=model['baseline'];t10=model['t10']
        r=cfg['sipm']['readout'];lsb=(r['adc_max_mV']-r['adc_min_mV'])/2**r['adc_bits']
        signal=np.maximum(model['signal_mV']/lsb,0.)
        m=fluctuation_metrics(adc,baseline,t10)[:,:,3];target=data['metrics'][:,:,3].mean(0)
        h0=np.array(cfg['signal_dependent_noise']['coefficients'])
        hl=np.exp(-np.arange(1025)*2/100.);hl/=np.sqrt(hl@hl)
        kernels=[h0,hl,np.array([1.])]
        rng=np.random.default_rng(291029430+run)
        white=rng.normal(size=(8,len(adc),adc.shape[1]+1024))
        contributions=[]
        for h in kernels:
            rows=[]
            for noise in white:
                z=colored(h,noise[:,:adc.shape[1]+len(h)-1])*np.sqrt(signal)
                pair=[fluctuation_metrics(adc+sign*z,baseline,t10)[:,:,3] for sign in (-1,1)]
                rows.append((pair[0]+pair[1])/2-m)
            contributions.append(np.array(rows).mean(axis=(0,1)))
        A=np.array(contributions).T
        v=positive_fit(A/target[:,None],(target-m.mean(0))/target)
        h,strength=combine_kernels(kernels,v)
        perturb=np.array([colored(h,noise)*np.sqrt(signal) for noise in white])
        added=[]
        for z in perturb:
            pair=[fluctuation_metrics(adc+sign*z,baseline,t10)[:,:,3] for sign in (-1,1)]
            added.append((pair[0]+pair[1])/2-m)
        delta=np.array(added).mean(axis=(0,1))
        strength=max(0.,float(np.dot(delta/target,(target-m.mean(0))/target)/np.dot(delta/target,delta/target)))
        values=[];added_squared=[]
        for z in perturb*np.sqrt(strength):
            pair=[fluctuation_metrics(adc+sign*z,baseline,t10)[:,:,3] for sign in (-1,1)]
            values.extend(pair);added_squared.append(((pair[0]+pair[1])/2-m)**2)
        values=np.array(values);var=values.reshape(-1,3).var(0,ddof=1);dv=data['metrics'][:,:,3].var(0,ddof=1)
        denom=np.array(added_squared).mean(axis=(0,1));cv2=max(0.,float((dv[0]-var[0])/denom[0]));assert cv2<4
        noise=cfg['signal_dependent_noise'];noise['coefficients']=h.tolist();noise['variance_per_signal_adc']=strength;noise['event_variance_cv']=float(np.sqrt(cv2))
        noise['provenance']=str((output/'fit.json').relative_to(PROJECT))
        noise['spectral_components']='Positive combination of earlier residual spectrum,100ns exponential covariance and white component;1025 taps'
        cfg['status']='Frozen2026-10-29 training calibration for fresh full-Geant4 validation'
        cfg['evidence']['final_training_closure']={'record':str((output/'fit.json').relative_to(PROJECT)),
            'no_check_pulse_tuning':True,'electron_yield_scale':float(se),'alpha_yield_scale':float(sa)}
        save(output/f'{source}_calibrated.json',cfg)
        report['sources'][source]={'positive_component_variances':v.tolist(),'unit_component_F_contributions':A.tolist(),
          'base_F':m.mean(0).tolist(),'target_F':target.tolist(),'predicted_F':values.mean(axis=(0,1)).tolist(),
          'predicted_relative':(values.mean(axis=(0,1))/target-1).tolist(),'variance_per_signal_adc':strength,
          'event_variance_cv':float(np.sqrt(cv2)),'config_sha256':sha(output/f'{source}_calibrated.json')}
        print(source,report['sources'][source],flush=True)
    save(output/'fit.json',report)


if __name__=='__main__':main()
