"""Refit stochastic residuals on run3/12 training with new native optical histories."""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
from reference import PROJECT, wave_features
from diagnose_charge import sha
from diagnose_fluctuations import fluctuation_metrics
from calibrate_signal_fluctuations import save
from fit_signal_fluctuations import colored, fir_from_power

OLD=PROJECT/'simulation/results/signal_fluctuations_20261028'
BASE=PROJECT/'simulation/results/waveform_refit_20261029'


def arrays(path):
    with np.load(path) as z:return dict(z)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();args.native=args.native.resolve();args.out=args.out.resolve()
    args.out.mkdir(parents=True,exist_ok=False)
    native=json.loads((args.native/'validation.json').read_text())
    report={'scope':__doc__,'code_sha256':sha(Path(__file__)),
      'native_validation_sha256':sha(args.native/'validation.json'),
      'training_only':'Frozen run3/12 first60; no other measured pulse samples fitted.',
      'method':'Same257-tap conditional spectrum residual as earlier phase.16 antithetic Gaussian electronics replicas calibrate three F means and training F50 variance. Native full transport/readout mean frozen.',
      'limits':'Acquisition-specific empirical covariance, not a unique detector mechanism. Quantization/t10 changes tested later on native ADC. Optical parameters conditional on effective mean fit.',
      'sources':{}}
    for source,name,run in [('Co60','co60',3),('Am241','am241',12)]:
        data=arrays(OLD/f'run{run}_first60.npz');model=arrays(args.native/f'{source}_native.npz')
        cfg=copy.deepcopy(native['sources'][source]['config']);r=cfg['sipm']['readout']
        lsb=(r['adc_max_mV']-r['adc_min_mV'])/2**r['adc_bits']
        excess=data['power'].mean(0)-model['power'].mean(0)
        h=fir_from_power(data['frequency_MHz'],excess)
        h/=np.sqrt(np.dot(h,h))
        signal=np.maximum(model['signal_mV']/lsb,0.)
        adc=model['waves'].astype(float);baseline=model['baseline'];t10=model['t10']
        base=fluctuation_metrics(adc,baseline,t10)[:,:,3]
        rng=np.random.default_rng(291029300+run)
        innovations=rng.normal(size=(8,len(adc),adc.shape[1]+len(h)-1))
        perturb=np.array([colored(h,x) for x in innovations])*np.sqrt(signal)[None,:,:]
        delta=[]
        for z in perturb:
            for sign in (-1,1):delta.append(fluctuation_metrics(adc+sign*z,baseline,t10)[:,:,3]-base)
        added=np.array(delta).mean(axis=(0,1))
        target=data['metrics'][:,:,3].mean(0)
        strength=max(0.,float(np.dot(added/target,(target-base.mean(0))/target)/np.dot(added/target,added/target)))
        values=[];squared_added=[]
        for z in perturb*np.sqrt(strength):
            pair=[fluctuation_metrics(adc+sign*z,baseline,t10)[:,:,3] for sign in (-1,1)]
            values.extend(pair);squared_added.append(((pair[0]+pair[1])/2-base)**2)
        values=np.array(values);variance=values.reshape(-1,3).var(axis=0,ddof=1)
        data_variance=data['metrics'][:,:,3].var(axis=0,ddof=1)
        denominator=np.array(squared_added).mean(axis=(0,1))
        cv2=max(0.,float((data_variance[0]-variance[0])/denominator[0]));assert cv2<4
        cfg['signal_dependent_noise']={'model':'gaussian_fir_signal_variance','coefficients':h.tolist(),
          'variance_per_signal_adc':strength,'sample_period_ns':2.,'event_variance_cv':float(np.sqrt(cv2)),
          'calibration_run':run,'parameter_status':'Training-only empirical acquisition residual; no source-dependent SiPM gain inferred.',
          'provenance':str((args.out/'fit.json').relative_to(PROJECT))}
        cfg['status']='2026-10-29 native waveform calibration; later-segment and fresh-seed validation pending'
        save(args.out/f'{source}_calibrated.json',cfg)
        report['sources'][source]={'run':run,'base_F':base.mean(0).tolist(),'target_F':target.tolist(),
          'predicted_F':values.mean(axis=(0,1)).tolist(),'predicted_relative':(values.mean(axis=(0,1))/target-1).tolist(),
          'residual_variance_per_signal_ADC':strength,'event_variance_cv':float(np.sqrt(cv2)),
          'data_F_variance':data_variance.tolist(),'fixed_model_F_variance':variance.tolist(),
          'predicted_F_variance':(variance+cv2*denominator).tolist(),
          'negative_spectral_bins':int((excess<0).sum()),'unprojected_residual_power':excess.tolist(),
          'config_sha256':sha(args.out/f'{source}_calibrated.json'),
          'parent_ROOT':native['sources'][source]['ROOT_path'],'parent_ROOT_sha256':native['sources'][source]['root_sha256']}
        print(source,'variance',strength,'cv',np.sqrt(cv2),'predicted rel',values.mean(axis=(0,1))/target-1,flush=True)
    save(args.out/'fit.json',report)


if __name__=='__main__':main()
