"""Second-stage training-only fit of event-to-event residual variance scatter.

Motivated by the too-narrow F50 distribution of the constant-variance model.
Only training F50 variance is fitted. Other filter distributions and later
segments are checks; this adaptive extension is not a preregistered test.
"""
import json
from pathlib import Path
import numpy as np
import ROOT
from calibrate_signal_fluctuations import OUT,save
from fit_signal_fluctuations import colored
from diagnose_fluctuations import fluctuation_metrics
from diagnose_charge import sha


def main():
    target=OUT/'variance_scatter';target.mkdir(exist_ok=True)
    report={'scope':__doc__,'code_sha256':sha(Path(__file__)),
      'parent_fit_sha256':sha(OUT/'fit.json'),'parent_validation_sha256':sha(OUT/'validation.json'),
      'method':'Mean-one lognormal residual-variance multiplier. CV^2=max(0,(Var_data(F50)-Var_fixed_model(F50))/E[extra_F50^2]). Model moments from16 antithetic electronics replicas over fixed parent histories. Fixed t10/baseline for this moment estimate. No new transport; no physical gain assigned.',
      'sources':{}}
    for source,name,run in [('Co60','co60',3),('Am241','am241',12)]:
        with np.load(OUT/f'run{run}_first60.npz') as z:data=dict(z)
        with np.load(OUT/f'{source}_parent.npz') as z:parent=dict(z)
        config=json.loads((OUT/f'{source}_calibrated.json').read_text());r=config['sipm']['readout'];lsb=(r['adc_max_mV']-r['adc_min_mV'])/2**r['adc_bits']
        file=ROOT.TFile.Open(str(OUT/name/'events.root'));t=file.Get('Events');t.SetBranchStatus('*',0);t.SetBranchStatus('signal_mV',1)
        signals=[]
        for idx in parent['ids']:
            t.GetEntry(int(idx));signals.append(np.array(t.signal_mV))
        file.Close();signals=np.array(signals)
        analog=(signals+parent['analog_mV']-parent['signal_mV']-r['adc_min_mV'])/lsb
        base=fluctuation_metrics(analog,parent['baseline'],parent['t10'])[:,:,3]
        model=config['signal_dependent_noise'];h=np.array(model['coefficients'])*np.sqrt(model['variance_per_signal_adc'])
        rng=np.random.default_rng(281028330+run);observed=[];squared_added=[]
        for i in range(8):
            z=colored(h,rng.normal(size=(len(signals),2496+len(h)-1)))*np.sqrt(np.maximum(signals/lsb,0.))
            pair=[fluctuation_metrics(analog+sign*z,parent['baseline'],parent['t10'])[:,:,3] for sign in (-1,1)]
            observed.extend(pair);squared_added.append(((pair[0]+pair[1])/2-base)**2)
        values=np.array(observed);variance=values.reshape(-1,3).var(axis=0,ddof=1)
        data_variance=data['metrics'][:,:,3].var(axis=0,ddof=1)
        denominator=np.array(squared_added).mean(axis=(0,1))
        cv2=max(0.,float((data_variance[0]-variance[0])/denominator[0]));assert cv2<4
        model['event_variance_cv']=float(np.sqrt(cv2))
        model['event_variance_provenance']='Training F50 variance only; positive mean-one lognormal residual multiplier. Not physical gain or isotope-dependent response.'
        config['status']='Conditional acquisition calibration with empirical residual covariance and event scatter; not an independently identified detector mechanism'
        save(target/f'{source}_calibrated.json',config)
        report['sources'][source]={'run':run,'data_F_variance':data_variance.tolist(),'model_fixed_F_variance':variance.tolist(),
          'extra_F_squared_mean':denominator.tolist(),'event_variance_cv':float(np.sqrt(cv2)),
          'predicted_F_variance':(variance+cv2*denominator).tolist(),'config_sha256':sha(target/f'{source}_calibrated.json')}
        print(source,'event variance CV',np.sqrt(cv2),'predicted/data var',(variance+cv2*denominator)/data_variance,flush=True)
    save(target/'fit.json',report)


if __name__=='__main__':main()
