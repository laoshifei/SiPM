"""Conditional mean-shape fit, with shared scintillation times and passive RC.

This is an effective forward-model initialization for Geant4, not an independent
measurement of intrinsic scintillation lifetimes, sensor gain or source distance.
The Am-dominant component is provisionally assigned to alpha in this hypothesis.
"""
from pathlib import Path
import json
import hashlib
import numpy as np
import ROOT
from reference import PROJECT,TIME_NS

def convolve(a,b):
    n=1<<(len(a)+len(b)-2).bit_length()
    return np.fft.irfft(np.fft.rfft(a,n)*np.fft.rfft(b,n),n)[:len(a)]

def expectation(tau1,tau2,rc,fraction):
    dt=2.;edges=np.arange(0,16002,dt)
    # Integrated photons per bin, convolved with charge-conserving SPE and RC.
    emission=fraction*(-np.diff(np.exp(-edges/tau1)))+(1-fraction)*(-np.diff(np.exp(-edges/tau2)))
    rise=2.;decay=29.
    cdf=1-(decay*np.exp(-edges/decay)-rise*np.exp(-edges/rise))/(decay-rise)
    spe=np.diff(cdf);rc_kernel=-np.diff(np.exp(-edges/rc))
    signal=convolve(convolve(emission,spe),rc_kernel)/dt
    peak=int(signal.argmax());before=np.flatnonzero(signal[:peak]<=.1*signal[peak])[-1]
    crossing=before+(.1*signal[peak]-signal[before])/(signal[before+1]-signal[before])
    t=(np.arange(len(signal))-crossing)*dt
    y=np.interp(TIME_NS,t,signal,left=0.,right=0.)
    return y/(y[(TIME_NS>=-40)&(TIME_NS<3000)].sum()*dt)

def metrics(observed,predicted):
    mask=(TIME_NS>=0)&(TIME_NS<=3500)
    residual=predicted[mask]-observed[mask]
    return {'nrmse_peak':float(np.sqrt(np.mean(residual**2))/observed.max()),
            'max_abs_residual_over_peak':float(np.max(np.abs(residual))/observed.max()),
            'integral_short_long_predicted':float(predicted[(TIME_NS>=-40)&(TIME_NS<500)].sum()*2),
            'integral_short_long_measured_mean':float(observed[(TIME_NS>=-40)&(TIME_NS<500)].sum()*2)}

def main():
    ROOT.gROOT.SetBatch(True)
    target_dir=PROJECT/'simulation/results/reference';out=PROJECT/'simulation/results/calibration';out.mkdir(parents=True,exist_ok=True)
    with np.load(target_dir/'references.npz') as z: data={k:z[k] for k in z.files}
    target=[data['run3_train_mean'],data['run12_train_mean']]
    mask=(TIME_NS>=0)&(TIME_NS<=3500)
    def objective(p):
        a,b,rc=np.exp([p[0],p[1],p[2]])
        if b<=a:return 1e6+(a-b)**2
        return sum(float(np.mean(((expectation(a,b,rc,p[3+j])[mask]-target[j][mask])/target[j].max())**2)) for j in (0,1))
    candidates=[]
    for start in [(679,3340,256,.638,.75),(400,2000,100,.7,.2),(1000,6000,50,.8,.3),(250,1600,400,.6,.1)]:
        m=ROOT.Math.Factory.CreateMinimizer('Minuit2','Migrad');m.SetPrintLevel(-1);m.SetMaxFunctionCalls(5000);m.SetTolerance(1e-7)
        fun=ROOT.Math.Functor(objective,5);m.SetFunction(fun)
        for i,(name,lower,upper) in enumerate([('log_fast',100,2000),('log_slow',800,12000),('log_rc',2,1000)]):
            m.SetLimitedVariable(i,name,float(np.log(start[i])),.02,float(np.log(lower)),float(np.log(upper)))
        m.SetLimitedVariable(3,'electron_fast_fraction',start[3],.01,.005,.995)
        m.SetLimitedVariable(4,'Am_dominant_fast_fraction',start[4],.01,.005,.995)
        ok=m.Minimize();x=[float(m.X()[i]) for i in range(5)]
        candidates.append({'parameters':x,'objective':objective(x),'success':bool(ok),'status':m.Status()})
    best=min(candidates,key=lambda c:c['objective']);x=best['parameters'];fast,slow,rc=map(float,np.exp(x[:3]));fco,fam=x[3:]
    cfg_path=PROJECT/'simulation/configs/csi_tl_initial.json';cfg=json.loads(cfg_path.read_text())
    cfg['optical'].update(electron_tau_ns=[fast,slow],alpha_tau_ns=[fast,slow],electron_fast_fraction=fco,alpha_fast_fraction=fam)
    cfg['sipm']['readout']['lowpass_ns']=rc
    ref=json.loads((target_dir/'references.json').read_text());lsb=(cfg['sipm']['readout']['adc_max_mV']-cfg['sipm']['readout']['adc_min_mV'])/16384
    cfg['sipm']['readout']['baseline_mV']=ref['groups']['run3_train']['baseline_adc']*lsb
    cfg['sipm']['readout']['noise_rms_mV']=ref['groups']['run3_train']['baseline_rms_adc']*lsb
    cfg['status']='Effective mean-shape initialization fit to run3/run12 training only; optical and particle interpretation remain hypotheses'
    cfg['evidence']['fit_policy']='Shared tau_fast/tau_slow and passive RC; separate source-dominant fractions. Am-dominant -> alpha is a testable hypothesis, not truth. No source distance was fitted. No preamplifier added.'
    (out/'csi_tl_effective.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2)+'\n')
    curves={'time_ns':TIME_NS,'Co60':expectation(fast,slow,rc,fco),'Am241_dominant_hypothesis':expectation(fast,slow,rc,fam)}
    results={}
    for key,group in ref['groups'].items():
        pred=curves['Co60' if group['source']=='Co60' else 'Am241_dominant_hypothesis']
        results[key]={'role':group['role'],**metrics(data[key+'_mean'],pred)}
    summary={'parameters':{'fast_ns':fast,'slow_ns':slow,'passive_rc_ns':rc,'electron_fast_fraction':fco,'Am_dominant_fraction':fam},
        'fits':candidates,'metrics':results,'reference_sha256':hashlib.sha256((target_dir/'references.npz').read_bytes()).hexdigest(),
        'scope':'Effective normalized mean shape only; unweighted least squares, correlated residuals, no chi-square probability or intrinsic parameter errors claimed.',
        'caveats':['Waveforms already explored; check groups are not blind','No noise or light-yield distribution validation from mean fit','Short SPE constants fixed to 2/29 ns as proxy','Optical transport omitted in this fast initialization and checked with actual Geant4 afterward','Different RC/scintillation combinations may be non-identifiable; minima and bounds retained','Gain/PDE/source gap are not inferred from normalized shapes']}
    (out/'fit.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n');np.savez_compressed(out/'expectations.npz',**curves)
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
