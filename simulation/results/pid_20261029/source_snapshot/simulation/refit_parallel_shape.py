"""Train a common parallel-path readout with particle-dependent emission times.

Analytic convolutions only propose native Geant4 parameters. All evaluated
families are saved; no measured curve is used as a native response template.
"""
import copy
import json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.signal import savgol_filter
from refit_waveform_shape import PROJECT, OLD, OUT, T, MASK, arrays, sha, save, normalize, descriptors


EDGES = np.arange(0., 10001., 1.)


def convolved_exponentials(taus):
    """Exact bin masses of a convolution of exponential unit-area kernels."""
    taus = np.array(taus, dtype=float)
    if min(abs(a-b) for i,a in enumerate(taus) for b in taus[i+1:]) < .05:
        # Numerically stable fallback when two poles nearly coincide.
        from scipy.signal import fftconvolve
        y = -np.diff(np.exp(-EDGES/taus[0]))
        for tau in taus[1:]:
            y = fftconvolve(y, -np.diff(np.exp(-EDGES/tau)))[:len(y)]
        return y
    coefficients = np.array([tau**(len(taus)-1)/np.prod(tau-np.delete(taus,i)) for i,tau in enumerate(taus)])
    masses = -np.diff(np.exp(-EDGES[None,:]/taus[:,None]),axis=1)
    return coefficients @ masses


def expected(times, rc, fraction, prompt):
    curves = []
    for tau in times:
        slow = convolved_exponentials([tau,2.,29.,rc])
        fast = convolved_exponentials([tau,2.,5.])
        curves.append((1-prompt)*slow+prompt*fast)
    signal = fraction*curves[0]+(1-fraction)*curves[1]
    peak = int(signal.argmax())
    before = np.flatnonzero(signal[:peak] <= .1*signal[peak])[-1]
    crossing = before+(.1*signal[peak]-signal[before])/(signal[before+1]-signal[before])
    return normalize(np.interp(T,np.arange(len(signal))-crossing,signal,left=0.))


def unpack(x, separate):
    if separate:
        e1,e2,a1,a2,rc = np.exp(x[:5]); fe,fa,prompt=x[5:]
    else:
        e1,e2,rc=np.exp(x[:3]); a1,a2=e1,e2;fe,fa,prompt=x[3:]
    return {'electron_tau_ns':[e1,e2], 'alpha_tau_ns':[a1,a2], 'lowpass_ns':rc,
            'electron_fast_fraction':fe, 'alpha_fast_fraction':fa, 'prompt_charge_fraction':prompt}


def main():
    out = OUT/'parallel_shape';out.mkdir(exist_ok=False)
    targets={};closures={};configs={};hashes={}
    for source,run,kind in [('Co60',3,'electron'),('Am241',12,'alpha')]:
        dp=OLD/f'run{run}_first60.npz';npz=OLD/'independent'/f'{source}_native.npz'
        cp=OLD/'independent'/f'{source}_calibrated.json'
        targets[source]=arrays(dp)['shapes'].mean(0);cfg=json.loads(cp.read_text());configs[source]=cfg
        optical=cfg['optical'];readout=cfg['sipm']['readout']
        base=expected(optical[kind+'_tau_ns'],readout['lowpass_ns'],optical[kind+'_fast_fraction'],0.)
        closures[source]=savgol_filter(arrays(npz)['shapes'].mean(0)-base,51,3)
        hashes.update({str(p.relative_to(PROJECT)):sha(p) for p in [dp,npz,cp]})
    weights=np.ones(MASK.sum());weights[(T[MASK]>=0)&(T[MASK]<600)]=4.;weights/=weights.sum()
    cases=[]
    for separate in (False,True):
        def predict(x):
            p=unpack(x,separate);curves={}
            for source,kind in [('Co60','electron'),('Am241','alpha')]:
                y=expected(p[kind+'_tau_ns'],p['lowpass_ns'],p[kind+'_fast_fraction'],p['prompt_charge_fraction'])
                curves[source]=normalize(y+closures[source])
            return p,curves
        def residual(x):
            p,curves=predict(x);rows=[]
            for source,y in curves.items():
                target=targets[source]
                rows.extend((y[MASK]-target[MASK])/target.max()/.01*np.sqrt(weights))
                rows.extend((descriptors(y)-descriptors(target))/np.array([10.,.004,.00003,8.,8.]))
            for kind in ('electron','alpha'):
                a,b=p[kind+'_tau_ns'];rows.append(max(0.,a-b)/10.)
            return np.array(rows)
        attempts=[]
        for initial in [(479,1373,479,1373,105,.62,.15,.03),
                        (300,1400,800,4000,110,.4,.7,.05),
                        (500,3000,900,8000,80,.75,.5,.06),
                        (200,800,850,8000,120,.2,.5,.04)]:
            e1,e2,a1,a2,rc,fe,fa,prompt=initial
            times=[e1,e2,a1,a2,rc] if separate else [e1,e2,rc]
            low=[80,700,150,700,20] if separate else [80,700,20]
            high=[1800,15000,1800,15000,200] if separate else [1800,15000,200]
            x=np.r_[np.log(times),fe,fa,prompt]
            fit=least_squares(residual,x,bounds=(np.r_[np.log(low),.005,.005,0.],np.r_[np.log(high),.995,.995,.15]),
                               max_nfev=150,ftol=1e-9,xtol=1e-9,gtol=1e-7,diff_step=1e-4)
            p,curves=predict(fit.x)
            row={'parameters':p,'objective':float(np.sum(residual(fit.x)**2)), 'success':bool(fit.success),
                 'nfev':fit.nfev,'status':int(fit.status),'metrics':{}}
            for source,y in curves.items():
                row['metrics'][source]={'mean_nrmse':float(np.sqrt(np.mean((y[MASK]-targets[source][MASK])**2))/targets[source].max()),
                  'predicted_descriptors':descriptors(y).tolist(),'target_descriptors':descriptors(targets[source]).tolist()}
            attempts.append(row)
            print('separate',separate,'loss',row['objective'],'parameters',p,flush=True)
        cases.append({'separate_particle_times':separate,'attempts':attempts,'best':min(attempts,key=lambda r:r['objective'])})
    shared=cases[0]
    selected=shared if max(m['mean_nrmse'] for m in shared['best']['metrics'].values())<.01 else min(cases,key=lambda c:c['best']['objective'])
    p=selected['best']['parameters'];curves={}
    for source,kind in [('Co60','electron'),('Am241','alpha')]:
        cfg=copy.deepcopy(configs[source]);old_optical=copy.deepcopy(cfg['optical'])
        for key in ('electron_tau_ns','alpha_tau_ns','electron_fast_fraction','alpha_fast_fraction'):cfg['optical'][key]=p[key]
        cfg['sipm']['readout']['lowpass_ns']=p['lowpass_ns']
        cfg['readout_response']={'model':'parallel_paths','prompt_charge_fraction':p['prompt_charge_fraction'],
                                'prompt_decay_ns':5.,'prompt_lowpass_ns':0.}
        cfg.pop('signal_dependent_noise',None)
        cfg['status']='2026-10-29 training parallel-response proposal; full Geant4 and residual calibration pending'
        cfg['evidence']['shape_refit']={'record':str((out/'shape_fit.json').relative_to(PROJECT)),
            'source_identity_not_readout_parameter':True,'proposal_closure_not_in_native_output':True,
            'training_only':'run3/run12 frozen first60; optical times conditional on apparatus assumptions'}
        save(out/f'{source}_shape_candidate.json',cfg)
        curves[source]=expected(p[kind+'_tau_ns'],p['lowpass_ns'],p[kind+'_fast_fraction'],p['prompt_charge_fraction'])
    np.savez_compressed(out/'proposal_curves.npz',time_ns=T,**{s+'_prediction':v for s,v in curves.items()},
                        **{s+'_data':v for s,v in targets.items()},**{s+'_closure':v for s,v in closures.items()})
    save(out/'shape_fit.json',{'scope':__doc__,'code_sha256':sha(Path(__file__)),'inputs_sha256':hashes,
        'cases':cases,'selected':selected,
        'selection_rule':'Shared times preferred when both corrected NRMSE<1%; otherwise minimum objective.',
        'closure_scope':'Fixed smooth discrepancy from old full native MC used in proposal only; no template is installed.',
        'parameter_bounds':{'fast_ns':[80,1800],'alpha_fast_ns':[150,1800],'slow_ns':[700,15000],'RC_ns':[20,200],
                            'fractions':[.005,.995],'prompt_fraction':[0,.15]},
        'validation_requirement':'Fresh native full Geant4 optical transport; no charge matching by individual amplitude rescaling.'})
    print('Selected',json.dumps(p),flush=True)


if __name__=='__main__':main()
