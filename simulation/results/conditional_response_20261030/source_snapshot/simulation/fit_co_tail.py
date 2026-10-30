"""Train a joint two-source tail proposal; native Geant4 must validate it."""
import copy
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.signal import savgol_filter
from prepare_npe_profile import OUT as PARENT,BUILD,PROJECT
from prepare_pid import read
from extract_pid import load
from reference import wave_features
from fit_pid_bandwidth import average_bins
from refit_parallel_shape import convolved_exponentials
from publication_stats import save,sha

OUT=PROJECT/'simulation/results/co_tail_20261029'
DATA=PROJECT/'simulation/results/signal_fluctuations_20261028'
WIDTH=PROJECT/'simulation/results/pid_bandwidth_20261029'
STATE=PROJECT/'simulation/results/run12_noise_state_20261029'
T=np.arange(-40.,3000.,2.)
SOURCES=[('Co60',3,'electron'),('Am241',12,'alpha')]
CASES=[{'name':'legacy_objective','slow_lower_ns':700.,'tail_weight':0.,'early_weight':4.},
       {'name':'tail_old_bound','slow_lower_ns':700.,'tail_weight':1.,'early_weight':1.},
       {'name':'tail_relaxed_bound','slow_lower_ns':400.,'tail_weight':1.,'early_weight':1.},
       {'name':'tail_half_weight','slow_lower_ns':400.,'tail_weight':.5,'early_weight':1.}]
PRIMARY='tail_relaxed_bound'


def gate_density(waves):
    f=wave_features(waves);anchor=np.floor(f['t10_sample']).astype(int)+1
    gate=waves[np.arange(len(waves))[:,None],anchor[:,None]+np.arange(-20,1500)].astype(float)-f['baseline'][:,None]
    q=gate.sum(1);np.testing.assert_allclose(q,f['charge_long'],rtol=0,atol=1e-8)
    return gate/(2*q[:,None]),f


def normalize(y):return y/(2*y.sum())


def expected(times,rc,fraction,prompt):
    curves=[]
    for tau in times:
        curves.append((1-prompt)*convolved_exponentials([tau,2.,29.,rc])+
                      prompt*convolved_exponentials([tau,2.,5.]))
    signal=fraction*curves[0]+(1-fraction)*curves[1]
    peak=int(signal.argmax());i=int(np.flatnonzero(signal[:peak]<=.1*signal[peak])[-1])
    crossing=i+(.1*signal[peak]-signal[i])/(signal[i+1]-signal[i])
    # Average integer-anchor phase is approximated by1ns; the fixed native
    # closure below absorbs its baseline error. Native ADC uses its own anchor.
    raw=np.interp(T+crossing+1.,np.arange(len(signal)),signal,left=0.)
    area=2*raw.sum()
    return raw/area,float(area)


def descriptors(y):
    smooth=savgol_filter(y,11,3);i=int(smooth.argmax());l,m,r=smooth[i-1:i+2]
    shift=.5*(l-r)/(l-2*m+r);peak=m-.25*(l-r)*shift
    rise=[]
    for fraction in [.5,.9]:
        j=int(np.flatnonzero(smooth[:i]<=fraction*peak)[-1])
        rise.append(T[j]+2*(fraction*peak-smooth[j])/(smooth[j+1]-smooth[j]))
    return np.array([T[i]+2*shift,2*y[T<500].sum(),peak,*rise])


def tail_metrics(y):
    late=T>=2000
    return np.array([2*np.dot(T,y),2*y[late].sum(),2*np.dot(T[late],y[late])])


def unpack(x):
    e1,e2,a1,a2,rc=np.exp(x[:5])
    return {'electron_tau_ns':[e1,e2],'alpha_tau_ns':[a1,a2],'lowpass_ns':rc,
            'electron_fast_fraction':x[5],'alpha_fast_fraction':x[6]}


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    for folder in ['fit','configs','native','arrays','extraction','logs','figures']:(OUT/folder).mkdir()
    protocol=read(PARENT/'validation/protocol.json');edges=protocol['q_edges']
    cfgs={};targets={};closures={};baseline={};inputs={};support={};data_counts={};mean_files={'time_ns':T}
    state_fit=read(WIDTH/'covariance_fit/fit.json')
    for source,run,kind in SOURCES:
        e=protocol['entries'][f'run{run}_{source}'];cp=PROJECT/e['config'];cfgs[source]=read(cp)
        dp=DATA/f'run{run}_first60.npz';sp=WIDTH/'arrays'/f'measured_run{run}_first60.npz'
        mp=PARENT/'validation/arrays'/f'run{run}_{source}.npz'
        d=load(dp);s=load(sp);m=load(mp);np.testing.assert_array_equal(d['ids'],s['ids'])
        use=s['main_band'];support[source]=state_fit['states'][f'run{run}']['support_bins']
        dy,df=gate_density(d['waves'][use]);my,mf=gate_density(m['waves'])
        targets[source]=average_bins(dy,s['q'][use],edges,support[source])
        baseline[source]=average_bins(my,m['q'],edges,support[source])
        # Exact check against the existing formal mean-time scores.
        np.testing.assert_allclose(tail_metrics(targets[source])[0],
            -1000*average_bins(s['scores'][use,1],s['q'][use],edges,support[source]),atol=1e-9)
        np.testing.assert_allclose(tail_metrics(baseline[source])[0],
            -1000*average_bins(m['scores'][:,1],m['q'],edges,support[source]),atol=1e-9)
        c=cfgs[source];a,_=expected(c['optical'][kind+'_tau_ns'],c['sipm']['readout']['lowpass_ns'],
            c['optical'][kind+'_fast_fraction'],c['readout_response']['prompt_charge_fraction'])
        closures[source]=savgol_filter(baseline[source]-a,51,3)
        mean_files.update({source+'_data':targets[source],source+'_native_baseline':baseline[source],source+'_closure':closures[source]})
        data_counts[source]={'experimental_first60':int(use.sum()),'reused_MC':len(m['ids'])}
        for path in [cp,dp,sp,mp]:inputs[str(path.relative_to(PROJECT))]=sha(path)
    base=cfgs['Co60'];o=base['optical'];rc=base['sipm']['readout']['lowpass_ns']
    prompt=base['readout_response']['prompt_charge_fraction']
    initial=np.r_[np.log(o['electron_tau_ns']+o['alpha_tau_ns']+[rc]),o['electron_fast_fraction'],o['alpha_fast_fraction']]
    starts=[initial,np.r_[np.log([100.,800.,800.,9000.,120.]),.2,.5],
            np.r_[np.log([100.,600.,700.,10000.,140.]),.1,.45]]
    save(OUT/'design.json',{'scope':__doc__,'checkpoint_commit':'455da70','code_sha256':sha(Path(__file__)),
        'inputs_sha256':inputs,'parent_record_sha256':sha(PARENT/'record.json'),'noise_state_record_sha256':sha(STATE/'record.json'),
        'q_edges':edges,'support_bins':support,'counts':data_counts,'cases':CASES,'predeclared_primary':PRIMARY,
        'starts': [x.tolist() for x in starts],
        'training':'Only frozen run3/run12 first60 and previous MC, now calibration support; no later experimental pulse or new native output enters fitting.',
        'objective':'Equal-Q mean ADC/Q density, original5 waveform descriptors; tail targets add mean time/5ns, late fraction/0.002 and late contribution/5ns. Descriptive correlated penalties, not a chi-square likelihood.',
        'closure':'Fixed102ns Savitzky-Golay smoothed discrepancy between previous native mean and exponential convolution. Proposal only; never added to native signal or used as a measured waveform template.',
        'frozen':['geometry','nominal_PDE0.3','passive_signal_transmission0.5','electronics_noise','signal_dependent_noise','SPE2ns/29ns'],
        'normalization':'One source-independent yield multiplier preserves analytic electron3us gate area; alpha relative yield adjusts by the corresponding alpha/electron ratio. No per-event amplitude scaling.',
        'native_validation':'Full Geant4 required because optical times change. New seeds8000Co/24000Am; all previously explored experimental segments checked, no validation-driven refit.',
        'limits':'Conditional apparatus and emission/response parameters. First60 main-band separator is diagnostic, not experimental particle truth. Old MC is used by the proposal, so old/new transport comparison is not independent paired validation.'})
    fits={}
    for case in CASES:
        low=np.r_[np.log([80.,case['slow_lower_ns'],400.,3000.,60.]),.005,.005]
        high=np.r_[np.log([500.,3000.,1800.,15000.,200.]),.995,.995]
        weights=np.where((T>=0)&(T<600),case['early_weight'],1.);weights/=weights.sum()
        def predict(x):
            p=unpack(x);curves={}
            for source,run,kind in SOURCES:
                y,_=expected(p[kind+'_tau_ns'],p['lowpass_ns'],p[kind+'_fast_fraction'],prompt)
                curves[source]=normalize(y+closures[source])
            return p,curves
        def residual(x):
            p,curves=predict(x);rows=[]
            for source in curves:
                y=curves[source];target=targets[source]
                rows.extend((y-target)/target.max()/.01*np.sqrt(weights))
                rows.extend((descriptors(y)-descriptors(target))/np.array([10.,.004,.00003,8.,8.]))
                rows.extend(case['tail_weight']*(tail_metrics(y)-tail_metrics(target))/np.array([5.,.002,5.]))
            for kind in ['electron','alpha']:rows.append(max(0.,p[kind+'_tau_ns'][0]-p[kind+'_tau_ns'][1])/10.)
            return np.asarray(rows)
        attempts=[]
        for i,start in enumerate(starts):
            fit=least_squares(residual,np.clip(start,low+1e-6,high-1e-6),bounds=(low,high),max_nfev=220,
                              ftol=1e-10,xtol=1e-10,gtol=1e-7,diff_step=1e-4)
            parameters,curves=predict(fit.x)
            row={'start':i,'parameters':parameters,'objective':float(np.sum(residual(fit.x)**2)),
                'success':bool(fit.success),'status':int(fit.status),'nfev':int(fit.nfev),'metrics':{}}
            for source,y in curves.items():
                row['metrics'][source]={'shape_NRMSE':float(np.sqrt(np.mean((y-targets[source])**2))/targets[source].max()),
                    'tail_metrics':tail_metrics(y).tolist(),'data_tail_metrics':tail_metrics(targets[source]).tolist(),
                    'tail_difference':(tail_metrics(y)-tail_metrics(targets[source])).tolist(),
                    'descriptors':descriptors(y).tolist(),'data_descriptors':descriptors(targets[source]).tolist()}
            attempts.append(row)
            print(case['name'],i,'loss',round(row['objective'],4),'tau_e',np.round(parameters['electron_tau_ns'],3),
                'mean-time differences', {s:round(row['metrics'][s]['tail_difference'][0],3) for s in row['metrics']},'success',fit.success,flush=True)
        valid=[x for x in attempts if x['success']];assert valid,'No converged fit for '+case['name']
        best=min(valid,key=lambda r:r['objective']);fits[case['name']]={'case':case,'attempts':attempts,'best':best,
            'bounds':{'low_transformed':low.tolist(),'high_transformed':high.tolist()}}
        p=best['parameters']
        for source,run,kind in SOURCES:
            y,_=expected(p[kind+'_tau_ns'],p['lowpass_ns'],p[kind+'_fast_fraction'],prompt)
            mean_files[f"{case['name']}__{source}"]=normalize(y+closures[source])
        save(OUT/'fit'/f"{case['name']}.json",fits[case['name']])
    chosen=fits[PRIMARY]['best'];p=chosen['parameters'];areas={}
    for kind in ['electron','alpha']:
        _,oldarea=expected(o[kind+'_tau_ns'],rc,o[kind+'_fast_fraction'],prompt)
        _,newarea=expected(p[kind+'_tau_ns'],p['lowpass_ns'],p[kind+'_fast_fraction'],prompt)
        areas[kind]=oldarea/newarea
    assert all(.75<x<1.25 for x in areas.values()),'Large gate-area change requires a new explicit study'
    entries={}
    for source,run,kind in SOURCES:
        c=copy.deepcopy(cfgs[source])
        for field in ['electron_tau_ns','alpha_tau_ns','electron_fast_fraction','alpha_fast_fraction']:c['optical'][field]=p[field]
        c['sipm']['readout']['lowpass_ns']=p['lowpass_ns']
        c['optical']['yield_photons_per_MeV']*=areas['electron']
        c['optical']['alpha_light_yield_ratio']*=areas['alpha']/areas['electron']
        c['status']='Frozen joint first60 mean-time/tail proposal; native validation pending; effective conditional parameters'
        c['tail_refit']={'design':str((OUT/'design.json').relative_to(PROJECT)),'primary':PRIMARY,
            'analytic_area_preservation':areas,'native_template_added':False}
        key=f'run{run}_{source}';cp=OUT/'configs'/f'{key}.json';rp=OUT/'native'/f'{key}.root';save(cp,c)
        events=8000 if source=='Co60' else 24000;seed=301029503 if source=='Co60' else 301029512
        entries[key]={'key':key,'run':run,'source':source,'events':events,'seed':seed,'native':True,
            'config':str(cp.relative_to(PROJECT)),'config_sha256':sha(cp),'path':str(rp.relative_to(PROJECT)),
            'command':[str(BUILD/'csi_sources'),str(cp),source,str(events),str(seed),str(rp)]}
    np.savez_compressed(OUT/'proposal_curves.npz',**mean_files)
    save(OUT/'selection.json',{'scope':__doc__,'code_sha256':sha(Path(__file__)),'design_sha256':sha(OUT/'design.json'),
        'cases':fits,'selected_case':PRIMARY,'chosen_parameters':p,'analytic_gate_area_multipliers':areas,
        'baseline_training_tail_metrics':{s:tail_metrics(y).tolist() for s,y in baseline.items()},
        'data_training_tail_metrics':{s:tail_metrics(y).tolist() for s,y in targets.items()},
        'note':'Predeclared objective/bound family, best converged start within family. Other families retained as sensitivities; no later-data selection.'})
    save(OUT/'native_protocol.json',{'scope':__doc__,'selection_sha256':sha(OUT/'selection.json'),
        'design_sha256':sha(OUT/'design.json'),'q_edges':edges,'support_bins':support,'entries':entries,
        'binary':str((BUILD/'csi_sources').relative_to(PROJECT)),'binary_sha256':sha(BUILD/'csi_sources'),
        'independent_new_mothers':32000,'source_sha256':{str(x.relative_to(PROJECT)):sha(x) for x in [Path(__file__),PROJECT/'simulation/refit_parallel_shape.py']},
        'evaluation':'Independent new native seeds. Baseline MC has been used for proposal closure; no same-history comparison or new experimental holdout claim.',
        'noise':'Original first60 baseline and signal-residual model frozen; run12 cross-fitted later noise variants remain a separate experiment.'})
    print('Frozen joint tail candidate:',p,'area multipliers',areas,flush=True)


if __name__=='__main__':main()
