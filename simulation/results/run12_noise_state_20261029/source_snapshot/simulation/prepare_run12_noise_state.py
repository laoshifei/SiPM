"""Freeze pretrigger-only cross-fitted noise states for the current PID model."""
import copy
from pathlib import Path
import numpy as np
from prepare_npe_profile import OUT as PARENT,BUILD,PROJECT
from prepare_pid import read
from extract_pid import load
from prepare_noise_model import make_kernel
from refit_baseline_mixture import fit_mixture,derive_states
from publication_stats import save,sha

OUT=PROJECT/'simulation/results/run12_noise_state_20261029'
SEGMENTS=['first60','middle20','last20']
VARIANTS=['baseline','scale','states','spectrum_states']
DATA=PROJECT/'simulation/results/signal_fluctuations_20261028'
WIDTH=PROJECT/'simulation/results/pid_bandwidth_20261029'


def sample_variance_factor(h,n=128):
    ac=np.correlate(h,h,'full')[len(h)-1:]
    c=ac[np.abs(np.arange(n)[:,None]-np.arange(n)[None,:])]
    return float((np.trace(c)-c.sum()/n)/n)


def fit_states(pre,h,lsb):
    assert pre.shape[1]==128
    x=np.c_[pre.mean(1),np.log(np.maximum(pre.var(1)-(1/12)*127/128,1e-6))]
    attempts=[]
    for k,split in [(1,.5),(2,.5),(2,.9),(2,.97)]:
        fit=fit_mixture(x,k,split)
        row={'fit':fit,'valid':False}
        if min(fit['component_n'])>=20:
            try:
                states,diagnostic=derive_states(fit,h,lsb)
                row.update(valid=True,states=states,diagnostic=diagnostic)
            except (AssertionError,ValueError,FloatingPointError) as error:
                row['failure']=str(error) or type(error).__name__
        else:row['failure']='Fewer than20 effective observations in a component'
        attempts.append(row)
    valid=[x for x in attempts if x['valid']];assert valid,'No valid pretrigger state model'
    chosen=min(valid,key=lambda r:r['fit']['BIC'])
    return chosen['states'],{'attempts':attempts,'chosen':chosen}


def apply_states(cfg,states,h,provenance):
    c=copy.deepcopy(cfg);noise=c['electronics_noise']
    noise['states']=states;noise['coefficients']=h.tolist()
    noise['state_provenance']=provenance
    c['sipm']['readout']['baseline_mV']=sum(s['weight']*s['baseline_mV'] for s in states)
    c['sipm']['readout']['noise_rms_mV']=float(np.sqrt(sum(s['weight']*s['noise_rms_mV']**2 for s in states)))
    return c


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    for folder in ['configs','fits','arrays','extraction','logs','roots']:(OUT/folder).mkdir()
    pp=read(PARENT/'validation/protocol.json');parent=pp['entries']['run12_Am241']
    base=read(PROJECT/parent['config']);h0=np.asarray(base['electronics_noise']['coefficients'])
    ro=base['sipm']['readout'];lsb=(ro['adc_max_mV']-ro['adc_min_mV'])/2**ro['adc_bits']
    current_power=sum(s['weight']*(s['noise_rms_mV']/lsb)**2 for s in base['electronics_noise']['states'])
    parent_hash=sha(PROJECT/parent['path']);entries={};groups={};inputs={}
    rng=np.random.default_rng(3010291200);uniform=rng.random(parent['events']);assignments={}
    baseline=dict(key='baseline',variant='baseline',segment=None,eval_fold=None,config=parent['config'],
        config_sha256=parent['config_sha256'],path=parent['path'],source='Am241',events=parent['events'],seed=parent['seed'],
        parent_existing=True,command=None)
    entries['baseline']=baseline
    for seg in SEGMENTS:
        path=DATA/f'run12_{seg}.npz';a=load(path);ids=a['ids'];fold=(ids//64)%2
        inputs[str(path.relative_to(PROJECT))]=sha(path)
        # Every measured event is evaluated using a state fitted on the opposite
        # fold. MC mothers get exactly one response per segment, never duplicate.
        p0=float(np.mean(fold==0));assigned=(uniform>=p0).astype(np.int8);assignments[seg]=assigned
        groups[seg]={'n_measured':len(ids),'fold0_fraction':p0,'MC_mothers_per_fold':np.bincount(assigned,minlength=2).tolist(),
            'folds':{},'support_bins':[0,1,2]}
        for ef in [0,1]:
            train=fold!=ef;test=~train
            pre=a['waves'][train,:256].astype(float)
            assert len(pre)>=100 and not np.intersect1d(ids[train],ids[test]).size
            hnew,_,freq,power,spectrum_diagnostic=make_kernel(pre)
            target=max(float(pre[:,:128].var(1).mean()-(1/12)*127/128),1e-8)
            factor=float(np.sqrt(target/(sample_variance_factor(h0)*current_power)))
            state0,fit0=fit_states(pre[:,:128],h0,lsb)
            state1,fit1=fit_states(pre[:,:128],hnew,lsb)
            label=f'{seg}_eval{ef}'
            provenance={'record':str((OUT/'fits'/f'{label}.json').relative_to(PROJECT)),
                'group':label,'training_fold':1-ef,'evaluation_fold':ef,
                'input_region':'first128 samples for state mixture; first256 linearly detrended samples for PSD; no pulse samples or PID scores',
                'role':'Measured pretrigger condition from opposite raw-index block64 parity; not a blind forecast'}
            configs={}
            scale=copy.deepcopy(base)
            for state in scale['electronics_noise']['states']:state['noise_rms_mV']*=factor
            scale['sipm']['readout']['noise_rms_mV']*=factor
            scale['electronics_noise']['state_provenance']=provenance
            configs['scale']=scale
            configs['states']=apply_states(base,state0,h0,provenance)
            configs['spectrum_states']=apply_states(base,state1,hnew,provenance)
            row={'segment':seg,'eval_fold':ef,'calibration_ids':ids[train].tolist(),'evaluation_ids':ids[test].tolist(),
                'calibration_n':int(train.sum()),'evaluation_n':int(test.sum()),'RMS_scale':factor,
                'target_first128_centered_variance_minus_quantization':target,
                'state_fit_original_spectrum':fit0,'state_fit_new_spectrum':fit1,
                'spectrum_diagnostic':spectrum_diagnostic,'code_sha256':sha(Path(__file__)),
                'selection':'Frozen previous waveform cohort, no PID/main-band selection for noise estimation.',
                'calibration_pretrigger_only':True}
            save(OUT/'fits'/f'{label}.json',row)
            np.savez_compressed(OUT/'fits'/f'{label}_spectrum.npz',frequency_mhz=freq,calibration_PSD=power,
                old_kernel=h0,new_kernel=hnew,calibration_ids=ids[train],evaluation_ids=ids[test])
            groups[seg]['folds'][str(ef)]={'fit':str((OUT/'fits'/f'{label}.json').relative_to(PROJECT)),
                'fit_sha256':sha(OUT/'fits'/f'{label}.json'),'evaluation_n':int(test.sum()),'calibration_n':int(train.sum())}
            for variant,cfg in configs.items():
                key=f'{label}_{variant}';cfg['noise_state_study']={'variant':variant,**provenance}
                cfg['status']='Frozen run12 pretrigger-only cross-fit; optical, avalanche and signal-residual parameters unchanged'
                cp=OUT/'configs'/f'{key}.json';rp=OUT/'roots'/f'{key}.root';save(cp,cfg)
                entries[key]={'key':key,'variant':variant,'segment':seg,'eval_fold':ef,'source':'Am241',
                    'events':parent['events'],'seed':parent['seed'],'parent_existing':False,
                    'config':str(cp.relative_to(PROJECT)),'config_sha256':sha(cp),'path':str(rp.relative_to(PROJECT)),
                    'command':[str(BUILD/'replay_response'),str(PROJECT/parent['path']),str(cp),str(rp)]}
            print(label,'train/eval',train.sum(),test.sum(),'RMS scale',round(factor,3),'states',len(state0),len(state1),flush=True)
    np.savez_compressed(OUT/'mother_folds.npz',**assignments)
    save(OUT/'protocol.json',{'scope':__doc__,'code_sha256':sha(Path(__file__)),'checkpoint_commit':'c13601b',
        'parent_record_sha256':sha(PARENT/'record.json'),'parent_ROOT':parent['path'],'parent_ROOT_sha256':parent_hash,
        'parent_config':parent['config'],'parent_config_sha256':parent['config_sha256'],
        'binary':str((BUILD/'replay_response').relative_to(PROJECT)),'binary_sha256':sha(BUILD/'replay_response'),
        'source_sha256':{str(p.relative_to(PROJECT)):sha(p) for p in [Path(__file__),PROJECT/'simulation/refit_baseline_mixture.py',
            PROJECT/'simulation/prepare_noise_model.py',PROJECT/'simulation/characterize_readout.py',PROJECT/'simulation/src/ReadoutNoise.cc',
            PROJECT/'simulation/src/ReadoutResponse.cc',PROJECT/'simulation/src/SignalDependentNoise.cc']},
        'inputs_sha256':inputs,'q_edges':pp['q_edges'],'support_bins':[0,1,2],'groups':groups,'entries':entries,
        'mother_fold_sha256':sha(OUT/'mother_folds.npz'),'MC_mother_fold_seed':3010291200,
        'noise_variants':VARIANTS,'unique_parent_mothers':24000,'new_native_mothers':0,
        'frozen_parameters':['geometry','optical','acquisition','sipm.sensor','readout_response','signal_dependent_noise'],
        'experimental_evaluation':'Opposite raw-index64-event block parity within each segment. Existing first60 pulse/signal calibration remains shared; only new pretrigger state estimation is cross-fitted.',
        'MC_evaluation':'One fixed uniform per mother assigns exactly one evaluation fold per segment, using the measured fold fraction. Same mothers paired across all variants and segments; no duplicated independent MC.',
        'PID_selection':'Experimental IDs/Q/PCA frozen; main-band0.4 diagnostic only at evaluation. MC has no PID cut. Report reselected and fixed parent IDs/Q, plus fixed-anchor sensitivity.',
        'uncertainty':'512 paired whole-mother bootstrap plus experimental block64 bootstrap, with block128 check. Noise fits/kernels and MC fold assignment held fixed; no fit/model uncertainty.',
        'PSD_limits':'First256=512ns; linear detrend, quantization correction, flat extrapolation below first positive Fourier frequency,257-tap FIR. Long correlations and pulse-like transients not uniquely inferred.',
        'decision':'Compare all frozen variants; do not refit to evaluation PID scores or select an unreported optimum.'})
    print('Frozen',len(entries)-1,'native noise replays plus existing parent.',flush=True)


if __name__=='__main__':main()
