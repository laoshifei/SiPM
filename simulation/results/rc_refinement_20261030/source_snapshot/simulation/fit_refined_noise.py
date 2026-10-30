"""Fit positive spectra, then select native event-variance CV with joint distributions."""
import copy
import argparse
import sys
import numpy as np
from scipy.optimize import nnls
from rc_refinement_common import *
from fit_state_signal_noise import design
from fit_pid_bandwidth import projected_cov
from fit_signal_fluctuations import colored
from refit_final_parameters import combine_kernels
from publication_stats import block_counts


def proposal(key,entry,p):
    source=entry['source'];run=entry['run'];seg=entry['segment'];B=entry['block'];ef=entry['evaluation_fold']
    data,_=measured(run,seg);train=(data['ids']//B)%2!=ef;target=summary(data,p,source,train.astype(int))
    m=load(OUT/'arrays'/f'{key}_base.npz');base=summary(m,p,source);cfg=read(OUT/'configs'/f'{key}_base.json')
    kernels=[np.asarray(cfg['signal_dependent_noise']['coefficients'])]
    for tau in [400.,1500.]:
        h=np.exp(-np.arange(1025)*2/tau);kernels.append(h/np.linalg.norm(h))
    w=event_weights(m,p,source);sqrt_signal=np.sqrt(np.maximum(m['signal'],0));fm=F_metrics(m['waves'],m['baseline'],m['t10'])
    rng=np.random.default_rng(710031100+run+100*SEGMENTS.index(seg)+B+10*ef)
    white=rng.normal(size=(16,len(w),2496+1024))
    def contribution(h):
        C=np.einsum('n,nij->ij',w,projected_cov(m['signal'],m['anchor'],m['scores'],m['qcalc'],h,PCA))
        fs=[]
        for z in white:
            noise=colored(h,z[:,:2496+len(h)-1])*sqrt_signal
            v=F_metrics(noise,np.zeros(len(w)),m['t10']);fs.append(w@((v[:,:,2]-v[:,:,1])/fm[:,:,0]))
        return C,np.mean(fs,axis=0)
    C,F=map(np.asarray,zip(*(contribution(h) for h in kernels)));families={};jobs=[]
    oldcv=cfg['signal_dependent_noise']['event_variance_cv'];cvs=sorted(set([oldcv]+p['CV_grid']))
    for family,nc in [('inherited',1),('spectrum',3)]:
        A,t,vector=design(target,C[:nc],F[:nc],base);v,loss=nnls(A,t)
        if v.sum()<=1e-15:h=kernels[0];initial=0.
        elif nc==1:h=kernels[0];initial=float(v[0])
        else:h,initial=combine_kernels(kernels,v)
        c,f=contribution(h);u=vector(c,f);strength=max(0.,float(u@t/(u@u))) if v.sum()>1e-15 else 0.
        params=copy.deepcopy(cfg);params['signal_dependent_noise'].update(coefficients=h.tolist(),variance_per_signal_adc=strength,
            provenance=relative(OUT/'fits'/f'{key}_noise.json'))
        choices=[]
        for index,cv in enumerate(cvs):
            candidate=copy.deepcopy(params);candidate['signal_dependent_noise']['event_variance_cv']=cv
            candidate['rc_refinement'].update(stage='noise_grid',family=family,CV_index=index)
            nk=f'{key}_{family}_cv{index}';jobs.append(job(nk,candidate));choices.append({'CV':cv,'native_key':nk,'inherited_CV':cv==oldcv})
        rngb=np.random.default_rng(710031500+run+100*SEGMENTS.index(seg)+B+10*ef);refits=[]
        for ib in range(128):
            count=block_counts(data['ids'],tuple(data['span']),B,rngb)*train;tt=summary(data,p,source,count)
            if tt is None:continue
            AA,tb,_=design(tt,C[:nc],F[:nc],base);vv,_=nnls(AA,tb);refits.append(vv)
        families[family]={'component_powers':v,'initial_strength':initial,'strength':strength,
            'base':base,'target':target,'predicted_cov':base['cov']+strength*c,'predicted_F':base['F']+strength*f,
            'linear_loss_squared':loss**2,'combined_loss':np.sum((strength*u-t)**2),'choices':choices,
            'conditional_power_refits':refits}
    report={'scope':__doc__,'code_sha256':sha(__file__),'context':key,'entry':entry,'families':families,
        'component_cov':C,'component_F':F,'calibration_ids':data['ids'][train],'evaluation_ids':data['ids'][~train],
        'limits':'Full covariance/F moment proposal uses linearized scores and16 effective-noise replicas. Native grid validates own-anchor ADC. Power bootstrap fixes RC/electronics/spectra/MC. CV selection subsequently uses native training loss only.'}
    save(OUT/'fits'/f'{key}_noise.json',serialize(report))
    print(key,'spectrum powers',np.round(v,4),'native CV candidates',len(cvs)*2,flush=True)
    return jobs


def select(key,entry,p):
    data,_=measured(entry['run'],entry['segment']);B=entry['block'];ef=entry['evaluation_fold'];source=entry['source']
    train=(data['ids']//B)%2!=ef;target=summary(data,p,source,train.astype(int));fit=read(OUT/'fits'/f'{key}_noise.json');result={}
    for family,f in fit['families'].items():
        choices=[];ms=[]
        for candidate in f['choices']:
            m=summary(load(OUT/'arrays'/f"{candidate['native_key']}.npz"),p,source)
            loss,parts=joint_loss(m,target);choices.append({**candidate,'loss':loss,'loss_parts':parts,'native_summary':m});ms.append(m)
        winner=int(np.argmin([r['loss'] for r in choices]));old=next(i for i,r in enumerate(choices) if r['inherited_CV'])
        rng=np.random.default_rng(710032100+entry['run']+100*SEGMENTS.index(entry['segment'])+B+10*ef);stability=[]
        for ib in range(128):
            count=block_counts(data['ids'],tuple(data['span']),B,rng)*train;tt=summary(data,p,source,count)
            if tt is None:continue
            losses=[joint_loss(m,tt)[0] for m in ms];stability.append(int(np.argmin(losses)))
        result[family]={'choices':choices,'chosen_index':winner,'chosen_native_key':choices[winner]['native_key'],
            'chosen_CV':choices[winner]['CV'],'inherited_native_key':choices[old]['native_key'],
            'inherited_CV':choices[old]['CV'],'conditional_selection_refits':stability}
    record={'scope':'Opposite-block native joint distribution selection','source':source,'context':key,'entry':entry,
        'code_sha256':sha(__file__),'calibration_ids':data['ids'][train],'evaluation_ids':data['ids'][~train],
        'families':result,'target':target,'limits':'Only CV selected on this grid. Conditional selection bootstrap fixes RC, powers, spectra, pretrigger parameters and native mothers. No claim of independent validation.'}
    save(OUT/'fits'/f'{key}_selection.json',serialize(record))
    print(key,'selected CVs',[(f,r['chosen_CV']) for f,r in result.items()],flush=True)
    return {'fit':relative(OUT/'fits'/f'{key}_selection.json'),'fit_sha256':sha(OUT/'fits'/f'{key}_selection.json'),
        'families':{f:{k:r[k] for k in ['chosen_native_key','chosen_CV','inherited_native_key','inherited_CV']} for f,r in result.items()}}


def main(source_filter):
    destination=OUT/f'noise_selection_{source_filter}.json'
    assert not destination.exists()
    p=read(OUT/'protocol.json');cal=read(OUT/'mean_calibration.json');records={}
    for source,run in SOURCES:
        if source!=source_filter:continue
        jobs=[]
        for key,r in cal['groups'].items():
            if r['entry']['source']==source:jobs+=proposal(key,r['entry'],p)
        verification=read(OUT/'cache_verification.json')
        binary=PROJECT/verification['cached_binary']
        assert sha(binary)==verification['cached_binary_sha256'] and verification['passed']
        execute(source,jobs,'noise_grid',binary=binary)
        for key,r in cal['groups'].items():
            if r['entry']['source']==source:records[key]=select(key,r['entry'],p)
    save(destination,{'scope':__doc__,'code_sha256':sha(__file__),
        'protocol_sha256':sha(OUT/'protocol.json'),'groups':records,'new_independent_mothers':0,'default_changed':False})


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--source',choices=['Co60','Am241']);args=ap.parse_args()
    if args.source:main(args.source)
    else:
        assert not (OUT/'noise_selection.json').exists()
        processes=[subprocess.Popen([sys.executable,'-u',str(Path(__file__)), '--source',source]) for source,_ in SOURCES]
        codes=[process.wait() for process in processes];assert all(code==0 for code in codes),codes
        groups={}
        for source,_ in SOURCES:groups.update(read(OUT/f'noise_selection_{source}.json')['groups'])
        save(OUT/'noise_selection.json',{'scope':__doc__,'code_sha256':sha(__file__),
            'protocol_sha256':sha(OUT/'protocol.json'),'groups':groups,'new_independent_mothers':0,'default_changed':False})
