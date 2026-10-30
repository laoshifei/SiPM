"""Training-block joint PID covariance/F proposals; native ADC replay required."""
import copy
import numpy as np
from scipy.optimize import nnls
from state_response_common import *
from fit_pid_bandwidth import projected_cov
from fit_signal_fluctuations import colored
from refit_final_parameters import combine_kernels
from publication_stats import block_counts


def design(target,C,F,base):
    i,j=np.triu_indices(3);normal=np.sqrt(np.diag(target['cov'])[i]*np.diag(target['cov'])[j])
    assert np.all(normal>0) and np.all(target['F']>0)
    def vector(c,f):return np.r_[c[i,j]/normal/.25,f/target['F']/.10]
    return np.column_stack([vector(c,f) for c,f in zip(C,F)]),vector(target['cov']-base['cov'],target['F']-base['F']),vector


def fit_one(key,entry,p):
    source=entry['source'];run=entry['run'];B=entry['block'];ef=entry['evaluation_fold'];seg=entry['segment']
    a,_=measured(run,seg);train=(a['ids']//B)%2!=ef
    target=moments(a,p,source,train.astype(int));m=load(OUT/'arrays'/f'{key}_base.npz')
    cfg=read(OUT/'configs'/f'{key}_base.json');base=moments(m,p,source);w=event_weights(m,p,source)
    kernels=[np.asarray(cfg['signal_dependent_noise']['coefficients'])]
    for tau in [400.,1500.]:
        h=np.exp(-np.arange(1025)*2/tau);kernels.append(h/np.linalg.norm(h))
    rng=np.random.default_rng(610031100+run+100*SEGMENTS.index(seg)+B+10*ef)
    white=rng.normal(size=(16,len(m['ids']),2496+1024))
    sqrt_signal=np.sqrt(np.maximum(m['signal'],0))
    base_metrics=F_metrics(m['waves'],m['baseline'],m['t10'])
    def contribution(h):
        C=np.einsum('n,nij->ij',w,projected_cov(m['signal'],m['anchor'],m['scores'],m['qcalc'],h,PCA))
        values=[]
        for z in white:
            noise=colored(h,z[:,:2496+len(h)-1])*sqrt_signal
            fm=F_metrics(noise,np.zeros(len(w)),m['t10'])
            values.append(w@((fm[:,:,2]-fm[:,:,1])/base_metrics[:,:,0]))
        return C,np.mean(values,axis=0)
    C,F=map(np.asarray,zip(*(contribution(h) for h in kernels)));report={};jobs=[]
    for variant,nc in [('response_inherited',1),('response_spectrum',3)]:
        A,t,vector=design(target,C[:nc],F[:nc],base);v,loss=nnls(A,t)
        if v.sum()<=1e-15:h=kernels[0];initial=0.
        elif nc==1:h=kernels[0];initial=float(v[0])
        else:h,initial=combine_kernels(kernels,v)
        c,f=contribution(h);unit=vector(c,f);strength=max(0.,float(unit@t/(unit@unit))) if v.sum()>1e-15 else 0.
        candidate=copy.deepcopy(cfg);candidate['signal_dependent_noise'].update(coefficients=h.tolist(),
            variance_per_signal_adc=strength,provenance=relative(OUT/'fits'/f'{key}_signal.json'),
            spectral_components=variant)
        candidate['state_response_study']['stage']=variant
        candidate['status']='Conditional acquisition-segment calibration; internal native checks, not blind validation or physical parameter identification.'
        jobs.append(create_job(key+'_'+variant,candidate))
        rngb=np.random.default_rng(610031500+run+100*SEGMENTS.index(seg)+B+10*ef);stability=[]
        for ib in range(128):
            dc=block_counts(a['ids'],tuple(a['span']),B,rngb)*train
            tt=moments(a,p,source,dc)
            if tt is None:continue
            Ab,tb,_=design(tt,C[:nc],F[:nc],base);vb,_=nnls(Ab,tb);stability.append(vb.tolist())
        report[variant]={'component_powers':v,'raw_linear_loss_squared':loss**2,
            'combined_initial_strength':initial,'final_strength':strength,
            'predicted_cov':base['cov']+strength*c,'predicted_F':base['F']+strength*f,
            'predicted_SD_ratio':np.sqrt(np.diag(base['cov']+strength*c)/np.diag(target['cov'])),
            'predicted_F_ratio':(base['F']+strength*f)/target['F'],
            'final_prediction_loss':np.sum((strength*unit-t)**2),'unit_combined_cov':c,'unit_combined_F':f,
            'kernel_compression_cov_relative_error':np.linalg.norm(initial*c-np.einsum('k,kij->ij',v,C[:nc]))/max(np.linalg.norm(np.einsum('k,kij->ij',v,C[:nc])),1e-20),
            'training_block_power_refits':stability}
    report.update(scope=__doc__,source=source,context=key,code_sha256=sha(__file__),
        calibration_ids=a['ids'][train],evaluation_ids=a['ids'][~train],target=target,base=base,
        component_cov=C,component_F=F,components=p['signal_noise_components'],
        limitations='Linearized fixed-anchor covariance and16 noise-replica F proposal; compare actual own-anchor ADC. Event variance CV inherited. Training-block refits hold MC, electronics fits and selected RC fixed. RC selection stability reported separately; no full-fit uncertainty claimed.')
    save(OUT/'fits'/f'{key}_signal.json',serialize(report))
    print(key,'powers',np.round(v,4),'strength',round(strength,4),flush=True)
    return jobs


def main():
    assert not (OUT/'signal_calibration.json').exists()
    p=read(OUT/'protocol.json');cal=read(OUT/'mean_calibration.json');record={}
    for source,run in SOURCES:
        jobs=[]
        for key,entry in cal['groups'].items():
            if entry['source']!=source:continue
            jobs+=fit_one(key,entry,p)
            record[key]={'fit':relative(OUT/'fits'/f'{key}_signal.json'),'fit_sha256':sha(OUT/'fits'/f'{key}_signal.json')}
        execute_jobs(source,jobs,'signal_candidates')
    save(OUT/'signal_calibration.json',{'scope':__doc__,'code_sha256':sha(__file__),
        'protocol_sha256':sha(OUT/'protocol.json'),'groups':record,'new_independent_mothers':0,'default_changed':False})


if __name__=='__main__':main()
