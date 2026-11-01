"""Fixed-cohort diagnostics of PID quantile widths, tails and waveform covariance."""
from pathlib import Path
import numpy as np
from rc_refinement_common import (PROJECT,SOURCES,SEGMENTS,read,load,save,sha,relative,serialize,
    rows,event_weights,weighted_quantile)
from analyze_acquisition_state import bins_of,interval

PARENT=PROJECT/'simulation/results/rc_refinement_20261030'
ACQ=PROJECT/'simulation/results/acquisition_state_20261030'
OUT=PROJECT/'simulation/results/pid_core_tail_20261030'
VARIANTS=['frozen','coarse_spectrum','mean_only','inherited_CV_fit','spectrum_CV_fit']
METHODS=['QDC','negative_mean_time','PCA']
COVERAGES=[.50,.80,.90,.95]
FENCES=[2.,3.]
NORMAL_IQR=1.3489795003921634
AUX=['log_baseline_RMS','F50','F102','F202']


def measured(run,seg):
    z=load(ACQ/'arrays'/f'run{run}_{seg}.npz')
    a={k:z[k] for k in ['ids','q','shape','F','baseline','baseline_rms','t10','anchor','state','span','n_raw']}
    a['scores']=z['scores'][:,0];a['qcalc']=z['q_recomputed']
    a['mean_wave']=a['shape']*a['qcalc'][:,None]/20.
    return a


def native(run,seg,B,v):return load(PARENT/'arrays'/f'run{run}_{seg}_B{B}_{v}_evaluation.npz')


def fit_reference(a,p,source,B):
    result={};fold=(a['ids']//B)%2
    for ef in [0,1]:
        mask=fold!=ef;rr=rows(a,p,source,mask.astype(int));assert rr is not None
        centers={};scales={};edges={}
        for b,ix,w,_ in rr:
            q=np.array([weighted_quantile(a['scores'][ix,j],w,[.25,.5,.75]) for j in range(3)])
            centers[str(b)]=q[:,1];scales[str(b)]=(q[:,2]-q[:,0])/NORMAL_IQR
            assert np.all(scales[str(b)]>0)
            edges[str(b)]={str(k):np.c_[q[:,1]-k*scales[str(b)],q[:,1]+k*scales[str(b)]] for k in FENCES}
        result[str(ef)]={'training_ids':a['ids'][mask].tolist(),'evaluation_ids':a['ids'][~mask].tolist(),
            'median':centers,'robust_scale':scales,'fences':edges}
    return serialize(result)


def standardized(a,p,source,reference,fold):
    bins=bins_of(a,p);z=np.full_like(a['scores'],np.nan)
    for ef in [0,1]:
        r=reference[str(ef)]
        for b in p['support_bins'][source]:
            use=(bins==b)&(fold==ef)
            z[use]=(a['scores'][use]-np.array(r['median'][str(b)]))/np.array(r['robust_scale'][str(b)])
    assert np.isfinite(z[np.isin(bins,p['support_bins'][source])]).all()
    return z


def diagnostics(a,p,source,z,counts=None,full=False):
    rr=rows(a,p,source,counts)
    if rr is None:return None
    widths=[];covs=[];prob=[];variance_tail=[];joint=[];joint_cov=[];means=[]
    cross=[];cross_tail=[];auxcov=[];auxvar=[];details={}
    quantiles=np.array([[(1-c)/2,(1+c)/2] for c in COVERAGES]).ravel()
    for b,ix,w,_ in rr:
        x=a['scores'][ix];mu=w@x;r=x-mu;cov=r.T@(w[:,None]*r)
        q=np.array([weighted_quantile(x[:,j],w,quantiles) for j in range(3)]).reshape(3,len(COVERAGES),2)
        widths.append((q[:,:,1]-q[:,:,0]).T);means.append(mu);covs.append(cov)
        tail=np.abs(z[ix])[None,:,:]>np.array(FENCES)[:,None,None]
        prob.append(np.einsum('n,knj->kj',w,tail));variance_tail.append(np.einsum('n,nj,knj->kj',w,r*r,tail))
        jt=tail.any(2);joint.append(jt@w)
        joint_cov.append(np.array([r.T@((w*t)[:,None]*r) for t in jt]))
        if full:
            s=a['shape'][ix];s=s-w@s
            cross.append(r.T@(w[:,None]*s))
            cross_tail.append(np.array([(w*r[:,j]*tail[0,:,j])@s for j in range(3)]))
            aux=np.c_[np.log(a['baseline_rms'][ix]),a['F'][ix]];ar=aux-w@aux
            auxcov.append(r.T@(w[:,None]*ar));auxvar.append(w@(ar*ar))
            details[str(b)]={'events':len(ix),'effective_events':float(1/(w@w)),'mean':mu,'cov':cov,
                'widths':widths[-1],'tail_probability':prob[-1],'tail_variance_contribution':variance_tail[-1],
                'joint_tail_probability':joint[-1],'joint_tail_covariance_contribution':joint_cov[-1]}
    cov=np.mean(covs,axis=0);sd=np.sqrt(np.diag(cov));v=np.mean(variance_tail,axis=0)
    result={'mean':np.mean(means,axis=0),'widths':np.sqrt(np.mean(np.array(widths)**2,axis=0)),
        'cov':cov,'sd':sd,'tail_probability':np.mean(prob,axis=0),'tail_variance_share':v/(sd*sd),
        'joint_tail_probability':np.mean(joint,axis=0),'joint_tail_covariance':np.mean(joint_cov,axis=0)}
    if full:
        c=np.mean(cross,axis=0);ct=np.mean(cross_tail,axis=0);av=np.mean(auxvar,axis=0)
        result.update(per_Q=details,score_shape_covariance=c,tail_score_shape_covariance=ct,
            core_score_shape_covariance=c-ct,score_shape_per_SD=c/sd[:,None],tail_score_shape_per_SD=ct/sd[:,None],
            auxiliary_correlations=np.mean(auxcov,axis=0)/np.sqrt(sd[:,None]**2*av[None,:]))
        result['histogram']={'edges':np.linspace(-5,5,51),'density':[],'outside_range_mass':[]}
        ew=event_weights(a,p,source,counts)
        result['supported_events']=int(np.count_nonzero(ew));result['effective_events']=float(1/(ew@ew))
        result['tail_event_counts']=np.array([np.sum((np.abs(z[ew>0])>k),axis=0) for k in FENCES])
        for j in range(3):
            result['histogram']['density'].append(np.histogram(z[:,j],bins=result['histogram']['edges'],weights=ew)[0]/.2)
            result['histogram']['outside_range_mass'].append(float(ew[np.abs(z[:,j])>5].sum()))
    return result


NAMES=[f'width{int(c*100)}_{m}_ratio' for c in COVERAGES for m in METHODS]+[f'SD_{m}_ratio' for m in METHODS]+[
    f'core_over_SD_{m}_ratio' for m in METHODS]+[f'tail{k:g}_{m}_mass_delta_pp' for k in FENCES for m in METHODS]+[
    f'tail{k:g}_{m}_variance_share_delta_pp' for k in FENCES for m in METHODS]+[
    f'joint_tail{k:g}_mass_delta_pp' for k in FENCES]


def contrasts(m,d):
    return np.r_[(m['widths']/d['widths']).ravel(),m['sd']/d['sd'],
        (m['widths'][0]/m['sd'])/(d['widths'][0]/d['sd']),
        100*(m['tail_probability']-d['tail_probability']).ravel(),
        100*(m['tail_variance_share']-d['tail_variance_share']).ravel(),
        100*(m['joint_tail_probability']-d['joint_tail_probability'])]


def truth_parts(a,p,source,z,truth):
    rr=rows(a,p,source);ew=event_weights(a,p,source);labels=truth['label_energy90'];q=bins_of(a,p)
    total=np.zeros((3,3));between=np.zeros((3,3));within=np.zeros((3,3));count={}
    for b,ix,w,_ in rr:
        x=a['scores'][ix];mu=w@x;r=x-mu;total+=r.T@(w[:,None]*r)
        for label in range(5):
            take=labels[ix]==label;mass=w[take].sum()
            if mass==0:continue
            mean=(w[take]@x[take])/mass;delta=mean-mu;rrr=x[take]-mean
            between+=mass*np.outer(delta,delta);within+=rrr.T@(w[take,None]*rrr)
    total/=len(rr);between/=len(rr);within/=len(rr)
    np.testing.assert_allclose(total,between+within,rtol=1e-11,atol=1e-15)
    joint=(np.abs(z)>2).any(1)
    for label,name in enumerate(['alpha_family','gamma_family','decay_electron_family','mixed_other','no_deposit']):
        use=(labels==label)&(ew>0);mass=ew[use].sum()
        count[name]={'events':int(use.sum()),'weighted_fraction':float(mass),
            'joint_tail_events':int(np.sum(use&joint)),'joint_tail_weight':float(ew[use&joint].sum()),
            'tail_probability_given_class':float(ew[use&joint].sum()/mass) if mass else None,
            'Q_bins_present':np.unique(q[use]).tolist()}
    return {'counts':count,'covariance_total':total,'between_truth_mean_covariance':between,
        'within_truth_covariance':within,'between_truth_mean_variance_fraction':np.diag(between)/np.diag(total),
        'label_sensitivity':{key:{'counts':np.bincount(truth[key][ew>0],minlength=5),
            'weighted_fractions':np.bincount(truth[key],weights=ew,minlength=5)} for key in ['label_energy90','label_energy95','label_charge90']},
        'scope':'Labels describe stored simulated deposition/avalanche ancestry only; never experimental particle truth. No class removal or new selection.'}
