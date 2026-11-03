"""Charge reassignment and joint fluctuation-scale diagnostics on fixed cohorts."""
from pathlib import Path
import numpy as np
from noise_replay_common import (PROJECT,OUT as REPLAY,PARENT,QDC,SOURCES,SEGMENTS,EVENTS,
    read,load,save,sha,relative,serialize,rows,weighted_quantile,interval)

OUT=PROJECT/'simulation/results/charge_scales_20261102'
CONDITIONS=['original_Q','retained_original_Q','current_Q']
VARIANTS=['current','zero_cv']
FEATURES=['QDC','negative_mean_time_us','PCA','F50','F102','F202','F_common','F_short_long','F_curvature']
ROTATION=np.array([[1,1,1],[1,0,-1],[1,-2,1]])/np.sqrt(np.array([3,2,6]))[:,None]
METRICS=['QDC_mean_delta','mean_time_delta_ns']+[f'F{s}_mean_ratio' for s in [50,102,202]]+[
    f'{n}_SD_ratio' for n in FEATURES]+[f'{n}_width{q}_ratio' for q in [50,95] for n in FEATURES[:3]]+[
    'F50_F102_corr_delta','F50_F202_corr_delta','F102_F202_corr_delta',
    'F_covariance_relative_error','mean_ADC_relative_L2','PID_covariance_relative_error']
INHERITED_PAIRS=[(i,i) for i in [*range(11),*range(14,20)]]+[(24,20),(25,21)]


def charge_bin(q,p):
    """8 categories: underflow, 6 original bins, overflow. Last edge inclusive."""
    edges=np.asarray(p['q_edges']);b=np.searchsorted(edges,q,side='right')
    b=np.where(q==edges[-1],6,b);return b


def charge_from_profile(profile):
    # 1520 integer ADC samples minus 1520*(sum of 128 pretrigger ADCs)/128
    # lies exactly on an eighth-ADC lattice. Restore that exact value before
    # comparing to half-bin thresholds, avoiding floating summation tie changes.
    q=np.sum(profile*20,axis=-1);exact=np.rint(q*8)/8
    assert np.all(np.abs(q-exact)<1e-7),'Profile cannot reconstruct the original integer-ADC gate charge'
    return exact


def reference(a,p,source):
    result={}
    for fold in [0,1]:
        rr=rows(a,p,source,(a['fold']!=fold).astype(int));assert rr is not None
        result[str(fold)]={}
        for b,ix,w,_ in rr:
            m=w@a['X'][ix,3:6];assert np.all(m>0) and np.isfinite(m).all()
            result[str(fold)][str(b)]={'mean_F':m,'training_ids':a['ids'][ix],
                'training_half_targets':p['conditional_cuts'][source][str(b)]['q2_target']}
    return result


def cohort(a,p,source,ref,condition):
    old=a['q'];current=a.get('qcalc',charge_from_profile(a['mean_wave']))
    qb=charge_bin(current,p);supported=np.isin(qb,np.array(p['support_bins'][source])+1)
    q=current if condition=='current_Q' else old
    allowed=np.ones(len(q),bool) if condition=='original_Q' else supported
    bins=charge_bin(q,p)-1;Y=np.full((len(q),len(FEATURES)),np.nan);Y[:,:6]=a['X'][:,:6]
    for fold in [0,1]:
        for b in p['support_bins'][source]:
            take=(a['fold']==fold)&(bins==b)
            scale=np.asarray(ref[str(fold)][str(b)]['mean_F'])
            z=a['X'][take,3:6]/scale-1
            Y[take,6:]=z@ROTATION.T
    cells=[]
    for b in p['support_bins'][source]:
        ix=np.flatnonzero(allowed&(bins==b));cut=p['conditional_cuts'][source][str(b)]
        h=(q[ix]>=cut['q_quartiles'][1]).astype(int);target=np.asarray(cut['q2_target'])
        cells.append((b,ix,h,target))
    return {'Y':Y,'mean_wave':a['mean_wave'],'cells':cells,'qcalc':current,'q':q,
        'ids':a['ids'],'source':source,'condition':condition}


def statistics(a,counts=None,keep_Q=False):
    result=[]
    for b,ix,half,target in a['cells']:
        if counts is not None:
            h=np.repeat(half,counts[ix]);ix=np.repeat(ix,counts[ix])
        else:h=half
        number=np.bincount(h,minlength=2)
        if len(ix)<5 or np.any((target>0)&(number==0)):return None
        w=(target/np.where(number>0,number,1))[h];assert np.isclose(w.sum(),1.)
        x=a['Y'][ix]
        if not np.isfinite(x).all():return None
        m=w@x;dx=x-m;cov=dx.T@(w[:,None]*dx)
        q=np.array([weighted_quantile(x[:,j],w,[.025,.25,.75,.975]) for j in range(3)])
        width2=np.stack([q[:,2]-q[:,1],q[:,3]-q[:,0]])**2
        result.append({'Q_bin':b,'mean':m,'cov':cov,'width2':width2,'profile':w@a['mean_wave'][ix],
                       'events':len(ix),'half_counts':number,'ESS':1/(w@w)})
    s={k:np.mean([r[k] for r in result],axis=0) for k in ['mean','cov','width2','profile']}
    if keep_Q:s['per_Q']=result
    return s


def average_statistics(values):
    # Every predeclared repeat and original support bin is required.
    if any(x is None for x in values):return None
    return {k:np.mean([x[k] for x in values],axis=0) for k in ['mean','cov','width2','profile']}


def correlation(C):
    sd=np.sqrt(np.diag(C));return C/(sd[:,None]*sd[None,:])


def contrasts(m,d):
    if m is None or d is None:return np.full(len(METRICS),np.nan)
    mc,dc=m['cov'],d['cov'];sd=np.sqrt(np.diag(dc));delta=(mc-dc)/(sd[:,None]*sd[None,:])
    cm=correlation(mc[3:6,3:6]);cd=correlation(dc[3:6,3:6])
    with np.errstate(divide='ignore',invalid='ignore'):
        z=np.r_[m['mean'][0]-d['mean'][0],-1000*(m['mean'][1]-d['mean'][1]),
            m['mean'][3:6]/d['mean'][3:6],np.sqrt(np.diag(mc)/np.diag(dc)),
            np.sqrt(m['width2']/d['width2']).ravel(),(cm-cd)[np.triu_indices(3,1)],
            np.sqrt(np.mean(delta[3:6,3:6]**2)),np.linalg.norm(m['profile']-d['profile'])/np.linalg.norm(d['profile']),
            np.sqrt(np.mean(delta[:3,:3]**2))]
    assert len(z)==len(METRICS);return z


def interpolation(data,current,zero):
    if any(x is None for x in [data,current,zero]):return np.full(6,np.nan)
    d,c,z=[np.diag(s['cov'])[3:6] for s in [data,current,zero]];den=c-z
    lam=np.divide(d-z,den,out=np.full(3,np.nan),where=np.abs(den)>1e-12*np.maximum(np.abs(c),np.abs(z)))
    # This is an algebraic moment-line fraction, not a physical CV estimate.
    return np.r_[lam,lam[0]-lam[2],lam[0]-lam[1],lam[1]-lam[2]]


def migration(a,p,source):
    old=charge_bin(a['q'],p);new=charge_bin(a.get('qcalc',charge_from_profile(a['mean_wave'])),p)
    assert np.all((old>=1)&(old<=6))
    counts=np.zeros((8,8),int);np.add.at(counts,(old,new),1)
    support=np.array(p['support_bins'][source])+1;o=np.isin(old,support);n=np.isin(new,support)
    return {'matrix':counts,'old_support':int(o.sum()),'current_support':int(n.sum()),
            'leave_support':int(np.sum(o&~n)),'enter_support':int(np.sum(~o&n)),
            'bin_changed':int(np.sum(old!=new)),'underflow':int(np.sum(new==0)),'overflow':int(np.sum(new==7)),
            'total_original_selected':len(old),'q_shift_quantiles':np.quantile(a.get('qcalc',charge_from_profile(a['mean_wave']))-a['q'],[0,.05,.5,.95,1])}


def intervals(x,names):return {n:interval(x[:,j]) for j,n in enumerate(names)}


LAMBDA_NAMES=['lambda_F50','lambda_F102','lambda_F202','lambda_F50_minus_F202','lambda_F50_minus_F102','lambda_F102_minus_F202']
