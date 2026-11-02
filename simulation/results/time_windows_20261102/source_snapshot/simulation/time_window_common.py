"""Exact time-resolved decomposition of the frozen fluctuation diagnostics."""
import numpy as np
from spectral_split_common import (PROJECT,OUT as SPLIT,REPLAY,SOURCES,EVENTS,read,load,save,sha,relative,serialize,interval)
from charge_scale_common import charge_bin
from state_response_common import F_metrics

OUT=PROJECT/'simulation/results/time_windows_20261102'
QDC=PROJECT/'simulation/results/qdc_fluctuation_20261101'
LOW=PROJECT/'simulation/results/lowfreq_taper_20261102'
SCALES=[50,102,202]
TERMS=['window_500_1000','window_1000_1500','window_1500_2000','window_2000_2500',
       'between_window_means','negative_pretrigger','normalization']
VARIANTS=['split_r0','split_r1']
NAMES=['global_SD_ratio','fixed_denominator_SD_ratio','global_variance_excess']+[
    f'local_mean_ratio_w{i}' for i in range(4)]+[f'local_SD_ratio_w{i}' for i in range(4)]+[
    'excess_'+t for t in TERMS]+['quarter_diagonal_excess','quarter_cross_covariance_excess','other_covariance_excess']


def decompose(waves,baseline,t10,mu_reference):
    y=waves.astype(float)-baseline[:,None];c=np.c_[np.zeros(len(y)),y.cumsum(1)]
    pre=np.ceil(t10-300).astype(int)[:,None]+np.arange(200)
    post=np.ceil(t10+250).astype(int)[:,None]+np.arange(1000);ix=np.arange(len(y))[:,None]
    window_mu=y[ix,post].reshape(len(y),4,250).mean(2);mu=window_mu.mean(1)
    assert np.all(mu>0) and np.all(mu_reference>0)
    terms=[];local=[]
    for width in [25,51,101]:
        half=width//2;r=y[:,half:-half]-(c[:,width:]-c[:,:-width])/width
        z=r[ix,post-half].reshape(len(y),4,250);means=z.mean(2);within=z.var(2,ddof=1)
        between=250*np.sum((means-means.mean(1)[:,None])**2,1)/999
        before=r[ix,pre-half].var(1,ddof=1)
        parts=np.c_[249*within/999,between,-before]
        post_var=r[ix,post-half].var(1,ddof=1)
        np.testing.assert_allclose(parts[:,:5].sum(1),post_var,rtol=2e-12,atol=2e-10)
        numerator=parts.sum(1);fixed=parts/mu_reference[:,None]
        norm=numerator*(1/mu-1/mu_reference)
        terms.append(np.c_[fixed,norm])
        local.append(np.divide(within-before[:,None],window_mu,out=np.full_like(within,np.nan),where=window_mu>0))
    T=np.stack(terms,1);F=F_metrics(waves,baseline,t10)[:,:,3]
    np.testing.assert_allclose(T.sum(2),F,rtol=2e-10,atol=2e-10)
    return {'T':T,'local_F':np.stack(local,1),'mu':mu,'window_mu':window_mu,'F':F,'mu_reference':mu_reference}


def cells(a,p,source):
    bins=charge_bin(a['qcalc'],p)-1;out=[]
    for b in p['support_bins'][source]:
        ix=np.flatnonzero(a['active']&(bins==b));cut=p['conditional_cuts'][source][str(b)]
        out.append((b,ix,(a['qcalc'][ix]>=cut['q_quartiles'][1]).astype(int),np.asarray(cut['q2_target'])))
    return out


def moments(a,p,source,counts):
    """Original within-Q weighted moments, all draws together; no event removal."""
    if counts.ndim==1:counts=counts[None,:]
    D=len(counts);result=[];valid=np.ones(D,bool);details=[]
    for b,ix,h,target in cells(a,p,source):
        number=np.stack([counts[:,ix][:,h==j].sum(1) for j in [0,1]],1)
        ok=(number.sum(1)>=5)&np.all((number>0)|(target==0),1);valid&=ok
        w=counts[:,ix]*np.divide(target,number,out=np.zeros_like(number,dtype=float),where=number>0)[:,h]
        t=a['T'][ix];assert np.isfinite(t).all()
        mean=(w@t.reshape(len(ix),21)).reshape(D,3,7)
        outer=(t[:,:,:,None]*t[:,:,None,:]).reshape(len(ix),147)
        cov=(w@outer).reshape(D,3,7,7)-mean[:,:,:,None]*mean[:,:,None,:]
        local=a['local_F'][ix];lmean=(w@local.reshape(len(ix),12)).reshape(D,3,4)
        lc=(w@(local[:,:,:,None]*local[:,:,None,:]).reshape(len(ix),48)).reshape(D,3,4,4)-lmean[:,:,:,None]*lmean[:,:,None,:]
        record={'mean':mean,'cov':cov,'local_mean':lmean,'local_cov':lc,'window_mu':w@a['window_mu'][ix]}
        for x in record.values():x[~ok]=np.nan
        result.append(record);details.append({'Q_bin':b,'events':len(ix),'half_events':np.bincount(h,minlength=2).tolist(),'local_invalid':int(np.count_nonzero(~np.isfinite(local)))})
    averaged={k:np.mean([r[k] for r in result],0) for k in result[0]}
    for value in averaged.values():value[~valid]=np.nan
    return averaged,details


def contrast(m,d):
    cm,cd=m['cov'],d['cov'];vm=cm.sum((-1,-2));vd=cd.sum((-1,-2));dm=np.diagonal(cm,axis1=-2,axis2=-1)[:,:,:4].sum(2)
    dd=np.diagonal(cd,axis1=-2,axis2=-1)[:,:,:4].sum(2);xc=cm[:,:,:4,:4].sum((-1,-2))-dm;xd=cd[:,:,:4,:4].sum((-1,-2))-dd
    with np.errstate(invalid='ignore',divide='ignore'):
        diagonal=(dm-dd)/vd;cross=(xc-xd)/vd;excess=(vm-vd)/vd
        values=np.concatenate([np.stack([np.sqrt(vm/vd),np.sqrt(cm[:,:,:6,:6].sum((-1,-2))/cd[:,:,:6,:6].sum((-1,-2))),excess],2),
            m['local_mean']/d['local_mean'],np.sqrt(np.diagonal(m['local_cov'],axis1=-2,axis2=-1)/np.diagonal(d['local_cov'],axis1=-2,axis2=-1)),
            (cm.sum(3)-cd.sum(3))/vd[:,:,None],np.stack([diagonal,cross,excess-diagonal-cross],2)],2)
    assert values.shape[2]==len(NAMES)
    np.testing.assert_allclose(values[:,:,11:18].sum(2),values[:,:,2],rtol=1e-9,atol=1e-10)
    np.testing.assert_allclose(values[:,:,18:21].sum(2),values[:,:,2],rtol=1e-9,atol=1e-10)
    return values


def atomic_npz(path,**arrays):
    assert not path.exists();temporary=path.with_suffix('.pending.npz');np.savez_compressed(temporary,**arrays);temporary.replace(path)
