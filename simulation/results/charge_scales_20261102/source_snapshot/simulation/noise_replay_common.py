"""Selection audit and repeated downstream noise on fixed Geant4 mothers."""
import numpy as np
from qdc_fluctuation_common import (PROJECT, OUT as QDC, PARENT, WAVE, RC, SOURCES,
    SEGMENTS, VARIANTS, BATCHES, EVENTS, assemble, pool, mother_counts,
    read, load, save, sha, relative, serialize, data_waves)
from state_response_common import features, F_metrics
from prepare_acquisition_state import DATA, WIDTH
from analyze_acquisition_state import rows, event_weights, weighted_quantile, interval

OUT = PROJECT/'simulation/results/noise_replay_20261102'
BUILD = PROJECT/'simulation/build/noise_replay_20261102'
PYTHON = '/home/xc/software/jupyter/.venv/bin/python'
FEATURES = ['QDC', 'negative_mean_time_us', 'PCA', 'F50', 'F102', 'F202',
            'fixed_F50', 'fixed_F102', 'fixed_F202']
METRICS = ['QDC_mean_delta', 'mean_time_delta_ns']+[f'F{s}_mean_ratio' for s in [50,102,202]]+[
    f'{name}_SD_ratio' for name in FEATURES]+[f'{name}_width{level}_ratio' for level in [50,95] for name in FEATURES[:3]]
CASES = ['all_selected', 'inherited', 'matched_band', 'fixed_MC_band']
REPLAY_CASES = {'current_all': ('all_readout',16,False), 'current_signal': ('signal_only',8,False),
                'current_electronics': ('electronics_only',8,False), 'zero_cv_all': ('all_readout',16,True)}


def band(scores, run):
    return scores[:,0] >= .4 if run == 3 else scores[:,0] < .4


def mu_reference(a, p, source, ref):
    bins = np.clip(np.searchsorted(p['q_edges'],a['q'],'right')-1,0,5)
    mu = np.ones(len(bins))
    for fold in [0,1]:
        for b in p['support_bins'][source]:
            mu[(bins==b)&(a['fold']==fold)] = ref[str(fold)]['Q_bins'][str(b)]['pulse_mean']
    return mu


def compact(raw, p, source, ref):
    detail = F_metrics(raw['waves'],raw['baseline'],raw['t10'])
    out = {k:raw[k] for k in ['ids','q','fold','n_parent','batch','span','n_raw'] if k in raw}
    out['mu_reference'] = mu_reference(raw,p,source,ref)
    fixed = (detail[:,:,2]-detail[:,:,1])/out['mu_reference'][:,None]
    out['X'] = np.c_[raw['scores'],detail[:,:,3],fixed]
    out['mu'] = detail[:,0,0]
    out['mean_wave'] = raw['mean_wave']
    out['t10'] = raw['t10'];out['baseline'] = raw['baseline'];out['qcalc'] = raw['qcalc']
    return out


def old_compact(a):
    return {**{k:a[k] for k in ['ids','q','fold','n_parent','batch','span','n_raw'] if k in a},
            'X':np.c_[a['scores'],a['F'],(a['detail'][:,:,2]-a['detail'][:,:,1])/a['mu_reference'][:,None]],
            'mu_reference':a['mu_reference']}


def summary(a,p,source,counts=None,mask=None):
    if mask is not None: counts = mask.astype(int) if counts is None else counts*mask
    rr = rows(a,p,source,counts)
    if rr is None: return None
    out = []
    for b,ix,w,_ in rr:
        x = a['X'][ix];m=w@x;r=x-m;c=r.T@(w[:,None]*r)
        quant = np.array([weighted_quantile(x[:,j],w,[.025,.25,.75,.975]) for j in range(3)])
        out.append({'Q_bin':b,'mean':m,'cov':c,'widths':np.stack([quant[:,2]-quant[:,1],quant[:,3]-quant[:,0]]),
                    'events':len(ix),'ESS':float(1/(w@w)), 'invalid_feature_events':np.sum(~np.isfinite(x),0)})
    return {'mean':np.mean([x['mean'] for x in out],0),'cov':np.mean([x['cov'] for x in out],0),
            'widths':np.sqrt(np.mean([x['widths']**2 for x in out],0)), 'per_Q':out}


def contrast(m,d):
    if m is None or d is None:return np.full(len(METRICS),np.nan)
    with np.errstate(invalid='ignore',divide='ignore'):
        x=[m['mean'][0]-d['mean'][0],-1000*(m['mean'][1]-d['mean'][1])]
        x+=list(m['mean'][3:6]/d['mean'][3:6])
        x+=list(np.sqrt(np.diag(m['cov'])/np.diag(d['cov'])))
        x+=list((m['widths']/d['widths']).ravel())
    assert len(x)==len(METRICS)
    return np.asarray(x)


def nested_moments(X,w):
    """Fixed-mother mixture moments and finite-repeat correction; no clipping.

    X is mother x repeat x feature; weights sum to one over original mothers.
    Between-mean sampling correction is sum w_i(1-w_i) S_i/R. Whole mothers,
    not flattened readout replicas, are the inferential sampling units.
    """
    n,R,k=X.shape;assert R>=2 and np.isclose(w.sum(),1.)
    means=X.mean(1);res=X-means[:,None,:]
    within=np.einsum('nri,nrj->nij',res,res)/(R-1)
    m=w@means;center=means-m
    between_raw=center.T@(w[:,None]*center)
    W=np.einsum('n,nij->ij',w,within)
    correction=np.einsum('n,nij->ij',w*(1-w)/R,within)
    B=between_raw-correction
    W_one=np.einsum('n,nij->ij',w*(1-w),within)
    one_repeat=[]
    for r in range(R):
        y=X[:,r];dy=y-w@y;one_repeat.append(dy.T@(w[:,None]*dy))
    flat=X-m[None,None,:]
    flat_cov=np.einsum('n,nri,nrj->ij',w,flat,flat)/R
    np.testing.assert_allclose(flat_cov,between_raw+(R-1)/R*W,rtol=1e-10,atol=1e-10)
    return {'mean':m,'within':W,'between_raw':between_raw,'between_correction':correction,
            'between_corrected':B,'mixture_covariance':B+W,'one_readout_within':W_one,
            'one_readout_expected_covariance':B+W_one,'one_readout_empirical_covariance':np.mean(one_repeat,0),
            'flat_covariance':flat_cov,'repeat_count':R,'mothers':n}


def nested_summary(a,X,p,source,counts=None,mask=None):
    if mask is not None: counts=mask.astype(int) if counts is None else counts*mask
    rr=rows(a,p,source,counts)
    if rr is None:return None
    values=[]
    for b,ix,w,_ in rr:
        # Repeated mother IDs in a bootstrap are one fixed conditional mean;
        # coalesce their weights before finite-repeat corrections.
        unique,inverse=np.unique(ix,return_inverse=True);weights=np.bincount(inverse,weights=w)
        values.append({'Q_bin':b,**nested_moments(X[unique],weights)})
    names=['mean','within','between_raw','between_correction','between_corrected','mixture_covariance',
           'one_readout_within','one_readout_expected_covariance','one_readout_empirical_covariance','flat_covariance']
    return {**{k:np.mean([r[k] for r in values],0) for k in names},'per_Q':values}
