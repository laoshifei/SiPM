"""Descriptive intensity conditioning of existing temporal fluctuation terms."""
from pathlib import Path
import numpy as np
from time_window_common import (PROJECT, OUT as TIME, SPLIT, SOURCES, SCALES, VARIANTS,
    TERMS, cells, load, read, save, sha, relative, serialize, interval, atomic_npz)

OUT = PROJECT / 'simulation/results/intensity_condition_20261102'
PREDICTORS = ['mean_500_1000', 'mean_1000_1500', 'mean_1500_2000', 'mean_2000_2500', 'whole_mean']
GLOBAL_NAMES = ['original_SD_ratio', 'residual_SD_ratio', 'crossfit_residual_SD_ratio',
    'variance_excess', 'linear_excess', 'residual_excess', 'data_linear_fraction', 'MC_linear_fraction',
    'data_crossfit_variance_fraction', 'MC_crossfit_variance_fraction', 'data_slope', 'MC_slope',
    'data_x_SD', 'MC_x_SD', 'data_F_mean', 'MC_F_mean']
LOCAL_NAMES = ['original_SD_ratio', 'residual_SD_ratio', 'crossfit_residual_SD_ratio',
    'variance_excess', 'linear_excess', 'residual_excess', 'data_linear_fraction', 'MC_linear_fraction',
    'data_crossfit_variance_fraction', 'MC_crossfit_variance_fraction', 'data_slope', 'MC_slope',
    'data_y_mean', 'MC_y_mean']


def design(a, ix):
    mu0 = a['mu_reference'][ix]
    x = np.c_[a['window_mu'][ix], a['mu'][ix]] / mu0[:, None]
    y = a['T'][ix]
    assert np.isfinite(x).all() and np.isfinite(y).all()
    return x, y


def raw_moments(w, x, y):
    d = len(w); s, j = y.shape[1:]; p = x.shape[1]
    return {'mass': w.sum(1), 'x': w @ x, 'xx': w @ (x*x),
        'y': (w @ y.reshape(len(y), s*j)).reshape(d,s,j),
        'yy': (w @ (y[:,:,:,None]*y[:,:,None,:]).reshape(len(y),s*j*j)).reshape(d,s,j,j),
        'xy': (w @ (x[:,:,None,None]*y[:,None,:,:]).reshape(len(y),p*s*j)).reshape(d,p,s,j)}


def centered(raw):
    x, y = raw['x'], raw['y']
    vx = raw['xx']-x*x
    c = raw['xy']-x[:,:,None,None]*y[:,None,:,:]
    cy = raw['yy']-y[:,:,:,None]*y[:,:,None,:]
    valid = np.all(vx > 1e-14,1)
    beta = np.divide(c, vx[:,:,None,None], out=np.full_like(c,np.nan), where=vx[:,:,None,None]>1e-14)
    intercept = y[:,None,:,:]-beta*x[:,:,None,None]
    linear = c[:,:,:,:,None]*beta[:,:,:,None,:]
    return {'mean':y,'x_mean':x,'x_var':vx,'cov':cy,'slope':beta,'intercept':intercept,
        'linear_cov':linear,'residual_cov':cy[:,None,:,:,:]-linear,'valid':valid}


def weights(counts, h, target, mask=None):
    c = counts if mask is None else counts*mask[None,:]
    number = np.stack([c[:,h==half].sum(1) for half in [0,1]],1)
    valid = (number.sum(1)>=5)&np.all((number>0)|(target==0),1)
    w = c*np.divide(target,number,out=np.zeros_like(number,dtype=float),where=number>0)[:,h]
    return w, valid


def crossfit_cov(raw_folds, fit_folds):
    """Apply each opposite-fold fit to evaluation raw moments, preserving full-Q weights."""
    mean = 0.; second = 0.
    for ef in [0,1]:
        r = raw_folds[ef]; f = fit_folds[1-ef]; b = f['slope']; a = f['intercept']
        ym = r['y'][:,None,:,:]; xy = r['xy']; xm = r['x'][:,:,None,None]
        mean = mean + ym-a*r['mass'][:,None,None,None]-b*xm
        second = second+r['yy'][:,None,:,:,:]
        second = second-a[:,:,:,:,None]*ym[:,:,:,None,:]-ym[:,:,:,:,None]*a[:,:,:,None,:]
        second = second-b[:,:,:,:,None]*xy[:,:,:,None,:]-xy[:,:,:,:,None]*b[:,:,:,None,:]
        second = second+r['mass'][:,None,None,None,None]*a[:,:,:,:,None]*a[:,:,:,None,:]
        second = second+xm[:,:,:,:,None]*(a[:,:,:,:,None]*b[:,:,:,None,:]+b[:,:,:,:,None]*a[:,:,:,None,:])
        second = second+r['xx'][:,:,None,None,None]*b[:,:,:,:,None]*b[:,:,:,None,:]
    return second-mean[:,:,:,:,None]*mean[:,:,:,None,:], mean


def summarize(a, p, source, counts, keep_cells=False):
    if counts.ndim==1:counts=counts[None,:]
    combined=[];details=[];valid=np.ones(len(counts),bool)
    for qbin, ix, h, target in cells(a,p,source):
        x,y=design(a,ix);w,ok=weights(counts[:,ix],h,target);raw=raw_moments(w,x,y);s=centered(raw)
        fit=[];rr=[]
        for ef in [0,1]:
            mask=a['fold'][ix]==ef;train_w,train_ok=weights(counts[:,ix],h,target,mask)
            f=centered(raw_moments(train_w,x,y));fit.append(f);rr.append(raw_moments(w*mask[None,:],x,y))
            ok &= train_ok & f['valid']
        cf,cfmean=crossfit_cov(rr,fit);ok &= s.pop('valid');valid &= ok
        s.pop('intercept');s['crossfit_cov']=cf;s['crossfit_mean']=cfmean
        for value in s.values():value[~ok]=np.nan
        combined.append(s)
        if keep_cells:
            details.append({'Q_bin':qbin,'events':len(ix),'fold_events':np.bincount(a['fold'][ix],minlength=2).tolist(),
                'point':{k:v[0] for k,v in s.items()},
                'fold_fits':[{k:f[k][0] for k in ['slope','intercept','x_mean','x_var']} for f in fit]})
    result={k:np.mean([s[k] for s in combined],0) for k in combined[0]}
    for v in result.values():v[~valid]=np.nan
    return result,details,valid


def contrast(mc, data):
    values=[];local=[]
    for p in range(len(PREDICTORS)):
        def global_terms(s):
            v=s['cov'].sum((-1,-2));fit=s['linear_cov'][:,p].sum((-1,-2))
            res=s['residual_cov'][:,p].sum((-1,-2));cf=s['crossfit_cov'][:,p].sum((-1,-2))
            return v,fit,res,cf
        vm,fm,rm,cm=global_terms(mc);vd,fd,rd,cd=global_terms(data)
        with np.errstate(invalid='ignore',divide='ignore'):
            z=np.stack([np.sqrt(vm/vd),np.sqrt(rm/rd),np.sqrt(cm/cd),(vm-vd)/vd,(fm-fd)/vd,(rm-rd)/vd,
                fd/vd,fm/vm,cd/vd,cm/vm,data['slope'][:,p].sum(2),mc['slope'][:,p].sum(2),
                np.broadcast_to(np.sqrt(data['x_var'][:,p,None]),vd.shape),np.broadcast_to(np.sqrt(mc['x_var'][:,p,None]),vd.shape),
                data['mean'].sum(2),mc['mean'].sum(2)],2)
        values.append(z)
        np.testing.assert_allclose(z[:,:,4]+z[:,:,5],z[:,:,3],rtol=1e-8,atol=1e-9)
        if p<4:
            def local_terms(s):
                return (s['cov'][:,:,p,p],s['linear_cov'][:,p,:,p,p],s['residual_cov'][:,p,:,p,p],s['crossfit_cov'][:,p,:,p,p])
            vm,fm,rm,cm=local_terms(mc);vd,fd,rd,cd=local_terms(data)
            with np.errstate(invalid='ignore',divide='ignore'):
                z=np.stack([np.sqrt(vm/vd),np.sqrt(rm/rd),np.sqrt(cm/cd),(vm-vd)/vd,(fm-fd)/vd,(rm-rd)/vd,
                    fd/vd,fm/vm,cd/vd,cm/vm,data['slope'][:,p,:,p],mc['slope'][:,p,:,p],data['mean'][:,:,p],mc['mean'][:,:,p]],2)
            local.append(z)
            np.testing.assert_allclose(z[:,:,4]+z[:,:,5],z[:,:,3],rtol=1e-8,atol=1e-9)
    return np.stack(values,1),np.stack(local,1)


def range_overlap(a, data, p, source):
    """Weighted point coverage; never used to filter or reweight the comparison."""
    result=[]
    dc={b:(ix,h,t) for b,ix,h,t in cells(data,p,source)}
    for b,ix,h,t in cells(a,p,source):
        di,dh,dt=dc[b];dx,_=design(data,di);mx,_=design(a,ix)
        dw,_=weights(np.ones((1,len(di))),dh,dt);mw,_=weights(np.ones((1,len(ix))),h,t)
        for j,name in enumerate(PREDICTORS):
            order=np.argsort(dx[:,j]);q=np.interp([.05,.95],np.cumsum(dw[0,order]),dx[order,j])
            result.append({'Q_bin':b,'predictor':name,'measured_5_95':q.tolist(),
                'MC_mass_inside':float(mw[0]@((mx[:,j]>=q[0])&(mx[:,j]<=q[1]))),
                'measured_mass_inside':float(dw[0]@((dx[:,j]>=q[0])&(dx[:,j]<=q[1])))})
    return result
