"""Exploratory fixed-dof Student-t core sensitivity, added after Gaussian points.

Same training IDs, Q cells and three methods. This tests core-shape assumptions;
it neither selects a preferred physical model nor relabels cores as particles.
"""
import json
import math
from pathlib import Path
import numpy as np
import publication_stats as p
from shape_mixture import diagnostics


def fit_t(x,group,nu):
    x=np.asarray(x,float);groups,gi=np.unique(group,return_inverse=True)
    q05,q95=np.quantile(x,[.05,.95]);width=max(q95-q05,np.std(x)*.05,1e-8)
    center=(q05+q95)/2;floor=width*.01
    background=1/(np.pi*width*(1+((x-center)/width)**2))
    logconstant=math.lgamma((nu+1)/2)-math.lgamma(nu/2)-.5*math.log(nu*np.pi)
    seeds=[np.sort([np.median(x[gi==j]) for j in range(2)]),np.quantile(x,[.2,.8]),np.quantile(x,[.35,.65])]
    best=None
    for seed in seeds:
        mu=seed.copy();scale=np.full(2,width/8);weights=np.array([[.475,.475,.05],[.475,.475,.05]])
        previous=-np.inf;converged=False;minimum=np.inf
        for iteration in range(600):
            z=(x[:,None]-mu)/scale
            density=np.exp(logconstant-(nu+1)/2*np.log1p(z*z/nu))/scale
            weighted=np.c_[density,background]*weights[gi]
            total=np.maximum(weighted.sum(axis=1),1e-300);ll=np.log(total).sum();responsibility=weighted/total[:,None]
            if iteration:minimum=min(minimum,ll-previous)
            if iteration and abs(ll-previous)/len(x)<1e-7:converged=True;break
            previous=ll
            u=(nu+1)/(nu+z*z);ru=responsibility[:,:2]*u
            mu=(ru*x[:,None]).sum(axis=0)/ru.sum(axis=0)
            scale=np.maximum(np.sqrt((ru*(x[:,None]-mu)**2).sum(axis=0)/responsibility[:,:2].sum(axis=0)),floor)
            order=np.argsort(mu);mu=mu[order];scale=scale[order]
            for j in range(2):
                weights[j]=np.maximum(responsibility[gi==j].mean(axis=0)[np.r_[order,2]],1e-12)
                weights[j]/=weights[j].sum()
        fwhm=2*scale*np.sqrt(nu*(2**(2/(nu+1))-1))
        model={'nu':nu,'means':mu.tolist(),'scales':scale.tolist(),'fwhm':fwhm.tolist(),
          'weights':weights.tolist(),'converged':converged,'at_scale_floor':bool(np.any(scale<=1.01*floor)),
          'iterations':iteration+1,'history_min_step':float(minimum),'log_likelihood':float(ll),
          'fom_fwhm':float((mu[1]-mu[0])/sum(fwhm))}
        if best is None or ll>best['log_likelihood']:best=model
    return best


def main(out=p.OUT):
    point=json.loads((out/'point.json').read_text());d=p.training(out);scores,_=p.pca_scores(d,2,np.ones(len(d['ids']),int))
    report={'scope':__doc__,'point_sha256':p.sha(out/'point.json'),'code_sha256':p.sha(Path(__file__)),
      'comparison':'Student-t has fixed dof5/10 and shared location/scale, separate source weights, same fixed-per-fit broad Cauchy. FOM uses each density actual FWHM; t scale is not a standard deviation. Point sensitivity only, added after Gaussian results. Cauchy width checks alone were uninformative because fitted nuisance weights were tiny.',
      'core_families':{},'gaussian_adequacy':[]}
    for nu in (5,10):
        foms=np.empty((3,6));fits=[]
        for m in range(3):
            for b in range(6):
                ix=d['charge_bin']==b;fit=fit_t(scores[ix,m],d['run'][ix],nu)
                if not fit['converged'] or fit['at_scale_floor']:raise ValueError('Student-t sensitivity fit failed')
                foms[m,b]=fit['fom_fwhm'];fits.append({'method':p.METHODS[m],'bin':b,**fit})
        report['core_families'][str(nu)]={'foms':foms.tolist(),'medians':np.median(foms,axis=1).tolist(),'fits':fits}
        print('Student-t dof',nu,'FWHM FOM medians',np.median(foms,axis=1),flush=True)
    for fit in point['windows']['3000']['fits']:
        m=p.METHODS.index(fit['method']);b=fit['bin']
        for j,run in enumerate(fit['groups']):
            ix=(d['charge_bin']==b)&(d['run']==run)
            diag=diagnostics(scores[ix,m],fit,fit['weights'][j])
            report['gaussian_adequacy'].append({'method':p.METHODS[m],'bin':b,'run':run,'n':int(ix.sum()),
                'binned_total_variation':diag['binned_total_variation'],'mean_log_density':diag['mean_log_density']})
    # Independent numerical check of the nearly-Gaussian limit and affine equivariance.
    rng=np.random.default_rng(92);x=np.r_[rng.normal(-1,.2,150),rng.normal(1,.2,150)];g=np.repeat([3,12],150)
    a=fit_t(x,g,5);b=fit_t(2*x+7,g,5)
    np.testing.assert_allclose(a['fom_fwhm'],b['fom_fwhm'],rtol=1e-9)
    gaussian=p.fit_mixture(x,g,max_iter=600);limit=fit_t(x,g,1000000)
    np.testing.assert_allclose(gaussian['shape_fom_fwhm'],limit['fom_fwhm'],rtol=1e-5)
    report['checks']={'affine_equivariance':True,'large_nu_gaussian_limit':True}
    p.save(out/'core_model_sensitivity.json',report)


if __name__=='__main__':main()
