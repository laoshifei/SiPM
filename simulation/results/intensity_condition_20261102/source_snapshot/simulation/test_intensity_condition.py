"""Independent direct weighted residual checks for the conditioning algebra."""
import numpy as np
from intensity_condition_common import raw_moments, centered, crossfit_cov, weights


def run():
    rng=np.random.default_rng(731104011);n=120;x=rng.normal(size=(n,5));y=rng.normal(size=(n,3,7))+x[:,0,None,None]*rng.normal(size=(3,7))
    folds=np.tile([0,1],n//2);half=np.tile([0,0,1,1],n//4);target=np.array([.37,.63]);counts=np.vstack([np.ones(n,int),rng.integers(1,4,n),rng.integers(1,5,n)])
    w,ok=weights(counts,half,target);assert ok.all();s=centered(raw_moments(w,x,y));f=[];rr=[]
    for ef in [0,1]:
        mask=folds==ef;fw,valid=weights(counts,half,target,mask);assert valid.all()
        f.append(centered(raw_moments(fw,x,y)));rr.append(raw_moments(w*mask[None,:],x,y))
    cf,cfmean=crossfit_cov(rr,f)
    for d in range(3):
        for p in range(5):
            residual=y-s['mean'][d]-s['slope'][d,p]*(x[:,p]-s['x_mean'][d,p])[:,None,None]
            cov=np.einsum('n,nsi,nsj->sij',w[d],residual,residual)
            np.testing.assert_allclose(cov,s['residual_cov'][d,p],rtol=1e-11,atol=1e-11)
            pred=np.empty_like(y)
            for ef in [0,1]:
                mask=folds==ef;pred[mask]=f[1-ef]['intercept'][d,p]+f[1-ef]['slope'][d,p]*x[mask,p,None,None]
            r=y-pred;mean=np.einsum('n,nsi->si',w[d],r);z=r-mean
            cov=np.einsum('n,nsi,nsj->sij',w[d],z,z)
            np.testing.assert_allclose(cov,cf[d,p],rtol=1e-11,atol=1e-11)
            np.testing.assert_allclose(mean,cfmean[d,p],rtol=1e-11,atol=1e-11)
    np.testing.assert_allclose(s['linear_cov']+s['residual_cov'],s['cov'][:,None]+np.zeros_like(s['linear_cov']),rtol=1e-11,atol=1e-11)
    return {'synthetic_events':n,'multiplicity_patterns':3,'predictors':5,'direct_residual_covariance_checks':30,'passed':True}


if __name__=='__main__':print(run())
