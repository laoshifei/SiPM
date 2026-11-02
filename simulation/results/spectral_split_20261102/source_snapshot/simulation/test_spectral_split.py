"""Independent matrix checks of complementary covariance and mixture moments."""
from spectral_split_common import *


def operators(n,d):
    center=np.zeros((n,n+2*d));high=np.zeros_like(center);aux=np.zeros_like(center)
    for i in range(n):
        center[i,i+d]=1;high[i,[i,i+d,i+2*d]]=[-.25,.5,-.25];aux[i,[i,i+2*d]]=[-.25,.25]
    return center-high,-aux,high,aux


def main():
    for d in DELAYS:
        L,V,H,G=operators(64,d);lc=L@L.T+V@V.T;hc=H@H.T+G@G.T
        np.testing.assert_array_equal(L@H.T+V@G.T,np.zeros((64,64)))
        np.testing.assert_array_equal(lc+hc,np.eye(64))
        expected=.5*np.eye(64)
        for i in range(64-d):expected[i,i+d]=expected[i+d,i]=-.25
        np.testing.assert_array_equal(hc,expected)
    n=32;d=4;L,V,H,G=operators(n,d);hc=H@H.T+G@G.T
    filters=[]
    for width in [3,7,11]:
        f=np.eye(n)
        for i in range(n):
            ix=np.arange(max(0,i-width//2),min(n,i+width//2+1));f[i,ix]-=1/len(ix)
        filters.append(f.T@(np.eye(n)-np.ones((n,n))/n)@f/(n-1))
    A=np.array(filters);mu=.5*np.sin(np.arange(n)/5);C=1.2*np.eye(n)
    def kernel(S):return np.array([[2*np.trace(a@S@b@S)+np.trace(a@S)*np.trace(b@S) for b in A] for a in A])
    base=np.array([[2*np.trace(a@C@b@C)+4*mu@a@C@b@mu for b in A] for a in A])
    common=.2;high_cv=.6;expected=base+common**2*kernel(np.eye(n))+(1+common**2)*high_cv**2*kernel(hc)
    rng=np.random.default_rng(731102901);q=[]
    for _ in range(40):
        w=rng.normal(size=(1000,n+2*d));v=rng.normal(size=w.shape)
        lo=w@L.T+v@V.T;hi=w@H.T+v@G.T
        def multiplier(cv):
            s2=np.log1p(cv*cv);return np.exp(np.sqrt(s2)*rng.normal(size=1000)-.5*s2)
        x=mu+rng.normal(size=(1000,n))*np.sqrt(.2)+np.sqrt(multiplier(common))[:,None]*(lo+np.sqrt(multiplier(high_cv))[:,None]*hi)
        q.append(np.einsum('ni,kij,nj->nk',x,A,x))
    q=np.concatenate(q);observed=np.cov(q,rowvar=False,bias=True)
    error=np.max(np.abs(observed-expected)/np.sqrt(np.diag(expected)[:,None]*np.diag(expected)[None,:]));assert error<.045,error
    mean=np.einsum('kii->k',A@C)+np.einsum('i,kij,j->k',mu,A,mu)
    assert np.max(abs(q.mean(0)-mean)/mean)<.012
    save(OUT/'algebra_tests.json',{'exact_complementary_operator_covariance':True,'mixture_trials':len(q),
        'joint_quadratic_covariance_relative_error':error,'observed_cov':observed.tolist(),'expected_cov':expected.tolist(),
        'code_sha256':sha(__file__),'fixed_mean_spectrum_test':True})
    print('Complementary covariance and full common/high Gaussian-mixture quadratic formula passed.',flush=True)


if __name__=='__main__':main()
