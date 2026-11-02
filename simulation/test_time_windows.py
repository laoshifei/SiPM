"""Independent direct-filter and explicit-replica checks of temporal attribution."""
from time_window_common import *


def run():
    rng=np.random.default_rng(731103800);n=24;t10=rng.uniform(530,600,n);t=np.arange(2496)[None,:]
    y=600*np.exp(-np.maximum(t-t10[:,None],0)/700)*(t>=t10[:,None])+rng.normal(0,5,(n,2496))
    baseline=np.full(n,3000.);waves=y+baseline[:,None];mu0=np.full(n,250.);parts=decompose(waves,baseline,t10,mu0)
    direct=np.empty((n,3))
    for j,w in enumerate([25,51,101]):
        kernel=np.full(w,-1/w);kernel[w//2]+=1
        for i in range(n):
            filtered=np.convolve(y[i],kernel,mode='same');before=int(np.ceil(t10[i]-300));after=int(np.ceil(t10[i]+250))
            direct[i,j]=(filtered[after:after+1000].var(ddof=1)-filtered[before:before+200].var(ddof=1))/y[i,after:after+1000].mean()
    np.testing.assert_allclose(parts['T'].sum(2),direct,rtol=1e-9,atol=1e-9)
    a={**parts,'ids':np.arange(n),'qcalc':np.linspace(.1,.9,n),'active':np.ones(n,bool)}
    p={'q_edges':list(range(7)),'support_bins':{'test':[0]},'conditional_cuts':{'test':{'0':{'q_quartiles':[.2,.5,.8],'q2_target':[.3,.7]}}}}
    counts=np.vstack([np.ones(n,int),rng.integers(1,4,n),rng.integers(1,4,n)]);summary,_=moments(a,p,'test',counts)
    for d,c in enumerate(counts):
        ix=np.repeat(np.arange(n),c);h=(a['qcalc'][ix]>=.5).astype(int);weight=(np.array([.3,.7])/np.bincount(h,minlength=2))[h]
        for k in range(3):
            x=parts['T'][ix,k];m=weight@x;center=x-m;cov=center.T@(weight[:,None]*center)
            np.testing.assert_allclose(cov,summary['cov'][d,k],rtol=1e-10,atol=1e-10)
            f=direct[ix,k];vf=weight@((f-weight@f)**2);np.testing.assert_allclose(cov.sum(),vf,rtol=1e-9,atol=1e-10)
    z=contrast(summary,summary);np.testing.assert_allclose(z[:,:,:2],1,rtol=1e-12,atol=1e-12)
    return {'direct_filter_waveforms':n,'scales':3,'explicit_replica_draws':3,'exact_sum_and_covariance_checks':True,'code_sha256':sha(__file__)}


if __name__=='__main__':print(run())
