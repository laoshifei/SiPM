"""Check charge-lattice recovery, explicit overflow and weighted scale algebra."""
from charge_scale_common import *


def run():
    rng=np.random.default_rng(2026110211)
    adc=rng.integers(0,16384,size=(100,2048));base=adc[:,:128].mean(1)
    gate=adc[:,256:1776]-base[:,None];profile=gate.reshape(100,76,20).mean(2)
    np.testing.assert_array_equal(charge_from_profile(profile),gate.sum(1))
    p={'q_edges':list(range(7)),'support_bins':{'test':[0,1]},'conditional_cuts':{'test':{
        str(b):{'q_quartiles':[b+.25,b+.5,b+.75],'q2_target':[.4,.6]} for b in [0,1]}}}
    np.testing.assert_array_equal(charge_bin(np.array([-.1,0.,.999,1.,5.999,6.,6.001]),p),[0,1,1,2,6,6,7])
    q=np.r_[np.arange(8)/10+.1,np.arange(8)/10+1.1]
    a={'q':q,'qcalc':q.copy(),'ids':np.arange(16),'fold':np.arange(16)%2,
       'X':rng.uniform(.2,2.,(16,9)),'mean_wave':np.ones((16,76))}
    ref={str(f):{str(b):{'mean_F':[1.,2.,3.]} for b in [0,1]} for f in [0,1]}
    np.testing.assert_allclose(ROTATION@ROTATION.T,np.eye(3),atol=1e-15)
    c=cohort(a,p,'test',ref,'original_Q');count=np.tile([2,1,1,2,1,0,1,1],2);s=statistics(c,count)
    independent=[];trace=[]
    for b in [0,1]:
        ix=np.repeat(np.flatnonzero((q>=b)&(q<b+1)),count[(q>=b)&(q<b+1)])
        half=(q[ix]>=b+.5).astype(int);num=np.bincount(half,minlength=2);w=np.array([.4,.6])[half]/num[half]
        z=a['X'][ix,3:6]/np.array([1.,2.,3.])-1
        Y=np.c_[a['X'][ix,:6],z@ROTATION.T]
        independent.append(np.cov(Y.T,aweights=w,ddof=0));trace.append(np.trace(np.cov(z.T,aweights=w,ddof=0)))
    np.testing.assert_allclose(s['cov'],np.mean(independent,0),rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(np.trace(s['cov'][6:,6:]),np.mean(trace),rtol=1e-12,atol=1e-12)
    moved={**a,'qcalc':q.copy()};moved['qcalc'][0]=-.1;moved['qcalc'][8]=2.1
    m=migration(moved,p,'test');assert m['underflow']==1 and m['leave_support']==2
    assert statistics(cohort({**a,'qcalc':np.full(16,-1.)},p,'test',ref,'current_Q')) is None
    assert np.isnan(interpolation(s,s,s)).all()
    return {'exact_integer_ADC_charge_examples':100,'boundary_cases':7,'orthonormal_rotation':True,
            'independent_weighted_covariance_cells':2,'trace_preserved':True,'missing_support_propagated':True,
            'degenerate_interpolation_is_unestimable':True}


if __name__=='__main__':print(run())
