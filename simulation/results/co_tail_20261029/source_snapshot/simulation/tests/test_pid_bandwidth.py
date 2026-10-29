"""Check correlated-noise covariance against independent random waveform draws."""
import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from diagnose_pid_bandwidth import linear_scores,weights
from fit_pid_bandwidth import projected_cov
from fit_signal_fluctuations import colored


class PIDBandwidthTest(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(101)
        v=rng.normal(size=76);v/=np.linalg.norm(v)
        self.pca={'blocks':76,'vector':v.tolist(),'score_scale':.07,'center':np.ones(76)/76}
        self.anchor=np.array([500])
        t=np.arange(2496)-500
        self.signal=(600*(1-np.exp(-np.maximum(t,0)/25))*np.exp(-np.maximum(t,0)/420))[None,:]
        self.signal[:,t<0]=0

    def test_exact_ratio_decomposition_including_cross_covariance(self):
        rng=np.random.default_rng(102);n=128
        s=np.repeat(self.signal,n,axis=0)*rng.uniform(.7,1.4,size=(n,1))
        noise=rng.normal(size=s.shape)*np.sqrt(s+1)*.5
        # Deliberately correlated components: diagonal variances alone must fail.
        parts=np.stack([s,noise,.7*noise],axis=1)
        parts-=parts[:,:,:128].mean(2)[:,:,None]
        a=np.full(n,500);total=parts.sum(1)
        score,q,_=linear_scores(total,a,np.zeros(n),self.pca)
        gates=parts[:,:,480:2000];w,c=weights(self.pca)
        r0=score.mean(0)
        u=((gates@w.T)-(r0+c)[None,None,:]*gates.sum(2)[:,:,None])/q[:,None,None]
        np.testing.assert_allclose(u.sum(1),score-r0,atol=1e-12)
        for j in range(3):
            covariance=np.cov(u[:,:,j],rowvar=False)
            self.assertAlmostEqual(float(covariance.sum()),float(score[:,j].var(ddof=1)),places=13)
            self.assertGreater(abs(covariance.sum()-np.trace(covariance)),.1*covariance.sum())

    def test_fir_covariance_matches_independent_noise_waveforms(self):
        score,q,_=linear_scores(self.signal,self.anchor,np.zeros(1),self.pca)
        h=np.exp(-np.arange(137)/23);h/=np.linalg.norm(h)
        predicted=projected_cov(self.signal,self.anchor,score,q,h,self.pca)[0]
        w,c=weights(self.pca)
        rng=np.random.default_rng(103);draws=[]
        for _ in range(64):
            z=colored(h,rng.normal(size=(128,2496+len(h)-1)))*np.sqrt(self.signal)
            gate=z[:,480:2000]
            draws.append((gate@w.T-gate.sum(1)[:,None]*(score+c))/q[:,None])
        observed=np.cov(np.concatenate(draws),rowvar=False)
        scale=np.sqrt(np.diag(predicted)[:,None]*np.diag(predicted)[None,:])
        self.assertLess(np.max(abs(observed-predicted)/scale),.05)


if __name__=='__main__':unittest.main()
