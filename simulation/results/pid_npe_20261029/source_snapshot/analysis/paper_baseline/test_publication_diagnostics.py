"""Moment identities and paired-candidate bootstrap checks."""
import unittest
import numpy as np
from publication_covariance import weighted_parts, covariance_parts
from publication_mc_uncertainty import weighted_ks, wmean
from publication_transfer import distance


class DiagnosticChecks(unittest.TestCase):
    def test_covariance_reestimates_means(self):
        rng=np.random.default_rng(8)
        z=rng.normal(size=(25,2,3,20))+np.arange(20)[None,None,None,:]*np.arange(25)[:,None,None,None]/100
        z-=z.mean(axis=-1,keepdims=True)
        weights=rng.integers(0,4,25)
        raw,across=weighted_parts(z,weights)
        expanded=np.repeat(z,weights,axis=0)
        a,_,b,_=covariance_parts(expanded,lags=np.array([0]))
        np.testing.assert_allclose(raw,a[...,0],atol=1e-12)
        np.testing.assert_allclose(across,b[...,0],atol=1e-12)
        # Holding the original across-event mean fixed would not reproduce this identity.
        fixed=np.mean(wmean((z-z.mean(axis=0))**2,weights),axis=-1)*weights.sum()/(weights.sum()-1)
        self.assertGreater(np.max(abs(fixed-across)),1e-3)

    def test_covariance_common_structure_is_removed(self):
        t=np.linspace(-1,1,20)
        z=np.broadcast_to(t,(20,1,1,20)).copy()
        raw,across=weighted_parts(z,np.ones(20,int))
        self.assertGreater(float(raw[0,0]),.3)
        np.testing.assert_allclose(across,0,atol=1e-14)

    def test_weighted_ks_matches_expanded_tied_samples(self):
        x=np.array([0.,1.,1.,3.]);y=np.array([0.,.5,2.,3.])
        wx=np.array([1,0,3,2]);wy=np.array([2,3,0,1])
        self.assertAlmostEqual(weighted_ks(x,wx,y,wy),distance(np.repeat(x,wx),np.repeat(y,wy)))

    def test_shared_parent_draw_keeps_paired_constant_shift(self):
        selected=np.array([3,17,55]);signal=np.array([[1.],[2.],[4.]])
        rng=np.random.default_rng(3)
        for _ in range(50):
            w=np.bincount(rng.integers(100,size=100),minlength=100)[selected]
            if not w.sum():continue
            self.assertAlmostEqual(float((wmean(signal+3,w)-wmean(signal,w))[0]),3.)


if __name__=='__main__':unittest.main()
