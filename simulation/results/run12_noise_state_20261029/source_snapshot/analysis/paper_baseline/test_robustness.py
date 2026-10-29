"""Numerical checks for sufficient-statistic and common-support estimates."""
import unittest
import numpy as np
from robustness_full_cohort import add_moments,from_moments,coarsen_q,combine_blocks,focus_residuals
from publication_covariance import weighted_parts
from analyze_residual_covariance import extract_windows
from robustness_mc_charge import bin_ids,cell_sums,event_weights,matched_moments


class RobustnessChecks(unittest.TestCase):
    def test_grouped_moments_match_expanded_covariance(self):
        rng=np.random.default_rng(81);z=rng.normal(size=(100,2,200));z-=z.mean(axis=-1,keepdims=True)
        keys=rng.integers(0,4,100);n=np.zeros(4,int);total=np.zeros((4,2,200));square=np.zeros((4,2))
        for ids in np.array_split(np.arange(100),7):add_moments(n,total,square,keys[ids],z[ids])
        value=from_moments(n,total,square)
        for k in range(4):
            ix=keys==k;_,signal=weighted_parts(z[ix,0],np.ones(ix.sum(),int));_,pre=weighted_parts(z[ix,1],np.ones(ix.sum(),int))
            self.assertAlmostEqual(value[k],float(signal-pre),places=12)

    def test_fine_Q_merging_keeps_between_cell_means(self):
        rng=np.random.default_rng(82);keys=np.repeat(np.arange(24),10)
        z=rng.normal(size=(240,2,200))+np.arange(200)[None,None,:]*keys[:,None,None]/100
        z-=z.mean(axis=-1,keepdims=True)
        n=np.zeros(24,int);s=np.zeros((24,2,200));ss=np.zeros((24,2));add_moments(n,s,ss,keys,z)
        a,b,c=coarsen_q(n,s,ss,6)
        nn=np.zeros(6,int);bb=np.zeros((6,2,200));cc=np.zeros((6,2));add_moments(nn,bb,cc,keys//4,z)
        np.testing.assert_equal(a,nn);np.testing.assert_allclose(b,bb,atol=1e-12);np.testing.assert_allclose(c,cc,atol=1e-12)
        np.testing.assert_allclose(from_moments(a,b,c),from_moments(nn,bb,cc),atol=1e-12)

    def test_odd_number_of_blocks_preserves_partial_last(self):
        n=np.arange(2*2*5*6).reshape(2,2,5,6)
        d={'block_n':n,'block_total':np.repeat(n[...,None,None],400,axis=-1).reshape(2,2,5,6,2,200),
           'block_square':np.repeat(n[...,None],2,axis=-1)}
        a=combine_blocks(d,2)
        np.testing.assert_equal(a['block_n'][:,:,0],n[:,:,0]+n[:,:,1])
        np.testing.assert_equal(a['block_n'][:,:,2],n[:,:,4])
        np.testing.assert_equal(a['block_total'].sum(axis=2),d['block_total'].sum(axis=2))

    def test_focused_filter_matches_original_definition(self):
        rng=np.random.default_rng(83);t=np.arange(2496)[None,:]
        t10=np.array([430.2,501.,598.4]);baseline=np.array([30.,31.,32.])
        waves=baseline[:,None]+np.where(t>=t10[:,None],300*np.exp(-(t-t10[:,None])/800),0)+rng.normal(size=(3,2496))
        mu,res=extract_windows(waves,baseline,t10)
        expected=np.stack([res[:,0,2],res[:,0,0]],axis=1)/np.sqrt(mu[:,1])[:,None,None]
        np.testing.assert_array_equal(focus_residuals(waves,baseline,t10),expected)

    def test_Q_matching_removes_composition_only_difference(self):
        def data(counts):
            b=np.repeat([0,1],counts);f=(1+4*b)[:,None]*np.ones((1,3))
            return {'bins':b,'fluctuations':f,'shapes':f.copy()}
        measured=data([90,10]);simulation=data([10,90]);target=np.array([.5,.5]);bins=np.array([0,1])
        self.assertGreater(abs(measured['fluctuations'].mean()-simulation['fluctuations'].mean()),3.)
        values=[]
        for d in (measured,simulation):
            counts=np.ones(len(d['bins']));moments=cell_sums(d,counts)
            f,_=matched_moments(moments,bins,target);values.append(f)
            w=event_weights(d['bins'],counts,bins,target)
            np.testing.assert_allclose(np.bincount(d['bins'],weights=w),target)
            np.testing.assert_allclose(w@d['fluctuations'],f)
        np.testing.assert_array_equal(values[0],values[1])

    def test_empty_resampled_cell_is_not_redefined(self):
        n=np.array([3.,0.,0.,0.,0.,0.]);f=np.ones((6,3));s=np.ones((6,4))
        with self.assertRaises(ValueError):matched_moments((n,f,s),np.array([0,1]),np.array([.5,.5]))

    def test_charge_bin_endpoint_and_outside_guard(self):
        edges=np.arange(7.)
        np.testing.assert_array_equal(bin_ids(np.array([0.,1.,6.]),edges),[0,1,5])
        with self.assertRaises(ValueError):bin_ids(np.array([6.1]),edges)


if __name__=='__main__':unittest.main()
