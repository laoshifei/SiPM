import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from check_frozen_psd_transfer import selected_blocks, frozen_scores, distance
from refine_waveforms import advanced_features


class TransferTests(unittest.TestCase):
    def test_new_blocks_equal_original_feature_extraction(self):
        rng=np.random.default_rng(9);waves=3000+rng.normal(size=(3,2496))
        starts=np.array([400.2,401.,405.9]);baseline=np.full(3,3000.)
        previous=advanced_features(waves,starts,np.full(3,500),np.full(3,200),baseline,3.)
        actual=selected_blocks(waves,baseline,previous['anchor_sample'],76)
        np.testing.assert_array_equal(actual,previous['block_charge'][:,:76])

    def test_scores_are_scale_invariant_without_retraining(self):
        pca={'blocks':2,'center':[.2,.3],'vector':[1.,-1.],'score_scale':.5}
        data={'block_charge':np.array([[2.,3.],[4.,1.]]),'long_charge':np.full((2,4),10.),'short_charge':np.full((2,3),4.),'mean_time_us':np.ones((2,4))}
        a=frozen_scores(data,pca)
        scaled={k:(v*7 if k!='mean_time_us' else v) for k,v in data.items()}
        b=frozen_scores(scaled,pca)
        for key in a:np.testing.assert_allclose(a[key],b[key],atol=1e-15)
        np.testing.assert_allclose(a['pca_projection'],[0.,.8])

    def test_empirical_ks_handles_ties(self):
        self.assertEqual(distance([1,1,2],[1,1,2]),0)
        self.assertEqual(distance([0,0],[1,1]),1)


if __name__=='__main__':unittest.main(verbosity=2)
