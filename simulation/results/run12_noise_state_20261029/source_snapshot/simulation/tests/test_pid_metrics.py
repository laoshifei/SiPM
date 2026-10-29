"""Check ties, finite zero-error bounds, truth labels and weighted PID statistics."""
import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pid_metrics import auc,roc,binomial,gaussian_equivalent_fom
from extract_pid import label_truth


class PIDTests(unittest.TestCase):
    def test_ties_and_roc_area(self):
        a=np.array([0.,1.]);g=np.array([1.,2.]);r=roc(a,g)
        self.assertEqual(auc(a,g),.875)
        self.assertAlmostEqual(np.trapezoid(r['alpha_acceptance'],r['gamma_leakage']),.875)
        self.assertAlmostEqual(auc(a,g)+auc(g,a),1.)
        self.assertEqual(auc([1,1],[1,1]),.5)

    def test_zero_errors_have_positive_upper_limit(self):
        r=binomial(0,100)
        self.assertEqual(r['fraction'],0.)
        self.assertAlmostEqual(r['one_sided95_upper'],1-.05**.01)
        self.assertGreater(r['exact_two_sided95'][1],r['one_sided95_upper'])
        self.assertLess(binomial(100,100)['exact_two_sided95'][0],1.)
        for pair in [(0,0),(-1,10),(11,10),(1.2,10)]:
            with self.assertRaises(ValueError):binomial(*pair)

    def test_weighted_replicated_observations(self):
        a=np.array([0.,1.,2.]);g=np.array([1.,2.,4.]);wa=np.array([2,0,3]);wg=np.array([0,3,1])
        ar=np.repeat(a,wa);gr=np.repeat(g,wg)
        self.assertAlmostEqual(auc(a,g,wa,wg),auc(ar,gr))
        self.assertAlmostEqual(gaussian_equivalent_fom(a,g,wa,wg),gaussian_equivalent_fom(ar,gr))

    def test_fom_scale_and_translation(self):
        a=np.array([1.,2.,4.]);g=np.array([6.,7.,9.])
        self.assertAlmostEqual(gaussian_equivalent_fom(a,g),gaussian_equivalent_fom(8*a+31,8*g+31))

    def test_ancestry_dominance_and_zero_deposit(self):
        e=np.array([[0,9,1,0,0,0],[0,0,9,1,0,0],[0,0,0,10,0,0],
                    [0,5,5,0,0,0],[0,0,0,0,0,0]],float)
        labels,_=label_truth(e,.9)
        np.testing.assert_array_equal(labels,[0,1,2,3,4])
        strict,_=label_truth(e,.95)
        np.testing.assert_array_equal(strict,[3,3,2,3,4])


if __name__=='__main__':unittest.main(verbosity=2)
