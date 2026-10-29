"""Numerical controls for baseline projection, spectrum units and sign selection."""
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from characterize_readout import detrend, covariance, periodogram, candidate_indices
from prepare_noise_model import make_kernel
from profile_signal_response import variance_integrals


class ReadoutChecks(unittest.TestCase):
    def test_offset_and_slope_do_not_change_noise_statistics(self):
        rng=np.random.default_rng(281030); y=rng.normal(0,3,(100,256))
        shifted=y+3000+np.arange(256)*.2
        np.testing.assert_allclose(detrend(y),detrend(shifted),atol=2e-12)
        np.testing.assert_allclose(covariance(y),covariance(shifted),atol=2e-12)

    def test_periodogram_parseval_and_units(self):
        rng=np.random.default_rng(281031); y=rng.normal(0,3,(100,256)); f,p=periodogram(y)
        w=np.hanning(256); expected=((detrend(y)*w)**2).sum(axis=1)/(w@w)
        np.testing.assert_allclose(p.sum(axis=1)*(f[1]-f[0]),expected,atol=1e-13)
        self.assertAlmostEqual(f[-1],250.)

    def test_positive_and_negative_control_have_exact_sign_symmetry(self):
        rng=np.random.default_rng(281032); y=rng.normal(0,3,(400,256))
        y[::3,90:105]+=12
        positive=candidate_indices(y,2.,4.,1);negative=candidate_indices(-y,2.,4.,-1)
        for a,b in zip(positive,negative):np.testing.assert_array_equal(a,b)
        self.assertGreater(len(positive[0]),100)

    def test_white_noise_projection_bias_is_accounted_for(self):
        rng=np.random.default_rng(281033); c=covariance(rng.normal(0,3,(3000,256)))
        # Two fitted coefficients remove two degrees of freedom from each trace.
        self.assertAlmostEqual(c[:,0].mean(),9*(254/256),delta=.06)
        self.assertLess(abs(c[:,1].mean()),.12)

    def test_learned_noise_model_predicts_independent_colored_traces(self):
        rng=np.random.default_rng(281034)
        white=rng.normal(size=(2000,257));waves=3000+3*(white[:,:256]+white[:,1:])/np.sqrt(2)
        kernel,sigma,_,_,_=make_kernel(np.floor(waves))
        white=rng.normal(size=(2000,256+len(kernel)-1))
        # FFT valid convolution uses independent simulated Gaussian samples.
        size=1024;y=np.fft.irfft(np.fft.rfft(white,size,axis=1)*np.fft.rfft(kernel,size),size,axis=1)[:,len(kernel)-1:len(kernel)-1+256]
        c=covariance(np.floor(3000+sigma*y)).mean(axis=0)
        self.assertAlmostEqual(c[0],covariance(np.floor(waves)).mean(axis=0)[0],delta=.25)
        self.assertAlmostEqual(c[1]/c[0],.5,delta=.04)

    def test_shot_noise_frequency_integral_converges(self):
        a=variance_integrals([0.,30.,94.],bins=8000)
        b=variance_integrals([0.,30.,94.],bins=32000)
        np.testing.assert_allclose(a,b,rtol=2e-4,atol=1e-9)


if __name__ == '__main__':unittest.main(verbosity=2)
