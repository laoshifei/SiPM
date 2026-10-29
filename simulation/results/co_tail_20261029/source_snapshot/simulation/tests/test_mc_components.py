"""Exact component checks including correlated electronics and quantization."""
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from diagnose_mc_components import decompose
from diagnose_fluctuations import fluctuation_metrics


class ComponentsChecks(unittest.TestCase):
    def test_correlated_noise_and_quantization_preserve_variance_identity(self):
        rng=np.random.default_rng(283331);t=np.arange(2496);start=np.full(12,400.)
        shape=np.where(t>=400,np.exp(-(t-400)/450.)-np.exp(-(t-400)/60.),0.)
        signal=rng.uniform(300,600,(12,1))*shape
        electronics=rng.normal(0,3,signal.shape)-.25*signal
        analog=3000+signal+electronics;adc=np.floor(analog);quant=adc-analog
        baseline=adc[:,:128].mean(axis=1);full=fluctuation_metrics(adc,baseline,start)
        components=decompose(signal,electronics,quant,start,full[:,:,0])
        np.testing.assert_allclose(components.sum(axis=2),full[:,:,3],atol=2e-12,rtol=2e-12)
        self.assertLess(components[:,:,3].mean(),0.)

    def test_noiseless_case_has_only_signal_component(self):
        t=np.arange(2496);start=np.array([400.,410.]);signal=np.array([np.where(t>=s,400*np.exp(-(t-s)/400.),0.) for s in start])
        metrics=fluctuation_metrics(signal,np.zeros(2),start)
        components=decompose(signal,np.zeros_like(signal),np.zeros_like(signal),start,metrics[:,:,0])
        np.testing.assert_allclose(components[:,:,0],metrics[:,:,3],atol=1e-14)
        np.testing.assert_array_equal(components[:,:,1:],0.)


if __name__=='__main__':unittest.main(verbosity=2)
