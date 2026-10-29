"""Controls for time partition, spectral contrast and response identifiability."""
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from analyze_bias_response import acquisition_segments, evenly_spaced, pulse_spectra
from profile_bias_response import kernel_grid, profile
from profile_signal_response import variance_integrals


class BiasResponseChecks(unittest.TestCase):
    def test_segments_use_original_indices_before_selection(self):
        segment=acquisition_segments(4776)
        np.testing.assert_array_equal(np.bincount(segment),[2865,955,956])
        ids=np.array([5,40,2864,2865,3819,3820,4775])
        np.testing.assert_array_equal(segment[ids],[0,0,0,1,1,2,2])
        sampled=evenly_spaced(ids,4)
        self.assertEqual(sampled[0],5);self.assertEqual(sampled[-1],4775)
        self.assertTrue(np.all(np.diff(sampled)>0))

    def test_spectral_band_recovers_known_sinusoidal_variance(self):
        n=20;t=np.arange(2496);anchor=400
        signal=np.zeros((n,len(t)))
        signal[:,anchor:]=200+10*np.sin(2*np.pi*20*t[anchor:]/256)
        f,pre,excess,means,bands=pulse_spectra(3000+signal,np.full(n,3000.),np.full(n,anchor))
        self.assertEqual(f[-1],250.);self.assertTrue(np.all(pre==0))
        np.testing.assert_allclose(means,200.,atol=1e-12)
        # 39.0625 MHz lies inside20-80 MHz; sine variance50 ADC^2 / mean200 ADC.
        np.testing.assert_allclose(bands[:,:,1],.25,rtol=.003)
        self.assertLess(abs(bands[:,:,2]).max(),1e-7)

    def test_stationary_noise_subtraction_has_zero_expected_excess(self):
        rng=np.random.default_rng(283225);n=600
        waves=3000+rng.normal(0,3,(n,2496));waves[:,400:]+=200
        _,_,_,_,bands=pulse_spectra(waves,np.full(n,3000.),np.full(n,400.))
        np.testing.assert_allclose(bands.mean(axis=(0,1)),0.,atol=.0005)

    def test_matrix_integral_matches_direct_quadrature(self):
        d=np.array([5.,29.,80.]);r=np.array([0.,12.,94.]);grid=kernel_grid(d,r)
        for i,decay in enumerate(d):
            np.testing.assert_allclose(grid[i],variance_integrals(r,decay=decay,bins=8000),rtol=2e-12,atol=1e-15)

    def test_pole_allocation_is_exactly_degenerate(self):
        values=np.array([3.,29.,94.,150.]);grid=kernel_grid(values,values,bins=3000)
        np.testing.assert_allclose(grid,grid.transpose(1,0,2),rtol=2e-12,atol=1e-15)

    def test_free_area_recovers_signal_scale_without_identifying_gain(self):
        grid=kernel_grid([8.,29.,70.],[0.,20.,94.],bins=3000)
        obs=75*grid[1,1]
        area,prediction,loss=profile(obs,grid)
        self.assertAlmostEqual(area[1,1],75.,places=10)
        self.assertLess(loss[1,1],1e-25)
        scaled,_,loss_scaled=profile(obs*4,grid)
        np.testing.assert_allclose(scaled,area*4,rtol=1e-13)
        np.testing.assert_allclose(loss_scaled,loss,atol=1e-14)


if __name__=='__main__':unittest.main(verbosity=2)
