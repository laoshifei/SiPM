"""Check the audit's selection boundaries and timestamp arithmetic."""
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diagnose_charge import hardware_flag, charge_bands, neighboring_intervals_us
from diagnose_fluctuations import fluctuation_metrics, WINDOWS
from prepare_response_test import native_rise, finite_rise_pdf


class ChargeAuditChecks(unittest.TestCase):
    def test_combined_bit_mask_retains_other_status_bits(self):
        np.testing.assert_array_equal(hardware_flag([0, 0x800, 0x200, 0x8000, 0x8800]), [False, False, False, True, True])

    def test_charge_partition_is_disjoint_and_finite(self):
        np.testing.assert_array_equal(charge_bands([0, -1, np.nan, np.inf, 1, 10, 20, 30, 31], [10, 20, 30]), [-1, -1, -1, -1, 0, 1, 2, 3, 3])
        with self.assertRaises(ValueError): charge_bands([1], [1, 1, 2])

    def test_intervals_preserve_large_timestamp_precision(self):
        t = np.array([2**60, 2**60+2000, 2**60+5000], dtype=np.uint64)
        before, after = neighboring_intervals_us(t)
        np.testing.assert_allclose(before[1:], [.002, .003]); np.testing.assert_allclose(after[:-1], [.002, .003])
        self.assertTrue(np.isnan(before[0]) and np.isnan(after[-1]))
        with self.assertRaises(ValueError): neighboring_intervals_us(t[::-1])

    def test_fluctuation_metric_offset_and_linear_trend(self):
        x = np.arange(2496, dtype=float)
        traces = np.array([20+x*.1, 20+x*.1+3000])
        metrics = fluctuation_metrics(traces, [20, 3020], [400, 400])
        np.testing.assert_allclose(metrics[0], metrics[1], atol=1e-10)
        self.assertLess(np.max(np.abs(metrics[:, :, 1:])), 1e-18)

    def test_poisson_plateau_variance_scale(self):
        rng = np.random.default_rng(281020)
        waves = rng.normal(0, 3, size=(100, 2496))
        waves[:, 400:] += rng.poisson(100, size=(100, 2096))
        metrics = fluctuation_metrics(waves, np.zeros(100), np.full(100, 400.))
        expected = 1.-1./np.array(WINDOWS)
        np.testing.assert_allclose(np.median(metrics[:, :, 3], axis=0), expected, rtol=.08)

    def test_native_geant4_rise_conversion_preserves_linear_kernel(self):
        rc=94.47173460998194;t=np.arange(0.,30000.,.5)
        for decay in (478.76494123873897,1372.5347670910692):
            rise=float(native_rise(decay,rc));actual=finite_rise_pdf(t,decay,rise)
            expected=(np.exp(-t/decay)-np.exp(-t/rc))/(decay-rc)
            np.testing.assert_allclose(actual,expected,atol=1e-17,rtol=1e-13)
            self.assertAlmostEqual(float(np.trapezoid(actual,t)),1.,places=6)
            self.assertAlmostEqual(float(np.trapezoid(t*actual,t)),decay+rc,places=4)
        for pole in (-1.,0.,480.,float('nan')):
            with self.assertRaises(ValueError):native_rise([480.,1370.],pole)


if __name__ == '__main__': unittest.main(verbosity=2)
