"""Statistical identities and finite-window controls for local covariance."""
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze_residual_covariance import covariance, covariance_parts, extract_windows, charge_bins, analyze, OFFSETS, N


class CovarianceTests(unittest.TestCase):
    def test_coherent_term_matches_distinct_event_pairs(self):
        rng = np.random.default_rng(44)
        x = rng.normal(size=(7, 31))+np.sin(np.arange(31)/3)
        x -= x.mean(axis=1, keepdims=True)
        lags = np.array([0, 1, 7])
        raw, coherent, across, event = covariance_parts(x, lags)
        expected = np.array([np.mean([np.mean(x[i,:31-k]*x[j,k:]) for i in range(7) for j in range(7) if i != j]) for k in lags])
        np.testing.assert_allclose(coherent, expected, atol=1e-14)
        np.testing.assert_allclose(raw, coherent+across, atol=1e-14)
        np.testing.assert_allclose(event.mean(axis=0), across)

    def test_identical_shapes_are_removed(self):
        x = np.tile(np.sin(np.arange(200)/8), (20,1))
        raw, coherent, across, _ = covariance_parts(x)
        self.assertGreater(raw[0], .4)
        np.testing.assert_allclose(raw, coherent, atol=1e-14)
        np.testing.assert_allclose(across, 0, atol=1e-28)

    def test_noise_covariance_preserved_after_mean_removal(self):
        rng = np.random.default_rng(7)
        noise = rng.normal(size=(3000,200))
        signal = noise + 10*np.sin(np.arange(200)/12)
        _, _, actual, _ = covariance_parts(signal)
        _, _, expected, _ = covariance_parts(noise)
        np.testing.assert_allclose(actual, expected, atol=1e-12)
        self.assertAlmostEqual(actual[0], 1-1/200, delta=.01)

    def test_integer_windows_and_constant_baseline(self):
        x = np.arange(2496, dtype=float)[None,:]*.1
        mu, r = extract_windows(x, [0], [400.3])
        mu2, r2 = extract_windows(x+1234, [1234], [400.3])
        np.testing.assert_allclose(mu, mu2, atol=1e-12)
        np.testing.assert_allclose(r, r2, atol=1e-10)
        np.testing.assert_allclose(r, 0, atol=1e-10)
        expected = .1*(401+OFFSETS[1:]+(N-1)/2)
        np.testing.assert_allclose(mu[0], expected)
        with self.assertRaises(ValueError): extract_windows(x, [0], [100])

    def test_upper_charge_edge_and_small_cells(self):
        np.testing.assert_array_equal(charge_bins([1,2,3],[1,2,3]), [0,1,1])
        with self.assertRaises(ValueError): charge_bins([.9],[1,2,3])
        x = np.tile(np.arange(2496, dtype=float)*.1, (9,1))
        result, _ = analyze(x, np.zeros(9), np.full(9,400.1), np.full(9,1.5), np.full(9,.32), np.arange(9), [1,2,3])
        self.assertEqual(result['views']['all']['n_used'], 0)
        self.assertEqual(result['views']['all']['cells'][0]['n'], 9)


if __name__ == '__main__': unittest.main()
