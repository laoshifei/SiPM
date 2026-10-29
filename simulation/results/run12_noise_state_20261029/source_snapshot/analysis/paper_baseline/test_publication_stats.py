"""Numerical checks for paired resampling and the publication cohort."""
import unittest
import numpy as np
import publication_stats as p


class StatisticsChecks(unittest.TestCase):
    def test_block_multiplicities_preserve_acquisition_gaps(self):
        ids = np.array([0, 1, 8, 9, 30, 31])
        a = p.block_counts(ids, (0, 32), 4, np.random.default_rng(42))
        rng = np.random.default_rng(42)
        expected = np.bincount(rng.integers(0, 8, size=8), minlength=8)
        np.testing.assert_equal(a, expected[ids//4])
        self.assertEqual(a[0], a[1])
        self.assertEqual(a[2], a[3])

    def test_pca_weighted_draw_equals_expanded_events(self):
        d = p.training(p.OUT)
        counts = np.random.default_rng(91).integers(0, 4, len(d['ids']))
        scores, model = p.pca_scores(d, 2, counts)
        repeated = np.repeat(np.arange(len(counts)), counts)
        expanded = p.subset(d, repeated)
        scores2, model2 = p.pca_scores(expanded, 2, np.ones(len(repeated), int))
        np.testing.assert_allclose(model['center'], model2['center'], rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(model['vector'], model2['vector'], atol=1e-11)
        np.testing.assert_allclose(scores[repeated], scores2, atol=1e-11)
        _, original = p.pca_scores(d, 2, np.ones(len(counts), int))
        self.assertGreater(np.linalg.norm(np.array(model['vector'])-original['vector']), 1e-5)

    def test_common_q_edges_do_not_clip_outside_events(self):
        np.testing.assert_equal(p.charge_bin(np.array([1., 2., 7.]), np.arange(1., 8.)), [0, 1, 5])
        with self.assertRaises(ValueError):
            p.charge_bin(np.array([.9]), np.arange(1., 8.))

    def test_refit_reestimates_background(self):
        rng = np.random.default_rng(3)
        x = np.r_[rng.normal(-1, .15, 80), rng.normal(1, .15, 80)]
        groups = np.repeat([3, 12], 80)
        a = p.fit_mixture(x, groups)
        b = p.fit_mixture(3*x+4, groups)
        self.assertAlmostEqual(b['background_center'], 3*a['background_center']+4)
        self.assertAlmostEqual(b['background_scale'], 3*a['background_scale'])
        self.assertAlmostEqual(b['shape_fom_fwhm'], a['shape_fom_fwhm'], places=6)

    def test_pretrigger_alternative_charge_integrals(self):
        d = p.training(p.OUT)
        alt = p.training(p.OUT, 'recent_baseline')
        for j, end in enumerate(p.ENDS):
            np.testing.assert_allclose(alt['long_charge'][:, j],
                d['long_charge'][:, j]-(end+20)*d['recent_baseline_delta'])
            np.testing.assert_allclose(alt['block_charge'][:, :(end+20)//20].sum(axis=1),
                alt['long_charge'][:, j], rtol=1e-6, atol=.1)


if __name__ == '__main__':
    unittest.main()
