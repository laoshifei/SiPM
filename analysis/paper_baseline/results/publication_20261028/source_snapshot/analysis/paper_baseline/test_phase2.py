#!/usr/bin/env python3
import unittest
import hashlib

import numpy as np

from compare_methods import training_pca
from refine_waveforms import PHASE2, advanced_features
from prepare import HERE, save_json
from shape_mixture import cdf, fit_mixture, fit_weights


class PhaseTwoChecks(unittest.TestCase):
    def features(self, waves, noise=1):
        n = len(waves)
        return advanced_features(waves, np.full(n, 399.1), np.full(n, 400),
                                 np.full(n, 100.), np.full(n, 1000.), noise)

    def test_exact_moments_and_block_integrals(self):
        wave = np.full((1, 2496), 1000.)
        wave[:, 400:2000] += 100
        result = self.features(wave)
        self.assertEqual(result["long_charge"][0, 2], 150000)
        self.assertEqual(result["short_charge"][0, 2], 25000)
        self.assertAlmostEqual(result["mean_time_us"][0, 2], 1.499)
        self.assertAlmostEqual(result["block_charge"][0, :76].sum(), 150000)
        self.assertFalse(result["pretrigger_flag_primary"][0])

    def test_pretrigger_pulse_and_colored_noise_reference(self):
        wave = np.full((2, 2496), 1000.)
        wave[:, 400:2000] += 100
        wave[0, 200:240] += 80
        wave[1, :400] += 10*np.sin(np.arange(400)*2*np.pi/128)
        result = self.features(wave, noise=7.1)
        self.assertTrue(result["pretrigger_flag_primary"][0])
        self.assertFalse(result["pretrigger_flag_primary"][1])
        self.assertTrue(result["pretrigger_flag_primary_v1"][1])

    def test_negative_tail_is_not_a_pretrigger_cut(self):
        wave = np.full((1, 2496), 1000.)
        wave[:, 400:2000] += 100
        wave[:, 2000:] -= 40
        result = self.features(wave)
        self.assertFalse(result["pretrigger_flag_primary"][0])

    def test_secondary_rise_remains_a_separate_diagnostic(self):
        wave = np.full((1, 2496), 1000.)
        wave[:, 400:600] += 100
        wave[:, 1000:1100] += 70
        result = self.features(wave)
        self.assertFalse(result["pretrigger_flag_primary"][0])
        self.assertTrue(result["secondary_rise_flag"][0])

    def test_shared_mixture_recovers_shapes_not_source_labels(self):
        rng = np.random.default_rng(1937)
        groups = np.repeat([3, 12], 1200)
        fractions = np.where(groups == 3, 0.2, 0.9)
        low = rng.random(len(groups)) < fractions
        x = np.where(low, rng.normal(-0.7, 0.12, len(groups)), rng.normal(0.9, 0.16, len(groups)))
        model = fit_mixture(x, groups)
        one = fit_mixture(x, groups, cores=1)
        self.assertTrue(model["converged"])
        self.assertGreater(one["bic"] - model["bic"], 200)
        self.assertLess(abs(model["means"][0] + 0.7), 0.03)
        self.assertLess(abs(model["means"][1] - 0.9), 0.03)
        self.assertLess(abs(model["weights"][0][0] - 0.2), 0.05)
        self.assertLess(abs(model["weights"][1][0] - 0.9), 0.05)
        self.assertGreaterEqual(model["history_min_step"], -1e-7)
        transfer = rng.normal(-0.7, 0.12, 1500)
        weights = fit_weights(transfer, model)
        self.assertGreater(weights[0], 0.97)
        np.testing.assert_allclose(cdf(np.array([-np.inf, np.inf]), model, weights), [0, 1], atol=1e-12)

    def test_pca_never_uses_validation_or_external_values(self):
        rng = np.random.default_rng(99)
        data = {}
        for run in [3, 12, 18]:
            blocks = np.exp(-np.arange(101)[None, :] / (20 if run == 3 else 40)) * np.ones((600, 1))
            blocks *= 1 + rng.normal(0, 0.02, blocks.shape)
            charges = np.column_stack([blocks[:, :n].sum(axis=1) for n in [26, 51, 76, 101]])
            split = np.where(np.arange(600) < 360, 0, 1)
            if run == 18:
                split[:] = 3
            data[run] = {"block_charge": blocks, "long_charge": charges, "eligible_primary": np.ones(600, bool),
                         "common_charge": np.ones(600, bool), "split": split,
                         "charge_bin": np.arange(600) % 6,
                         "pilot_ratio": np.full(600, 0.46 if run == 3 else 0.32)}
        original = training_pca(data, 2)
        for run in data:
            mask = data[run]["split"] != 0
            data[run]["block_charge"][mask] *= 10000
            data[run]["pilot_ratio"][mask] = -1e8
        changed = training_pca(data, 2)
        np.testing.assert_array_equal(original["vector"], changed["vector"])
        np.testing.assert_array_equal(original["center"], changed["center"])


if __name__ == "__main__":
    outcome = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(PhaseTwoChecks))
    save_json(PHASE2 / "unit_checks.json", {
        "tests_run": outcome.testsRun, "passed": outcome.wasSuccessful(),
        "failures": len(outcome.failures), "errors": len(outcome.errors),
        "code_sha256": {name: hashlib.sha256((HERE/name).read_bytes()).hexdigest()
                        for name in ["refine_waveforms.py", "shape_mixture.py", "compare_methods.py", "test_phase2.py"]}})
    if not outcome.wasSuccessful():
        raise SystemExit(1)
