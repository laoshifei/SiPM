#!/usr/bin/env python3
"""Explore shape-core separation, without converting source runs into truth labels."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time

import numpy as np

from prepare import HERE, OUT, save_csv, save_json
from refine_waveforms import PHASE2, LONG_ENDS, confirmed
from shape_mixture import diagnostics, fit_mixture, fit_weights

RUNS = [3, 12, 18]


def load_data():
    labels = confirmed()
    for run in RUNS:
        if run not in labels or float(labels[run]["bias_voltage_V"]) != 53.2:
            raise ValueError("Benchmark requires confirmed runs 3, 12, 18 at 53.2 V")
    edges = np.linspace(*json.loads((OUT / "confirmed_cohort.json").read_text())["common_charge_range_adc_samples"], 7)
    result = {}
    for run in RUNS:
        with np.load(PHASE2 / f"run{run:04d}_features.npz") as file:
            result[run] = {key: file[key] for key in file.files}
        with np.load(OUT / f"pha_run{run:04d}_features.npz") as file:
            q = file["charge_long"]
            result[run]["pilot_ratio"] = file["short_long_ratio"]
        result[run]["charge_bin"] = np.clip(np.searchsorted(edges, q, side="right") - 1, 0, 5)
    return result, edges


def training_pca(data, window):
    """Unlabelled PCA with equal weight per run and charge bin, training indices only."""
    blocks = int((LONG_ENDS[window] + 20) // 20)
    arrays, weights, reference = [], [], []
    cells = []
    for run in [3, 12]:
        d = data[run]
        for b in range(6):
            mask = d["eligible_primary"] & d["common_charge"] & (d["split"] == 0) & (d["charge_bin"] == b)
            n = int(mask.sum())
            if n < 20:
                raise ValueError(f"Insufficient PCA training cell: run {run}, bin {b}: {n}")
            arrays.append(d["block_charge"][mask, :blocks].astype(float) / d["long_charge"][mask, window, None])
            weights.append(np.full(n, 1 / (12 * n)))
            reference.append(d["pilot_ratio"][mask])
            cells.append({"run": run, "bin": b, "events": n})
    matrix = np.vstack(arrays)
    w = np.concatenate(weights)
    ref = np.concatenate(reference)
    mean = (matrix * w[:, None]).sum(axis=0)
    centered = matrix - mean
    covariance = (centered * w[:, None]).T @ centered
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    vector = eigenvectors[:, -1]
    projection = centered @ vector
    if float(np.sum(w * projection * (ref - np.sum(w * ref)))) < 0:
        vector = -vector
        projection = -projection
    scale = float(np.sqrt(np.sum(w * projection**2)))
    if scale <= 0:
        raise ValueError("Degenerate PCA training variance")
    return {"blocks": blocks, "center": mean.tolist(), "vector": vector.tolist(),
            "score_scale": scale, "explained_variance_fraction": float(eigenvalues[-1] / eigenvalues.sum()),
            "training_cells": cells,
            "training_policy": "First 60% of runs 3 and 12; equal run/bin weight; no event-level particle labels",
            "sign_policy": "Positive correlation with training QDC ratio; orientation only"}


def make_scores(data):
    scores, projections = {}, []
    timing = []
    for window, end in enumerate(LONG_ENDS):
        pca = training_pca(data, window)
        projections.append({"end_samples": int(end), **pca})
        mean, vector = np.asarray(pca["center"]), np.asarray(pca["vector"])
        for method in ["qdc", "negative_mean_time", "pca_projection"]:
            key = f"{method}_{end * 2}ns"
            scores[key] = {}
            started = time.perf_counter()
            for run in RUNS:
                d = data[run]
                if method == "qdc":
                    value = np.divide(d["short_charge"][:, 2], d["long_charge"][:, window],
                                      out=np.full(len(d["split"]), np.nan), where=d["long_charge"][:, window] > 0)
                elif method == "negative_mean_time":
                    value = -d["mean_time_us"][:, window]
                else:
                    matrix = d["block_charge"][:, :pca["blocks"]].astype(float)
                    q = d["long_charge"][:, window, None]
                    matrix = np.divide(matrix, q, out=np.full_like(matrix, np.nan), where=q > 0)
                    value = ((matrix - mean) @ vector) / pca["score_scale"]
                scores[key][run] = value
            timing.append({"configuration": key, "input_window_samples": int(end + 20),
                           "pca_blocks": pca["blocks"] if method == "pca_projection" else 0,
                           "elapsed_score_seconds": time.perf_counter() - started,
                           "events_scored": sum(len(d["split"]) for d in data.values()),
                           "scope": "One NumPy score pass after feature extraction; excludes I/O, preprocessing and training; not online throughput"})
    save_json(PHASE2 / "pca_models.json", projections)
    save_csv(PHASE2 / "score_cost.csv", timing)
    return scores


def training_arrays(data, scores, b, quality="primary"):
    values, groups, indices = [], [], []
    for run in [3, 12]:
        d = data[run]
        if quality == "basic":
            eligible = d["basic_eligible"] & d["comparison_support"]
        elif quality == "primary_no_secondary":
            eligible = d["eligible_primary"] & ~d["secondary_rise_flag"]
        else:
            eligible = d[f"eligible_{quality}"]
        mask = eligible & d["common_charge"] & (d["split"] == 0) & (d["charge_bin"] == b) & np.isfinite(scores[run])
        values.append(scores[run][mask])
        groups.append(np.full(mask.sum(), run, dtype=np.int32))
        indices.append(d["event_index"][mask])
    return np.concatenate(values), np.concatenate(groups), np.concatenate(indices)


def block_bootstrap(x, group, indices, model, draws, seed):
    """Conditional fit uncertainty, resampling 64-event acquisition blocks per source."""
    rng = np.random.default_rng(seed)
    blocks = {}
    for run in np.unique(group):
        # Empty blocks are retained within the occupied acquisition span.
        count = int(indices[group == run].max()) // 64 + 1
        blocks[int(run)] = [np.flatnonzero((group == run) & (indices // 64 == b)) for b in range(count)]
    foms = []
    failures = 0
    for _ in range(draws):
        selected = []
        for run, collection in blocks.items():
            picked = rng.integers(0, len(collection), len(collection))
            selected.extend(collection[i] for i in picked)
        selected = np.concatenate(selected)
        try:
            if any(np.count_nonzero(group[selected] == run) < 20 for run in blocks):
                raise ValueError("Sparse bootstrap source")
            fit = fit_mixture(x[selected], group[selected], initial=model, max_iter=200, tolerance=1e-6)
            if not fit["converged"] or fit["at_sigma_floor"]:
                raise ValueError("Unstable bootstrap fit")
            foms.append(fit["shape_fom_fwhm"])
        except ValueError:
            failures += 1
    interval = np.quantile(foms, [0.025, 0.975]).tolist() if len(foms) >= max(20, 0.8 * draws) else [None, None]
    return {"draws": draws, "successful": len(foms), "failures": failures,
            "fom_interval_95": interval, "block_size_events": 64,
            "scope": "Exploratory within-model block-bootstrap interval; fixed PCA and fixed Cauchy nuisance shape; not particle efficiency uncertainty"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", type=int, default=128)
    args = parser.parse_args()
    data, edges = load_data()
    scores = make_scores(data)
    np.savez_compressed(PHASE2 / "benchmark_scores.npz",
                        **{f"{key}_run{run}": score for key, by_run in scores.items() for run, score in by_run.items()})
    models, rows, evaluations, plot_diagnostics = [], [], [], []
    for configuration, by_run in scores.items():
        for b in range(6):
            x, group, indices = training_arrays(data, by_run, b)
            if any(np.count_nonzero(group == run) < 25 for run in [3, 12]):
                raise ValueError(f"Insufficient mixture training observations: {configuration}, bin {b}")
            two = fit_mixture(x, group, cores=2)
            one = fit_mixture(x, group, cores=1)
            delta_bic = one["bic"] - two["bic"]
            bootstrap = {"fom_interval_95": [None, None], "successful": 0, "failures": 0}
            if configuration.endswith("_3000ns") and args.bootstrap:
                bootstrap = block_bootstrap(x, group, indices, two, args.bootstrap,
                                            20261027 + len(models))
            models.append({"configuration": configuration, "charge_bin": b,
                           "charge_low": float(edges[b]), "charge_high": float(edges[b+1]),
                           "two_core": two, "one_core": one, "delta_bic_one_minus_two": delta_bic,
                           "bootstrap": bootstrap})
            row = {"configuration": configuration, "charge_bin": b,
                   "charge_low": edges[b], "charge_high": edges[b+1],
                   "n_train_run3": int(np.count_nonzero(group == 3)), "n_train_run12": int(np.count_nonzero(group == 12)),
                   "mean_L": two["means"][0], "mean_H": two["means"][1],
                   "sigma_L": two["sigmas"][0], "sigma_H": two["sigmas"][1],
                   "shape_fom_fwhm": two["shape_fom_fwhm"], "fom_interval_low": bootstrap["fom_interval_95"][0],
                   "fom_interval_high": bootstrap["fom_interval_95"][1],
                   "delta_bic_one_minus_two": delta_bic, "converged": two["converged"],
                   "at_sigma_floor": two["at_sigma_floor"],
                   "bootstrap_successful": bootstrap["successful"], "bootstrap_failed": bootstrap["failures"]}
            rows.append(row)
            for run, split in [(3,0), (12,0), (3,1), (12,1), (3,2), (12,2), (18,3)]:
                d = data[run]
                mask = (d["eligible_primary"] & d["common_charge"] & (d["charge_bin"] == b)
                        & (d["split"] == split) & np.isfinite(by_run[run]))
                test = by_run[run][mask]
                if len(test) == 0:
                    continue
                fitted = fit_weights(test, two)
                diagnostic = diagnostics(test, two, fitted)
                frozen = None
                if run in two["groups"]:
                    frozen = diagnostics(test, two, np.asarray(two["weights"][two["groups"].index(run)]))
                evaluations.append({"configuration": configuration, "charge_bin": b, "run": run, "split": split,
                                    "events": len(test), "weight_L": fitted[0], "weight_H": fitted[1], "weight_background": fitted[2],
                                    "near_weight_boundary": bool(np.any(fitted[:2] * len(test) < 1)),
                                    "fixed_shapes_refitted_weights_TV": diagnostic["binned_total_variation"],
                                    "fixed_shapes_refitted_weights_mean_log_density": diagnostic["mean_log_density"],
                                    "frozen_train_weights_TV": frozen["binned_total_variation"] if frozen else None,
                                    "frozen_train_weights_mean_log_density": frozen["mean_log_density"] if frozen else None})
                if configuration.endswith("_3000ns"):
                    plot_diagnostics.append({"configuration": configuration, "charge_bin": b, "run": run, "split": split,
                                             "weights": fitted.tolist(), **diagnostic})
        print(f"Fitted {configuration}: six charge bins", flush=True)
    save_json(PHASE2 / "mixture_models.json", models)
    save_csv(PHASE2 / "method_comparison.csv", rows)
    save_csv(PHASE2 / "transfer_checks.csv", evaluations)
    save_json(PHASE2 / "fit_diagnostics.json", plot_diagnostics)

    sensitivity = []
    reference = scores["qdc_3000ns"]
    variants = [(mode, 1.0, reference) for mode in ["basic", "loose", "primary", "strict", "primary_no_secondary"]]
    variants += [("primary_bg_half", 0.5, reference), ("primary_bg_double", 2.0, reference)]
    alternate = {}
    for run, d in data.items():
        delta = d["recent_baseline_delta"]
        alternate[run] = (d["short_charge"][:, 2] - delta * 270) / (d["long_charge"][:, 2] - delta * 1520)
    variants.append(("primary_recent_baseline", 1.0, alternate))
    for variant, bg_scale, values in variants:
        quality = variant if variant in ("basic", "loose", "primary", "strict", "primary_no_secondary") else "primary"
        for b in range(6):
            x, group, _ = training_arrays(data, values, b, quality)
            fit = fit_mixture(x, group, background_scale_factor=bg_scale)
            sensitivity.append({"variant": variant, "charge_bin": b, "training_events": len(x),
                                "shape_fom_fwhm": fit["shape_fom_fwhm"], "mean_L": fit["means"][0],
                                "mean_H": fit["means"][1], "sigma_L": fit["sigmas"][0], "sigma_H": fit["sigmas"][1],
                                "converged": fit["converged"], "at_sigma_floor": fit["at_sigma_floor"]})
    save_csv(PHASE2 / "sensitivity_checks.csv", sensitivity)
    save_json(PHASE2 / "analysis_scope.json", {
        "trained_runs": [3,12], "external_check_run": 18,
        "components": "Statistical low-score L and high-score H shape cores, not particle truth labels",
        "uniform_comparison_support": "Same pretrigger-quality and complete 4.04 us record support for all window/method comparisons",
        "reference_charge": "Frozen first-phase 3.04 us charge range and six bins; not recomputed to improve separation",
        "FOM": "Gaussian core centroid separation / sum of FWHM; source-constrained exploratory model",
        "BIC_scope": "One versus two cores for the SAME observable and SAME events; not a ranking across observables",
        "background": "Fixed Cauchy nuisance density; it is not a measured physical background spectrum",
        "transfer_scope": "Shape parameters frozen; source weights fitted separately on check samples. Frozen training-weight diagnostics also supplied for runs 3/12.",
        "blindness": "All runs were explored in phase 1. Temporal splits and run18 checks are internal consistency checks, not untouched blind validation.",
        "bootstrap": "128 draws by default, 64-event blocks, conditional on learned PCA and background density",
        "uncomputed_claims": ["alpha acceptance", "gamma leakage", "true component purity", "keVee threshold", "temperature stability", "online throughput"],
        "labels_sha256": hashlib.sha256((HERE/'labels.csv').read_bytes()).hexdigest(),
        "protocol_sha256": hashlib.sha256((PHASE2/'protocol.json').read_bytes()).hexdigest()})


if __name__ == "__main__":
    main()
