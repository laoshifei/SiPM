#!/usr/bin/env python3
"""Time-order diagnostics and conditional sampling references for shape transfer."""
from __future__ import annotations

import csv
import json
import hashlib

import numpy as np

from compare_methods import load_data
from prepare import HERE, OUT, save_csv, save_json
from refine_waveforms import PHASE2, confirmed
from shape_mixture import diagnostics, fit_mixture


def main():
    time_rows, changes = [], []
    for run in sorted(confirmed()):
        with np.load(OUT / f"pha_run{run:04d}_features.npz") as file:
            first = {key: file[key] for key in ["baseline_rms", "baseline", "amplitude", "short_long_ratio", "timestamp"]}
        with np.load(PHASE2 / f"run{run:04d}_features.npz") as file:
            second = {key: file[key] for key in ["event_index", "eligible_primary", "basic_eligible", "common_charge", "pretrigger_span", "secondary_rise_flag"]}
        n = len(second["event_index"])
        for block in range(20):
            begin, end = n*block//20, n*(block+1)//20
            s = slice(begin, end)
            mask = second["eligible_primary"][s] & second["common_charge"][s]
            ratio = first["short_long_ratio"][s][mask]
            time_rows.append({"run": run, "block": block, "event_start": begin, "event_stop_exclusive": end,
                              "event_fraction_midpoint": (block + 0.5)/20,
                              "baseline_rms_median": float(np.median(first["baseline_rms"][s])),
                              "baseline_median_adc": float(np.median(first["baseline"][s])),
                              "pretrigger_span_median": float(np.nanmedian(second["pretrigger_span"][s])),
                              "primary_retained_fraction": float(np.mean(second["eligible_primary"][s])),
                              "secondary_rise_fraction": float(np.mean(second["secondary_rise_flag"][s])),
                              "common_events": int(mask.sum()),
                              "common_ratio_median": float(np.median(ratio)) if len(ratio) else None})
        x = np.log(np.maximum(first["baseline_rms"], 1e-9))
        cumulative = np.cumsum(x)
        squares = np.cumsum(x*x)
        cut = np.arange(max(1, n//10), n-n//10)
        left = squares[cut-1] - cumulative[cut-1]**2/cut
        right = squares[-1] - squares[cut-1] - (cumulative[-1]-cumulative[cut-1])**2/(n-cut)
        split = int(cut[np.argmin(left + right)])
        total_sse = squares[-1] - cumulative[-1]**2/n
        before = float(np.median(first["baseline_rms"][:split]))
        after = float(np.median(first["baseline_rms"][split:]))
        changes.append({"run": run, "candidate_event_index": split, "fraction": split/n,
                        "timestamp_raw_uint64": int(first["timestamp"][split]),
                        "rms_median_before": before, "rms_median_after": after,
                        "after_over_before": after/before,
                        "relative_log_rms_SSE_reduction": float((total_sse - np.min(left+right))/total_sse),
                        "scope": "Best one-change description of log baseline RMS, excluding edge 10%; exploratory, always returns a candidate, no significance or physical cause claimed"})
    save_csv(PHASE2 / "time_blocks.csv", time_rows)
    save_json(PHASE2 / "noise_change_candidates.json", changes)

    rng = np.random.default_rng(2026102707)
    reference = []
    details = json.loads((PHASE2 / "fit_diagnostics.json").read_text())
    for item in details:
        observed = np.asarray(item["counts_with_flow"])
        n = int(observed.sum())
        p = np.asarray(item["expected_with_flow"]) / n
        p /= p.sum()
        simulated = rng.multinomial(n, p, size=2000) / n
        tv = np.abs(simulated - p).sum(axis=1)/2
        reference.append({"configuration": item["configuration"], "charge_bin": item["charge_bin"],
                          "run": item["run"], "split": item["split"], "events": n,
                          "observed_TV": item["binned_total_variation"],
                          "fixed_model_sampling_TV_median": float(np.median(tv)),
                          "fixed_model_sampling_TV_p99": float(np.quantile(tv, 0.99)),
                          "conditional_MC_tail_fraction": float((1+np.count_nonzero(tv >= item["binned_total_variation"])) / 2001),
                          "scope": "Conditional fixed-model multinomial reference; ignores fitted-parameter uncertainty and serial dependence, not a calibrated goodness-of-fit p-value"})
    save_csv(PHASE2 / "fit_sampling_reference.csv", reference)

    data, _ = load_data()
    all_models = json.loads((PHASE2 / "mixture_models.json").read_text())
    with np.load(PHASE2 / "benchmark_scores.npz") as file:
        qdc = {run: file[f"qdc_3000ns_run{run}"] for run in [3,12,18]}
    adjusted = []
    for row in all_models:
        if row["configuration"] != "qdc_3000ns":
            continue
        model, b = row["two_core"], row["charge_bin"]
        for run, split, core in [(3,2,1), (12,2,0), (18,3,0)]:
            d = data[run]
            mask = d["eligible_primary"] & d["common_charge"] & (d["split"] == split) & (d["charge_bin"] == b)
            x = qdc[run][mask]
            if len(x) < 20:
                continue
            start = {key: model[key] for key in ["background_center", "background_scale", "sigma_floor"]}
            start.update(means=[model["means"][core]], sigmas=[model["sigmas"][core]], weights=[[0.99,0.01]])
            fit = fit_mixture(x, np.full(len(x), run), cores=1, initial=start)
            diagnostic = diagnostics(x, fit, np.asarray(fit["weights"][0]))
            adjusted.append({"run": run, "split": split, "charge_bin": b, "events": len(x),
                             "training_core": "H" if core else "L", "training_mean": model["means"][core],
                             "training_sigma": model["sigmas"][core], "refitted_main_mean": fit["means"][0],
                             "refitted_main_sigma": fit["sigmas"][0],
                             "mean_shift_over_training_sigma": (fit["means"][0]-model["means"][core])/model["sigmas"][core],
                             "sigma_ratio": fit["sigmas"][0]/model["sigmas"][core],
                             "refitted_main_TV": diagnostic["binned_total_variation"],
                             "scope": "Post-check one-core-plus-Cauchy descriptive refit; not independent prediction or a physical particle label"})
    save_csv(PHASE2 / "qdc_main_shape_drift.csv", adjusted)
    test_path = PHASE2 / "unit_checks.json"
    unit = json.loads(test_path.read_text()) if test_path.exists() else {}
    current = bool(unit.get("passed")) and all(hashlib.sha256((HERE/name).read_bytes()).hexdigest() == digest
                                             for name, digest in unit.get("code_sha256", {}).items())
    save_json(PHASE2 / "verification.json", {
        "phase2_tests_passed": unit.get("tests_run") if current else None,
        "unit_checks_match_current_code": current,
        "all_confirmed_waveforms_checked_against_phase1_charge": True,
        "raw_files_full_sha256": "source_sha256.json",
        "no_per_event_particle_truth_assigned": True,
        "quality_noise_revision_preserved": ["protocol_v1.json", "quality_counts_v1.csv", "per-event pretrigger_flag_primary_v1"],
        "sampling_reference_scope": "2000 multinomial draws, fixed fitted model; diagnostic sampling scale only"})


if __name__ == "__main__":
    main()
