#!/usr/bin/env python3
"""Source-independent pretrigger diagnostics and window features for confirmed PHA data."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from prepare import HERE, OUT, PROJECT, bin_chunks, bin_layout, save_csv, save_json

PHASE2 = OUT / "phase2"
LONG_ENDS = np.array([500, 1000, 1500, 2000], dtype=np.int32)
SHORT_ENDS = np.array([100, 150, 250, 400], dtype=np.int32)
SPEC = {
    "version": 2,
    "scope": "Exploratory continuation; first-phase distributions have already been inspected.",
    "sample_period_ns": 2,
    "baseline": "First 128 samples, unchanged from phase 1",
    "alignment": "First-phase t10; anchor is floor(t10)+1, checked against first-phase QDC",
    "pretrigger": "Range of 32-sample moving means in [0, anchor-64)",
    "noise_scale": "Maximum of per-event first-difference MAD/sqrt(2), 1 ADC, and per-run reference baseline RMS",
    "reference_baseline_noise": "Median first-128-sample RMS over first 20% of original events of each run; baseline data only, no PSD or source-class selection",
    "revision_reason": "Version 1 used first-difference noise alone; this underestimated the observed correlated/low-frequency baseline variation. Version-1 counts and flags are retained.",
    "pretrigger_thresholds": {"loose": {"noise_multiplier": 12.0, "amplitude_fraction": 0.05},
                              "primary": {"noise_multiplier": 8.0, "amplitude_fraction": 0.03},
                              "strict": {"noise_multiplier": 5.0, "amplitude_fraction": 0.015}},
    "pretrigger_rule": "Flag only when moving-mean span exceeds BOTH thresholds (their maximum)",
    "secondary_rise": "Maximum 32-sample-lag rise of 16-sample means after peak+80; flag above max(8*noise, 0.15*amplitude). Diagnostic only in primary analysis.",
    "negative_tail_policy": "Report, never reject an event solely for a negative tail",
    "alternative_baseline": "Mean over [anchor-192, anchor-64), sensitivity only",
    "integration_start_offset_samples": -20,
    "long_end_offsets_samples": LONG_ENDS.tolist(),
    "short_end_offsets_samples": SHORT_ENDS.tolist(),
    "block_width_samples": 20,
    "benchmark_runs": [3, 12, 18],
    "training": "First 60% of original event indices of runs 3 and 12",
    "validation": "Next 20% of original event indices of runs 3 and 12",
    "temporal_check": "Last 20% of original event indices of runs 3 and 12",
    "external_run_check": "All eligible run 18 events; already inspected in exploratory phase, not a blind test",
    "labels": "Source provenance only; no event-level alpha/gamma labels",
}


def confirmed():
    with (HERE / "labels.csv").open(encoding="utf-8-sig") as f:
        return {int(row["run"]): row for row in csv.DictReader(f)
                if row["system"] == "PHA" and row["analysis_status"] == "include"}


def advanced_features(waves, t10, peak, amplitude, baseline, noise_reference):
    waves = np.asarray(waves, dtype=np.float64)
    n, length = waves.shape
    ix = np.arange(n)
    valid_anchor = np.isfinite(t10)
    anchor = np.where(valid_anchor, np.floor(np.nan_to_num(t10)) + 1, 0).astype(np.int32)
    y = waves - baseline[:, None]
    cumulative = np.c_[np.zeros(n), np.cumsum(y, axis=1)]
    weighted = np.c_[np.zeros(n), np.cumsum(y * np.arange(length)[None, :], axis=1)]
    differences = np.diff(waves[:, :128], axis=1)
    median_difference = np.median(differences, axis=1)
    noise_difference = np.maximum(1.0, 1.4826022185 * np.median(np.abs(differences - median_difference[:, None]), axis=1) / np.sqrt(2))
    noise = np.maximum(noise_difference, noise_reference)

    # Only a strictly pretrigger interval is used for the primary additional cut.
    means32 = (cumulative[:, 32:] - cumulative[:, :-32]) / 32
    endpoints = np.arange(32, length + 1)[None, :]
    pre = endpoints <= (anchor - 64)[:, None]
    have_pre = valid_anchor & (anchor >= 256)
    pre_max = np.where(pre, means32, -np.inf).max(axis=1)
    pre_min = np.where(pre, means32, np.inf).min(axis=1)
    pre_span = np.where(have_pre, pre_max - pre_min, np.nan)
    result = {"anchor_sample": anchor, "noise_difference_mad": noise_difference,
              "noise_reference": noise, "pretrigger_span": pre_span,
              "pretrigger_flag_primary_v1": (~have_pre) | (pre_span > np.maximum(8 * noise_difference, 0.03 * amplitude))}
    for name, threshold in SPEC["pretrigger_thresholds"].items():
        limit = np.maximum(threshold["noise_multiplier"] * noise,
                           threshold["amplitude_fraction"] * amplitude)
        result[f"pretrigger_flag_{name}"] = (~have_pre) | (pre_span > limit)

    means16 = (cumulative[:, 16:] - cumulative[:, :-16]) / 16
    rise = means16[:, 32:] - means16[:, :-32]
    rise_start = np.arange(rise.shape[1])[None, :]
    after_peak = rise_start >= (peak + 80)[:, None]
    largest_rise = np.maximum(0, np.where(after_peak, rise, -np.inf).max(axis=1))
    result["secondary_rise_adc"] = largest_rise
    result["secondary_rise_flag"] = largest_rise > np.maximum(8 * noise, 0.15 * amplitude)
    recent_start, recent_stop = anchor - 192, anchor - 64
    recent_delta = (cumulative[ix, np.clip(recent_stop, 0, length)]
                    - cumulative[ix, np.clip(recent_start, 0, length)]) / 128
    result["recent_baseline_delta"] = np.where(have_pre, recent_delta, np.nan)

    start = anchor - 20
    long_charge, mean_time, supported = [], [], []
    for end in LONG_ENDS:
        stop = anchor + end
        good = valid_anchor & (start >= 128) & (stop <= length)
        q = cumulative[ix, np.clip(stop, 0, length)] - cumulative[ix, np.clip(start, 0, length)]
        moment = weighted[ix, np.clip(stop, 0, length)] - weighted[ix, np.clip(start, 0, length)] - anchor * q
        time = np.divide(moment * 0.002, q, out=np.full(n, np.nan), where=good & (q > 0))
        long_charge.append(np.where(good, q, np.nan))
        mean_time.append(time)
        supported.append(good)
    short_charge = []
    for end in SHORT_ENDS:
        stop = anchor + end
        good = valid_anchor & (start >= 128) & (stop <= length)
        q = cumulative[ix, np.clip(stop, 0, length)] - cumulative[ix, np.clip(start, 0, length)]
        short_charge.append(np.where(good, q, np.nan))
    result["long_charge"] = np.column_stack(long_charge)
    result["short_charge"] = np.column_stack(short_charge)
    result["mean_time_us"] = np.column_stack(mean_time)
    result["window_supported"] = np.column_stack(supported)
    boundaries = anchor[:, None] - 20 + np.arange(102)[None, :] * 20
    integrated = cumulative[ix[:, None], np.clip(boundaries, 0, length)]
    blocks = np.diff(integrated, axis=1)
    block_valid = (boundaries[:, :-1] >= 128) & (boundaries[:, 1:] <= length) & valid_anchor[:, None]
    result["block_charge"] = np.where(block_valid, blocks, np.nan).astype(np.float32)
    return result


def main():
    PHASE2.mkdir(parents=True, exist_ok=True)
    previous = PHASE2 / "protocol.json"
    if previous.exists() and json.loads(previous.read_text()).get("version") == 1:
        (PHASE2 / "protocol_v1.json").write_text(previous.read_text())
        counts = PHASE2 / "quality_counts.csv"
        if counts.exists():
            (PHASE2 / "quality_counts_v1.csv").write_bytes(counts.read_bytes())
    save_json(PHASE2 / "protocol.json", SPEC)
    cohort = json.loads((OUT / "confirmed_cohort.json").read_text())
    qlo, qhi = cohort["common_charge_range_adc_samples"]
    summaries, band_rows, source_hashes = [], [], []
    for run, label in sorted(confirmed().items()):
        path = PROJECT / label["path"]
        header, dtype, count = bin_layout(path)
        with np.load(OUT / f"pha_run{run:04d}_features.npz") as original:
            f = {k: original[k] for k in original.files}
        if count != len(f["event_index"]):
            raise ValueError("First-phase data count changed")
        noise_reference = float(np.median(f["baseline_rms"][:max(1, int(count * 0.2))]))
        parts = {}
        digest = hashlib.sha256(int(header).to_bytes(2, "little"))
        for offset, chunk in bin_chunks(path, dtype, count, chunk_size=512):
            digest.update(chunk.tobytes())
            sl = slice(offset, offset + len(chunk))
            a = advanced_features(chunk["data"], f["t10_sample"][sl], f["peak_sample"][sl],
                                  f["amplitude"][sl], f["baseline"][sl], noise_reference)
            # Independent cumulative implementation must reproduce the pilot windows.
            if not np.allclose(a["long_charge"][:, 2], f["charge_long"][sl], equal_nan=True):
                raise ValueError(f"Alignment/integral mismatch: run {run}, event {offset}")
            for key, val in a.items():
                if key == "block_charge" and run not in SPEC["benchmark_runs"]:
                    continue
                parts.setdefault(key, []).append(val)
        a = {key: np.concatenate(value) for key, value in parts.items()}
        basic = f["diagnostic_flags"] == 0
        common = (f["charge_long"] >= qlo) & (f["charge_long"] <= qhi)
        support = np.all(a["window_supported"], axis=1) & np.all(a["long_charge"] > 0, axis=1)
        a["basic_eligible"] = basic
        a["common_charge"] = common
        a["comparison_support"] = support
        a["event_index"] = f["event_index"]
        a["split"] = np.where(f["event_index"] < int(count * 0.6), 0,
                              np.where(f["event_index"] < int(count * 0.8), 1, 2)).astype(np.int8)
        if run == 18:
            a["split"][:] = 3
        elif run not in (3, 12):
            a["split"][:] = 4
        for mode in ["loose", "primary", "strict"]:
            a[f"eligible_{mode}"] = basic & support & ~a[f"pretrigger_flag_{mode}"]
        np.savez_compressed(PHASE2 / f"run{run:04d}_features.npz", **a)
        row = {"run": run, "source": label["source_or_background"], "bias_voltage_V": label["bias_voltage_V"],
               "baseline_reference_rms_adc": noise_reference,
               "saved_events": count, "basic_eligible": int(basic.sum()),
               "basic_with_all_windows": int((basic & support).sum()),
               "common_before_additional_quality": int((basic & common).sum()),
               "secondary_rise_among_basic": int((basic & a["secondary_rise_flag"]).sum())}
        for mode in ["loose", "primary", "strict"]:
            good = a[f"eligible_{mode}"]
            row[f"eligible_{mode}"] = int(good.sum())
            row[f"common_{mode}"] = int((good & common).sum())
        primary = a["eligible_primary"]
        row["primary_plus_no_secondary_rise"] = int((primary & ~a["secondary_rise_flag"]).sum())
        row["median_pretrigger_span_adc"] = float(np.nanmedian(a["pretrigger_span"][basic]))
        row["recent_baseline_delta_p05"] = float(np.nanquantile(a["recent_baseline_delta"][primary], 0.05))
        row["recent_baseline_delta_p95"] = float(np.nanquantile(a["recent_baseline_delta"][primary], 0.95))
        summaries.append(row)
        for lo, hi in [(0.28, 0.38), (0.43, 0.60)]:
            band = basic & common & (f["short_long_ratio"] >= lo) & (f["short_long_ratio"] < hi)
            band_rows.append({"run": run, "ratio_low": lo, "ratio_high": hi, "before": int(band.sum()),
                              **{mode: int((band & a[f"eligible_{mode}"]).sum()) for mode in ["loose", "primary", "strict"]},
                              "primary_plus_no_secondary_rise": int((band & primary & ~a["secondary_rise_flag"]).sum())})
        source_hashes.append({"run": run, "path": label["path"], "sha256": digest.hexdigest(),
                              "bytes": path.stat().st_size, "events": count})
        print(f"run {run}: {count} events; pretrigger-clean {row['eligible_primary']}; common {row['common_primary']}", flush=True)
    save_csv(PHASE2 / "quality_counts.csv", summaries)
    save_csv(PHASE2 / "band_retention.csv", band_rows)
    save_json(PHASE2 / "source_sha256.json", source_hashes)


if __name__ == "__main__":
    main()
