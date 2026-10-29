#!/usr/bin/env python3
"""Inspect deterministic examples from two PSD bands, without assigning truth labels."""
import csv
import json

import numpy as np
import ROOT

from prepare import HERE, OUT, PROJECT, bin_layout, save_csv, save_json


def main():
    ROOT.gROOT.SetBatch(True)
    ROOT.gStyle.SetOptStat(0)
    with (HERE / "labels.csv").open(encoding="utf-8-sig") as handle:
        labels = {int(r["run"]): r for r in csv.DictReader(handle)
                  if r["system"] == "PHA" and r["analysis_status"] == "include"}
    qlo, qhi = json.loads((OUT / "confirmed_cohort.json").read_text())["common_charge_range_adc_samples"]
    canvas = ROOT.TCanvas("wave_audit", "Waveform band inspection", 1250, 1150)
    canvas.Divide(2, 3, 0.005, 0.005)
    colors = [ROOT.TColor.GetColor(c) for c in ["#215A8C", "#CB593A", "#258275", "#8964A8"]]
    keep, rows, groups = [], [], []
    panel = 0
    for run in [3, 12, 18]:
        if run not in labels:
            raise ValueError("Only confirmed included data may be inspected here")
        with np.load(OUT / f"pha_run{run:04d}_features.npz") as features:
            path = PROJECT / labels[run]["path"]
            _, dtype, _ = bin_layout(path)
            base = ((features["diagnostic_flags"] == 0) & (features["charge_long"] >= qlo)
                    & (features["charge_long"] <= qhi))
            for lower, upper in [(0.28, 0.38), (0.43, 0.60)]:
                panel += 1
                candidates = np.flatnonzero(base & (features["short_long_ratio"] >= lower)
                                            & (features["short_long_ratio"] < upper))
                selected = candidates[np.unique(np.linspace(0, len(candidates) - 1, min(4, len(candidates)), dtype=int))] if len(candidates) else []
                flag_values, flag_counts = np.unique(features["flags"][candidates], return_counts=True)
                groups.append({"run": run, "ratio_low": lower, "ratio_high": upper,
                               "candidates": len(candidates), "inspected_event_indices": [int(i) for i in selected],
                               "hardware_flags": {str(int(k)): int(v) for k, v in zip(flag_values, flag_counts)},
                               "tail_to_amplitude_median": float(np.median(features["tail_level"][candidates] / features["amplitude"][candidates])) if len(candidates) else None})
                pad = canvas.cd(panel)
                pad.SetLeftMargin(0.13)
                pad.SetBottomMargin(0.14)
                pad.SetTopMargin(0.12)
                pad.SetRightMargin(0.04)
                title = f"run {run}, {labels[run]['source_or_background']}: {lower:.2f} #leq R < {upper:.2f}, N={len(candidates)}"
                frame = pad.DrawFrame(-1.0, -0.25, 4.4, 1.6, title + ";Time relative to t10 (#mus);Baseline-subtracted ADC / peak")
                legend = ROOT.TLegend(0.60, 0.63, 0.94, 0.86)
                legend.SetFillStyle(0)
                legend.SetBorderSize(0)
                legend.SetTextSize(0.031)
                with path.open("rb") as handle:
                    for j, index in enumerate(selected):
                        handle.seek(2 + int(index) * dtype.itemsize)
                        record = np.fromfile(handle, dtype=dtype, count=1)[0]
                        x = (np.arange(len(record["data"])) - features["t10_sample"][index]) * 0.002
                        y = (record["data"].astype(float) - features["baseline"][index]) / features["amplitude"][index]
                        graph = ROOT.TGraph(len(x), x.astype("float64"), y.astype("float64"))
                        graph.SetLineColor(colors[j])
                        graph.Draw("L SAME")
                        legend.AddEntry(graph, f"event {int(index)}, flags {int(record['flags'])}", "l")
                        keep.append(graph)
                        rows.append({"run": run, "event_index": int(index), "source": labels[run]["source_or_background"],
                                     "ratio_band_low": lower, "ratio_band_high": upper,
                                     "ratio": float(features["short_long_ratio"][index]),
                                     "charge_long_adc_samples": float(features["charge_long"][index]),
                                     "timestamp_raw_uint64": int(record["timestamp"]),
                                     "hardware_flags_uint32": int(record["flags"]),
                                     "raw_path": labels[run]["path"]})
                if len(selected):
                    legend.Draw()
                else:
                    text = ROOT.TLatex()
                    text.SetNDC()
                    text.SetTextSize(0.04)
                    text.DrawLatex(0.28, 0.52, "No events in this diagnostic band")
                    keep.append(text)
                keep.extend([frame, legend])
    save_csv(OUT / "waveform_inspection_events.csv", rows)
    save_json(OUT / "waveform_band_audit.json", {
        "charge_range_adc_samples": [qlo, qhi], "groups": groups,
        "example_rule": "At most four equally spaced positions in the time-ordered candidate list",
        "scope": "Exploratory PSD bands for raw waveform inspection only; not truth labels or a validated selection",
        "raw_waveforms_inspected": len(rows)})
    for extension in ["png", "svg"]:
        canvas.SaveAs(str(OUT / f"pha_waveform_band_examples.{extension}"))


if __name__ == "__main__":
    main()
