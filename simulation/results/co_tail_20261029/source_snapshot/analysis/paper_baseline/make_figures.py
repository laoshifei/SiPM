#!/usr/bin/env python3
"""Make diagnostic PNG/SVG figures from the saved PHA features (no raw reread)."""
from array import array
import csv

import numpy as np
import ROOT

from prepare import HERE, OUT, save_csv, save_json

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetTitleFont(42, "XYZ")
ROOT.gStyle.SetLabelFont(42, "XYZ")
ROOT.gStyle.SetTitleFontSize(0.045)
ROOT.gStyle.SetTitleOffset(1.5, "X")
ROOT.gStyle.SetTitleSize(0.040, "X")
ROOT.gStyle.SetLabelSize(0.035, "X")
KEEP = []
COLORS = [ROOT.TColor.GetColor(c) for c in ["#215A8C", "#CB593A", "#258275", "#8964A8"]]


def confirmed_labels():
    with (HERE / "labels.csv").open(encoding="utf-8-sig") as handle:
        return {int(r["run"]): r for r in csv.DictReader(handle)
                if r["system"] == "PHA" and r["analysis_status"] == "include"}


def save(canvas, name):
    canvas.Modified()
    canvas.Update()
    # Build vector output first so lazy ROOT title/legend painting has completed
    # before the raster capture in a batch process.
    for suffix in ["svg", "png"]:
        canvas.SaveAs(str(OUT / f"{name}.{suffix}"))


def style_pad(pad):
    pad.SetLeftMargin(0.15)
    pad.SetBottomMargin(0.18)
    pad.SetTopMargin(0.10)
    pad.SetRightMargin(0.04)
    pad.SetTicks(1, 1)


def source_labels(entries, x=0.57, y=0.84, spacing=0.065):
    # Explicit pad labels also survive ROOT's batch raster export consistently.
    for i, entry in enumerate(entries):
        label = ROOT.TLatex(x, y - i * spacing, entry)
        label.SetNDC()
        label.SetTextFont(42)
        label.SetTextSize(0.031)
        label.SetTextColor(COLORS[i])
        label.Draw()
        KEEP.append(label)


def summary():
    labels = confirmed_labels()
    with (OUT / "pha_qc.csv").open(encoding="utf-8-sig") as handle:
        rows = [r for r in csv.DictReader(handle) if int(r["run"]) in labels]
    x = np.array([int(r["run"]) for r in rows], dtype="float64")
    canvas = ROOT.TCanvas("summary", "PHA diagnostic overview", 1250, 900)
    canvas.Divide(2, 2, 0.005, 0.005)
    metrics = [("baseline_rms", "Baseline RMS (ADC)", "Baseline noise", False),
               ("amplitude", "Peak above baseline (ADC)", "Pulse amplitude", True),
               ("charge_long", "Aligned charge (ADC #times sample)", "Finite-window light output", True)]
    for panel, (key, ylabel, title, log) in enumerate(metrics, 1):
        pad = canvas.cd(panel)
        style_pad(pad)
        if log:
            pad.SetLogy()
        low, med, high = [np.array([float(r[f"{key}_{q}"]) for r in rows]) for q in ["p05", "p50", "p95"]]
        graph = ROOT.TGraphAsymmErrors(len(x), x, med, np.zeros(len(x)), np.zeros(len(x)), med - low, high - med)
        graph.SetTitle(f"{title};PHA run (CH0);{ylabel}")
        graph.SetMarkerStyle(20)
        graph.SetMarkerSize(0.75)
        graph.SetMarkerColor(COLORS[0])
        graph.SetLineColor(COLORS[0])
        graph.Draw("AP")
        graph.GetXaxis().SetLimits(0, max(x) + 2)
        graph.GetXaxis().SetNdivisions(510)
        graph.GetYaxis().SetTitleOffset(1.55)
        note = ROOT.TLatex()
        note.SetNDC()
        note.SetTextSize(0.030)
        note.DrawLatex(0.18, 0.84, "Median; bars: event 5th-95th percentiles")
        KEEP.extend([graph, note])
    pad = canvas.cd(4)
    style_pad(pad)
    frac = np.array([100 * (1 - int(r["basic_window_eligible"]) / int(r["events"])) for r in rows])
    graph = ROOT.TGraph(len(x), x, frac)
    graph.SetTitle("Pilot diagnostic flags;PHA run (CH0);Events with any diagnostic flag (%)")
    graph.SetMarkerStyle(20)
    graph.SetMarkerColor(COLORS[1])
    graph.SetMinimum(0)
    graph.SetMaximum(max(0.6, float(frac.max()) * 1.3))
    graph.Draw("AP")
    graph.GetXaxis().SetLimits(0, max(x) + 2)
    graph.GetYaxis().SetTitleOffset(1.55)
    note = ROOT.TLatex()
    note.SetNDC()
    note.SetTextSize(0.029)
    note.DrawLatex(0.18, 0.84, "Flags overlap; this is not a particle-selection efficiency")
    KEEP.extend([graph, note])
    save(canvas, "pha_qc_overview")


def coverage():
    labels = confirmed_labels()
    runs = [3, 12, 18]
    for run in runs:
        if run not in labels or float(labels[run]["bias_voltage_V"]) != 53.2:
            raise ValueError("53.2 V comparison requires confirmed runs 3, 12, 18")
    values, ratios = {}, {}
    for run in runs:
        with np.load(OUT / f"pha_run{run:04d}_features.npz") as data:
            good = data["diagnostic_flags"] == 0
            values[run] = data["charge_long"][good]
            ratios[run] = data["short_long_ratio"][good]
    all_q = np.concatenate(list(values.values()))
    low, high = np.quantile(all_q, [0.001, 0.999])
    edges = np.geomspace(max(1, low), high, 71)
    common_low = max(float(np.quantile(q, 0.05)) for q in values.values())
    common_high = min(float(np.quantile(q, 0.95)) for q in values.values())
    if common_low >= common_high:
        raise ValueError("No overlap of central 90% charge ranges")
    common_edges = np.linspace(common_low, common_high, 7)
    summary_rows = []
    for run in runs:
        for b in range(6):
            selected = (values[run] >= common_edges[b]) & (values[run] < common_edges[b + 1])
            if b == 5:
                selected |= values[run] == common_edges[b + 1]
            subset = ratios[run][selected]
            q05, median, q95 = np.quantile(subset, [0.05, 0.5, 0.95]) if len(subset) else [None] * 3
            summary_rows.append({"run": run, "source": labels[run]["source_or_background"],
                                 "bias_voltage_V": 53.2, "charge_bin": b,
                                 "charge_low": common_edges[b], "charge_high": common_edges[b + 1],
                                 "events": len(subset), "ratio_p05": q05, "ratio_p50": median,
                                 "ratio_p95": q95})
    save_csv(OUT / "matched_charge_bins.csv", summary_rows)
    save_json(OUT / "confirmed_cohort.json", {
        "included_pha_runs": sorted(labels), "matched_bias_runs": runs,
        "bias_voltage_V": 53.2, "common_charge_range_adc_samples": [common_low, common_high],
        "overlap_rule": "Intersection of per-run 5th-95th charge percentiles after pilot diagnostic flags == 0",
        "binning": "Six equal charge bins within that intersection; no PSD optimization",
        "particle_label_policy": "Source identity only; 241Am alpha/gamma mixture and source/background contamination retained",
        "event_counts_in_common_range": {str(run): int(np.count_nonzero((values[run] >= common_low) & (values[run] <= common_high))) for run in runs},
        "performance_claim": "No alpha efficiency, gamma leakage, energy threshold, FOM or independent run validation claimed"})
    canvas = ROOT.TCanvas("coverage", "PHA charge coverage", 1250, 900)
    canvas.Divide(2, 2, 0.005, 0.005)
    pad = canvas.cd(1)
    style_pad(pad)
    pad.SetLogx()
    hists = []
    for i, run in enumerate(runs):
        q = values[run]
        counts, _ = np.histogram(q, edges)
        hist = ROOT.TH1D(f"charge_{run}", "", len(edges) - 1, array("d", edges))
        hist.SetDirectory(0)
        for k, count in enumerate(counts, 1):
            hist.SetBinContent(k, float(count / len(q)))
        hist.SetLineColor(COLORS[i])
        hist.SetLineWidth(2)
        hist.SetTitle("Same detector, nominal 53.2 V;Aligned charge (ADC #times sample);Fraction per logarithmic bin")
        hist.GetYaxis().SetTitleOffset(1.55)
        hist.GetXaxis().SetMoreLogLabels()
        hists.append(hist)
    maximum = max(h.GetMaximum() for h in hists)
    hists[0].SetMaximum(maximum * 1.3)
    for i, hist in enumerate(hists):
        hist.Draw("HIST" if i == 0 else "HIST SAME")
    source_labels([f"run {run}: {labels[run]['source_or_background']}" for run in runs], x=0.70)
    for boundary in [common_low, common_high]:
        line = ROOT.TLine(boundary, 0, boundary, maximum * 1.3)
        line.SetLineStyle(2)
        line.SetLineColor(ROOT.kGray + 2)
        line.Draw()
        KEEP.append(line)
    KEEP.extend(hists)
    for panel, run in enumerate(runs, 2):
        pad = canvas.cd(panel)
        style_pad(pad)
        pad.SetRightMargin(0.13)
        pad.SetLogx()
        pad.SetLogz()
        yedges = np.linspace(0.0, 0.75, 101)
        counts, _, _ = np.histogram2d(values[run], ratios[run], bins=[edges, yedges])
        source = labels[run]["source_or_background"]
        identity = "gamma-source data" if source == "60Co" else "alpha/gamma mixed source"
        hist = ROOT.TH2D(f"pid_{run}", f"run {run}: {source}, {identity};Aligned charge (ADC #times sample);Q_{{short}} / Q_{{long}}",
                         len(edges) - 1, array("d", edges), len(yedges) - 1, array("d", yedges))
        hist.SetDirectory(0)
        for ix, iy in np.argwhere(counts > 0):
            hist.SetBinContent(int(ix) + 1, int(iy) + 1, counts[ix, iy])
        hist.GetYaxis().SetTitleOffset(1.45)
        hist.GetXaxis().SetMoreLogLabels()
        hist.GetZaxis().SetTitle("Events / bin")
        hist.SetMinimum(1)
        hist.Draw("COLZ")
        KEEP.append(hist)
    save(canvas, "pha_charge_coverage")

    canvas = ROOT.TCanvas("matched", "Matched charge projections", 1250, 1100)
    canvas.Divide(2, 3, 0.005, 0.005)
    for b in range(6):
        style_pad(canvas.cd(b + 1))
        hists = []
        for i, run in enumerate(runs):
            sel = (values[run] >= common_edges[b]) & (values[run] < common_edges[b + 1])
            if b == 5:
                sel |= values[run] == common_edges[b + 1]
            subset = ratios[run][sel]
            redges = np.linspace(0.15, 0.60, 91)
            counts, _ = np.histogram(subset, redges)
            hist = ROOT.TH1D(f"projection_{run}_{b}", "", len(redges) - 1, array("d", redges))
            hist.SetDirectory(0)
            for k, count in enumerate(counts, 1):
                hist.SetBinContent(k, float(count / len(subset)) if len(subset) else 0)
            hist.SetLineColor(COLORS[i])
            hist.SetLineWidth(2)
            hist.SetTitle(f"Q: {common_edges[b]/1000:.1f}-{common_edges[b+1]/1000:.1f} #times 10^{{3}} ADC samples;Q_{{short}} / Q_{{long}};Fraction per bin")
            hist.GetYaxis().SetTitleOffset(1.55)
            hists.append(hist)
        hists[0].SetMaximum(max(h.GetMaximum() for h in hists) * 1.65)
        for i, hist in enumerate(hists):
            hist.Draw("HIST" if i == 0 else "HIST SAME")
        source_labels([f"{labels[run]['source_or_background']} r{run}, N={int(next(r['events'] for r in summary_rows if r['run'] == run and r['charge_bin'] == b)):,}" for run in runs], x=0.60)
        KEEP.extend(hists)
    save(canvas, "pha_matched_projections")


if __name__ == "__main__":
    summary()
    coverage()
