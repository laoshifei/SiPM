#!/usr/bin/env python3
"""Scientific diagnostic figures for quality, model-based FOM and cross-run checks."""
from array import array
import csv
import json

import numpy as np
import ROOT

from refine_waveforms import PHASE2

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetTitleFontSize(0.042)
ROOT.gStyle.SetTitleFont(42, "XYZ")
ROOT.gStyle.SetLabelFont(42, "XYZ")
ROOT.gStyle.SetLabelSize(0.035, "XYZ")
ROOT.gStyle.SetTitleSize(0.040, "XYZ")
ROOT.gStyle.SetTitleOffset(1.35, "X")
ROOT.gStyle.SetTitleOffset(1.45, "Y")
COLORS = [ROOT.TColor.GetColor(c) for c in ["#215A8C", "#CB593A", "#258275", "#8964A8"]]
KEEP = []
METHODS = [("qdc", "Short / long charge"), ("negative_mean_time", "Mean time"), ("pca_projection", "PCA projection")]


def rows(name):
    with (PHASE2 / name).open(encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def pad(canvas, index):
    p = canvas.cd(index)
    p.SetLeftMargin(0.15)
    p.SetRightMargin(0.04)
    p.SetBottomMargin(0.18)
    p.SetTopMargin(0.10)
    p.SetTicks(1, 1)
    return p


def frame(p, xmin, ymin, xmax, ymax, title):
    h = p.DrawFrame(xmin, ymin, xmax, ymax, title)
    KEEP.append(h)
    return h


def label(value, x, y, color=ROOT.kBlack, size=0.031):
    t = ROOT.TLatex(x, y, value)
    t.SetNDC()
    t.SetTextFont(42)
    t.SetTextSize(size)
    t.SetTextColor(color)
    t.Draw()
    KEEP.append(t)


def graph(x, y, color, low=None, high=None, style=1, markers=True):
    x, y = np.asarray(x, dtype="float64"), np.asarray(y, dtype="float64")
    if low is None:
        g = ROOT.TGraph(len(x), x, y)
    else:
        g = ROOT.TGraphAsymmErrors(len(x), x, y, np.zeros(len(x)), np.zeros(len(x)),
                                   y-np.asarray(low, dtype="float64"), np.asarray(high, dtype="float64")-y)
    g.SetLineColor(color)
    g.SetLineWidth(2)
    g.SetLineStyle(style)
    g.SetMarkerColor(color)
    g.SetMarkerStyle(20)
    g.SetMarkerSize(0.75)
    g.Draw("LP SAME" if markers else "L SAME")
    KEEP.append(g)
    return g


def save(canvas, name):
    canvas.Modified()
    canvas.Update()
    for extension in ["svg", "png"]:
        canvas.SaveAs(str(PHASE2 / f"{name}.{extension}"))


def method_figure():
    table = rows("method_comparison.csv")
    c = ROOT.TCanvas("phase2_methods", "Shape method comparison", 1300, 650)
    c.Divide(2, 1, 0.005, 0.005)
    p = pad(c, 1)
    frame(p, 195, 1.4, 425, 4.6, "Three methods at 3 #mus;Reference charge (10^{3} ADC samples);FWHM FOM of fitted shape cores")
    for i, (key, title) in enumerate(METHODS):
        a = [r for r in table if r["configuration"] == key+"_3000ns"]
        x = [(float(r["charge_low"])+float(r["charge_high"]))/2000 + (i-1)*1.3 for r in a]
        graph(x, [float(r["shape_fom_fwhm"]) for r in a], COLORS[i],
              [float(r["fom_interval_low"]) for r in a], [float(r["fom_interval_high"]) for r in a])
        label(title, 0.18, 0.85-i*0.055, COLORS[i])
    label("95% conditional block-bootstrap intervals", 0.18, 0.65, size=0.028)
    p = pad(c, 2)
    frame(p, 0.75, 0.5, 4.25, 4.8, "Window endpoint scan;Window end after t10 (#mus);Median shape FOM over six charge bins")
    for i, (key, title) in enumerate(METHODS):
        y, low, high = [], [], []
        for end in [1000, 2000, 3000, 4000]:
            values = [float(r["shape_fom_fwhm"]) for r in table if r["configuration"] == f"{key}_{end}ns"]
            y.append(np.median(values));low.append(min(values));high.append(max(values))
        graph(np.arange(1,5)+(i-1)*0.025, y, COLORS[i], low, high)
        label(title, 0.18, 0.85-i*0.055, COLORS[i])
    label("Bars: spread across charge bins, not uncertainty", 0.18, 0.69, size=0.027)
    save(c, "phase2_methods")


def quality_figure():
    time = rows("time_blocks.csv")
    sensitivity = rows("sensitivity_checks.csv")
    c = ROOT.TCanvas("phase2_quality", "Quality and acquisition-order checks", 1300, 950)
    c.Divide(2, 2, 0.005, 0.005)
    p = pad(c, 1)
    frame(p, 0, 2, 100, 8, "Baseline variation within runs;Acquisition order (% of saved events);Median baseline RMS (ADC)")
    for i, run in enumerate([3,12,18]):
        a = [r for r in time if int(r["run"]) == run]
        graph([100*float(r["event_fraction_midpoint"]) for r in a], [float(r["baseline_rms_median"]) for r in a], COLORS[i])
        label(f"run {run}", 0.18, 0.85-i*0.055, COLORS[i])
    label("run 12 change candidate: event 5029", 0.18, 0.65, size=0.028)
    p = pad(c, 2)
    frame(p, 0, 65, 100, 106, "Retention under default pretrigger check;Acquisition order (% of saved events);Retained saved events (%)")
    for i, run in enumerate([3,12,18]):
        a = [r for r in time if int(r["run"]) == run]
        graph([100*float(r["event_fraction_midpoint"]) for r in a], [100*float(r["primary_retained_fraction"]) for r in a], COLORS[i])
        label(f"run {run}", 0.18, 0.62-i*0.055, COLORS[i])
    for x in [60,80]:
        line=ROOT.TLine(x,65,x,106);line.SetLineStyle(2);line.SetLineColor(ROOT.kGray+1);line.Draw();KEEP.append(line)
    label("run 3/12 split boundaries: 60%, 80%",0.18,0.38,size=0.027)
    p = pad(c, 3)
    hist = ROOT.TH1D("band_categories", "Diagnostic-band retention;Quality condition;Events in the original diagnostic band", 4, 0.5, 4.5)
    hist.SetDirectory(0);hist.SetMinimum(0);hist.SetMaximum(56)
    for i, title in enumerate(["Basic", "Loose", "Default", "Strict"], 1):hist.GetXaxis().SetBinLabel(i,title)
    hist.Draw("AXIS");KEEP.append(hist)
    label("Diagnostic-band retention",0.28,0.95,size=0.040)
    bands = rows("band_retention.csv")
    for run, band, i, title in [(3,"0.28",0,"60Co run 3, low-R band"),(18,"0.43",2,"241Am run 18, high-R band")]:
        r = next(r for r in bands if int(r["run"])==run and r["ratio_low"]==band)
        graph([1,2,3,4],[int(r[k]) for k in ["before","loose","primary","strict"]],COLORS[i])
        label(title,0.18,0.85-(0 if i==0 else 0.065),COLORS[i])
    p = pad(c,4)
    frame(p,195,-3,425,3.6,"QDC sensitivity at 3 #mus;Reference charge (10^{3} ADC samples);FOM change from default (%)")
    default = {int(r["charge_bin"]):float(r["shape_fom_fwhm"]) for r in sensitivity if r["variant"]=="primary"}
    for i,(variant,title) in enumerate([("basic","Basic checks only"),("strict","Strict pretrigger check"),("primary_recent_baseline","Recent baseline estimate")]):
        a=[r for r in sensitivity if r["variant"]==variant]
        x=[201.8241125+(int(r["charge_bin"])+0.5)*(418.7332-201.8241125)/6 for r in a]
        y=[100*(float(r["shape_fom_fwhm"])/default[int(r["charge_bin"])]-1) for r in a]
        graph(x,y,COLORS[i]);label(title,0.18,0.85-i*0.055,COLORS[i],size=0.028)
    save(c,"phase2_quality")


def transfer_figure():
    detail=json.loads((PHASE2/"fit_diagnostics.json").read_text())
    c=ROOT.TCanvas("phase2_transfer","Cross-run shape checks",1300,950)
    c.Divide(2,2,0.005,0.005)
    for panel,b in [(1,0),(2,4)]:
        r=next(r for r in detail if r["configuration"]=="qdc_3000ns" and r["run"]==18 and r["charge_bin"]==b)
        edges=np.asarray(r["score_edges"]);centers=(edges[1:]+edges[:-1])/2
        counts=np.asarray(r["counts_with_flow"]);expected=np.asarray(r["expected_with_flow"])
        n=counts.sum();obs=counts[1:-1]/n;pred=expected[1:-1]/n
        p=pad(c,panel)
        frame(p,0.26,0,0.39,max(obs.max(),pred.max())*1.55,f"Run 18, charge bin {b+1}: main-band zoom;Q_{{short}} / Q_{{long}};Fraction per score bin")
        graph(centers,obs,COLORS[2],markers=False)
        graph(centers,pred,ROOT.kBlack,style=2,markers=False)
        label("Observed run 18",0.18,0.85,COLORS[2])
        label("Frozen shape model; proportions refitted",0.18,0.79,size=0.027)
    p=pad(c,3)
    frame(p,195,0,425,0.15,"Run 18 fixed-shape model discrepancy;Reference charge (10^{3} ADC samples);Binned total variation distance")
    reference=rows("fit_sampling_reference.csv")
    for i,(key,title) in enumerate(METHODS):
        a=[r for r in reference if r["configuration"]==key+"_3000ns" and int(r["run"])==18]
        x=[201.8241125+(int(r["charge_bin"])+0.5)*(418.7332-201.8241125)/6 for r in a]
        graph(x,[float(r["observed_TV"]) for r in a],COLORS[i])
        graph(x,[float(r["fixed_model_sampling_TV_p99"]) for r in a],COLORS[i],style=2,markers=False)
        label(title,0.18,0.85-i*0.055,COLORS[i])
    label("Dashed: 99% fixed-model sampling reference",0.18,0.34,size=0.026)
    p=pad(c,4)
    frame(p,195,0.65,425,1.2,"Run 18 main-core width change;Reference charge (10^{3} ADC samples);Refitted width / training L-core width")
    a=[r for r in rows("qdc_main_shape_drift.csv") if int(r["run"])==18]
    x=[201.8241125+(int(r["charge_bin"])+0.5)*(418.7332-201.8241125)/6 for r in a]
    graph(x,[float(r["sigma_ratio"]) for r in a],COLORS[2])
    line=ROOT.TLine(195,1,425,1);line.SetLineStyle(2);line.Draw();KEEP.append(line)
    label("QDC, 3 us; descriptive refit after the check",0.18,0.85,size=0.028)
    save(c,"phase2_transfer")


if __name__ == "__main__":
    method_figure()
    quality_figure()
    transfer_figure()
