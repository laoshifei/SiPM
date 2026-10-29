#!/usr/bin/env python3
"""Associate PHA waveform files with saved energy histograms without date guesses."""
import json
import platform

import numpy as np
import ROOT

from prepare import HERE, OUT, PROJECT, relative, save_json


def main():
    ROOT.gROOT.SetBatch(True)
    comparisons = []
    inventory = json.loads((OUT / "pha_inventory.json").read_text())
    for run in inventory:
        if not run["entries"]:
            continue
        with np.load(OUT / f"pha_run{run['run']:04d}_features.npz") as data:
            energy = data["energy_ch"]
        for path in sorted((PROJECT / run["path"]).parent.glob("Hcompass*.root")):
            file = ROOT.TFile.Open(str(path))
            # Read only known TH1 objects: the files also contain application-
            # specific calibration classes for which ROOT has no local dictionary.
            hist = file.Get("Energy/EnergyCH0@DT5730_1258")
            if not hist:
                raise RuntimeError(f"Missing energy histogram: {path}")
            edges = np.array([hist.GetXaxis().GetBinLowEdge(i)
                              for i in range(1, hist.GetNbinsX() + 2)])
            bins = np.searchsorted(edges, energy, side="right")
            reconstructed = np.bincount(bins, minlength=hist.GetNbinsX() + 2)
            stored = np.array([hist.GetBinContent(i) for i in range(hist.GetNbinsX() + 2)])
            comparisons.append({"run": run["run"], "snapshot": relative(path),
                                "waveform_events": len(energy),
                                "histogram_entries": int(hist.GetEntries()),
                                "histogram_sum_including_flow": float(stored.sum()),
                                "all_bin_counts_equal": bool(np.array_equal(reconstructed, stored)),
                                "sum_absolute_bin_difference": float(np.abs(reconstructed - stored).sum())})
            file.Close()
    save_json(OUT / "histogram_association.json", comparisons)
    source_stats = []
    for filename in ["pha_inventory.json", "root_inventory.json"]:
        for record in json.loads((OUT / filename).read_text()):
            path = PROJECT / record["path"]
            stat = path.stat()
            source_stats.append({"path": record["path"], "bytes": stat.st_size,
                                 "mtime_ns": stat.st_mtime_ns})
    save_json(OUT / "execution_environment.json", {
        "python": platform.python_version(), "numpy": np.__version__,
        "ROOT": ROOT.gROOT.GetVersion(), "source_file_stats": source_stats,
        "source_fingerprint_scope": "Paths, sizes and mtimes; not full-file cryptographic hashes.",
        "source_data_modified": False,
        "run_info_duration_policy": "Do not infer exposure/rates from a reused run's latest run.info.",
        "histogram_association_scope": "All hardware-energy bin contents, including underflow and overflow; no absolute waveform timestamps inferred."})
    exact = sum(row["all_bin_counts_equal"] for row in comparisons)
    print(f"Histogram audit: {exact} exact histogram matches among {len(comparisons)} snapshots")


if __name__ == "__main__":
    main()
