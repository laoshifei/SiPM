#!/usr/bin/env python3
"""Read existing data; produce a provenance inventory and label-free PHA QC.

Only results/ and labels.csv next to this script are written. Source data and
existing experimental settings are never modified. Requires numpy and PyROOT.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import struct
import xml.etree.ElementTree as ET

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
OUT = HERE / "results"
PRIMARY_PHA = {3, 7, 10, 11, 12, 13, 14, 18}


def relative(path):
    return str(Path(path).resolve().relative_to(PROJECT))


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def save_csv(path, rows, fields=None):
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def settings(run_dir):
    path = run_dir / "settings.xml"
    if not path.exists():
        return {}
    board = ET.parse(path).getroot().find("board")
    defaults = {e.findtext("key"): e.findtext("value/value")
                for e in board.findall("parameters/entry")}
    units = {e.findtext("key"): e.findtext("value/descriptor/udm")
             for e in board.findall("parameters/entry")}
    channels = {}
    for node in board.findall("channel"):
        effective = dict(defaults)
        effective.update({e.findtext("key"): e.findtext("value")
                          for e in node.findall("values/entry")})
        channels[node.findtext("index")] = effective
    info = {}
    info_path = run_dir / "run.info"
    if info_path.exists():
        info = dict(line.split("=", 1) for line in info_path.read_text().splitlines()
                    if "=" in line)
    snapshots = sorted({match.group(1) for p in run_dir.rglob("Hcompass*.root")
                        if (match := re.search(r"_(\d{8}_\d{6})\.root$", p.name))})
    return {"path": relative(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "board": {k: board.findtext(k) for k in
                      ("modelName", "serialNumber", "dppType", "adcBitCount", "sampleTime")},
            "units": units, "channels": channels, "run_info": info,
            "histogram_snapshot_times": snapshots,
            "metadata_association_warning": len(snapshots) > 1}


def compass_dtype(header, length):
    if not header & 8:
        raise ValueError("No saved waveform in CoMPASS header")
    fields = [("board", "<i2"), ("channel", "<i2"), ("timestamp", "<u8")]
    if header & 1:
        fields.append(("energy_ch", "<u2"))
    if header & 2:
        fields.append(("energy_keV", "<f8"))
    if header & 4:
        fields.append(("energy_short", "<u2"))
    fields.extend([("flags", "<u4"), ("probe", "u1"), ("size", "<u4"),
                   ("data", "<i2", (length,))])
    return np.dtype(fields)


def bin_layout(path):
    """Fixed-length fast path. Every event length is checked again while reading.

    A variable-length or truncated file is rejected, never silently reinterpreted.
    """
    total = path.stat().st_size
    if total < 2:
        raise ValueError(f"Truncated file header: {path}")
    with path.open("rb") as handle:
        header = struct.unpack("<H", handle.read(2))[0]
        if total == 2:
            return header, None, 0
        prefix = compass_dtype(header, 0)
        raw = handle.read(prefix.itemsize)
        if len(raw) != prefix.itemsize:
            raise ValueError(f"Truncated event prefix: {path}")
        length = int(np.frombuffer(raw, dtype=prefix)["size"][0])
    if not 1 <= length <= 10000:
        raise ValueError(f"Invalid waveform length {length}: {path}")
    dtype = compass_dtype(header, length)
    count, remainder = divmod(total - 2, dtype.itemsize)
    if remainder:
        raise ValueError(f"Variable-length or truncated file (remainder {remainder}): {path}")
    return header, dtype, count


def bin_chunks(path, dtype, count, chunk_size=1024):
    with path.open("rb") as handle:
        handle.seek(2)
        seen = 0
        while seen < count:
            records = np.fromfile(handle, dtype=dtype, count=min(chunk_size, count - seen))
            if len(records) == 0 or np.any(records["size"] != dtype["data"].shape[0]):
                raise ValueError(f"Variable-length or truncated records near {seen}: {path}")
            if np.any(records["board"] < 0) or np.any(records["channel"] < 0):
                raise ValueError(f"Invalid board/channel near {seen}: {path}")
            yield seen, records
            seen += len(records)
        if handle.read(1):
            raise ValueError(f"Unexpected trailing bytes: {path}")


def wave_features(waves, adc_bits=14):
    """Diagnostic features in ADC and samples, without particle labels or cuts.

    A 10% rising-edge estimate uses the last below-threshold sample preceding the
    global maximum after sample 127. It is a pilot alignment, not a pileup finder.
    Long/short windows share [t10-20, ...); endpoints are t10+1500 / t10+250.
    Integrals outside the recorded window are NaN, never padded or truncated.
    """
    wave = np.asarray(waves, dtype=np.float64)
    if wave.ndim != 2 or wave.shape[1] < 256:
        raise ValueError("QC needs at least 256 samples per waveform")
    n, length = wave.shape
    baseline = wave[:, :128].mean(axis=1)
    rms = wave[:, :128].std(axis=1)
    y = wave - baseline[:, None]
    peak = y[:, 128:].argmax(axis=1) + 128
    index = np.arange(n)
    amp = y[index, peak]
    positions = np.arange(length)[None, :]
    below = (positions < peak[:, None]) & (y <= 0.1 * amp[:, None])
    before = np.where(below, positions, -1).max(axis=1)
    has_crossing = (before >= 0) & (before < peak) & (amp > 0)
    lo = np.maximum(before, 0)
    hi = np.minimum(lo + 1, length - 1)
    step = y[index, hi] - y[index, lo]
    fraction = np.divide(0.1 * amp - y[index, lo], step,
                         out=np.zeros(n), where=step > 0)
    t10 = np.where(has_crossing, lo + fraction, np.nan)
    anchor = np.where(has_crossing, hi, 0)
    start = anchor - 20
    stop = anchor + 1500
    full_window = has_crossing & (start >= 128) & (stop <= length)
    cumulative = np.c_[np.zeros(n), np.cumsum(y, axis=1)]

    def charge(end_offset):
        end = anchor + end_offset
        good = has_crossing & (start >= 128) & (end <= length)
        q = cumulative[index, np.clip(end, 0, length)] - cumulative[index, np.clip(start, 0, length)]
        return np.where(good, q, np.nan)

    qlong = charge(1500)
    qshort = charge(250)
    ratio = np.divide(qshort, qlong, out=np.full(n, np.nan), where=qlong > 0)
    rails = (wave.min(axis=1) <= 0) | (wave.max(axis=1) >= 2**adc_bits - 1)
    # These are reporting flags. No events are removed from saved features.
    flags = (rails.astype(np.uint8) | ((amp <= 0).astype(np.uint8) << 1)
             | ((amp <= 5 * rms).astype(np.uint8) << 2)
             | ((~full_window).astype(np.uint8) << 3)
             | ((~np.isfinite(qlong) | (qlong <= 0)).astype(np.uint8) << 4))
    return {"baseline": baseline, "baseline_rms": rms,
            "baseline_first64": wave[:, :64].mean(axis=1),
            "baseline_half_difference": wave[:, 64:128].mean(axis=1) - wave[:, :64].mean(axis=1),
            "amplitude": amp, "peak_sample": peak.astype(np.int32), "t10_sample": t10,
            "tail_level": y[:, -128:].mean(axis=1), "charge_full": y.sum(axis=1),
            "charge_long": qlong, "charge_short": qshort,
            "charge_1us": charge(500), "charge_2us": charge(1000),
            "short_long_ratio": ratio, "diagnostic_flags": flags}


def quantiles(values):
    good = np.asarray(values)[np.isfinite(values)]
    return [float(x) for x in np.quantile(good, [0.05, 0.5, 0.95])] if len(good) else [None] * 3


def pha_inventory():
    rows, summaries = [], []
    base = PROJECT / "data/DT5730_DPP_PHA/DAQ"
    paths = sorted(base.glob("run_*/UNFILTERED/*.BIN"),
                   key=lambda p: int(p.parents[1].name.split("_")[1]))
    for path in paths:
        run = int(path.parents[1].name.split("_")[1])
        header, dtype, count = bin_layout(path)
        config = settings(path.parents[1])
        rec = {"system": "PHA", "run": run, "path": relative(path),
               "file_bytes": path.stat().st_size, "header_hex": hex(header),
               "entries": count, "config": config, "channels": []}
        if not count:
            rows.append(rec)
            print(f"PHA {run}: empty", flush=True)
            continue
        length = dtype["data"].shape[0]
        rec["samples"] = length
        fields = Counter()
        hardware_flags = Counter()
        probes = Counter()
        feature_parts = {}
        backsteps = duplicates = 0
        last = None
        sample_ids = np.unique(np.linspace(0, count - 1, min(count, 32), dtype=np.int64))
        sampled = []
        bits = int(config["board"]["adcBitCount"])
        for offset, chunk in bin_chunks(path, dtype, count):
            fields.update(zip(chunk["board"].tolist(), chunk["channel"].tolist()))
            hardware_flags.update(int(v) for v in chunk["flags"])
            probes.update(int(v) for v in chunk["probe"])
            stamps = chunk["timestamp"]
            if last is not None:
                backsteps += int(stamps[0] < last)
                duplicates += int(stamps[0] == last)
            backsteps += int(np.count_nonzero(stamps[1:] < stamps[:-1]))
            duplicates += int(np.count_nonzero(stamps[1:] == stamps[:-1]))
            last = stamps[-1]
            f = wave_features(chunk["data"], bits)
            f.update({key: chunk[key].copy() for key in chunk.dtype.names if key != "data"})
            f["event_index"] = np.arange(offset, offset + len(chunk), dtype=np.int64)
            for key, val in f.items():
                feature_parts.setdefault(key, []).append(val)
            select = sample_ids[(sample_ids >= offset) & (sample_ids < offset + len(chunk))] - offset
            if len(select):
                sampled.append(chunk["data"][select].copy())
        features = {key: np.concatenate(value) for key, value in feature_parts.items()}
        rec["timestamp_backsteps"] = backsteps
        rec["adjacent_equal_timestamps"] = duplicates
        rec["hardware_flags"] = {str(k): v for k, v in sorted(hardware_flags.items())}
        rec["waveform_probe_codes"] = {str(k): v for k, v in sorted(probes.items())}
        for (board, channel), events in sorted(fields.items()):
            effective = config["channels"].get(str(channel), {})
            if effective.get("SRV_PARAM_CH_POLARITY") != "POLARITY_POSITIVE":
                raise ValueError(f"Positive-polarity feature assumption invalid: {path}, CH{channel}")
            period = float(effective["SRV_PARAM_RECLEN"]) / length
            if config["units"].get("SRV_PARAM_RECLEN") != "NANOS" or period != 2:
                raise ValueError("Pilot 1/2 us windows require a 2 ns sample interval")
            rec["channels"].append({"board": board, "channel": channel, "entries": events,
                                    "sample_period_ns": period,
                                    "period_basis": "settings record length (NANOS) / saved sample count"})
            mask = (features["board"] == board) & (features["channel"] == channel)
            flags = features["diagnostic_flags"][mask]
            summary = {"system": "PHA", "run": run, "board": board, "channel": channel,
                       "events": events, "samples": length, "sample_period_ns": period,
                       "threshold_adc": effective.get("SRV_PARAM_CH_THRESHOLD"),
                       "timestamp_backsteps": backsteps,
                       "metadata_association_warning": config["metadata_association_warning"],
                       "basic_window_eligible": int(np.count_nonzero(flags == 0))}
            for label, bit in [("rail_touch", 0), ("nonpositive_amplitude", 1),
                               ("amplitude_le_5rms", 2), ("incomplete_window", 3),
                               ("nonpositive_or_missing_charge", 4)]:
                summary[label] = int(np.count_nonzero(flags & (1 << bit)))
            for key in ["baseline", "baseline_rms", "amplitude", "t10_sample", "charge_long", "tail_level"]:
                for q, value in zip(("p05", "p50", "p95"), quantiles(features[key][mask])):
                    summary[key + "_" + q] = value
            summaries.append(summary)
        np.savez_compressed(OUT / f"pha_run{run:04d}_features.npz", **features)
        np.savez_compressed(OUT / f"pha_run{run:04d}_wave_sample.npz", event_index=sample_ids,
                            data=np.concatenate(sampled))
        rows.append(rec)
        print(f"PHA {run}: {count} records, channels {dict(fields)}, QC saved", flush=True)
    save_json(OUT / "pha_inventory.json", rows)
    save_csv(OUT / "pha_qc.csv", summaries)


def root_inventory():
    import ROOT
    ROOT.gROOT.SetBatch(True)
    rows = []

    def scalars(path, tree_name, columns):
        frame = ROOT.RDataFrame(tree_name, str(path))
        for name, expression in columns.items():
            frame = frame.Define("audit_" + name, expression)
        values = frame.AsNumpy(["audit_" + name for name in columns])
        return {name: np.asarray(values["audit_" + name]) for name in columns}

    def group_counts(values, columns):
        data = np.column_stack([np.asarray(values[c], dtype=np.int64) for c in columns])
        groups, counts = np.unique(data, axis=0, return_counts=True)
        return [dict(zip(columns, (int(v) for v in group)), entries=int(count))
                for group, count in zip(groups, counts)]

    for path in sorted((PROJECT / "data/DT5730_DPP_PSD/DAQ").glob("run_*/UNFILTERED/Data_*.root")):
        run = int(path.parents[1].name.split("_")[1])
        arr = scalars(path, "Data", {"board": "int(Board)", "channel": "int(Channel)",
                                     "samples": "int(Samples.GetSize())", "flags": "ULong64_t(Flags)"})
        raw = path.parents[1] / "RAW" / f"DataR_run_{run}.root"
        f = ROOT.TFile.Open(str(raw))
        raw_count = int(f.Get("Data_R").GetEntries())
        f.Close()
        rows.append({"system": "PSD_DT5730", "run": run, "path": relative(path),
                     "tree": "Data", "entries": len(arr["channel"]),
                     "groups": group_counts(arr, ["board", "channel", "samples"]),
                     "hardware_flags": {str(int(k)): int(v) for k, v in zip(*np.unique(arr["flags"], return_counts=True))},
                     "raw_path": relative(raw), "raw_entries": raw_count,
                     "config": settings(path.parents[1])})
        print(f"Native PSD {run}: {rows[-1]['groups']}", flush=True)

    for path in sorted((PROJECT / "DT5720_DPP_PSD/raw2root").glob("run*.root")):
        f = ROOT.TFile.Open(str(path))
        names = sorted({k.GetName() for k in f.GetListOfKeys() if k.GetClassName() == "TTree"})
        for name in names:
            tree = f.Get(name)
            if not tree.GetEntries():
                continue
            arr = scalars(path, name, {"board": "int(board)", "channel": "int(channel)", "samples": "int(size)"})
            rows.append({"system": "PSD_legacy", "run": int(path.stem[3:]),
                         "path": relative(path), "tree": name, "entries": int(tree.GetEntries()),
                         "groups": group_counts(arr, ["board", "channel", "samples"]),
                         "config": {}, "provenance": "decoded ROOT only; acquisition configuration not linked"})
            print(f"Legacy PSD {path.name}: {rows[-1]['groups']}", flush=True)
        f.Close()

    xia_hash = hashlib.sha256()
    xia_paths = sorted((PROJECT / "data/XIA/rootfile").glob("data_C1_*_wave.root"))
    key_dtype = np.dtype([("crate", "<i4"), ("slot", "<i4"), ("channel", "<i4"),
                          ("timestamp", "<i8"), ("energy", "<i4"), ("samples", "<i4")])
    xia_columns = {"crate": "int(cid)", "slot": "int(sid)", "channel": "int(ch)",
                   "samples": "int(ltra)", "sample_rate": "int(sr)",
                   "timestamp": "Long64_t(ts)", "energy": "int(evte)",
                   "pileup": "int(pileup)", "out_of_range": "int(outofr)"}

    def hash_keys(arr, hasher):
        packed = np.empty(len(arr["channel"]), dtype=key_dtype)
        for key in key_dtype.names:
            packed[key] = arr[key]
        hasher.update(packed.tobytes())

    for path in xia_paths:
        arr = scalars(path, "tree", xia_columns)
        hash_keys(arr, xia_hash)
        rows.append({"system": "XIA", "run": int(path.stem.split("_")[-2]),
                     "path": relative(path), "tree": "tree", "entries": len(arr["channel"]),
                     "groups": group_counts(arr, ["crate", "slot", "channel", "samples", "sample_rate"]),
                     "pileup_flag_events": int(np.count_nonzero(arr["pileup"])),
                     "out_of_range_flag_events": int(np.count_nonzero(arr["out_of_range"])),
                     "config": {}})
        print(f"XIA {path.name}: {rows[-1]['entries']} entries", flush=True)
    merged = PROJECT / "data/XIA/rootfile/data_all.root"
    arr = scalars(merged, "tree", xia_columns)
    merged_hash = hashlib.sha256()
    hash_keys(arr, merged_hash)
    match = merged_hash.hexdigest() == xia_hash.hexdigest()
    wave_checks = []
    mf = ROOT.TFile.Open(str(merged))
    mt = mf.Get("tree")
    offset = 0
    for path in xia_paths:
        sf = ROOT.TFile.Open(str(path))
        st = sf.Get("tree")
        for entry in sorted({0, int(st.GetEntries()) // 2, int(st.GetEntries()) - 1}):
            st.GetEntry(entry)
            mt.GetEntry(offset + entry)
            same = int(st.ltra) == int(mt.ltra) and list(st.data) == list(mt.data)
            wave_checks.append({"source": relative(path), "entry": entry, "equal_waveform": same})
        offset += int(st.GetEntries())
        sf.Close()
    mf.Close()
    save_json(OUT / "xia_merge_audit.json", {"merged_path": relative(merged),
                "entries": len(arr["channel"]), "source_concatenation": [relative(p) for p in xia_paths],
                "ordered_scalar_keys_equal": match, "source_key_sha256": xia_hash.hexdigest(),
                "merged_key_sha256": merged_hash.hexdigest(), "sample_waveform_checks": wave_checks,
                "key_fields": list(key_dtype.names),
                "scope": "All scalar event keys and six waveforms; not a full waveform byte comparison.",
                "use_as_additional_sample": False})
    save_json(OUT / "root_inventory.json", rows)


def label_inventory():
    rows = []
    pha_path, root_path = OUT / "pha_inventory.json", OUT / "root_inventory.json"
    if pha_path.exists():
        for run in json.loads(pha_path.read_text()):
            for channel in run["channels"]:
                rows.append({"key": f"PHA:{run['run']}:B{channel['board']}:CH{channel['channel']}",
                             "priority": "1" if run["run"] in PRIMARY_PHA else "2",
                             "system": "PHA", "run": run["run"], "crate": "", "slot": "",
                             "board": channel["board"], "channel": channel["channel"],
                             "events": channel["entries"], "path": run["path"]})
    if root_path.exists():
        for run in json.loads(root_path.read_text()):
            grouped = Counter()
            for g in run["groups"]:
                grouped[(g.get("crate", ""), g.get("slot", ""), g.get("board", ""), g["channel"])] += g["entries"]
            for (crate, slot, board, channel), entries in sorted(grouped.items()):
                key = f"{run['system']}:{run['run']}:C{crate}:S{slot}:B{board}:CH{channel}"
                rows.append({"key": key, "priority": "3", "system": run["system"], "run": run["run"],
                             "crate": crate, "slot": slot, "board": board, "channel": channel,
                             "events": entries, "path": run["path"]})
    if not rows:
        raise RuntimeError("Run an inventory stage first")
    save_csv(OUT / "run_channels.csv", rows)
    label_path = HERE / "labels.csv"
    existing = {}
    if label_path.exists():
        with label_path.open(encoding="utf-8-sig", newline="") as handle:
            existing = {row["key"]: row for row in csv.DictReader(handle)}
    manual = ["analysis_status", "source_or_background", "particle_content", "label_basis", "detector_setup",
              "bias_voltage_V", "temperature_C", "repeat_group", "run_reused_or_changed", "notes"]
    fields = list(rows[0]) + manual
    current_keys = set()
    for row in rows:
        current_keys.add(row["key"])
        for col in manual:
            row[col] = existing.get(row["key"], {}).get(col, "")
        if row["analysis_status"] == "exclude_user":
            row["priority"] = "excluded"
    # Retain prior user records even if an optional source becomes unavailable.
    for key, old in existing.items():
        if key not in current_keys:
            rows.append({col: old.get(col, "") for col in fields})
    save_csv(label_path, rows, fields)
    print(f"Label template: {len(rows)} run/channel rows (existing answers preserved)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["pha", "root", "labels", "all"])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage in ("pha", "all"):
        pha_inventory()
    if args.stage in ("root", "all"):
        root_inventory()
    if args.stage in ("labels", "all"):
        label_inventory()


if __name__ == "__main__":
    main()
