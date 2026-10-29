#!/usr/bin/env python3
"""Small correctness checks for the new parser/features, plus raw-to-ROOT parity."""
from pathlib import Path
import csv
import json
import struct
import tempfile
import unittest

import numpy as np

import prepare


class ParserAndFeatures(unittest.TestCase):
    def test_analytic_rectangular_pulse(self):
        wave = np.full((1, 2496), 1000, dtype=np.int16)
        wave[:, 500:2000] += 100
        result = prepare.wave_features(wave)
        self.assertAlmostEqual(result["baseline"][0], 1000)
        self.assertAlmostEqual(result["t10_sample"][0], 499.1)
        self.assertAlmostEqual(result["charge_long"][0], 150000)
        self.assertAlmostEqual(result["charge_short"][0], 25000)
        self.assertAlmostEqual(result["short_long_ratio"][0], 1 / 6)
        self.assertEqual(result["diagnostic_flags"][0], 0)
        shifted = np.full_like(wave, 1000)
        shifted[:, 600:2100] += 100
        self.assertAlmostEqual(prepare.wave_features(shifted)["charge_long"][0], 150000)

    def test_short_record_is_not_silently_completed(self):
        wave = np.full((1, 2496), 1000, dtype=np.int16)
        wave[:, 1500:2496] += 100
        result = prepare.wave_features(wave)
        self.assertTrue(np.isnan(result["charge_long"][0]))
        self.assertTrue(result["diagnostic_flags"][0] & 8)

    def test_rail_and_absent_signal_flags(self):
        wave = np.full((2, 2496), 1000, dtype=np.int16)
        wave[0, 500:2000] = 16383
        result = prepare.wave_features(wave)
        self.assertTrue(result["diagnostic_flags"][0] & 1)
        self.assertTrue(result["diagnostic_flags"][1] & 2)

    def test_binary_layout_integrity_and_timestamp_precision(self):
        # Independently pack the documented byte sequence, not the NumPy dtype.
        prefix = struct.pack("<hhQHIBI", 0, 2, 2**60 + 3, 1234, 0xFEDCBA98, 1, 256)
        values = np.arange(256, dtype="<i2").tobytes()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "raw.BIN"
            path.write_bytes(struct.pack("<H", 0xCAE9) + prefix + values)
            _, dtype, count = prepare.bin_layout(path)
            self.assertEqual(count, 1)
            _, records = next(prepare.bin_chunks(path, dtype, count))
            self.assertEqual(int(records["timestamp"][0]), 2**60 + 3)
            self.assertEqual(int(records["flags"][0]), 0xFEDCBA98)
            self.assertEqual(records["data"][0, -1], 255)
            path.write_bytes(path.read_bytes()[:-1])
            with self.assertRaises(ValueError):
                prepare.bin_layout(path)

    def test_per_event_lengths_are_checked(self):
        dtype = prepare.compass_dtype(0xCAE9, 256)
        events = np.zeros(2, dtype=dtype)
        events["size"] = [256, 128]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "raw.BIN"
            path.write_bytes(struct.pack("<H", 0xCAE9) + events.tobytes())
            _, parsed, count = prepare.bin_layout(path)
            with self.assertRaises(ValueError):
                list(prepare.bin_chunks(path, parsed, count, chunk_size=1))

    def test_refresh_preserves_source_answers_and_exclusions(self):
        old_here, old_out = prepare.HERE, prepare.OUT
        try:
            with tempfile.TemporaryDirectory() as tmp:
                prepare.HERE = Path(tmp)
                prepare.OUT = Path(tmp) / "results"
                prepare.OUT.mkdir()
                inventory = [{"run": 3, "path": "example.BIN", "channels": [
                    {"board": 0, "channel": 0, "entries": 10}]}]
                (prepare.OUT / "pha_inventory.json").write_text(json.dumps(inventory))
                prepare.label_inventory()
                path = prepare.HERE / "labels.csv"
                with path.open(encoding="utf-8-sig", newline="") as f:
                    rows = list(csv.DictReader(f))
                rows[0]["source_or_background"] = "241Am"
                rows[0]["particle_content"] = "α/γ 混合源"
                rows[0]["analysis_status"] = "exclude_user"
                prepare.save_csv(path, rows)
                prepare.label_inventory()
                with path.open(encoding="utf-8-sig", newline="") as f:
                    result = next(csv.DictReader(f))
                self.assertEqual(result["source_or_background"], "241Am")
                self.assertEqual(result["particle_content"], "α/γ 混合源")
                self.assertEqual(result["analysis_status"], "exclude_user")
                self.assertEqual(result["priority"], "excluded")
        finally:
            prepare.HERE, prepare.OUT = old_here, old_out


def raw_root_parity(test_count):
    import ROOT
    path = prepare.PROJECT / "data/DT5730_DPP_PHA/DAQ/run_3/UNFILTERED/Data_run_3.BIN"
    _, dtype, count = prepare.bin_layout(path)
    root_path = prepare.PROJECT / "DT5730_DPP_PHA/raw2root/run0003.root"
    f = ROOT.TFile.Open(str(root_path))
    tree = f.Get("tr_ch00")
    assert int(tree.GetEntries()) == count
    checks = []
    with path.open("rb") as handle:
        for index in [0, count // 2, count - 1]:
            handle.seek(2 + index * dtype.itemsize)
            record = np.fromfile(handle, dtype=dtype, count=1)[0]
            tree.GetEntry(index)
            for key in ["board", "channel", "timestamp", "energy_ch", "size"]:
                assert int(record[key]) == int(getattr(tree, key)), (index, key)
            assert np.array_equal(record["data"], np.asarray(list(tree.data), dtype=np.int16))
            checks.append(index)
    f.Close()
    prepare.save_json(prepare.OUT / "verification.json", {
        "parser_feature_and_provenance_tests_passed": test_count,
        "raw_root_parity": {"run": 3, "channel": 0, "entries": count,
                            "event_indices": checks, "all_samples_equal": True,
                            "scope": "Three complete waveforms and their stored scalar fields"},
        "raw_path": prepare.relative(path), "root_path": prepare.relative(root_path)})
    print("Raw BIN vs existing ROOT: count and three complete records agree")


if __name__ == "__main__":
    outcome = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ParserAndFeatures))
    if not outcome.wasSuccessful():
        raise SystemExit(1)
    raw_root_parity(outcome.testsRun)
