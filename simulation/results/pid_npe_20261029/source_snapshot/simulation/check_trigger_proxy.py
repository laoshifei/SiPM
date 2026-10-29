"""Check a declared FIR timing proxy against recorded training pulses.

This does not implement or validate CAEN firmware: decimation, integer arithmetic,
arming/zero crossing, baseline recovery and dead time are not reproduced.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from reference import PROJECT, bin_layout
from screen_joint import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=False)
    report = {'scope': __doc__, 'formula': '4-sample causal moving average, followed by y[n]-2*y[n-48]+y[n-96]; 2 ns ADC samples',
              'setting_status': 'Current settings use 96 ns TTF delay and smoothing 4; equivalence of this FIR to the actual firmware and configuration-to-BIN association are unconfirmed.',
              'interpretation': 'Fractions below nominal thresholds demonstrate that this proxy cannot simply be assumed to reproduce all recorded triggers. They are not trigger inefficiencies.',
              'script_sha256': sha(Path(__file__)), 'runs': []}
    for run, threshold in ((3, 120.), (12, 160.)):
        path = PROJECT/f'data/DT5730_DPP_PHA/DAQ/run_{run}/UNFILTERED/Data_run_{run}.BIN'
        phase = PROJECT/f'analysis/paper_baseline/results/phase2/run{run:04d}_features.npz'
        _, dtype, count = bin_layout(path); raw = np.memmap(path, dtype=dtype, offset=2, mode='r', shape=(count,))
        with np.load(phase) as z: indices = np.flatnonzero(z['eligible_primary'] & (z['split'] == 0))
        peaks = []
        for start in range(0, len(indices), 256):
            y = np.array(raw['data'][indices[start:start+256]], dtype=float); y -= y[:, :128].mean(axis=1, keepdims=True)
            cumulative = np.c_[np.zeros(len(y)), np.cumsum(y, axis=1)]
            smoothed = (cumulative[:, 4:]-cumulative[:, :-4])/4
            filtered = smoothed[:, 96:]-2*smoothed[:, 48:-48]+smoothed[:, :-96]
            peaks.extend(filtered[:, 128:].max(axis=1))
        peaks = np.array(peaks)
        np.savez_compressed(args.out/f'run{run}_proxy.npz', event_indices=indices, peak_adc=peaks)
        report['runs'].append({'run': run, 'n': len(indices), 'nominal_threshold_adc': threshold,
            'peak_p01_p10_p50_p90_adc': np.quantile(peaks, [.01, .1, .5, .9]).tolist(),
            'recorded_fraction_below_nominal_threshold': float(np.mean(peaks < threshold)),
            'input_sha256': sha(path), 'phase_sha256': sha(phase)})
    (args.out/'trigger_proxy.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report['runs'], indent=2))


if __name__ == '__main__': main()
