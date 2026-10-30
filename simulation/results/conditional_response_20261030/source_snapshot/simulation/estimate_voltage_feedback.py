"""Conditional ADC-terminal voltage bounds; not a sensor feedback calibration."""
import argparse
import json
from pathlib import Path
from diagnose_charge import sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--covariance', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    report = json.loads(args.covariance.read_text())
    ranges = (500., 2000.)
    overvoltages = (.5, 1., 3., 5.)
    result = {
        'scope': __doc__, 'adc_bits': 14, 'adc_range_mV': ranges,
        'conditional_overvoltage_V': overvoltages,
        'selection': 'First60 frozen common-charge pulse cohorts in covariance.json; event peak amplitudes, not the peak of the average waveform.',
        'limitations': [
            'Historical BIN-to-input-range association is not established; present XML supports 500 mV while 2000 mV is the older alternative.',
            'Overvoltages are scenarios, not inferred from nominal bias or typical breakdown voltage.',
            'Terminal voltage is not necessarily the SiPM bias-node change. Unknown summing, coupling and load prevent assigning or excluding a feedback mechanism.',
            'The reported fraction assumes equal terminal and sensor-node excursions; it is not a general upper bound on gain or PDE changes.'
        ],
        'input_sha256': sha(args.covariance), 'code_sha256': sha(Path(__file__)), 'runs': {}
    }
    for run in (3, 7, 10, 11, 12, 13, 14, 18):
        amplitude = report['groups'][f'run{run}_first60']['adc_amplitude_p10_p50_p90']
        rows = {}
        for span in ranges:
            volts = [a * span / (2**14) for a in amplitude]
            rows[str(span)] = {
                'amplitude_mV_p10_p50_p90': volts,
                'median_terminal_fraction_of_overvoltage': {str(ov): volts[1] / (1000*ov) for ov in overvoltages}
            }
        result['runs'][str(run)] = rows
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
