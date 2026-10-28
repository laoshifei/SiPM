"""Record current CH0 settings and distinguish them from the historical BIN association."""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

PROJECT = Path(__file__).resolve().parents[1]
KEYS = ('SRV_PARAM_CH_COARSE_GAIN', 'SRV_PARAM_CH_THRESHOLD', 'SRV_PARAM_CH_TTF_DELAY',
        'SRV_PARAM_CH_TTF_SMOOTHING', 'SRV_PARAM_CH_POLARITY', 'SRV_PARAM_CH_PRETRG',
        'SRV_PARAM_CH_SELF_TRG_ENABLE', 'SRV_PARAM_CH_TRG_HOLDOFF', 'SRV_PARAM_CH_TRAP_TRISE',
        'SRV_PARAM_CH_TRAP_TFLAT', 'SRV_PARAM_CH_TRAP_POLEZERO', 'SRV_PARAM_CH_ENERGY_FINE_GAIN',
        'SW_PARAMETER_CH_ENERGYCUTENABLE', 'SW_PARAMETER_CH_ENERGYLOWCUT', 'SW_PARAMETER_CH_ENERGYHIGHCUT')


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=False)
    associations = json.loads((PROJECT/'analysis/paper_baseline/results/histogram_association.json').read_text())
    report = {'scope': 'Current saved settings, not proof of the settings used for every earlier BIN in a reused run directory',
              'hardware_sources': {
                  'range': {'url': 'https://caen.it/product/dt5730-dt5730s/', 'checked': '2026-10-28',
                            'paraphrase': 'DT5730 X1 corresponds to 2 Vpp, X4 to 0.5 Vpp; analog inputs have 50 ohm impedance.'},
                  'trigger': {'url': 'https://caen.it/products/dpp-pha/', 'checked': '2026-10-28',
                              'paraphrase': 'DPP-PHA self-trigger uses the RC-CR2 digital algorithm; raw pulse peak is not that filter output.'}},
              'snapshot_keys_checked': 'Matched run 3/12 Hcompass ROOT files contain spectra and live/real times, no saved settings object at top level.',
              'runs': []}
    for run in (3, 7, 10, 11, 12, 13, 14, 18):
        path = PROJECT/f'data/DT5730_DPP_PHA/DAQ/run_{run}/settings.xml'
        board = ET.parse(path).getroot().find('board')
        values = {entry.findtext('key'): entry.findtext('value/value') for entry in board.findall('parameters/entry')}
        channel = next(c for c in board.findall('channel') if c.findtext('index') == '0')
        overrides = {entry.findtext('key'): entry.findtext('value') for entry in channel.findall('values/entry')}
        values.update(overrides)
        report['runs'].append({'run': run, 'channel': 0, 'settings_path': str(path.relative_to(PROJECT)),
            'settings_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'board': board.findtext('modelName'),
            'sample_time_ps': board.findtext('sampleTime'), 'adc_bits': board.findtext('adcBitCount'),
            'parameters': {key: {'value': values[key], 'scope': 'channel override' if key in overrides else 'board inherited'} for key in KEYS},
            'range_implied_by_current_settings_mV': {'COARSE_GAIN_X1': 2000, 'COARSE_GAIN_X4': 500}[values['SRV_PARAM_CH_COARSE_GAIN']],
            'historical_bin_settings_association': 'unconfirmed',
            'exact_energy_histogram_matches': [r['snapshot'] for r in associations if r['run'] == run and r['all_bin_counts_equal']]})
    (args.out/'settings_audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'runs': len(report['runs']), 'saved_ranges_mV': [r['range_implied_by_current_settings_mV'] for r in report['runs']]}))


if __name__ == '__main__': main()
