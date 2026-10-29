"""Export frozen native-waveform checks, without changing calibration parameters."""
import csv
import json
from pathlib import Path
import numpy as np
from calibrate_signal_fluctuations import OUT, save
from reference import wave_features
from screen_response_candidates import ks
from diagnose_charge import sha


def arrays(path):
    with np.load(path) as z:
        return dict(z)


def features(waves):
    f = wave_features(waves)
    return {
        'peak_ADC': f['amplitude'],
        'peak_per_Q': f['amplitude'] / f['charge_long'],
        'tpeak_minus_t10_ns': 2 * (f['peak_sample'] - f['t10_sample']),
        'short_long_ratio': f['short_long_ratio'],
        'baseline_RMS_ADC': f['baseline_rms'],
    }


def main():
    rows = []
    extra = {
        'code_sha256': sha(Path(__file__)),
        'scope': 'Additional descriptive waveform checks in the unchanged common Q selection; no retuning.',
        'limits': 'KS is a distance, not a p-value. Unweighted Q distributions differ. Check segments were explored previously. Mean shapes are area-normalized; individual ADC examples are not amplitude-scaled.',
        'variants': {},
    }
    for variant in ('variance_scatter', 'independent'):
        folder = OUT / variant
        report = json.loads((folder / 'validation.json').read_text())
        entry = {'validation_sha256': sha(folder / 'validation.json'), 'sources': {}}
        for source, model_report in report['sources'].items():
            model = arrays(folder / f'{source}_native.npz')
            fm = features(model['waves'])
            source_extra = {}
            for group, c in model_report['comparisons'].items():
                draws = np.load(folder / f'{source}_{group}_relative_draws.npy')
                np.testing.assert_allclose(np.quantile(draws, [.025, .975], axis=0), c['relative_interval95'])
                for j, width in enumerate((50, 102, 202)):
                    rows.append({
                        'variant': variant, 'source': source, 'measured_group': group,
                        'independent_transport': report['independent_new_transport'],
                        'parent_events': model_report['nparent'], 'MC_common_n': model_report['common_n'],
                        'measured_n': c['n'], 'filter_ns': width,
                        'MC_F': model_report['F'][j], 'measured_F': c['measured_F'][j],
                        'relative': c['relative'][j], 'relative95_low': c['relative_interval95'][0][j],
                        'relative95_high': c['relative_interval95'][1][j],
                        'area_normalized_mean_NRMSE': c['shape_nrmse'],
                        'shape95_low': c['shape_interval95'][0], 'shape95_high': c['shape_interval95'][1],
                        'F_distribution_KS': c['filter_distribution_KS'][j],
                        'F_IQR_ratio_MC_over_data': c['filter_IQR_ratios'][j],
                        'common_Q_KS': c['common_Q_KS'],
                    })
                data = arrays(OUT / f'{group}.npz')
                fd = features(data['waves'])
                source_extra[group] = {}
                for name, x in fm.items():
                    y = fd[name]
                    assert np.isfinite(x).all() and np.isfinite(y).all()
                    source_extra[group][name] = {
                        'MC_q25_median_q75': np.quantile(x, [.25, .5, .75]).tolist(),
                        'data_q25_median_q75': np.quantile(y, [.25, .5, .75]).tolist(),
                        'KS': ks(x, y),
                    }
            entry['sources'][source] = source_extra
        extra['variants'][variant] = entry
    with (OUT / 'waveform_validation.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    save(OUT / 'waveform_feature_checks.json', extra)
    print(f'Exported {len(rows)} comparison rows; all stored 95% intervals reproduced.')


if __name__ == '__main__':
    main()
