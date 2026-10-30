"""Check frozen joint calibration with new native mothers and unchanged data."""
from pathlib import Path
import numpy as np
from prepare_joint_constraints import OUT, TAIL, NPE, PROJECT, DATA, WIDTH, T, SOURCES, relative
from prepare_pid import read
from extract_pid import load
from analyze_co_tail import features, bootstrap, finite, SLICES, SEGMENTS
from analyze_npe_validation import raw_shapes
from analyze_bandwidth_validation import per_bin, widths, rms_width, interval
from fit_pid_bandwidth import average_bins
from select_joint_constraints import evaluate
from reference import TIME_NS, wave_features
from diagnose_pid_bandwidth import METHODS
from publication_stats import save, sha


def main():
    assert not (OUT/'assessment.json').exists()
    protocol = read(OUT/'validation/protocol.json'); design = read(OUT/'design.json')
    assert len(read(OUT/'validation/execution.json')['jobs']) == 2
    edges = np.asarray(protocol['q_edges']); mc = {}; draws = {}; means = {}; inputs = {}
    previous = read(TAIL/'assessment.json'); oldboot = load(TAIL/'bootstrap.npz')
    inputs[relative(TAIL/'bootstrap.npz')] = sha(TAIL/'bootstrap.npz')
    report = {'scope': __doc__, 'code_sha256': sha(Path(__file__)), 'protocol_sha256': sha(OUT/'validation/protocol.json'),
        'selection_sha256': sha(OUT/'selection.json'), 'methods': METHODS, 'checks': {}, 'mean_time_slices_ns': SLICES.tolist(),
        'new_independent_mothers': 32000, 'calibration_independent_mothers': 96000,
        'models': {'npe': 'Earlier photon-statistics candidate', 'tail': 'Previous tail-only candidate', 'joint': 'Fresh native joint candidate'},
        'uncertainty': '512 whole-mother MC draws; original-index data block64/128. Reuse frozen previous reference/data draws exactly; independently draw new validation mothers. Fits, selection, closure and apparatus fixed; pointwise intervals, no multiplicity correction.',
        'limits': 'First60 used in calibration; later data previously explored, not new blind experiment. Old models descriptive references. Source/run-specific response calibration; no updated matched-state PID ROC or experimental particle efficiency.'}
    for source, run, kind in SOURCES:
        key = f'run{run}_{source}'; support = protocol['support_bins'][source]
        for name, folder in [('npe', NPE/'validation'), ('tail', TAIL), ('joint', OUT/'validation')]:
            path = folder/'arrays'/f'{key}.npz'; a = load(path)
            t10 = wave_features(a['waves'])['t10_sample']
            assert np.all(2*(a['waves'].shape[1]-1-t10) >= 3800), 'Extended tail must have real recorded support'
            density, tail, pieces = features(a['waves'], a['scores'])
            row = {'ids': a['ids'], 'q': a['q'], 'scores': a['scores'], 'F': a['metrics'], 'density': density,
                'tail': tail, 'pieces': pieces, 'shape': a['shapes'], 'raw': raw_shapes(a['waves']), 'waves': a['waves'], 'metrics': a['metrics']}
            mc[(run, name)] = row
            if name == 'joint':
                qs, ms = bootstrap(row, edges, support, int(a['n_parent']), 401029550+run)
            else:
                oldname = 'old' if name == 'npe' else 'new'
                qs = oldboot[f'{key}__{oldname}__quartiles']; ms = oldboot[f'{key}__{oldname}__moments']
            draws[f'{key}__{name}__quartiles'] = qs; draws[f'{key}__{name}__moments'] = ms
            means[f'{key}__{name}__density'] = average_bins(density, a['q'], edges, support)
            means[f'{key}__{name}__ADC'] = average_bins(row['raw'], a['q'], edges, support)
            inputs[relative(path)] = sha(path)
        for segment in SEGMENTS:
            group = f'run{run}_{segment}'; dp = DATA/f'{group}.npz'; sp = WIDTH/'arrays'/f'measured_{group}.npz'
            data = load(dp); ds = load(sp); np.testing.assert_array_equal(data['ids'], ds['ids']); u = ds['main_band']
            t10 = wave_features(data['waves'][u])['t10_sample']
            assert np.all(2*(data['waves'].shape[1]-1-t10) >= 3800), 'Do not extrapolate measured samples'
            density, tail, pieces = features(data['waves'][u], ds['scores'][u])
            d = {'ids': ds['ids'][u], 'q': ds['q'][u], 'scores': ds['scores'][u], 'F': data['metrics'][u, :, 3],
                'tail': tail, 'pieces': pieces, 'density': density, 'shape': data['shapes'][u], 'raw': raw_shapes(data['waves'][u])}
            dq = per_bin(d['q'], d['scores'], edges); dt = average_bins(tail, d['q'], edges, support)
            df = average_bins(d['F'], d['q'], edges, support); da = average_bins(d['raw'], d['q'], edges, support)
            dshape = average_bins(d['shape'], d['q'], edges, support); den = rms_width(widths(dq), support)
            means[group+'__data_density'] = average_bins(density, d['q'], edges, support); means[group+'__data_ADC'] = da
            for size in [64, 128]:
                for what in ['quartiles', 'moments']:
                    k = f'{group}__block{size}__{what}'; draws[k] = oldboot[k]
            row = {'run': run, 'source': source, 'segment': segment, 'experimental_n': len(d['ids']), 'support_bins': support,
                'data_tail_metrics': dt.tolist(), 'data_F': df.tolist(), 'data_quartiles': dq.tolist(), 'models': {}}
            region = (TIME_NS >= -40)&(TIME_NS <= 3000)
            extra = (TIME_NS >= 3000)&(TIME_NS < 3800)
            data_extra = average_bins(d['raw'][:, extra].sum(axis=1)/d['q'], d['q'], edges, support)
            for name in ['npe', 'tail', 'joint']:
                a = mc[(run, name)]; point = per_bin(a['q'], a['scores'], edges)
                am = average_bins(a['tail'], a['q'], edges, support); af = average_bins(a['F'], a['q'], edges, support)
                aq = draws[f'{key}__{name}__quartiles']; ab = draws[f'{key}__{name}__moments']
                aa = average_bins(a['raw'], a['q'], edges, support); ashape = average_bins(a['shape'], a['q'], edges, support)
                delta = average_bins(a['pieces'], a['q'], edges, support)-average_bins(d['pieces'], d['q'], edges, support)
                np.testing.assert_allclose(delta.sum(), am[0]-dt[0], atol=1e-9)
                item = {'n_MC': len(a['ids']), 'bin_counts': np.histogram(a['q'], edges)[0].tolist(), 'MC_quartiles': point.tolist(),
                    'IQR_ratio': (rms_width(widths(point), support)/den).tolist(), 'mean_tail_metrics': am.tolist(),
                    'tail_difference': (am-dt).tolist(), 'mean_time_slice_difference_ns': delta.tolist(),
                    'median_shift_in_data_IQR': np.mean(((point[:, 1, :]-dq[:, 1, :])/widths(dq))[support], axis=0).tolist(),
                    'F_ratio': (af/df).tolist(), 'MC_F': af.tolist(),
                    'mean_shape_normalized_RMSE': float(np.sqrt(np.mean((ashape[region]-dshape[region])**2))/dshape.max()),
                    'mean_ADC_normalized_RMSE': float(np.sqrt(np.mean((aa[region]-da[region])**2))/da.max()),
                    'mean_ADC_peak_ratio': float(aa.max()/da.max()), 'uncertainty_by_data_block': {}}
                model_extra = average_bins(a['raw'][:, extra].sum(axis=1)/a['q'], a['q'], edges, support)
                item['extended_tail_3000_3800_ns'] = {'data_charge_over_Qlong': float(data_extra),
                    'MC_charge_over_Qlong': float(model_extra), 'difference': float(model_extra-data_extra),
                    'ADC_NRMSE_over_full_peak': float(np.sqrt(np.mean((aa[extra]-da[extra])**2))/da.max()),
                    'scope': 'Post-selection point diagnostic outside the fitted/formal 3us gate; fractionally aligned ADC sum over 3000<=t<3800ns divided by the original integer-gate Qlong. Not a probability or an equivalence test; no additional interval or refit.'}
                for size in [64, 128]:
                    bq = draws[f'{group}__block{size}__quartiles']; bm = draws[f'{group}__block{size}__moments']
                    br = rms_width(widths(aq), support)/rms_width(widths(bq), support)
                    dm = ab[:, :3]-bm[:, :3]; fr = ab[:, 3:]/bm[:, 3:]
                    item['uncertainty_by_data_block'][str(size)] = {
                        'IQR_ratio_CI95': [interval(br[:, j]) for j in range(3)],
                        'tail_difference_CI95': [interval(dm[:, j]) for j in range(3)],
                        'F_ratio_CI95': [interval(fr[:, j]) for j in range(3)]}
                if name != 'joint':
                    ref = previous['checks'][group]['models']['old' if name == 'npe' else 'new']
                    for field in ['IQR_ratio', 'F_ratio', 'tail_difference', 'mean_ADC_normalized_RMSE']:
                        np.testing.assert_allclose(item[field], ref[field], atol=1e-11, rtol=1e-11)
                item['joint_criterion_check'] = evaluate(a, d, data['waves'][u], edges, support, design['ranking_scales'])
                row['models'][name] = item
            report['checks'][group] = row
            for p in [dp, sp]: inputs[relative(p)] = sha(p)
            new = row['models']['joint']
            print(group, 'mean ns', round(new['tail_difference'][0], 3), 'width', np.round(new['IQR_ratio'], 3),
                  'ADC%', round(100*new['mean_ADC_normalized_RMSE'], 3), 'F', np.round(new['F_ratio'], 3), flush=True)
    report['inputs_sha256'] = inputs
    np.savez_compressed(OUT/'bootstrap.npz', **draws)
    np.savez_compressed(OUT/'means.npz', gate_time_ns=T, time_ns=TIME_NS, **means)
    save(OUT/'assessment.json', finite(report))


if __name__ == '__main__': main()
