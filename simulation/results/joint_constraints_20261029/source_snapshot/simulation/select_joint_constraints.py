"""Rank actual native calibration ADC using the frozen first60 joint criterion."""
import copy
from pathlib import Path
import numpy as np
from prepare_joint_constraints import OUT, PROJECT, BUILD, DATA, WIDTH, FAMILIES, SOURCES, relative
from prepare_pid import read
from extract_pid import load
from fit_pid_bandwidth import average_bins, conditional_cov
from analyze_bandwidth_validation import per_bin, widths, rms_width
from analyze_npe_validation import raw_shapes
from reference import TIME_NS
from publication_stats import sha, save


def evaluate(m, d, waves, edges, support, scales):
    dq = per_bin(d['q'], d['scores'], edges); mq = per_bin(m['q'], m['scores'], edges)
    iqr_ratio = rms_width(widths(mq), support)/rms_width(widths(dq), support)
    median_shift = np.mean(((mq[:, 1, :]-dq[:, 1, :])/widths(dq))[support], axis=0)
    mu = -1000*(average_bins(m['scores'][:, 1], m['q'], edges, support)-average_bins(d['scores'][:, 1], d['q'], edges, support))
    dc = conditional_cov(d['scores'], d['q'], edges, support, robust=True)
    mc = conditional_cov(m['scores'], m['q'], edges, support)
    i, j = np.triu_indices(3); normal = np.sqrt(np.diag(dc)[i]*np.diag(dc)[j])
    cov_delta = (mc-dc)[i, j]/normal
    df = average_bins(d['F'], d['q'], edges, support); mf = average_bins(m['metrics'], m['q'], edges, support)
    da = average_bins(raw_shapes(waves), d['q'], edges, support)
    ma = average_bins(raw_shapes(m['waves']), m['q'], edges, support)
    region = (TIME_NS >= -40)&(TIME_NS <= 3000)
    nrmse = float(np.sqrt(np.mean((ma[region]-da[region])**2))/da.max())
    residuals = {'median_in_data_IQR': median_shift, 'mean_time_ns': np.array([mu]),
        'IQR_ratio': iqr_ratio-1, 'covariance_normalized': cov_delta, 'F_ratio': mf/df-1,
        'mean_ADC_NRMSE': np.array([nrmse])}
    terms = {k: float(np.mean((v/scales[k])**2)) for k, v in residuals.items()}
    return {'n_MC': len(m['q']), 'IQR_ratio': iqr_ratio.tolist(), 'median_shift_in_data_IQR': median_shift.tolist(),
        'mean_time_difference_ns': float(mu), 'F_ratio': (mf/df).tolist(), 'mean_ADC_NRMSE': nrmse,
        'covariance_difference_normalized': cov_delta.tolist(), 'objective_terms': terms, 'objective': sum(terms.values())}


def main():
    assert not (OUT/'selection.json').exists()
    design = read(OUT/'design.json'); protocol = read(OUT/'calibration/protocol.json')
    execution = read(OUT/'calibration/execution.json'); assert len(execution['jobs']) == 6
    edges = np.asarray(protocol['q_edges']); results = {}; inputs = {}
    for family in FAMILIES:
        rows = {}
        for source, run, kind in SOURCES:
            key = f'{family}_run{run}_{source}'; mp = OUT/'calibration/arrays'/f'{key}.npz'
            dp = DATA/f'run{run}_first60.npz'; sp = WIDTH/'arrays'/f'measured_run{run}_first60.npz'
            m = load(mp); data = load(dp); ds = load(sp); u = ds['main_band']
            np.testing.assert_array_equal(data['ids'], ds['ids'])
            d = {'q': ds['q'][u], 'scores': ds['scores'][u], 'F': data['metrics'][u, :, 3]}
            rows[source] = evaluate(m, d, data['waves'][u], edges, protocol['support_bins'][source], design['ranking_scales'])
            for path in [mp, dp, sp]: inputs[relative(path)] = sha(path)
        results[family] = {'sources': rows, 'objective': float(np.mean([x['objective'] for x in rows.values()]))}
    selected = min(results, key=lambda k: results[k]['objective'])
    save(OUT/'selection.json', {'scope': __doc__, 'code_sha256': sha(Path(__file__)),
        'design_sha256': sha(OUT/'design.json'), 'calibration_protocol_sha256': sha(OUT/'calibration/protocol.json'),
        'inputs_sha256': inputs, 'ranking': results, 'selected_family': selected,
        'note': 'Minimum predeclared equal-source first60 criterion on actual native replay ADC. These are training results; fresh native transport below has not yet been run.'})
    entries = {}
    for source, run, kind in SOURCES:
        parentkey = f'{selected}_run{run}_{source}'; e = protocol['entries'][parentkey]
        c = copy.deepcopy(read(PROJECT/e['config'])); c['joint_constraints']['stage'] = 'frozen_independent_validation'
        c['status'] = 'Joint bounded-family first60 calibration frozen for independent native transport; conditional effective model'
        key = f'run{run}_{source}'; folder = OUT/'validation'/key; folder.mkdir()
        cp = folder/'config.json'; rp = folder/'events.root'; save(cp, c)
        seed = design['native']['validation_seeds'][source]
        entries[key] = {'key': key, 'run': run, 'source': source, 'family': selected, 'events': e['events'], 'seed': seed,
            'native': True, 'config': relative(cp), 'config_sha256': sha(cp), 'path': relative(rp),
            'command': [str(BUILD/'csi_sources'), str(cp), source, str(e['events']), str(seed), str(rp)]}
    save(OUT/'validation/protocol.json', {'scope': __doc__, 'selection_sha256': sha(OUT/'selection.json'),
        'design_sha256': sha(OUT/'design.json'), 'q_edges': protocol['q_edges'], 'support_bins': protocol['support_bins'],
        'binary_sha256': protocol['binary_sha256'], 'entries': entries, 'independent_mothers': 32000, 'stage': 'validation'})
    for name, row in results.items(): print(name, row, flush=True)
    print('Frozen selection:', selected, flush=True)


if __name__ == '__main__': main()
