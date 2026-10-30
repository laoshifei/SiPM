"""Freeze bounded timing families before joint native waveform/PID calibration.

The analytic convolution proposes parameters only. Every evaluated ADC comes
from full Geant4 transport or its native C++ noise-only response replay.
"""
import copy
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares, brentq
from scipy.signal import savgol_filter
from fit_co_tail import (OUT as TAIL, PARENT as NPE, DATA, WIDTH, PROJECT, BUILD,
                        T, SOURCES, expected, normalize, gate_density, descriptors, tail_metrics)
from refit_parallel_shape import convolved_exponentials
from fit_pid_bandwidth import average_bins
from prepare_pid import read
from extract_pid import load
from publication_stats import save, sha

OUT = PROJECT/'simulation/results/joint_constraints_20261029'
FAMILIES = {'a4000': 4000., 'a6000': 6000., 'a8439': 8438.62820399502}


def relative(p):
    return str(p.relative_to(PROJECT))


def components(times, rc, prompt):
    return [(1-prompt)*convolved_exponentials([tau, 2., 29., rc]) +
            prompt*convolved_exponentials([tau, 2., 5.]) for tau in times]


def density(curves, fraction):
    # Align the combined signal, not separately aligned exponential components.
    signal = fraction*curves[0]+(1-fraction)*curves[1]
    peak = int(signal.argmax())
    i = int(np.flatnonzero(signal[:peak] <= .1*signal[peak])[-1])
    crossing = i+(.1*signal[peak]-signal[i])/(signal[i+1]-signal[i])
    raw = np.interp(T+crossing+1., np.arange(len(signal)), signal, left=0.)
    return normalize(raw)


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    for folder in ['fit', 'noise_fit', 'profile', 'calibration', 'validation', 'figures']:
        (OUT/folder).mkdir()
    old = read(TAIL/'native_protocol.json'); edges = old['q_edges']
    cfgs = {}; targets = {}; closures = {}; inputs = {}; curves_saved = {'time_ns': T}
    for source, run, kind in SOURCES:
        key = f'run{run}_{source}'; cp = PROJECT/old['entries'][key]['config']
        cfgs[source] = read(cp)
        dp = DATA/f'run{run}_first60.npz'; sp = WIDTH/'arrays'/f'measured_run{run}_first60.npz'
        mp = TAIL/'arrays'/f'{key}.npz'
        d = load(dp); s = load(sp); m = load(mp); u = s['main_band']
        np.testing.assert_array_equal(d['ids'], s['ids'])
        support = old['support_bins'][source]
        targets[source] = average_bins(gate_density(d['waves'][u])[0], s['q'][u], edges, support)
        native = average_bins(gate_density(m['waves'])[0], m['q'], edges, support)
        c = cfgs[source]
        analytical, _ = expected(c['optical'][kind+'_tau_ns'], c['sipm']['readout']['lowpass_ns'],
                                 c['optical'][kind+'_fast_fraction'], c['readout_response']['prompt_charge_fraction'])
        closures[source] = savgol_filter(native-analytical, 51, 3)
        curves_saved.update({source+'_data': targets[source], source+'_closure': closures[source], source+'_tail': native})
        for p in [dp, sp, mp, cp]: inputs[relative(p)] = sha(p)
    base = cfgs['Co60']; o = base['optical']; rc = base['sipm']['readout']['lowpass_ns']
    prompt = base['readout_response']['prompt_charge_fraction']
    starts = [np.log([*o['electron_tau_ns'], o['alpha_tau_ns'][0], rc]),
              np.log([100., 850., 750., 130.]), np.log([250., 1000., 950., 90.])]
    scales = {'median_in_data_IQR': .20, 'mean_time_ns': 3., 'IQR_ratio': .15,
              'covariance_normalized': .25, 'F_ratio': .10, 'mean_ADC_NRMSE': .01}
    design = {'scope': __doc__, 'checkpoint_commit': 'f7a840c', 'code_sha256': sha(Path(__file__)),
        'inputs_sha256': inputs, 'parent_record_sha256': {relative(p): sha(p/'record.json') for p in [TAIL, NPE]},
        'families_alpha_slow_ns': FAMILIES, 'starts_log_parameters': [x.tolist() for x in starts],
        'q_edges': edges, 'support_bins': old['support_bins'], 'ranking_scales': scales,
        'ranking': 'Equal source weights. Sum of six group mean-squared residuals: three median shifts/data IQR, mean time, three IQR ratios, six covariance entries normalized by data diagonal products, three F ratios, raw mean ADC NRMSE. Descriptive correlated penalties, not likelihood. Choose smallest total on native calibration ADC first60 only.',
        'training': 'Frozen run3/run12 first60 only; tail-study MC reused for fixed proposal closure, then independent calibration mothers for each family. No later experimental segment or fresh validation output enters selection.',
        'timing': 'Fixed alpha slow times are conditional sensitivity families, not confidence limits. Fractions solve the first60 mean-time target when feasible. Four other timing parameters fit uniform density, five descriptors and late-tail moments, three starts.',
        'closure': 'Fixed smoothed tail-native minus analytic difference is used only to propose parameters; never inserted into native waveform.',
        'noise': 'Native calibration transport disables signal residual. Refit nonnegative positive-PSD mixture against full PID covariance and F50/102/202; inherited event-variance CV fixed. Then native noise-only replay, own-anchor ADC selection and ranking.',
        'normalization': 'Common optical-yield and alpha-relative-yield scalars preserve each analytical 3us gate area. No per-event scaling, PID-score broadening or imported measured template.',
        'native': {'calibration_mothers': 96000, 'validation_mothers': 32000,
                   'calibration_Co_seeds': [401029003+i*1000 for i in range(3)],
                   'calibration_Am_seeds': [401029012+i*1000 for i in range(3)],
                   'validation_seeds': {'Co60': 401029903, 'Am241': 401029912}},
        'frozen': ['geometry', 'PDE0.3', 'passive transmission0.5', 'SPE2/29ns', 'first60 baseline noise', 'event variance CV', 'experimental IDs and PCA'],
        'limits': 'Main-band data labels are diagnostic, not particle truth. Am includes alpha/gamma/electrons. Unknown circuit and geometry remain conditional. Later data previously explored, not new experimental blind validation.'}
    save(OUT/'design.json', design)
    fits = {}; entries = {}
    low = np.log([80., 400., 400., 60.]); high = np.log([500., 3000., 1800., 200.])
    for fi, (family, slow) in enumerate(FAMILIES.items()):
        def predict(x):
            e1, e2, a1, rcnew = np.exp(x)
            p = {'electron_tau_ns': [e1, e2], 'alpha_tau_ns': [a1, slow], 'lowpass_ns': rcnew}
            ys = {}
            for source, run, kind in SOURCES:
                cc = components(p[kind+'_tau_ns'], rcnew, prompt)
                target_mean = tail_metrics(targets[source])[0]
                def response(f): return normalize(density(cc, f)+closures[source])
                def equation(f): return tail_metrics(response(f))[0]-target_mean
                if equation(.005)*equation(.995) <= 0:
                    fraction = brentq(equation, .005, .995, xtol=1e-12)
                else:
                    fraction = min([.005, .995], key=lambda f: abs(equation(f)))
                p[kind+'_fast_fraction'] = fraction; ys[source] = response(fraction)
            return p, ys
        def residual(x):
            p, ys = predict(x); rows = []
            for source, y in ys.items():
                target = targets[source]
                rows.extend((y-target)/target.max()/.01/np.sqrt(len(T)))
                rows.extend((descriptors(y)-descriptors(target))/np.array([10., .004, .00003, 8., 8.]))
                rows.extend((tail_metrics(y)-tail_metrics(target))/np.array([.2, .002, 5.]))
            rows.append(max(0., p['electron_tau_ns'][0]-p['electron_tau_ns'][1])/10.)
            return np.asarray(rows)
        attempts = []
        for i, start in enumerate(starts):
            fit = least_squares(residual, np.clip(start, low+1e-6, high-1e-6), bounds=(low, high),
                max_nfev=220, ftol=1e-9, xtol=1e-9, gtol=1e-7, diff_step=1e-4)
            p, ys = predict(fit.x)
            row = {'start': i, 'parameters': p, 'objective': float(np.sum(residual(fit.x)**2)),
                   'success': bool(fit.success), 'nfev': int(fit.nfev),
                   'mean_time_difference_ns': {s: float(tail_metrics(y)[0]-tail_metrics(targets[s])[0]) for s, y in ys.items()}}
            attempts.append(row); print(family, i, row, flush=True)
        best = min([r for r in attempts if r['success']], key=lambda r: r['objective'])
        assert max(abs(x) for x in best['mean_time_difference_ns'].values()) < 1e-6
        fits[family] = {'attempts': attempts, 'best': best, 'bounds_log': [low.tolist(), high.tolist()]}
        p = best['parameters']; areas = {}
        for kind in ['electron', 'alpha']:
            _, oldarea = expected(o[kind+'_tau_ns'], rc, o[kind+'_fast_fraction'], prompt)
            _, newarea = expected(p[kind+'_tau_ns'], p['lowpass_ns'], p[kind+'_fast_fraction'], prompt)
            areas[kind] = oldarea/newarea
        # The shorter alpha tail intentionally reduces the extrapolated total yield.
        assert all(.4 < x < 1.5 for x in areas.values())
        fits[family]['analytic_gate_area_multipliers'] = areas
        save(OUT/'fit'/f'{family}.json', fits[family])
        for source, run, kind in SOURCES:
            c = copy.deepcopy(cfgs[source])
            for field in ['electron_tau_ns', 'alpha_tau_ns', 'electron_fast_fraction', 'alpha_fast_fraction']:
                c['optical'][field] = p[field]
            c['sipm']['readout']['lowpass_ns'] = p['lowpass_ns']
            c['optical']['yield_photons_per_MeV'] *= areas['electron']
            c['optical']['alpha_light_yield_ratio'] *= areas['alpha']/areas['electron']
            c['signal_dependent_noise']['variance_per_signal_adc'] = 0.
            c['status'] = 'Bounded timing family, native calibration base without signal residual; not a default configuration'
            c['joint_constraints'] = {'family': family, 'design': relative(OUT/'design.json'), 'stage': 'calibration_base',
                'analytic_gate_area_multipliers': areas, 'template_added': False}
            key = f'{family}_run{run}_{source}'; folder = OUT/'profile'/key; folder.mkdir()
            cp = folder/'config.json'; rp = folder/'events.root'; save(cp, c)
            events = 8000 if source == 'Co60' else 24000
            seed = (401029003 if source == 'Co60' else 401029012)+fi*1000
            entries[key] = {'key': key, 'run': run, 'source': source, 'family': family, 'events': events,
                'seed': seed, 'native': True, 'config': relative(cp), 'config_sha256': sha(cp), 'path': relative(rp),
                'command': [str(BUILD/'csi_sources'), str(cp), source, str(events), str(seed), str(rp)]}
            y, _ = expected(p[kind+'_tau_ns'], p['lowpass_ns'], p[kind+'_fast_fraction'], prompt)
            curves_saved[f'{family}__{source}'] = normalize(y+closures[source])
    save(OUT/'timing_proposals.json', {'design_sha256': sha(OUT/'design.json'), 'families': fits})
    np.savez_compressed(OUT/'proposal_curves.npz', **curves_saved)
    save(OUT/'profile/protocol.json', {'scope': __doc__, 'design_sha256': sha(OUT/'design.json'),
        'q_edges': edges, 'support_bins': old['support_bins'], 'entries': entries,
        'binary_sha256': {name: sha(BUILD/name) for name in ['csi_sources', 'replay_response']},
        'independent_mothers': 96000, 'stage': 'profile'})
    print('Frozen three families and 96000 calibration mothers.', flush=True)


if __name__ == '__main__': main()
