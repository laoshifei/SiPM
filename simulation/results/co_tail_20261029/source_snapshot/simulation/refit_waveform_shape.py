"""Training-only optical-time/RC proposals, requiring subsequent full Geant4 runs.

This script uses the existing Jupyter scientific Python environment (SciPy).
The fast convolution is a proposal model, not a replacement for native transport.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scipy.optimize import least_squares
from scipy.signal import fftconvolve, savgol_filter

PROJECT = Path(__file__).resolve().parents[1]
OLD = PROJECT / 'simulation/results/signal_fluctuations_20261028'
OUT = PROJECT / 'simulation/results/waveform_refit_20261029'
T = np.arange(-200., 3802., 2.)
MASK = (T >= -100) & (T <= 3500)
sys.path.insert(0, str(PROJECT / 'analysis/paper_baseline'))
from prepare import wave_features


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def arrays(path):
    with np.load(path) as z:
        return dict(z)


def expected(fast, slow, rc, fraction):
    dt = 1.
    edges = np.arange(0., 16001., dt)
    emission = fraction*(-np.diff(np.exp(-edges/fast))) + (1-fraction)*(-np.diff(np.exp(-edges/slow)))
    rise, decay = 2., 29.
    spe = np.diff(1-(decay*np.exp(-edges/decay)-rise*np.exp(-edges/rise))/(decay-rise))
    lowpass = -np.diff(np.exp(-edges/rc))
    signal = fftconvolve(fftconvolve(emission, spe)[:len(emission)], lowpass)[:len(emission)] / dt
    peak = np.argmax(signal)
    low = np.flatnonzero(signal[:peak] <= .1*signal[peak])[-1]
    crossing = low + (.1*signal[peak]-signal[low])/(signal[low+1]-signal[low])
    y = np.interp(T, (np.arange(len(signal))-crossing)*dt, signal, left=0.)
    return normalize(y)


def normalize(y):
    return y / (2*y[(T >= -40) & (T < 3000)].sum())


def descriptors(y):
    # Smooth only the measured/MC mean for stable sub-bin peak timing.
    y = savgol_filter(y, 11, 3)
    imax = int(y.argmax())
    l, m, r = y[imax-1:imax+2]
    offset = .5*(l-r)/(l-2*m+r)
    peak_time = T[imax] + 2*offset
    peak = m - .25*(l-r)*offset
    ratio = y[(T >= -40) & (T < 500)].sum()/y[(T >= -40) & (T < 3000)].sum()
    rise_times = []
    for fraction in (.5, .9):
        before = np.flatnonzero(y[:imax] <= fraction*peak)[-1]
        rise_times.append(T[before] + 2*(fraction*peak-y[before])/(y[before+1]-y[before]))
    return np.array([peak_time, ratio, peak, *rise_times])


def parameters(x, separate):
    times = np.exp(x[:5] if separate else x[:3])
    if separate:
        f1, s1, f2, s2, rc = times
        return {'electron_tau_ns': [f1, s1], 'alpha_tau_ns': [f2, s2],
                'lowpass_ns': rc, 'electron_fast_fraction': x[5], 'alpha_fast_fraction': x[6]}
    fast, slow, rc = times
    return {'electron_tau_ns': [fast, slow], 'alpha_tau_ns': [fast, slow],
            'lowpass_ns': rc, 'electron_fast_fraction': x[3], 'alpha_fast_fraction': x[4]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    target = {}; original = {}; cfgs = {}; closure = {}; input_hashes = {}
    for source, run in [('Co60', 3), ('Am241', 12)]:
        data_path = OLD / f'run{run}_first60.npz'
        data = arrays(data_path)
        target[source] = data['shapes'].mean(0)
        cfg_path = OLD / 'independent' / f'{source}_calibrated.json'
        cfg = json.loads(cfg_path.read_text()); cfgs[source] = cfg
        fast, slow = cfg['optical']['electron_tau_ns' if source == 'Co60' else 'alpha_tau_ns']
        fraction = cfg['optical']['electron_fast_fraction' if source == 'Co60' else 'alpha_fast_fraction']
        original[source] = expected(fast, slow, cfg['sipm']['readout']['lowpass_ns'], fraction)
        native_path = OLD / 'independent' / f'{source}_native.npz'
        native = arrays(native_path)['shapes'].mean(0)
        # A fixed smooth closure residual is used only in proposal generation.
        # It is NOT installed as a waveform template or added to native output.
        closure[source] = savgol_filter(native-original[source], 51, 3)
        for path in (data_path, cfg_path, native_path):
            input_hashes[str(path.relative_to(PROJECT))] = sha(path)
    names = ['Co60', 'Am241']
    weights = np.ones(MASK.sum())
    weights[(T[MASK] >= 0) & (T[MASK] < 600)] = 4.
    weights /= weights.sum()
    cases = []
    for use_closure in (False, True):
        for separate in (False, True):
            def predictions(x):
                p = parameters(x, separate)
                curves = {}
                for source, kind in [('Co60', 'electron'), ('Am241', 'alpha')]:
                    y = expected(*p[kind+'_tau_ns'], p['lowpass_ns'], p[kind+'_fast_fraction'])
                    if use_closure: y = normalize(y + closure[source])
                    curves[source] = y
                return p, curves

            def residual(x):
                p, curves = predictions(x)
                rows = []
                for source in names:
                    y, data = curves[source], target[source]
                    rows.extend((y[MASK]-data[MASK])/data.max()/.01*np.sqrt(weights))
                    # Practical scales specify a descriptive fit objective,
                    # not independent errors or a chi-square likelihood.
                    rows.extend((descriptors(y)-descriptors(data))/np.array([10., .004, .00003, 8., 8.]))
                for kind in ('electron', 'alpha'):
                    a, b = p[kind+'_tau_ns']
                    rows.append(max(0., a-b)/10.)
                return np.array(rows)

            attempts = []
            for fast, slow, rc, fe, fa in [(479,1373,105,.623,.15),(450,2000,65,.65,.2),
                                            (350,1500,80,.6,.05),(650,3500,40,.85,.6)]:
                values = [fast, slow, fast, slow, rc] if separate else [fast, slow, rc]
                x = np.r_[np.log(values), fe, fa]
                low = [150,700,150,700,20] if separate else [150,700,20]
                high = [1000,8000,1000,8000,180] if separate else [1000,8000,180]
                fit = least_squares(residual, x, bounds=(np.r_[np.log(low),.005,.005], np.r_[np.log(high),.995,.995]),
                                    max_nfev=180, ftol=1e-10, xtol=1e-10, gtol=1e-8, diff_step=1e-4)
                p, curves = predictions(fit.x)
                row = {'parameters': p, 'objective': float(np.sum(residual(fit.x)**2)),
                       'success': bool(fit.success), 'nfev': fit.nfev, 'status': int(fit.status),
                       'metrics': {source: {
                           'mean_nrmse': float(np.sqrt(np.mean((curves[source][MASK]-target[source][MASK])**2))/target[source].max()),
                           'predicted_descriptors': descriptors(curves[source]).tolist(),
                           'target_descriptors': descriptors(target[source]).tolist(),
                       } for source in names}}
                attempts.append(row)
            best = min(attempts, key=lambda r:r['objective'])
            case = {'separate_particle_times': separate, 'closure_correction': use_closure,
                    'attempts': attempts, 'best': best}
            cases.append(case)
            print('closure',use_closure,'separate',separate,'loss',best['objective'],best['parameters'],flush=True)
    # Prefer the common-time hypothesis if both average-shape NRMSEs are below
    # 1%; otherwise retain the better separate-time proposal. All cases saved.
    corrected = [c for c in cases if c['closure_correction']]
    shared = next(c for c in corrected if not c['separate_particle_times'])
    selected = shared if max(v['mean_nrmse'] for v in shared['best']['metrics'].values()) < .01 else min(corrected,key=lambda c:c['best']['objective'])
    p = selected['best']['parameters']
    for source, kind in [('Co60','electron'),('Am241','alpha')]:
        cfg = copy.deepcopy(cfgs[source])
        for key in ('electron_tau_ns','alpha_tau_ns','electron_fast_fraction','alpha_fast_fraction'):
            cfg['optical'][key] = p[key]
        cfg['sipm']['readout']['lowpass_ns'] = p['lowpass_ns']
        cfg.pop('signal_dependent_noise',None)
        cfg['status'] = '2026-10-29 training mean-shape proposal; native optical validation and residual refit required'
        cfg['evidence']['shape_refit'] = {'training_runs':[3,12], 'training_segments':'original first60 only',
            'proposal_record':str((out/'shape_fit.json').relative_to(PROJECT)),
            'no_template_in_native_output':True, 'unknown_apparatus_not_inferred':True}
        save(out/f'{source}_shape_candidate.json',cfg)
    save(out/'shape_fit.json', {
        'scope':__doc__, 'code_sha256':sha(Path(__file__)), 'inputs_sha256':input_hashes,
        'training':'Exactly 2450 run3 + 1547 run12 frozen first60 measured waveforms; no other measured segments used.',
        'mean_shape_model':'Binned biexponential photon emission convolved with fixed 2/29 ns SPE and variable passive RC. Smooth closure residual from earlier native MC is only a proposal correction, never a native template.',
        'descriptor_names':['mean_peak_time_ns','mean_short_long_ratio','mean_peak_per_area_per_ns','mean_t50_ns','mean_t90_ns'],
        'selection_rule':'Prefer shared particle time constants if both corrected mean NRMSE <1%; otherwise minimum corrected objective.',
        'cases':cases, 'selected':selected,
        'limits':['Proposal fit only; requires full G4 with changed optical parameters.',
                  'Finite MC closure, broad parameter degeneracy and experimental changes not represented by a parameter confidence interval.',
                  'Am alpha-dominant interpretation remains conditional; full G4 must retain its gamma branch.'],
    })
    print('Selected',json.dumps(p),flush=True)


if __name__ == '__main__':
    main()
