"""Training-only joint response screening using Geant4 deposited-energy libraries.

This linear surrogate proposes full-optical Geant4 configurations. It is not a
replacement for optical transport, SiPM recovery, noise or hardware triggering.
"""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import ROOT
from reference import PROJECT, wave_features
from calibrate import convolve

HERE = Path(__file__).resolve().parent
PROBABILITIES = np.array([.1, .25, .5, .75, .9])
ANALYSIS_AMPLITUDE_ADC = 200.


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


class LinearResponse:
    def __init__(self, config, coefficients):
        self.config = config; self.coefficients = coefficients
        optical = config['optical']; readout = config['sipm']['readout']
        if optical['electron_tau_ns'] != optical['alpha_tau_ns']:
            raise ValueError('This surrogate requires the shared time constants of the baseline fit')
        edges = np.arange(0., 16002., 2.)
        rise, decay = readout['rise_ns'], readout['decay_ns']
        spe = np.diff(1-(decay*np.exp(-edges/decay)-rise*np.exp(-edges/rise))/(decay-rise))
        rc = -np.diff(np.exp(-edges/readout['lowpass_ns']))
        kernels = [convolve(convolve(-np.diff(np.exp(-edges/tau)), spe), rc)/2 for tau in optical['electron_tau_ns']]
        self.fractions = np.linspace(0., 1., 401); long = []; ratio = []; peak = []
        for fraction in self.fractions:
            y = fraction*kernels[0]+(1-fraction)*kernels[1]; maximum = int(y.argmax())
            before = int(np.flatnonzero(y[:maximum] <= .1*y[maximum])[-1]); anchor = before+1
            start = max(0, anchor-20); qlong = y[start:anchor+1500].sum()
            long.append(qlong); ratio.append(y[start:anchor+250].sum()/qlong); peak.append(y[maximum])
        self.long = np.array(long); self.ratio = np.array(ratio); self.peak = np.array(peak)
        # C -> mV ns -> ADC ns using the 2 V pilot convention. The shared net
        # ADC scale is converted to a 0.5 V scenario only when writing configs.
        self.area_factor = readout['transimpedance_ohm']*1e12*16384/2000.

    def predict(self, species_energy, alpha_ratio=.5, alpha_fraction=None):
        if alpha_fraction is None: alpha_fraction = self.config['optical']['alpha_fast_fraction']
        alpha, electron, other = species_energy.T
        qa = alpha*self.coefficients['alpha_C_per_MeV']*(alpha_ratio/.5)
        qe = electron*self.coefficients['electron_C_per_MeV']
        qo = other*self.coefficients['alpha_C_per_MeV']*(self.config['optical']['ion_light_yield_ratio']/.5)
        total = qa+qe+qo
        fraction = ((qa+qo)*alpha_fraction+qe*self.config['optical']['electron_fast_fraction'])/np.maximum(total, 1e-300)
        area = total*self.area_factor
        return (area*np.interp(fraction, self.fractions, self.long),
                np.interp(fraction, self.fractions, self.ratio),
                area*np.interp(fraction, self.fractions, self.peak))


def read_pilot(path):
    file = ROOT.TFile.Open(str(path)); info = json.loads(str(file.Get('RunInfo').GetString().Data())); tree = file.Get('Events')
    if not info.get('completed') or info['schema_version'] < 4: raise ValueError('Complete corrected pilot required')
    tree.SetBranchStatus('*', 0)
    for name in ('adc', 'edep_MeV', 'edep_alpha_MeV', 'edep_electron_MeV', 'avalanche_q_C', 'avalanche_emitter_pdg'):
        tree.SetBranchStatus(name, 1)
    energies = []; charges = []; waves = []
    for event in tree:
        energies.append([event.edep_alpha_MeV, event.edep_electron_MeV,
                         max(0., event.edep_MeV-event.edep_alpha_MeV-event.edep_electron_MeV)])
        q = np.array(event.avalanche_q_C); emitter = np.array(event.avalanche_emitter_pdg)
        charges.append([q[emitter == 1000020040].sum(), q[np.abs(emitter) == 11].sum()])
        waves.append(np.array(event.adc, dtype=float))
    file.Close()
    return info, np.array(energies), np.array(charges), wave_features(np.array(waves))


def training_targets():
    targets = {}; records = {}
    for source, run in [('Co60', 3), ('Am241', 12)]:
        feature = PROJECT/f'analysis/paper_baseline/results/pha_run{run:04d}_features.npz'
        phase = PROJECT/f'analysis/paper_baseline/results/phase2/run{run:04d}_features.npz'
        with np.load(feature) as z: f = {k: z[k] for k in z.files}
        with np.load(phase) as z: a = {k: z[k] for k in z.files}
        mask = a['eligible_primary'] & (a['split'] == 0) & (f['amplitude'] >= ANALYSIS_AMPLITUDE_ADC)
        targets[source] = {'q': np.quantile(f['charge_long'][mask], PROBABILITIES),
                           'r': np.quantile(f['short_long_ratio'][mask], PROBABILITIES)}
        records[source] = {'run': run, 'indices': np.flatnonzero(mask).tolist(), 'n': int(mask.sum()),
                           'charge_quantiles': targets[source]['q'].tolist(), 'ratio_quantiles': targets[source]['r'].tolist(),
                           'feature_sha256': sha(feature), 'phase_sha256': sha(phase)}
    return targets, records


def score(q, ratio, amplitude, scale, target, minimum=50):
    mask = amplitude*scale >= ANALYSIS_AMPLITUDE_ADC
    n = int(mask.sum())
    if n < minimum: return 1e6, n
    qquant = np.quantile(q[mask], PROBABILITIES)*scale
    rquant = np.quantile(ratio[mask], PROBABILITIES)
    # Fixed practical discrepancy scales, not sampling errors or likelihood.
    loss = np.mean((np.log(qquant/target['q'])/.15)**2)+np.mean(((rquant-target['r'])/.025)**2)
    return float(loss), n


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--transport', type=Path, required=True)
    parser.add_argument('--pilot', type=Path, default=HERE/'results/corrected_optics_20261027')
    parser.add_argument('--out', type=Path, required=True); args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    scan = json.loads((args.transport/'summary.json').read_text())
    if not scan.get('completed'): raise ValueError('Transport library must be complete')
    pilots = {s: read_pilot(args.pilot/name/'events.root') for s, name in [('Co60', 'co60'), ('Am241', 'am241')]}
    config = pilots['Co60'][0]['config']; optical = config['optical']; tau = optical['electron_tau_ns']
    duration = config['acquisition']['samples']*2-config['acquisition']['pretrigger_ns']
    def window(f): return f*(1-np.exp(-duration/tau[0]))+(1-f)*(1-np.exp(-duration/tau[1]))
    coefficients = {'electron_C_per_MeV': float(pilots['Co60'][2][:, 1].sum()/pilots['Co60'][1][:, 1].sum()/window(optical['electron_fast_fraction'])),
                    'alpha_C_per_MeV': float(pilots['Am241'][2][:, 0].sum()/pilots['Am241'][1][:, 0].sum()/window(optical['alpha_fast_fraction']))}
    response = LinearResponse(config, coefficients); targets, training = training_targets(); closure = {}
    for source, (_, energies, _, f) in pilots.items():
        q, ratio, amplitude = response.predict(energies)
        mask = (f['diagnostic_flags'] == 0) & (f['amplitude'] >= ANALYSIS_AMPLITUDE_ADC) & (q > 0)
        closure[source] = {'n': int(mask.sum()), 'relative_charge_error_p10_p50_p90': np.quantile(q[mask]/f['charge_long'][mask]-1, [.1, .5, .9]).tolist(),
                           'ratio_error_p10_p50_p90': np.quantile(ratio[mask]-f['short_long_ratio'][mask], [.1, .5, .9]).tolist()}
    deposits = {}
    for row in scan['scenarios']:
        path = args.transport/row['name']/'deposits.npz'
        if sha(path) != row['deposits_sha256']: raise ValueError('Transport input changed')
        with np.load(path) as z: energies = z['edep_origin_species_MeV'].sum(axis=1)
        deposits[row['name']] = energies[energies.sum(axis=1) > 0]
    scales = np.geomspace(.2, 3., 45)
    co_grid = []
    for scale in scales:
        alternatives = []
        for row in scan['scenarios']:
            if row['source'] != 'Co60': continue
            loss, n = score(*response.predict(deposits[row['name']]), scale, targets['Co60'])
            alternatives.append({'scenario': row['name'], 'loss': loss, 'n': n, 'scale': float(scale)})
        co_grid.append({'best': min(alternatives, key=lambda x: x['loss']), 'alternatives': alternatives})
    rows = []; best = None
    for scenario in scan['scenarios']:
        if scenario['source'] != 'Am241': continue
        for alpha_ratio in (.03, .05, .075, .1, .15, .2, .25, .3, .4, .5, .65, .8):
            for fraction in (0., .025, .05, .1, .15, .2, .3):
                prediction = response.predict(deposits[scenario['name']], alpha_ratio, fraction)
                for i, scale in enumerate(scales):
                    am_loss, am_n = score(*prediction, scale, targets['Am241'])
                    co = co_grid[i]['best']
                    row = {'am_scenario': scenario['name'], 'co_scenario': co['scenario'], 'net_adc_scale': float(scale),
                           'alpha_light_yield_ratio': alpha_ratio, 'alpha_fast_fraction': fraction,
                           'co_loss': co['loss'], 'am_loss': am_loss, 'joint_loss': co['loss']+am_loss, 'co_n': co['n'], 'am_n': am_n}
                    rows.append(row)
                    if best is None or row['joint_loss'] < best['joint_loss']: best = row
        print(json.dumps({'scanned': scenario['name'], 'best_so_far': best}), flush=True)
    with (args.out/'grid.csv').open('w') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    report = {'scope': __doc__, 'training_only': [3, 12], 'analysis_amplitude_adc': ANALYSIS_AMPLITUDE_ADC,
              'selection': 'Same 200 ADC analysis-amplitude cut applied to observed and simulated samples; not a hardware trigger or efficiency estimate. Measured selection also uses frozen phase2 quality.',
              'quantile_probabilities': PROBABILITIES.tolist(), 'objective': 'Sum over sources of mean squared log-Q quantile residual /0.15 and ratio quantile residual /0.025; diagnostic ranking, not chi square or likelihood.',
              'coefficients': coefficients, 'surrogate_closure': closure, 'training': training, 'co_grid': co_grid, 'best': best,
              'n_grid_points': len(rows), 'transport_summary_sha256': sha(args.transport/'summary.json'), 'script_sha256': sha(Path(__file__)),
              'limits': ['Charge conversion inferred from previous Geant4/SiPM pilot, not measured particle truth.',
                         'Linear surrogate neglects changed recovery/saturation, photon statistics and spatial collection variation; full optical confirmation is required.',
                         'Shared output normalization is represented by effective scintillation yield at fixed PDE/gain; it is not an intrinsic light-yield measurement.',
                         'Planar Al foil, source distance, array wiring, actual input range and full hardware trigger remain conditional assumptions.',
                         'No experimental particle labels, source-specific output gains or check-run tuning.']}
    references = json.loads((HERE/'results/reference/references.json').read_text())
    for source, run, key in [('Co60', 3, 'co_scenario'), ('Am241', 12, 'am_scenario')]:
        scenario_config = json.loads((args.transport/best[key]/'config.json').read_text())
        candidate = copy.deepcopy(config); candidate['geometry'] = scenario_config['geometry']
        candidate['optical']['yield_photons_per_MeV'] *= best['net_adc_scale']/4.
        candidate['optical']['alpha_light_yield_ratio'] = best['alpha_light_yield_ratio']
        candidate['optical']['alpha_fast_fraction'] = best['alpha_fast_fraction']
        candidate['sipm']['readout']['adc_max_mV'] = 500.
        ref = references['groups'][f'run{run}_train']
        candidate['sipm']['readout']['baseline_mV'] = ref['baseline_adc']*500/16384
        candidate['sipm']['readout']['noise_rms_mV'] = ref['baseline_rms_adc']*500/16384
        candidate['acquisition']['amplitude_threshold_adc'] = 120. if source == 'Co60' else 160.
        candidate['status'] = 'Training-only joint surrogate candidate; must be tested by full Geant4 optical transport'
        candidate['evidence']['joint_screening'] = {'record': str((args.out/'screen.json').resolve()),
            'range': '500 mV follows current X4 settings; association to historical BIN remains unconfirmed',
            'normalization': 'Effective common light-yield scale at fixed gain/PDE, not a measured intrinsic light yield',
            'source_geometry': 'Fitted candidate among declared scenarios; not recovered apparatus history',
            'training_only': [3, 12], 'no_hardware_trigger_claim': True}
        (args.out/f'{source}_candidate.json').write_text(json.dumps(candidate, indent=2)+'\n')
    (args.out/'screen.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'closure': closure, 'best': best, 'grid_points': len(rows)}, indent=2))


if __name__ == '__main__': main()
