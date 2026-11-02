"""Freeze independent seed batches, unchanged responses and all evaluation definitions."""
import shutil
import subprocess
from mc_statistics_common import *


def main():
    assert not OUT.exists()
    OUT.mkdir()
    for name in ['configs', 'native', 'replay', 'arrays', 'logs', 'figures', 'source_snapshot',
                 'transport/arrays', 'transport/extraction']:
        (OUT/name).mkdir(parents=True)
    p = read(WAVE/'protocol.json'); rp = read(RC/'protocol.json'); noise = read(RC/'noise_selection.json')
    jp = read(JOINT/'validation/protocol.json'); rr = read(RC/'record.json')
    assert sha(NATIVE) == jp['binary_sha256']['csi_sources']
    needed = [WAVE/n for n in ['record.json', 'protocol.json', 'decision.json', 'analysis.json']]
    needed += [CORE/n for n in ['record.json', 'protocol.json', 'diagnostics.json']]
    needed += [RC/n for n in ['record.json', 'protocol.json', 'noise_selection.json', 'mean_calibration.json', 'evaluation.json']]
    needed += [JOINT/'validation/protocol.json', PROJECT/'analysis/paper_baseline/results/publication_20261028/point.json']
    responses = {}; entries = {}
    for source, run in SOURCES:
        base_path = PROJECT/rp['native'][source]['config']
        assert sha(base_path) == rp['native'][source]['config_sha256']
        base = read(base_path); cp = OUT/'configs'/f'run{run}_native.json'
        shutil.copy2(base_path, cp); needed.append(base_path)
        for batch in BATCHES:
            key = batch_key(batch, run); seed = 731030000+1000*batch+run
            root = OUT/'native'/f'{key}.root'
            entries[key] = {'key': key, 'batch': batch, 'run': run, 'source': source, 'events': EVENTS[source],
                            'seed': seed, 'config': relative(cp), 'config_sha256': sha(cp), 'path': relative(root),
                            'command': [str(NATIVE), str(cp), source, str(EVENTS[source]), str(seed), str(root)]}
        for seg in SEGMENTS:
            group = f'run{run}_{seg}'
            needed += [WAVE/'arrays'/f'{group}_data.npz']
            for B in p['block_sizes']:
                needed += [WAVE/'designs'/f'{group}_B{B}.json', CORE/'references'/f'{group}_B{B}.json']
                for v in VARIANTS:
                    needed.append(RC/'arrays'/f'{group}_B{B}_{v}_evaluation.npz')
                    needed.append(CORE/'arrays'/f'{group}_B{B}_{v}_z.npz')
                needed.append(CORE/'arrays'/f'{group}_B{B}_data_z.npz')
                for ef in [0, 1]:
                    group_fold = f'{group}_B{B}_eval{ef}'
                    for family in ['inherited', 'spectrum']:
                        chosen = noise['groups'][group_fold]['families'][family]['chosen_native_key']
                        old = RC/'configs'/f'{chosen}.json'
                        assert sha(old) == rr['outputs_sha256'][relative(old)]
                        cfg = read(old)
                        for name in ['geometry', 'optical', 'acquisition']: assert cfg[name] == base[name]
                        assert cfg['sipm']['sensor'] == base['sipm']['sensor']
                        dest = OUT/'configs'/f'{group_fold}_{family}.json'
                        shutil.copy2(old, dest); needed.append(old)
                        responses[f'{group_fold}_{family}'] = {'config': relative(dest), 'config_sha256': sha(dest),
                            'parent_config': relative(old), 'parent_choice': chosen}
    seeds = [e['seed'] for e in entries.values()]
    assert len(set(seeds)) == 8
    existing_seeds = []
    for f in [JOINT/'profile/protocol.json', JOINT/'validation/protocol.json']:
        existing_seeds += [x['seed'] for x in read(f)['entries'].values()]
    assert not set(seeds)&set(existing_seeds)
    hashes = {relative(f): sha(f) for f in sorted(set(needed))}
    protocol = {'scope': __doc__, 'code_sha256': sha(__file__),
        'common_code_sha256': sha(PROJECT/'simulation/mc_statistics_common.py'),
        'checkpoint_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECT).decode().strip(),
        'inputs_sha256': hashes, 'binary_sha256': {relative(x): sha(x) for x in [NATIVE, REPLAY]},
        'entries': entries, 'responses': responses, 'q_edges': p['q_edges'], 'support_bins': p['support_bins'],
        'conditional_cuts': p['conditional_cuts'], 'block_sizes': p['block_sizes'], 'variants': VARIANTS,
        'metrics': METRICS, 'batches': BATCHES, 'events_per_batch': EVENTS,
        'planned_new_independent_mothers': 256000, 'old_independent_mothers': {'Co60': 8000, 'Am241': 24000},
        'transport': 'Four independent seeds per source, same full Geant4 binary and byte-identical baseline configs as original joint candidate. No geometry/optical/sensor/gain fit. All mothers including zero deposits retained in ROOT. Native and replay programs are frozen existing binaries.',
        'selection': 'Apply inherited mc_wave_features quality with per-batch global baseline-RMS median, amplitude>=200 ADC, t10>=384 samples and same inclusive original common-Q interval to baseline ADC only. Freeze these IDs and baseline Q for every response. No truth/main-band/diagnostic-tail deletion or selection on refitted ADC. Report selection, unsupported bins, cell counts and ESS; no adaptive extra events.',
        'responses_definition': 'Original frozen plus two prior chosen RC/noise families, unchanged per segment/block/evaluation fold. Replay original native avalanches with C++ ReadoutSampling. Require phase0 original-config ADC identity for every selected new mother. Replays do not increase independent count.',
        'fold_assignment': 'Inherited method: full-parent Bernoulli assignments with probability equal to frozen measured Q2-weighted evaluation-fold mass. RNG seed731030700+1000*batch+100*segment_index+run+block. Same one assignment for all models/diagnostics; independent of outcomes. Existing mothers retain original assignments.',
        'evaluation': 'Original experimental IDs/Q/PCA, Q-half targets, PID robust references and rank3 waveform modes remain frozen. Report first60/middle20/last20, block128 primary/block64 sensitivity, each seed batch and pooled fresh only; never pool old with fresh. All51 declared metrics retain original definitions. Joint full covariance, unscaled mean ADC, F means/SD and all quantile widths retained.',
        'uncertainty': '512 paired measured original-index block draws and stratified whole-parent bootstrap within each fresh seed batch. Selected counts sampled by exact multinomial marginal with one combined unselected category. Same counts across model variants. Independent original-cohort whole-parent draws for fresh-minus-old contrasts, sharing each measured draw. Report fresh joint, MC-only, data-only intervals/SD and paired response changes. Responses, assignment, modes, fences and Q targets fixed; no fit or model uncertainty/equivalence test.',
        'bootstrap_seed': '731031900+run+100*segment_index+block',
        'small_sample_check': 'Secondary last20/block128 only:512 draws without replacement of8000 Co/24000 Am distinct full-parent IDs from the fresh pool, retaining only selected IDs, same frozen assignments/targets. Compare diagnostic95% ranges and empirical rank of old points, not independent new histories, predictive validation or calibrated p-values. Reject unestimable draws; require128 valid for interval. RNG731032500+run.',
        'candidate_boundary': 'No selection/refit/promotion based on these fresh results. If same residual direction persists, propose a separate future constrained model. Experimental segments remain previously explored internal comparisons. New simulation is statistical replication, not new experimental evidence.',
        'new_experimental_events': 0, 'default_changed': False, 'official_PID_changed': False}
    save(OUT/'protocol.json', protocol)
    print('Frozen', len(entries), 'full-transport jobs,', len(responses), 'response configurations and51 metrics.', flush=True)


if __name__ == '__main__': main()
