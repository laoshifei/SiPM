"""Freeze the minimal spectral-state family before calibration outcomes."""
from spectral_split_common import *


def main():
    assert not OUT.exists(),'Refuse existing study'
    for folder in ['configs','fits','native','logs','arrays','evaluation','figures','source_snapshot']:(OUT/folder).mkdir(parents=True,exist_ok=True)
    p=read(FULL/'protocol.json');record=read(FULL/'record.json');parent=read(PARENT/'protocol.json')
    needed=[FULL/'record.json',FULL/'protocol.json',FULL/'analysis.json',REPLAY/'record.json',REPLAY/'replay_protocol.json',PARENT/'protocol.json']
    entries={}
    for key,e in p['entries'].items():
        entries[key]={**e,'reference':e['output'],'reference_sha256':record['outputs_sha256'][e['output']]}
        needed += [PROJECT/e['path'],PROJECT/e['output'],PROJECT/e['plan'],PARENT/'arrays'/f'{key}_frozen.npz']
        for v in ['original','current_r0','zero_cv_r0','current_r1','zero_cv_r1']:needed.append(FULL/'arrays'/f'{key}_{v}.npz')
        for ef in [0,1]:needed.append(PARENT/'arrays'/f'{key}_last20_B128_eval{ef}_spectrum.npz')
    for source,run in SOURCES:
        context=f'run{run}_last20_B128'
        needed += [REPLAY/'point_arrays'/f'{context}_{v}.npz' for v in ['data','current','zero_cv']]
        needed += [REPLAY/'repeat_arrays'/f'{source}_{v}.npz' for v in ['current_all','zero_cv_all']]
        needed += [SCALES/'references'/f'{context}.json',FULL/'evaluation'/f'{source}_counts.npz',FULL/'evaluation'/f'{source}_bootstrap.npz']
        for ef in [0,1]:needed.append(PROJECT/parent['responses'][f'{context}_eval{ef}_spectrum']['config'])
    for path in needed:
        if relative(path) in record['outputs_sha256']:assert sha(path)==record['outputs_sha256'][relative(path)],path
    code=['simulation/spectral_split_common.py','simulation/prepare_spectral_split.py','simulation/fit_spectral_split.py',
        'simulation/run_spectral_split.py','simulation/extract_spectral_split.py','simulation/replay_spectral_pool.cc',
        'simulation/include/SpectralVarianceSplit.hh','simulation/src/SpectralVarianceSplit.cc',
        'simulation/src/SignalDependentNoise.cc','simulation/tests/test_spectral_split.cc','simulation/CMakeLists.txt']
    inherited=['simulation/full_selection_common.py','simulation/charge_scale_common.py','simulation/state_response_common.py',
        'simulation/mc_statistics_common.py','simulation/extract_full_selection.py','simulation/fit_signal_fluctuations.py',
        'simulation/src/ReadoutReplica.cc','simulation/src/ReadoutNoise.cc','simulation/src/ReadoutSampling.cc',
        'simulation/src/ReadoutResponse.cc','simulation/vendor/SiPMResponse.cc',
        'analysis/paper_baseline/prepare.py','analysis/paper_baseline/refine_waveforms.py','simulation/prepare_acquisition_state.py',
        'simulation/analyze_acquisition_state.py']
    save(OUT/'protocol.json',{'scope':'Minimal complementary spectral-state candidate; opposite experimental folds for proposal and exhaustive existing-pool native checks, not blind validation.',
        'checkpoint_commit':'4a00b1c4c6fabd9e8289b51842d28f6bab9cff71','entries':entries,
        'q_edges':p['q_edges'],'support_bins':p['support_bins'],'conditional_cuts':p['conditional_cuts'],
        'code_sha256':{n:sha(PROJECT/n) for n in code},'inherited_code_sha256':{n:sha(PROJECT/n) for n in inherited},
        'inputs_sha256':{relative(path):sha(path) for path in sorted(set(needed))},
        'delays_samples':DELAYS,'proposal_repeats':16,'proposal_seed':731102900,'bootstrap_draws':512,'extension_seed':731102950,
        'model':'Legacy Gaussian innovations w split into independent Gaussian L,H with L+H=w and spectra P_L=cos²(omega*d/2), P_H=sin²(omega*d/2). Replace w by L+sqrt(M_H)*H before the unchanged FIR and signal envelope; M_H is independent, positive, mean-one lognormal. The original common variance multiplier M_C is retained with a fitted CV. No waveform-energy normalization, clipping correction, measured pulse injection, or transport change.',
        'spectral_boundary':'Complementary delay filters have frequency lobes; H is a delayed-difference component, not a disjoint physical high-frequency band. Ensemble covariance/mean PSD stays fixed; individual-event PSD varies. This is an effective acquisition-dependent model, not unique detector physics.',
        'proposal':'For each source/evaluation fold, use opposite measured last20/B128 blocks. Baseline is the mean within-Q covariance of the inherited16 zero-CV native readouts on the assigned original mother cohort. Estimate G_all and G_H=average sum w_i(1-w_i) E[Q_i Q_i^T] using16 independent Gaussian innovations and fixed zero-CV r0 anchors/noiseless post means. Predict Cov(F)=C0+a*G_all+b*G_H, a=CV_C², b=(1+a)*CV_H². Choose delay and bounded least squares on all six F covariance elements normalized by measured SD products. This ignores random denominators/anchors and selection; native full-pool checks are required.',
        'bounds':'0<=a<=inherited_common_CV²;0<=b<=4, so CV_H<=2. Delay grid1,4,16 samples fixed. Ties choose first delay. Keep every candidate prediction and first8/last8 sensitivity. No expansion/refit after native outcomes.',
        'floor':'G diagonals are nonnegative. With unchanged mean spectrum this family cannot lower a predicted F variance below the zero-CV baseline; report the floor even if the candidate fails.',
        'native':'All256000 original mothers; current_r0 and zero_cv_r0 must reproduce the prior full-pool ADC exactly. Candidate r0/r1 uses the same original/positive readout seeds. Preserve fold assignments, current full-batch QC, source Q support and half-bin targets.',
        'uncertainty':'Extend the previous complete-pool512 mother counts conditionally to newly entering categories; preserve data-block counts. Parameters, references, QC thresholds and realized noise fixed; no fit/systematic uncertainty. No independent mothers from repeated readouts.',
        'joint_guard':{'F_SD_bounds':[.9,1.1],'F_mean_bounds':[.9,1.1],'PID_SD_bounds':[.9,1.1],
            'PID_width50_95_bounds':[.85,1.15],'maximum_mean_ADC_L2_regression':.002,'maximum_PID_covariance_error_regression':.02,
            'F_covariance_error':'Must improve versus the corresponding current readout in both r0 and r1. Numerical guards only; no automatic default promotion.'},
        'new_independent_mothers':0,'new_experimental_events':0,'default_changed':False,'official_PID_changed':False})
    print('Frozen complementary spectral-state family, covariance objective, bounds and full-pool checks.',flush=True)


if __name__=='__main__':main()
