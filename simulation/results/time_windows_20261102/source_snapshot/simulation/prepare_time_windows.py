"""Freeze exact post-window ANOVA and inherited paired counts before extraction."""
import subprocess
from time_window_common import *
from test_time_windows import run as test_algebra


def main():
    assert not OUT.exists()
    tests=test_algebra()
    for directory in ['arrays','evaluation','figures','source_snapshot']:(OUT/directory).mkdir(parents=True,exist_ok=True)
    p=read(SPLIT/'protocol.json');record=read(SPLIT/'record.json')
    inputs=[SPLIT/x for x in ['record.json','protocol.json','analysis.json','verification.json']]+[LOW/'record.json',LOW/'decision.json']
    for source,run in SOURCES:
        inputs += [REPLAY/'point_arrays'/f'run{run}_last20_B128_data.npz',QDC/'references'/f'run{run}_last20_B128.json',
            PROJECT/f'data/DT5730_DPP_PHA/DAQ/run_{run}/UNFILTERED/Data_run_{run}.BIN',SPLIT/'evaluation'/f'{source}_counts.npz',SPLIT/'evaluation'/f'{source}_bootstrap.npz']
        inputs += [SPLIT/'evaluation'/f'{source}_{v}.npz' for v in VARIANTS]
        inputs += [SPLIT/'native'/f'b{b:02d}_run{run}.root' for b in range(4)]
    for path in inputs:
        if relative(path) in record['outputs_sha256']:assert sha(path)==record['outputs_sha256'][relative(path)],path
    code=['simulation/time_window_common.py','simulation/prepare_time_windows.py','simulation/extract_time_windows.py','simulation/analyze_time_windows.py','simulation/test_time_windows.py']
    inherited=['simulation/state_response_common.py','simulation/charge_scale_common.py','simulation/full_selection_common.py',
        'simulation/reference.py','analysis/paper_baseline/prepare.py']
    save(OUT/'protocol.json',{'checkpoint_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        **{k:p[k] for k in ['q_edges','support_bins','conditional_cuts']},'scales_ns':SCALES,'terms':TERMS,'metrics':NAMES,
        'inputs_sha256':{relative(x):sha(x) for x in sorted(set(inputs))},'code_sha256':{x:sha(PROJECT/x) for x in code},
        'inherited_code_sha256':{x:sha(PROJECT/x) for x in inherited},
        'scope':'Time-resolved diagnosis of the previous split full-pool ADC and existing late measured segment; no parameter fitting, waveform modification or new readout.',
        'windows_ns':[[500,1000],[1000,1500],[1500,2000],[2000,2500]],
        'exact_ANOVA':'Use unchanged high-pass widths25/51/101 samples, pre200 and post1000 samples at ceil(t10-300)/ceil(t10+250). Split post into four contiguous250-sample windows. Vpost=sum(249*Vwindow)/999+250*sum((mean_window-mean_post)^2)/999, using ddof1 throughout.',
        'terms':'Divide four within-window variance terms, between-window-mean variance, and negative pretrigger variance by inherited opposite-fold per-Q mu0. Add normalization term (Vpost-Vpre)*(1/mu-1/mu0); seven signed terms sum exactly to original F.',
        'variance_attribution':'Within each current-Q bin, preserve frozen half-bin masses. Cov(term,F) row sums add to Var(F). Compare MC-data contributions divided by measured F variance. Contributions may be negative; they are not independent physical noise powers.',
        'cross_window':'Separately split fixed-denominator quarter terms into sum of four diagonal variances, twice all off-diagonal quarter covariances, and all remaining covariance terms. These three groups exactly sum to total variance; no causal attribution.',
        'local_diagnostic':'Local F=(quarter residual variance-pre variance)/quarter pulse mean. Retain all events. Nonpositive local means produce NaN and are reported; do not silently drop or clip them. Local diagnostics are not official PID.',
        'selection':'Reuse all previously supported mothers under complete-pool split r0/r1 current Q/QC. Experimental IDs, source Q support, half targets and folds unchanged. No truth selection. All three smoothing scales and both sources retained.',
        'uncertainty':'Exactly inherit512 experimental-block and complete-mother multiplicities from spectral split. No new RNG, resampling of windows or refitting. Point and every global F SD draw must reproduce the previous complete-pool result. Fold results are point sensitivity checks, not independent validation.',
        'limits':'Variance terms also contain deterministic pulse curvature, source mixtures, residual correlation, normalization and finite reference noise. No unique SiPM/electronics mechanism or experimental particle truth inferred. Existing data are explored; no blind or equivalence claim.',
        'new_independent_mothers':0,'new_native_readouts':0,'new_experimental_events':0,'default_changed':False,'official_PID_changed':False})
    save(OUT/'algebra_tests.json',tests)
    print('Frozen four contiguous windows, exact variance attribution and inherited512 draws.',flush=True)


if __name__=='__main__':main()
