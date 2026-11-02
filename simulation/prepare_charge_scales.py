"""Freeze current-charge sensitivity and joint F-scale diagnostics before evaluation."""
import subprocess
from charge_scale_common import *
from test_charge_scales import run as tests


def main():
    assert not OUT.exists();test_result=tests()
    rp=read(REPLAY/'replay_protocol.json');archive=read(REPLAY/'record.json');inputs={}
    wanted=[REPLAY/n for n in ['record.json','verification.json','replay_protocol.json','point_analysis.json','repeat_analysis.json']]
    wanted += list((REPLAY/'point_arrays').glob('*.npz'))+list((REPLAY/'point_evaluation').glob('*_bootstrap.npz'))
    for source,run in SOURCES:
        for case in ['current_all','zero_cv_all']:
            wanted.append(REPLAY/'repeat_arrays'/f'{source}_{case}.npz')
            for part in rp['groups'][source]['logical_cases'][case]:
                wanted += [REPLAY/'arrays'/f'{key}.npz' for key in part['keys']]
    for path in sorted(set(wanted)):
        h=sha(path);name=relative(path)
        if path!=REPLAY/'record.json':assert h==archive['outputs_sha256'][name],name
        inputs[name]=h
    inherited={n:sha(PROJECT/n) for n in ['simulation/noise_replay_common.py','simulation/analyze_acquisition_state.py',
        'analysis/paper_baseline/publication_stats.py','simulation/qdc_fluctuation_common.py','simulation/mc_statistics_common.py']}
    codes=['simulation/charge_scale_common.py','simulation/prepare_charge_scales.py','simulation/analyze_charge_scales.py','simulation/test_charge_scales.py']
    p={k:rp[k] for k in ['q_edges','support_bins','conditional_cuts','block_sizes']}
    p.update(scope=__doc__,checkpoint_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
        inputs_sha256=inputs,code_sha256={n:sha(PROJECT/n) for n in codes},inherited_sources_sha256=inherited,
        contexts={k:{n:v[n] for n in ['source','run','segment','block']} for k,v in rp['point_contexts'].items()},
        conditions=CONDITIONS,variants=VARIANTS,features=FEATURES,metrics=METRICS,rotation=ROTATION,
        selection='Same original selected IDs only. Explicit under/overflow, final common-Q edge inclusive. original_Q inherits old support/half targets. retained_original_Q retains only IDs whose current Q is in source support, then weights by old Q. current_Q uses current Q bins and old physical half-Q thresholds/target masses. No truth or new main-band cut. Missing cells propagate unestimability; never drop support bins. Events outside the originally selected pool cannot enter, so this is not full trigger/current-selection validation.',
        charge='Reconstruct gate charge as sum(20*mean_wave bins), then restore its exact 1/8 ADC lattice (1520 integer samples and mean baseline of128 integer samples); require rounding residue<1e-7. This avoids numerical tie changes at physical half-bin thresholds. Positive replicas must match every saved native qcalc; data must match original q exactly. No ADC modification.',
        scales='For each original evaluation fold/Q bin, normalize F50/F102/F202 by opposite-fold measured means, then subtract one. Fixed orthonormal axes: common=(z50+z102+z202)/sqrt3; short_long=(z50-z202)/sqrt2; curvature=(z50-2*z102+z202)/sqrt6. Current-Q conditions select the reference for the current bin. These are feature combinations, not independent frequency bands or identified physical components.',
        repeats='Primary last20/block128, current_all and zero_cv_all, all16 positive readouts. Compute/reweight every readout separately then average within-Q covariances, means, squared quantile widths and profiles; do not flatten repeats or add between-repeat means. Every repeat/support bin required. This also remains valid when Q-dependent weights vary with noise. Do not use the prior fixed-weight within/between formula after rebinning.',
        uncertainty='Use all512 inherited paired measured block/full-mother counts. Same mother count across all readouts, conditions and variants. Opposite-fold references and half targets stay fixed. Report conditional percentile ranges and valid counts, no coverage/equivalence claim; no refit, complete noise-resampling or systematics uncertainty. Already explored data and MC, not blind validation.',
        interpolation='For each F variance, lambda=(Vdata-Vzero)/(Vcurrent-Vzero). Keep out-of-range and negative fractions; degenerate denominator unestimable. Also report paired lambda50-lambda202, lambda50-lambda102, lambda102-lambda202. Pure moment-line sensitivity; no assertion that intermediate native CV follows this line, no mapping to physical CV, no tuning.',
        response_policy='No fitting, parameter promotion or new transport in this phase. Determine charge-selection sensitivity and cross-scale constraints before implementing more flexible residual-noise models.',
        new_independent_mothers=0,new_readout_replays=0,new_experimental_events=0,default_changed=False,official_PID_changed=False)
    OUT.mkdir()
    for n in ['references','point_evaluation','repeat_evaluation','figures','source_snapshot']:(OUT/n).mkdir()
    save(OUT/'algebra_tests.json',test_result)
    save(OUT/'protocol.json',serialize(p));print('Frozen charge/scale design with',len(inputs),'verified input hashes.',flush=True)


if __name__=='__main__':main()
