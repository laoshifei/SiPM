"""Freeze the observable-band audit before examining its derived comparisons."""
import subprocess
from noise_replay_common import *


def main():
    assert not OUT.exists()
    for name in ['selection/arrays','selection/evaluation','configs','replay','arrays','logs','figures','source_snapshot']:
        (OUT/name).mkdir(parents=True,exist_ok=True)
    p=read(QDC/'protocol.json');r=read(QDC/'record.json');inputs={}
    for k,v in r['outputs_sha256'].items():
        if '/arrays/' in k or '/evaluation/' in k or '/references/' in k or k.endswith('/analysis.json'):
            inputs[k]=v
    for f in [QDC/'record.json',QDC/'protocol.json',PARENT/'record.json',PARENT/'protocol.json']:
        inputs[relative(f)]=sha(f)
    for source,run in SOURCES:
        for seg in SEGMENTS:
            for f in [DATA/f'run{run}_{seg}.npz',WIDTH/'arrays'/f'measured_run{run}_{seg}.npz']:
                inputs[relative(f)]=sha(f)
    for k,v in inputs.items():assert sha(PROJECT/k)==v,k
    protocol={k:p[k] for k in ['q_edges','support_bins','conditional_cuts','block_sizes','variants']}
    code=['simulation/noise_replay_common.py','simulation/prepare_noise_selection.py','simulation/audit_noise_selection.py']
    protocol.update(scope=__doc__,checkpoint_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT).decode().strip(),
        inputs_sha256=inputs,code_sha256={k:sha(PROJECT/k) for k in code},features=FEATURES,metrics=METRICS,cases=CASES,
        selection='Existing quality/common-Q primary experiment before old main-band cut versus its exact old band IDs. Co QDC>=0.4; Am QDC<0.4. MC retains original quality/common-Q IDs and Q; compare all, own-response band and original-frozen-response band. Source names are not particle truth. No truth selection, no removal based on F or tails.',
        cases_definition={'all_selected':'Pre-band measured versus all original MC. Response remains trained on the prior main-band data; this is a population sensitivity, not a new fit.',
            'inherited':'Prior main-band measured versus all MC; must reproduce prior F means/SDs and QDC widths.',
            'matched_band':'Prior measured band versus same observable cut on each MC response ADC.',
            'fixed_MC_band':'Prior measured band versus MC mask from frozen baseline ADC, shared across responses.'},
        weighting='Original within-Q variances, equal Q-bin weights and original half-Q target masses. Reweight retained members within each half. Empty positive-mass cells are unestimable; retain all original support bins. Full-cohort survival masses also reported before renormalization.',
        uncertainty='Same512 parent original-index measured block draws and full-mother MC multinomials; consume old-parent counts. Data pre-band and band use identical block draws; MC cuts use identical mother counts. Intervals conditional on frozen references/response/targets; no refit/systematic/multiplicity correction.',
        invalid_F='Nonpositive pulse-window means make original F undefined; count such events and propagate nonfinite F summaries without silently removing events. Other finite PID quantities remain reportable.',
        new_independent_mothers=0,new_native_replays=0,default_changed=False,official_PID_changed=False)
    save(OUT/'selection_protocol.json',protocol);print('Selection audit frozen:',len(inputs),'input hashes.',flush=True)


if __name__=='__main__':main()
