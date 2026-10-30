"""Freeze acquisition-segment calibration, front-only controls and native phase tests."""
import subprocess
from pathlib import Path
import numpy as np
from prepare_acquisition_state import PROJECT, DATA, WIDTH, PARENT, SOURCES, SEGMENTS, read, load, relative
from publication_stats import save, sha

OUT=PROJECT/'simulation/results/state_response_20261030'
ACQ=PROJECT/'simulation/results/acquisition_state_20261030'
JOINT=PROJECT/'simulation/results/joint_constraints_20261029'
BUILD=PROJECT/'simulation/build/state_response_20261030'
BLOCKS=[64,128]
RC_GRID=[96.,112.,128.21494566172396,144.,160.]
PHASES=[0.,.5,1.,1.5]
VARIANTS=['frozen','electronics','response_inherited','response_spectrum']


def main():
    OUT.mkdir(exist_ok=False)
    for d in ['arrays','fits','configs','native','figures','logs','source_snapshot']:(OUT/d).mkdir()
    p=read(ACQ/'protocol.json');jp=read(JOINT/'validation/protocol.json')
    inputs=[ACQ/'record.json',ACQ/'protocol.json',PARENT/'record.json',JOINT/'validation/protocol.json',
            PROJECT/'analysis/paper_baseline/results/publication_20261028/point.json',
            PROJECT/'jupyter/run_records/library_update_20261030/research_record.json']
    groups={};native={}
    for source,run in SOURCES:
        entry=jp['entries'][f'run{run}_{source}'];cfg=PROJECT/entry['config'];inputs.append(cfg)
        old=load(PARENT/'arrays'/f'run{run}_{source}__identity.npz');ids=old['ids'][old['fixed']]
        save(OUT/'configs'/f'run{run}_ids.json',ids.tolist())
        native[source]={**entry,'selected_ids':ids.tolist(),'n_selected':len(ids),
            'id_file':relative(OUT/'configs'/f'run{run}_ids.json')}
        inputs.append(PARENT/'arrays'/f'run{run}_{source}__identity.npz')
        for seg in SEGMENTS:
            key=f'run{run}_{seg}';a=load(ACQ/'arrays'/f'{key}.npz')
            inputs += [ACQ/'arrays'/f'{key}.npz',DATA/f'{key}.npz',WIDTH/'arrays'/f'measured_{key}.npz']
            rr={'source':source,'run':run,'segment':seg,'ids':a['ids'].tolist(),'span':a['span'].tolist(),'folds':{}}
            for block in BLOCKS:
                fold=(a['ids']//block)%2
                rr['folds'][str(block)]={str(k):{'calibration_ids':a['ids'][fold!=k].tolist(),
                    'evaluation_ids':a['ids'][fold==k].tolist()} for k in [0,1]}
            groups[key]=rr
    save(OUT/'protocol.json',{'scope':__doc__,'code_sha256':sha(__file__),
        'checkpoint':subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT).decode().strip(),
        'inputs_sha256':{relative(f):sha(f) for f in inputs},'native':native,'groups':groups,
        'q_edges':p['q_edges'],'support_bins':p['support_bins'],'conditional_cuts':p['conditional_cuts'],
        'block_sizes':BLOCKS,'phase_offsets_ns':PHASES,'RC_grid_ns':RC_GRID,
        'transmission_bounds':[.45,.55],'variants':VARIANTS,
        'signal_noise_components':['inherited','exponential_400ns','exponential_1500ns'],
        'state_definition':'Pre-existing first60/middle20/last20 acquisition segments, not measured temperature or proven discrete physical states. New fits use opposite raw-block parity within each segment. Block128 primary,64 sensitivity.',
        'selection':'Original main-band IDs, original Q/support/first60 Q2 targets and PCA fixed; source groups are not event truth. Recalibration/checks are internal and previously explored, not blind.',
        'front_alignment':'Training-block average normalized ADC shape from original fractional t10; template-only interpolation on[-40,400]ns. Location fit uses original samples in[0,200]ns, nuisance amplitude/offset and shifts[-10,10]ns. ADC amplitude/shape are never replaced or rescaled for PID integration; template does not enter native simulation.',
        'phase':'Fixed original native avalanche times/charges, readout sampled with0/0.5/1/1.5ns aperture offsets. Fine-grid native integration, no interpolation of ADC/noise; no new transport history. Observed t10 fractional part is a noisy estimator, not true phase.',
        'regression':'Within each segment, blocked out-of-fold Q-bin intercept/slope+pretrigger state regressions, with and without sin/cos of estimated t10 phase at harmonics1,2. Report prediction error ratios; no event correction or causal attribution.',
        'calibration':'Pretrigger electronics from opposite-fold first128/256 samples. Deterministic5-point RC family and global transmission fit to training mean ADC and mean-time/tail constraints, fixed optical/sensor physics. Residual powers fit nonnegative full PID covariance and F50/102/202, inherited spectrum alone or inherited+400/1500ns spectra. No direct PID smearing.',
        'uncertainty':'Paired512 original-index block draws at64/128 and whole original mother resampling for final fixed-fit contrasts. Calibration-block refits separately report selection/power stability; conditional native intervals do not include complete fit/model uncertainty.',
        'next_native_check':'All candidates retained for internal comparisons. Any final recalibrated candidate requires frozen parameters and fresh independent Geant4 mothers to check simulation statistics; that does not create independent experimental validation.',
        'new_experimental_events':0,'new_independent_mothers':0,'default_changed':False})
    print('Frozen',len(groups),'segment groups,',sum(v['n_selected'] for v in native.values()),'original native mothers.',flush=True)


if __name__=='__main__':main()
