"""Freeze a truth-aware, common-readout PID study using existing optical histories."""
import copy
import json
from pathlib import Path
import numpy as np
from reference import PROJECT
from publication_stats import sha, save

OUT=PROJECT/'simulation/results/pid_20261029'
PARENT=PROJECT/'simulation/results/waveform_refit_20261029'
SOURCES=('Co60','Am241')
STATES=('run3','run12','run18')


def read(path):return json.loads(Path(path).read_text())


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    parent=read(PARENT/'record.json')
    configs={source:read(PARENT/f'independent/{source}_calibrated.json') for source in SOURCES}
    profiles={'run3':configs['Co60'],'run12':configs['Am241'],
              'run18':read(PARENT/'run18/Am241_run18_first60.json')}
    assert configs['Co60']['optical']==configs['Am241']['optical']
    assert configs['Co60']['sipm']['sensor']==configs['Am241']['sipm']['sensor']
    parents={s:PARENT/'independent'/s.lower()/'events.root' for s in SOURCES}
    inherited={('run3','Co60'):parents['Co60'],('run12','Am241'):parents['Am241'],
               ('run18','Am241'):PARENT/'run18/first60/events.root'}
    paths={};jobs=[];splits={}
    for source,count in [('Co60',4000),('Am241',24000)]:
        rng=np.random.default_rng(np.random.SeedSequence([291029701, 60 if source=='Co60' else 241]))
        split=np.ones(count,dtype=np.int8);split[rng.permutation(count)[:count//2]]=0
        splits[source]=split
    np.savez_compressed(OUT/'mother_splits.npz',**splits)
    for state in STATES:
        for source in SOURCES:
            cfg=copy.deepcopy(configs[source]);profile=profiles[state]
            for key in ['sipm','electronics_noise','signal_dependent_noise','readout_response']:
                cfg[key]=copy.deepcopy(profile[key])
            cfg['status']=f'PID common-readout scenario {state}: same response for both sources; geometry and optical histories retained'
            cfg['pid_study']={'common_readout':state,'role':'Conditioned sensitivity scenario, not a newly measured universal response.',
                'source_identity_is_not_a_feature':True,'threshold_proxy_not_used_for_selection':True}
            cfgpath=OUT/'configs'/f'{state}_{source}.json';save(cfgpath,cfg)
            if (state,source) in inherited:
                root=inherited[state,source];kind='existing_native' if state!='run18' else 'existing_replay'
            else:
                folder=OUT/state/source.lower();folder.mkdir(parents=True)
                root=folder/'events.root';kind='new_replay'
                jobs.append({'state':state,'source':source,'command':[str(PROJECT/'simulation/build/replay_response'),
                     str(parents[source]),str(cfgpath),str(root)]})
            paths[f'{state}_{source}']={'path':str(root.relative_to(PROJECT)),'kind':kind,
                'scenario_config':str(cfgpath.relative_to(PROJECT)),'scenario_config_sha256':sha(cfgpath),
                'parent':str(parents[source].relative_to(PROJECT))}
    point=PROJECT/'analysis/paper_baseline/results/publication_20261028/point.json'
    protocol={'scope':__doc__,'code_sha256':sha(Path(__file__)),'date':'2026-10-29',
      'parent_record_sha256':sha(PARENT/'record.json'),
      'inputs_sha256':{str(point.relative_to(PROJECT)):sha(point),
        **{str(path.relative_to(PROJECT)):sha(path) for path in parents.values()},
        str((PARENT/'run18/first60/events.root').relative_to(PROJECT)):sha(PARENT/'run18/first60/events.root')},
      'source_rows':paths,'replay_jobs':jobs,
      'replay_binary_sha256':sha(PROJECT/'simulation/build/replay_response'),
      'primary_state':'run12','sensitivity_states':['run3','run18'],
      'common_readout':'Identical SiPM/readout, baseline-noise states and signal residual for Co and Am within each scenario. Original source geometries retained. No refitting.',
      'acquisition':'All sampling/record settings equal except source-specific historical amplitude proxy (120/160), ignored by PID. Both sources use observed amplitude>=200 ADC selection.',
      'truth':'Primary: >=90% CsI deposited energy from one ancestry family: alpha, gamma, decay electron; otherwise mixed/other. Zero deposition is no-deposit. Alpha secondary electrons stay in alpha family. Charge ancestry is a separate sensitivity, never the ADC score.',
      'truth_sensitivity':{'energy_dominance':[.8,.9,.95],'avalanche_charge_dominance':.9},
      'selection':'Native ADC recomputed quality + primary pretrigger veto, amplitude>=200 ADC, t10>=384 and complete 3us integrals; no cut on any PID score. Measured IDs remain frozen.',
      'q_edges':read(PARENT/'independent/analysis/validation.json')['q_edges'],
      'methods':['qdc','negative_mean_time','pca_projection'],
      'features':'Fixed [-40,500)ns short / [-40,3000)ns long charge; negative mean time in long gate; published frozen experimental-training PCA (76 blocks). Lower scores are alpha-like. No MC PCA retraining.',
      'split':'Before score extraction, uniformly split original mothers 50/50 per source, seed [291029701,60/241]. Same mother split across all response replays. Reused old validation histories; no blind/new experimental claim.',
      'split_sha256':sha(OUT/'mother_splits.npz'),
      'operating_point':'In primary run12 scenario, freeze per-method global score<=threshold at empirical95th percentile of training-alpha scores in common Q. Apply unchanged to test mothers in every scenario. No test threshold tuning.',
      'evaluation':'Conditional on selected recorded pulses, not detection efficiency. ROC on held-out mothers; exact binomial intervals for unweighted acceptance/leakage. Report counts, mixed exclusions, per-Q bins and equal-bin AUC; do not interpret zero errors as zero true leakage.',
      'minimum_counts':{'ROC_per_Q_test_per_class':5,'descriptive_truth_FOM_per_class':10,'source_mixture_FOM_per_source':10},
      'FOM':'Truth moment Gaussian-equivalent FOM=|meanA-meanG|/(2sqrt(2ln2)*(sdA+sdG)); no Gaussian-tail claim. For experiment-MC comparison also fit identical two Gaussian cores + broad Cauchy nuisance per source/Q, with convergence/floor checks; cores are not measured particle labels.',
      'uncertainty':'512 paired whole-mother multinomial draws for ROC/FOM/scenario differences, fixed calibration, PCA and threshold. Measured median/IQR comparisons use original-index block64 where reported. Exact binomial intervals condition on frozen operating point and selection.',
      'experimental_checks':'Use frozen run3/run12/run18 first/middle/last arrays. Compare distributions/retained fractions only, never label measured events by source or report measured particle error rates.',
      'MC_accounting':{'unique_parent_histories':28000,'new_transport':0,'planned_new_replay_parent_entries':32000,
                       'note':'All three states reuse the same4000 Co and24000 Am mothers. Do not add states as independent samples.'},
      'known_limits':['Previous waveforms and Q spectra do not agree in every observable.',
                     'Only129/172 common-Q events in parent configurations; low leakage cannot be established with this pilot size.',
                     'Particle-dependent scintillation parameters were fitted to source-enriched experimental shapes; simulation separation is conditional on those assumptions.',
                     'Source distance/capsule, PDE/gain and summing network remain conditional hypotheses.']}
    save(OUT/'protocol.json',protocol)
    print('Frozen PID protocol; new replay jobs:',len(jobs),flush=True)
    for job in jobs:print(' '.join(job['command']),flush=True)


if __name__=='__main__':main()
