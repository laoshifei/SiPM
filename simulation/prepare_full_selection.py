"""Freeze an exhaustive existing-mother selection audit before new replay."""
from full_selection_common import *


def main():
    assert not (OUT/'protocol.json').exists(),'Refuse frozen protocol overwrite'
    assert not list((OUT/'native').glob('*')) and not list((OUT/'arrays').glob('*')),'Refuse preparation after outcomes'
    for name in ['configs','native','logs','arrays','evaluation','figures','source_snapshot']:
        (OUT/name).mkdir(parents=True,exist_ok=True)
    parent=read(PARENT/'protocol.json');previous=read(SCALES/'protocol.json')
    inputs=[PARENT/'protocol.json',PARENT/'record.json',SCALES/'protocol.json',SCALES/'record.json',
        REPLAY/'selection_protocol.json',REPLAY/'replay_protocol.json',PROJECT/'analysis/paper_baseline/results/publication_20261028/point.json']
    entries={}
    for source,run in SOURCES:
        context=f'run{run}_last20_B128';data_path=REPLAY/'point_arrays'/f'{context}_data.npz'
        data=load(data_path);inputs += [data_path,REPLAY/'point_arrays'/f'{context}_current.npz',
            REPLAY/'point_arrays'/f'{context}_zero_cv.npz',REPLAY/'point_evaluation'/f'{context}_bootstrap.npz',
            SCALES/'references'/f'{context}.json',SCALES/'point_evaluation'/f'{context}.json']
        configs=[]
        for ef in [0,1]:
            cp=PROJECT/parent['responses'][f'{context}_eval{ef}_spectrum']['config'];inputs.append(cp)
            configs.append(str(cp))
        for batch in range(4):
            key=f'b{batch:02d}_run{run}';entry=parent['entries'][key]
            old_path=PARENT/'arrays'/f'{key}_frozen.npz';old=load(old_path)
            folds=full_folds(entry['events'],data,parent,source,run,batch)
            from mc_statistics_common import assign
            np.testing.assert_array_equal(folds[old['ids']],assign(old,data,parent,source,run,2,128,batch))
            plan={'key':key,'source':source,'run':run,'batch':batch,'seed':entry['seed'],
                'events':entry['events'],'configs':configs,'folds':folds.tolist(),
                'output':str(OUT/'native'/f'{key}.root')}
            plan_path=OUT/'configs'/f'{key}.json'
            if plan_path.exists():assert read(plan_path)==plan,'Existing preparation plan differs'
            else:save(plan_path,plan)
            inputs += [old_path,PROJECT/entry['path'],PROJECT/entry['config'],
                PARENT/'transport/arrays'/f'{key}.npz',PARENT/'transport/extraction'/f'{key}.json']
            expected=read(PARENT/'transport/extraction'/f'{key}.json')['root_sha256']
            assert sha(PROJECT/entry['path'])==expected,'Native pool no longer matches archived parent'
            entries[key]={**entry,'plan':relative(plan_path),'plan_sha256':sha(plan_path),
                'output':relative(OUT/'native'/f'{key}.root'),'full_parent_sha256':expected}
    code=['simulation/full_selection_common.py','simulation/prepare_full_selection.py',
        'simulation/run_full_selection.py','simulation/extract_full_selection.py',
        'simulation/replay_full_pool.cc','simulation/CMakeLists.txt']
    inherited=['simulation/src/ReadoutReplica.cc','simulation/src/ReadoutSampling.cc',
        'simulation/src/SignalDependentNoise.cc','simulation/src/ReadoutNoise.cc','simulation/src/ReadoutResponse.cc',
        'simulation/vendor/SiPMResponse.cc','simulation/charge_scale_common.py','simulation/compare.py',
        'simulation/state_response_common.py','simulation/mc_statistics_common.py','simulation/extract_pid.py',
        'analysis/paper_baseline/prepare.py','analysis/paper_baseline/refine_waveforms.py',
        'simulation/prepare_acquisition_state.py','simulation/analyze_acquisition_state.py']
    save(OUT/'protocol.json',{
        'scope':'Exhaustive existing native mother pool, frozen last20/B128/extended response. Diagnostic, not blind validation.',
        'checkpoint_commit':'4a00b1c4c6fabd9e8289b51842d28f6bab9cff71',
        'entries':entries,'q_edges':parent['q_edges'],'support_bins':parent['support_bins'],
        'conditional_cuts':parent['conditional_cuts'],'variants':VARIANTS,'conditions':CONDITIONS,
        'replicas':[0,1],'primary_replica':0,'primary_context':'last20_B128',
        'code_sha256':{name:sha(PROJECT/name) for name in code},
        'inherited_code_sha256':{name:sha(PROJECT/name) for name in inherited},
        'inputs_sha256':{relative(path):sha(path) for path in sorted(set(inputs))},
        'selection':'Every native mother, including zero deposits, is replayed. No prefilter by old Q, amplitude, deposited energy or truth. Current QC uses the original wave_features/advanced_features with global RMS median across all mothers in each batch; amplitude >=200 ADC, t10>=384, inclusive old common-Q endpoints. Source support is applied only when comparing weighted statistics. Also audit fold-specific global RMS as a sensitivity.',
        'folds':'Extend the original Bernoulli full-parent assignment; preserve every originally selected assignment. Each mother gets the fixed opposite-data-fold response. No refitting.',
        'comparisons':'Original cohort and original Q; original cohort and current Q; original cohort and current QC/Q; complete pool and current QC/Q. Keep all Q bins, half-Q masses, PCA and opposite-fold F references frozen. Compare F50/F102/F202 means, SD/covariance, PID SD/50%/95% widths/covariance, and mean ADC.',
        'uncertainty':'512 inherited paired data-block and old full-mother category counts. Extend to the union of newly needed mothers by the exact conditional multinomial over previously excluded mothers, separately by batch. One count per mother shared by all variants. Global QC thresholds, response and references remain fixed; intervals exclude fit/systematic errors and fresh readout randomization.',
        'extension_seed':731102801,'replica_boundary':'Replica0 is the inherited matched-stream comparison. Replica1 is one prespecified positive all-readout stream sensitivity, not a converged expectation over electronics. Show separately, never count replicas as new mothers.',
        'acceptance_boundary':'No automatic parameter/default/PID promotion. Selection fractions are conditional MC software cuts, not validated hardware trigger efficiency. MC ancestry may describe entrants; measured Am has alpha and gamma/e components without event truth.',
        'available_existing_mothers':256000,'new_independent_mothers':0,'new_experimental_events':0,
        'logical_readout_records':1024000,'default_changed':False,'official_PID_changed':False})
    print('Frozen complete-pool selection audit: 256000 mothers, replicas0/1, current/zero-CV.',flush=True)


if __name__=='__main__':main()
