"""Freeze repeated-noise cases and a zero-CV control on existing Geant4 histories."""
import copy
import subprocess
from pathlib import Path
from noise_replay_common import *


def main():
    assert not (OUT/'replay_protocol.json').exists()
    p=read(OUT/'selection_protocol.json');parent=read(PARENT/'protocol.json');pr=read(PARENT/'record.json')
    tests=read(OUT/'replay_tests.json');assert tests['ctest_passed']==6
    inputs={relative(OUT/f):sha(OUT/f) for f in ['selection_protocol.json','selection_analysis.json','replay_tests.json']}
    for name,entry in parent['entries'].items():
        path=PROJECT/entry['path'];value=sha(path);assert value==pr['outputs_sha256'][entry['path']]
        inputs[entry['path']]=value
    plans={};contexts={};groups={};zero_configs={};jobs_by_key={}
    def zero_config(path):
        path=Path(path);cfg=read(path)
        if cfg['signal_dependent_noise'].get('event_variance_cv',0.)==0.:return path
        key=sha(path)
        if key not in zero_configs:
            cfg=copy.deepcopy(cfg);cfg['signal_dependent_noise']['event_variance_cv']=0.
            cfg['status']='Fixed-history zero signal-variance-CV diagnostic; all other response and physical parameters inherited.'
            dest=OUT/'configs'/f'zero_cv_{key[:16]}.json';save(dest,cfg);zero_configs[key]=dest
        return zero_configs[key]
    def new_plan(name,entry,ids,kind):
        plan={'name':name,'entry_key':entry['key'],'parent':entry['path'],'ids':list(map(int,ids)),
              'kind':kind,'jobs':[],'registry':{}}
        plans[name]=plan;return plan
    def add(plan,config,context,fold,array,replica=0,scope='all_readout',identity=False):
        config=Path(config);signature=(sha(config),scope,replica,context,fold,identity)
        if signature in plan['registry']:return plan['registry'][signature]
        key=f"{plan['name']}_j{len(plan['jobs']):03d}"
        job={'key':key,'config':str(config),'output':str(OUT/'replay'/f'{key}.root'),
            'phase_ns':0.,'noise_scope':scope,'noise_replica':replica,'require_identity':identity,
            'context':context,'fold':fold,'reference_array':relative(array),'native_identity':identity}
        plan['jobs'].append(job);plan['registry'][signature]=key;jobs_by_key[key]=job
        for path in [config,array]:
            if relative(path) not in inputs:inputs[relative(path)]=sha(path)
        return key
    for source,run in SOURCES:
        primary=f'run{run}_last20_B128';a=load(QDC/'arrays'/f'{primary}_spectrum_CV_fit.npz')
        inputs[relative(QDC/'arrays'/f'{primary}_spectrum_CV_fit.npz')]=sha(QDC/'arrays'/f'{primary}_spectrum_CV_fit.npz')
        groups[source]={'run':run,'context':primary,'logical_cases':{k:[] for k in REPLAY_CASES},'current_identity':[],'zero_identity':[]}
        for batch in BATCHES:
            entry=parent['entries'][f'b{batch:02d}_run{run}'];base_path=PARENT/'arrays'/f'b{batch:02d}_run{run}_frozen.npz';base=load(base_path)
            auxiliary=new_plan(f'b{batch:02d}_run{run}_point',entry,base['ids'],'point')
            add(auxiliary,PROJECT/entry['config'],primary,0,base_path,identity=True)
            for fold in [0,1]:
                take=(a['batch']==batch)&(a['fold']==fold);ids=a['ids'][take]-batch*EVENTS[source]
                plan=new_plan(f'b{batch:02d}_run{run}_eval{fold}',entry,ids,'repeated_noise')
                original=PROJECT/parent['responses'][f'{primary}_eval{fold}_spectrum']['config'];zero=zero_config(original)
                reference=PARENT/'arrays'/f'b{batch:02d}_run{run}_last20_B128_eval{fold}_spectrum.npz'
                current_key=add(plan,original,primary,fold,reference);zero_key=add(plan,zero,primary,fold,reference)
                groups[source]['current_identity'].append(current_key);groups[source]['zero_identity'].append(zero_key)
                for case,(scope,R,is_zero) in REPLAY_CASES.items():
                    sequence=[add(plan,zero if is_zero else original,primary,fold,reference,replica,scope) for replica in range(1,R+1)]
                    groups[source]['logical_cases'][case].append({'batch':batch,'fold':fold,'keys':sequence})
            for seg in SEGMENTS:
                for B in p['block_sizes']:
                    context=f'run{run}_{seg}_B{B}'
                    if context not in contexts:contexts[context]={'source':source,'run':run,'segment':seg,'block':B,'parts':[]}
                    for fold in [0,1]:
                        original=PROJECT/parent['responses'][f'{context}_eval{fold}_spectrum']['config'];zero=zero_config(original)
                        reference=PARENT/'arrays'/f'b{batch:02d}_run{run}_{seg}_B{B}_eval{fold}_spectrum.npz'
                        if context==primary:
                            primary_plan=plans[f'b{batch:02d}_run{run}_eval{fold}']
                            key=add(primary_plan,zero,context,fold,reference)
                        elif zero==original:key=None
                        else:key=add(auxiliary,zero,context,fold,reference)
                        for path in [original,reference]:
                            if relative(path) not in inputs:inputs[relative(path)]=sha(path)
                        contexts[context]['parts'].append({'batch':batch,'fold':fold,'key':key,'config':relative(zero),
                            'reference_array':relative(reference),'unchanged_zero_CV':zero==original})
    for plan in plans.values():
        plan.pop('registry');path=OUT/'configs'/f"{plan['name']}_plan.json"
        save(path,{'ids':plan['ids'],'jobs':plan['jobs']});plan['path']=relative(path);plan['sha256']=sha(path)
    for group in groups.values():
        context=group['context'];key=context.rsplit('_B',1)[0]
        ref=QDC/'references'/f'{context}.json';inputs[relative(ref)]=sha(ref)
    for context in contexts:
        for f in [QDC/'references'/f'{context}.json',QDC/'arrays'/f'{context}_data.npz',
                  QDC/'arrays'/f'{context}_spectrum_CV_fit.npz']:
            inputs[relative(f)]=sha(f)
    code=['simulation/noise_replay_common.py','simulation/prepare_noise_replay.py','simulation/run_noise_replay.py',
          'simulation/test_noise_replay.py','simulation/include/ReadoutReplica.hh','simulation/src/ReadoutReplica.cc',
          'simulation/replay_readout.cc','simulation/CMakeLists.txt','simulation/tests/test_readout_replica.cc',
          'simulation/src/ReadoutSampling.cc','simulation/src/ReadoutNoise.cc','simulation/src/SignalDependentNoise.cc']
    protocol={k:p[k] for k in ['q_edges','support_bins','conditional_cuts','block_sizes','variants']}
    protocol.update(scope=__doc__,checkpoint_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT).decode().strip(),
        inputs_sha256=inputs,code_sha256={k:sha(PROJECT/k) for k in code},binary_sha256=tests['binary_sha256'],
        parent_protocol_sha256=sha(PARENT/'protocol.json'),features=FEATURES,metrics=METRICS,plans=plans,groups=groups,point_contexts=contexts,
        repeat_cases={k:{'scope':v[0],'replicates':v[1],'zero_signal_variance_CV':v[2]} for k,v in REPLAY_CASES.items()},
        physical_replay_entries=sum(len(x['ids'])*len(x['jobs']) for x in plans.values()),
        fixed_selected_mothers=3661,new_independent_mothers=0,new_experimental_events=0,
        design='Primary last20/block128/RC+extended, both sources, all four original transport seeds and all original selected IDs/Q. Replica0 checks inherited ADC. Positive replicas1..16 all-readout and zero-CV all-readout;1..8 signal-only and electronics-only. Distinct source streams share replica indices for paired controls; exact identical cases reuse one physical file. Secondary zero-CV replica0 checks all original segments/blocks.',
        seed='ReadoutReplica seed=EventSeed(original_readout_seed,0x5245504c00000000+index), index0 exactly original; scope independently chooses baseline and signal seeds. Parent readout seed remains EventSeed(transport_seed+31,event_id). No transport or avalanche RNG touched.',
        zero_CV='Set only signal_dependent_noise.event_variance_cv=0; retain variance_per_signal_adc, FIR, electronics, geometry, scintillation, SiPM and readout mean response. This is a paired diagnostic hypothesis, not a fitted/promoted candidate. Already-zero cases are exact identities.',
        finite_repeat='For within-Q normalized mother weights w, sample conditional covariance S_i over R replicas: W=sum w_i S_i; B_raw=Cov_w(replica means); B=B_raw-sum w_i(1-w_i)S_i/R. Keep negative corrected entries. Expected original one-readout weighted covariance is B+sum w_i(1-w_i)S_i. Exact flat empirical covariance=B_raw+(R-1)W/R. Between-mother term still includes signal-dependent conditional noise means and original ADC selection.',
        uncertainty='512 paired original acquisition-block/full-mother count vectors for point controls. Repeated-noise uncertainty resamples whole mothers with all of their readouts together; never count replicas as independent mothers. Coalesce duplicated mother weights before finite-R correction. Compare first8/last8/all16 for repeat sensitivity; no outcome-driven extension or source/event exclusion.',
        selection='All original selected mothers retained across replicas and CV cases. Report crossing of the old0.4 band but do not reselect repeated readouts. Selection audit showed unchanged matched-band F202 contrasts on frozen cohorts; this does not prove general trigger/source-population equivalence.',
        promotion_guard='Report paired changes in absolute log ratios for F means, F SDs, PID SDs and central50/95% widths, plus unscaled mean-wave residuals in all12 contexts. If reducing CV clearly worsens any of these previously constrained quantities, retain the current defaults and do not spend new transport on this control. A favorable control would still need a separately frozen independent-mother validation, not be declared experimentally blind or physically identified.',
        default_changed=False,official_PID_changed=False)
    save(OUT/'replay_protocol.json',protocol)
    print('Frozen',len(plans),'plans,',len(jobs_by_key),'physical readout files and',protocol['physical_replay_entries'],'replay entries; new mothers=0.',flush=True)


if __name__=='__main__':main()
