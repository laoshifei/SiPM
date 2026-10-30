"""Independent gate reconstruction, fit separation, paired intervals and frozen provenance."""
import argparse
import base64
import platform
import shutil
import numpy as np
import nbformat
from state_response_common import *
from verify_acquisition_state import independent_prefix_scores,check_interval
from diagnose_fluctuations import fluctuation_metrics
from analyze_acquisition_state import interval
from evaluate_state_response import NAMES,VARIANTS,comparison,assemble


def sources():
    names=['prepare_state_response.py','state_response_common.py','analyze_state_alignment.py','calibrate_state_response.py',
        'fit_state_signal_noise.py','evaluate_state_response.py','plot_state_response.py','verify_state_response.py',
        'diagnose_state_fold_mixture.py','summarize_state_response.py',
        'replay_readout.cc','CMakeLists.txt','README.md','reference.py','prepare_acquisition_state.py','analyze_acquisition_state.py',
        'prepare_noise_model.py','prepare_run12_noise_state.py','refit_baseline_mixture.py','characterize_readout.py',
        'fit_pid_bandwidth.py','diagnose_pid_bandwidth.py','fit_signal_fluctuations.py','diagnose_fluctuations.py',
        'refit_final_parameters.py','verify_acquisition_state.py','vendor/SiPMResponse.cc','vendor/SiPMResponse.hh',
        'include/ReadoutSampling.hh','src/ReadoutSampling.cc','src/ReadoutNoise.cc','src/ReadoutResponse.cc','src/SignalDependentNoise.cc',
        'tests/test_readout_sampling.cc']
    return [PROJECT/'simulation'/n for n in names]+[PROJECT/'analysis/paper_baseline'/n for n in ['prepare.py','publication_stats.py']]+[
        PROJECT/'jupyter/codex_0001_build_state_response_notebook.py']


def hashes(mapping):
    for path,value in mapping.items():assert sha(PROJECT/path)==value,path
    return len(mapping)


def archive():
    assert not (OUT/'record.json').exists();v=read(OUT/'verification.json');hashes(v['verified_outputs_sha256']);hashes(v['verified_sources_sha256'])
    snapshot={}
    for f in sources():
        dest=OUT/'source_snapshot'/f.relative_to(PROJECT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dest)
        snapshot[relative(f)]=sha(dest)
    outputs={relative(f):sha(f) for f in sorted(OUT.rglob('*')) if f.is_file() and 'source_snapshot' not in f.parts}
    nb=PROJECT/'jupyter/codex_0001_csi_state_response.ipynb';outputs[relative(nb)]=sha(nb)
    save(OUT/'record.json',{'scope':__doc__,'outputs_sha256':outputs,'sources_sha256':snapshot,
        'unique_original_mothers_with_fixed_waveforms':470,'new_independent_mothers':0,'new_experimental_events':0,
        'native_jobs':v['native_jobs'],'native_selected_replay_entries':v['native_replay_entries'],
        'promoted_to_default':False,'limits':'Conditional state response on previously explored data, fixed-cohort replays; no independent experimental or MC validation of fitted candidates.'})
    print('Archived',len(outputs),'outputs and',len(snapshot),'source files.',flush=True)


def main():
    assert not (OUT/'verification.json').exists()
    p=read(OUT/'protocol.json');a=read(OUT/'alignment.json');e=read(OUT/'evaluation.json');mc=read(OUT/'mean_calibration.json')
    checks={'scope':__doc__,'input_hashes_checked':hashes(p['inputs_sha256']),'native_jobs':0,'native_replay_entries':0}
    assert p['code_sha256']==sha(PROJECT/'simulation/prepare_state_response.py')
    assert read(OUT/'fit_design.json')['code_sha256']==sha(OUT/'source_snapshot/state_response_common_before_root_reader_fix.py')
    n_rebuilt=0
    for stage in ['initial','state_base','signal_candidates']:
        for source,run in SOURCES:
            log=read(OUT/'logs'/f'{stage}_{source}.json');assert log['binary_sha256']==sha(BUILD/'replay_readout')
            assert log['plan_sha256']==sha(OUT/'configs'/f'jobs_{stage}_{source}.json')
            plan=read(OUT/'configs'/f'jobs_{stage}_{source}.json');base=read(PROJECT/p['native'][source]['config'])
            original=load(OUT/'arrays'/f'run{run}_phase00.npz')
            assert len(plan['ids'])==len(set(plan['ids']))==p['native'][source]['n_selected']
            for job in plan['jobs']:
                key=job['key'];cfg=read(job['config']);r=log['jobs'][key]
                assert sha(job['config'])==r['config_sha256']==job['config_sha256']
                assert sha(PROJECT/r['root'])==r['root_sha256']
                assert sha(OUT/'arrays'/f'{key}.npz')==r['array_sha256']
                for k in ['geometry','optical','acquisition']:assert cfg[k]==base[k]
                assert cfg['sipm']['sensor']==base['sipm']['sensor']
                z=load(OUT/'arrays'/f'{key}.npz');np.testing.assert_array_equal(z['ids'],original['ids']);np.testing.assert_array_equal(z['q'],original['q'])
                score,q,shape=independent_prefix_scores(z['waves'],z['anchor'],z['baseline'],PCA)
                if job['phase_ns']>0:assert (z['anchor']+1500).max()<z['waves'].shape[1]-1
                np.testing.assert_allclose(score,z['scores'],rtol=2e-10,atol=2e-10)
                np.testing.assert_allclose(q,z['qcalc'],rtol=1e-12,atol=1e-7);np.testing.assert_allclose(shape,z['shape'],atol=1e-12)
                # Independent legacy loop checks each native array and integer-boundary handling.
                ix=np.arange(min(8,len(q)))
                fm=fluctuation_metrics(z['waves'][ix],z['baseline'][ix],z['t10'][ix])[:,:,3]
                np.testing.assert_allclose(fm,z['F'][ix],rtol=2e-12,atol=2e-12)
                checks['native_jobs']+=1;checks['native_replay_entries']+=len(q);n_rebuilt+=len(q)
            if stage=='initial':assert log['jobs'][f'run{run}_phase00']['equal_parent_ADC']==len(original['ids'])
    print('Native jobs: hashes, fixed sensor/transport, original IDs/Q and independent gate algebra passed.',flush=True)
    for key,r in mc['groups'].items():
        fit=read(PROJECT/r['fit']);assert sha(PROJECT/r['fit'])==r['fit_sha256']
        B=r['block'];ef=r['evaluation_fold'];cal=np.array(fit['calibration_ids']);ev=np.array(fit['evaluation_ids'])
        assert not set(cal)&set(ev) and np.all((cal//B)%2!=ef) and np.all((ev//B)%2==ef)
        assert np.all((np.array(fit['pretrigger_calibration_ids'])//B)%2!=ef)
        assert fit['selected_RC_ns'] in p['RC_grid_ns'] and .45<=fit['selected_transmission']<=.55
        sf=read(OUT/'fits'/f'{key}_signal.json');assert sf['calibration_ids']==fit['calibration_ids'] and sf['evaluation_ids']==fit['evaluation_ids']
        for v in ['response_inherited','response_spectrum']:assert min(sf[v]['component_powers'])>=0 and sf[v]['final_strength']>=0
    boot=load(OUT/'evaluation_bootstrap.npz');ab=load(OUT/'alignment_bootstrap.npz');minimum=512
    for source,run in SOURCES:
        for seg in SEGMENTS:
            data,_=measured(run,seg);dp=moments(data,p,source);key=f'run{run}_{seg}'
            for B in p['block_sizes']:
                front=load(OUT/'arrays'/f'{key}_alignment_B{B}.npz');fold=(data['ids']//B)%2
                np.testing.assert_array_equal(front['ids'],data['ids'])
                s,q,_=independent_prefix_scores(data['waves'],front['data_anchor'],data['baseline'],PCA)
                np.testing.assert_allclose(s,front['data_scores'],atol=1e-10);np.testing.assert_allclose(q,front['data_qcalc'],atol=1e-7)
                n_rebuilt+=len(q)
                for ef,models in a['groups'][key][str(B)]['regression_fits'].items():
                    for model in models.values():assert not set(model['training_ids'])&set(data['ids'][fold==int(ef)])
                row=e['groups'][key][str(B)];draw=boot[f'{key}_B{B}']
                for iv,v in enumerate(VARIANTS):
                    m=assemble(run,seg,B,v);mp=moments(m,p,source);r=row['variants'][v]
                    np.testing.assert_allclose(comparison(mp,dp),[r['point'][name] for name in NAMES],atol=1e-12)
                    for j,name in enumerate(NAMES):
                        check_interval(draw[:,iv,j],r['CI95'][name]);check_interval(draw[:,iv,j]-draw[:,0,j],r['paired_change_from_frozen'][name])
                        minimum=min(minimum,r['CI95'][name]['valid_draws'])
                    cv=np.array(r['shape_cov']);np.testing.assert_allclose(cv,cv.T,atol=1e-15)
                    assert np.linalg.eigvalsh(cv).min()>-1e-14;np.testing.assert_allclose(cv@np.ones(76),0.,atol=1e-14)
                fd=read(OUT/'fold_mixture_diagnostic.json')['groups'][key][str(B)]
                for parts in [fd['data_parts']]+[r['parts'] for r in fd['variants'].values()]:
                    parts=np.asarray(parts);np.testing.assert_allclose(parts[0],parts[1]+parts[2],rtol=1e-12,atol=1e-15)
                for j in range(4):check_interval((ab[f'{key}_B{B}_front'][:,1]-ab[f'{key}_B{B}_front'][:,0])[:,j],a['groups'][key][str(B)]['paired_front_change_intervals'][j])
    checks.update(independent_gate_event_controls=n_rebuilt,minimum_valid_bootstrap_draws=minimum,
        calibration_evaluation_blocks_disjoint=True,paired_intervals_recomputed=True,shape_covariances_PSD_with_zero_sum=True)
    assert minimum>=128
    nbpath=PROJECT/'jupyter/codex_0001_csi_state_response.ipynb';nb=nbformat.read(nbpath,as_version=4);nbformat.validate(nb)
    code=[c for c in nb.cells if c.cell_type=='code'];assert all(c.execution_count for c in code)
    assert not any(o.output_type=='error' for c in code for o in c.outputs)
    pics=[base64.b64decode(o.data['image/png']) for c in code for o in c.outputs if 'data' in o and 'image/png' in o.data]
    figs=read(OUT/'figure_record.json');assert len(pics)==len(figs['figures'])==5
    for blob,name in zip(pics,figs['figures']):assert blob==(OUT/'figures'/f'{name}.png').read_bytes()
    hashes(figs['inputs_sha256']);assert not read(OUT/'decision.json')['promoted_to_default']
    checks['notebook']={'cells':len(nb.cells),'executed_cells':len(code),'embedded_pngs':len(pics),'sha256':sha(nbpath)}
    checks['environment']={'python':platform.python_version(),'numpy':np.__version__,'ROOT':ROOT.gROOT.GetVersion()}
    checks['verified_outputs_sha256']={relative(f):sha(f) for f in sorted(OUT.rglob('*')) if f.is_file() and 'source_snapshot' not in f.parts}
    checks['verified_outputs_sha256'][relative(nbpath)]=sha(nbpath)
    checks['verified_sources_sha256']={relative(f):sha(f) for f in sources()}
    save(OUT/'verification.json',checks);print('Statistical separation, notebook, figures and provenance verified.',flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--archive',action='store_true');args=ap.parse_args()
    archive() if args.archive else main()
