"""Audit native fixed-cohort replays, training separation and paired scientific outputs."""
import argparse
import base64
import platform
import shutil
import nbformat
import numpy as np
from rc_refinement_common import *
from verify_acquisition_state import independent_prefix_scores,check_interval
from evaluate_refined_rc import VARIANTS,NAMES,contrasts,selected_arrays
from diagnose_fluctuations import fluctuation_metrics


def source_files():
    names=['rc_refinement_common.py','calibrate_refined_rc.py','fit_refined_noise.py','evaluate_refined_rc.py',
        'plot_refined_rc.py','summarize_refined_rc.py','verify_refined_rc.py','verify_rc_cache.py',
        'state_response_common.py','prepare_state_response.py','prepare_acquisition_state.py','analyze_acquisition_state.py',
        'evaluate_state_response.py','fit_state_signal_noise.py','fit_pid_bandwidth.py','diagnose_pid_bandwidth.py',
        'fit_signal_fluctuations.py','refit_final_parameters.py','diagnose_fluctuations.py','diagnose_state_fold_mixture.py',
        'replay_readout.cc','CMakeLists.txt','README.md','vendor/SiPMResponse.cc','vendor/SiPMResponse.hh',
        'src/ReadoutSampling.cc','include/ReadoutSampling.hh','src/ReadoutNoise.cc','src/ReadoutResponse.cc','src/SignalDependentNoise.cc',
        'reference.py','verify_acquisition_state.py']
    return [PROJECT/'simulation'/n for n in names]+[PROJECT/'analysis/paper_baseline'/n for n in ['prepare.py','publication_stats.py']]+[
        PROJECT/'jupyter/codex_0001_build_refined_rc_notebook.py']


def check_hashes(m):
    for path,value in m.items():assert sha(PROJECT/path)==value,path
    return len(m)


def archive():
    assert not (OUT/'record.json').exists();v=read(OUT/'verification.json');check_hashes(v['outputs_sha256']);check_hashes(v['sources_sha256'])
    sources={}
    for f in source_files():
        dest=OUT/'source_snapshot'/f.relative_to(PROJECT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dest)
        sources[relative(f)]=sha(dest)
    outputs={relative(f):sha(f) for f in sorted(OUT.rglob('*')) if f.is_file() and 'source_snapshot' not in f.parts}
    nb=PROJECT/'jupyter/codex_0001_csi_refined_rc.ipynb';outputs[relative(nb)]=sha(nb)
    save(OUT/'record.json',{'scope':__doc__,'outputs_sha256':outputs,'sources_sha256':sources,
        'parent_record_sha256':sha(PARENT/'record.json'),'unique_original_mothers':470,'new_independent_mothers':0,
        'research_native_jobs':v['native_jobs'],'selected_native_replay_entries':v['native_replay_entries'],
        'engineering_paired_event_cases':read(OUT/'cache_verification.json')['paired_event_cases'],
        'default_changed':False,'limitations':'Refined conditional calibration, fixed-fit internal checks on previously explored experiments and reused MC. No independent final validation or particle efficiency measurement.'})
    print('Archived',len(outputs),'outputs and',len(sources),'sources.',flush=True)


def main():
    assert not (OUT/'verification.json').exists()
    p=read(OUT/'protocol.json');report={'scope':__doc__,'inputs_checked':check_hashes(p['inputs_sha256']),
        'inherited_inputs_checked':check_hashes(p['inherited_inputs_sha256'])}
    assert p['code_sha256']==sha(OUT/'source_snapshot/rc_refinement_common_frozen.py')
    cache=read(OUT/'cache_verification.json');assert cache['passed'] and cache['paired_event_cases']==216
    assert cache['source_sha256']==sha(PROJECT/'simulation/replay_readout.cc')
    assert cache['cached_binary_sha256']==sha(PROJECT/cache['cached_binary'])
    assert cache['legacy_binary_sha256']==sha(PROJECT/cache['legacy_binary'])
    m=read(OUT/'mean_calibration.json');s=read(OUT/'noise_selection.json');evaluation=read(OUT/'evaluation.json')
    for r,script in [(m,'calibrate_refined_rc.py'),(s,'fit_refined_noise.py'),(evaluation,'evaluate_refined_rc.py')]:
        assert r['code_sha256']==sha(PROJECT/'simulation'/script) and r['protocol_sha256']==sha(OUT/'protocol.json')
    native_jobs=0;entries=0;arrays_checked=0;cache_generations={}
    for stage in ['quiet_grid','mean_round0','mean_round1','mean_round2','base','noise_grid']:
        for source,run in SOURCES:
            log=read(OUT/'logs'/f'{stage}_{source}.json');assert sha(log['command'][0])==log['binary_sha256']
            assert sha(OUT/'configs'/f'jobs_{stage}_{source}.json')==log['plan_sha256']
            plan=read(OUT/'configs'/f'jobs_{stage}_{source}.json');base=read(PROJECT/p['native'][source]['config'])
            original=load(PARENT/'arrays'/f'run{run}_phase00.npz')
            for job in plan['jobs']:
                key=job['key'];r=log['jobs'][key];cfg=read(job['config'])
                assert sha(job['output'])==r['root_sha256'] and sha(job['config'])==r['config_sha256']
                ap=OUT/'arrays'/f'{key}.npz';assert sha(ap)==r['array_sha256']
                for field in ['geometry','optical','acquisition']:assert cfg[field]==base[field]
                assert cfg['sipm']['sensor']==base['sipm']['sensor']
                assert 96.<=cfg['sipm']['readout']['lowpass_ns']<=160.
                assert .45<=cfg['readout_response']['signal_transmission']<=.55
                z=load(ap);np.testing.assert_array_equal(z['ids'],original['ids']);np.testing.assert_array_equal(z['q'],original['q'])
                sc,q,shape=independent_prefix_scores(z['waves'],z['anchor'],z['baseline'],PCA)
                np.testing.assert_allclose(sc,z['scores'],rtol=2e-10,atol=2e-10);np.testing.assert_allclose(q,z['qcalc'],atol=1e-7)
                np.testing.assert_allclose(shape,z['shape'],atol=1e-12)
                ix=np.arange(min(5,len(q)));fm=fluctuation_metrics(z['waves'][ix],z['baseline'][ix],z['t10'][ix])[:,:,3]
                np.testing.assert_allclose(fm,z['F'][ix],rtol=2e-12,atol=2e-12)
                if stage=='noise_grid':
                    f=ROOT.TFile.Open(job['output']);info=json.loads(str(f.Get('RunInfo').GetString().Data()));f.Close()
                    assert info['completed'] and info['clipped_samples']==0 and info['config']==cfg
                    cache_generations[source]=max(cache_generations.get(source,0),info['base_response_cache']['generation'])
                native_jobs+=1;entries+=len(q);arrays_checked+=len(q)
            print('Verified',stage,source,len(plan['jobs']),'native jobs',flush=True)
    for key,r in m['groups'].items():
        fit=read(PROJECT/r['fit']);assert sha(PROJECT/r['fit'])==r['fit_sha256'];e=r['entry'];B=e['block'];ef=e['evaluation_fold']
        a=np.array(fit['calibration_ids']);b=np.array(fit['evaluation_ids'])
        assert not set(a)&set(b) and np.all((a//B)%2!=ef) and np.all((b//B)%2==ef)
        chosen=min(range(3),key=lambda i:fit['rounds'][i]['native_training_mean_loss']);assert chosen==fit['chosen_round']==r['selected_round']
        assert 96<=r['RC_ns']<=160 and .45<=r['transmission']<=.55
        selection=read(PROJECT/s['groups'][key]['fit']);assert sha(PROJECT/s['groups'][key]['fit'])==s['groups'][key]['fit_sha256']
        assert selection['calibration_ids']==fit['calibration_ids'] and selection['evaluation_ids']==fit['evaluation_ids']
        for family,c in selection['families'].items():
            target=as_arrays(selection['target'])
            for choice in c['choices']:
                loss,parts=joint_loss(as_arrays(choice['native_summary']),target)
                np.testing.assert_allclose(loss,choice['loss'],rtol=1e-12)
            assert c['chosen_index']==int(np.argmin([x['loss'] for x in c['choices']]))
            assert c['chosen_CV'] in p['CV_grid']+[c['inherited_CV']]
    bootstrap=load(OUT/'evaluation_bootstrap.npz');minimum=512
    for source,run in SOURCES:
        for seg in SEGMENTS:
            data,_=measured(run,seg);dp=summary(data,p,source);key=f'run{run}_{seg}'
            for B in p['block_sizes']:
                row=evaluation['groups'][key][str(B)];draw=bootstrap[f'{key}_B{B}']
                for iv,v in enumerate(VARIANTS):
                    z=selected_arrays(run,seg,B,v,m,s);r=row['variants'][v]
                    np.testing.assert_allclose(contrasts(summary(z,p,source),dp),[r['point'][n] for n in NAMES],atol=1e-12)
                    control=VARIANTS.index(r['control_comparator'])
                    for j,n in enumerate(NAMES):
                        check_interval(draw[:,iv,j],r['CI95'][n]);check_interval(draw[:,iv,j]-draw[:,1,j],r['paired_from_coarse'][n])
                        check_interval(draw[:,iv,j]-draw[:,control,j],r['paired_from_control'][n]);minimum=min(minimum,r['CI95'][n]['valid_draws'])
                    cv=np.array(r['shape_cov']);np.testing.assert_allclose(cv,cv.T,atol=1e-15)
                    assert np.linalg.eigvalsh(cv).min()>-1e-14;np.testing.assert_allclose(cv@np.ones(76),0.,atol=1e-14)
                    parts=np.array(r['fold_covariance_parts']);np.testing.assert_allclose(parts[0],parts[1]+parts[2],rtol=1e-12,atol=1e-15)
    assert minimum>=128
    nbpath=PROJECT/'jupyter/codex_0001_csi_refined_rc.ipynb';nb=nbformat.read(nbpath,as_version=4);nbformat.validate(nb)
    code=[c for c in nb.cells if c.cell_type=='code'];assert all(c.execution_count for c in code)
    assert not any(o.output_type=='error' for c in code for o in c.outputs)
    pics=[base64.b64decode(o.data['image/png']) for c in code for o in c.outputs if 'data' in o and 'image/png' in o.data]
    figures=read(OUT/'figure_record.json');assert len(pics)==len(figures['figures'])==5
    for blob,name in zip(pics,figures['figures']):assert blob==(OUT/'figures'/f'{name}.png').read_bytes()
    check_hashes(figures['inputs_sha256']);assert figures['code_sha256']==sha(PROJECT/'simulation/plot_refined_rc.py')
    decision=read(OUT/'decision.json');check_hashes(decision['inputs_sha256'])
    assert decision['code_sha256']==sha(PROJECT/'simulation/summarize_refined_rc.py')
    assert decision['validation_table_sha256']==sha(OUT/'validation_table.csv')
    assert not decision['promoted_to_default'] and decision['native_jobs']==native_jobs and decision['native_replay_entries']==entries
    report.update(native_jobs=native_jobs,native_replay_entries=entries,independent_gate_event_checks=arrays_checked,
        maximum_base_cache_generations=cache_generations,minimum_valid_bootstrap_draws=minimum,
        native_identity_contract_passed=True,calibration_evaluation_separated=True,training_selection_recomputed=True,
        paired_intervals_recomputed=True,notebook={'cells':len(nb.cells),'executed_cells':len(code),'exact_embedded_pngs':len(pics)},
        environment={'python':platform.python_version(),'numpy':np.__version__,'ROOT':ROOT.gROOT.GetVersion()})
    report['outputs_sha256']={relative(f):sha(f) for f in sorted(OUT.rglob('*')) if f.is_file() and 'source_snapshot' not in f.parts}
    report['outputs_sha256'][relative(nbpath)]=sha(nbpath);report['sources_sha256']={relative(f):sha(f) for f in source_files()}
    save(OUT/'verification.json',report);print('All scientific, notebook, cache and provenance checks passed.',flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--archive',action='store_true');args=ap.parse_args()
    archive() if args.archive else main()
