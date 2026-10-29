"""Verify and freeze pretrigger cross-fit, paired native replay and notebook."""
import argparse,base64,copy,functools,json,platform,shutil
from pathlib import Path
import numpy as np
import scipy
import ROOT
import nbformat
from prepare_run12_noise_state import OUT,PARENT,PROJECT,DATA,WIDTH,SEGMENTS,VARIANTS
from prepare_pid import read
from extract_pid import load
from verify_pid import independent_scores
from publication_stats import save,sha
from analyze_run12_noise_state import combine,VIEWS


@functools.lru_cache(None)
def digest(path):return sha(Path(path))


def hashes(mapping,base=PROJECT):
    for rel,value in mapping.items():assert digest(base/rel)==value,rel
    return len(mapping)


def main(archive=False):
    if (OUT/'record.json').exists():
        assert not archive,'Archive exists; use read-only verification'
        r=read(OUT/'record.json')
        print('Verified frozen outputs:',hashes(r['outputs_sha256']),flush=True)
        print('Verified source snapshots:',hashes(r['source_sha256'],OUT/'source_snapshot'),flush=True)
        return
    protocol=read(OUT/'protocol.json');report=read(OUT/'assessment.json')
    assert digest(PARENT/'record.json')==protocol['parent_record_sha256']
    parent=read(PARENT/'record.json')
    checks={'scope':__doc__,'parent_archive':{'record_sha256':protocol['parent_record_sha256'],
        'unchanged_outputs':hashes(parent['outputs_sha256']),
        'unchanged_source_snapshots':hashes(parent['source_sha256'],PARENT/'source_snapshot')}}
    print('Parent archive unchanged.',flush=True)
    checks['frozen_noise_sources']=hashes(protocol['source_sha256']);checks['frozen_inputs']=hashes(protocol['inputs_sha256'])
    assert digest(WIDTH/'record.json')==parent['parent_record_sha256']
    width_record=read(WIDTH/'record.json');analysis_inputs={}
    for seg in SEGMENTS:
        path=WIDTH/'arrays'/f'measured_run12_{seg}.npz';rel=str(path.relative_to(PROJECT))
        assert digest(path)==width_record['outputs_sha256'][rel]
        analysis_inputs[rel]=digest(path)
    pub=PROJECT/'analysis/paper_baseline/results/publication_20261028';point=pub/'point.json'
    assert digest(pub/'record.json')=='69d1858dafa473c58d795cbfbb40deb1aca5ad6a9a20e1d5920eb67de0bf18e1'
    point_rel=str(point.relative_to(PROJECT))
    assert digest(point)==read(pub/'record.json')['outputs_sha256'][point_rel]
    analysis_inputs[point_rel]=digest(point)
    countpath=PROJECT/'simulation/results/pid_20261029/arrays/measured_run12_first60.npz'
    countrel=str(countpath.relative_to(PROJECT))
    assert digest(countpath)==read(PROJECT/'simulation/results/pid_20261029/record.json')['outputs_sha256'][countrel]
    analysis_inputs[countrel]=digest(countpath);checks['other_frozen_analysis_inputs']=analysis_inputs
    assert digest(PROJECT/protocol['binary'])==protocol['binary_sha256']
    for filename,source in [('protocol.json','prepare_run12_noise_state.py'),('execution.json','run_run12_noise_state.py'),
        ('assessment.json','analyze_run12_noise_state.py'),('plot_manifest.json','plot_run12_noise_state.py')]:
        assert read(OUT/filename)['code_sha256']==digest(PROJECT/'simulation'/source)
    assert report['protocol_sha256']==digest(OUT/'protocol.json')
    assert read(OUT/'plot_manifest.json')['assessment_sha256']==digest(OUT/'assessment.json')
    assert digest(PROJECT/protocol['parent_ROOT'])==protocol['parent_ROOT_sha256']
    assert digest(PROJECT/protocol['parent_config'])==protocol['parent_config_sha256']
    assert digest(OUT/'mother_folds.npz')==protocol['mother_fold_sha256']
    assignment=load(OUT/'mother_folds.npz');rng=np.random.default_rng(protocol['MC_mother_fold_seed'])
    uniform=rng.random(protocol['unique_parent_mothers']);checks['crossfit']={}
    for seg in SEGMENTS:
        a=load(DATA/f'run12_{seg}.npz');fold=(a['ids']//64)%2;counts=[]
        for ef in [0,1]:
            g=protocol['groups'][seg]['folds'][str(ef)];f=read(PROJECT/g['fit'])
            assert digest(PROJECT/g['fit'])==g['fit_sha256']
            assert f['code_sha256']==protocol['code_sha256']
            np.testing.assert_array_equal(f['calibration_ids'],a['ids'][fold!=ef])
            np.testing.assert_array_equal(f['evaluation_ids'],a['ids'][fold==ef])
            assert not set(f['calibration_ids'])&set(f['evaluation_ids'])
            assert sorted(f['calibration_ids']+f['evaluation_ids'])==a['ids'].tolist()
            for name in ['state_fit_original_spectrum','state_fit_new_spectrum']:
                fit=f[name];valid=[r for r in fit['attempts'] if r['valid']]
                assert fit['chosen']==min(valid,key=lambda r:r['fit']['BIC'])
                assert min(fit['chosen']['fit']['component_n'])>=20
            counts.append([len(f['calibration_ids']),len(f['evaluation_ids'])])
        np.testing.assert_array_equal(assignment[seg],(uniform>=np.mean(fold==0)).astype(np.int8))
        checks['crossfit'][seg]={'calibration_evaluation_sizes':counts,'disjoint':True,
            'one_response_assignment_per_mother':True}
    pca=read(PROJECT/'analysis/paper_baseline/results/publication_20261028/point.json')['windows']['3000']['pca']
    original=load(PARENT/'validation/arrays/run12_Am241.npz');base=read(PROJECT/protocol['parent_config'])
    arrays={};checks['native_replays']={}
    for key,e in protocol['entries'].items():
        assert digest(PROJECT/e['config'])==e['config_sha256']
        cfg=read(PROJECT/e['config'])
        if key!='baseline':
            # Only state noise and its documented scalar summary may change.
            normalized=copy.deepcopy(cfg)
            normalized.pop('noise_state_study');normalized['status']=base['status']
            normalized['electronics_noise']=copy.deepcopy(base['electronics_noise'])
            for field in ['baseline_mV','noise_rms_mV']:
                normalized['sipm']['readout'][field]=base['sipm']['readout'][field]
            assert normalized==base,key
        h=np.asarray(cfg['electronics_noise']['coefficients']);assert len(h)==257
        np.testing.assert_allclose(h@h,1,atol=1e-12,rtol=0)
        meta=read(OUT/'extraction'/f'{key}.json')
        assert meta['root_sha256']==digest(PROJECT/e['path'])
        assert meta['config_sha256']==e['config_sha256']
        assert meta['code_sha256']==digest(PROJECT/'simulation/extract_run12_noise_state.py')
        assert meta['protocol_sha256']==digest(OUT/'protocol.json')
        a=load(OUT/'arrays'/f'{key}.npz');arrays[key]=a
        np.testing.assert_array_equal(a['all_energy'],original['all_energy'])
        np.testing.assert_array_equal(a['ids'][a['fixed']],original['ids'])
        np.testing.assert_array_equal(a['ids'][a['common']],np.flatnonzero(a['all_common']))
        np.testing.assert_allclose(independent_scores(a['waves'],pca),a['scores'],rtol=1e-10,atol=1e-10)
        root=ROOT.TFile.Open(str(PROJECT/e['path']));info=json.loads(str(root.Get('RunInfo').GetString().Data()))
        assert info['completed'] and info['config']==cfg and info['seed']==e['seed']
        tree=root.Get('Events');assert tree.GetEntries()==e['events']==24000
        if key!='baseline':
            assert info['response_replay']['equal_original_avalanche_events']==24000
            assert info['response_replay']['max_relative_charge_balance']<1e-9
        tree.SetBranchStatus('*',0);tree.SetBranchStatus('signal_mV',1);tree.SetBranchStatus('adc',1)
        ro=cfg['sipm']['readout'];lsb=(ro['adc_max_mV']-ro['adc_min_mV'])/2**ro['adc_bits']
        for i,event_id in enumerate(a['ids']):
            tree.GetEntry(int(event_id));np.testing.assert_array_equal(np.asarray(tree.adc),a['waves'][i])
            if a['fixed'][i]:
                j=int(np.searchsorted(original['ids'],event_id))
                np.testing.assert_array_equal(np.asarray(tree.signal_mV)/lsb,original['signal'][j])
        root.Close()
        checks['native_replays'][key]={'mothers':24000,'same_deposited_energy':True,
            'independent_ADC_score_check':len(a['ids']),'bit_exact_signal_parent_ids':len(original['ids']),
            'same_avalanche_metadata':True,'selected_before_fold_assignment':int(a['common'].sum())}
        print('Native ADC, scores and preserved signal:',key,flush=True)
    combined=load(OUT/'combined_scores.npz');boot=load(OUT/'bootstrap.npz')
    original_result=read(PARENT/'validation/assessment.json');checks['paired_analysis']={}
    minimum=512
    for seg in SEGMENTS:
        for v in VARIANTS:
            for view in VIEWS:
                key=f'{seg}__{v}__{view}';a=combine(seg,v,arrays,assignment,view)
                for k in ['ids','q','scores','anchor']:np.testing.assert_array_equal(combined[key+'__'+k],a[k])
                assert len(np.unique(a['ids']))==len(a['ids'])
                row=report['checks'][seg]['variants'][v][view]
                assert row['n_MC']==len(a['ids'])
                if v=='baseline':
                    np.testing.assert_allclose(row['IQR_ratio'],original_result['checks'][f'run12_{seg}']['IQR_ratio'],rtol=1e-10,atol=1e-10)
                    np.testing.assert_array_equal(row['paired_delta_ratio'],np.zeros(3))
                for b in ['64','128']:
                    for field in ['IQR_ratio_CI95','paired_delta_ratio_CI95']:
                        for ci in row['uncertainty_by_data_block'][b][field]:
                            minimum=min(minimum,ci['valid_draws']);assert ci['valid_draws']>=480 and ci['estimate95'] is not None
                assert boot[key].shape==(512,6,3,3)
        checks['paired_analysis'][seg]={'fixed_common_ids':len(original['ids']),
            'old_baseline_reproduced':True,'duplicate_mothers':0}
    assert report['new_native_mothers']==0 and report['unique_parent_mothers']==24000
    checks['minimum_valid_bootstrap_draws']=minimum
    nbpath=PROJECT/'jupyter/codex_0001_csi_run12_noise_state.ipynb'
    nb=nbformat.read(nbpath,as_version=4);nbformat.validate(nb);images=0;nc=0
    for c in nb.cells:
        if c.cell_type!='code':continue
        nc+=1;assert c.execution_count is not None and all(o.output_type!='error' for o in c.outputs)
        for o in c.outputs:
            if 'data' in o and 'image/png' in o.data:
                assert base64.b64decode(o.data['image/png']).startswith(b'\x89PNG');images+=1
    assert images==4
    for p in (OUT/'figures').glob('*'):
        if p.suffix=='.png':assert p.read_bytes().startswith(b'\x89PNG')
        if p.suffix=='.pdf':assert p.read_bytes().startswith(b'%PDF-')
        if p.suffix=='.svg':assert '<svg' in p.read_text()[:3000]
    checks['notebook']={'executed_code_cells':nc,'embedded_images':images,'valid':True}
    prior=read(PARENT/'verification.json')['tests'];assert prior['passed']
    assert '100% tests passed' in (PARENT/'logs/ctest.log').read_text()
    checks['native_tests']={'reused_unchanged_binary_sha256':protocol['binary_sha256'],
        'parent_ctest_targets':prior['ctest_targets'],'parent_replay_integration_tests':prior['response_replay_integration_tests'],
        'rerun_this_phase':False,'reason':'No native source or binary change; new configs exercised by18 complete native replays and independent checks above.'}
    checks['versions']={'Python':platform.python_version(),'NumPy':np.__version__,'SciPy':scipy.__version__,'ROOT':ROOT.gROOT.GetVersion()}
    checks['remaining_limits']='Fits, kernels, fold assignment and signal parameters held fixed in intervals; later pretrigger conditioned; all data previously explored. See decision.json.'
    save(OUT/'verification.json',checks)
    if archive:
        sources=set((PROJECT/'simulation').glob('*.py'))
        for folder in ['include','src','vendor','tests']:
            sources.update(p for p in (PROJECT/'simulation'/folder).rglob('*') if p.suffix in ['.py','.cc','.hh','.json'])
        sources.update([PROJECT/'simulation/main.cc',PROJECT/'simulation/replay_response.cc',PROJECT/'simulation/CMakeLists.txt',
            PROJECT/'simulation/README.md',PROJECT/'jupyter/codex_0001_build_run12_noise_state_notebook.py'])
        sources.update((PROJECT/'analysis/paper_baseline').glob('*.py'))
        snapshot=OUT/'source_snapshot';source_hashes={}
        for p in sorted(sources):
            rel=p.relative_to(PROJECT);target=snapshot/rel;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(p,target);source_hashes[str(rel)]=sha(target)
        outputs={str(p.relative_to(PROJECT)):digest(p) for p in sorted(OUT.rglob('*')) if p.is_file() and 'source_snapshot' not in p.parts}
        outputs[str(nbpath.relative_to(PROJECT))]=digest(nbpath)
        save(OUT/'record.json',{'scope':__doc__,'outputs_sha256':outputs,'source_sha256':source_hashes,
            'checkpoint_commit':protocol['checkpoint_commit'],'parent_record_sha256':protocol['parent_record_sha256'],
            'verification_sha256':digest(OUT/'verification.json'),
            'result':'Conditional pretrigger noise test; no new independent MC mothers; full bandwidth closure not established.'})
        print('Archived',len(outputs),'outputs and',len(source_hashes),'source snapshots.',flush=True)
    print('All run12 cross-fit verification checks passed.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive',action='store_true');args=p.parse_args();main(args.archive)
