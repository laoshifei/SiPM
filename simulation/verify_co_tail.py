"""Verify native two-source tail checks and freeze an auditable experiment."""
import argparse,base64,copy,functools,json,platform,shutil
from pathlib import Path
import numpy as np
import scipy
import ROOT
import nbformat
from fit_co_tail import OUT,PARENT,PROJECT,STATE,DATA,WIDTH,SOURCES,PRIMARY
from prepare_pid import read
from extract_pid import load,label_truth
from verify_pid import independent_scores
from publication_stats import sha,save
from analyze_bandwidth_validation import widths,rms_width,interval


@functools.lru_cache(None)
def digest(path):return sha(Path(path))


def hashes(mapping,base=PROJECT):
    for rel,value in mapping.items():assert digest(base/rel)==value,rel
    return len(mapping)


def independent_moments(waves):
    # Direct sample sums, independent of the proposal's density/moment helpers.
    from reference import wave_features
    f=wave_features(waves);rows=[]
    time=np.arange(-20,1500)*2.
    for i,w in enumerate(waves):
        anchor=int(np.floor(f['t10_sample'][i]))+1
        gate=w[anchor-20:anchor+1500].astype(float)-f['baseline'][i]
        q=gate.sum();late=time>=2000
        rows.append([np.dot(time,gate)/q,gate[late].sum()/q,np.dot(time[late],gate[late])/q])
    return np.asarray(rows)


def equal_bins(values,q,edges,support):
    bins=np.clip(np.searchsorted(edges,q,'right')-1,0,5)
    return np.array([values[bins==b].mean(0) for b in support]).mean(0)


def main(archive=False):
    if (OUT/'record.json').exists():
        assert not archive,'Archive already exists'
        record=read(OUT/'record.json')
        print('Frozen outputs:',hashes(record['outputs_sha256']),flush=True)
        print('Frozen source snapshots:',hashes(record['source_sha256'],OUT/'source_snapshot'),flush=True)
        return
    design=read(OUT/'design.json');fit=read(OUT/'selection.json');protocol=read(OUT/'native_protocol.json')
    report=read(OUT/'assessment.json');checks={'scope':__doc__,'parent_archives':{}}
    for folder,expected in [(PARENT,design['parent_record_sha256']),(STATE,design['noise_state_record_sha256'])]:
        assert digest(folder/'record.json')==expected
        old=read(folder/'record.json')
        checks['parent_archives'][folder.name]={'record_sha256':expected,'unchanged_outputs':hashes(old['outputs_sha256']),
            'unchanged_sources':hashes(old['source_sha256'],folder/'source_snapshot')}
        print('Preserved parent archive:',folder.name,flush=True)
    checks['training_inputs']=hashes(design['inputs_sha256']);checks['analysis_inputs']=hashes(report['inputs_sha256'])
    assert fit['design_sha256']==protocol['design_sha256']==digest(OUT/'design.json')
    assert protocol['selection_sha256']==report['selection_sha256']==digest(OUT/'selection.json')
    assert report['protocol_sha256']==digest(OUT/'native_protocol.json')
    assert fit['code_sha256']==design['code_sha256']==digest(PROJECT/'simulation/fit_co_tail.py')
    for file,script in [('execution.json','run_co_tail.py'),('assessment.json','analyze_co_tail.py'),('figure_record.json','plot_co_tail.py')]:
        assert read(OUT/file)['code_sha256']==digest(PROJECT/'simulation'/script)
    assert read(OUT/'figure_record.json')['assessment_sha256']==digest(OUT/'assessment.json')
    assert read(OUT/'figure_record.json')['selection_sha256']==digest(OUT/'selection.json')
    assert fit['selected_case']==design['predeclared_primary']==PRIMARY
    assert fit['chosen_parameters']==fit['cases'][PRIMARY]['best']['parameters']
    for row in fit['cases'].values():
        assert row['best']==min([a for a in row['attempts'] if a['success']],key=lambda r:r['objective'])
        assert len(row['attempts'])==3
    checks['fit_families']={'retained':len(fit['cases']),'starts_per_family':3,'primary_predeclared':PRIMARY}
    assert digest(PROJECT/protocol['binary'])==protocol['binary_sha256']
    parent_native=read(PARENT/'validation/protocol.json')
    assert parent_native['binary_sha256']['csi_sources']==protocol['binary_sha256']
    native_sources={k:v for k,v in read(PARENT/'protocol.json')['source_sha256'].items()
        if Path(k).suffix in ['.cc','.hh','.h'] or Path(k).name=='CMakeLists.txt'}
    checks['unchanged_native_sources']=hashes(native_sources)
    checks['proposal_sources']=hashes(protocol['source_sha256'])
    pca=read(PROJECT/'analysis/paper_baseline/results/publication_20261028/point.json')['windows']['3000']['pca']
    pub=PROJECT/'analysis/paper_baseline/results/publication_20261028'
    assert digest(pub/'record.json')=='69d1858dafa473c58d795cbfbb40deb1aca5ad6a9a20e1d5920eb67de0bf18e1'
    assert digest(pub/'point.json')==read(pub/'record.json')['outputs_sha256'][str((pub/'point.json').relative_to(PROJECT))]
    arrays={};checks['native']={};total=0
    for key,e in protocol['entries'].items():
        cfg=read(PROJECT/e['config']);oldcfg=read(PROJECT/parent_native['entries'][key]['config'])
        normalized=copy.deepcopy(cfg);normalized.pop('tail_refit');normalized['status']=oldcfg['status']
        for k in ['electron_tau_ns','alpha_tau_ns','electron_fast_fraction','alpha_fast_fraction','yield_photons_per_MeV','alpha_light_yield_ratio']:
            normalized['optical'][k]=oldcfg['optical'][k]
        normalized['sipm']['readout']['lowpass_ns']=oldcfg['sipm']['readout']['lowpass_ns']
        assert normalized==oldcfg,key
        for k in ['electron_tau_ns','alpha_tau_ns','electron_fast_fraction','alpha_fast_fraction']:
            assert cfg['optical'][k]==fit['chosen_parameters'][k]
        factors=fit['analytic_gate_area_multipliers']
        assert cfg['optical']['yield_photons_per_MeV']==oldcfg['optical']['yield_photons_per_MeV']*factors['electron']
        assert cfg['optical']['alpha_light_yield_ratio']==oldcfg['optical']['alpha_light_yield_ratio']*(factors['alpha']/factors['electron'])
        assert cfg['sipm']['readout']['lowpass_ns']==fit['chosen_parameters']['lowpass_ns']
        assert cfg['sipm']['sensor']['constant_pde']==.3 and cfg['readout_response']['signal_transmission']==.5
        assert digest(PROJECT/e['config'])==e['config_sha256']
        meta=read(OUT/'extraction'/f'{key}.json');assert meta['root_sha256']==digest(PROJECT/e['path'])
        assert meta['code_sha256']==digest(PROJECT/'simulation/extract_npe_profile.py')
        native_check=read((PROJECT/e['path']).with_name(Path(e['path']).stem+'_validation.json'))
        assert native_check['passed'] and native_check['ancestry_checks'] and native_check['events']==e['events']
        assert native_check['max_relative_charge_balance_error']<1e-9
        root=ROOT.TFile.Open(str(PROJECT/e['path']));info=json.loads(str(root.Get('RunInfo').GetString().Data()))
        assert info['completed'] and info['config']==cfg and info['seed']==e['seed']
        assert 'response_replay' not in info and e['seed']!=parent_native['entries'][key]['seed']
        tree=root.Get('Events');assert tree.GetEntries()==e['events'];total+=e['events']
        a=load(OUT/'arrays'/f'{key}.npz');arrays[(e['run'],'new')]=a
        assert int(a['n_parent'])==e['events'] and len(a['ids'])==meta['n_common']
        np.testing.assert_array_equal(a['ids'],np.flatnonzero(a['all_common']))
        np.testing.assert_array_equal(a['energy'],a['all_energy'][a['ids']])
        np.testing.assert_array_equal(label_truth(a['energy'])[0],a['truth'])
        np.testing.assert_allclose(independent_scores(a['waves'],pca),a['scores'],rtol=1e-10,atol=1e-10)
        tree.SetBranchStatus('*',0);tree.SetBranchStatus('adc',1)
        for i,event_id in enumerate(a['ids']):
            tree.GetEntry(int(event_id));np.testing.assert_array_equal(np.asarray(tree.adc),a['waves'][i])
        root.Close()
        checks['native'][key]={'independent_mothers':e['events'],'common_events':len(a['ids']),
            'physics_and_charge_checks_all_mothers':True,'independent_ADC_scores':len(a['ids']),'config_changes_limited_to_frozen_timing_and_normalization':True}
        print('Verified new native source:',key,flush=True)
    assert total==protocol['independent_new_mothers']==report['new_independent_mothers']==32000
    checks['independent_mothers']=total
    boot=load(OUT/'bootstrap.npz');edges=np.asarray(protocol['q_edges']);minimum=512;aggregate_checks=0
    for source,run,kind in SOURCES:
        key=f'run{run}_{source}';support=protocol['support_bins'][source]
        arrays[(run,'old')]=load(PARENT/'validation/arrays'/f'{key}.npz')
        moment={name:equal_bins(independent_moments(a['waves']),a['q'],edges,support)
            for (rr,name),a in arrays.items() if rr==run}
        for seg in ['first60','middle20','last20']:
            group=f'run{run}_{seg}';data=load(DATA/f'{group}.npz');d=load(WIDTH/'arrays'/f'measured_{group}.npz');u=d['main_band']
            dm=equal_bins(independent_moments(data['waves'][u]),d['q'][u],edges,support)
            row=report['checks'][group];np.testing.assert_allclose(dm,row['data_tail_metrics'],rtol=1e-12,atol=1e-9)
            for name in ['old','new']:
                item=row['models'][name]
                np.testing.assert_allclose(moment[name]-dm,item['tail_difference'],rtol=1e-10,atol=1e-9)
                np.testing.assert_allclose(sum(item['mean_time_slice_difference_ns']),item['tail_difference'][0],atol=1e-9)
                am=boot[f'{key}__{name}__moments'];aq=boot[f'{key}__{name}__quartiles']
                assert am.shape==(512,6) and aq.shape==(512,6,3,3)
                for size in ['64','128']:
                    bm=boot[f'{group}__block{size}__moments'];bq=boot[f'{group}__block{size}__quartiles']
                    values={'tail_difference_CI95':am[:,:3]-bm[:,:3],'F_ratio_CI95':am[:,3:]/bm[:,3:],
                        'IQR_ratio_CI95':rms_width(widths(aq),support)/rms_width(widths(bq),support)}
                    for field,bs in values.items():
                        for j in range(3):
                            saved=item['uncertainty_by_data_block'][size][field][j]
                            assert interval(bs[:,j])==saved;minimum=min(minimum,saved['valid_draws']);assert saved['valid_draws']>=480
                aggregate_checks+=1
    checks['independent_formal_moment_aggregates']=aggregate_checks;checks['minimum_valid_bootstrap_draws']=minimum
    nbpath=PROJECT/'jupyter/codex_0001_csi_co_tail_refit.ipynb';nb=nbformat.read(nbpath,as_version=4);nbformat.validate(nb)
    code=[c for c in nb.cells if c.cell_type=='code'];images=[]
    for c in code:
        assert c.execution_count is not None and all(o.output_type!='error' for o in c.outputs)
        images.extend(base64.b64decode(o.data['image/png']) for o in c.outputs if 'data' in o and 'image/png' in o.data)
    figures=read(OUT/'figure_record.json')['figures'];assert len(images)==len(figures)==5
    for image,name in zip(images,figures):
        assert image==(OUT/'figures'/f'{name}.png').read_bytes()
        assert (OUT/'figures'/f'{name}.pdf').read_bytes().startswith(b'%PDF-')
        assert '<svg' in (OUT/'figures'/f'{name}.svg').read_text()[:3000]
    checks['notebook']={'executed_code_cells':len(code),'embedded_images':5,'byte_identical_PNGs':5,'valid':True}
    prior=read(PARENT/'verification.json')['tests'];assert prior['passed']
    checks['native_tests']={'reused_exact_binary_sha256':protocol['binary_sha256'],'parent_passed':prior,
        'new_all_mother_physics_checks':32000,'rerun_unit_tests':False,'reason':'No native source change; new optical/readout configurations verified with complete native transport.'}
    checks['versions']={'Python':platform.python_version(),'NumPy':np.__version__,'SciPy':scipy.__version__,'ROOT':ROOT.gROOT.GetVersion()}
    checks['scientific_limits']='Conditional model and pointwise statistics; old MC used in proposal; alpha slow component at15us bound and effective alpha normalization above1; see decision.json for remaining width and center residuals.'
    save(OUT/'verification.json',checks)
    if archive:
        sources=set((PROJECT/'simulation').glob('*.py'))
        for folder in ['include','src','vendor','tests']:
            sources.update(p for p in (PROJECT/'simulation'/folder).rglob('*') if p.suffix in ['.py','.cc','.hh','.json'])
        sources.update([PROJECT/'simulation/main.cc',PROJECT/'simulation/replay_response.cc',PROJECT/'simulation/CMakeLists.txt',
            PROJECT/'simulation/README.md',PROJECT/'jupyter/codex_0001_build_co_tail_notebook.py'])
        sources.update((PROJECT/'analysis/paper_baseline').glob('*.py'))
        snapshot=OUT/'source_snapshot';source_hashes={}
        for p in sorted(sources):
            rel=p.relative_to(PROJECT);target=snapshot/rel;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(p,target);source_hashes[str(rel)]=sha(target)
        outputs={str(p.relative_to(PROJECT)):digest(p) for p in sorted(OUT.rglob('*')) if p.is_file() and 'source_snapshot' not in p.parts}
        outputs[str(nbpath.relative_to(PROJECT))]=digest(nbpath)
        save(OUT/'record.json',{'scope':__doc__,'outputs_sha256':outputs,'source_sha256':source_hashes,
            'checkpoint_commit':design['checkpoint_commit'],'parent_record_sha256':design['parent_record_sha256'],
            'noise_state_record_sha256':design['noise_state_record_sha256'],'verification_sha256':digest(OUT/'verification.json'),
            'result':'Tail and center correction has native support but does not close all PID widths; no default promotion or unique physical calibration.'})
        print('Archived',len(outputs),'outputs and',len(source_hashes),'source snapshots.',flush=True)
    print('All co-tail verification checks passed.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive',action='store_true');args=p.parse_args();main(args.archive)
