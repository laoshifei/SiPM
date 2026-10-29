"""Verify and archive the 2026-10-29 refit without changing earlier results."""
import argparse
import base64
import hashlib
import json
import platform
import shutil
from pathlib import Path
import numpy as np
import ROOT
from diagnose_charge import sha
from reference import PROJECT
from screen_response_candidates import info_of
from validate_signal_fluctuations import arrays, error

BASE=PROJECT/'simulation/results/waveform_refit_20261029'
OLD=PROJECT/'simulation/results/signal_fluctuations_20261028'


def read(path):
    return json.loads(path.read_text())


def hashes(mapping,base=PROJECT):
    for relative,digest in mapping.items():
        assert sha(base/relative)==digest, relative
    return len(mapping)


def statistical_checks(folder,source,row,data_root):
    model=arrays(folder/f'{source}_native.npz')
    assert len(model['ids'])==row['common_n']
    np.testing.assert_array_equal(model['metrics'][:,:,3].mean(0),row['F'])
    for group,c in row['comparisons'].items():
        data=arrays(data_root/f'{group}.npz')
        assert len(data['ids'])==c['n']
        np.testing.assert_array_equal(data['metrics'][:,:,3].mean(0),c['measured_F'])
        np.testing.assert_array_equal(np.array(row['F'])/c['measured_F']-1,c['relative'])
        assert error(model['shapes'].mean(0),data['shapes'].mean(0))==c['shape_nrmse']
        draws=np.load(folder/f'{source}_{group}_relative_draws.npy')
        assert draws.shape==(512,3) and np.isfinite(draws).all()
        np.testing.assert_array_equal(np.quantile(draws,[.025,.975],axis=0),c['relative_interval95'])
    return len(row['comparisons'])


def physical(config):
    # Baseline level/RMS and empirical readout noise are allowed to vary.
    ro={k:v for k,v in config['sipm']['readout'].items() if k not in ('baseline_mV','noise_rms_mV')}
    return [config['geometry'],config['optical'],config['sipm']['sensor'],ro,
            config['readout_response'],config['acquisition']]


def main(verify=False):
    if verify:
        record=read(BASE/'record.json')
        print('Archived outputs:',hashes(record['outputs_sha256']),flush=True)
        print('Archived sources:',hashes(record['source_sha256'],BASE/'source_snapshot'),flush=True)
        return
    assert not (BASE/'record.json').exists(), 'Frozen record already exists; use --verify'
    checks={'parent_archives':{}}
    expected={
      'publication_20261028':'69d1858dafa473c58d795cbfbb40deb1aca5ad6a9a20e1d5920eb67de0bf18e1',
      'robustness_20261028':'44c4b4a7af6d59c5379aaea826e9d02b2097534caec237c7f0b29dfc3c632cc3'}
    for folder in [PROJECT/'analysis/paper_baseline/results'/k for k in expected]+[OLD]:
        record=read(folder/'record.json');digest=sha(folder/'record.json')
        if folder.name in expected:assert digest==expected[folder.name]
        checks['parent_archives'][folder.name]={'record_sha256':digest,
            'outputs_unchanged':hashes(record['outputs_sha256']),
            'source_snapshots_unchanged':hashes(record['source_sha256'],folder/'source_snapshot')}
        print('Verified earlier archive:',folder.name,flush=True)

    checks['training_inputs']={}
    scripts={'shape_fit.json':'refit_waveform_shape.py',
      'parallel_shape/shape_fit.json':'refit_parallel_shape.py',
      'baseline_states.json':'refit_readout_baseline.py',
      'baseline_mixture_fit.json':'refit_baseline_mixture.py',
      'calibrated/fit.json':'refit_signal_residual.py',
      'final_parameters/fit.json':'refit_final_parameters.py','run18/fit.json':'refit_run18_state.py'}
    for relative,script in scripts.items():
        r=read(BASE/relative)
        assert r['code_sha256']==sha(PROJECT/'simulation'/script),script
        checks['training_inputs'][relative]=hashes(r.get('inputs_sha256',r.get('input_sha256',{})))
    checks['proposal_source_snapshot_files']=hashes(read(BASE/'proposal_source_snapshot/record.json'),BASE/'proposal_source_snapshot')
    proposal=read(BASE/'parallel_shape/native/native_run.json')
    assert proposal['shape_fit_sha256']==sha(BASE/'parallel_shape/shape_fit.json')
    assert proposal['baseline_sha256']==sha(BASE/'baseline_states.json')
    proposal_validation=read(BASE/'parallel_shape/analysis/validation.json')
    for source,name in [('Co60','co60'),('Am241','am241')]:
        frozen=proposal['sources'][source];r=proposal_validation['sources'][source]
        info=info_of(BASE/'parallel_shape/native'/name/'events.root')
        assert info['completed'] and 'response_replay' not in info
        assert info['seed']==frozen['seed'] and info['events_requested']==frozen['events']
        assert info['config']==read(PROJECT/frozen['config'])
        assert sha(PROJECT/frozen['config'])==frozen['config_sha256']
        assert r['charge_balance']['passed']
        assert r['root_sha256']==sha(PROJECT/r['ROOT_path'])
        statistical_checks(BASE/'parallel_shape/analysis',source,r,OLD)
    checks['proposal_native']={'mothers':28000,'binary_sha256':proposal['binary_sha256'],
        'note':'Proposal ran before final baseline-mixture and zero-signal changes; its own source snapshot is preserved. Binary digest is recorded; old binary was not retained.'}
    fresh=read(BASE/'independent/analysis/validation.json')
    assert fresh['code_sha256']==sha(PROJECT/'simulation/refit_native_analysis.py')
    frozen=read(BASE/'independent/native_run.json')
    assert frozen['binary_sha256']==sha(PROJECT/'simulation/build/csi_sources')
    assert frozen['fit_sha256']==sha(BASE/'final_parameters/fit.json')
    assert frozen['baseline_mixture_sha256']==sha(BASE/'baseline_mixture_fit.json')
    checks['fresh_full_transport']={}
    for source,name in [('Co60','co60'),('Am241','am241')]:
        r=fresh['sources'][source];s=frozen['sources'][source]
        info=info_of(PROJECT/r['ROOT_path'])
        assert info['completed'] and info['schema_version']==9 and 'response_replay' not in info
        assert info['seed']==r['seed']==s['seed']
        assert info['events_requested']==r['nparent']==s['events']
        assert info['config']==r['config']==read(PROJECT/s['config'])
        assert sha(PROJECT/s['config'])==s['config_sha256']==sha(BASE/f'final_parameters/{source}_calibrated.json')
        assert sha(PROJECT/r['ROOT_path'])==r['root_sha256']
        assert r['charge_balance']['passed'] and r['charge_balance']['events']==r['nparent']
        statistical_checks(BASE/'independent/analysis',source,r,OLD)
        checks['fresh_full_transport'][source]={'mothers':r['nparent'],'common_n':r['common_n'],'seed':r['seed'],
            'config_sha256':s['config_sha256'],'ROOT_sha256':r['root_sha256']}
    checks['replays']={}
    for stage in ['calibrated_mixture','conditioned_last20']:
        result=read(BASE/stage/'analysis/validation.json')
        assert result['code_sha256']==sha(PROJECT/'simulation/refit_native_analysis.py')
        for source,name in [('Co60','co60'),('Am241','am241')]:
            r=result['sources'][source];info=info_of(PROJECT/r['ROOT_path'])
            assert info['completed'] and r['charge_balance']['passed']
            assert info['response_replay']['equal_original_avalanche_events']==r['nparent']
            assert r['root_sha256']==sha(PROJECT/r['ROOT_path'])
            assert info['config']==r['config']
            statistical_checks(BASE/stage/'analysis',source,r,OLD)
            if stage=='conditioned_last20':
                manifest=read(BASE/stage/'native_run.json')
                assert manifest['binary_sha256']==sha(PROJECT/'simulation/build/replay_response')
                assert sha(PROJECT/manifest['parents'][source]['path'])==manifest['parents'][source]['sha256']
                assert sha(BASE/stage/f'{source}_calibrated.json')==manifest['config_sha256'][source]
                assert physical(r['config'])==physical(fresh['sources'][source]['config'])
                assert r['config']['signal_dependent_noise']==fresh['sources'][source]['config']['signal_dependent_noise']
            checks['replays'][stage+'/'+source]={'mothers_reused':r['nparent'],'new_mothers':0,'common_n':r['common_n']}

    r18=read(BASE/'run18/validation.json');manifest=read(BASE/'run18/native_run.json')
    assert r18['code_sha256']==sha(PROJECT/'simulation/validate_run18_state.py')
    assert manifest['fit_sha256']==sha(BASE/'run18/fit.json')==r18['fit_sha256']
    assert r18['parent_validation_sha256']==sha(BASE/'independent/analysis/validation.json')
    assert manifest['binary_sha256']==sha(PROJECT/'simulation/build/replay_response')
    assert manifest['parent']['sha256']==fresh['sources']['Am241']['root_sha256']
    pool=arrays(PROJECT/'analysis/paper_baseline/results/publication_20261028/pool_run18.npz')
    # Split arrays and raw BIN reselection were checked when the branch was prepared.
    assert read(BASE/'run18/fit.json')['training'].startswith('Only existing frozen3000 run18 first60')
    for variant,r in r18['variants'].items():
        folder=BASE/'run18'/variant;info=info_of(folder/'events.root')
        assert info['completed'] and info['seed']==fresh['sources']['Am241']['seed']
        assert info['events_requested']==24000==info['response_replay']['equal_original_avalanche_events']
        assert r['root_sha256']==sha(folder/'events.root')
        config=BASE/f'run18/Am241_run18_{variant}.json'
        assert info['config']==read(config) and manifest['config_sha256'][variant]==sha(config)
        assert physical(info['config'])==physical(fresh['sources']['Am241']['config'])
        statistical_checks(folder,'Am241',r,BASE/'run18')
        checks['replays']['run18/'+variant]={'mothers_reused':24000,'new_mothers':0,'common_n':r['common_n']}
    # Check frozen run18 membership directly, including the segment labels.
    for split,group in enumerate(['first60','middle20','last20']):
        data=arrays(BASE/f'run18/run18_{group}.npz')
        np.testing.assert_array_equal(data['ids'],pool['ids'][pool['primary']&(pool['split']==split)])
    checks['run18_frozen_counts']=[3000,3000,3000]

    for stage in ['independent','conditioned_last20']:
        folder=BASE/stage/'analysis';r=read(folder/'feature_intervals.json')
        assert r['code_sha256']==sha(PROJECT/'simulation/refit_feature_intervals.py')
        assert r['validation_sha256']==sha(folder/'validation.json')
        for source,groups in r['sources'].items():
            for group,values in groups.items():
                draws=arrays(folder/f'{source}_{group}_feature_draws.npz')
                for key,c in values.items():
                    assert len(draws[key])==512
                    np.testing.assert_array_equal(np.quantile(draws[key],[.025,.975]),c['difference_interval95'])
    checks['conditional_intervals']='All saved 512-draw F intervals and feature-median intervals reconstructed; fitted parameters held fixed.'

    summary=read(BASE/'final_summary.json');hashes(summary['inputs_sha256'])
    assert summary['code_sha256']==sha(PROJECT/'simulation/summarize_waveform_refit.py')
    for preset in summary['presets'].values():
        assert sha(PROJECT/preset['path'])==preset['sha256']==sha(PROJECT/preset['source'])
    figure=read(BASE/'figures/figure_record.json')
    assert figure['code_sha256']==sha(PROJECT/'simulation/plot_waveform_refit.py')
    for filename in figure['figures']:
        path=PROJECT/filename
        assert path.read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
        assert path.with_suffix('.pdf').read_bytes().startswith(b'%PDF')
        assert '<svg' in path.with_suffix('.svg').read_text()
    for e in figure['examples']:
        root=BASE/'run18' if e['measured_group'].startswith('run18') else OLD
        data=arrays(root/(e['measured_group']+'.npz'))
        model=arrays(PROJECT/e['model_analysis']/(e['source']+'_native.npz'))
        i=np.argsort(data['q'])[int(e['rank']*(len(data['q'])-1))]
        j=int(np.argmin(abs(model['q']-data['q'][i])))
        assert int(data['ids'][i])==e['measured_id'] and int(model['ids'][j])==e['MC_id']
        assert float(model['q'][j]/data['q'][i]-1)==e['Q_relative_difference']
    checks['figures']={'PNG_groups_visually_reviewed':5,'PDF_SVG_headers_checked':True,
        'PDF_pages_not_separately_rendered':True,'fixed_Q_rank_examples_checked':len(figure['examples'])}
    notebook=PROJECT/'jupyter/codex_0001_csi_geant4_waveform_calibration.ipynb'
    nb=read(notebook);code=[c for c in nb['cells'] if c['cell_type']=='code']
    assert [c['execution_count'] for c in code]==list(range(1,len(code)+1))
    assert not any(o['output_type']=='error' for c in code for o in c['outputs'])
    images=[o['data']['image/png'] for c in code for o in c['outputs'] if 'image/png' in o.get('data',{})]
    assert len(images)==5
    for png,relative in zip(images,figure['figures']):
        assert hashlib.sha256(base64.b64decode(png)).hexdigest()==sha(PROJECT/relative)
    assert nb['metadata']['kernelspec']['name']=='python3'
    checks['notebook']={'path':str(notebook.relative_to(PROJECT)),'executed_cells':len(code),
        'errors':0,'inline_PNG_byte_identity':5,'schema_validated_by_builder':True}
    logs=BASE/'logs';logs.mkdir(exist_ok=True)
    for path in sorted(Path('/tmp').glob('sipm_refit*.log')):
        if 'archive' not in path.name:shutil.copy2(path,logs/path.name)
    assert '100% tests passed, 0 tests failed out of 5' in (logs/'sipm_refit_frozen_ctest.log').read_text()
    tests=(logs/'sipm_refit_frozen_replay_tests.log').read_text()
    assert 'Ran 2 tests' in tests and tests.rstrip().endswith('OK')
    checks['tests']={'ctest_targets':5,'integration_tests':2,'passed':True}

    sources=list((PROJECT/'simulation').glob('*.py'))+list((PROJECT/'simulation').glob('*.cc'))
    for sub in ['include','src','tests','vendor']:
        sources += [p for p in (PROJECT/'simulation'/sub).rglob('*') if p.is_file() and p.suffix in ['.py','.cc','.hh','.h','.json']]
    sources += [PROJECT/p for p in ['README.md','simulation/README.md','simulation/CMakeLists.txt',
        'analysis/paper_baseline/prepare.py','analysis/paper_baseline/publication_stats.py',
        'analysis/paper_baseline/publication_mc_uncertainty.py','jupyter/codex_0001_build_notebook.py']]
    sources += [notebook]+[PROJECT/p['path'] for p in summary['presets'].values()]
    source_hashes={}
    for path in sorted(set(sources)):
        rel=path.relative_to(PROJECT);dest=BASE/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,dest);source_hashes[str(rel)]=sha(path)
        assert sha(dest)==source_hashes[str(rel)]
    outputs={str(p.relative_to(PROJECT)):sha(p) for p in sorted(BASE.rglob('*')) if p.is_file()
        and p!=BASE/'record.json' and 'source_snapshot' not in p.relative_to(BASE).parts}
    record={'date':'2026-10-29','scope':__doc__,'checks':checks,
        'environment':{'python':platform.python_version(),'numpy':np.__version__,'ROOT':ROOT.gROOT.GetVersion(),
            'Geant4':'11.4.2','notebook_python':'/home/xc/software/jupyter/.venv/bin/python'},
        'MC_accounting':summary['MC_accounting'],
        'raw_provenance':'Previously archived raw BIN hashes retained; run18 waveforms were reread from BIN at frozen IDs. No source BIN was changed.',
        'limits':['Conditional empirical readout calibration; not a unique physical mechanism.',
            'Later data previously explored; no blind experimental validation.',
            'Run18 separately trained; failed run12 transfer retained.',
            'Last20-pretrigger states use later pretrigger data, not training-only predictions.',
            'Some shape-time parameters are at bounds or beyond recording support.',
            'Charge spectra and peak-time distributions do not all agree; fixed-parameter intervals omit model selection.'],
        'source_sha256':source_hashes,'outputs_sha256':outputs,
        'repository_actions':'No commit or push this phase; pre-existing changes preserved.'}
    (BASE/'record.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    print('Archived',len(source_hashes),'source files and',len(outputs),'outputs.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--verify',action='store_true')
    main(parser.parse_args().verify)
