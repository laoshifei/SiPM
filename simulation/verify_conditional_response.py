"""Verify native identity, conditional estimators, uncertainty and provenance."""
import argparse
import base64
import functools
import json
import shutil
from pathlib import Path
import numpy as np
import nbformat
from prepare_conditional_response import OUT, PARENT, PROJECT, BUILD, SOURCES, SEGMENTS, VARIANTS, relative
from prepare_pid import read
from extract_pid import load
from publication_stats import sha, save
from analyze_conditional_response import quantile, bin_weights, summary, design_for, cohort, contrasts
from verify_pid import independent_scores


@functools.lru_cache(None)
def digest(path):return sha(path)


def hashes(mapping,base=PROJECT):
    for path,expected in mapping.items():assert digest(base/path)==expected,path
    return len(mapping)


def check_interval(values,record):
    finite=np.asarray(values)[np.isfinite(values)]
    assert len(finite)==record['valid_draws']
    if len(finite)<128:assert record['CI95'] is None
    else:np.testing.assert_allclose(record['CI95'],np.quantile(finite,[.025,.975]),atol=1e-12,rtol=1e-12)


def sources():
    files=set((PROJECT/'simulation').glob('*.py'))
    for name in ['include','src','vendor','tests']:
        files.update(p for p in (PROJECT/'simulation'/name).rglob('*') if p.suffix in ['.cc','.hh','.h','.py','.json'])
    files.update((PROJECT/'analysis/paper_baseline').glob('*.py'))
    files.update([PROJECT/'simulation/replay_response.cc',PROJECT/'simulation/main.cc',PROJECT/'simulation/CMakeLists.txt',
                  PROJECT/'simulation/README.md',PROJECT/'jupyter/codex_0001_build_conditional_response_notebook.py'])
    return files


def archive():
    assert not (OUT/'record.json').exists()
    checks=read(OUT/'verification.json')
    hashes(checks['verified_outputs_sha256']);hashes(checks['verified_sources_sha256'])
    assert digest(PARENT/'record.json')==checks['parent_record_sha256']
    source_hashes={}
    for path in sorted(sources()):
        target=OUT/'source_snapshot'/path.relative_to(PROJECT);target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target);source_hashes[relative(path)]=sha(target)
    outputs={relative(path):digest(path) for path in sorted(OUT.rglob('*')) if path.is_file() and 'source_snapshot' not in path.parts}
    nb=PROJECT/'jupyter/codex_0001_csi_conditional_response.ipynb';outputs[relative(nb)]=digest(nb)
    save(OUT/'record.json',{'scope':__doc__,'outputs_sha256':outputs,'source_sha256':source_hashes,
        'new_independent_mothers':0,'replay_entries':192000,'unique_reused_mothers':32000,
        'limits':'Diagnostic exploration on reused MC and previously explored data. No new fit, detector-parameter identification, equivalence proof or experimental PID efficiency.'})
    print('Frozen',len(outputs),'outputs and',len(source_hashes),'source snapshots.',flush=True)


def main():
    if (OUT/'record.json').exists():
        r=read(OUT/'record.json');print('Frozen output hashes:',hashes(r['outputs_sha256']))
        print('Frozen source hashes:',hashes(r['source_sha256'],OUT/'source_snapshot'));return
    assert not (OUT/'verification.json').exists()
    p=read(OUT/'protocol.json');c=read(OUT/'conditional.json');m=read(OUT/'mechanisms.json');ex=read(OUT/'execution.json')
    checks={'scope':__doc__,'parent_record_sha256':digest(PARENT/'record.json')}
    assert checks['parent_record_sha256']==p['parent_record_sha256']
    parent=read(PARENT/'record.json')
    checks['preserved_parent_outputs']=hashes(parent['outputs_sha256'])
    checks['preserved_parent_sources']=hashes(parent['source_sha256'],PARENT/'source_snapshot')
    checks['protocol_inputs']=hashes(p['inputs_sha256']);checks['protocol_sources']=hashes(p['source_sha256'])
    assert digest(BUILD/'replay_response')==p['binary_sha256']
    assert digest(PROJECT/p['literature_note'])==p['literature_note_sha256']
    for name,script in [('conditional','analyze_conditional_response.py'),('mechanisms','analyze_conditional_response.py'),('execution','run_conditional_response.py'),('figure_record','plot_conditional_response.py')]:
        r=read(OUT/f'{name}.json');assert r['code_sha256']==digest(PROJECT/'simulation'/script)
    assert ex['replay_entries']==sum(e['events'] for e in p['entries'].values())==192000
    assert ex['unique_reused_mothers']==32000 and ex['new_independent_mothers']==0
    assert len(ex['jobs'])==12 and all(all(s.get('exit',0)==0 for s in e['steps']) for e in ex['jobs'])
    checks['native_cases']={};pca=read(PROJECT/'analysis/paper_baseline/results/publication_20261028/point.json')['windows']['3000']['pca']
    for key,e in p['entries'].items():
        a=load(OUT/'arrays'/f'{key}.npz');r=read(OUT/'extraction'/f'{key}.json');root=PROJECT/e['path']
        assert digest(root)==r['root_sha256'];assert digest(PROJECT/e['parent'])==e['parent_sha256']
        assert digest(PROJECT/e['config'])==e['config_sha256']==r['config_sha256']
        assert r['code_sha256']==digest(PROJECT/'simulation/run_conditional_response.py')
        check=read(root.with_name(root.stem+'_validation.json'));assert check['passed'] and check['events']==e['events']
        assert r['energy_identity'] and r['diagnostic_charge_count_closure']
        np.testing.assert_allclose(independent_scores(a['waves'],pca),a['scores'],rtol=1e-11,atol=1e-11)
        assert np.all(np.diff(a['ids'])>0) and int(a['n_parent'])==e['events']
        assert not a['diag_n_dark'].any() and not a['diag_n_crosstalk'].any()
        if e['variant'] in ['identity','no_signal_residual','recovery60ns']:assert not a['diag_n_afterpulse'].any()
        if e['variant']=='identity':
            assert r['replay']['equal_original_adc_events']==r['replay']['equal_original_avalanche_events']==e['events']
        if e['variant']=='no_signal_residual':assert r['replay']['equal_original_avalanche_events']==e['events']
        checks['native_cases'][key]={'all_mothers':e['events'],'independent_PID_recomputed':len(a['ids']),
            'common_events':int(a['common'].sum()),'zero_dark_and_crosstalk':True,'source_ancestry_and_charge_validated':True}
    print('Native ADC / diagnostic checks passed.',flush=True)
    rng=np.random.default_rng(31030)
    for n in [5,20,100]:
        x=rng.normal(size=n);np.testing.assert_allclose(quantile(x,np.ones(n)),np.quantile(x,[.25,.5,.75]),atol=1e-14)
    cb=load(OUT/'conditional_bootstrap.npz');mb=load(OUT/'mechanism_bootstrap.npz');checks['statistical_checks']={}
    n_mixture=0
    for source,run in SOURCES:
        design=design_for(p,source)
        for segment in SEGMENTS:
            group=f'run{run}_{segment}';data=load(OUT/'arrays'/f'data_{group}.npz')
            for mode in p['conditional_modes']:
                r=c['groups'][group]['modes'][mode];value=summary(data,design,mode)
                np.testing.assert_allclose(value,np.asarray(r['data'],float),equal_nan=True,atol=1e-14)
                for b in design['support']:
                    out=bin_weights(data,design,mode,b)
                    if out is None:continue
                    ix,w,cell,t=out;x=data['scores'][ix];mu=w@x
                    np.testing.assert_allclose(np.bincount(cell,weights=w,minlength=len(t)),t,atol=1e-14)
                    within=np.zeros((3,3));between=within.copy()
                    for j,prob in enumerate(t):
                        if prob==0:continue
                        u=cell==j;local=np.average(x[u],axis=0,weights=w[u]);resid=x[u]-local
                        within+=resid.T@(w[u,None]*resid);between+=prob*np.outer(local-mu,local-mu)
                    np.testing.assert_allclose((x-mu).T@(w[:,None]*(x-mu)),within+between,rtol=1e-11,atol=1e-16);n_mixture+=1
                for block in [64,128]:
                    B=np.array([contrasts(x,y) for x,y in zip(cb[f'run{run}__MC__{mode}'],cb[f'{group}__block{block}__{mode}'])])
                    check_interval(B[:,0],r['uncertainty'][str(block)]['mean_time'])
                    for j in range(3):check_interval(B[:,j+1],r['uncertainty'][str(block)]['IQR_ratio'][j])
        for variant in VARIANTS:
            a=load(OUT/'arrays'/f'run{run}_{source}__{variant}.npz')
            for scheme in ['fixed','reselected','fixed_anchor']:
                row=m['sources'][source]['variants'][variant]['schemes'][scheme]
                np.testing.assert_allclose(summary(cohort(a,scheme),design,'q2'),np.asarray(row['point'],float),atol=1e-14)
                b=mb[f'run{run}__{variant}__{scheme}'];ref=mb[f'run{run}__no_signal_residual__{scheme}']
                effects=np.c_[-1000*(b[:,1]-ref[:,1]),b[:,3:6]/ref[:,3:6]]
                for j in range(4):check_interval(effects[:,j],row['paired_effect_CI95'][j])
    assert c['original_estimator_closure']
    checks['statistical_checks']={'original_estimator_closure':True,'weighted_quantile_equal_weight_type7':True,
        'mixture_covariance_checks':n_mixture,'bootstrap_intervals_recomputed':True,'support_weights_sum_to_targets':True}
    tests=BUILD/'Testing/Temporary/LastTest.log';text=tests.read_text()
    assert text.count('Test Passed.')==4
    shutil.copy2(tests,OUT/'logs/cpp_tests.log');checks['cpp_tests_passed']=4
    nbpath=PROJECT/'jupyter/codex_0001_csi_conditional_response.ipynb';nb=nbformat.read(nbpath,as_version=4);nbformat.validate(nb)
    codes=[cell for cell in nb.cells if cell.cell_type=='code'];assert all(cell.execution_count for cell in codes)
    assert not any(o.output_type=='error' for cell in codes for o in cell.outputs)
    embedded=[base64.b64decode(o.data['image/png']) for cell in codes for o in cell.outputs if 'data' in o and 'image/png' in o.data]
    figs=read(OUT/'figure_record.json')['figures'];assert len(embedded)==len(figs)==4
    for name,blob in zip(figs,embedded):assert (OUT/'figures'/f'{name}.png').read_bytes()==blob
    checks['notebook']={'cells':len(nb.cells),'executed_code_cells':len(codes),'exact_embedded_pngs':len(embedded),'sha256':digest(nbpath)}
    assert not read(OUT/'decision.json')['promoted_to_default']
    checks['verified_outputs_sha256']={relative(f):digest(f) for f in sorted(OUT.rglob('*')) if f.is_file() and 'source_snapshot' not in f.parts}
    checks['verified_outputs_sha256'][relative(nbpath)]=digest(nbpath)
    checks['verified_sources_sha256']={relative(f):digest(f) for f in sources() if f.name!='README.md'}
    save(OUT/'verification.json',checks)
    print('Conditional statistics, C++ tests, executed notebook and provenance verified.',flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--archive',action='store_true');args=ap.parse_args()
    archive() if args.archive else main()
