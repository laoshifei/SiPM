"""Independent gate algebra, prediction separation and provenance checks."""
import argparse
import base64
import functools
import platform
import shutil
from pathlib import Path
import numpy as np
import scipy
import nbformat
from prepare_acquisition_state import OUT,PARENT,DATA,WIDTH,PROJECT,SOURCES,SEGMENTS,CONTROLS,MC_VARIANTS,read,load,relative
from analyze_acquisition_state import (event_weights,statistics,contrasts,deciles,permuted_order,
                                       design_matrix,predict,MODELS)
from publication_stats import save,sha


@functools.lru_cache(None)
def digest(path):return sha(path)


def hashes(mapping,base=PROJECT):
    for path,value in mapping.items():assert digest(base/path)==value,path
    return len(mapping)


def independent_prefix_scores(waves,anchor,baseline,pca):
    raw=waves.astype(float);n,length=raw.shape;i=np.arange(n)
    cum=np.c_[np.zeros(n),np.cumsum(raw,axis=1)]
    cumt=np.c_[np.zeros(n),np.cumsum(raw*np.arange(length),axis=1)]
    start=anchor-20;end=anchor+1500
    q=cum[i,end]-cum[i,start]-1520*baseline
    short=cum[i,anchor+250]-cum[i,start]-270*baseline
    time=cumt[i,end]-cumt[i,start]-baseline*(start+end-1)*1520/2-anchor*q
    boundaries=start[:,None]+20*np.arange(77)
    block=np.diff(cum[i[:,None],boundaries],axis=1)-20*baseline[:,None]
    pc=(block/q[:,None]-np.array(pca['center']))@np.array(pca['vector'])/pca['score_scale']
    return np.c_[short/q,-.002*time/q,pc],q,block/q[:,None]


def check_interval(x,row):
    x=np.asarray(x);x=x[np.isfinite(x)];assert len(x)==row['valid_draws']
    if len(x)<128:assert row['CI95'] is None
    else:np.testing.assert_allclose(row['CI95'],np.quantile(x,[.025,.975]),rtol=1e-12,atol=1e-12)


def sources():
    files=[PROJECT/'simulation'/name for name in ['prepare_acquisition_state.py','analyze_acquisition_state.py',
        'plot_acquisition_state.py','verify_acquisition_state.py','reference.py','README.md']]
    files += [PROJECT/'analysis/paper_baseline'/name for name in ['prepare.py','publication_stats.py','shape_mixture.py']]
    files += [PROJECT/'jupyter/codex_0001_build_acquisition_state_notebook.py']
    return files


def freeze():
    assert not (OUT/'record.json').exists();v=read(OUT/'verification.json')
    hashes(v['verified_outputs_sha256']);hashes(v['verified_sources_sha256'])
    snapshots={}
    for f in sources():
        target=OUT/'source_snapshot'/f.relative_to(PROJECT);target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(f,target);snapshots[relative(f)]=sha(target)
    outputs={relative(f):digest(f) for f in sorted(OUT.rglob('*')) if f.is_file() and 'source_snapshot' not in f.parts}
    nb=PROJECT/'jupyter/codex_0001_csi_acquisition_state.ipynb';outputs[relative(nb)]=digest(nb)
    save(OUT/'record.json',{'scope':__doc__,'outputs_sha256':outputs,'source_sha256':snapshots,
        'parent_record_sha256':digest(PARENT/'record.json'),'new_independent_mothers':0,'new_native_replay_entries':0,
        'limits':'Existing-data estimator sensitivity and acquisition-state exploration. Fixed first60 targets/PCA; nuisance regression crossvalidation only. No causal noise attribution or new independent experimental validation.'})
    print('Archived',len(outputs),'outputs and',len(snapshots),'source files.',flush=True)


def main():
    if (OUT/'record.json').exists():
        r=read(OUT/'record.json');print('Frozen outputs:',hashes(r['outputs_sha256']));print('Source snapshots:',hashes(r['source_sha256'],OUT/'source_snapshot'));return
    assert not (OUT/'verification.json').exists()
    p=read(OUT/'protocol.json');c=read(OUT/'controls.json');s=read(OUT/'states.json');checks={'scope':__doc__}
    assert digest(PARENT/'record.json')==p['parent_record_sha256']
    checks['inputs_unchanged']=hashes(p['inputs_sha256']);checks['parent_record_sha256']=p['parent_record_sha256']
    assert p['checkpoint_commit']=='f5f8a1e55768141d9ceef17bc91772f8ee5689b4'
    for report,script in [(p,'prepare_acquisition_state.py'),(c,'analyze_acquisition_state.py'),(s,'analyze_acquisition_state.py'),(read(OUT/'figure_record.json'),'plot_acquisition_state.py')]:
        assert report['code_sha256']==digest(PROJECT/'simulation'/script)
    for r in [c,s]:assert r['protocol_sha256']==digest(OUT/'protocol.json')
    checks['figure_inputs_unchanged']=hashes(read(OUT/'figure_record.json')['inputs_sha256'])
    pca=read(PROJECT/'analysis/paper_baseline/results/publication_20261028/point.json')['windows']['3000']['pca']
    checked=0
    for source,run in SOURCES:
        for key in [f'run{run}_{seg}' for seg in SEGMENTS]+[f'run{run}_{source}__{v}' for v in MC_VARIANTS]:
            a=load(OUT/'arrays'/f'{key}.npz')
            if '__' in key:
                old=load(PARENT/'arrays'/f'{key}.npz');u=old['fixed'];waves=old['waves'][u]
                np.testing.assert_array_equal(a['ids'],old['ids'][u]);np.testing.assert_array_equal(a['q'],old['original_q'][u])
                np.testing.assert_allclose(a['F'],old['metrics'][u],atol=1e-14)
                np.testing.assert_allclose(statistics(a,p,source),c['MC'][source][key.split('__')[-1]],atol=1e-12)
            else:
                old=load(DATA/f'{key}.npz');u=load(WIDTH/'arrays'/f'measured_{key}.npz')['main_band'];waves=old['waves'][u]
                frozen=load(PARENT/'arrays'/f'data_{key}.npz');np.testing.assert_array_equal(a['ids'],frozen['ids'])
                np.testing.assert_array_equal(a['q'],frozen['q']);np.testing.assert_allclose(a['F'],old['metrics'][u,:,3],atol=1e-14)
            for j,name in enumerate(CONTROLS):
                anchor=a['smooth_anchor'] if j>=2 else a['anchor'];baseline=a['recent_baseline'] if j in [1,3] else a['baseline']
                score,q,shape=independent_prefix_scores(waves,anchor,baseline,pca)
                np.testing.assert_allclose(score,a['scores'][:,j],rtol=1e-10,atol=1e-10)
                np.testing.assert_allclose(q,a['control_q'][:,j],rtol=1e-12,atol=1e-7)
                if j==0:np.testing.assert_allclose(shape,a['shape'],atol=1e-12)
            db=a['recent_baseline']-a['baseline'];np.testing.assert_allclose(a['control_q'][:,1],a['control_q'][:,0]-1520*db,atol=1e-7)
            np.testing.assert_allclose(a['hp_ac'][:,0],a['F'][:,0],rtol=1e-10,atol=1e-10)
            np.testing.assert_allclose(a['shape'].sum(1),1.,atol=1e-12);checked+=len(waves)*4
    checks['independent_prefix_PID_event_controls']=checked
    print('All four gate/baseline controls independently reconstructed.',flush=True)
    b=load(OUT/'control_bootstrap.npz');diag=load(OUT/'state_diagnostics.npz');pred=load(OUT/'predictions.npz');null=load(OUT/'permutation_reference.npz')
    for key,v in diag.items():
        if key.endswith('__shape_cov'):
            np.testing.assert_allclose(v,v.T,atol=1e-15)
            assert np.linalg.eigvalsh(v).min()>=-1e-14
            np.testing.assert_allclose(v@np.ones(76),0.,atol=1e-14)
    for source,run in SOURCES:
        first=load(OUT/'arrays'/f'run{run}_first60.npz');w=event_weights(first,p,source);fold=deciles(first)
        for model,fit in s['sources'][source]['models'].items():
            assert set(fit['training_ids'])==set(first['ids'][w>0])
            x=design_matrix(first,p,source,np.array(fit['state_center']),np.array(fit['state_scale']),fit['time_center'],fit['time_scale'],model)
            residual=x@np.array(fit['beta'])-first['scores'][:,0]
            np.testing.assert_allclose(x.T@(w[:,None]*residual),0.,atol=1e-10)
            cv=pred[f'run{run}__first60_CV__{model}'];mse=w@((cv-first['scores'][:,0])**2)
            np.testing.assert_allclose(mse,s['sources'][source]['CV_MSE'][model],atol=1e-14)
            for r in s['sources'][source]['folds'][model]:
                use=fold==r['fold'];assert set(r['test_ids'])==set(first['ids'][use]);assert not set(r['test_ids'])&set(r['fit']['training_ids'])
                np.testing.assert_allclose(cv[use],predict(first,p,source,r['fit'])[use],atol=1e-12)
        for seg in SEGMENTS:
            key=f'run{run}_{seg}';a=load(OUT/'arrays'/f'{key}.npz');w=event_weights(a,p,source)
            np.testing.assert_allclose(statistics(a,p,source),c['groups'][key]['data'],atol=1e-12)
            bounds=np.array(s['sources'][source]['segments'][seg]['state_support']['bounds_p01_p99'])
            for j in range(3):assert bounds[j,0]<bounds[j,1]
            for model,fit in s['sources'][source]['models'].items():
                np.testing.assert_allclose(pred[key+'__'+model],predict(a,p,source,fit),atol=1e-12)
            for block in [64,128]:
                raw=np.arange(*map(int,a['span']));rng=np.random.default_rng(773+run+block)
                np.testing.assert_array_equal(np.sort(permuted_order(raw,a['span'],block,rng)),raw)
                t=s['temporal_covariance'][key];parts=np.array(t['covariance_parts'])
                np.testing.assert_allclose(parts[0],parts[1:].sum(0),atol=1e-15)
                reference=null[key+f'__block{block}'];np.testing.assert_allclose(np.quantile(reference,[.025,.5,.975],axis=0),t['permutation_reference'][str(block)]['null_p025_p50_p975'],atol=1e-14)
                for variant in MC_VARIANTS:
                    B=contrasts(b[f'run{run}__{variant}'],b[f'{key}__block{block}'])
                    point=contrasts(np.array(c['MC'][source][variant]),np.array(c['groups'][key]['data']))
                    for j,name in enumerate(CONTROLS):
                        row=c['groups'][key]['comparisons'][variant][name]
                        np.testing.assert_allclose(point[j],np.r_[row['mean_time_difference_ns'],row['IQR_ratio']],atol=1e-12)
                        r=row['uncertainty'][str(block)]
                        for k in range(4):
                            check_interval(B[:,j,k],r['contrasts'][k]);check_interval(B[:,j,k]-B[:,0,k],r['paired_change_from_original'][k])
    assert c['original_conditional_estimator_closure']
    checks['statistical_checks']={'prior_point_estimator_closure':True,'paired_intervals_recomputed':True,'time_covariance_identity':True,
        'partial_raw_block_permutations_bijective':True,'shape_covariance_PSD_and_sum_constraint':True,
        'OLS_normal_equations':True,'all_prediction_folds_disjoint':True,'no_later_data_in_regression_fit':True}
    nbpath=PROJECT/'jupyter/codex_0001_csi_acquisition_state.ipynb';nb=nbformat.read(nbpath,as_version=4);nbformat.validate(nb)
    codes=[x for x in nb.cells if x.cell_type=='code'];assert all(x.execution_count for x in codes)
    assert not any(o.output_type=='error' for cell in codes for o in cell.outputs)
    pictures=[base64.b64decode(o.data['image/png']) for cell in codes for o in cell.outputs if 'data' in o and 'image/png' in o.data]
    names=read(OUT/'figure_record.json')['figures'];assert len(pictures)==len(names)==4
    for name,blob in zip(names,pictures):assert blob==(OUT/'figures'/f'{name}.png').read_bytes()
    checks['notebook']={'cells':len(nb.cells),'executed_code_cells':len(codes),'exact_embedded_pngs':len(pictures),'sha256':digest(nbpath)}
    checks['environment']={'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__}
    assert not read(OUT/'decision.json')['promoted_to_default'] and p['new_native_replay_entries']==0
    checks['verified_outputs_sha256']={relative(f):digest(f) for f in sorted(OUT.rglob('*')) if f.is_file() and 'source_snapshot' not in f.parts}
    checks['verified_outputs_sha256'][relative(nbpath)]=digest(nbpath)
    checks['verified_sources_sha256']={relative(f):digest(f) for f in sources() if f.name!='README.md'}
    save(OUT/'verification.json',checks);print('Estimator, state, notebook and provenance checks passed.',flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--archive',action='store_true');args=ap.parse_args()
    freeze() if args.archive else main()
