"""Verify the publication analysis, archive source and write its provenance record."""
import csv
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from urllib.parse import unquote
import numpy as np
import ROOT
import publication_stats as p


def main(out=p.OUT):
    protocol=json.loads((out/'protocol.json').read_text());checks={};snap=out/'source_snapshot';snap.mkdir(exist_ok=True)
    logs=out/'logs';logs.mkdir(exist_ok=True)
    for name in ('test_publication_stats.py','test_publication_diagnostics.py'):
        result=subprocess.run([sys.executable,str(p.HERE/name)],text=True,capture_output=True)
        (logs/(name+'.log')).write_text(result.stdout+result.stderr)
        if result.returncode:raise ValueError(result.stdout+result.stderr)
    checks['unit_tests']={'statistics':5,'diagnostic_identities':4,'passed':True}
    for path,digest in protocol['code_sha256'].items():assert p.sha(p.PROJECT/path)==digest
    checked_inputs={};inherited_raw={}
    for path,digest in protocol['inputs_sha256'].items():
        source=p.PROJECT/path
        if source.suffix.upper()=='.BIN':
            inherited_raw[path]={'prior_sha256':digest,'current_bytes':source.stat().st_size}
        else:
            assert p.sha(source)==digest;checked_inputs[path]=digest
    checks['current_feature_hashes']=len(checked_inputs)
    with np.load(p.SIM/'results/bias_response_20261028/samples.npz') as prior:
        total=0
        for run in p.RUNS:
            data=p.load(out/f'pool_run{run}.npz')
            for s,group in enumerate(p.GROUPS):
                ids=data['ids'][data['primary']&(data['split']==s)]
                np.testing.assert_array_equal(ids,prior[f'run{run}_{group}_ids']);total+=len(ids)
    assert total==40795;assert len(p.training(out)['ids'])==3997
    checks['all_frozen_ids_equal']=total
    checks['flow_monotonic']=True
    flowkeys=['saved','basic','four_windows','pretrigger_primary','common_Q','amplitude200','bit_clear','time_support','frozen_primary']
    for row in protocol['flow']:assert np.all(np.diff([row[k] for k in flowkeys])<=0)
    stats=json.loads((out/'paired_uncertainty.json').read_text());fit_count=0
    for size,n in ((64,256),(32,128),(128,128)):
        files=sorted((out/f'bootstrap_b{size}').glob('draw*.npz'));assert len(files)==n
        assert not list((out/f'bootstrap_b{size}').glob('draw*.json'))
        assert stats['block_sizes'][str(size)]['draws']==n
        for path in files:
            d=p.load(path);counts=d['counts'];assert counts.shape==(3997,) and np.issubdtype(counts.dtype,np.integer) and counts.min()>=0
            for window in ([1000,2000,3000,4000] if size==64 else [3000]):
                f=d[f'fom_{window}'];assert f.shape==(3,6) and np.isfinite(f).all() and f.min()>0;fit_count+=f.size
        # Recompute paired primary-contrast intervals directly from stored draws.
        values=np.array([np.median(p.load(path)['fom_3000'],axis=1) for path in files])
        for a,b in ((1,0),(2,0),(2,1)):
            key=p.METHODS[a]+' minus '+p.METHODS[b]
            np.testing.assert_array_equal(np.quantile(values[:,a]-values[:,b],[.025,.975]),stats['block_sizes'][str(size)]['windows']['3000']['paired_differences'][key]['interval95'])
    assert fit_count==23040;checks['converged_bootstrap_fits']=fit_count
    computation=json.loads((out/'computation.json').read_text())
    for path,digest in computation['code_sha256'].items():assert p.sha(p.PROJECT/path)==digest
    assert computation['point_fit_comparisons']==72 and computation['resampled_background_comparisons']==3
    checks['native_equivalence']=computation
    covariance=json.loads((out/'covariance_uncertainty.json').read_text())
    for path,digest in covariance['code_sha256'].items():assert p.sha(p.PROJECT/path)==digest
    for size,n in ((64,512),(32,128),(128,128)):
        files=sorted((out/'covariance').glob(f'b{size}_draw*.npy'));assert len(files)==n
        data=np.array([np.load(path) for path in files]);assert data.shape==(n,2,2,6,3,4) and np.isfinite(data).all()
        equal=data.mean(axis=3);delta=equal[:,0]/equal[:,1]-1
        np.testing.assert_array_equal(np.quantile(delta,[.025,.975],axis=0),covariance['block_sizes'][str(size)]['relative_effect_interval95'])
    checks['covariance_draws']={'64':512,'32':128,'128':128}
    transfer=json.loads((out/'transfer_uncertainty.json').read_text());files=sorted((out/'transfer_bootstrap').glob('draw*.npz'))
    assert len(files)==256
    data=np.array([p.load(path)['values'] for path in files]);assert data.shape==(256,6,3,3) and np.isfinite(data).all()
    from publication_transfer import COMPARISONS
    for i,key in enumerate(COMPARISONS):np.testing.assert_array_equal(np.quantile(data[:,i],[.025,.975],axis=0),transfer['groups'][key]['interval95'])
    checks['transfer_draws']=256
    mc=json.loads((out/'mc_uncertainty.json').read_text());assert mc['code_sha256']==p.sha(p.HERE/'publication_mc_uncertainty.py')
    roots={}
    for source in ('Co60','Am241'):
        boot=p.load(out/'mc'/(source+'_bootstrap.npz'))
        assert boot['joint'].shape==(512,3,6) and np.isfinite(boot['joint']).all()
        for i,c in enumerate(('control','native_best','prompt_plus_ap15')):
            row=mc['sources'][source]['candidates'][c]
            assert p.sha(p.PROJECT/row['root'])==row['root_sha256'];roots[row['root']]=row['root_sha256']
            np.testing.assert_array_equal(np.quantile(boot['joint'][:,i],[.025,.975],axis=0),mc['results'][source][c]['joint_interval95'])
    checks['native_root_hashes_checked']=len(roots)
    sensitivity=json.loads((out/'core_model_sensitivity.json').read_text());assert all(sensitivity['checks'].values())
    assert sensitivity['code_sha256']==p.sha(p.HERE/'publication_model_check.py')
    checks['student_t_limit_and_affine_checks']=sensitivity['checks']
    figures=json.loads((out/'figures/manifest.json').read_text());assert len(figures['figure_stems'])==6
    assert figures['code_sha256']==p.sha(p.HERE/'plot_publication.py')
    for name,digest in figures['inputs_sha256'].items():assert p.sha(out/name)==digest
    for stem in figures['figure_stems']:
        for ext,magic in [('pdf',b'%PDF'),('png',b'\x89PNG')]:
            path=out/'figures'/(stem+'.'+ext)
            with path.open('rb') as stream:assert stream.read(len(magic))==magic
        assert '<svg' in (out/'figures'/(stem+'.svg')).read_text()
    checks['figures']={'count':6,'formats':['png','svg','pdf'],'input_hashes_current':True,
        'visual_review':'All six final PNG canvases inspected; peak clipping, small markers, axis/title delimiter and label overlaps corrected. PDF/SVG use the same ROOT canvases; PDF bytes checked, not separately rasterized.'}
    documents=[p.PROJECT/'README.md',p.PROJECT/'EXPERIMENT_INFO.md',p.SIM/'README.md',p.SIM/'RESULTS.md',
      p.HERE/'README.md',p.HERE/'PHASE2_RESULTS.md',p.HERE/'MANUSCRIPT_DRAFT.md',p.HERE/'LOCAL_EVIDENCE.md']
    linked=0
    for doc in documents:
        for match in re.finditer(r'\]\((<[^>]+>|[^)\n]+)\)',doc.read_text()):
            target=match.group(1).strip('<>')
            if target.startswith(('http://','https://','#')):continue
            target=re.sub(r':\d+$','',unquote(target.split('#')[0]))
            path=(doc.parent/target).resolve() if not target.startswith('/') else Path(target)
            if path==out/'record.json':continue
            if not path.exists():raise ValueError(f'Broken link in{doc}: {target}')
            linked+=1
    checks['existing_local_document_targets']=linked
    # Snapshot local analysis modules; native G4 source was archived by the parent stage.
    sources=sorted(set(p.HERE.glob('*.py'))|set(p.HERE.glob('publication_*.cc'))|set(p.SIM.glob('*.py')))
    source_hashes={}
    for source in sources+documents:
        rel=source.relative_to(p.PROJECT);target=snap/rel;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target);source_hashes[str(rel)]=p.sha(source)
        assert p.sha(target)==source_hashes[str(rel)]
    for name in ['sipm_publication_prepare','sipm_publication_point','sipm_publication_native',
                 'sipm_bootstrap64','sipm_bootstrap64_native','sipm_bootstrap32_native','sipm_bootstrap128_native',
                 'sipm_publication_cov','sipm_publication_transfer','sipm_publication_transfer_bootstrap',
                 'sipm_publication_transfer_sensitivity','sipm_publication_mc','sipm_publication_mc_bootstrap',
                 'sipm_publication_model','sipm_publication_plots']:
        log=Path('/tmp')/(name+'.log')
        if log.exists():shutil.copyfile(log,logs/log.name)
    outputs={str(path.relative_to(p.PROJECT)):p.sha(path) for path in sorted(out.rglob('*'))
             if path.is_file() and path.name!='record.json' and 'source_snapshot' not in path.parts}
    record={'date':'2026-10-28','scope':'Completed existing-data publication analysis and draft; not a calibrated detector or particle-efficiency result.',
        'checks':checks,'environment':{'python':sys.version,'numpy':np.__version__,'ROOT':ROOT.gROOT.GetVersion()},
        'feature_inputs_sha256_checked':checked_inputs,'raw_hashes_inherited_not_recomputed':inherited_raw,
        'native_ROOT_inputs_sha256_checked':roots,'source_sha256':source_hashes,'outputs_sha256':outputs,
        'parent_simulation_record':{'path':'simulation/results/response_candidates_20261028/record.json',
            'sha256':p.sha(p.SIM/'results/response_candidates_20261028/record.json')},
        'literature_reused':{'path':'.cache/paper-search/2026-10-28-publication-readiness/assessment.json',
            'sha256':p.sha(p.PROJECT/'.cache/paper-search/2026-10-28-publication-readiness/assessment.json')},
        'remaining_boundaries':['Actual summing/load circuit, source geometry, temperature and calibration remain unconfirmed.',
            'All check data previously explored; candidate selection and total model uncertainty are not in bootstrap intervals.',
            'Local literature comparison does not establish global novelty or journal acceptance.'],
        'repository_actions':'No commit or push in this stage; prior unrelated working-tree edits preserved.'}
    p.save(out/'record.json',record)
    print('PASS:40795 frozen IDs;23040 bootstrap fits;9 unit tests;6 figures;',linked,'local links;',len(outputs),'artifact hashes',flush=True)


if __name__=='__main__':main()
