"""Verify the PID analysis and optionally freeze its sources and artifacts."""
import argparse
import base64
import csv
import hashlib
import json
import platform
import re
import shutil
from pathlib import Path
import numpy as np
import scipy
from scipy.stats import mannwhitneyu
import ROOT
import nbformat
from prepare_pid import OUT,PARENT,PROJECT,STATES,SOURCES,read
from analyze_pid import combined,METHODS,CASES,ci
from extract_pid import load,profile,label_truth
from reference import wave_features
from pid_metrics import binomial,gaussian_equivalent_fom
from publication_stats import sha,save


def hashes(mapping,base=PROJECT):
    for relative,digest in mapping.items():assert sha(base/relative)==digest,relative
    return len(mapping)


def independent_auc(a,g):
    return float(1-mannwhitneyu(a,g,alternative='less',method='asymptotic').statistic/(len(a)*len(g)))


def independent_scores(waves,pca):
    f=wave_features(waves);scores=[]
    for i,w in enumerate(waves):
        y=w.astype(float)-f['baseline'][i];anchor=int(np.floor(f['t10_sample'][i]))+1
        start=anchor-20;stop=anchor+1500;short=anchor+250
        q=y[start:stop].sum();qs=y[start:short].sum()
        time=np.dot(y[start:stop],np.arange(-20,1500))*.002/q
        blocks=np.array([y[start+20*k:start+20*k+20].sum() for k in range(76)],dtype=np.float32)
        pc=np.dot(blocks.astype(float)/q-np.array(pca['center']),np.array(pca['vector']))/pca['score_scale']
        scores.append([qs/q,-time,pc])
    return np.array(scores)


def main(archive=False):
    if (OUT/'record.json').exists():
        assert not archive,'Archive exists; use read-only verification'
        record=read(OUT/'record.json')
        print('Frozen PID outputs checked:',hashes(record['outputs_sha256']),flush=True)
        print('Frozen PID source snapshots checked:',hashes(record['source_sha256'],OUT/'source_snapshot'),flush=True)
        return
    protocol=read(OUT/'protocol.json');report=read(OUT/'pid.json');checks={}
    assert sha(PARENT/'record.json')==protocol['parent_record_sha256']
    parent=read(PARENT/'record.json')
    checks['parent_waveform_archive']={'record_sha256':sha(PARENT/'record.json'),
        'unchanged_outputs':hashes(parent['outputs_sha256']),
        'unchanged_source_snapshots':hashes(parent['source_sha256'],PARENT/'source_snapshot')}
    print('Previous waveform archive unchanged.',flush=True)
    assert sha(PROJECT/'simulation/prepare_pid.py')==protocol['code_sha256']
    checks['input_hashes']=hashes(protocol['inputs_sha256'])
    assert sha(PROJECT/'simulation/build/replay_response')==protocol['replay_binary_sha256']
    assert report['code_sha256']==sha(PROJECT/'simulation/analyze_pid.py')
    assert report['metrics_code_sha256']==sha(PROJECT/'simulation/pid_metrics.py')
    assert report['protocol_sha256']==sha(OUT/'protocol.json')
    assert protocol['split_sha256']==sha(OUT/'mother_splits.npz')
    splits=load(OUT/'mother_splits.npz')
    for source,n in [('Co60',4000),('Am241',24000)]:
        rng=np.random.default_rng(np.random.SeedSequence([291029701,60 if source=='Co60' else 241]))
        split=np.ones(n,dtype=np.int8);split[rng.permutation(n)[:n//2]]=0
        np.testing.assert_array_equal(splits[source],split)
    pca=read(PROJECT/'analysis/paper_baseline/results/publication_20261028/point.json')['windows']['3000']['pca']
    checks['native_inputs']={}
    for key,entry in protocol['source_rows'].items():
        r=read(OUT/'extraction'/f'{key}.json');d=load(OUT/'arrays'/f'{key}.npz')
        assert r['code_sha256']==sha(PROJECT/'simulation/extract_pid.py')
        assert r['protocol_sha256']==sha(OUT/'protocol.json')
        path=PROJECT/entry['path'];assert sha(path)==r['ROOT_sha256']
        f=ROOT.TFile.Open(str(path));info=json.loads(str(f.Get('RunInfo').GetString().Data()));t=f.Get('Events')
        assert info['completed'] and t.GetEntries()==r['nparent']
        cfg=read(PROJECT/entry['scenario_config']);assert sha(PROJECT/entry['scenario_config'])==entry['scenario_config_sha256']
        assert profile(info['config'])==profile(cfg)
        if entry['kind']!='existing_native':assert info['response_replay']['equal_original_avalanche_events']==r['nparent']
        t.SetBranchStatus('*',0);t.SetBranchStatus('adc',1)
        ids=np.flatnonzero(d['common']);waves=[]
        for i in ids:t.GetEntry(int(i));waves.append(np.array(t.adc,dtype=np.uint16))
        f.Close()
        np.testing.assert_allclose(independent_scores(np.array(waves),pca),d['scores'][ids],rtol=1e-11,atol=1e-11)
        labels,_=label_truth(d['energy_by_origin']);np.testing.assert_array_equal(labels,d['truth'])
        np.testing.assert_array_equal(d['split'],splits[r['source']])
        checks['native_inputs'][key]={'mothers':r['nparent'],'common_n':len(ids),'kind':entry['kind'],'ADC_scores_independently_recomputed':len(ids)}
    for state in STATES:
        a=read(PROJECT/protocol['source_rows'][state+'_Co60']['scenario_config'])
        b=read(PROJECT/protocol['source_rows'][state+'_Am241']['scenario_config'])
        assert profile(a)==profile(b)
        assert a['optical']==b['optical']
        aa={k:v for k,v in a['acquisition'].items() if k!='amplitude_threshold_adc'}
        bb={k:v for k,v in b['acquisition'].items() if k!='amplitude_threshold_adc'}
        assert aa==bb
    checks['common_readout_profiles_verified']=3
    print('Native profiles, replay identity and ADC scores verified.',flush=True)

    measured=read(OUT/'extraction/measured.json')
    assert measured['code_sha256']==sha(PROJECT/'simulation/extract_pid.py')
    for run in [3,12,18]:
        poolpath=PROJECT/f'analysis/paper_baseline/results/publication_20261028/pool_run{run}.npz'
        pool=load(poolpath)
        for segment_index,segment in enumerate(['first60','middle20','last20']):
            key=f'run{run}_{segment}';m=measured['groups'][key]
            assert sha(PROJECT/m['input'])==m['input_sha256'] and sha(poolpath)==m['pool_sha256']
            d=load(OUT/'arrays'/f'measured_{key}.npz');use=pool['primary']&(pool['split']==segment_index)
            np.testing.assert_array_equal(d['ids'],pool['ids'][use]);q=pool['long_charge'][use,2]
            scores=np.c_[pool['short_charge'][use,2]/q,-pool['mean_time_us'][use,2],
                ((pool['block_charge'][use,:76].astype(float)/q[:,None]-np.array(pca['center']))@np.array(pca['vector']))/pca['score_scale']]
            np.testing.assert_allclose(d['scores'],scores,rtol=1e-11,atol=1e-11)
            np.testing.assert_array_equal(d['q'],q)
    checks['measured_frozen_ID_and_score_identity']=sum(x['n'] for x in measured['groups'].values())
    datasets={case:combined(case,protocol) for case in CASES};primary=datasets['run12'];thresholds=read(OUT/'thresholds.json')
    assert thresholds['code_sha256']==report['code_sha256']
    assert report['thresholds_sha256']==sha(OUT/'thresholds.json')
    training=primary['common']&(primary['split']==0)&(primary['truth']==0)
    np.testing.assert_array_equal(primary['mother'][training],thresholds['alpha_mothers'])
    for j,method in enumerate(METHODS):
        assert np.quantile(primary['scores'][training,j],.95,method='higher')==thresholds['methods'][method]['threshold']
    counts=load(OUT/'bootstrap_mother_counts.npz')['counts'];assert counts.shape==(512,28000)
    np.testing.assert_array_equal(counts[:,:4000].sum(1),np.full(512,4000))
    np.testing.assert_array_equal(counts[:,4000:].sum(1),np.full(512,24000))
    draws={};checks['tested_working_points']=0
    for case,d in datasets.items():
        r=report['cases'][case];test=d['common']&(d['split']==1)
        a=test&(d['truth']==0);g=test&(d['truth']==1)
        aa=d['common']&(d['truth']==0);gg=d['common']&(d['truth']==1)
        draws[case]=load(OUT/'curves'/f'{case}_bootstrap.npz')
        for j,method in enumerate(METHODS):
            score=d['scores'][:,j];s=r['methods'][method];threshold=thresholds['methods'][method]['threshold']
            assert independent_auc(score[a],score[g])==s['test_AUC']
            for mask,key in [(a,'alpha_acceptance'),(g,'gamma_leakage')]:
                actual=binomial(int(np.sum(score[mask]<=threshold)),int(mask.sum()))
                assert actual==s['operating'][key]
            checks['tested_working_points']+=1
            curve=load(OUT/'curves'/f'{case}_{method}.npz')
            assert np.all(np.diff(curve['gamma_leakage'])>=0) and np.all(np.diff(curve['alpha_acceptance'])>=0)
            assert abs(np.trapezoid(curve['alpha_acceptance'],curve['gamma_leakage'])-s['test_AUC'])<1e-12
            assert ci(draws[case]['auc'][:,j])==s['test_AUC_uncertainty']
            assert ci(draws[case]['fom'][:,j])==s['truth_FOM_uncertainty']
            assert ci(draws[case]['binned_auc'][:,j])==s['binned_AUC_uncertainty']
            for b,cell in enumerate(s['per_Q']):
                if 'truth_FOM' in cell:assert ci(draws[case]['fom_per_bin'][:,j,b])==cell['truth_FOM_uncertainty']
            for fit in s['source_mixture_fits']:
                if fit['valid']:
                    f=fit['fit'];value=np.diff(f['means'])[0]/(2*np.sqrt(2*np.log(2))*sum(f['sigmas']))
                    assert value==f['shape_fom_fwhm'] and f['converged'] and not f['at_sigma_floor']
            for k in [0,255,511]:
                x=np.repeat(score[a],counts[k,d['mother'][a]]);y=np.repeat(score[g],counts[k,d['mother'][g]])
                assert abs(independent_auc(x,y)-draws[case]['auc'][k,j])<1e-12
                x=np.repeat(score[aa],counts[k,d['mother'][aa]]);y=np.repeat(score[gg],counts[k,d['mother'][gg]])
                value=abs(x.mean()-y.mean())/(2*np.sqrt(2*np.log(2))*(x.std(ddof=1)+y.std(ddof=1)))
                np.testing.assert_allclose(value,draws[case]['fom'][k,j],rtol=1e-12,atol=1e-12)
    for case in CASES:
        if case=='run12':continue
        for j,method in enumerate(METHODS):
            delta=load(OUT/'curves'/f'{case}_minus_run12_{method}.npz')
            for metric in ['auc','fom','binned_auc']:
                np.testing.assert_array_equal(delta[metric],draws[case][metric][:,j]-draws['run12'][metric][:,j])
                assert ci(delta[metric])==report['paired_state_differences'][case][method][metric]
    checks['bootstrap']='All saved intervals and paired differences reconstructed; three whole-mother draws per case/method independently reproduced from replicated observations.'
    checks['thresholds_use_only_training_mothers']=True
    print('Frozen measured cohorts, thresholds, ROC and uncertainty verified.',flush=True)

    exported=read(OUT/'exports.json');assert exported['code_sha256']==sha(PROJECT/'simulation/export_pid.py')
    for kind,r in exported['tables'].items():
        assert sha(PROJECT/r['path'])==r['sha256']
        with (PROJECT/r['path']).open() as f:rows=list(csv.DictReader(f))
        assert len(rows)==r['rows']
        if kind=='measured':assert all(r['particle_truth']=='unavailable' for r in rows)
        for row in rows:
            for method in METHODS:
                region='low_score' if float(row[method])<=thresholds['methods'][method]['threshold'] else 'high_score'
                assert row[method+'_region']==region
    assert exported['classifier_sha256']==sha(OUT/'pid_classifier.json')
    figures=read(OUT/'figures/figure_record.json')
    assert figures['code_sha256']==sha(PROJECT/'simulation/plot_pid.py')
    assert figures['PID_sha256']==sha(OUT/'pid.json')
    notebook=PROJECT/'jupyter/codex_0001_csi_geant4_pid.ipynb'
    nb=nbformat.read(notebook,as_version=4);nbformat.validate(nb)
    code=[c for c in nb.cells if c.cell_type=='code']
    assert [c.execution_count for c in code]==list(range(1,len(code)+1))
    assert not any(o.output_type=='error' for c in code for o in c.outputs)
    images=[o.data['image/png'] for c in code for o in c.outputs if 'image/png' in o.get('data',{})]
    assert len(images)==4
    for image,relative in zip(images,figures['figures']):
        path=PROJECT/relative
        assert hashlib.sha256(base64.b64decode(image)).hexdigest()==sha(path)
        assert path.with_suffix('.pdf').read_bytes().startswith(b'%PDF')
        assert '<svg' in path.with_suffix('.svg').read_text()
    for link in re.findall(r'\]\(([^)]+)\)','\n'.join(c.source for c in nb.cells)):
        target=(notebook.parent/link).resolve()
        if target!=OUT/'record.json':assert target.exists(),link
    checks['notebook']={'executed_code_cells':len(code),'embedded_PNGs_byte_verified':len(images),'errors':0,
        'schema_valid':True,'execution_method':'Actual in-process IPython cell execution'}
    checks['figures']={'PNG_groups_visually_reviewed':4,'PDF_SVG_headers_checked':True,'PDF_not_separately_rendered':True}
    testlog=Path('/tmp/sipm_pid_tests.log').read_text()
    assert 'Ran 5 tests' in testlog and testlog.rstrip().endswith('OK')
    checks['statistical_unit_tests']={'count':5,'passed':True}
    if not archive:
        print(json.dumps(checks,indent=2),flush=True);return
    logs=OUT/'logs';logs.mkdir(exist_ok=True)
    for path in Path('/tmp').glob('sipm_pid*.log'):
        if 'verify' not in path.name:shutil.copy2(path,logs/path.name)
    sources=list((PROJECT/'simulation').glob('*.py'))+list((PROJECT/'simulation').glob('*.cc'))
    for sub in ['include','src','tests','vendor']:
        sources += [p for p in (PROJECT/'simulation'/sub).rglob('*') if p.is_file() and p.suffix in ['.py','.cc','.hh','.h','.json']]
    sources += [PROJECT/p for p in ['README.md','simulation/README.md','simulation/CMakeLists.txt',
        'analysis/paper_baseline/prepare.py','analysis/paper_baseline/refine_waveforms.py',
        'analysis/paper_baseline/publication_stats.py','analysis/paper_baseline/shape_mixture.py',
        'jupyter/codex_0001_build_pid_notebook.py']]+[notebook]
    source_hashes={}
    for path in sorted(set(sources)):
        relative=path.relative_to(PROJECT);target=OUT/'source_snapshot'/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target);source_hashes[str(relative)]=sha(path);assert sha(target)==source_hashes[str(relative)]
    outputs={str(p.relative_to(PROJECT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()
        and p!=OUT/'record.json' and 'source_snapshot' not in p.relative_to(OUT).parts}
    save(OUT/'record.json',{'date':'2026-10-29','scope':__doc__,'checks':checks,
        'environment':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'ROOT':ROOT.gROOT.GetVersion()},
        'MC_accounting':protocol['MC_accounting'],
        'raw_data':'Existing frozen waveforms and feature pools only; no raw experiment data changed.',
        'limitations':protocol['known_limits']+['Zero observed errors do not establish zero true leakage.',
             'Measured particle truth unavailable; exported score regions are conditional predictions.',
             'PID widths and source spectra not completely reproduced.'],
        'source_sha256':source_hashes,'outputs_sha256':outputs,
        'repository_actions':'No commit/push; pre-existing changes and earlier frozen archives preserved.'})
    print('Archived',len(source_hashes),'sources and',len(outputs),'outputs.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--archive',action='store_true')
    main(parser.parse_args().archive)
