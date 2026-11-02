"""Verify native coverage, default compatibility, candidate scores and inference."""
import argparse
import json
import shutil
import sys
import nbformat
import ROOT
from spectral_split_common import *
from reference import wave_features
from refine_waveforms import advanced_features


def verify_fit(p):
    fit=read(OUT/'fit.json');checked=0
    for source,group in fit.items():
        for ef,f in group['folds'].items():
            train=np.array(f['training_data_ids']);test=np.array(f['evaluation_data_ids'])
            assert not np.intersect1d(train,test).size
            assert np.all((train//128)%2!=int(ef)) and np.all((test//128)%2==int(ef))
            run=dict(SOURCES)[source];context=f'run{run}_last20_B128'
            raw=load(OUT/'fits'/f'{context}_eval{ef}_quadratic.npz');G=raw['G'].mean(0)
            np.testing.assert_array_equal(G,f['quadratic_matrices']);assert np.all(np.diagonal(G,axis1=1,axis2=2)>=0)
            cfg=read(PROJECT/f['config']);old=read(PROJECT/f['source_config']);assert sha(PROJECT/f['config'])==f['config_sha256']
            for key in ['sipm','electronics_noise','readout_response','optical','geometry','acquisition']:assert cfg[key]==old[key]
            noise=cfg['signal_dependent_noise'];original=old['signal_dependent_noise']
            for key in original:
                if key!='event_variance_cv':assert noise[key]==original[key]
            target=np.array(f['target']['cov']);base=np.array(f['baseline']['zero_cv']['cov']);sd=np.sqrt(np.diag(target));norm=sd[:,None]*sd[None,:];tri=np.triu_indices(3)
            y=((target-base)/norm)[tri];upper=original['event_variance_cv']**2
            for k,c in enumerate(f['candidates'],1):
                a,b=c['a'],c['b'];assert 0<=a<=upper+1e-12 and 0<=b<=4
                assert np.isclose(c['common_variance_cv']**2,a) and np.isclose((1+a)*c['high_variance_cv']**2,b)
                matrix=np.column_stack([(G[j]/norm)[tri] for j in [0,k]]);x=np.array([a,b]);grad=matrix.T@(matrix@x-y)
                limits=np.array([upper,4.]);scale=1+np.linalg.norm(matrix.T@y)
                for j in range(2):
                    if limits[j]==0:continue
                    if x[j]<1e-10:assert grad[j]>=-1e-7*scale
                    elif x[j]>limits[j]-1e-10:assert grad[j]<=1e-7*scale
                    else:assert abs(grad[j])<1e-7*scale
                np.testing.assert_allclose(base+a*G[0]+b*G[k],c['predicted_cov'],rtol=1e-12,atol=1e-12)
                assert np.isclose(np.sum((matrix@x-y)**2),c['objective']);checked+=1
            assert f['chosen_index']==int(np.argmin([c['objective'] for c in f['candidates']]))
            c=f['candidates'][f['chosen_index']];assert noise['spectral_split']['delay_samples']==c['delay_samples']
            assert noise['spectral_split']['high_variance_cv']==c['high_variance_cv'] and noise['event_variance_cv']==c['common_variance_cv']
    return checked


def native_checks(p):
    native=read(OUT/'native_protocol.json');coverage=legacy=scores=sampled=0
    assert native['fit_sha256']==sha(OUT/'fit.json') and native['binary_sha256']==sha(BUILD/'replay_spectral_pool')
    for key,e in native['entries'].items():
        log=read(OUT/'logs'/f'{key}_native.json');extraction=read(OUT/'logs'/f'{key}_extraction.json')
        assert sha(PROJECT/e['output'])==log['output_sha256'] and sha(PROJECT/e['plan'])==e['plan_sha256']
        assert log['binary_sha256']==native['binary_sha256']
        froot=ROOT.TFile.Open(str(PROJECT/e['output']));tree=froot.Get('Readout');info=json.loads(str(froot.Get('RunInfo').GetString().Data()))
        assert info['completed'] and info['events']==e['events'] and tree.GetEntries()==e['events']
        assert info['plan']==read(PROJECT/e['plan']) and info['legacy_control_readouts_bit_identical']==2*e['events']
        assert info['new_independent_mothers']==0;coverage+=e['events'];legacy+=2*e['events']
        for j,case in enumerate(info['plan']['cases']):
            for ef in [0,1]:assert info['configs'][j][ef]==read(case['configs'][ef])
        for vi,v in enumerate(['split_r0','split_r1']):
            path=OUT/'arrays'/f'{key}_{v}.npz';a=load(path);r=extraction['variants'][v];assert sha(path)==r['array_sha256']
            np.testing.assert_array_equal(a['ids'],np.arange(e['events']));np.testing.assert_array_equal(a['fold'],info['plan']['folds'])
            ref=float(np.median(a['baseline_rms']));assert ref==r['global_baseline_RMS']
            common=a['selected']&(a['qcalc']>=p['q_edges'][0])&(a['qcalc']<=p['q_edges'][-1])
            np.testing.assert_array_equal(common,a['common']);assert np.isfinite(a['X'][common]).all()
            rng=np.random.default_rng(731103200+e['run']+100*e['batch']+vi)
            boundary=np.unique(np.concatenate([np.argsort(abs(a['qcalc']-edge))[:16] for edge in p['q_edges']]))
            ids=np.unique(np.r_[np.flatnonzero(common),rng.choice(e['events'],256,replace=False),boundary])
            tree.SetBranchStatus('*',0);tree.SetBranchStatus('adc_'+v,1)
            for start in range(0,len(ids),512):
                ix=ids[start:start+512];waves=[]
                for id_ in ix:assert tree.GetEntry(int(id_))>0;waves.append(np.asarray(getattr(tree,'adc_'+v)).copy())
                waves=np.asarray(waves,np.uint16);f=wave_features(waves)
                advanced=advanced_features(waves,f['t10_sample'],f['peak_sample'],f['amplitude'],f['baseline'],ref)
                selected=(f['diagnostic_flags']==0)&~advanced['pretrigger_flag_primary']&(f['amplitude']>=200)&(f['t10_sample']>=384)
                np.testing.assert_array_equal(selected,a['selected'][ix]);np.testing.assert_array_equal(f['charge_long'],a['qcalc'][ix]);sampled+=len(ix)
                use=common[ix]
                if np.any(use):
                    got=features(waves[use],ix[use],a['qcalc'][ix[use]])
                    np.testing.assert_array_equal(np.c_[got['scores'],got['F']],a['X'][ix[use]])
                    np.testing.assert_array_equal(got['mean_wave'],a['mean_wave'][ix[use]]);scores+=int(use.sum())
        froot.Close();print('Verified candidate ADC/QC/features and legacy closure:',key,flush=True)
    assert coverage==256000 and legacy==512000
    return {'complete_parent_events':coverage,'legacy_readouts_bit_identical':legacy,'legacy_ADC_samples_bit_identical':legacy*2496,
            'candidate_ADC_QC_checks':sampled,'all_candidate_common_features_recomputed':scores}


def verify():
    assert not (OUT/'verification.json').exists()
    p=read(OUT/'protocol.json');ap=read(OUT/'analysis_protocol.json')
    assert ap['code_sha256']==sha(PROJECT/'simulation/analyze_spectral_split.py')
    export_fix=read(OUT/'pre_export_outputs.json')
    for path,h in export_fix['outputs_sha256'].items():assert sha(PROJECT/path)==h,path
    for path,h in {**p['inputs_sha256'],**p['code_sha256'],**p['inherited_code_sha256'],**ap['inputs_sha256']}.items():assert sha(PROJECT/path)==h,path
    proposals=verify_fit(p);native=native_checks(p);analysis=read(OUT/'analysis.json');valid={};cells=0
    for source,g in analysis.items():
        counts=load(OUT/'evaluation'/f'{source}_counts.npz');old=load(FULL/'evaluation'/f'{source}_counts.npz')
        ix=np.searchsorted(counts['ids'],old['ids']);np.testing.assert_array_equal(counts['MC_counts'][:,ix],old['MC_counts'])
        np.testing.assert_array_equal(counts['data_counts'],old['data_counts']);draw=load(OUT/'evaluation'/f'{source}_bootstrap.npz')['contrasts']
        ref=read(SCALES/'references'/f"{g['context']}.json");valid[source]={}
        for k,v in enumerate(VARIANTS):
            a=load(OUT/'evaluation'/f'{source}_{v}.npz');pack=full_pack(a,p,source,ref);covs=[];means=[];profiles=[]
            for _,indices,h,target in pack['cells']:
                count=np.bincount(h,minlength=2);w=target[h]/count[h];x=pack['Y'][indices];m=np.sum(w[:,None]*x,0)
                means.append(m);covs.append((x*w[:,None]).T@x-np.outer(m,m));profiles.append(np.sum(w[:,None]*pack['mean_wave'][indices],0));cells+=1
            np.testing.assert_allclose(np.mean(covs,0),g['summaries'][v]['cov'],rtol=2e-9,atol=2e-10)
            np.testing.assert_allclose(np.mean(means,0),g['summaries'][v]['mean'],rtol=1e-12,atol=1e-11)
            np.testing.assert_allclose(np.mean(profiles,0),g['summaries'][v]['profile'],rtol=1e-12,atol=1e-10)
            assert {n:interval(draw[:,k,j]) for j,n in enumerate(METRICS)}==g['variants'][v]['CI95']
            valid[source][v]=int(np.isfinite(draw[:,k]).all(1).sum())
        for batch in range(4):
            mask=(counts['ids']>=batch*EVENTS[source])&(counts['ids']<(batch+1)*EVENTS[source]);assert np.all(counts['MC_counts'][:,mask].sum(1)<=EVENTS[source])
    nbpath=PROJECT/'jupyter/codex_0001_csi_spectral_variance_split.ipynb';nb=nbformat.read(nbpath,as_version=4);nbformat.validate(nb)
    code=[c for c in nb.cells if c.cell_type=='code'];assert all(c.execution_count is not None for c in code)
    assert all(o.output_type!='error' for c in code for o in c.outputs)
    images=sum('image/png' in o.get('data',{}) for c in code for o in c.outputs);assert images==5
    import csv
    tables={n:len(list(csv.DictReader((OUT/n).open()))) for n in ['proposal_table.csv','comparison_table.csv']};assert set(tables.values())=={12}
    decision=read(OUT/'decision.json');assert decision['code_sha256']==sha(PROJECT/'simulation/report_spectral_split.py')
    build=read(OUT/'build_tests.json');assert build['ctest_passed']==7 and build['ctest_failed']==0
    assert build['csi_sources_sha256']==sha(BUILD/'csi_sources') and build['replay_spectral_pool_sha256']==sha(BUILD/'replay_spectral_pool')
    assert build['ctest_log_sha256']==sha(OUT/'logs/ctest.log') and build['math_tests_sha256']==sha(OUT/'algebra_tests.json')
    save(OUT/'verification.json',{'inputs_verified':len(p['inputs_sha256']),**native,'bounded_proposals_verified':proposals,
        'numeric_outputs_unchanged_after_export_fix':len(export_fix['outputs_sha256']),
        'direct_weighted_moment_cells':cells,'bootstrap_valid_draws':valid,'notebook_cells':len(nb.cells),
        'executed_code_cells':len(code),'embedded_PNG':images,'table_rows':tables,'notebook_sha256':sha(nbpath),
        'code_sha256':sha(__file__),'new_independent_mothers':0,'default_changed':False,'official_PID_changed':False})
    print('Spectral candidate provenance, full coverage, raw ADC, statistics and notebook verified.',flush=True)


def archive():
    assert (OUT/'verification.json').exists() and not (OUT/'record.json').exists()
    p=read(OUT/'protocol.json');names=set(p['code_sha256'])|set(p['inherited_code_sha256'])
    names.update(['simulation/analyze_spectral_split.py','simulation/report_spectral_split.py','simulation/verify_spectral_split.py',
                  'simulation/test_spectral_split.py','jupyter/codex_0001_build_spectral_split_notebook.py'])
    for module in list(sys.modules.values()):
        name=getattr(module,'__file__',None)
        if name:
            path=Path(name).resolve()
            if path.is_relative_to(PROJECT) and path.suffix=='.py':names.add(relative(path))
    for directory in ['simulation/include','simulation/vendor']:
        for path in (PROJECT/directory).glob('*.h*'):names.add(relative(path))
    sources={}
    for name in sorted(names):
        path=OUT/'source_snapshot'/name;path.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(PROJECT/name,path);sources[name]=sha(path)
        assert sources[name]==sha(PROJECT/name)
    outputs={relative(path):sha(path) for path in sorted(OUT.rglob('*')) if path.is_file() and 'source_snapshot' not in path.parts and path.name!='record.json'}
    nb=PROJECT/'jupyter/codex_0001_csi_spectral_variance_split.ipynb'
    save(OUT/'record.json',{'outputs_sha256':outputs,'sources_snapshot_sha256':sources,'notebook':relative(nb),'notebook_sha256':sha(nb),
        'protocol_sha256':sha(OUT/'protocol.json'),'new_independent_mothers':0,'default_changed':False,'official_PID_changed':False})
    print('Archived',len(outputs),'outputs and',len(sources),'source snapshots.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--archive',action='store_true');args=parser.parse_args()
    archive() if args.archive else verify()
