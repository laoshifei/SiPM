"""Audit bounded FIR changes, native controls, paired statistics and notebook."""
import argparse
import csv
import json
import shutil
import sys
import nbformat
import ROOT
from lowfreq_common import *
from reference import wave_features
from refine_waveforms import advanced_features


def verify():
    assert not (OUT/'verification.json').exists();p=read(OUT/'protocol.json');fit=read(OUT/'fit.json');a=read(OUT/'analysis.json')
    for name,h in {**p['inputs_sha256'],**p['code_sha256'],**p['inherited_code_sha256'],**p['binary_sha256']}.items():assert sha(PROJECT/name)==h,name
    resume=read(OUT/'resume_record.json')
    recovery=read(OUT/'interrupted_output_recovery.json')['relocated_incomplete_outputs']
    for name,h in resume['outputs_sha256'].items():
        path=recovery[name]['preserved_path'] if name in recovery else name
        assert sha(PROJECT/path)==h,name
    for ef,configs in p['configs'].items():
        for label,record in configs.items():
            new=read(PROJECT/record['path']);old=read(PROJECT/record['source_config']);assert sha(PROJECT/record['path'])==record['sha256']
            h,lost=tapered_coefficients(old['signal_dependent_noise']['coefficients'],record['corner_MHz'],record['dc_amplitude'])
            assert np.isclose(lost,record['outside_original_FIR_power_fraction'])
            for key in old:
                if key not in ['signal_dependent_noise','status']:assert new[key]==old[key]
            for key in old['signal_dependent_noise']:
                if key not in ['coefficients','variance_per_signal_adc']:assert new['signal_dependent_noise'][key]==old['signal_dependent_noise'][key]
            if label=='identity':assert new==old
            else:
                np.testing.assert_allclose(new['signal_dependent_noise']['coefficients'],h/np.linalg.norm(h),rtol=1e-13,atol=1e-15)
                assert np.isclose(new['signal_dependent_noise']['variance_per_signal_adc'],old['signal_dependent_noise']['variance_per_signal_adc']*(h@h))
    identity=jobs=readouts=0
    for name,e in p['grid_jobs'].items():
        record=read(OUT/'logs'/f'{name}.json');plan=read(PROJECT/e['path']);assert sha(PROJECT/e['path'])==e['sha256']
        identity+=record['identity_ADC_readouts']
        for job in plan['jobs']:
            r=record['jobs'][job['key']];assert sha(job['output'])==r['root_sha256']
            assert sha(OUT/'grid_arrays'/f"{job['key']}.npz")==r['array_sha256'];jobs+=1;readouts+=r['mothers']
    assert identity==2972 and jobs==160 and readouts==29720
    data,ref,dp=read_data(p);calibrations=0
    for ef,f in fit['folds'].items():
        ef_i=int(ef);train=np.asarray(f['training_ids']);test=np.asarray(f['evaluation_ids']);assert not np.intersect1d(train,test).size
        np.testing.assert_array_equal(train,data['ids'][data['fold']!=ef_i]);d=statistics(dp,(data['fold']!=ef_i).astype(int))
        for c in f['candidates']:
            values=[]
            for r in [0,1]:
                arrays=merge([load(OUT/'grid_arrays'/f"b{b:02d}_run12_eval{ef}_{c['candidate']}_r{r}.npz") for b in range(4)])
                assert np.all(arrays['fold']==ef_i);values.append(statistics(full_pack(arrays,p,SOURCE,ref)))
            np.testing.assert_allclose(objective(average(values),d)['total'],c['loss']['total'],rtol=1e-13,atol=1e-14);calibrations+=1
        assert f['chosen_index']==int(np.argmin([c['loss']['total'] for c in f['candidates']]))
    native=read(OUT/'native_protocol.json');assert native['fit_sha256']==sha(OUT/'fit.json') and native['binary_sha256']==sha(BUILD/'replay_spectral_pool')
    if (OUT/'full_pool_reuse.json').exists():
        reuse=read(OUT/'full_pool_reuse.json');assert all(f['candidate']=='identity' for f in fit['folds'].values())
        for path,h in reuse['reused_inputs_sha256'].items():assert sha(PROJECT/path)==h,path
        for path,r in reuse['copied_arrays'].items():assert sha(PROJECT/path)==r['sha256']==sha(PROJECT/r['source'])
    sampled=feature_checks=parents=legacy=0
    for key,e in native['entries'].items():
        log=read(OUT/'logs'/f'{key}_native.json');extraction=read(OUT/'logs'/f'{key}_extraction.json')
        assert sha(PROJECT/e['output'])==log['output_sha256'];assert sha(PROJECT/e['plan'])==e['plan_sha256']
        root=ROOT.TFile.Open(str(PROJECT/e['output']));info=json.loads(str(root.Get('RunInfo').GetString().Data()));tree=root.Get('Readout')
        assert info['completed'] and tree.GetEntries()==EVENTS and info['events']==EVENTS and info['plan']==read(PROJECT/e['plan'])
        assert info['legacy_control_readouts_bit_identical']==2*EVENTS and not any(info['clipped_samples']);parents+=EVENTS;legacy+=2*EVENTS
        for j,case in enumerate(info['plan']['cases']):
            for ef in [0,1]:assert info['configs'][j][ef]==read(case['configs'][ef])
        for r in [0,1]:
            path=OUT/'arrays'/f'{key}_split_r{r}.npz';raw=load(path);report=extraction['variants'][f'split_r{r}'];assert sha(path)==report['array_sha256']
            np.testing.assert_array_equal(raw['ids'],np.arange(EVENTS));np.testing.assert_array_equal(raw['fold'],info['plan']['folds'])
            baseline=float(np.median(raw['baseline_rms']));assert baseline==report['global_baseline_RMS']
            common=raw['selected']&(raw['qcalc']>=p['q_edges'][0])&(raw['qcalc']<=p['q_edges'][-1]);np.testing.assert_array_equal(common,raw['common'])
            rng=np.random.default_rng(731103500+e['batch']*10+r);boundary=np.concatenate([np.argsort(abs(raw['qcalc']-edge))[:8] for edge in p['q_edges']])
            ids=np.unique(np.r_[np.flatnonzero(common),rng.choice(EVENTS,128,replace=False),boundary]);tree.SetBranchStatus('*',0);tree.SetBranchStatus(f'adc_split_r{r}',1)
            for start in range(0,len(ids),512):
                ix=ids[start:start+512];waves=[]
                for id_ in ix:assert tree.GetEntry(int(id_))>0;waves.append(np.asarray(getattr(tree,f'adc_split_r{r}')).copy())
                waves=np.asarray(waves,np.uint16);f=wave_features(waves);ad=advanced_features(waves,f['t10_sample'],f['peak_sample'],f['amplitude'],f['baseline'],baseline)
                selected=(f['diagnostic_flags']==0)&~ad['pretrigger_flag_primary']&(f['amplitude']>=200)&(f['t10_sample']>=384)
                np.testing.assert_array_equal(selected,raw['selected'][ix]);np.testing.assert_array_equal(f['charge_long'],raw['qcalc'][ix]);sampled+=len(ix)
                use=common[ix]
                if np.any(use):
                    got=features(waves[use],ix[use],raw['qcalc'][ix[use]]);np.testing.assert_array_equal(np.c_[got['scores'],got['F']],raw['X'][ix[use]])
                    np.testing.assert_array_equal(got['mean_wave'],raw['mean_wave'][ix[use]]);feature_checks+=int(use.sum())
        root.Close();print('Verified complete-pool low-frequency ADC and features:',key,flush=True)
    assert parents==192000 and legacy==384000
    ap=read(OUT/'analysis_protocol.json');assert ap['code_sha256']==sha(PROJECT/'simulation/analyze_lowfreq.py')
    for name,h in ap['inputs_sha256'].items():assert sha(PROJECT/name)==h,name
    counts=load(OUT/'evaluation/Am241_counts.npz');old=load(SPLIT/'evaluation/Am241_counts.npz');ix=np.searchsorted(counts['ids'],old['ids'])
    np.testing.assert_array_equal(counts['MC_counts'][:,ix],old['MC_counts']);np.testing.assert_array_equal(counts['data_counts'],old['data_counts'])
    draws=load(OUT/'evaluation/Am241_bootstrap.npz')['contrasts'];old_draws=load(SPLIT/'evaluation/Am241_bootstrap.npz')['contrasts'];valid={};cells=0
    for r in [0,1]:np.testing.assert_allclose(draws[:,2*r],old_draws[:,[2,5][r]],rtol=1e-9,atol=1e-10)
    for j,v in enumerate(VARIANTS):
        pack=full_pack(load(OUT/'evaluation'/f'Am241_{v}.npz'),p,SOURCE,ref);means=[];covs=[]
        for _,indices,half,target in pack['cells']:
            weight=(target/np.bincount(half,minlength=2))[half];x=pack['Y'][indices];m=weight@x
            means.append(m);covs.append((x*weight[:,None]).T@x-np.outer(m,m));cells+=1
        np.testing.assert_allclose(np.mean(covs,0),a['summaries'][v]['cov'],rtol=2e-9,atol=2e-10)
        np.testing.assert_allclose(np.mean(means,0),a['summaries'][v]['mean'],rtol=1e-12,atol=1e-11)
        assert {n:interval(draws[:,j,k]) for k,n in enumerate(METRICS)}==a['variants'][v]['CI95'];valid[v]=int(np.isfinite(draws[:,j]).all(1).sum())
    nbpath=PROJECT/'jupyter/codex_0001_csi_low_frequency_noise.ipynb';nb=nbformat.read(nbpath,as_version=4);nbformat.validate(nb)
    code=[c for c in nb.cells if c.cell_type=='code'];assert all(c.execution_count is not None for c in code)
    assert all(o.output_type!='error' for c in code for o in c.outputs);images=sum('image/png' in o.get('data',{}) for c in code for o in c.outputs);assert images==5
    assert len(list(csv.DictReader((OUT/'grid_table.csv').open())))==20 and len(list(csv.DictReader((OUT/'comparison_table.csv').open())))==4
    decision=read(OUT/'decision.json');assert decision['code_sha256']==sha(PROJECT/'simulation/report_lowfreq.py')
    save(OUT/'verification.json',{'inputs_verified':len(p['inputs_sha256']),'preserved_pre_resume_files':len(resume['outputs_sha256']),
        'incomplete_arrays_preserved_and_reextracted':len(recovery),
        'grid_jobs':jobs,'grid_readouts':readouts,'grid_identity_ADC_readouts':identity,'joint_calibrations_recomputed':calibrations,
        'full_parent_events':parents,'legacy_control_ADC_readouts_identical':legacy,'candidate_ADC_QC_checks':sampled,
        'all_candidate_common_features_recomputed':feature_checks,'direct_weighted_moment_cells':cells,'bootstrap_valid_draws':valid,
        'notebook_cells':len(nb.cells),'executed_code_cells':len(code),'embedded_PNG':images,'notebook_sha256':sha(nbpath),'code_sha256':sha(__file__),
        'new_full_pool_readouts':0 if (OUT/'full_pool_reuse.json').exists() else 768000,
        'new_independent_mothers':0,'default_changed':False,'official_PID_changed':False})
    print('Low-frequency candidate provenance, native grid, full pool and notebook verified.',flush=True)


def archive():
    assert (OUT/'verification.json').exists() and not (OUT/'record.json').exists();p=read(OUT/'protocol.json')
    names=set(p['code_sha256'])|set(p['inherited_code_sha256'])
    names.update(['simulation/run_lowfreq_full.py','simulation/extract_lowfreq.py','simulation/reuse_lowfreq_full.py','simulation/analyze_lowfreq.py','simulation/report_lowfreq.py',
        'simulation/verify_lowfreq.py','simulation/README.md','jupyter/codex_0001_build_lowfreq_notebook.py','simulation/CMakeLists.txt',
        'simulation/replay_readout.cc','simulation/replay_spectral_pool.cc'])
    for directory,pattern in [('simulation/include','*.hh'),('simulation/src','*.cc'),('simulation/vendor','*.h*'),('simulation/vendor','*.cc')]:
        names.update(relative(x) for x in (PROJECT/directory).glob(pattern))
    for module in list(sys.modules.values()):
        name=getattr(module,'__file__',None)
        if name:
            path=Path(name).resolve()
            if path.is_relative_to(PROJECT) and path.suffix=='.py':names.add(relative(path))
    sources={}
    for name in sorted(names):
        path=OUT/'source_snapshot'/name;path.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(PROJECT/name,path);sources[name]=sha(path)
    outputs={relative(x):sha(x) for x in sorted(OUT.rglob('*')) if x.is_file() and 'source_snapshot' not in x.parts and x.name!='record.json'}
    nb=PROJECT/'jupyter/codex_0001_csi_low_frequency_noise.ipynb'
    save(OUT/'record.json',{'outputs_sha256':outputs,'sources_snapshot_sha256':sources,'notebook':relative(nb),'notebook_sha256':sha(nb),
        'new_independent_mothers':0,'default_changed':False,'official_PID_changed':False});print('Archived',len(outputs),'outputs and',len(sources),'source snapshots.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--archive',action='store_true');args=parser.parse_args();archive() if args.archive else verify()
