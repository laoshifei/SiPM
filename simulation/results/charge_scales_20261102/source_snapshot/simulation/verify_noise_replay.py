"""Independent seeds, ADC windows, covariance identities and provenance checks."""
import argparse
import base64
import csv
import platform
import shutil
from pathlib import Path
import nbformat
import ROOT
from noise_replay_common import *
from analyze_noise_replay import POINT_METRICS,NESTED_METRICS,losses

CHECKED={}
MAPPING=[(2,2),(3,3),(4,4),(8,5),(9,6),(10,7),(14,0),(17,1)]


def digest(path):
    path=Path(path);s=path.stat();stamp=(s.st_size,s.st_mtime_ns)
    if path not in CHECKED or CHECKED[path][0]!=stamp:CHECKED[path]=(stamp,sha(path))
    return CHECKED[path][1]


def hashes(items):
    for name,value in items.items():assert digest(PROJECT/name)==value,name


def source_paths():
    names=set(read(OUT/'selection_protocol.json')['code_sha256'])|set(read(OUT/'replay_protocol.json')['code_sha256'])
    # Include inherited Python implementations as used by this study, rather than
    # copying a prior C++ snapshot over the newly extended readout implementation.
    for name in read(QDC/'record.json')['sources_sha256']:
        if name.endswith('.py'):names.add(name)
    names.update(['simulation/analyze_noise_replay.py','simulation/report_noise_replay.py',
        'simulation/verify_noise_replay.py','simulation/README.md','simulation/vendor/SiPMResponse.cc',
        'simulation/vendor/SiPMResponse.hh','simulation/include/ReadoutSampling.hh',
        'simulation/include/ReadoutNoise.hh','simulation/include/SignalDependentNoise.hh',
        'jupyter/codex_0001_build_noise_replay_notebook.py'])
    return [PROJECT/n for n in sorted(names)]


def mix(x):
    mask=(1<<64)-1;x=(x+0x9e3779b97f4a7c15)&mask
    x=((x^(x>>30))*0xbf58476d1ce4e5b9)&mask
    x=((x^(x>>27))*0x94d049bb133111eb)&mask
    return x^(x>>31)


def event_seed(base,event):return mix(base^mix(event))


def independent_adc(wave,a,i):
    baseline=float(wave[:128].mean());np.testing.assert_equal(baseline,a['baseline'][i])
    y=wave-baseline;peak=128+np.argmax(y[128:]);before=np.flatnonzero(y[:peak]<=.1*y[peak])[-1]
    t10=before+(.1*y[peak]-y[before])/(y[before+1]-y[before]);np.testing.assert_allclose(t10,a['t10'][i],rtol=0,atol=1e-10)
    pre=int(np.ceil(t10-300));post=int(np.ceil(t10+250));mu=y[post:post+1000].mean()
    fs=[];fixed=[]
    for width in [25,51,101]:
        residual=y-np.convolve(y,np.ones(width)/width,mode='same')
        dv=residual[post:post+1000].var(ddof=1)-residual[pre:pre+200].var(ddof=1)
        fs.append(dv/mu);fixed.append(dv/a['mu_reference'][i])
    np.testing.assert_allclose(fs,a['X'][i,3:6],rtol=1e-9,atol=1e-9)
    np.testing.assert_allclose(fixed,a['X'][i,6:9],rtol=1e-9,atol=1e-9)
    anchor=int(np.floor(t10))+1;gate=y[anchor-20:anchor+1500];q=gate.sum()
    np.testing.assert_allclose(q,a['qcalc'][i],rtol=1e-12,atol=1e-8)
    np.testing.assert_allclose(gate.reshape(76,20).mean(1),a['mean_wave'][i],rtol=1e-12,atol=1e-9)
    np.testing.assert_allclose([gate[:270].sum()/q,-np.dot(gate,np.arange(-20,1500)*.002)/q],a['X'][i,:2],rtol=1e-10,atol=1e-10)


def verify_native(p):
    execution=read(OUT/'replay_execution.json');assert execution['protocol_sha256']==digest(OUT/'replay_protocol.json')
    total=files=seed_checks=windows=identity=0
    for name,plan in p['plans'].items():
        record=execution['plans'][name];assert record==read(OUT/'logs'/f'{name}_receipt.json')
        assert record['plan_sha256']==plan['sha256']==digest(PROJECT/plan['path'])
        hashes(record['outputs_sha256'])
        run=int(name.split('_')[1][3:]);batch=int(name.split('_')[0][1:])
        base=load(PARENT/'arrays'/f'b{batch:02d}_run{run}_frozen.npz')
        for job in plan['jobs']:
            rec=record['jobs'][job['key']];path=Path(job['output'])
            assert digest(path)==rec['root_sha256'] and digest(job['config'])==rec['config_sha256']
            f=ROOT.TFile.Open(str(path));assert f and not f.IsZombie()
            meta=__import__('json').loads(str(f.Get('RunInfo').GetString().Data()));t=f.Get('Readout')
            assert meta['completed'] and meta['new_independent_mothers']==0 and meta['clipped_samples']==0
            assert meta['config']==read(job['config']) and meta['ids']==plan['ids']
            assert t.GetEntries()==rec['entries']==len(plan['ids'])
            assert meta['noise_scope']==job['noise_scope'] and meta['noise_replica']==job['noise_replica']
            assert rec['unchanged_noiseless_signal'] and rec['clipped_samples']==0
            t.SetBranchStatus('*',0)
            for field in ['event_id','adc','baseline_readout_seed','signal_readout_seed']:t.SetBranchStatus(field,1)
            if job['native_identity']:
                assert meta['equal_parent_ADC']==len(plan['ids'])
                for row in t:
                    k=int(row.event_id);j=np.searchsorted(base['ids'],k);assert base['ids'][j]==k
                    np.testing.assert_array_equal(np.asarray(row.adc),base['waves'][j]);identity+=1
            else:
                ap=OUT/'arrays'/f"{job['key']}.npz";assert digest(ap)==rec['array_sha256'];a=load(ap)
                np.testing.assert_array_equal(a['ids'],plan['ids']);assert np.isfinite(a['X']).all()
                original=np.array([event_seed(int(meta['seed'])+31,int(k)) for k in a['ids']],dtype=np.uint64)
                replica=job['noise_replica'];scope=job['noise_scope']
                moved=np.array([event_seed(int(s),0x5245504c00000000+replica) for s in original],dtype=np.uint64) if replica else original
                np.testing.assert_array_equal(a['baseline_seed'],original if scope=='signal_only' else moved)
                np.testing.assert_array_equal(a['signal_seed'],original if scope=='electronics_only' else moved)
                seed_checks+=2*len(a['ids'])
                for i in [0,len(a['ids'])-1]:
                    t.GetEntry(i);assert int(t.event_id)==int(a['ids'][i])
                    assert int(t.baseline_readout_seed)==int(a['baseline_seed'][i])
                    assert int(t.signal_readout_seed)==int(a['signal_seed'][i])
                    independent_adc(np.asarray(t.adc,dtype=float),a,i);windows+=1
            f.Close();files+=1;total+=len(plan['ids'])
        print('Verified native seeds, ADC examples and hashes:',name,flush=True)
    assert total==p['physical_replay_entries']==execution['physical_replay_entries']==198819
    assert files==788 and identity==p['fixed_selected_mothers']==3661
    return {'physical_replay_entries':total,'physical_output_files':files,'native_identity_events':identity,
            'independent_seed_checks':seed_checks,'independent_ADC_examples':windows,'independent_highpass_windows':3*windows}


def intervals(saved,draw,names):
    for j,n in enumerate(names):assert saved[n]==interval(draw[:,j]),n


def verify_summaries(p):
    selection=read(OUT/'selection_analysis.json');points=read(OUT/'point_analysis.json');repeated=read(OUT/'repeat_analysis.json')
    inherited=0;flat_checks=0
    for group,bb in selection['groups'].items():
        for block,g in bb.items():
            key=f'{group}_B{block}';a=load(OUT/'selection/arrays'/f'{key}_data.npz')
            assert digest(OUT/'selection/arrays'/f'{key}_data.npz')==g['data_array_sha256']
            run=int(group.split('_')[0][3:]);np.testing.assert_array_equal(a['main_band'],band(a['X'],run))
            prior=load(QDC/'arrays'/f'{key}_data.npz');np.testing.assert_array_equal(a['ids'][a['main_band']],prior['ids'])
            bp=OUT/'selection/evaluation'/f'{key}_bootstrap.npz';assert digest(bp)==g['bootstrap_sha256'];boot=load(bp)['contrasts']
            old=load(QDC/'evaluation'/f'{key}_bootstrap.npz')['joint']
            for j,k in MAPPING:np.testing.assert_allclose(boot[:,:,1,j],old[:,:,k],rtol=1e-8,atol=1e-10);inherited+=512*3
            for iv,v in enumerate(VARIANTS):
                assert band(old_compact(load(QDC/'arrays'/f'{key}_{v}.npz'))['X'],run).all()
                for ic,case in enumerate(CASES):
                    r=g['comparisons'][v][case];intervals(r['CI95'],boot[:,iv,ic],METRICS)
                    intervals(r['change_CI95'],boot[:,iv,ic]-boot[:,iv,1],METRICS)
                np.testing.assert_array_equal(boot[:,iv,1],boot[:,iv,2]);np.testing.assert_array_equal(boot[:,iv,1],boot[:,iv,3])
    for context,g in points.items():
        assert g==read(OUT/'point_evaluation'/f'{context}.json');hashes(g['arrays_sha256'])
        bp=OUT/'point_evaluation'/f'{context}_bootstrap.npz';assert digest(bp)==g['bootstrap_sha256'];b=load(bp)['contrasts']
        old=load(QDC/'evaluation'/f'{context}_bootstrap.npz')['joint']
        for j,k in MAPPING:np.testing.assert_allclose(b[:,0,j],old[:,2,k],rtol=1e-8,atol=1e-10);inherited+=512
        for j,v in enumerate(['current','zero_cv']):intervals(g['variants'][v]['CI95'],b[:,j],POINT_METRICS)
        intervals(g['paired_change_zero_minus_current']['CI95'],b[:,1]-b[:,0],POINT_METRICS)
        intervals(g['loss_change_zero_minus_current']['CI95'],losses(b[:,1])-losses(b[:,0]),POINT_METRICS)
    for source,g in repeated.items():
        assert g==read(OUT/'repeat_evaluation'/f'{source}.json');hashes(g['arrays_sha256'])
        bp=OUT/'repeat_evaluation'/f'{source}_bootstrap.npz';assert digest(bp)==g['bootstrap_sha256'];b=load(bp)
        a=load(OUT/'point_arrays'/f"{g['context']}_current.npz")
        for case,r in g['cases'].items():
            X=load(OUT/'repeat_arrays'/f'{source}_{case}.npz')['X'];R=X.shape[1]
            assert R==p['repeat_cases'][case]['replicates'];intervals(r['CI95'],b[case],NESTED_METRICS)
            rr=rows(a,p,source)
            for (_,ix,w,_),saved in zip(rr,r['summary']['per_Q']):
                z=X[ix];mean=z.mean(1)
                S=np.array([np.cov(v.T,ddof=1) for v in z]);within=np.average(S,axis=0,weights=w)
                raw=np.cov(mean.T,aweights=w,ddof=0);correction=np.einsum('n,nij->ij',w*(1-w)/R,S)
                flat=np.cov(z.reshape(-1,len(FEATURES)).T,aweights=np.repeat(w/R,R),ddof=0)
                np.testing.assert_allclose(flat,raw+(R-1)/R*within,rtol=1e-8,atol=1e-9)
                np.testing.assert_allclose(flat,saved['flat_covariance'],rtol=1e-8,atol=1e-9)
                np.testing.assert_allclose(raw-correction,saved['between_corrected'],rtol=1e-8,atol=1e-9)
                Wone=np.einsum('n,nij->ij',w*(1-w),S)
                np.testing.assert_allclose(raw-correction+Wone,saved['one_readout_expected_covariance'],rtol=1e-8,atol=1e-9)
                flat_checks+=1
            r0=r['summary'];np.testing.assert_allclose(np.array(r0['between_corrected'])+np.array(r0['one_readout_within']),r0['one_readout_expected_covariance'],rtol=1e-8,atol=1e-9)
        diff=b['zero_minus_current_components'];np.testing.assert_allclose(diff[:,:,:2].sum(2),diff[:,:,2],rtol=1e-8,atol=1e-9)
        saved=g['zero_minus_current_components']
        for j,f in enumerate(FEATURES):intervals(saved['CI95'][f],diff[:,j],['within','between','total'])
    return {'parent_scalar_checks':inherited,'independent_covariance_Q_cells':flat_checks}


def verify_artifacts():
    points=read(OUT/'point_analysis.json');repeated=read(OUT/'repeat_analysis.json');selection=read(OUT/'selection_analysis.json')
    d=read(OUT/'decision.json');figs=read(OUT/'figure_record.json');hashes(d['inputs_sha256']);hashes(figs['inputs_sha256'])
    assert d['code_sha256']==figs['code_sha256']==digest(PROJECT/'simulation/report_noise_replay.py')
    regressions=[]
    for context,g in points.items():
        for n in d['guard_metrics']:
            r=g['loss_change_zero_minus_current'];c=r['CI95'][n]['CI95']
            if c is not None and c[0]>0:regressions.append({'context':context,'metric':n,'loss_change':r['point'][n],'CI95':c})
    assert regressions==d['clear_loss_regressions']
    benefits=[points[f'run{run}_last20_B128']['loss_change_zero_minus_current']['CI95']['F202_SD_ratio']['CI95'][1]<0 for _,run in SOURCES]
    assert benefits==d['primary_F202_benefits']
    assert d['candidate_guard_passed']==(not regressions and all(benefits))
    assert not d['default_changed'] and not d['official_PID_changed'] and d['new_independent_mothers']==0
    nbpath=PROJECT/'jupyter/codex_0001_csi_noise_replay.ipynb';nb=nbformat.read(nbpath,as_version=4);nbformat.validate(nb)
    code=[c for c in nb.cells if c.cell_type=='code'];assert all(c.execution_count for c in code)
    assert not any(o.output_type=='error' for c in code for o in c.outputs)
    blobs=[base64.b64decode(o.data['image/png']) for c in code for o in c.outputs if 'data' in o and 'image/png' in o.data]
    assert len(blobs)==len(figs['figures'])==6
    assert sorted(blobs)==sorted((OUT/'figures'/f'{n}.png').read_bytes() for n in figs['figures'])
    counts={}
    for name,expected in [('selection_table.csv',144),('zero_cv_point_table.csv',24),('zero_cv_loss_table.csv',264),('repeat_table.csv',8)]:
        with (OUT/name).open(newline='') as f:tab=list(csv.DictReader(f))
        assert len(tab)==expected;counts[name]=len(tab)
        for row in tab:
            if name=='zero_cv_loss_table.csv':
                r=points[row['context']]['loss_change_zero_minus_current'];n=row['metric']
                assert float(row['change'])==r['point'][n]
                assert [float(row['low']),float(row['high'])]==r['CI95'][n]['CI95'];continue
            if name=='selection_table.csv':r=selection['groups'][row['group']][row['block']]['comparisons'][row['variant']][row['case']];names=METRICS
            elif name=='zero_cv_point_table.csv':r=points[row['context']]['variants'][row['variant']];names=POINT_METRICS
            else:r=repeated[row['source']]['cases'][row['case']];names=NESTED_METRICS
            for n in names:
                assert (row[n]=='') if r['point'][n] is None else (float(row[n])==r['point'][n])
                ci=r['CI95'][n]['CI95'];assert int(row[n+'_valid'])==r['CI95'][n]['valid_draws']
                if ci is not None:assert [float(row[n+'_lo']),float(row[n+'_hi'])]==ci
    return {'table_rows':counts,'descriptive_regression_alerts':len(regressions),'candidate_guard_passed':d['candidate_guard_passed'],
            'notebook':{'cells':len(nb.cells),'executed_cells':len(code),'exact_embedded_pngs':len(blobs)}}


def verify():
    assert not (OUT/'verification.json').exists() and not (OUT/'record.json').exists()
    p=read(OUT/'replay_protocol.json');sp=read(OUT/'selection_protocol.json');ap=read(OUT/'analysis_protocol.json')
    hashes(p['inputs_sha256']);hashes(sp['inputs_sha256']);hashes(p['code_sha256']);hashes(sp['code_sha256']);hashes(p['binary_sha256'])
    hashes({k:v for k,v in read(QDC/'record.json')['sources_sha256'].items() if k.endswith('.py')})
    assert ap['code_sha256']==digest(PROJECT/'simulation/analyze_noise_replay.py')
    assert ap['common_sha256']==digest(PROJECT/'simulation/noise_replay_common.py')
    assert ap['replay_protocol_sha256']==digest(OUT/'replay_protocol.json')
    tests=read(OUT/'replay_tests.json');assert tests['ctest_passed']==6 and tests['algebra']['trials']==2000
    assert tests['ctest_sha256']==digest(BUILD/'Testing/Temporary/LastTest.log')
    assert tests['code_sha256']==digest(PROJECT/'simulation/test_noise_replay.py');hashes(tests['binary_sha256'])
    assert tests['algebra']['negative_between_correction_retained']<0
    api_entries=sum(2*x['mothers']*x['cases'] for x in tests['native_tests']);assert api_entries==240
    native=verify_native(p);summaries=verify_summaries(p);artifacts=verify_artifacts()
    nbpath=PROJECT/'jupyter/codex_0001_csi_noise_replay.ipynb'
    report={'scope':__doc__,**native,**summaries,**artifacts,'API_test_entries_separate':api_entries,
        'input_paths_checked':len(set(p['inputs_sha256'])|set(sp['inputs_sha256'])),'ctest_passed':6,
        'new_independent_mothers':0,'fixed_selected_mothers':3661,'default_changed':False,'official_PID_changed':False,
        'environment':{'python':platform.python_version(),'numpy':np.__version__,'ROOT':ROOT.gROOT.GetVersion(),'nbformat':nbformat.__version__},
        'outputs_sha256':{relative(f):digest(f) for f in sorted(OUT.rglob('*')) if f.is_file() and 'source_snapshot' not in f.parts},
        'sources_sha256':{relative(f):digest(f) for f in source_paths()}}
    report['outputs_sha256'][relative(nbpath)]=digest(nbpath)
    save(OUT/'verification.json',report);print('All noise-replay provenance, ADC and statistical checks passed.',flush=True)


def archive():
    assert not (OUT/'record.json').exists()
    v=read(OUT/'verification.json');hashes(v['outputs_sha256']);hashes(v['sources_sha256'])
    snapshots={}
    for name,value in v['sources_sha256'].items():
        dest=OUT/'source_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(PROJECT/name,dest);assert digest(dest)==value;snapshots[relative(dest)]=value
    outputs={**v['outputs_sha256'],**snapshots,relative(OUT/'verification.json'):digest(OUT/'verification.json')}
    save(OUT/'record.json',{'scope':__doc__,'outputs_sha256':outputs,'sources_sha256':v['sources_sha256'],
        'parent_record_sha256':digest(PARENT/'record.json'),'qdc_record_sha256':digest(QDC/'record.json'),
        'preceding_gaussian_record_sha256':digest(PROJECT/'simulation/results/gaussian_closure_20261102/record.json'),
        'fixed_selected_mothers':3661,'new_independent_mothers':0,'physical_replay_entries':198819,
        'API_test_entries_separate':240,'default_changed':False,'official_PID_changed':False})
    print('Archived',len(outputs),'outputs and',len(v['sources_sha256']),'source files.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--archive',action='store_true');args=parser.parse_args()
    archive() if args.archive else verify()
