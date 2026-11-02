"""Verify complete-pool coverage, native ADC identity and paired inference."""
import argparse
import json
import shutil
import sys
from pathlib import Path
import ROOT
import nbformat
from full_selection_common import *


def native_closure(p):
    rp=read(REPLAY/'replay_protocol.json');total=0;parent_events=0;clipped=[]
    archived=read(REPLAY/'record.json')['outputs_sha256']
    inputs={relative(path):sha(path) for path in [REPLAY/'record.json',REPLAY/'replay_protocol.json',OUT/'protocol.json',BUILD/'replay_full_pool']}
    for key,e in p['entries'].items():
        fullpath=PROJECT/e['output'];f=ROOT.TFile.Open(str(fullpath));tree=f.Get('Readout')
        info=json.loads(str(f.Get('RunInfo').GetString().Data()))
        assert info['completed'] and info['events']==e['events'] and tree.GetEntries()==e['events']
        assert info['new_independent_mothers']==0 and info['seed']==e['seed']
        assert info['plan']==read(PROJECT/e['plan']);parent_events+=e['events'];clipped.append(info['clipped_samples'])
        record=read(OUT/'logs'/f'{key}_native.json');assert record['output_sha256']==sha(fullpath)
        inputs[relative(fullpath)]=record['output_sha256']
        assert record['binary_sha256']==sha(BUILD/'replay_full_pool')
        for fold in [0,1]:
            plan=rp['plans'][f'{key}_eval{fold}'];jobs={j['key']:j for j in plan['jobs']}
            picks={}
            for variant,config in [('current_r0',info['configs'][fold]),('zero_cv_r0',info['zero_configs'][fold])]:
                # Older zero-CV files annotate the nonphysical status string.
                # Every numeric/physics/readout field must still match exactly.
                signature=lambda c:{k:v for k,v in c.items() if k!='status'}
                picks[variant]=next(job for job in plan['jobs'] if job['noise_replica']==0
                    and job['noise_scope']=='all_readout' and signature(read(job['config']))==signature(config))
            for case,variant in [('current_all','current_r1'),('zero_cv_all','zero_cv_r1')]:
                part=next(x for x in rp['groups'][e['source']]['logical_cases'][case] if x['batch']==e['batch'] and x['fold']==fold)
                picks[variant]=jobs[part['keys'][0]]
            for variant,job in picks.items():
                old_path=Path(job['output']);old_relative=relative(old_path)
                assert sha(old_path)==archived[old_relative]
                inputs[old_relative]=archived[old_relative]
                other=ROOT.TFile.Open(job['output']);old=other.Get('Readout')
                old.SetBranchStatus('*',0)
                for name in ['event_id','adc']:old.SetBranchStatus(name,1)
                tree.SetBranchStatus('*',0)
                for name in ['event_id','fold','adc_'+variant]:tree.SetBranchStatus(name,1)
                seen=[]
                for row in old:
                    id_=int(row.event_id);assert tree.GetEntry(id_)>0 and tree.event_id==id_ and tree.fold==fold
                    np.testing.assert_array_equal(np.asarray(getattr(tree,'adc_'+variant)),np.asarray(row.adc))
                    seen.append(id_);total+=1
                np.testing.assert_array_equal(seen,plan['ids']);other.Close()
        f.Close();print('Native selected-ADC identity verified:',key,flush=True)
    assert parent_events==256000 and total==4*3661
    return {'complete_parent_events':parent_events,'old_cohort_readouts_bit_identical':total,
            'ADC_samples_bit_identical':total*2496,'clipped_samples_by_batch':clipped,'inputs_sha256':inputs}


def verify():
    p=read(OUT/'protocol.json');analysis=read(OUT/'analysis.json')
    ap=read(OUT/'analysis_protocol.json')
    assert ap['code_sha256']==sha(PROJECT/'simulation/analyze_full_selection.py')
    assert ap['protocol_sha256']==sha(OUT/'protocol.json')
    for path,h in ap['inputs_sha256'].items():assert sha(PROJECT/path)==h,path
    for path,h in {**p['inputs_sha256'],**p['code_sha256'],**p['inherited_code_sha256']}.items():assert sha(PROJECT/path)==h,path
    for key,e in p['entries'].items():
        assert sha(PROJECT/e['plan'])==e['plan_sha256']
        report=read(OUT/'logs'/f'{key}_extraction.json');old=load(PARENT/'transport/arrays'/f'{key}.npz')
        assert report['original_selection_closure']
        for v in ['original',*VARIANTS]:
            path=OUT/'arrays'/f'{key}_{v}.npz';a=load(path);r=report['variants'][v]
            assert sha(path)==r['array_sha256'] and len(a['ids'])==e['events']
            np.testing.assert_array_equal(a['ids'],np.arange(e['events']))
            bins=charge_bin(a['qcalc'],p)
            np.testing.assert_array_equal(a['common'],a['selected']&(bins>=1)&(bins<=6)&np.isfinite(a['qcalc']))
            np.testing.assert_array_equal(a['supported'],a['common']&np.isin(bins,np.asarray(p['support_bins'][e['source']])+1))
            take=a['common'];q=np.rint(a['mean_wave'][take].sum(1)*160)/8
            np.testing.assert_array_equal(q,a['qcalc'][take])
            assert np.isfinite(a['X'][take]).all()
            if v=='original':
                np.testing.assert_array_equal(a['qcalc'],old['all_q']);np.testing.assert_array_equal(a['selected'],old['all_selected']);np.testing.assert_array_equal(a['common'],old['all_common']);np.testing.assert_array_equal(a['energy'],old['all_energy'])
    direct=0;valid={}
    for source,g in analysis.items():
        counts=load(OUT/'evaluation'/f'{source}_counts.npz');run=dict(SOURCES)[source];context=g['context']
        old=load(REPLAY/'point_evaluation'/f'{context}_bootstrap.npz')
        ix=np.searchsorted(counts['ids'],counts['old_ids'])
        np.testing.assert_array_equal(counts['MC_counts'][:,ix],old['MC_counts']);np.testing.assert_array_equal(counts['data_counts'],old['data_counts'])
        for batch in range(4):
            take=(counts['ids']>=batch*EVENTS[source])&(counts['ids']<(batch+1)*EVENTS[source])
            assert np.all(counts['MC_counts'][:,take].sum(1)<=EVENTS[source])
        ref=read(SCALES/'references'/f'{context}.json');draw=load(OUT/'evaluation'/f'{source}_bootstrap.npz')['contrasts']
        valid[source]={}
        for j,c in enumerate(CONDITIONS):
            valid[source][c]={}
            for k,v in enumerate(VARIANTS):
                a=load(OUT/'evaluation'/f'{source}_{v}.npz');pack=prepare_pack(a,p,source,ref,c)
                summary=g['conditions'][c]['summaries'][v]
                # Independently accumulate first/second raw weighted moments;
                # avoid the centered-residual covariance implementation.
                covs=[];means=[];profiles=[]
                for b,ids,halves,target in pack['cells']:
                    count=np.bincount(halves,minlength=2);w=target[halves]/count[halves]
                    x=pack['Y'][ids];m=np.sum(w[:,None]*x,axis=0)
                    raw=(x*w[:,None]).T@x;cov=raw-np.outer(m,m)
                    covs.append(cov);means.append(m);profiles.append(np.sum(w[:,None]*pack['mean_wave'][ids],axis=0));direct+=1
                np.testing.assert_allclose(np.mean(covs,0),summary['cov'],rtol=2e-9,atol=2e-10)
                np.testing.assert_allclose(np.mean(means,0),summary['mean'],rtol=1e-12,atol=1e-11)
                np.testing.assert_allclose(np.mean(profiles,0),summary['profile'],rtol=1e-12,atol=1e-10)
                expected={n:interval(draw[:,j,k,m]) for m,n in enumerate(METRICS)}
                assert expected==g['conditions'][c]['variants'][v]['CI95']
                valid[source][c][v]=int(np.isfinite(draw[:,j,k]).all(1).sum())
    cached=OUT/'native_compatibility.json'
    if not cached.exists():save(cached,native_closure(p))
    n=read(cached)
    for path,h in n['inputs_sha256'].items():assert sha(PROJECT/path)==h,path
    n={k:v for k,v in n.items() if k!='inputs_sha256'}
    nbpath=PROJECT/'jupyter/codex_0001_csi_full_pool_selection.ipynb';nb=nbformat.read(nbpath,as_version=4);nbformat.validate(nb)
    code=[c for c in nb.cells if c.cell_type=='code'];assert all(c.execution_count is not None for c in code)
    assert all(o.output_type!='error' for c in code for o in c.outputs)
    images=sum('image/png' in o.get('data',{}) for c in code for o in c.outputs);assert images==5
    import csv
    tables={name:len(list(csv.DictReader((OUT/name).open()))) for name in ['comparison_table.csv','selection_table.csv']}
    assert tables=={'comparison_table.csv':32,'selection_table.csv':8}
    save(OUT/'verification.json',{'inputs_verified':len(p['inputs_sha256']),**n,
        'direct_weighted_moment_cells':direct,'bootstrap_valid_draws':valid,
        'notebook_cells':len(nb.cells),'executed_code_cells':len(code),'embedded_PNG':images,'table_rows':tables,
        'notebook_sha256':sha(nbpath),'code_sha256':sha(__file__),'new_independent_mothers':0,
        'default_changed':False,'official_PID_changed':False})
    print('Complete-pool coverage, ADC, statistics, bootstrap and notebook verification passed.',flush=True)


def archive():
    assert (OUT/'verification.json').exists() and not (OUT/'record.json').exists()
    p=read(OUT/'protocol.json');code=set(p['code_sha256'])|set(p['inherited_code_sha256'])
    code.update(['simulation/analyze_full_selection.py','simulation/report_full_selection.py',
        'simulation/verify_full_selection.py','jupyter/codex_0001_build_full_selection_notebook.py'])
    for module in list(sys.modules.values()):
        path=getattr(module,'__file__',None)
        if path:
            from pathlib import Path
            path=Path(path).resolve()
            if path.is_relative_to(PROJECT) and path.suffix=='.py':code.add(relative(path))
    for directory in ['simulation/include','simulation/vendor']:
        for path in (PROJECT/directory).glob('*.h*'):code.add(relative(path))
    snapshots={}
    for name in sorted(code):
        path=OUT/'source_snapshot'/name;path.parent.mkdir(exist_ok=True,parents=True);shutil.copyfile(PROJECT/name,path)
        snapshots[name]=sha(path);assert snapshots[name]==sha(PROJECT/name)
    outputs={relative(path):sha(path) for path in sorted(OUT.rglob('*')) if path.is_file() and 'source_snapshot' not in path.parts and path.name!='record.json'}
    nbpath=PROJECT/'jupyter/codex_0001_csi_full_pool_selection.ipynb'
    save(OUT/'record.json',{'protocol_sha256':sha(OUT/'protocol.json'),'outputs_sha256':outputs,
        'sources_snapshot_sha256':snapshots,'notebook':relative(nbpath),'notebook_sha256':sha(nbpath),
        'binary_sha256':sha(BUILD/'replay_full_pool'),'checkpoint_commit':p['checkpoint_commit'],
        'new_independent_mothers':0,'default_changed':False,'official_PID_changed':False})
    print('Archived',len(outputs),'outputs and',len(snapshots),'source snapshots.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group()
    group.add_argument('--archive',action='store_true');group.add_argument('--native-only',action='store_true');args=parser.parse_args()
    if args.native_only:
        assert not (OUT/'native_compatibility.json').exists()
        save(OUT/'native_compatibility.json',native_closure(read(OUT/'protocol.json')))
    elif args.archive:archive()
    else:verify()
