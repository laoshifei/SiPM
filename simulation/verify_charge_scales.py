"""Verify exact current charge, explicit support and joint-scale statistics."""
import argparse
import base64
import csv
import platform
import shutil
import nbformat
from charge_scale_common import *
from test_charge_scales import run as algebra_tests


def hashes(items):
    for name,h in items.items():assert sha(PROJECT/name)==h,name


def direct(a,p,source,ref,condition,counts=None):
    qnew=np.rint(a['mean_wave'].sum(axis=1)*160)/8
    edges=np.asarray(p['q_edges']);newbin=np.searchsorted(edges,qnew,'right')-1
    newbin[qnew==edges[-1]]=5
    allowed=np.ones(len(qnew),bool) if condition=='original_Q' else np.isin(newbin,p['support_bins'][source])
    q=qnew if condition=='current_Q' else a['q'];results=[]
    for b in p['support_bins'][source]:
        use=allowed&(q>=edges[b])&((q<edges[b+1]) if b<5 else (q<=edges[b+1]))
        ix=np.flatnonzero(use)
        if counts is not None:ix=np.repeat(ix,counts[ix])
        h=(q[ix]>=p['conditional_cuts'][source][str(b)]['q_quartiles'][1]).astype(int)
        number=np.bincount(h,minlength=2);target=np.array(p['conditional_cuts'][source][str(b)]['q2_target'])
        if len(ix)<5 or np.any((target>0)&(number==0)):return None
        w=target[h]/number[h];scales=np.array([ref[str(f)][str(b)]['mean_F'] for f in a['fold'][ix]])
        z=a['X'][ix,3:6]/scales-1
        modes=np.c_[z.sum(1)/np.sqrt(3),(z[:,0]-z[:,2])/np.sqrt(2),(z[:,0]-2*z[:,1]+z[:,2])/np.sqrt(6)]
        Y=np.c_[a['X'][ix,:6],modes];cov=np.cov(Y.T,aweights=w,ddof=0)
        np.testing.assert_allclose(np.trace(cov[6:,6:]),np.trace(np.cov(z.T,aweights=w,ddof=0)),rtol=1e-9,atol=1e-12)
        quant=[]
        for j in range(3):
            order=np.argsort(Y[:,j]);xp=Y[order,j];wp=w[order];position=np.cumsum(wp)-wp/2
            quant.append(np.interp([.025,.25,.75,.975],(position-position[0])/(position[-1]-position[0]),xp))
        qq=np.asarray(quant)
        results.append({'mean':np.average(Y,axis=0,weights=w),'cov':cov,
            'width2':np.stack([qq[:,2]-qq[:,1],qq[:,3]-qq[:,0]])**2,
            'profile':np.average(a['mean_wave'][ix],axis=0,weights=w)})
    return {k:np.mean([r[k] for r in results],axis=0) for k in results[0]}


def compare(got,saved):
    if saved is None:assert got is None;return
    assert got is not None
    for k in got:np.testing.assert_allclose(got[k],saved[k],rtol=1e-8,atol=1e-9)


def check_interval(saved,draw,names):
    for j,n in enumerate(names):assert saved[n]==interval(draw[:,j]),n


def source_paths():
    p=read(OUT/'protocol.json');names=set(p['code_sha256'])|set(p['inherited_sources_sha256'])
    names.update(n for n in read(REPLAY/'record.json')['sources_sha256'] if n.endswith('.py'))
    names.update(['simulation/report_charge_scales.py','simulation/verify_charge_scales.py','simulation/README.md',
                  'jupyter/codex_0001_build_charge_scale_notebook.py'])
    return [PROJECT/n for n in sorted(names)]


def verify():
    assert not (OUT/'verification.json').exists() and not (OUT/'record.json').exists()
    p=read(OUT/'protocol.json');hashes(p['inputs_sha256']);hashes(p['code_sha256']);hashes(p['inherited_sources_sha256'])
    assert algebra_tests()==read(OUT/'algebra_tests.json')
    points=read(OUT/'point_analysis.json');repeats=read(OUT/'repeat_analysis.json');parent=read(REPLAY/'replay_protocol.json')
    inherited=independent=native_charge=reference_cells=0
    for context,g in points.items():
        assert g==read(OUT/'point_evaluation'/f'{context}.json');source=g['source']
        rp=OUT/'references'/f'{context}.json';assert sha(rp)==g['reference_sha256'];ref=read(rp)
        a={v:load(REPLAY/'point_arrays'/f'{context}_{v}.npz') for v in ['data',*VARIANTS]}
        np.testing.assert_array_equal(np.rint(a['data']['mean_wave'].sum(1)*160)/8,a['data']['q'])
        for fold in [0,1]:
            for b,r in ref[str(fold)].items():
                train=a['data'];ix=np.searchsorted(train['ids'],r['training_ids']);np.testing.assert_array_equal(train['ids'][ix],r['training_ids'])
                assert np.all(train['fold'][ix]!=fold)
                h=(train['q'][ix]>=p['conditional_cuts'][source][b]['q_quartiles'][1]).astype(int)
                number=np.bincount(h,minlength=2);w=np.array(p['conditional_cuts'][source][b]['q2_target'])[h]/number[h]
                np.testing.assert_allclose(np.average(train['X'][ix,3:6],axis=0,weights=w),r['mean_F'],rtol=1e-12,atol=1e-12);reference_cells+=1
        bp=OUT/'point_evaluation'/f'{context}_bootstrap.npz';assert sha(bp)==g['bootstrap_sha256'];boot=load(bp)
        old=load(REPLAY/'point_evaluation'/f'{context}_bootstrap.npz')
        for j,k in INHERITED_PAIRS:np.testing.assert_allclose(boot['contrasts'][:,0,:,j],old['contrasts'][:,:,k],rtol=1e-8,atol=1e-10);inherited+=512*2
        for ic,c in enumerate(CONDITIONS):
            row=g['conditions'][c];check_interval(row['interpolation']['CI95'],boot['interpolation'][:,ic],LAMBDA_NAMES)
            for v in ['data',*VARIANTS]:compare(direct(a[v],p,source,ref,c),row['summaries'][v]);independent+=1
            for iv,v in enumerate(VARIANTS):
                r=row['variants'][v];check_interval(r['CI95'],boot['contrasts'][:,ic,iv],METRICS)
                check_interval(r['change_from_original_Q']['CI95'],boot['contrasts'][:,ic,iv]-boot['contrasts'][:,0,iv],METRICS)
            for draw in [0,255,511]:
                ds=direct(a['data'],p,source,ref,c,old['data_counts'][draw])
                models={v:direct(a[v],p,source,ref,c,old['MC_counts'][draw]) for v in VARIANTS}
                for iv,v in enumerate(VARIANTS):np.testing.assert_allclose(contrasts(models[v],ds),boot['contrasts'][draw,ic,iv],rtol=1e-8,atol=1e-9,equal_nan=True)
                np.testing.assert_allclose(interpolation(ds,models['current'],models['zero_cv']),boot['interpolation'][draw,ic],rtol=1e-8,atol=1e-8,equal_nan=True);independent+=3
        for v,x in a.items():assert serialize(migration(x,p,source))==g['migration'][v]
        print('Verified independent charge/scale summaries:',context,flush=True)
    for source,g in repeats.items():
        assert g==read(OUT/'repeat_evaluation'/f'{source}.json');context=g['context']
        original=load(REPLAY/'point_arrays'/f'{context}_current.npz');data=load(REPLAY/'point_arrays'/f'{context}_data.npz')
        ref=read(OUT/'references'/f'{context}.json');arrays={};source_checks=0
        for v,case in [('current','current_all'),('zero_cv','zero_cv_all')]:
            raw=load(REPLAY/'repeat_arrays'/f'{source}_{case}.npz');np.testing.assert_array_equal(raw['ids'],original['ids'])
            q=np.rint(raw['mean_wave'].sum(2)*160)/8
            for part in parent['groups'][source]['logical_cases'][case]:
                use=(original['batch']==part['batch'])&(original['fold']==part['fold'])
                for j,key in enumerate(part['keys']):
                    child=load(REPLAY/'arrays'/f'{key}.npz')
                    np.testing.assert_array_equal(child['ids']+part['batch']*EVENTS[source],original['ids'][use])
                    np.testing.assert_array_equal(q[use,j],child['qcalc']);np.testing.assert_array_equal(raw['X'][use,j],child['X']);source_checks+=int(use.sum())
            arrays[v]=[{**original,'X':raw['X'][:,j],'mean_wave':raw['mean_wave'][:,j]} for j in range(16)]
            for j,x in enumerate(arrays[v]):assert serialize(migration(x,p,source))==g['migration'][v][j]
        assert source_checks==g['native_charge_entries_verified'];native_charge+=source_checks
        bp=OUT/'repeat_evaluation'/f'{source}_bootstrap.npz';assert sha(bp)==g['bootstrap_sha256'];boot=load(bp)
        old=load(REPLAY/'point_evaluation'/f'{context}_bootstrap.npz')
        for ic,c in enumerate(CONDITIONS):
            row=g['conditions'][c];check_interval(row['interpolation']['CI95'],boot['interpolation'][:,ic],LAMBDA_NAMES)
            ds=direct(data,p,source,ref,c);compare(ds,row['data_summary'])
            for iv,v in enumerate(VARIANTS):
                direct_repeats=[direct(x,p,source,ref,c) for x in arrays[v]];sm=average_statistics(direct_repeats)
                compare(sm,row['model_summaries'][v]);independent+=16
                r=row['variants'][v];check_interval(r['CI95'],boot['contrasts'][:,ic,iv],METRICS)
                check_interval(r['change_from_original_Q']['CI95'],boot['contrasts'][:,ic,iv]-boot['contrasts'][:,0,iv],METRICS)
                np.testing.assert_allclose([contrasts(s,ds) for s in direct_repeats],r['per_repeat_contrasts'],rtol=1e-8,atol=1e-9)
                for label,part in [('first8',slice(0,8)),('last8',slice(8,16))]:compare(average_statistics(direct_repeats[part]),r['repeat_sensitivity'][label]['summary'])
            for draw in [0,255,511]:
                ds=direct(data,p,source,ref,c,old['data_counts'][draw]);models={}
                for iv,v in enumerate(VARIANTS):
                    models[v]=average_statistics([direct(x,p,source,ref,c,old['MC_counts'][draw]) for x in arrays[v]])
                    np.testing.assert_allclose(contrasts(models[v],ds),boot['contrasts'][draw,ic,iv],rtol=1e-8,atol=1e-9,equal_nan=True);independent+=16
                np.testing.assert_allclose(interpolation(ds,models['current'],models['zero_cv']),boot['interpolation'][draw,ic],rtol=1e-8,atol=1e-8,equal_nan=True)
        print('Verified all native charges and independent repeated moments:',source,flush=True)
    assert native_charge==117152
    artifact=verify_artifacts(points,repeats)
    files={relative(f):sha(f) for f in sorted(OUT.rglob('*')) if f.is_file() and 'source_snapshot' not in f.parts}
    notebook=PROJECT/'jupyter/codex_0001_csi_charge_scale_constraints.ipynb';files[relative(notebook)]=sha(notebook)
    result={'scope':__doc__,'inputs_checked':len(p['inputs_sha256']),'parent_scalar_checks':inherited,
        'independent_cohort_covariances':independent,'reference_cells':reference_cells,'native_charge_entries_checked':native_charge,
        **artifact,'new_independent_mothers':0,'new_readout_replays':0,'default_changed':False,'official_PID_changed':False,
        'environment':{'python':platform.python_version(),'numpy':np.__version__,'nbformat':nbformat.__version__},
        'outputs_sha256':files,'sources_sha256':{relative(f):sha(f) for f in source_paths()}}
    save(OUT/'verification.json',result);print('All charge/scale checks passed.',flush=True)


def verify_artifacts(points,repeats):
    decision=read(OUT/'decision.json');figs=read(OUT/'figure_record.json');hashes(decision['inputs_sha256']);hashes(figs['inputs_sha256'])
    assert decision['code_sha256']==figs['code_sha256']==sha(PROJECT/'simulation/report_charge_scales.py')
    for source,g in repeats.items():
        c=g['conditions']['current_Q']['interpolation']['CI95']['lambda_F50_minus_F202']['CI95']
        assert c==decision['current_Q_lambda50_minus202_ranges'][source]
        assert decision['moment_line_incompatibility_flags'][source]==(c is not None and (c[0]>0 or c[1]<0))
    counts={}
    for name,expected in [('point_table.csv',72),('repeat_table.csv',12),('variance_line_table.csv',42),('migration_table.csv',100)]:
        with (OUT/name).open(newline='') as f:tab=list(csv.DictReader(f))
        assert len(tab)==expected;counts[name]=len(tab)
        for row in tab:
            if name=='migration_table.csv':
                saved=points[row['context']]['migration'][row['variant']] if row['stage']=='replica0' else repeats[row['context']]['migration'][row['variant']][int(row['replica'])-1]
                for k in ['old_support','current_support','leave_support','enter_support','bin_changed','underflow','overflow','total_original_selected']:assert int(row[k])==saved[k]
                continue
            if name=='variance_line_table.csv':collection=points if row['stage']=='replica0' else repeats;r=collection[row['context']]['conditions'][row['condition']]['interpolation'];names=LAMBDA_NAMES
            else:
                collection=points if name=='point_table.csv' else repeats;key=row['context'] if name=='point_table.csv' else row['source']
                r=collection[key]['conditions'][row['condition']]['variants'][row['variant']];names=METRICS
            for n in names:
                assert (row[n]=='') if r['point'][n] is None else float(row[n])==r['point'][n]
                assert int(row[n+'_valid'])==r['CI95'][n]['valid_draws']
                c=r['CI95'][n]['CI95']
                if c is not None:assert [float(row[n+'_lo']),float(row[n+'_hi'])]==c
    nb=nbformat.read(PROJECT/'jupyter/codex_0001_csi_charge_scale_constraints.ipynb',as_version=4);nbformat.validate(nb)
    code=[c for c in nb.cells if c.cell_type=='code'];assert all(c.execution_count for c in code)
    assert not any(o.output_type=='error' for c in code for o in c.outputs)
    blobs=[base64.b64decode(o.data['image/png']) for c in code for o in c.outputs if 'data' in o and 'image/png' in o.data]
    assert len(blobs)==len(figs['figures'])==6
    assert sorted(blobs)==sorted((OUT/'figures'/f'{n}.png').read_bytes() for n in figs['figures'])
    return {'table_rows':counts,'notebook':{'cells':len(nb.cells),'executed_cells':len(code),'exact_embedded_pngs':len(blobs)}}


def archive():
    assert not (OUT/'record.json').exists();v=read(OUT/'verification.json');hashes(v['outputs_sha256']);hashes(v['sources_sha256']);snapshots={}
    for name,h in v['sources_sha256'].items():
        dest=OUT/'source_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(PROJECT/name,dest);assert sha(dest)==h;snapshots[relative(dest)]=h
    save(OUT/'record.json',{'scope':__doc__,'parent_record_sha256':sha(REPLAY/'record.json'),
        'outputs_sha256':{**v['outputs_sha256'],**snapshots,relative(OUT/'verification.json'):sha(OUT/'verification.json')},
        'sources_sha256':v['sources_sha256'],'new_independent_mothers':0,'new_readout_replays':0,'default_changed':False,'official_PID_changed':False})
    print('Archived charge/scale outputs and',len(v['sources_sha256']),'source files.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--archive',action='store_true');args=parser.parse_args()
    archive() if args.archive else verify()
