"""Paired full-pool entrance/exit effects with unchanged physical readouts."""
import argparse
import concurrent.futures
from full_selection_common import *


def pooled(source,run,p):
    parts={v:[] for v in ['original',*VARIANTS]};audit={v:[] for v in VARIANTS}
    for batch in range(4):
        key=f'b{batch:02d}_run{run}';base=load(OUT/'arrays'/f'{key}_original.npz')
        arrays={v:load(OUT/'arrays'/f'{key}_{v}.npz') for v in VARIANTS}
        union=base['old_common'].copy()
        for a in arrays.values():union|=a['common']
        ix=np.flatnonzero(union)
        for v,a in {'original':base,**arrays}.items():
            fields=['q','qcalc','X','mean_wave','fold','old_common','selected','supported','common','truth']
            parts[v].append({**{k:a[k][ix] for k in fields},'ids':ix+batch*EVENTS[source]})
            if v=='original':continue
            old=base['supported'];now=a['supported'];entrant=~old&now;exit_=old&~now
            oldbins=charge_bin(base['qcalc'],p);newbins=charge_bin(a['qcalc'],p)
            transition=np.bincount(2*old.astype(int)+now.astype(int),minlength=4).reshape(2,2)
            reasons={
                'old_quality_or_amplitude_or_time_failure':entrant&~base['selected'],
                'old_below_supported_Q':entrant&base['selected']&(base['qcalc']<p['q_edges'][0]),
                'old_above_supported_Q':entrant&base['selected']&(base['qcalc']>p['q_edges'][max(p['support_bins'][source])+1]),
            }
            assert sum(int(x.sum()) for x in reasons.values())==int(entrant.sum())
            matrix=np.zeros((8,8),int)
            take=np.isfinite(base['qcalc'])&np.isfinite(a['qcalc'])&(old|now)
            np.add.at(matrix,(oldbins[take],newbins[take]),1)
            audit[v].append({'batch':batch,'mothers':len(old),'old_common':int(base['common'].sum()),
                'current_common':int(a['common'].sum()),'old_supported':int(old.sum()),'current_supported':int(now.sum()),
                'supported_transition':transition,'enter_support':int(entrant.sum()),'leave_support':int(exit_.sum()),
                'new_beyond_old_common':int((a['supported']&~a['old_common']).sum()),
                'fold_reference_QC_disagreement':int(np.count_nonzero(a['selected']!=a['selected_fold'])),
                'entrance_reasons':{k:int(x.sum()) for k,x in reasons.items()},
                'exit_current_selection_failure':int((exit_&~a['selected']).sum()),
                'exit_current_charge_failure':int((exit_&a['selected']).sum()),
                'enter_truth':np.bincount(a['truth'][entrant],minlength=5),
                'leave_truth':np.bincount(a['truth'][exit_],minlength=5),
                'current_truth':np.bincount(a['truth'][now],minlength=5),
                'migration_union_support':matrix})
    arrays={v:{k:np.concatenate([a[k] for a in blocks]) for k in blocks[0]} for v,blocks in parts.items()}
    return arrays,audit


def existing_closure(source,context,arrays):
    reports={};old_ids=None
    for v,label in [('current_r0','current'),('zero_cv_r0','zero_cv')]:
        old=load(REPLAY/'point_arrays'/f'{context}_{label}.npz');a=arrays[v]
        ix=np.searchsorted(a['ids'],old['ids']);np.testing.assert_array_equal(a['ids'][ix],old['ids'])
        for name in ['q','fold','mean_wave']:np.testing.assert_array_equal(a[name][ix],old[name])
        np.testing.assert_array_equal(a['X'][ix],old['X'][:,:6]);old_ids=old['ids']
        reports[v]=len(ix)
    for v,label in [('current_r1','current_all'),('zero_cv_r1','zero_cv_all')]:
        old=load(REPLAY/'repeat_arrays'/f'{source}_{label}.npz');a=arrays[v]
        ix=np.searchsorted(a['ids'],old['ids']);np.testing.assert_array_equal(a['ids'][ix],old['ids'])
        np.testing.assert_array_equal(a['X'][ix],old['X'][:,0,:6])
        np.testing.assert_array_equal(a['mean_wave'][ix],old['mean_wave'][:,0]);reports[v]=len(ix)
    return old_ids,reports


def one(source):
    p=read(OUT/'protocol.json');run=dict(SOURCES)[source];context=f'run{run}_last20_B128'
    dest=OUT/'evaluation'/f'{source}.json';assert not dest.exists()
    arrays,audit=pooled(source,run,p);old_ids,closure=existing_closure(source,context,arrays)
    ids=arrays['original']['ids'];data=load(REPLAY/'point_arrays'/f'{context}_data.npz')
    ref=read(SCALES/'references'/f'{context}.json')
    expected_ref=serialize(reference(data,p,source));assert ref==expected_ref
    dp=cohort(data,p,source,ref,'original_Q');ds=statistics(dp,keep_Q=True)
    packs={v:{c:prepare_pack(a,p,source,ref,c) for c in CONDITIONS} for v,a in arrays.items() if v!='original'}
    stats={c:{v:statistics(packs[v][c],keep_Q=True) for v in VARIANTS} for c in CONDITIONS}
    point=np.array([[contrasts(stats[c][v],ds) for v in VARIANTS] for c in CONDITIONS])
    inherited=read(SCALES/'point_evaluation'/f'{context}.json')
    for j,c in enumerate(['original_Q','current_Q']):
        for k,v in enumerate(['current','zero_cv']):
            prior=inherited['conditions'][c]['variants'][v]['point']
            np.testing.assert_allclose(point[j,k],[prior[n] for n in METRICS],rtol=1e-11,atol=1e-12)
    old=load(REPLAY/'point_evaluation'/f'{context}_bootstrap.npz')
    counts=extend_mother_counts(ids,old_ids,old['MC_counts'],source,p['extension_seed']+run)
    np.savez_compressed(OUT/'evaluation'/f'{source}_counts.npz',ids=ids,MC_counts=counts,data_counts=old['data_counts'],old_ids=old_ids)
    draw=np.full((512,len(CONDITIONS),len(VARIANTS),len(METRICS)),np.nan)
    for i in range(512):
        d=statistics(dp,old['data_counts'][i])
        for j,c in enumerate(CONDITIONS):
            for k,v in enumerate(VARIANTS):draw[i,j,k]=contrasts(statistics(packs[v][c],counts[i]),d)
        if (i+1)%128==0:print(source,'full-pool draws',i+1,flush=True)
    inherited_draw=load(SCALES/'point_evaluation'/f'{context}_bootstrap.npz')['contrasts']
    np.testing.assert_allclose(draw[:,0,:2],inherited_draw[:,0],rtol=1e-9,atol=1e-10)
    np.testing.assert_allclose(draw[:,1,:2],inherited_draw[:,2],rtol=1e-9,atol=1e-10)
    np.savez_compressed(OUT/'evaluation'/f'{source}_bootstrap.npz',contrasts=draw)
    report={'source':source,'context':context,'mothers':4*EVENTS[source],'union_mothers':len(ids),
        'inherited_feature_records_identical':closure,'inherited_bootstrap_draws_identical':512,
        'protocol_sha256':sha(OUT/'protocol.json'),'selection':audit,'data_summary':ds,'conditions':{}}
    for j,c in enumerate(CONDITIONS):
        report['conditions'][c]={'summaries':stats[c],'variants':{}}
        for k,v in enumerate(VARIANTS):
            row={'point':dict(zip(METRICS,point[j,k])),
                'CI95':{n:interval(draw[:,j,k,m]) for m,n in enumerate(METRICS)},'changes':{}}
            for reference_condition,index in [('old_cohort_current_Q',1),('old_cohort_current_selection',2)]:
                row['changes'][reference_condition]={'point':dict(zip(METRICS,point[j,k]-point[index,k])),
                    'CI95':{n:interval(draw[:,j,k,m]-draw[:,index,k,m]) for m,n in enumerate(METRICS)}}
            report['conditions'][c]['variants'][v]=row
    for v,a in arrays.items():np.savez_compressed(OUT/'evaluation'/f'{source}_{v}.npz',**a)
    save(dest,serialize(report));print(source,'complete selection statistics archived',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=2);args=parser.parse_args()
    p=read(OUT/'protocol.json');freeze=OUT/'analysis_protocol.json'
    if not freeze.exists():
        assert not list((OUT/'evaluation').glob('*'))
        expected={f'{key}_{v}.npz' for key in p['entries'] for v in ['original',*VARIANTS]}
        assert {path.name for path in (OUT/'arrays').glob('*.npz')}==expected,'Complete all batches before statistics'
        assert all((OUT/'logs'/f'{key}_extraction.json').exists() for key in p['entries'])
        inputs=list((OUT/'arrays').glob('*.npz'))+[REPLAY/'record.json']
        archived=read(REPLAY/'record.json')['outputs_sha256']
        for source,_ in SOURCES:
            for label in ['current_all','zero_cv_all']:
                path=REPLAY/'repeat_arrays'/f'{source}_{label}.npz'
                assert sha(path)==archived[relative(path)];inputs.append(path)
        save(freeze,{'code_sha256':sha(__file__),'protocol_sha256':sha(OUT/'protocol.json'),
            'metrics':METRICS,'bootstrap_replicates':512,'primary_scope':p['comparisons'],
            'inputs_sha256':{relative(path):sha(path) for path in sorted(inputs)}})
    f=read(freeze);assert f['code_sha256']==sha(__file__) and f['protocol_sha256']==sha(OUT/'protocol.json')
    for path,h in f['inputs_sha256'].items():assert sha(PROJECT/path)==h,path
    for path,h in {**p['code_sha256'],**p['inherited_code_sha256']}.items():assert sha(PROJECT/path)==h,path
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        for result in executor.map(one,[s for s,_ in SOURCES]):pass
    save(OUT/'analysis.json',{s:read(OUT/'evaluation'/f'{s}.json') for s,_ in SOURCES})
