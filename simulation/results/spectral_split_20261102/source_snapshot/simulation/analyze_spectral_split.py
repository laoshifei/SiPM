"""Evaluate frozen native spectral-state candidates on the complete mother pool."""
import argparse
import concurrent.futures
from spectral_split_common import *


def gather(source,run,p):
    previous=load(FULL/'evaluation'/f'{source}_counts.npz');parts={v:[] for v in VARIANTS};selection={v:[] for v in VARIANTS}
    for batch in range(4):
        key=f'b{batch:02d}_run{run}';arrays={v:load((OUT if v.startswith('split') else FULL)/'arrays'/f'{key}_{v}.npz') for v in VARIANTS}
        n=EVENTS[source];union=np.isin(np.arange(n)+batch*n,previous['ids'])
        for a in arrays.values():union|=a['common']
        ix=np.flatnonzero(union)
        for v,a in arrays.items():
            take=a['supported'];ref=arrays['current_r'+v[-1]]['supported']
            selection[v].append({'batch':batch,'mothers':n,'common':int(a['common'].sum()),'supported':int(take.sum()),
                'enter_vs_current':int((take&~ref).sum()),'leave_vs_current':int((ref&~take).sum()),
                'supported_truth':np.bincount(a['truth'][take],minlength=5).tolist()})
            fields=['q','qcalc','X','mean_wave','fold','selected','common','supported','old_common','truth']
            parts[v].append({**{k:a[k][ix] for k in fields},'ids':ix+batch*n})
    arrays={v:{k:np.concatenate([a[k] for a in blocks]) for k in blocks[0]} for v,blocks in parts.items()}
    return arrays,selection,previous


def guards(candidate,current,p):
    spec=p['joint_guard'];checks={}
    families={'F_SD_bounds':[f'F{s}_SD_ratio' for s in [50,102,202]],
        'F_mean_bounds':[f'F{s}_mean_ratio' for s in [50,102,202]],
        'PID_SD_bounds':[n+'_SD_ratio' for n in ['QDC','negative_mean_time_us','PCA']],
        'PID_width50_95_bounds':[f'{n}_width{w}_ratio' for w in [50,95] for n in ['QDC','negative_mean_time_us','PCA']]}
    for bound,names in families.items():
        lo,hi=spec[bound]
        for name in names:checks[name]=lo<=candidate[name]<=hi
    checks['mean_ADC_regression']=candidate['mean_ADC_relative_L2']<=current['mean_ADC_relative_L2']+spec['maximum_mean_ADC_L2_regression']
    checks['PID_covariance_regression']=candidate['PID_covariance_relative_error']<=current['PID_covariance_relative_error']+spec['maximum_PID_covariance_error_regression']
    checks['F_covariance_improved']=candidate['F_covariance_relative_error']<current['F_covariance_relative_error']
    checks={name:bool(value) for name,value in checks.items()}
    return {'passed':all(checks.values()),'checks':checks,'failed':[k for k,v in checks.items() if not v]}


def one(source):
    p=read(OUT/'protocol.json');run=dict(SOURCES)[source];context=f'run{run}_last20_B128';dest=OUT/'evaluation'/f'{source}.json'
    assert not dest.exists()
    arrays,selection,old=gather(source,run,p);ids=arrays['split_r0']['ids']
    data=load(REPLAY/'point_arrays'/f'{context}_data.npz');ref=read(SCALES/'references'/f'{context}.json')
    dp=cohort(data,p,source,ref,'original_Q');ds=statistics(dp,keep_Q=True)
    packs={v:full_pack(a,p,source,ref) for v,a in arrays.items()};ms={v:statistics(a,keep_Q=True) for v,a in packs.items()}
    point=np.array([contrasts(ms[v],ds) for v in VARIANTS]);previous=read(FULL/'analysis.json')[source]['conditions']['full_pool_current_selection']['variants']
    for k,v in enumerate(VARIANTS):
        if not v.startswith('split'):np.testing.assert_allclose(point[k],[previous[v]['point'][n] for n in METRICS],rtol=1e-11,atol=1e-12)
    counts=extend_mother_counts(ids,old['ids'],old['MC_counts'],source,p['extension_seed']+run)
    np.savez_compressed(OUT/'evaluation'/f'{source}_counts.npz',ids=ids,MC_counts=counts,data_counts=old['data_counts'],previous_ids=old['ids'])
    draw=np.full((512,len(VARIANTS),len(METRICS)),np.nan)
    for i in range(512):
        d=statistics(dp,old['data_counts'][i])
        for k,v in enumerate(VARIANTS):draw[i,k]=contrasts(statistics(packs[v],counts[i]),d)
        if (i+1)%128==0:print(source,'spectral full-pool paired draws',i+1,flush=True)
    old_draw=load(FULL/'evaluation'/f'{source}_bootstrap.npz')['contrasts'][:,3]
    for k,v in enumerate(['current_r0','zero_cv_r0','current_r1','zero_cv_r1']):
        np.testing.assert_allclose(draw[:,VARIANTS.index(v)],old_draw[:,k],rtol=1e-9,atol=1e-10)
    np.savez_compressed(OUT/'evaluation'/f'{source}_bootstrap.npz',contrasts=draw)
    report={'source':source,'context':context,'union_mothers':len(ids),'available_mothers':4*EVENTS[source],
        'selection':selection,'data_summary':ds,'summaries':ms,'variants':{},'fold_checks':{},
        'protocol_sha256':sha(OUT/'protocol.json'),'inherited_bootstrap_draws_identical':512}
    for k,v in enumerate(VARIANTS):
        baseline='current_r'+v[-1];index=VARIANTS.index(baseline)
        report['variants'][v]={'point':dict(zip(METRICS,point[k])),
            'CI95':{n:interval(draw[:,k,j]) for j,n in enumerate(METRICS)},
            'change_vs_current':{'point':dict(zip(METRICS,point[k]-point[index])),
                'CI95':{n:interval(draw[:,k,j]-draw[:,index,j]) for j,n in enumerate(METRICS)}}}
        np.savez_compressed(OUT/'evaluation'/f'{source}_{v}.npz',**arrays[v])
    for ef in [0,1]:
        d=statistics(dp,(data['fold']==ef).astype(int));report['fold_checks'][str(ef)]={}
        for v in VARIANTS:
            s=statistics(packs[v],(arrays[v]['fold']==ef).astype(int))
            report['fold_checks'][str(ef)][v]=dict(zip(METRICS,contrasts(s,d)))
    report['joint_guards']={str(r):guards(report['variants'][f'split_r{r}']['point'],report['variants'][f'current_r{r}']['point'],p) for r in [0,1]}
    save(dest,serialize(report));print(source,'spectral full-pool statistics complete',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=2);args=parser.parse_args();p=read(OUT/'protocol.json')
    expected={f'{k}_{v}.npz' for k in p['entries'] for v in ['split_r0','split_r1']}
    assert {path.name for path in (OUT/'arrays').glob('*.npz')}==expected
    freeze=OUT/'analysis_protocol.json'
    if not freeze.exists():
        assert not list((OUT/'evaluation').glob('*'))
        inputs=list((OUT/'arrays').glob('*.npz'))+[OUT/'fit.json',OUT/'native_protocol.json']
        save(freeze,{'code_sha256':sha(__file__),'protocol_sha256':sha(OUT/'protocol.json'),
            'inputs_sha256':{relative(path):sha(path) for path in inputs},'metrics':METRICS,'variants':VARIANTS})
    fp=read(freeze);assert fp['code_sha256']==sha(__file__)
    for name,h in {**p['code_sha256'],**p['inherited_code_sha256'],**fp['inputs_sha256']}.items():assert sha(PROJECT/name)==h,name
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        for result in executor.map(one,[s for s,_ in SOURCES]):pass
    save(OUT/'analysis.json',{s:read(OUT/'evaluation'/f'{s}.json') for s,_ in SOURCES})
