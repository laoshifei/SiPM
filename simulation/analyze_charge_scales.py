"""Evaluate frozen charge conditions and multiscale moments with paired counts."""
import argparse
import concurrent.futures
from charge_scale_common import *


def pack(a,p,source,ref):return {c:cohort(a,p,source,ref,c) for c in CONDITIONS}


def checked_data(context):
    a=load(REPLAY/'point_arrays'/f'{context}_data.npz');a['qcalc']=charge_from_profile(a['mean_wave'])
    np.testing.assert_array_equal(a['qcalc'],a['q']);return a


def point_one(context):
    p=read(OUT/'protocol.json');desc=p['contexts'][context];source=desc['source']
    dest=OUT/'point_evaluation'/f'{context}.json';assert not dest.exists()
    data=checked_data(context);ref=reference(data,p,source);save(OUT/'references'/f'{context}.json',serialize(ref))
    arrays={'data':data,**{v:load(REPLAY/'point_arrays'/f'{context}_{v}.npz') for v in VARIANTS}}
    packs={v:pack(a,p,source,ref) for v,a in arrays.items()}
    stats={c:{v:statistics(packs[v][c],keep_Q=True) for v in arrays} for c in CONDITIONS}
    point=np.array([[contrasts(stats[c][v],stats[c]['data']) for v in VARIANTS] for c in CONDITIONS])
    point_lambda=np.array([interpolation(stats[c]['data'],stats[c]['current'],stats[c]['zero_cv']) for c in CONDITIONS])
    old=load(REPLAY/'point_evaluation'/f'{context}_bootstrap.npz');draw=np.full((512,3,2,len(METRICS)),np.nan)
    lambdas=np.full((512,3,len(LAMBDA_NAMES)),np.nan)
    for i in range(512):
        for j,c in enumerate(CONDITIONS):
            ds=statistics(packs['data'][c],old['data_counts'][i]);mc={v:statistics(packs[v][c],old['MC_counts'][i]) for v in VARIANTS}
            for k,v in enumerate(VARIANTS):draw[i,j,k]=contrasts(mc[v],ds)
            lambdas[i,j]=interpolation(ds,mc['current'],mc['zero_cv'])
    for j,k in INHERITED_PAIRS:np.testing.assert_allclose(draw[:,0,:,j],old['contrasts'][:,:,k],rtol=1e-8,atol=1e-10)
    # Data have not changed any membership, scale reference, or waveform.
    for c in CONDITIONS[1:]:
        for key in ['mean','cov','width2','profile']:np.testing.assert_array_equal(stats[c]['data'][key],stats['original_Q']['data'][key])
    bp=OUT/'point_evaluation'/f'{context}_bootstrap.npz';np.savez_compressed(bp,contrasts=draw,interpolation=lambdas)
    report={'context':context,'source':source,'description':desc,'protocol_sha256':sha(OUT/'protocol.json'),
        'reference_sha256':sha(OUT/'references'/f'{context}.json'),'bootstrap_sha256':sha(bp),
        'migration':{v:migration(a,p,source) for v,a in arrays.items()},'conditions':{}}
    for j,c in enumerate(CONDITIONS):
        report['conditions'][c]={'summaries':stats[c],
            'interpolation':{'point':dict(zip(LAMBDA_NAMES,point_lambda[j])),'CI95':intervals(lambdas[:,j],LAMBDA_NAMES)},'variants':{}}
        for k,v in enumerate(VARIANTS):
            report['conditions'][c]['variants'][v]={'point':dict(zip(METRICS,point[j,k])),
                'CI95':intervals(draw[:,j,k],METRICS),
                'change_from_original_Q':{'point':dict(zip(METRICS,point[j,k]-point[0,k])),
                    'CI95':intervals(draw[:,j,k]-draw[:,0,k],METRICS)}}
    save(dest,serialize(report));print('Charge and joint-scale point checks:',context,'all512 inherited draws match.',flush=True)


def repeat_inputs(source,p,ref):
    rp=read(REPLAY/'replay_protocol.json');group=rp['groups'][source];context=group['context']
    original=load(REPLAY/'point_arrays'/f'{context}_current.npz');result={};migrations={};validated=0
    for v,case in [('current','current_all'),('zero_cv','zero_cv_all')]:
        raw=load(REPLAY/'repeat_arrays'/f'{source}_{case}.npz');np.testing.assert_array_equal(raw['ids'],original['ids'])
        qcalc=charge_from_profile(raw['mean_wave']);assert raw['X'].shape[1]==16
        for part in group['logical_cases'][case]:
            take=(original['batch']==part['batch'])&(original['fold']==part['fold'])
            ids=original['ids'][take]-part['batch']*EVENTS[source]
            for r,key in enumerate(part['keys']):
                a=load(REPLAY/'arrays'/f'{key}.npz');np.testing.assert_array_equal(a['ids'],ids)
                np.testing.assert_array_equal(qcalc[take,r],a['qcalc'])
                np.testing.assert_array_equal(raw['X'][take,r],a['X']);validated+=len(ids)
        result[v]={c:[] for c in CONDITIONS};migrations[v]=[]
        for r in range(16):
            a={**original,'X':raw['X'][:,r],'mean_wave':raw['mean_wave'][:,r],'qcalc':qcalc[:,r]}
            migrations[v].append(migration(a,p,source))
            for c in CONDITIONS:result[v][c].append(cohort(a,p,source,ref,c))
    return result,migrations,validated


def repeat_one(source):
    p=read(OUT/'protocol.json');run=dict(SOURCES)[source];context=f'run{run}_last20_B128'
    dest=OUT/'repeat_evaluation'/f'{source}.json';assert not dest.exists()
    data=checked_data(context);ref=read(OUT/'references'/f'{context}.json');dp=pack(data,p,source,ref)
    repeats,migrations,validated=repeat_inputs(source,p,ref)
    ds={c:statistics(dp[c],keep_Q=True) for c in CONDITIONS}
    per_repeat={c:{v:[statistics(x,keep_Q=True) for x in repeats[v][c]] for v in VARIANTS} for c in CONDITIONS}
    ms={c:{v:average_statistics(per_repeat[c][v]) for v in VARIANTS} for c in CONDITIONS}
    point=np.array([[contrasts(ms[c][v],ds[c]) for v in VARIANTS] for c in CONDITIONS])
    pl=np.array([interpolation(ds[c],ms[c]['current'],ms[c]['zero_cv']) for c in CONDITIONS])
    prior=read(REPLAY/'repeat_analysis.json')[source]
    for v,case in [('current','current_all'),('zero_cv','zero_cv_all')]:
        old=prior['cases'][case]['summary'];np.testing.assert_allclose(ms['original_Q'][v]['cov'][:6,:6],np.array(old['one_readout_empirical_covariance'])[:6,:6],rtol=1e-8,atol=1e-9)
        np.testing.assert_allclose(ms['original_Q'][v]['mean'][:6],old['mean'][:6],rtol=1e-8,atol=1e-9)
    boot=load(REPLAY/'point_evaluation'/f'{context}_bootstrap.npz')
    draw=np.full((512,3,2,len(METRICS)),np.nan);lambdas=np.full((512,3,len(LAMBDA_NAMES)),np.nan)
    for i in range(512):
        for j,c in enumerate(CONDITIONS):
            d=statistics(dp[c],boot['data_counts'][i]);models={}
            for k,v in enumerate(VARIANTS):
                models[v]=average_statistics([statistics(x,boot['MC_counts'][i]) for x in repeats[v][c]])
                draw[i,j,k]=contrasts(models[v],d)
            lambdas[i,j]=interpolation(d,models['current'],models['zero_cv'])
        if (i+1)%128==0:print('Charge repeats',source,'draws',i+1,flush=True)
    bp=OUT/'repeat_evaluation'/f'{source}_bootstrap.npz';np.savez_compressed(bp,contrasts=draw,interpolation=lambdas)
    report={'source':source,'context':context,'protocol_sha256':sha(OUT/'protocol.json'),'bootstrap_sha256':sha(bp),
        'native_charge_entries_verified':validated,'migration':migrations,'conditions':{}}
    for j,c in enumerate(CONDITIONS):
        report['conditions'][c]={'data_summary':ds[c],'model_summaries':ms[c],
            'interpolation':{'point':dict(zip(LAMBDA_NAMES,pl[j])),'CI95':intervals(lambdas[:,j],LAMBDA_NAMES)},'variants':{}}
        for k,v in enumerate(VARIANTS):
            result={'point':dict(zip(METRICS,point[j,k])),'CI95':intervals(draw[:,j,k],METRICS),
                'change_from_original_Q':{'point':dict(zip(METRICS,point[j,k]-point[0,k])),
                    'CI95':intervals(draw[:,j,k]-draw[:,0,k],METRICS)},
                'per_repeat_contrasts':[contrasts(x,ds[c]) for x in per_repeat[c][v]],'repeat_sensitivity':{}}
            for label,part in [('first8',slice(0,8)),('last8',slice(8,16))]:
                s=average_statistics(per_repeat[c][v][part]);result['repeat_sensitivity'][label]={'summary':s,'point':dict(zip(METRICS,contrasts(s,ds[c])))}
            report['conditions'][c]['variants'][v]=result
    save(dest,serialize(report));print('Finished repeated current-charge diagnostics:',source,flush=True)


def main(stage,workers):
    p=read(OUT/'protocol.json')
    for name,h in {**p['code_sha256'],**p['inherited_sources_sha256']}.items():assert sha(PROJECT/name)==h,name
    if stage=='points':
        dest=OUT/'point_analysis.json';assert not dest.exists();items=list(p['contexts']);fn=point_one
    else:
        assert (OUT/'point_analysis.json').exists();dest=OUT/'repeat_analysis.json';assert not dest.exists();items=[s for s,_ in SOURCES];fn=repeat_one
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures=[executor.submit(fn,item) for item in items]
        for f in concurrent.futures.as_completed(futures):f.result()
    sub='point_evaluation' if stage=='points' else 'repeat_evaluation'
    save(dest,{key:read(OUT/sub/f'{key}.json') for key in items})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['points','repeats']);parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args();main(args.stage,args.workers)
