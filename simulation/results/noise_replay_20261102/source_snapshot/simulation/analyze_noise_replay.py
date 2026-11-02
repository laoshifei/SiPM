"""Joint zero-CV controls and finite-repeat conditional covariance on fixed mothers."""
import argparse
import concurrent.futures
from noise_replay_common import *
from publication_stats import block_counts

POINT_METRICS=METRICS+['mean_ADC_relative_L2','PID_covariance_relative_error']
NESTED_METRICS=[f'{f}_{metric}' for f in FEATURES for metric in
                ['predicted_SD_ratio','within_fraction','between_fraction','finite_repeat_correction_fraction']]


def paired_counts(data,mc,run,seg,B):
    old=load(RC/'arrays'/f'run{run}_{seg}_B{B}_frozen_evaluation.npz')
    rng=np.random.default_rng(731031900+run+100*SEGMENTS.index(seg)+B)
    d=[];m=[]
    for _ in range(512):
        d.append(block_counts(data['ids'],tuple(data['span']),B,rng))
        m.append(mother_counts(mc,rng,[EVENTS[dict((r,s) for s,r in SOURCES)[run]]]*4))
        mother_counts(old,rng,[int(old['n_parent'])])
    return np.asarray(d),np.asarray(m)


def curve(a,p,source,counts=None):
    rr=rows(a,p,source,counts)
    if rr is None:return np.full(a['mean_wave'].shape[1],np.nan)
    return np.mean([w@a['mean_wave'][ix] for _,ix,w,_ in rr],0)


def full_contrast(ms,ds,mp,dp):
    c=contrast(ms,ds)
    if ms is None or ds is None:return np.r_[c,np.nan,np.nan]
    sd=np.sqrt(np.diag(ds['cov'])[:3]);delta=(ms['cov'][:3,:3]-ds['cov'][:3,:3])/(sd[:,None]*sd[None,:])
    return np.r_[c,np.linalg.norm(mp-dp)/np.linalg.norm(dp),np.sqrt(np.mean(delta*delta))]


def losses(x):
    z=x.copy()
    z[...,:2]=np.abs(z[...,:2])
    with np.errstate(invalid='ignore',divide='ignore'):z[...,2:len(METRICS)]=np.abs(np.log(z[...,2:len(METRICS)]))
    return z


def point_one(context):
    p=read(OUT/'replay_protocol.json');d=p['point_contexts'][context];source=d['source'];run=d['run'];seg=d['segment'];B=d['block']
    dest=OUT/'point_evaluation'/f'{context}.json';assert not dest.exists()
    data=old_compact(load(QDC/'arrays'/f'{context}_data.npz'));raw=data_waves(run,seg);raw['fold']=(raw['ids']//B)%2
    data['mean_wave']=raw['mean_wave']
    current=old_compact(load(QDC/'arrays'/f'{context}_spectrum_CV_fit.npz'))
    prior_raw=pool([assemble(batch,run,seg,B,'spectrum_CV_fit',raw,p,source) for batch in BATCHES])
    np.testing.assert_array_equal(current['ids'],prior_raw['ids']);current['mean_wave']=prior_raw['mean_wave'];del prior_raw,raw
    zero={k:v.copy() for k,v in current.items()}
    replacements=0
    for part in d['parts']:
        use=(current['batch']==part['batch'])&(current['fold']==part['fold'])
        if part['key'] is None:continue
        a=load(OUT/'arrays'/f"{part['key']}.npz");ids=current['ids'][use]-part['batch']*EVENTS[source]
        ix=np.searchsorted(a['ids'],ids);np.testing.assert_array_equal(a['ids'][ix],ids)
        np.testing.assert_array_equal(a['q'][ix],current['q'][use])
        for field in ['X','mean_wave']:zero[field][use]=a[field][ix]
        replacements+=int(use.sum())
    arrays={'data':data,'current':current,'zero_cv':zero}
    sums={k:summary(a,p,source) for k,a in arrays.items()};curves={k:curve(a,p,source) for k,a in arrays.items()}
    point=np.array([full_contrast(sums[v],sums['data'],curves[v],curves['data']) for v in ['current','zero_cv']])
    dc,mc=paired_counts(data,current,run,seg,B);draws=np.full((512,2,len(POINT_METRICS)),np.nan)
    for i in range(512):
        ds=summary(data,p,source,dc[i]);dp=curve(data,p,source,dc[i])
        for j,v in enumerate(['current','zero_cv']):
            a=arrays[v];draws[i,j]=full_contrast(summary(a,p,source,mc[i]),ds,curve(a,p,source,mc[i]),dp)
    previous=load(QDC/'evaluation'/f'{context}_bootstrap.npz')['joint']
    for j,k in [(2,2),(3,3),(4,4),(8,5),(9,6),(10,7),(14,0),(17,1)]:
        np.testing.assert_allclose(draws[:,0,j],previous[:,2,k],rtol=1e-8,atol=1e-10)
    files={}
    for v,a in arrays.items():
        path=OUT/'point_arrays'/f'{context}_{v}.npz';np.savez_compressed(path,**a);files[relative(path)]=sha(path)
    bp=OUT/'point_evaluation'/f'{context}_bootstrap.npz';np.savez_compressed(bp,contrasts=draws,data_counts=dc,MC_counts=mc)
    ld=losses(point[1])-losses(point[0]);ldb=losses(draws[:,1])-losses(draws[:,0])
    result={'context':context,'source':source,'block':B,'segment':seg,'metrics':POINT_METRICS,
        'protocol_sha256':sha(OUT/'replay_protocol.json'),'arrays_sha256':files,'bootstrap_sha256':sha(bp),
        'summaries':sums,'mean_profiles':curves,'new_response_entries':replacements,'variants':{},
        'loss_change_zero_minus_current':{'point':dict(zip(POINT_METRICS,ld)),
                'CI95':{n:interval(ldb[:,j]) for j,n in enumerate(POINT_METRICS)}},
        'paired_change_zero_minus_current':{'point':dict(zip(POINT_METRICS,point[1]-point[0])),
                'CI95':{n:interval(draws[:,1,j]-draws[:,0,j]) for j,n in enumerate(POINT_METRICS)}}}
    for j,v in enumerate(['current','zero_cv']):
        result['variants'][v]={'point':dict(zip(POINT_METRICS,point[j])),
            'CI95':{n:interval(draws[:,j,k]) for k,n in enumerate(POINT_METRICS)}}
    save(dest,serialize(result));print('Joint zero-CV point control',context,'all512 paired draws verified.',flush=True)


def cache_moments(X):
    mean=X.mean(1);res=X-mean[:,None,:];R=X.shape[1]
    cov=np.einsum('nri,nrj->nij',res,res)/(R-1)
    return {'X':X,'mean':mean,'cov':cov,'R':R}


def cached_nested(a,cache,p,source,counts=None):
    rr=rows(a,p,source,counts)
    if rr is None:return None
    result=[];R=cache['R']
    for b,ix,w,_ in rr:
        unique,inverse=np.unique(ix,return_inverse=True);w=np.bincount(inverse,weights=w)
        m=cache['mean'][unique];S=cache['cov'][unique];mean=w@m;delta=m-mean
        Braw=delta.T@(w[:,None]*delta);W=np.einsum('n,nij->ij',w,S)
        correction=np.einsum('n,nij->ij',w*(1-w)/R,S);B=Braw-correction
        Wone=np.einsum('n,nij->ij',w*(1-w),S)
        repeat_means=np.einsum('n,nri->ri',w,cache['X'][unique])
        second=np.einsum('n,ni,nj->ij',w,m,m)+(R-1)/R*W
        one=second-repeat_means.T@repeat_means/R
        result.append({'Q_bin':b,'mean':mean,'within':W,'between_raw':Braw,'between_correction':correction,
            'between_corrected':B,'mixture_covariance':B+W,'one_readout_within':Wone,
            'one_readout_expected_covariance':B+Wone,'one_readout_empirical_covariance':one,
            'flat_covariance':Braw+(R-1)/R*W,'repeat_count':R,'mothers':len(unique)})
    keys=['mean','within','between_raw','between_correction','between_corrected','mixture_covariance',
          'one_readout_within','one_readout_expected_covariance','one_readout_empirical_covariance','flat_covariance']
    return {**{k:np.mean([r[k] for r in result],0) for k in keys},'per_Q':result}


def nested_vector(s,ds):
    if s is None or ds is None:return np.full(len(NESTED_METRICS),np.nan)
    total=np.diag(s['one_readout_expected_covariance']);data=np.diag(ds['cov'])
    with np.errstate(invalid='ignore',divide='ignore'):
        return np.stack([np.sqrt(total/data),np.diag(s['one_readout_within'])/total,
            np.diag(s['between_corrected'])/total,np.diag(s['between_correction'])/total],1).ravel()


def repeat_one(source):
    p=read(OUT/'replay_protocol.json');g=p['groups'][source];context=g['context']
    a=load(OUT/'point_arrays'/f'{context}_current.npz');data=load(OUT/'point_arrays'/f'{context}_data.npz')
    identity=load(OUT/'point_arrays'/f'{context}_zero_cv.npz')
    boot=load(OUT/'point_evaluation'/f'{context}_bootstrap.npz');ds=summary(data,p,source)
    cases={};caches={};records={};files={};datacurve=curve(data,p,source)
    for case,parts in g['logical_cases'].items():
        R=p['repeat_cases'][case]['replicates'];X=np.full((len(a['ids']),R,len(FEATURES)),np.nan)
        profiles=np.full((len(a['ids']),R,a['mean_wave'].shape[1]),np.nan)
        for part in parts:
            use=(a['batch']==part['batch'])&(a['fold']==part['fold']);wanted=a['ids'][use]-part['batch']*EVENTS[source]
            for r,key in enumerate(part['keys']):
                child=load(OUT/'arrays'/f'{key}.npz');np.testing.assert_array_equal(child['ids'],wanted)
                np.testing.assert_array_equal(child['q'],a['q'][use]);X[use,r]=child['X'];profiles[use,r]=child['mean_wave']
        assert np.isfinite(X).all() and np.isfinite(profiles).all()
        cases[case]=X;caches[case]=cache_moments(X)
        s=cached_nested(a,caches[case],p,source);direct=nested_summary(a,X,p,source)
        for k in s:
            if k!='per_Q':np.testing.assert_allclose(s[k],direct[k],rtol=1e-8,atol=1e-9)
        per_repeat=[];migrations=[];w=event_weights(a,p,source)
        for r in range(R):
            item={**a,'X':X[:,r],'mean_wave':profiles[:,r]}
            per_repeat.append(full_contrast(summary(item,p,source),ds,curve(item,p,source),datacurve))
            mask=band(X[:,r],g['run']);migrations.append({'outside_band':int((~mask).sum()),'outside_band_weight':float(w@~mask)})
        path=OUT/'repeat_arrays'/f'{source}_{case}.npz';np.savez_compressed(path,ids=a['ids'],X=X,mean_wave=profiles)
        files[relative(path)]=sha(path)
        records[case]={'summary':s,'point':dict(zip(NESTED_METRICS,nested_vector(s,ds))),
            'per_repeat_contrasts':np.asarray(per_repeat),'band_migrations':migrations,
            'mean_profile':np.mean([curve({**a,'mean_wave':profiles[:,r]},p,source) for r in range(R)],0),
            'negative_between_diagonal':int(np.sum(np.diag(s['between_corrected'])<0)),'repeat_count':R}
        if R==16:
            records[case]['repeat_sensitivity']={label:cached_nested(a,cache_moments(X[:,sl]),p,source)
                                               for label,sl in [('first8',slice(0,8)),('last8',slice(8,16))]}
    draw={case:np.full((512,len(NESTED_METRICS)),np.nan) for case in cases}
    differences=np.full((512,len(FEATURES),3),np.nan)
    for i in range(512):
        db=summary(data,p,source,boot['data_counts'][i]);cache_result={}
        for case in cases:
            s=cached_nested(a,caches[case],p,source,boot['MC_counts'][i]);cache_result[case]=s;draw[case][i]=nested_vector(s,db)
            if i in [0,127,511]:
                direct=nested_summary(a,cases[case],p,source,boot['MC_counts'][i])
                for k in s:
                    if k!='per_Q':np.testing.assert_allclose(s[k],direct[k],rtol=1e-8,atol=1e-9)
        if db is not None and all(cache_result[x] is not None for x in ['current_all','zero_cv_all']):
            current=cache_result['current_all'];zero=cache_result['zero_cv_all'];dv=np.diag(db['cov'])
            differences[i]=np.stack([(np.diag(zero[k])-np.diag(current[k]))/dv for k in
                ['one_readout_within','between_corrected','one_readout_expected_covariance']],1)
    current=records['current_all']['summary'];zero=records['zero_cv_all']['summary'];dv=np.diag(ds['cov'])
    diff=np.stack([(np.diag(zero[k])-np.diag(current[k]))/dv for k in
        ['one_readout_within','between_corrected','one_readout_expected_covariance']],1)
    np.testing.assert_allclose(diff[:,:2].sum(1),diff[:,2],rtol=1e-9,atol=1e-9)
    np.testing.assert_allclose(differences[:,:,:2].sum(2),differences[:,:,2],rtol=1e-8,atol=1e-9)
    for case in records:records[case]['CI95']={n:interval(draw[case][:,j]) for j,n in enumerate(NESTED_METRICS)}
    bp=OUT/'repeat_evaluation'/f'{source}_bootstrap.npz';np.savez_compressed(bp,**draw,zero_minus_current_components=differences)
    result={'source':source,'context':context,'protocol_sha256':sha(OUT/'replay_protocol.json'),'arrays_sha256':files,
        'bootstrap_sha256':sha(bp),'features':FEATURES,'nested_metrics':NESTED_METRICS,'per_repeat_metrics':POINT_METRICS,
        'data_summary':ds,'data_mean_profile':datacurve,'cases':records,
        'zero_minus_current_components':{'columns':['within','between','total'],'point':diff,
            'CI95':{f:{name:interval(differences[:,j,k]) for k,name in enumerate(['within','between','total'])} for j,f in enumerate(FEATURES)}},
        'new_independent_mothers':0}
    save(OUT/'repeat_evaluation'/f'{source}.json',serialize(result))
    print('Repeated-noise decomposition complete',source,'finite-R corrections and mother-level paired intervals.',flush=True)


def main(stage,workers):
    if stage=='freeze':
        assert not (OUT/'analysis_protocol.json').exists()
        for n in ['point_arrays','point_evaluation','repeat_arrays','repeat_evaluation']:(OUT/n).mkdir(exist_ok=True)
        save(OUT/'analysis_protocol.json',{'code_sha256':sha(__file__),'common_sha256':sha(PROJECT/'simulation/noise_replay_common.py'),
            'replay_protocol_sha256':sha(OUT/'replay_protocol.json'),'point_metrics':POINT_METRICS,'nested_metrics':NESTED_METRICS,
            'loss_definition':'Absolute log ratio for every positive ratio, absolute signed center difference, and raw nonnegative mean-ADC/covariance errors. Paired zero-CV minus current loss intervals. Any lower interval endpoint>0 is a descriptive guard against model promotion, not a multiplicity-corrected significance claim.',
            'repeat_bootstrap':'Whole-mother weights, coalescing duplicates before within-repeat bias correction. Data blocks paired as parent. All repeats of a mother remain together. R sensitivity first8/last8/all16, no adaptive extension.'})
        return
    p=read(OUT/'analysis_protocol.json');assert sha(__file__)==p['code_sha256']
    assert sha(PROJECT/'simulation/noise_replay_common.py')==p['common_sha256']
    assert (OUT/'replay_execution.json').exists()
    if stage=='points':
        contexts=read(OUT/'replay_protocol.json')['point_contexts']
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            jobs=[executor.submit(point_one,k) for k in contexts]
            for j in concurrent.futures.as_completed(jobs):j.result()
        save(OUT/'point_analysis.json',{k:read(OUT/'point_evaluation'/f'{k}.json') for k in contexts})
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            jobs=[executor.submit(repeat_one,s) for s,_ in SOURCES]
            for j in concurrent.futures.as_completed(jobs):j.result()
        save(OUT/'repeat_analysis.json',{s:read(OUT/'repeat_evaluation'/f'{s}.json') for s,_ in SOURCES})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','points','repeats']);parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args();main(args.stage,args.workers)
