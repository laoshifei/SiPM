"""Matched estimator controls, blocked prediction and temporal covariance checks."""
import argparse
import numpy as np
from prepare_acquisition_state import (OUT,PARENT,PROJECT,SOURCES,SEGMENTS,CONTROLS,MC_VARIANTS,
                                       read,load,relative,finite)
from publication_stats import save,sha,block_counts

NBOOT=512
MODELS=['Q','Q_state','Q_time','Q_state_time']


def weighted_quantile(x,w,qs=(.25,.75)):
    order=np.argsort(x);x=x[order];w=w[order];pos=np.cumsum(w)-w/2
    if len(x)<2:return np.full(len(qs),np.nan)
    return np.interp(qs,(pos-pos[0])/(pos[-1]-pos[0]),x)


def bins_of(a,p):return np.clip(np.searchsorted(p['q_edges'],a['q'],'right')-1,0,5)


def rows(a,p,source,counts=None):
    bins=bins_of(a,p);result=[]
    for b in p['support_bins'][source]:
        ix=np.flatnonzero(bins==b)
        if counts is not None:ix=np.repeat(ix,counts[ix])
        cut=p['conditional_cuts'][source][str(b)];cell=(a['q'][ix]>=cut['q_quartiles'][1]).astype(int)
        number=np.bincount(cell,minlength=2);target=np.asarray(cut['q2_target'])
        if len(ix)<5 or np.any((target>0)&(number==0)):return None
        w=(target/number)[cell]
        assert np.isclose(w.sum(),1.)
        result.append((b,ix,w,cell))
    return result


def event_weights(a,p,source,counts=None):
    rr=rows(a,p,source,counts)
    if rr is None:return None
    result=np.zeros(len(a['ids']))
    for _,ix,w,_ in rr:np.add.at(result,ix,w/len(rr))
    return result


def statistics(a,p,source,counts=None):
    rr=rows(a,p,source,counts)
    if rr is None:return np.full((4,15),np.nan)
    mus=[];iqrs=[];covs=[]
    for _,ix,w,_ in rr:
        x=a['scores'][ix];mu=np.einsum('n,nci->ci',w,x);res=x-mu
        cov=np.einsum('n,nci,ncj->cij',w,res,res)
        iqr=np.array([[np.diff(weighted_quantile(x[:,i,j],w))[0] for j in range(3)] for i in range(4)])
        mus.append(mu);iqrs.append(iqr);covs.append(cov)
    return np.concatenate([np.mean(mus,axis=0),np.sqrt(np.mean(np.array(iqrs)**2,axis=0)),
        np.mean(covs,axis=0).reshape(4,9)],axis=1)


def interval(values):
    v=np.asarray(values);v=v[np.isfinite(v)]
    return {'CI95':np.quantile(v,[.025,.975]).tolist() if len(v)>=128 else None,'valid_draws':len(v)}


def contrasts(m,d):
    return np.concatenate([-1000*(m[...,1:2]-d[...,1:2]),m[...,3:6]/d[...,3:6]],axis=-1)


def controls():
    assert not (OUT/'controls.json').exists()
    p=read(OUT/'protocol.json');old=read(PARENT/'conditional.json');draws={};report={'code_sha256':sha(__file__),
        'protocol_sha256':sha(OUT/'protocol.json'),'groups':{},'MC':{},'scope':__doc__}
    for source,run in SOURCES:
        mc={v:load(OUT/'arrays'/f'run{run}_{source}__{v}.npz') for v in MC_VARIANTS}
        points={v:statistics(a,p,source) for v,a in mc.items()};n=int(mc['identity']['n_parent'])
        mb={v:np.empty((NBOOT,4,15)) for v in MC_VARIANTS};rng=np.random.default_rng(510030500+run)
        for ib in range(NBOOT):
            count=np.bincount(rng.integers(n,size=n),minlength=n)
            for v,a in mc.items():mb[v][ib]=statistics(a,p,source,count[a['ids']])
        report['MC'][source]={v:point.tolist() for v,point in points.items()}
        for v in MC_VARIANTS:draws[f'run{run}__{v}']=mb[v]
        for si,seg in enumerate(SEGMENTS):
            key=f'run{run}_{seg}';a=load(OUT/'arrays'/f'{key}.npz');dp=statistics(a,p,source);db={}
            np.testing.assert_allclose(dp[0],old['groups'][key]['modes']['q2']['data'][:15],atol=1e-13)
            np.testing.assert_allclose(points['identity'][0],old['groups'][key]['modes']['q2']['MC'][:15],atol=1e-13)
            for block in [64,128]:
                rng=np.random.default_rng(510030700+100*run+1000*si+block);b=np.empty((NBOOT,4,15))
                for ib in range(NBOOT):b[ib]=statistics(a,p,source,block_counts(a['ids'],tuple(a['span']),block,rng))
                db[block]=b;draws[f'{key}__block{block}']=b
            row={'data':dp.tolist(),'comparisons':{}}
            for v in MC_VARIANTS:
                value=contrasts(points[v],dp);B={block:contrasts(mb[v],db[block]) for block in [64,128]}
                row['comparisons'][v]={}
                for j,name in enumerate(CONTROLS):
                    r={'mean_time_difference_ns':float(value[j,0]),'IQR_ratio':value[j,1:].tolist(),'uncertainty':{}}
                    for block in [64,128]:
                        r['uncertainty'][str(block)]={'contrasts':[interval(B[block][:,j,k]) for k in range(4)],
                            'paired_change_from_original':[interval((B[block][:,j]-B[block][:,0])[:,k]) for k in range(4)]}
                    row['comparisons'][v][name]=r
            report['groups'][key]=finite(row)
            print(key,'identity contrasts',np.round(contrasts(points['identity'],dp),3).tolist(),flush=True)
    report['original_conditional_estimator_closure']=True
    np.savez_compressed(OUT/'control_bootstrap.npz',**draws);save(OUT/'controls.json',finite(report))


def mean_cov(a,x,p,source):
    rr=rows(a,p,source)
    if rr is None:return None,None
    means=[];covs=[]
    for _,ix,w,_ in rr:
        mu=w@x[ix];r=x[ix]-mu;means.append(mu);covs.append(r.T@(w[:,None]*r))
    return np.mean(means,axis=0),np.mean(covs,axis=0)


def deciles(a,ids=None):
    edges=np.array([int(i*int(a['n_raw'])/10) for i in range(11)])
    return np.clip(np.searchsorted(edges,a['ids'] if ids is None else ids,'right')-1,0,9)


def decomposition(a,p,source,labels):
    rr=rows(a,p,source);parts=np.zeros((4,3,3))
    for _,ix,w,cell in rr:
        x=a['scores'][ix,0];mu=w@x;res=x-mu;parts[0]+=res.T@(w[:,None]*res)
        for c in [0,1]:
            sel=cell==c;mass=w[sel].sum();cm=np.average(x[sel],axis=0,weights=w[sel]);delta=cm-mu
            parts[1]+=mass*np.outer(delta,delta)
            for t in np.unique(labels[ix][sel]):
                u=sel&(labels[ix]==t);g=w[u].sum();gm=np.average(x[u],axis=0,weights=w[u]);delta=gm-cm
                parts[2]+=g*np.outer(delta,delta);r=x[u]-gm;parts[3]+=r.T@(w[u,None]*r)
    parts/=len(rr);np.testing.assert_allclose(parts[0],parts[1:].sum(0),atol=1e-15,rtol=1e-10)
    return parts


def permuted_order(ids,span,size,rng):
    lo,hi=map(int,span);starts=np.arange(lo,hi,size);lengths=np.minimum(size,hi-starts)
    order=rng.permutation(len(starts));destination=np.empty(len(starts),int)
    destination[order]=np.r_[0,np.cumsum(lengths[order])[:-1]]+lo
    block=(ids-lo)//size;return destination[block]+(ids-starts[block])


def design_matrix(a,p,source,state_center,state_scale,time_center,time_scale,model):
    bins=bins_of(a,p);support=p['support_bins'][source];onehot=(bins[:,None]==np.array(support)).astype(float)
    edges=np.array(p['q_edges']);q=(a['q']-(edges[bins]+edges[bins+1])/2)/((edges[bins+1]-edges[bins])/2)
    cols=[onehot,onehot*q[:,None]]
    if 'state' in model:cols.append((a['state']-state_center)/state_scale)
    if 'time' in model:cols.append(((a['ids']/int(a['n_raw'])-time_center)/time_scale)[:,None])
    return np.concatenate(cols,axis=1)


def fit_model(a,p,source,model,mask):
    w=event_weights(a,p,source,mask.astype(int));assert w is not None
    center=w@a['state'];scale=np.sqrt(w@((a['state']-center)**2));assert np.all(scale>1e-10)
    time=a['ids']/int(a['n_raw']);tc=float(w@time);ts=float(np.sqrt(w@((time-tc)**2)))
    x=design_matrix(a,p,source,center,scale,tc,ts,model);use=w>0;sw=np.sqrt(w[use])
    beta,resid,rank,singular=np.linalg.lstsq(x[use]*sw[:,None],a['scores'][use,0]*sw[:,None],rcond=1e-12)
    assert rank==x.shape[1] and singular[0]/singular[-1]<1e8
    return {'model':model,'state_center':center.tolist(),'state_scale':scale.tolist(),'time_center':tc,'time_scale':ts,
        'beta':beta.tolist(),'rank':int(rank),'condition':float(singular[0]/singular[-1]),'training_ids':a['ids'][use].tolist()}


def predict(a,p,source,fit):
    x=design_matrix(a,p,source,np.array(fit['state_center']),np.array(fit['state_scale']),fit['time_center'],fit['time_scale'],fit['model'])
    return x@np.array(fit['beta'])


def state_support(train,a,p,source):
    w=event_weights(train,p,source);use=w>0;center=w@train['state'];r=train['state']-center
    cov=r.T@(w[:,None]*r);inv=np.linalg.pinv(cov,rcond=1e-12)
    distance=np.einsum('ni,ij,nj->n',r,inv,r)
    bounds=np.array([weighted_quantile(train['state'][use,j],w[use],[.01,.99]) for j in range(3)])
    d99=weighted_quantile(distance[use],w[use],[.99])[0]
    delta=a['state']-center;d=np.einsum('ni,ij,nj->n',delta,inv,delta)
    inside=np.all((a['state']>=bounds[:,0])&(a['state']<=bounds[:,1]),axis=1)&(d<=d99)
    inside&=np.isin(bins_of(a,p),p['support_bins'][source])
    return inside,{'bounds_p01_p99':bounds.tolist(),'mahalanobis2_p99':float(d99),'center':center.tolist(),'covariance':cov.tolist()}


def states():
    assert not (OUT/'states.json').exists()
    p=read(OUT/'protocol.json');report={'code_sha256':sha(__file__),'protocol_sha256':sha(OUT/'protocol.json'),
        'sources':{},'temporal_covariance':{},'timeline':{},'scope':__doc__};arrays={};predictions={};permutations={}
    for source,run in SOURCES:
        data={seg:load(OUT/'arrays'/f'run{run}_{seg}.npz') for seg in SEGMENTS};first=data['first60']
        first_w=event_weights(first,p,source);fold=deciles(first);assert set(fold)==set(range(6))
        fits={};cross={};fold_records={}
        for model in MODELS:
            cross[model]=np.full((len(first['ids']),3),np.nan);fold_records[model]=[]
            for k in range(6):
                fit=fit_model(first,p,source,model,fold!=k);test=fold==k
                assert not set(fit['training_ids'])&set(first['ids'][test])
                cross[model][test]=predict(first,p,source,fit)[test]
                fw=first_w[test];err=(cross[model][test]-first['scores'][test,0])**2
                fold_records[model].append({'fold':k,'test_n':int(test.sum()),'test_ids':first['ids'][test].tolist(),
                    'MSE':np.average(err,axis=0,weights=fw).tolist(),'fit':fit})
            fits[model]=fit_model(first,p,source,model,np.ones(len(first['ids']),bool))
            predictions[f'run{run}__first60_CV__{model}']=cross[model]
        cv_mse={model:first_w@((pred-first['scores'][:,0])**2) for model,pred in cross.items()}
        source_row={'models':fits,'folds':fold_records,'CV_MSE':{k:v.tolist() for k,v in cv_mse.items()},
            'CV_MSE_ratio_to_Q':{k:(v/cv_mse['Q']).tolist() for k,v in cv_mse.items()},'segments':{}}
        for si,seg in enumerate(SEGMENTS):
            a=data[seg];key=f'run{run}_{seg}';w=event_weights(a,p,source);inside,support=state_support(first,a,p,source)
            row={'state_support':support,'weighted_state_support_fraction':float(w@inside),
                'n_in_Q_support':int((w>0).sum()),'n_in_state_support':int(inside.sum()),'predictions':{}}
            predictions[key+'__state_supported']=inside
            for model,fit in fits.items():
                pred=predict(a,p,source,fit);predictions[key+'__'+model]=pred
                base=predict(first,p,source,fit);shift=-1000*(w@pred[:,1]-first_w@base[:,1])
                err=(pred-a['scores'][:,0])**2;bias=-1000*(w@(pred[:,1]-a['scores'][:,0,1]))
                wi=w*inside
                row['predictions'][model]={'mean_time_bias_ns':float(bias),'predicted_mean_time_change_from_first60_ns':float(shift),
                    'MSE':(w@err).tolist(),'MSE_on_state_support':(wi@err/wi.sum()).tolist() if wi.sum()>0 else None,
                    'time_extrapolation':bool(seg!='first60' and 'time' in model)}
            source_row['segments'][seg]=row
            mean,cv=mean_cov(a,a['shape'],p,source);arrays[key+'__shape_mean']=mean;arrays[key+'__shape_cov']=cv
            arrays[key+'__hp_ac']=w@a['hp_ac'];arrays[key+'__state_mean']=w@a['state']
            parts=decomposition(a,p,source,deciles(a));frac=np.diag(parts[2])/np.diag(parts[0])
            temporal={'covariance_parts':parts.tolist(),'between_decile_fraction':frac.tolist(),'permutation_reference':{}}
            for block in [64,128]:
                rng=np.random.default_rng(610030000+run*1000+si*200+block);null=np.empty((NBOOT,3))
                for ib in range(NBOOT):
                    pseudo=permuted_order(a['ids'],a['span'],block,rng);part=decomposition(a,p,source,deciles(a,pseudo))
                    null[ib]=np.diag(part[2])/np.diag(part[0])
                permutations[key+f'__block{block}']=null
                temporal['permutation_reference'][str(block)]={'null_p025_p50_p975':np.quantile(null,[.025,.5,.975],axis=0).tolist(),
                    'upper_tail_fraction_with_plus_one':((1+(null>=frac).sum(0))/(NBOOT+1)).tolist(),
                    'scope':'Whole-block exchangeability reference; not a causal attribution or correction for longer-range dependence.'}
            report['temporal_covariance'][key]=temporal
            print(key,'mean-time between-decile fraction',round(float(frac[1]),4),'state support',round(row['weighted_state_support_fraction'],3),flush=True)
        all_a={k:np.concatenate([data[seg][k] for seg in SEGMENTS]) for k in ['ids','q','scores','state','baseline','baseline_rms','recent_baseline','smooth_anchor','anchor','shape','F']}
        all_a['n_raw']=first['n_raw'];labels=deciles(all_a);timeline=[]
        for k in range(10):
            u=labels==k;sub={key:value[u] if np.ndim(value)>0 and len(value)==len(u) else value for key,value in all_a.items()}
            w=event_weights(sub,p,source);row={'decile':k,'n_all':int(u.sum()),'valid':w is not None}
            if w is not None:
                point=statistics(sub,p,source);row.update(n_in_Q_support=int((w>0).sum()),
                    baseline_RMS=float(w@sub['baseline_rms']),baseline=float(w@sub['baseline']),
                    low_frequency_fraction=float(w@sub['state'][:,2]),F=(w@sub['F']).tolist(),
                    mean_time_ns=(-1000*point[:,1]).tolist(),mean_time_width_ns=(1000*point[:,4]).tolist(),
                    recent_baseline_delta=float(w@(sub['recent_baseline']-sub['baseline'])),
                    anchor_shift_ns=float(w@(2*(sub['smooth_anchor']-sub['anchor']))),
                    tail_2_to_3us_fraction=float(w@sub['shape'][:,51:].sum(1)),
                    predicted_mean_time_ns={model:float(-1000*(w@predict(sub,p,source,fit)[:,1])) for model,fit in fits.items()})
            timeline.append(row)
        report['timeline'][source]=timeline;report['sources'][source]=source_row
        for variant in MC_VARIANTS:
            a=load(OUT/'arrays'/f'run{run}_{source}__{variant}.npz');key=f'run{run}__{variant}'
            mean,cv=mean_cov(a,a['shape'],p,source);arrays[key+'__shape_mean']=mean;arrays[key+'__shape_cov']=cv
            arrays[key+'__hp_ac']=event_weights(a,p,source)@a['hp_ac']
        print(source,'held-out first60 MSE ratios',source_row['CV_MSE_ratio_to_Q'],flush=True)
    np.savez_compressed(OUT/'state_diagnostics.npz',**arrays);np.savez_compressed(OUT/'predictions.npz',**predictions)
    np.savez_compressed(OUT/'permutation_reference.npz',**permutations);save(OUT/'states.json',finite(report))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['controls','states']);args=ap.parse_args()
    controls() if args.stage=='controls' else states()
