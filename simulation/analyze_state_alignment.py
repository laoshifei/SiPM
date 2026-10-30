"""Cross-fitted front-only location and observed/native sampling-phase diagnostics."""
import numpy as np
from state_response_common import *
from analyze_acquisition_state import design_matrix, bins_of, interval
from publication_stats import block_counts

GRID=np.arange(-40.,400.01,2.)
LAGS=np.arange(-10.,10.001,.25)


def template(a,weights):
    use=np.flatnonzero(weights>0);result=np.zeros(len(GRID))
    for i in use:
        time=(np.arange(a['waves'].shape[1])-a['t10'][i])*2
        result+=weights[i]*np.interp(GRID,time,(a['waves'][i]-a['baseline'][i])/a['qcalc'][i])
    return result


def front_fit(a,shape):
    lag=[];loss=[];amplitudes=[]
    for wave,t10,base in zip(a['waves'],a['t10'],a['baseline']):
        ix=int(np.ceil(t10))+np.arange(100);times=(ix-t10)*2
        y=wave[ix].astype(float)-base;y-=y.mean()
        x=np.interp((times[None,:]-LAGS[:,None]).ravel(),GRID,shape).reshape(len(LAGS),100)
        x-=x.mean(1)[:,None];den=(x*x).sum(1);amplitude=x@y/den
        # Positive amplitude is the physical front direction; no event is dropped.
        amplitude=np.maximum(amplitude,0.);sse=((y-x*amplitude[:,None])**2).sum(1)
        best=np.argmin(sse);lag.append(LAGS[best]);loss.append(sse[best]/100);amplitudes.append(amplitude[best])
    lag=np.asarray(lag);anchor=np.floor(a['t10']+lag/2).astype(int)+1
    scores,q,_=score_gate(a['waves'],anchor,a['baseline'],PCA)
    return {'lag_ns':lag,'anchor':anchor,'scores':scores,'qcalc':q,
        'front_MSE':np.array(loss),'front_amplitude_nuisance':np.array(amplitudes)}


def compact(a,scores):
    return {**a,'scores':scores}


def fit_phase(a,p,source,mask,phase):
    w=event_weights(a,p,source,mask.astype(int));assert w is not None
    center=w@a['state'];scale=np.sqrt(w@((a['state']-center)**2))
    x=design_matrix(a,p,source,center,scale,0.,1.,'Q_state')
    if phase:
        phi=(a['t10']%1)*2*np.pi
        x=np.c_[x,np.sin(phi),np.cos(phi),np.sin(2*phi),np.cos(2*phi)]
    use=w>0;sw=np.sqrt(w[use]);beta,_,rank,s=np.linalg.lstsq(x[use]*sw[:,None],a['scores'][use]*sw[:,None],rcond=1e-12)
    assert rank==x.shape[1] and s[0]/s[-1]<1e8
    return x@beta,{'rank':rank,'condition':s[0]/s[-1],'training_ids':a['ids'][use],
        'state_center':center,'state_scale':scale,'beta':beta}


def contrast(m,d):return np.r_[-1000*(m['mean'][1]-d['mean'][1]),m['iqr']/d['iqr']]


def main():
    assert not (OUT/'alignment.json').exists()
    p=read(OUT/'protocol.json');report={'scope':__doc__,'groups':{},'native_phase':{},
        'protocol_sha256':sha(OUT/'protocol.json'),'code_sha256':sha(__file__)};saved={}
    for source,run in SOURCES:
        mc=load(OUT/'arrays'/f'run{run}_phase00.npz');mp=moments(mc,p,source)
        n=int(mc['n_parent']);rng=np.random.default_rng(610030300+run)
        phases=[load(OUT/'arrays'/f'run{run}_phase{int(ph*10):02d}.npz') for ph in p['phase_offsets_ns']]
        values=[moments(v,p,source) for v in phases];boot=np.full((512,4,4),np.nan)
        for b in range(512):
            count=np.bincount(rng.integers(n,size=n),minlength=n)[mc['ids']]
            mm=[moments(v,p,source,count) for v in phases]
            if any(x is None for x in mm):continue
            for j in range(4):boot[b,j]=contrast(mm[j],mm[0])
        report['native_phase'][source]=[
            {'phase_ns':ph,'mean_time_change_ns':-1000*(v['mean'][1]-mp['mean'][1]),
             'IQR_ratio_to_phase0':v['iqr']/mp['iqr'],'mean_ADC_NRMSE_to_phase0':np.linalg.norm(v['wave']-mp['wave'])/np.linalg.norm(mp['wave']),
             'conditional_whole_mother_intervals':[interval(boot[:,j,k]) for k in range(4)]}
            for j,(ph,v) in enumerate(zip(p['phase_offsets_ns'],values))]
        saved[f'run{run}_native_phase_bootstrap']=boot
        for si,seg in enumerate(SEGMENTS):
            a,_=measured(run,seg);dp=moments(a,p,source);key=f'run{run}_{seg}';row={}
            for B in p['block_sizes']:
                folds=(a['ids']//B)%2;w=event_weights(a,p,source);assert w is not None
                # One mother assignment reused for both front and response checks.
                rng=np.random.default_rng(610030400+run+100*si+B)
                assigned=(rng.random(n)>=float(w@(folds==0)))[mc['ids']].astype(int)
                data_out={};mc_out={};pred=np.zeros((2,len(a['ids']),3));fits={};templates=[]
                for ef in [0,1]:
                    train=folds!=ef;wt=event_weights(a,p,source,train.astype(int));shape=template(a,wt);templates.append(shape)
                    dfit=front_fit(a,shape);mfit=front_fit(mc,shape)
                    for k in dfit:
                        if k not in data_out:data_out[k]=np.empty_like(dfit[k])
                        if k not in mc_out:mc_out[k]=np.empty_like(mfit[k])
                        data_out[k][folds==ef]=dfit[k][folds==ef];mc_out[k][assigned==ef]=mfit[k][assigned==ef]
                    fits[str(ef)]={}
                    for im,phase in enumerate([False,True]):
                        pr,fit=fit_phase(a,p,source,train,phase);pred[im,folds==ef]=pr[folds==ef]
                        fits[str(ef)][str(phase)]=fit
                dfront=compact(a,data_out['scores']);mfront=compact(mc,mc_out['scores'])
                dpoint=moments(dfront,p,source);mpoint=moments(mfront,p,source)
                errors=np.einsum('n,mni->mi',w,(pred-a['scores'][None,:,:])**2)
                contrasts=np.array([contrast(mp,dp),contrast(mpoint,dpoint)])
                rng=np.random.default_rng(610030700+run+100*si+B);bd=np.full((512,2,4),np.nan)
                phase_ratios=np.full((512,3),np.nan)
                for ib in range(512):
                    dc=block_counts(a['ids'],tuple(a['span']),B,rng)
                    count=np.bincount(rng.integers(n,size=n),minlength=n)[mc['ids']]
                    ds=[moments(z,p,source,dc) for z in [a,dfront]];ms=[moments(z,p,source,count) for z in [mc,mfront]]
                    if all(z is not None for z in ds+ms):
                        for j in range(2):bd[ib,j]=contrast(ms[j],ds[j])
                    wb=event_weights(a,p,source,dc)
                    if wb is not None:
                        e=np.einsum('n,mni->mi',wb,(pred-a['scores'][None,:,:])**2);phase_ratios[ib]=e[1]/e[0]
                row[str(B)]={'original_contrasts':contrasts[0],'front_contrasts':contrasts[1],
                    'paired_front_change_intervals':[interval((bd[:,1]-bd[:,0])[:,j]) for j in range(4)],
                    'front_data_boundary_fraction':w@(abs(data_out['lag_ns'])==10),
                    'front_MC_boundary_fraction':event_weights(mc,p,source)@(abs(mc_out['lag_ns'])==10),
                    'data_lag_ns_quantiles':weighted_quantile(data_out['lag_ns'][w>0],w[w>0],[.01,.5,.99]),
                    'phase_regression_OOF_MSE':errors,'phase_regression_MSE_ratio':errors[1]/errors[0],
                    'phase_prediction_conditional_intervals':[interval(phase_ratios[:,j]) for j in range(3)],
                    'regression_fits':fits,'MC_mothers_assigned_per_fold':np.bincount(assigned,minlength=2)}
                np.savez_compressed(OUT/'arrays'/f'{key}_alignment_B{B}.npz',ids=a['ids'],mc_ids=mc['ids'],
                    mother_assignment=assigned,template_grid=GRID,templates=templates,prediction=pred,
                    **{'data_'+k:v for k,v in data_out.items()},**{'mc_'+k:v for k,v in mc_out.items()})
                saved[f'{key}_B{B}_front']=bd;saved[f'{key}_B{B}_phase_prediction']=phase_ratios
                print(key,B,'front delta ns',round(contrasts[1,0]-contrasts[0,0],4),
                    'phase MSE ratios',np.round(errors[1]/errors[0],4),flush=True)
            report['groups'][key]=row
    report['limitations']='No true measured phase observed. t10 phase shares noise with PID and is only a proxy. Front template learned on opposite data blocks and used only for location; original ADC integrated. Intervals conditional on fixed templates/regression fits, not full refit uncertainty. All experimental checks previously explored and internal.'
    np.savez_compressed(OUT/'alignment_bootstrap.npz',**saved);save(OUT/'alignment.json',serialize(report))


if __name__=='__main__':main()
