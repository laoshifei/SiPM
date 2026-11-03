"""Original-ADC controls and fixed-rank waveform diagnostics for PID tails."""
import numpy as np
from pid_core_tail_common import (PROJECT,SOURCES,SEGMENTS,read,load,save,sha,relative,serialize,rows,
    event_weights,weighted_quantile,bins_of,interval,diagnostics,contrasts,NAMES as CORE_NAMES,
    standardized,fit_reference,COVERAGES,METHODS)
from prepare_acquisition_state import extract_features,score_gate,CONTROLS
from state_response_common import PCA
from reference import wave_features

PARENT=PROJECT/'simulation/results/pid_core_tail_20261030'
RC=PROJECT/'simulation/results/rc_refinement_20261030'
STATE=PROJECT/'simulation/results/state_response_20261030'
OUT=PROJECT/'simulation/results/waveform_tail_20261030'
VARIANTS=['frozen','inherited_CV_fit','spectrum_CV_fit']
OBSERVABLES=['baseline_delta_ADC','baseline_delta_RMS','pre_slope_ADC_per_us','anchor_shift_ns',
    'smooth_t10_shift_ns','middle_fraction_500_1500','late_fraction_2000_3000','late_positive_step_fraction']
FLAGS=['large_baseline_change','large_anchor_change','large_late_positive_step','extreme_late_fraction']
EXAMPLE_QUANTILES=[.025,.5,.975]


def extract(waves,ids,q):
    a=extract_features(waves,PCA);f=wave_features(waves)
    _,_,gate=score_gate(waves,a['anchor'],a['baseline'],PCA)
    a['mean_wave']=gate.reshape(len(waves),76,20).mean(2)
    a['ids']=ids;a['q']=q;a['qcalc']=a['control_q'][:,0]
    delta=a['recent_baseline']-a['baseline'];time=(np.arange(256)-127.5)*.002
    pre=waves[:,:256].astype(float);slope=(pre@time)/(time@time)
    rise=np.maximum(0.,np.diff(a['mean_wave'][:,16:71],axis=1).max(1))/f['amplitude']
    a['observables']=np.c_[delta,delta/a['baseline_rms'],slope,2*(a['smooth_anchor']-a['anchor']),
        2*(a['smooth_t10']-a['t10']),gate[:,270:770].sum(1)/a['qcalc'],
        gate[:,1020:1520].sum(1)/a['qcalc'],rise]
    assert np.isfinite(a['observables']).all()
    return a


def original(a):return {**a,'scores':a['scores'][:,0]}


def control_statistics(a,p,source,counts=None):
    rr=rows(a,p,source,counts)
    if rr is None:return None
    spans=np.array([[(1-c)/2,(1+c)/2] for c in COVERAGES]).ravel();means=[];widths=[];covs=[]
    for b,ix,w,_ in rr:
        x=a['scores'][ix];mu=np.einsum('n,ncj->cj',w,x);r=x-mu
        cov=np.einsum('n,nci,ncj->cij',w,r,r)
        quant=np.array([[weighted_quantile(x[:,c,j],w,spans).reshape(4,2) for j in range(3)] for c in range(4)])
        means.append(mu);covs.append(cov);widths.append(np.transpose(quant[:,:,:,1]-quant[:,:,:,0],(0,2,1)))
    return {'mean':np.mean(means,axis=0),'widths':np.sqrt(np.mean(np.array(widths)**2,axis=0)),
        'cov':np.mean(covs,axis=0),'sd':np.sqrt(np.diagonal(np.mean(covs,axis=0),axis1=1,axis2=2))}


CONTROL_NAMES=['mean_time_delta_ns']+[f'width{int(c*100)}_{m}_ratio' for c in COVERAGES for m in METHODS]+[f'SD_{m}_ratio' for m in METHODS]


def control_contrasts(m,d):
    return np.c_[-1000*(m['mean'][:,1]-d['mean'][:,1]),(m['widths']/d['widths']).reshape(4,12),m['sd']/d['sd']]


def training_design(a,p,source,B):
    fold=(a['ids']//B)%2;result={}
    for ef in [0,1]:
        rr=rows(a,p,source,(fold!=ef).astype(int));assert rr is not None
        covariance=np.zeros((76,76));templates={};bounds={}
        for b,ix,w,_ in rr:
            shape=a['shape'][ix];mu=w@shape;r=shape-mu;covariance+=r.T@(w[:,None]*r)
            templates[str(b)]=mu;obs=a['observables'][ix]
            bounds[str(b)]=[float(weighted_quantile(abs(obs[:,1]),w,[.99])[0]),
                float(weighted_quantile(abs(obs[:,3]),w,[.99])[0]),float(weighted_quantile(obs[:,7],w,[.99])[0]),
                weighted_quantile(obs[:,6],w,[.01,.99]).tolist()]
        covariance/=len(rr);values,vectors=np.linalg.eigh(covariance);order=np.argsort(values)[::-1]
        values=values[order];vectors=vectors[:,order[:3]]
        for j in range(3):
            if vectors[np.argmax(abs(vectors[:,j])),j]<0:vectors[:,j]*=-1
        assert values[2]>0
        result[str(ef)]={'training_ids':a['ids'][fold!=ef].tolist(),'evaluation_ids':a['ids'][fold==ef].tolist(),
            'templates':templates,'observable_bounds':bounds,'eigenvalues':values,'vectors':vectors,
            'training_rank3_variance_fraction':float(values[:3].sum()/values.sum()),'covariance':covariance}
    return serialize(result)


def apply_design(a,p,source,design,fold):
    b=bins_of(a,p);projection=np.full((len(b),3),np.nan);residual=np.full((len(b),76),np.nan)
    flags=np.zeros((len(b),4),bool)
    for ef in [0,1]:
        d=design[str(ef)];basis=np.array(d['vectors'])
        for bin0 in p['support_bins'][source]:
            use=(b==bin0)&(fold==ef);r=a['shape'][use]-np.array(d['templates'][str(bin0)])
            projection[use]=r@basis;residual[use]=r-projection[use]@basis.T
            x=a['observables'][use];lim=d['observable_bounds'][str(bin0)]
            flags[use]=np.c_[abs(x[:,1])>lim[0],abs(x[:,3])>lim[1],x[:,7]>lim[2],(x[:,6]<lim[3][0])|(x[:,6]>lim[3][1])]
    return {'projection':projection,'residual':residual,'flags':flags,'fold':fold}


MODE_NAMES=['PC1_variance_ratio','PC2_variance_ratio','PC3_variance_ratio','rank3_variance_ratio','orthogonal_variance_ratio','full_shape_variance_ratio','rank3_fraction_delta_pp']


def mode_statistics(a,p,source,applied,counts=None):
    rr=rows(a,p,source,counts)
    if rr is None:return None
    pc=np.zeros((3,3));perp=0.;whole=0.;mean_error=0.
    for b,ix,w,_ in rr:
        for ef in [0,1]:
            use=applied['fold'][ix]==ef;mass=w[use].sum()
            if mass==0:continue
            weight=w[use];u=ix[use];z=applied['projection'][u];r=applied['residual'][u];s=a['shape'][u]
            zm=(weight@z)/mass;rm=(weight@r)/mass;sm=(weight@s)/mass
            dz=z-zm;dr=r-rm;ds=s-sm;pc+=dz.T@(weight[:,None]*dz)
            perp+=float(weight@np.sum(dr*dr,axis=1));whole+=float(weight@np.sum(ds*ds,axis=1))
            mean_error+=mass*(float(zm@zm)+float(rm@rm))
    pc/=len(rr);perp/=len(rr);whole/=len(rr);mean_error/=len(rr)
    np.testing.assert_allclose(np.trace(pc)+perp,whole,rtol=2e-10,atol=1e-16)
    return {'mode_covariance':pc,'mode_variance':np.diag(pc),'parallel':float(np.trace(pc)),
        'orthogonal':perp,'whole':whole,'captured_fraction':float(np.trace(pc)/whole),'template_mean_squared_bias':mean_error}


def mode_contrasts(m,d):
    return np.r_[m['mode_variance']/d['mode_variance'],m['parallel']/d['parallel'],m['orthogonal']/d['orthogonal'],
        m['whole']/d['whole'],100*(m['captured_fraction']-d['captured_fraction'])]


def feature_audit(a,p,source,applied,z):
    w=event_weights(a,p,source);tail=(np.abs(z)>2).any(1);mass=float(w[tail].sum());flags=applied['flags']
    result={'support_events':int(np.count_nonzero(w)),'joint_PID_tail_events':int(np.sum(tail&(w>0))),
        'joint_PID_tail_mass':mass,'flag_events':np.sum(flags[w>0],axis=0),'flag_mass':w@flags,
        'flag_events_in_PID_tail':np.sum(flags[tail&(w>0)],axis=0),
        'flag_mass_given_PID_tail':w[tail]@flags[tail]/mass if mass else None,
        'PID_tail_events_without_any_flag':int(np.sum(tail&(w>0)&~flags.any(1))),
        'PID_tail_mass_without_any_flag':float(w[tail&~flags.any(1)].sum()),'observables':{}}
    for j,name in enumerate(OBSERVABLES):
        result['observables'][name]={'all_quantiles':weighted_quantile(a['observables'][w>0,j],w[w>0],[.025,.5,.975]),
            'PID_tail_quantiles':weighted_quantile(a['observables'][tail&(w>0),j],w[tail&(w>0)],[.025,.5,.975]) if np.sum(tail&(w>0))>=2 else None}
    return result


def select_examples(a,p,source,z):
    support=p['support_bins'][source];middle=support[(len(support)-1)//2]
    row=next(r for r in rows(a,p,source) if r[0]==middle);b,ix,w,_=row;selected=[]
    for j in range(3):
        quantiles=weighted_quantile(z[ix,j],w,EXAMPLE_QUANTILES)
        for q,target in zip(EXAMPLE_QUANTILES,quantiles):
            order=np.lexsort((a['ids'][ix],np.abs(z[ix,j]-target)));i=int(ix[order[0]])
            selected.append({'method':METHODS[j],'method_index':j,'quantile':q,'index':i,'event_id':int(a['ids'][i]),
                'Q_bin':b,'original_Q':float(a['q'][i]),'target_z':float(target),'actual_z':float(z[i,j]),
                'baseline_ADC':float(a['baseline'][i]),'anchor':int(a['anchor'][i]),'t10':float(a['t10'][i]),
                'observables':dict(zip(OBSERVABLES,a['observables'][i].tolist()))})
    return selected
