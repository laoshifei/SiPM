"""Joint native waveform/PID/F checks, paired raw-block and whole-mother intervals."""
import numpy as np
from state_response_common import *
from analyze_acquisition_state import interval,mean_cov
from publication_stats import block_counts

NAMES=['mean_time_delta_ns','mean_ratio_delta','mean_PCA_delta','IQR_ratio_ratio','IQR_time_ratio','IQR_PCA_ratio',
       'SD_ratio_ratio','SD_time_ratio','SD_PCA_ratio','F50_ratio','F102_ratio','F202_ratio',
       'covariance_normalized_RMSE','mean_ADC_NRMSE','tail_fraction_delta']
VARIANTS=['frozen','electronics','response_inherited','response_spectrum']


def comparison(m,d):
    norm=np.sqrt(np.diag(d['cov'])[:,None]*np.diag(d['cov'])[None,:])
    return np.r_[-1000*(m['mean'][1]-d['mean'][1]),m['mean'][0]-d['mean'][0],m['mean'][2]-d['mean'][2],
        m['iqr']/d['iqr'],np.sqrt(np.diag(m['cov'])/np.diag(d['cov'])),m['F']/d['F'],
        np.sqrt(np.mean(((m['cov']-d['cov'])/norm)**2)),
        np.linalg.norm(m['wave']-d['wave'])/np.linalg.norm(d['wave']),m['tail']-d['tail']]


def assemble(run,seg,B,variant):
    if variant=='frozen':return load(OUT/'arrays'/f'run{run}_phase00.npz')
    assignment=load(OUT/'arrays'/f'run{run}_{seg}_alignment_B{B}.npz')['mother_assignment']
    choices=[load(OUT/'arrays'/f'run{run}_{seg}_B{B}_eval{ef}_{variant}.npz') for ef in [0,1]]
    for k in ['ids','q']:np.testing.assert_array_equal(choices[0][k],choices[1][k])
    result={}
    for k,x in choices[0].items():
        if x.ndim and x.shape[0]==len(assignment):
            result[k]=x.copy();result[k][assignment==1]=choices[1][k][assignment==1]
        else:result[k]=x
    result['mother_assignment']=assignment
    return result


def main():
    assert not (OUT/'evaluation.json').exists()
    p=read(OUT/'protocol.json');report={'scope':__doc__,'metrics':NAMES,'groups':{},
        'code_sha256':sha(__file__),'protocol_sha256':sha(OUT/'protocol.json')};draws={}
    for source,run in SOURCES:
        for si,seg in enumerate(SEGMENTS):
            a,_=measured(run,seg);dp=moments(a,p,source);dshape,dcov=mean_cov(a,a['shape'],p,source)
            group=f'run{run}_{seg}';report['groups'][group]={}
            for B in p['block_sizes']:
                key=f'{group}_B{B}';native={v:assemble(run,seg,B,v) for v in VARIANTS}
                mp={v:moments(m,p,source) for v,m in native.items()};n=int(native['frozen']['n_parent'])
                points={v:comparison(m,dp) for v,m in mp.items()}
                boot=np.full((512,4,len(NAMES)),np.nan);rng=np.random.default_rng(610031900+run+100*si+B)
                for ib in range(512):
                    count=block_counts(a['ids'],tuple(a['span']),B,rng);db=moments(a,p,source,count)
                    if db is None:continue
                    mothers=np.bincount(rng.integers(n,size=n),minlength=n)
                    for iv,v in enumerate(VARIANTS):
                        m=native[v];mb=moments(m,p,source,mothers[m['ids']])
                        if mb is not None:boot[ib,iv]=comparison(mb,db)
                row={'data_moments':dp,'data_shape_mean':dshape,'data_shape_cov':dcov,'variants':{}}
                for iv,v in enumerate(VARIANTS):
                    m=native[v];sh,cv=mean_cov(m,m['shape'],p,source)
                    r={'point':dict(zip(NAMES,points[v])),'native_moments':mp[v],
                        'CI95':{name:interval(boot[:,iv,j]) for j,name in enumerate(NAMES)},
                        'paired_change_from_frozen':{name:interval((boot[:,iv]-boot[:,0])[:,j]) for j,name in enumerate(NAMES)},
                        'shape_mean':sh,'shape_cov':cv,'n_fixed':len(m['ids']),
                        'n_support':int(np.count_nonzero(event_weights(m,p,source))),
                        'Q_change_relative_quantiles':np.quantile(m['qcalc']/m['q']-1,[.01,.5,.99])}
                    row['variants'][v]=r
                    kept={k:x for k,x in m.items() if k not in ['waves','signal']}
                    np.savez_compressed(OUT/'arrays'/f'{key}_{v}_evaluation.npz',**kept)
                report['groups'][group][str(B)]=row;draws[key]=boot
                print(key,[(v,round(points[v][0],3),np.round(points[v][3:6],3).tolist(),np.round(points[v][9:12],3).tolist()) for v in VARIANTS],flush=True)
    report['limitations']='Conditional on fitted segment noise/response, fixed PCA/Q2 targets/IDs and MC fold assignment. Parent mothers reused for model calibration and internal replay, not independent validation. First60/middle20/last20 and both raw-block parities already explored. Source is not event truth. Native ADC unscaled; no measured waveform used by simulation.'
    report['new_independent_mothers']=0;report['default_changed']=False
    np.savez_compressed(OUT/'evaluation_bootstrap.npz',**draws);save(OUT/'evaluation.json',serialize(report))


if __name__=='__main__':main()
