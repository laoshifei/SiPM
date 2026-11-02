"""Paired native joint checks for continuous response and noise-distribution controls."""
import numpy as np
from rc_refinement_common import *
from evaluate_state_response import comparison,NAMES as OLD_NAMES,assemble as parent_assemble
from analyze_acquisition_state import interval,mean_cov
from publication_stats import block_counts
from diagnose_state_fold_mixture import decomposition

VARIANTS=['frozen','coarse_spectrum','mean_only','inherited_CV_fixed','inherited_CV_fit','spectrum_CV_fixed','spectrum_CV_fit']
NAMES=OLD_NAMES+['F50_SD_ratio','F102_SD_ratio','F202_SD_ratio']


def selected_arrays(run,seg,B,variant,mean,noise):
    if variant=='frozen':return load(PARENT/'arrays'/f'run{run}_phase00.npz')
    if variant=='coarse_spectrum':return parent_assemble(run,seg,B,'response_spectrum')
    assignment=load(PARENT/'arrays'/f'run{run}_{seg}_alignment_B{B}.npz')['mother_assignment']
    choices=[]
    for ef in [0,1]:
        key=f'run{run}_{seg}_B{B}_eval{ef}'
        if variant=='mean_only':nk=mean['groups'][key]['native_key']
        else:
            family=variant.split('_')[0];r=noise['groups'][key]['families'][family]
            nk=r['chosen_native_key' if variant.endswith('fit') else 'inherited_native_key']
        choices.append(load(OUT/'arrays'/f'{nk}.npz'))
    for k in ['ids','q']:np.testing.assert_array_equal(choices[0][k],choices[1][k])
    result={}
    for k,x in choices[0].items():
        if x.ndim and x.shape[0]==len(assignment):
            result[k]=x.copy();result[k][assignment==1]=choices[1][k][assignment==1]
        else:result[k]=x
    result['mother_assignment']=assignment
    return result


def contrasts(m,d):return np.r_[comparison(m,d),m['F_sd']/d['F_sd']]


def main():
    assert not (OUT/'evaluation.json').exists()
    p=read(OUT/'protocol.json');mean=read(OUT/'mean_calibration.json');noise=read(OUT/'noise_selection.json')
    old=read(PARENT/'evaluation.json');report={'scope':__doc__,'code_sha256':sha(__file__),'groups':{},'variants':VARIANTS,
        'metrics':NAMES,'protocol_sha256':sha(OUT/'protocol.json')};draws={}
    for source,run in SOURCES:
        for si,seg in enumerate(SEGMENTS):
            data,_=measured(run,seg);dp=summary(data,p,source);dshape,dcov=mean_cov(data,data['shape'],p,source);group=f'run{run}_{seg}'
            report['groups'][group]={}
            for B in p['block_sizes']:
                native={v:selected_arrays(run,seg,B,v,mean,noise) for v in VARIANTS}
                mp={v:summary(z,p,source) for v,z in native.items()};points={v:contrasts(m,dp) for v,m in mp.items()}
                for v,ov in [('frozen','frozen'),('coarse_spectrum','response_spectrum')]:
                    np.testing.assert_allclose(points[v][:len(OLD_NAMES)],[old['groups'][group][str(B)]['variants'][ov]['point'][n] for n in OLD_NAMES],atol=1e-12)
                n=int(native['frozen']['n_parent']);rng=np.random.default_rng(610031900+run+100*si+B)
                boot=np.full((512,len(VARIANTS),len(NAMES)),np.nan)
                for ib in range(512):
                    dc=block_counts(data['ids'],tuple(data['span']),B,rng);db=summary(data,p,source,dc)
                    if db is None:continue
                    counts=np.bincount(rng.integers(n,size=n),minlength=n)
                    for iv,v in enumerate(VARIANTS):
                        z=native[v];mb=summary(z,p,source,counts[z['ids']])
                        if mb is not None:boot[ib,iv]=contrasts(mb,db)
                dparts=decomposition(data,p,source,(data['ids']//B)%2)
                assignment=load(PARENT/'arrays'/f'{group}_alignment_B{B}.npz')['mother_assignment']
                row={'data_summary':dp,'data_shape_mean':dshape,'data_shape_cov':dcov,'data_fold_covariance_parts':dparts,'variants':{}}
                for iv,v in enumerate(VARIANTS):
                    z=native[v];sh,cv=mean_cov(z,z['shape'],p,source);parts=decomposition(z,p,source,assignment)
                    comparator='inherited_CV_fixed' if v=='inherited_CV_fit' else 'spectrum_CV_fixed' if v=='spectrum_CV_fit' else 'coarse_spectrum'
                    reference=VARIANTS.index(comparator)
                    row['variants'][v]={'point':dict(zip(NAMES,points[v])),'native_summary':mp[v],
                        'CI95':{name:interval(boot[:,iv,j]) for j,name in enumerate(NAMES)},
                        'paired_from_coarse':{name:interval(boot[:,iv,j]-boot[:,1,j]) for j,name in enumerate(NAMES)},
                        'control_comparator':comparator,'paired_from_control':{name:interval(boot[:,iv,j]-boot[:,reference,j]) for j,name in enumerate(NAMES)},
                        'shape_mean':sh,'shape_cov':cv,'fold_covariance_parts':parts,
                        'within_fold_SD_ratio':np.sqrt(np.diag(parts[2])/np.diag(dparts[2])),
                        'n_fixed':len(z['ids']),'n_support':int(np.count_nonzero(event_weights(z,p,source))),
                        'Q_change_relative_quantiles':np.quantile(z['qcalc']/z['q']-1,[.01,.5,.99])}
                    np.savez_compressed(OUT/'arrays'/f'{group}_B{B}_{v}_evaluation.npz',
                        **{k:x for k,x in z.items() if k not in ['waves','signal']})
                report['groups'][group][str(B)]=row;draws[f'{group}_B{B}']=boot
                print(group,B,[(v,round(points[v][0],3),np.round(points[v][3:6],3).tolist(),np.round(points[v][-3:],2).tolist()) for v in VARIANTS],flush=True)
    report['limits']='All checks are conditional and internal. Fitted parameters/model selection held fixed in paired intervals. Both block sizes and original mother assignments preserved. Source groups not particle truth; no new experiment or independent transport.'
    report['new_independent_mothers']=0;report['default_changed']=False
    np.savez_compressed(OUT/'evaluation_bootstrap.npz',**draws);save(OUT/'evaluation.json',serialize(report))


if __name__=='__main__':main()
