"""Compare fresh native tail-candidate ADC with frozen experimental segments."""
from pathlib import Path
import numpy as np
from fit_co_tail import OUT,PARENT,PROJECT,DATA,WIDTH,T,gate_density,SOURCES
from prepare_pid import read
from extract_pid import load
from analyze_npe_validation import raw_shapes
from analyze_bandwidth_validation import per_bin,widths,rms_width,interval
from fit_pid_bandwidth import average_bins
from reference import TIME_NS
from diagnose_pid_bandwidth import METHODS
from publication_stats import save,sha,block_counts

SEGMENTS=['first60','middle20','last20']
SLICES=np.array([-40.,0.,200.,500.,1000.,1500.,2000.,3000.])
NBOOT=512


def finite(obj):
    if isinstance(obj,float) and not np.isfinite(obj):return None
    if isinstance(obj,dict):return {k:finite(v) for k,v in obj.items()}
    if isinstance(obj,(tuple,list)):return [finite(v) for v in obj]
    return obj


def features(waves,scores):
    density,f=gate_density(waves)
    pieces=np.array([2*(density[:,(T>=a)&(T<b)]*T[(T>=a)&(T<b)]).sum(1) for a,b in zip(SLICES[:-1],SLICES[1:])]).T
    mu=pieces.sum(1);np.testing.assert_allclose(mu,-1000*scores[:,1],rtol=1e-12,atol=1e-9)
    tail=np.c_[mu,2*density[:,T>=2000].sum(1),pieces[:,-1]]
    return density,tail,pieces


def mean_bins(values,q,edges,support,counts=None):
    bins=np.clip(np.searchsorted(edges,q,'right')-1,0,5);rows=[]
    for b in support:
        use=bins==b
        if counts is None:rows.append(values[use].mean(0))
        elif counts[use].sum():rows.append(np.average(values[use],axis=0,weights=counts[use]))
        else:return np.full(values.shape[1:],np.nan)
    return np.mean(rows,axis=0)


def bootstrap(a,edges,support,nparent,seed,span=None,block=64):
    rng=np.random.default_rng(seed);quarts=np.full((NBOOT,6,3,3),np.nan)
    moments=np.full((NBOOT,6),np.nan)
    values=np.c_[a['tail'],a['F']]
    for b in range(NBOOT):
        if span is None:
            c=np.bincount(rng.integers(nparent,size=nparent),minlength=nparent)[a['ids']]
        else:c=block_counts(a['ids'],span,block,rng)
        quarts[b]=per_bin(a['q'],a['scores'],edges,c)
        moments[b]=mean_bins(values,a['q'],edges,support,c)
    return quarts,moments


def main():
    assert not (OUT/'assessment.json').exists(),'Refuse overwrite'
    protocol=read(OUT/'native_protocol.json');design=read(OUT/'design.json');execution=read(OUT/'execution.json')
    assert {x['key'] for x in execution['jobs']}==set(protocol['entries'])
    edges=np.asarray(protocol['q_edges']);mc={};draws={};means={};noise={};report={'scope':__doc__,
        'code_sha256':sha(Path(__file__)),'protocol_sha256':sha(OUT/'native_protocol.json'),
        'selection_sha256':sha(OUT/'selection.json'),'methods':METHODS,'checks':{},'mean_time_slices_ns':SLICES.tolist(),
        'new_independent_mothers':32000,'uncertainty':'512 whole-mother draws for each old/new MC separately; shared model draws across segments. Data original-index block64 and128. Fixed fit, closure and noise parameters; no model/fit uncertainty or multiplicity correction.',
        'reference_role':'Old MC is reused for proposal calibration. Old/new histories are different and are not paired; old comparison is a descriptive reference.',
        'experimental_role':'Frozen main-band diagnostics and Q/PCA; first60 used in fitting, later previously explored segments only checked afterward; no experimental particle truth or new blind data.',
        'no_updated_PID_efficiency':'This study compares run3-Co/run12-Am waveforms, with their respective acquisition states. No matched-response ROC or experimental alpha/gamma efficiency inference.'}
    source_inputs={}
    for source,run,kind in SOURCES:
        key=f'run{run}_{source}';support=protocol['support_bins'][source]
        for i,name in enumerate(['old','new']):
            path=(PARENT/'validation' if name=='old' else OUT)/'arrays'/f'{key}.npz';a=load(path)
            density,tail,pieces=features(a['waves'],a['scores'])
            row={'ids':a['ids'],'q':a['q'],'scores':a['scores'],'F':a['metrics'],'density':density,
                'tail':tail,'pieces':pieces,'shape':a['shapes'],'raw':raw_shapes(a['waves'])}
            mc[(run,name)]=row
            qs,ms=bootstrap(row,edges,support,int(a['n_parent']),301029550+run*10+i)
            draws[f'{key}__{name}__quartiles']=qs;draws[f'{key}__{name}__moments']=ms
            means[f'{key}__{name}__density']=average_bins(density,a['q'],edges,support)
            means[f'{key}__{name}__ADC']=average_bins(row['raw'],a['q'],edges,support)
            source_inputs[str(path.relative_to(PROJECT))]=sha(path)
        print('MC bootstrap',source,flush=True)
        nraw=int(load(PROJECT/f'simulation/results/pid_20261029/arrays/measured_run{run}_first60.npz')['original_count'])
        borders=[0,int(.6*nraw),int(.8*nraw),nraw]
        for si,seg in enumerate(SEGMENTS):
            group=f'run{run}_{seg}';path=DATA/f'{group}.npz';scorepath=WIDTH/'arrays'/f'measured_{group}.npz'
            d0=load(path);ds=load(scorepath);np.testing.assert_array_equal(d0['ids'],ds['ids']);use=ds['main_band']
            density,tail,pieces=features(d0['waves'][use],ds['scores'][use])
            d={'ids':ds['ids'][use],'q':ds['q'][use],'scores':ds['scores'][use],'F':d0['metrics'][use,:,3],
                'tail':tail,'pieces':pieces,'density':density,'shape':d0['shapes'][use],'raw':raw_shapes(d0['waves'][use])}
            dp=per_bin(d['q'],d['scores'],edges);dt=average_bins(d['tail'],d['q'],edges,support)
            dF=average_bins(d['F'],d['q'],edges,support);dshape=average_bins(d['shape'],d['q'],edges,support)
            dADC=average_bins(d['raw'],d['q'],edges,support);den=rms_width(widths(dp),support)
            means[group+'__data_density']=average_bins(density,d['q'],edges,support);means[group+'__data_ADC']=dADC
            data_draws={}
            for size in [64,128]:
                qs,ms=bootstrap(d,edges,support,nraw,301029700+run*10+si*1000+size,(borders[si],borders[si+1]),size)
                data_draws[size]=(qs,ms);draws[f'{group}__block{size}__quartiles']=qs;draws[f'{group}__block{size}__moments']=ms
            row={'run':run,'source':source,'segment':seg,'experimental_n':len(d['ids']),'support_bins':support,
                'data_tail_metrics':dt.tolist(),'data_F':dF.tolist(),'data_quartiles':dp.tolist(),'models':{}}
            region=(TIME_NS>=-40)&(TIME_NS<=3000)
            for name in ['old','new']:
                a=mc[(run,name)];point=per_bin(a['q'],a['scores'],edges)
                am=average_bins(a['tail'],a['q'],edges,support);af=average_bins(a['F'],a['q'],edges,support)
                aq=draws[f'{key}__{name}__quartiles'];ab=draws[f'{key}__{name}__moments']
                ashape=average_bins(a['shape'],a['q'],edges,support);aADC=average_bins(a['raw'],a['q'],edges,support)
                delta_pieces=average_bins(a['pieces'],a['q'],edges,support)-average_bins(d['pieces'],d['q'],edges,support)
                np.testing.assert_allclose(delta_pieces.sum(),am[0]-dt[0],atol=1e-9)
                item={'n_MC':len(a['ids']),'bin_counts':np.histogram(a['q'],edges)[0].tolist(),'MC_quartiles':point.tolist(),
                    'IQR_ratio':(rms_width(widths(point),support)/den).tolist(),'mean_tail_metrics':am.tolist(),
                    'tail_difference':(am-dt).tolist(),'mean_time_slice_difference_ns':delta_pieces.tolist(),
                    'median_shift_in_data_IQR':np.mean(((point[:,1,:]-dp[:,1,:])/widths(dp))[support],0).tolist(),
                    'F_ratio':(af/dF).tolist(),'MC_F':af.tolist(),
                    'mean_shape_normalized_RMSE':float(np.sqrt(np.mean((ashape[region]-dshape[region])**2))/dshape.max()),
                    'mean_ADC_normalized_RMSE':float(np.sqrt(np.mean((aADC[region]-dADC[region])**2))/dADC.max()),
                    'mean_ADC_peak_ratio':float(aADC.max()/dADC.max()),'uncertainty_by_data_block':{}}
                for size in [64,128]:
                    dq,db=data_draws[size]
                    br=rms_width(widths(aq),support)/rms_width(widths(dq),support)
                    dm=ab[:,:3]-db[:,:3];fr=ab[:,3:]/db[:,3:]
                    item['uncertainty_by_data_block'][str(size)]={'IQR_ratio_CI95':[interval(br[:,j]) for j in range(3)],
                        'tail_difference_CI95':[interval(dm[:,j]) for j in range(3)],
                        'F_ratio_CI95':[interval(fr[:,j]) for j in range(3)]}
                row['models'][name]=item
            oldcheck=read(PARENT/'validation/assessment.json')['checks'][group]
            np.testing.assert_allclose(row['models']['old']['IQR_ratio'],oldcheck['IQR_ratio'],rtol=1e-12,atol=1e-12)
            np.testing.assert_allclose(row['models']['old']['F_ratio'],oldcheck['F_ratio'],rtol=1e-12,atol=1e-12)
            report['checks'][group]=row
            for p in [path,scorepath]:source_inputs[str(p.relative_to(PROJECT))]=sha(p)
            new=row['models']['new'];print(group,'mean ns',round(new['tail_difference'][0],3),
                'tail fraction',round(new['tail_difference'][1],5),'width',np.round(new['IQR_ratio'],3),
                'ADC NRMSE%',round(100*new['mean_ADC_normalized_RMSE'],3),'F',np.round(new['F_ratio'],3),flush=True)
    report['inputs_sha256']=source_inputs
    np.savez_compressed(OUT/'bootstrap.npz',**draws)
    np.savez_compressed(OUT/'means.npz',gate_time_ns=T,time_ns=TIME_NS,**means)
    save(OUT/'assessment.json',finite(report))


if __name__=='__main__':main()
