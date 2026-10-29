"""Paired, pretrigger-conditioned run12 noise checks with unique MC mothers."""
from pathlib import Path
import numpy as np
from prepare_run12_noise_state import OUT,PARENT,PROJECT,DATA,WIDTH,SEGMENTS,VARIANTS
from prepare_pid import read
from extract_pid import load
from reference import TIME_NS
from analyze_npe_validation import raw_shapes
from analyze_bandwidth_validation import per_bin,widths,rms_width,interval
from fit_pid_bandwidth import average_bins
from characterize_readout import periodogram,covariance
from diagnose_pid_bandwidth import METHODS
from publication_stats import save,sha,block_counts

VIEWS=['reselected','fixed_ids','fixed_anchor']
NBOOT=512


def finite(obj):
    if isinstance(obj,float) and not np.isfinite(obj):return None
    if isinstance(obj,dict):return {k:finite(v) for k,v in obj.items()}
    if isinstance(obj,(tuple,list)):return [finite(v) for v in obj]
    return obj


def combine(segment,variant,arrays,assignments,view):
    """Exactly one response per original mother, never two fold replicas."""
    entries=[('baseline',None)] if variant=='baseline' else [(f'{segment}_eval{f}_{variant}',f) for f in [0,1]]
    result=[]
    for key,fold in entries:
        a=arrays[key];use=a['common' if view=='reselected' else 'fixed'].copy()
        if fold is not None:use&=assignments[segment][a['ids']]==fold
        row={k:a[k][use] for k in ['ids','scores','waves','baseline','baseline_rms','metrics','shapes','raw_shapes','anchor']}
        row['q']=a['q' if view=='reselected' else 'original_q'][use]
        if view=='fixed_anchor':
            idx=np.searchsorted(a['fixed_ids'],row['ids'])
            np.testing.assert_array_equal(a['fixed_ids'][idx],row['ids'])
            row['scores']=a['fixed_anchor_scores'][idx]
        result.append(row)
    merged={k:np.concatenate([r[k] for r in result]) for k in result[0]}
    order=np.argsort(merged['ids']);merged={k:v[order] for k,v in merged.items()}
    assert len(np.unique(merged['ids']))==len(merged['ids'])
    if view!='reselected':np.testing.assert_array_equal(merged['ids'],arrays['baseline']['fixed_ids'])
    return merged


def bin_counts(q,edges):
    return np.bincount(np.clip(np.searchsorted(edges,q,'right')-1,0,5),minlength=6).tolist()


def noise_summary(waves):
    pre=waves[:,:256].astype(float);rms=pre[:,:128].std(1)
    freq,power=periodogram(pre);cov=covariance(pre).mean(0)
    return {'n':len(pre),'RMS_p10_p50_p90':np.quantile(rms,[.1,.5,.9]).tolist(),
            'mean_variance':float(np.mean(rms**2)),
            'baseline_p10_p50_p90':np.quantile(pre[:,:128].mean(1),[.1,.5,.9]).tolist()},freq,power.mean(0),cov


def main():
    assert not (OUT/'assessment.json').exists(),'Refuse overwrite of a completed assessment'
    protocol=read(OUT/'protocol.json');execution=read(OUT/'execution.json')
    assert {r['key'] for r in execution['jobs']}==set(protocol['entries'])
    edges=np.asarray(protocol['q_edges']);support=protocol['support_bins']
    arrays={k:load(OUT/'arrays'/f'{k}.npz') for k in protocol['entries']}
    assignments=load(OUT/'mother_folds.npz');n=protocol['unique_parent_mothers']
    combined={};points={};draws={};means={};noise={};curves={};data={};data_points={};data_draws={}
    fixed_ids=arrays['baseline']['fixed_ids'];checks={};pairing={}
    original_n=int(load(PROJECT/'simulation/results/pid_20261029/arrays/measured_run12_first60.npz')['original_count'])
    borders=[0,int(.6*original_n),int(.8*original_n),original_n]
    for s,seg in enumerate(SEGMENTS):
        old=load(DATA/f'run12_{seg}.npz');scores=load(WIDTH/'arrays'/f'measured_run12_{seg}.npz')
        np.testing.assert_array_equal(old['ids'],scores['ids']);use=scores['main_band']
        data[seg]={'ids':scores['ids'][use],'q':scores['q'][use],'scores':scores['scores'][use],
            'F':average_bins(old['metrics'][use,:,3],scores['q'][use],edges,support),
            'shape':average_bins(old['shapes'][use],scores['q'][use],edges,support),
            'raw':average_bins(raw_shapes(old['waves'][use]),scores['q'][use],edges,support)}
        d=data[seg];data_points[seg]=per_bin(d['q'],d['scores'],edges)
        means[seg+'__data_ADC']=d['raw'];means[seg+'__data_shape']=d['shape']
        noise[seg]={}
        nr,freq,power,cov=noise_summary(old['waves']);noise[seg]['data']=nr
        curves[seg+'__data_PSD']=power;curves[seg+'__data_covariance']=cov;curves['frequency_mhz']=freq
        for variant in VARIANTS:
            for view in VIEWS:
                key=f'{seg}__{variant}__{view}'
                a=combine(seg,variant,arrays,assignments,view);combined[key]=a
                points[key]=per_bin(a['q'],a['scores'],edges)
                draws[key]=np.full((NBOOT,6,3,3),np.nan)
            fixed=combined[f'{seg}__{variant}__fixed_ids'];selected=combined[f'{seg}__{variant}__reselected']
            noise[seg][variant]={'folds':{}}
            nr,_,power,cov=noise_summary(fixed['waves']);noise[seg][variant]['pooled']=nr
            curves[f'{seg}__{variant}__PSD']=power;curves[f'{seg}__{variant}__covariance']=cov
            for fold in [0,1]:
                du=(old['ids']//64)%2==fold;mu=assignments[seg][fixed['ids']]==fold
                dr,_,dp,dc=noise_summary(old['waves'][du]);mr,_,mp,mc=noise_summary(fixed['waves'][mu])
                noise[seg][variant]['folds'][str(fold)]={'data':dr,'MC':mr,
                    'mean_variance_ratio':mr['mean_variance']/dr['mean_variance'],
                    'median_RMS_ratio':mr['RMS_p10_p50_p90'][1]/dr['RMS_p10_p50_p90'][1]}
                curves[f'{seg}__{variant}__fold{fold}__data_PSD']=dp
                curves[f'{seg}__{variant}__fold{fold}__MC_PSD']=mp
            pairing[f'{seg}__{variant}']={'selected_n':len(selected['ids']),'fixed_n':len(fixed_ids),
                'gained':np.setdiff1d(selected['ids'],fixed_ids).tolist(),
                'lost':np.setdiff1d(fixed_ids,selected['ids']).tolist(),
                'assigned_fixed_mothers_per_fold':np.bincount(assignments[seg][fixed_ids],minlength=2).tolist(),
                'unique_parent_ids':True}
        for size in [64,128]:
            rng=np.random.default_rng(3010291210+s*100+size)
            boot=np.full((NBOOT,6,3,3),np.nan)
            for b in range(NBOOT):
                c=block_counts(d['ids'],(borders[s],borders[s+1]),size,rng)
                boot[b]=per_bin(d['q'],d['scores'],edges,c)
            data_draws[f'{seg}__block{size}']=boot
    rng=np.random.default_rng(3010291250)
    for b in range(NBOOT):
        counts=np.bincount(rng.integers(n,size=n),minlength=n)
        for key,a in combined.items():draws[key][b]=per_bin(a['q'],a['scores'],edges,counts[a['ids']])
        if b%128==0:print('paired bootstrap',b,flush=True)
    region=(TIME_NS>=-40)&(TIME_NS<=3000)
    for seg in SEGMENTS:
        d=data[seg];dp=data_points[seg];den=rms_width(widths(dp),support)
        checks[seg]={'experimental_n':len(d['ids']),'data_bin_counts':bin_counts(d['q'],edges),
            'data_quartiles':dp.tolist(),'data_F_means':d['F'].tolist(),'variants':{}}
        for variant in VARIANTS:
            views={}
            for view in VIEWS:
                key=f'{seg}__{variant}__{view}';basekey=f'{seg}__baseline__{view}'
                a=combined[key];point=points[key]
                ratio=rms_width(widths(point),support)/den
                delta=ratio-rms_width(widths(points[basekey]),support)/den
                row={'n_MC':len(a['ids']),'bin_counts':bin_counts(a['q'],edges),'MC_quartiles':point.tolist(),
                    'IQR_ratio':ratio.tolist(),'paired_delta_ratio':delta.tolist(),
                    'median_shift_in_data_IQR':np.mean(((point[:,1,:]-dp[:,1,:])/widths(dp))[support],0).tolist(),
                    'uncertainty_by_data_block':{}}
                for size in [64,128]:
                    bd=rms_width(widths(data_draws[f'{seg}__block{size}']),support)
                    br=rms_width(widths(draws[key]),support)/bd
                    bdelta=br-rms_width(widths(draws[basekey]),support)/bd
                    row['uncertainty_by_data_block'][str(size)]={'IQR_ratio_CI95':[interval(br[:,j]) for j in range(3)],
                        'paired_delta_ratio_CI95':[interval(bdelta[:,j]) for j in range(3)]}
                if view!='fixed_anchor':
                    shape=average_bins(a['shapes'],a['q'],edges,support)
                    raw=average_bins(a['raw_shapes'],a['q'],edges,support)
                    F=average_bins(a['metrics'],a['q'],edges,support)
                    row.update(F_ratio=(F/d['F']).tolist(),MC_F_means=F.tolist(),
                        mean_shape_normalized_RMSE=float(np.sqrt(np.mean((shape[region]-d['shape'][region])**2))/d['shape'].max()),
                        mean_ADC_normalized_RMSE=float(np.sqrt(np.mean((raw[region]-d['raw'][region])**2))/d['raw'].max()),
                        mean_ADC_peak_ratio=float(raw.max()/d['raw'].max()))
                    means[key+'__ADC']=raw;means[key+'__shape']=shape
                views[view]=row
            checks[seg]['variants'][variant]=views
            print(seg,variant,'IQR',np.round(views['reselected']['IQR_ratio'],4),
                  'fixed',np.round(views['fixed_ids']['IQR_ratio'],4),
                  'baseline RMS',round(noise[seg][variant]['pooled']['RMS_p10_p50_p90'][1],3),flush=True)
        old=read(PARENT/'validation/assessment.json')['checks'][f'run12_{seg}']
        np.testing.assert_allclose(checks[seg]['variants']['baseline']['reselected']['IQR_ratio'],old['IQR_ratio'],rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(checks[seg]['variants']['baseline']['reselected']['F_ratio'],old['F_ratio'],rtol=1e-12,atol=1e-12)
    np.savez_compressed(OUT/'bootstrap.npz',**draws,**data_draws)
    np.savez_compressed(OUT/'means.npz',time_ns=TIME_NS,**means)
    np.savez_compressed(OUT/'noise_curves.npz',**curves)
    np.savez_compressed(OUT/'combined_scores.npz',**{f'{k}__{field}':a[field] for k,a in combined.items() for field in ['ids','q','scores','anchor']})
    report={'scope':__doc__,'code_sha256':sha(Path(__file__)),'protocol_sha256':sha(OUT/'protocol.json'),
        'methods':METHODS,'checks':checks,'noise':noise,'pairing':pairing,
        'unique_parent_mothers':n,'new_native_mothers':0,'old_common_MC':len(fixed_ids),
        'experimental_original_events':original_n,'support_bins':support,'q_edges':edges.tolist(),
        'uncertainty':'512 shared whole-mother draws, data raw-index blocks64 and128; paired comparisons between all response variants. Fits, kernels, fold assignment and all earlier parameters fixed; pointwise intervals without multiplicity adjustment.',
        'noise_population':'All frozen measured waveforms, no PID/main-band cut; model uses fixed original common IDs assigned exactly once to a fold. Per-fold test data were excluded from that fold noise fit. PSD and covariance are descriptive, finite512ns estimates.',
        'PID_population':'Frozen measured main-band0.4 diagnostic; not certified alpha truth. MC commonQ formally reselected or fixed original IDs/Q. Fixed-anchor uses original ADC anchor, with candidate baseline.',
        'crossfit_limit':'Only new pretrigger fit is cross-fitted; original pulse-response calibration used first60. State conditioning uses later-segment pretrigger data and is not a blind time forecast; adjacent blocks may remain dependent.',
        'prior_baseline_reproduced':True,'no_PID_based_candidate_refit':True}
    save(OUT/'assessment.json',finite(report))


if __name__=='__main__':main()
