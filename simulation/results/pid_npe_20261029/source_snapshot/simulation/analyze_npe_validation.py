"""Evaluate nominal-PDE full transport after the photon-statistics profile."""
from pathlib import Path
import numpy as np
from prepare_npe_profile import OUT,PARENT,WAVE,PROJECT
from prepare_pid import read
from extract_pid import load
from analyze_bandwidth_validation import per_bin,widths,rms_width,interval
from fit_pid_bandwidth import average_bins,conditional_cov
from reference import TIME_NS,wave_features
from pid_metrics import auc,operating_point
from publication_stats import sha,save


def raw_shapes(waves):
    f=wave_features(waves)
    return np.array([np.interp(TIME_NS,(np.arange(len(w))-t)*2,w.astype(float)-b)
        for w,t,b in zip(waves,f['t10_sample'],f['baseline'])])


def main():
    dest=OUT/'validation';assert not (dest/'assessment.json').exists(),'No overwrite'
    protocol=read(dest/'protocol.json');earlier=read(PARENT/'validation/assessment.json')
    edges=np.asarray(protocol['q_edges']);arrays={k:load(dest/'arrays'/f'{k}.npz') for k in protocol['entries']}
    (dest/'bootstrap').mkdir(exist_ok=True);(dest/'means').mkdir(exist_ok=True)
    draws={k:np.full((512,6,3,3),np.nan) for k in arrays};rng=np.random.default_rng(301029950)
    for source,n in [('Co60',8000),('Am241',24000)]:
        keys=[k for k,e in protocol['entries'].items() if e['source']==source]
        for b in range(512):
            counts=np.bincount(rng.integers(n,size=n),minlength=n)
            for k in keys:
                a=arrays[k];draws[k][b]=per_bin(a['q'],a['scores'],edges,counts[a['ids']])
        for k in keys:np.savez_compressed(dest/'bootstrap'/f'{k}.npz',quartiles=draws[k])
    report={'scope':__doc__,'code_sha256':sha(Path(__file__)),'protocol_sha256':sha(dest/'protocol.json'),
      'methods':earlier['methods'],'checks':{},'common_run12_PID':{},
      'comparison':'Previous noise-only candidate vs new effective-photon/transfer candidate. Different native seeds; no paired old/new transport claim.',
      'uncertainty':'New MC whole-mother bootstrap512, shared across same-source response states. Reuse frozen original-index block64 experimental draws. Fixed parameters; does not include fit uncertainty or parameter degeneracy.',
      'equal_bin_summary':'RMS IQR over previously frozen supported Q bins. Original experimental main-band0.4 diagnostic, no experimental particle truth.'}
    for run,source in [(3,'Co60'),(12,'Am241'),(18,'Am241')]:
        key=f'run{run}_{source}';a=arrays[key];point=per_bin(a['q'],a['scores'],edges)
        support=read(PARENT/'covariance_fit/fit.json')['states'][f'run{run}']['support_bins']
        new_cov=conditional_cov(a['scores'],a['q'],edges,support)
        new_shape=average_bins(a['shapes'],a['q'],edges,support)
        new_raw=average_bins(raw_shapes(a['waves']),a['q'],edges,support)
        new_F=average_bins(a['metrics'],a['q'],edges,support)
        for segment in ['first60','middle20','last20']:
            data_key=f'run{run}_{segment}';old=earlier['checks'][data_key]
            d=load(PARENT/'arrays'/f'measured_{data_key}.npz');use=d['main_band']
            original=load((WAVE/'run18' if run==18 else PROJECT/'simulation/results/signal_fluctuations_20261028')/(data_key+'.npz'))
            dpoint=per_bin(d['q'][use],d['scores'][use],edges)
            ddraws=load(PARENT/'validation/bootstrap'/f'{data_key}.npz')['quartiles']
            data_cov=conditional_cov(d['scores'][use],d['q'][use],edges,support,robust=True)
            data_shape=average_bins(original['shapes'][use],d['q'][use],edges,support)
            data_raw=average_bins(raw_shapes(original['waves'][use]),d['q'][use],edges,support)
            data_F=average_bins(original['metrics'][use,:,3],d['q'][use],edges,support)
            ratio=rms_width(widths(point),support)/rms_width(widths(dpoint),support)
            ratios=rms_width(widths(draws[key]),support)/rms_width(widths(ddraws),support)
            median_shift=(point[:,1,:]-dpoint[:,1,:])/widths(dpoint)
            region=(TIME_NS>=-40)&(TIME_NS<=3000)
            row={'n_MC':len(a['q']),'n_measured':int(use.sum()),'support_bins':support,
                'previous_noise_only_IQR_ratio':old['after']['IQR_ratio'],
                'previous_noise_only_IQR_ratio_CI95':old['after']['IQR_ratio_CI95'],
                'IQR_ratio':ratio.tolist(),'IQR_ratio_CI95':[interval(ratios[:,j]) for j in range(3)],
                'median_shift_in_data_IQR':np.mean(median_shift[support],axis=0).tolist(),
                'RMS_median_shift_in_data_IQR':np.sqrt(np.mean(median_shift[support]**2,axis=0)).tolist(),
                'MC_covariance':new_cov.tolist(),'data_covariance_robust':data_cov.tolist(),
                'robust_SD_ratio':np.sqrt(np.diag(new_cov)/np.diag(data_cov)).tolist(),
                'MC_F_means':new_F.tolist(),'data_F_means':data_F.tolist(),'F_ratio':(new_F/data_F).tolist(),
                'mean_shape_normalized_RMSE':float(np.sqrt(np.mean((new_shape[region]-data_shape[region])**2))/data_shape.max()),
                'mean_ADC_normalized_RMSE':float(np.sqrt(np.mean((new_raw[region]-data_raw[region])**2))/data_raw.max()),
                'mean_ADC_peak_ratio':float(new_raw.max()/data_raw.max()),
                'baseline_RMS_median_MC':float(np.median(a['baseline_rms'])),
                'baseline_RMS_median_data':float(np.median(wave_features(original['waves'][use])['baseline_rms'])),
                'per_Q':[]}
            for b in range(6):
                nq=int(np.sum((a['q']>=edges[b])&(a['q']<=edges[b+1])))
                br=widths(draws[key])[:,b]/widths(ddraws)[:,b]
                row['per_Q'].append({'bin':b,'in_summary':b in support,'n_MC':nq,
                  'MC_quartiles':point[b].tolist(),'data_quartiles':dpoint[b].tolist(),
                  'ratio_CI95':[interval(br[:,j]) for j in range(3)]})
            np.savez_compressed(dest/'means'/f'{data_key}.npz',time_ns=TIME_NS,MC_shape=new_shape,data_shape=data_shape,
                MC_ADC=new_raw,data_ADC=data_raw,MC_F=new_F,data_F=data_F)
            report['checks'][data_key]=row
            print(data_key,'IQR',np.round(ratio,3),'F',np.round(new_F/data_F,3),'ADCpeak',round(row['mean_ADC_peak_ratio'],4),flush=True)
    for source in ['Co60','Am241']:
        cfg=read(PROJECT/protocol['entries'][f'run12_{source}']['config'])
        assert cfg['sipm']['sensor']['constant_pde']==.3
        if source=='Co60':previous=cfg
        else:
            for k in ['sipm','readout_response','electronics_noise','signal_dependent_noise']:assert cfg[k]==previous[k]
    alpha=arrays['run12_Am241'];gamma=arrays['run12_Co60']
    sa=alpha['scores'][alpha['truth']==0];sg=gamma['scores'][gamma['truth']==1]
    cuts=read(PROJECT/'simulation/results/pid_20261029/thresholds.json')
    for j,m in enumerate(earlier['methods']):
        threshold=cuts['methods'][m]['threshold']
        report['common_run12_PID'][m]={'AUC':auc(sa[:,j],sg[:,j]),'fixed_previous_threshold':threshold,
             'operating':operating_point(sa[:,j],sg[:,j],threshold)}
    report['common_run12_note']='Same final run12 response for both sources; selected energy-ancestry90% truth only. Fixed previous thresholds, not new operating-point optimization or experimental efficiency.'
    def finite(obj):
        if isinstance(obj,float) and not np.isfinite(obj):return None
        if isinstance(obj,dict):return {k:finite(v) for k,v in obj.items()}
        if isinstance(obj,(tuple,list)):return [finite(x) for x in obj]
        return obj
    save(dest/'assessment.json',finite(report))


if __name__=='__main__':main()
