"""Read-only scientific audit to prioritize the next CsI/SiPM analysis."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/sipm-next-steps-mpl')
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reference import PROJECT,wave_features
from prepare_npe_profile import OUT as NPE,PARENT as WIDTH,WAVE
from prepare_pid import read
from extract_pid import load
from diagnose_pid_bandwidth import linear_scores,recent
from fit_pid_bandwidth import average_bins
from publication_stats import save,sha

OUT=PROJECT/'analysis/paper_baseline/results/next_steps_20261029'
WINDOWS=np.array([-40,0,200,500,1000,1500,2000,3000])


def summarize(waves,q,edges,support,pca):
    f=wave_features(waves);anchor=np.floor(f['t10_sample']).astype(int)+1
    scores,charge,gate=linear_scores(waves,anchor,f['baseline'],pca)
    times=np.arange(-20,1500)*2.
    pieces=np.array([(gate[:,(times>=lo)&(times<hi)]*times[(times>=lo)&(times<hi)]).sum(1)/charge
        for lo,hi in zip(WINDOWS[:-1],WINDOWS[1:])]).T
    np.testing.assert_allclose(pieces.sum(1),-1000*scores[:,1],rtol=1e-12,atol=1e-9)
    means=average_bins(pieces,q,edges,support)
    alternate=linear_scores(waves,anchor,recent(waves,anchor),pca)[0]
    variants={'original':scores,'recent_pretrigger_baseline':alternate}
    for shift in [-1,1]:
        variants[f'anchor_shift_{shift:+d}_sample']=linear_scores(waves,anchor+shift,f['baseline'],pca)[0]
    return {'mean_time_contributions_ns':means.tolist(),'mean_time_ns':float(means.sum()),
        'variant_mean_time_ns':{k:float(-1000*average_bins(s[:,1],q,edges,support)) for k,s in variants.items()},
        'baseline_RMS_median':float(np.median(f['baseline_rms'])),
        'recent_minus_first128_baseline_mean_ADC':float(np.mean(recent(waves,anchor)-f['baseline'])),
        'recent_minus_first128_baseline_SD_ADC':float(np.std(recent(waves,anchor)-f['baseline'],ddof=1))},f


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    report=read(NPE/'validation/assessment.json');edges=np.asarray(read(NPE/'validation/protocol.json')['q_edges'])
    pca=read(PROJECT/'analysis/paper_baseline/results/publication_20261028/point.json')['windows']['3000']['pca']
    inputs=[NPE/'record.json',NPE/'validation/assessment.json',NPE/'validation/protocol.json',
        PROJECT/'analysis/paper_baseline/MANUSCRIPT_DRAFT.md',WIDTH/'covariance_fit/fit.json',
        PROJECT/'analysis/paper_baseline/results/publication_20261028/point.json']
    audit={'scope':__doc__,'code_sha256':sha(Path(__file__)),'checkpoint_commit':'082607a2a6bdd3bb78fa3e0f92b716aaaefa0a30',
        'purpose':'Prioritization on existing frozen data; no new parameter fit, native simulation, cohort change, or particle relabeling.',
        'windows_ns':WINDOWS.tolist(),'q_edges':edges.tolist(),'groups':{},'MC':{},
        'limits':['Mean-time decomposition is a point estimate with exact additive sample weights, not a causal mechanism or a confidence interval.',
            'Mean time and earlier median-band displacement are different summaries; do not substitute one for the other.',
            'Alternative baseline and +/-one-sample anchor retain original IDs and Q bins; they are diagnostics, not a changed official analysis.',
            'Block-parity counts demonstrate cross-fit feasibility only. Neighboring acquisition blocks may remain correlated; block separation is not proof of statistical independence.',
            'All measured segments have been explored before. Noise-conditioned comparisons are not blind forecasts.']}
    fig,axes=plt.subplots(1,3,figsize=(13,4.2))
    for ax,(run,source,nraw) in zip(axes,[(3,'Co60',4776),(12,'Am241',7516),(18,'Am241',156966)]):
        support=report['checks'][f'run{run}_last20']['support_bins']
        mp=NPE/'validation/arrays'/f'run{run}_{source}.npz';m=load(mp);inputs.append(mp)
        ms,mf=summarize(m['waves'],m['q'],edges,support,pca);audit['MC'][str(run)]=ms
        for seg,color in zip(['first60','middle20','last20'],['#2876b4','#d47820','#278365']):
            key=f'run{run}_{seg}'
            path=(WAVE/'run18' if run==18 else PROJECT/'simulation/results/signal_fluctuations_20261028')/(key+'.npz')
            dp=WIDTH/'arrays'/f'measured_{key}.npz';inputs.extend([path,dp])
            d=load(path);a=load(dp);np.testing.assert_array_equal(d['ids'],a['ids']);use=a['main_band']
            ds,df=summarize(d['waves'][use],d['q'][use],edges,support,pca)
            ids=d['ids'][use];fold=(ids//64)%2
            counts=[[int(np.sum((fold==f)&(d['q'][use]>=edges[b])&(d['q'][use]<edges[b+1]))) for b in support] for f in [0,1]]
            delta=np.asarray(ms['mean_time_contributions_ns'])-np.asarray(ds['mean_time_contributions_ns'])
            row={'run':run,'segment':seg,'support_bins':support,'n':len(ids),'block64_parity_n':np.bincount(fold,minlength=2).tolist(),
                'block64_parity_supported_bin_counts':counts,'data':ds,
                'MC_minus_data_mean_time_contributions_ns':delta.tolist(),
                'MC_minus_data_mean_time_ns':float(delta.sum()),
                'MC_minus_data_variant_mean_time_ns':{k:ms['variant_mean_time_ns'][k]-v for k,v in ds['variant_mean_time_ns'].items()},
                'bandwidth_ratio':report['checks'][key]['IQR_ratio'],
                'bandwidth_ratio_CI95':report['checks'][key]['IQR_ratio_CI95']}
            audit['groups'][key]=row
            ax.scatter(ids/nraw,df['baseline_rms'],s=5,alpha=.22,color=color,rasterized=True,label=seg)
            print(key,'baseline',round(ds['baseline_RMS_median'],3),'mean shift ns',round(delta.sum(),3),
                  'recent baseline shift',round(row['MC_minus_data_variant_mean_time_ns']['recent_pretrigger_baseline'],3),flush=True)
        ax.axhline(ms['baseline_RMS_median'],color='k',ls='--',lw=1,label='Frozen MC median')
        ax.set_title(f'Run{run} / {source}');ax.set_xlabel('Original acquisition index / run entries')
        ax.set_ylim(0,15);ax.grid(alpha=.15)
    axes[0].set_ylabel('First128 baseline RMS (ADC)');axes[0].legend(fontsize=8)
    fig.suptitle('Existing selected waveforms: acquisition-state noise diagnostic',y=1.02)
    fig.text(.5,-.02,'Frozen cohort and main-band diagnostic. Index fraction is not live time; y-limit focuses on the core, not all transients.',ha='center',fontsize=9)
    fig.tight_layout()
    for ext in ['png','svg','pdf']:fig.savefig(OUT/f'noise_state_review.{ext}',dpi=170,bbox_inches='tight')
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.1))
    labels=[f'{a:g}:{b:g}' for a,b in zip(WINDOWS[:-1],WINDOWS[1:])]
    for ax,seg in zip(axes,['first60','last20']):
        r=audit['groups'][f'run3_{seg}'];delta=r['MC_minus_data_mean_time_contributions_ns']
        ax.bar(np.arange(len(delta)),delta,color=['#d47820' if x>=0 else '#2876b4' for x in delta])
        ax.axhline(0,color='k',lw=.7);ax.set_xticks(np.arange(len(delta)),labels,rotation=30)
        ax.set_title(f'Co / {seg}; total = {r["MC_minus_data_mean_time_ns"]:.2f} ns')
        ax.set_ylabel('Contribution to MC - data mean time (ns)');ax.set_xlabel('Time slice relative to original anchor (ns)')
    fig.suptitle('Exact mean-time decomposition at fixed IDs and Q bins',y=1.02)
    fig.text(.5,-.025,'Point estimates for prioritization. Signed contributions can cancel; this is not a unique physical noise decomposition.',ha='center',fontsize=9)
    fig.tight_layout()
    for ext in ['png','svg','pdf']:fig.savefig(OUT/f'co_mean_time_review.{ext}',dpi=170,bbox_inches='tight')
    plt.close(fig)
    gamma=report['common_run12_PID']['qdc']['operating']['gamma_leakage']
    audit['MC_statistics']={'current_common_gamma':gamma['total'],'current_leakage_upper95':gamma['one_sided95_upper'],
        'zero_leakage_formula':'upper=1-0.05**(1/n), one-sided95%; conditional on fixed selection and classifier',
        'gamma_needed_if_zero_for_upper_1e_minus3':int(np.ceil(np.log(.05)/np.log1p(-.001))),
        'note':'Independent accepted gamma histories, not total generated mother decays; no experimental efficiency claim.'}
    audit['manuscript_gap']={'path':'analysis/paper_baseline/MANUSCRIPT_DRAFT.md',
        'current_stage':'2026-10-28 model discussion; section4.4 still describes f=0.10 and ~81.6%/-25.2% short-scale residuals.',
        'needed':'Versioned update linking the new waveform, PID, and photon-statistics records; preserve old benchmark and frozen experimental main analysis.'}
    audit['inputs_sha256']={str(p.relative_to(PROJECT)):sha(p) for p in inputs}
    save(OUT/'audit.json',audit)
    print('Saved review:',OUT,flush=True)


if __name__=='__main__':main()
