"""Plot joint tail calibration, fresh native checks and unresolved tradeoffs."""
import os,csv
os.environ.setdefault('MPLCONFIGDIR','/tmp/sipm-co-tail-mpl')
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fit_co_tail import OUT,T,SOURCES,CASES,PRIMARY
from prepare_pid import read
from extract_pid import load
from publication_stats import save,sha

COLORS={'data':'#222222','old':'#64748b','new':'#c94c35','legacy_objective':'#9d7b9e',
        'tail_old_bound':'#ce9f23','tail_relaxed_bound':'#c94c35','tail_half_weight':'#27856f'}
LABELS={'data':'Measured','old':'Previous model','new':'New native tail candidate',
    'legacy_objective':'Legacy objective','tail_old_bound':'Tail / original bound',
    'tail_relaxed_bound':'Tail / relaxed bound','tail_half_weight':'Tail residual x0.5'}
SEGS=['first60','middle20','last20']


def finish(fig,name,caption,legend=None,ncol=3):
    top=1.
    if legend is not None:
        h,l=legend.get_legend_handles_labels();fig.legend(h,l,loc='upper center',ncol=ncol,fontsize=9,frameon=False)
        top=.94 if fig.get_figheight()>6 else .88
    fig.tight_layout(rect=(0,.045,1,top));fig.text(.5,.008,caption,ha='center',fontsize=8.5)
    for ext in ['png','svg','pdf']:fig.savefig(OUT/'figures'/f'{name}.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)


def bar(ax,x,y,ci,color):
    low,high=ci;ax.vlines(x,low,high,color=color,lw=1.2)
    ax.hlines([low,high],x-.025,x+.025,color=color,lw=1.2)
    ax.plot(x,y,'o',color=color,ms=5)


def main():
    report=read(OUT/'assessment.json');sel=read(OUT/'selection.json')
    proposals=load(OUT/'proposal_curves.npz');means=load(OUT/'means.npz')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for col,(source,run,kind) in enumerate(SOURCES):
        data=proposals[source+'_data']
        axes[0,col].plot(T,data,color=COLORS['data'],lw=2,label='Measured first60')
        axes[0,col].plot(T,proposals[source+'_native_baseline'],color=COLORS['old'],lw=1,label='Previous native MC')
        for case in CASES:
            name=case['name'];y=proposals[name+'__'+source]
            axes[0,col].plot(T,y,color=COLORS[name],lw=1.6 if name==PRIMARY else 1,label=LABELS[name],ls='--' if name=='tail_old_bound' else '-')
            axes[1,col].plot(T,100*(y-data)/data.max(),color=COLORS[name],lw=1.3)
        axes[0,col].set(title=f'{source}: proposal calibration',xlabel='Time from integer ADC anchor (ns)',ylabel='Mean gate-normalized density (1/ns)')
        axes[1,col].set(title=f'{source}: proposal residual',xlabel='Time from integer ADC anchor (ns)',ylabel='Difference / measured peak (%)')
        axes[1,col].axhline(0,color='k',lw=.7);axes[1,col].axvspan(2000,3000,color='#ddd',alpha=.3)
    finish(fig,'tail_training_families','Training-only analytic proposals plus fixed native closure. Curves are not inserted into Geant4 output; all four fit families retained.',axes[0,0],3)

    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for i,(source,run,kind) in enumerate(SOURCES):
        for j,(metric,scale,label) in enumerate([(0,1.,'MC - data mean time (ns)'),(1,100.,'MC - data late charge (percentage points)')]):
            ax=axes[i,j]
            for k,name in enumerate(['old','new']):
                for s,seg in enumerate(SEGS):
                    r=report['checks'][f'run{run}_{seg}']['models'][name]
                    ci=np.array(r['uncertainty_by_data_block']['64']['tail_difference_CI95'][metric]['estimate95'])*scale
                    bar(ax,s+(k-.5)*.17,r['tail_difference'][metric]*scale,ci,COLORS[name])
                ax.plot([],[],'o',color=COLORS[name],label=LABELS[name])
            ax.axhline(0,color='k',lw=.8,ls='--');ax.set_xticks(range(3),['First60','Middle20','Last20'])
            ax.set(title=f'{source} / run{run}',ylabel=label);ax.grid(axis='y',alpha=.15)
    finish(fig,'tail_native_moments','Late charge is the fraction in 2-3 us. Pointwise 95%:512 whole-mother / data block64 resamples; fit and closure fixed.',axes[0,0],2)

    fig,axes=plt.subplots(2,2,figsize=(12,8));time=means['time_ns'];region=(time>=-40)&(time<=3000)
    for col,(source,run,kind) in enumerate(SOURCES):
        key=f'run{run}_{source}';data=means[f'run{run}_last20__data_ADC']
        axes[0,col].plot(time[region],data[region],color=COLORS['data'],lw=2,label='Measured last20')
        for name in ['old','new']:
            y=means[f'{key}__{name}__ADC'];r=report['checks'][f'run{run}_last20']['models'][name]
            axes[0,col].plot(time[region],y[region],color=COLORS[name],lw=1.4,label=LABELS[name])
            axes[1,col].plot(time[region],100*(y[region]-data[region])/data.max(),color=COLORS[name],lw=1.2)
        axes[0,col].set(title=f'{source}: unscaled ADC mean',xlabel='Time from t10 (ns)',ylabel='Baseline-subtracted ADC')
        axes[1,col].set(title=f'{source}: mean waveform residual',xlabel='Time from t10 (ns)',ylabel='Difference / measured peak (%)')
        axes[1,col].axhline(0,color='k',lw=.6);axes[1,col].axvspan(2000,3000,color='#ddd',alpha=.25)
    finish(fig,'tail_native_waveforms','Last20 measured segments; equal weights in frozen Q bins. Fresh native transport; no per-event amplitude rescaling.',axes[0,0],3)

    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for col,(source,run,kind) in enumerate(SOURCES):
        for k,name in enumerate(['old','new']):
            r=report['checks'][f'run{run}_last20']['models'][name];u=r['uncertainty_by_data_block']['64']
            for j in range(3):
                bar(axes[0,col],j+(k-.5)*.16,r['IQR_ratio'][j],u['IQR_ratio_CI95'][j]['estimate95'],COLORS[name])
                bar(axes[1,col],j+(k-.5)*.16,r['F_ratio'][j],u['F_ratio_CI95'][j]['estimate95'],COLORS[name])
            axes[0,col].plot([],[],'o',color=COLORS[name],label=LABELS[name])
        axes[0,col].set(title=f'{source}: last20 PID widths',ylabel='Equal-Q RMS(IQR): MC / data')
        axes[1,col].set(title=f'{source}: last20 fluctuations',ylabel='Mean F: MC / data',xlabel='Scale (ns)')
        axes[0,col].set_xticks(range(3),['QDC','Mean time','PCA']);axes[1,col].set_xticks(range(3),['50','102','202'])
        for ax in axes[:,col]:ax.axhline(1,color='k',lw=.8,ls='--');ax.grid(axis='y',alpha=.15)
    finish(fig,'tail_native_pid_controls','Pointwise 95% with fixed response parameters. Old MC is a calibration reference; distinct native histories are not paired.',axes[0,0],2)

    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    labels=[f'{a:g}:{b:g}' for a,b in zip(report['mean_time_slices_ns'][:-1],report['mean_time_slices_ns'][1:])]
    for ax,seg in zip(axes,['first60','last20']):
        for k,name in enumerate(['old','new']):
            r=report['checks'][f'run3_{seg}']['models'][name]
            ax.bar(np.arange(7)+(k-.5)*.32,r['mean_time_slice_difference_ns'],width=.3,color=COLORS[name],label=f"{LABELS[name]}: total {r['tail_difference'][0]:+.2f} ns")
        ax.axhline(0,color='k',lw=.7);ax.set_xticks(range(7),labels,rotation=30)
        ax.set(title=f'Co / {seg}',xlabel='Time slice from original ADC anchor (ns)',ylabel='Contribution to MC - data mean time (ns)')
        ax.legend(fontsize=8)
    finish(fig,'tail_time_contributions','Exact additive gate-moment point estimates at equal Q-bin weights. Signed terms can cancel; no unique causal decomposition.')

    rows=[]
    for group,case in report['checks'].items():
        for name,m in case['models'].items():
            lo,hi=m['uncertainty_by_data_block']['64']['tail_difference_CI95'][0]['estimate95']
            rows.append({'group':group,'model':name,'n_MC':m['n_MC'],'mean_time_difference_ns':m['tail_difference'][0],
                'mean_time_CI95_low':lo,'mean_time_CI95_high':hi,'late_fraction_difference':m['tail_difference'][1],
                'late_moment_difference_ns':m['tail_difference'][2],'ADC_NRMSE_percent':100*m['mean_ADC_normalized_RMSE'],
                'ADC_peak_ratio':m['mean_ADC_peak_ratio'],**{f'{metric}_IQR_ratio':value for metric,value in zip(['QDC','mean_time','PCA'],m['IQR_ratio'])},
                **{f'F{scale}_ratio':value for scale,value in zip([50,102,202],m['F_ratio'])}})
    with (OUT/'validation_table.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    co=report['checks']['run3_last20']['models'];am=report['checks']['run12_last20']['models']
    conclusions=(f"完整Geant4后段检查：Co平均时间差由 {co['old']['tail_difference'][0]:+.2f} ns 变为 {co['new']['tail_difference'][0]:+.2f} ns，"
        f"2–3 µs电荷份额差由 {100*co['old']['tail_difference'][1]:+.3f} 变为 {100*co['new']['tail_difference'][1]:+.3f} 个百分点。"
        f"新候选Co/Am均ADC波形NRMSE分别为 {100*co['new']['mean_ADC_normalized_RMSE']:.2f}% / {100*am['new']['mean_ADC_normalized_RMSE']:.2f}%。"
        f"Am平均时间宽度比点估计由 {am['old']['IQR_ratio'][1]:.3f} 变为 {am['new']['IQR_ratio'][1]:.3f}，仍偏窄。"
        "宽度、起伏和参数边界仍需一起评估；本轮不自动替换默认配置。")
    save(OUT/'decision.json',{'conclusion_zh':conclusions,'promoted_to_default':False,
        'old_bound_vs_relaxed_bound_max_parameter_difference':float(np.max(np.abs(
            np.array(sel['cases']['tail_old_bound']['best']['parameters']['electron_tau_ns'])-
            np.array(sel['cases']['tail_relaxed_bound']['best']['parameters']['electron_tau_ns'])))),
        'alpha_slow_at_upper_bound':sel['chosen_parameters']['alpha_tau_ns'][1]>=14999.,
        'effective_alpha_yield_above_one':read(OUT/'configs/run12_Am241.json')['optical']['alpha_light_yield_ratio']>1,
        'parameter_warning':'Alpha slow decay reaches15us and analytic gate-area preservation raises effective alpha normalization above1. These are conditional finite-window parameters, not a measured decay constant or scintillation quenching factor.',
        'model_scope':'Training-objective sensitivity supported; lowered700ns electron bound is not driving this candidate because old/relaxed-bound optima agree. Full native mean-time and shape results can differ from analytic predictions.',
        'next_work':'Review joint native mean, PID width and F residuals before a constrained parameter-family study; retain the boundary/normalization tradeoff and run12 noise-state checks. No validation-driven retuning in this record.'})
    save(OUT/'figure_record.json',{'code_sha256':sha(Path(__file__)),'assessment_sha256':sha(OUT/'assessment.json'),
        'selection_sha256':sha(OUT/'selection.json'),'figures':['tail_training_families','tail_native_moments','tail_native_waveforms','tail_native_pid_controls','tail_time_contributions'],
        'table_rows':len(rows)})
    print(conclusions,flush=True)


if __name__=='__main__':main()
