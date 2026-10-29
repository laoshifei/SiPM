"""Plot all frozen noise variants and export their conditional comparisons."""
import csv,os
os.environ.setdefault('MPLCONFIGDIR','/tmp/sipm-run12-state-mpl')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from prepare_run12_noise_state import OUT,SEGMENTS,VARIANTS
from prepare_pid import read
from extract_pid import load
from publication_stats import save,sha
from pathlib import Path

COLORS={'data':'#222222','baseline':'#64748b','scale':'#e28b22','states':'#3182bd','spectrum_states':'#1b9e77'}
LABELS={'data':'Measured','baseline':'Original noise','scale':'RMS only','states':'Baseline + RMS states','spectrum_states':'States + spectrum'}
MLABELS=['Qshort / Qlong','Negative mean time','PCA']
SLABELS=['First 60%','Middle 20%','Last 20%']


def finish(fig,name,caption,legend_axis=None,legend_cols=4):
    top=1.
    if legend_axis is not None:
        handles,labels=legend_axis.get_legend_handles_labels()
        fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,1),ncol=legend_cols,fontsize=9,frameon=False)
        top=.94 if fig.get_figheight()>6 else .88
    fig.tight_layout(rect=(0,.045,1,top));fig.text(.5,.008,caption,ha='center',fontsize=8.5)
    for ext in ['png','svg','pdf']:fig.savefig(OUT/'figures'/f'{name}.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)


def bars(ax,x,y,interval,color,marker='o',fill=True):
    lo,hi=interval
    ax.vlines(x,lo,hi,color=color,lw=1)
    ax.hlines([lo,hi],x-.022,x+.022,color=color,lw=1)
    ax.plot(x,y,marker,color=color,mfc=color if fill else 'white',ms=5)


def main():
    report=read(OUT/'assessment.json');means=load(OUT/'means.npz');curves=load(OUT/'noise_curves.npz')
    (OUT/'figures').mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(2,2,figsize=(12,8.4))
    for j,key in enumerate(['data']+VARIANTS):
        offset=(j-2)*.12
        nr=[report['noise'][seg]['data' if key=='data' else key] for seg in SEGMENTS]
        if key!='data':nr=[r['pooled'] for r in nr]
        for i,r in enumerate(nr):
            q=r['RMS_p10_p50_p90'];bars(ax[0,0],i+offset,q[1],[q[0],q[2]],COLORS[key])
            q=r['baseline_p10_p50_p90'];bars(ax[1,1],i+offset,q[1],[q[0],q[2]],COLORS[key])
        ax[0,0].plot([],[],color=COLORS[key],marker='o',ls='',label=LABELS[key])
        prefix='last20__data_' if key=='data' else f'last20__{key}__'
        power=curves[prefix+'PSD'];freq=curves['frequency_mhz']
        ax[0,1].semilogy(freq[1:],power[1:],color=COLORS[key],lw=1.3,label=LABELS[key])
        cov=curves[prefix+'covariance']
        ax[1,0].plot(np.arange(len(cov))*2,cov/cov[0],color=COLORS[key],lw=1.4)
    ax[0,0].set(title='Pretrigger RMS distribution',ylabel='First 128 samples RMS (ADC)')
    ax[1,1].set(title='Pretrigger baseline distribution',ylabel='First 128 samples mean (ADC)')
    for a in [ax[0,0],ax[1,1]]:a.set_xticks(range(3),SLABELS);a.grid(axis='y',alpha=.15)
    ax[0,1].set(title='Last 20%: noise spectrum',xlabel='Frequency (MHz)',ylabel='PSD (ADC squared / MHz)',xlim=(0,250))
    ax[1,0].set(title='Last 20%: noise correlation',xlabel='Lag (ns)',ylabel='Normalized detrended covariance',xlim=(0,128));ax[1,0].axhline(0,color='k',lw=.5)
    finish(fig,'noise_state_closure','Bars: event p10 / median / p90, not confidence intervals. Cross-fitted pretrigger states; 512 ns PSD window; fixed MC IDs.',ax[0,0],5)

    fig,axes=plt.subplots(1,3,figsize=(13,4.6),sharey=True)
    for s,(seg,ax) in enumerate(zip(SEGMENTS,axes)):
        for k,v in enumerate(VARIANTS):
            row=report['checks'][seg]['variants'][v]['reselected']
            for j in range(3):bars(ax,j+(k-1.5)*.16,row['IQR_ratio'][j],row['uncertainty_by_data_block']['64']['IQR_ratio_CI95'][j]['estimate95'],COLORS[v])
            ax.plot([],[],color=COLORS[v],marker='o',ls='',label=LABELS[v])
        ax.axhline(1,color='k',ls='--',lw=.8);ax.set_xticks(range(3),['QDC','Mean time','PCA'])
        ax.set_title(SLABELS[s]);ax.grid(axis='y',alpha=.18)
    axes[0].set_ylabel('Equal-Q RMS(IQR): MC / measured')
    finish(fig,'noise_pid_widths','Pointwise 95%: 512 paired mother / data block64 resamples. Same 24000 mothers; fits fixed; formally reselected MC.',axes[0])

    fig,axes=plt.subplots(1,3,figsize=(13,4.6),sharey=True)
    views=[('reselected','o','Reselected IDs / Q'),('fixed_ids','s','Fixed IDs / Q'),('fixed_anchor','^','Fixed IDs / Q / anchor')]
    for j,ax in enumerate(axes):
        for k,v in enumerate(VARIANTS[1:]):
            for z,(view,marker,label) in enumerate(views):
                row=report['checks']['last20']['variants'][v][view]
                bars(ax,k+(z-1)*.17,row['paired_delta_ratio'][j],row['uncertainty_by_data_block']['64']['paired_delta_ratio_CI95'][j]['estimate95'],COLORS[v],marker,z==0)
        for view,marker,label in views:ax.plot([],[],marker,color='k',mfc='k' if view=='reselected' else 'white',ls='',label=label)
        ax.axhline(0,color='k',lw=.8,ls='--');ax.set_xticks(range(3),['RMS only','States','States +\nspectrum'])
        ax.set_title(MLABELS[j]);ax.grid(axis='y',alpha=.18)
    axes[0].set_ylabel('Last 20%: change in MC / data IQR ratio')
    finish(fig,'noise_pid_paired_controls','Change relative to original noise on the same mothers. Fixed IDs / Q and fixed-anchor controls separate selection and alignment effects.',axes[0],3)

    fig,ax=plt.subplots(2,2,figsize=(12,8));t=means['time_ns'];use=(t>=-40)&(t<=3000)
    d=means['last20__data_ADC'];ax[0,0].plot(t[use],d[use],color=COLORS['data'],lw=2,label='Measured')
    for k,v in enumerate(VARIANTS):
        m=means[f'last20__{v}__reselected__ADC'];row=report['checks']['last20']['variants'][v]['reselected']
        ax[0,0].plot(t[use],m[use],color=COLORS[v],lw=1.2,label=LABELS[v])
        ax[0,1].plot(t[use],100*(m[use]-d[use])/d.max(),color=COLORS[v],lw=1.2)
        ax[1,0].plot([50,102,202],row['F_ratio'],marker='o',color=COLORS[v],lw=1)
        ax[1,1].plot(np.arange(3)+(k-1.5)*.1,row['median_shift_in_data_IQR'],marker='o',ls='',color=COLORS[v])
    ax[0,0].set(title='Last 20%: unscaled mean ADC waveform',xlabel='Time from t10 (ns)',ylabel='Baseline-subtracted ADC');ax[0,0].legend(fontsize=8)
    ax[0,1].set(title='Last 20%: mean waveform residual',xlabel='Time from t10 (ns)',ylabel='(MC - data) / measured peak (%)');ax[0,1].axhline(0,color='k',lw=.5)
    ax[1,0].set(title='Short-scale fluctuation check',xlabel='Scale (ns)',ylabel='Mean F: MC / measured');ax[1,0].axhline(1,color='k',ls='--',lw=.8)
    ax[1,1].set(title='PID band centers',ylabel='Mean binwise median difference / data IQR');ax[1,1].set_xticks(range(3),['QDC','Mean time','PCA']);ax[1,1].axhline(0,color='k',ls='--',lw=.8)
    finish(fig,'noise_waveform_controls','Equal weights in frozen Q bins 0, 1, 2; measured main-band diagnostic. Point estimates; no pulse, optical or signal-residual refit.')

    rows=[]
    for seg in SEGMENTS:
        for variant in VARIANTS:
            for view,row in report['checks'][seg]['variants'][variant].items():
                for j,method in enumerate(report['methods']):
                    b64=row['uncertainty_by_data_block']['64'];b128=row['uncertainty_by_data_block']['128']
                    rows.append(dict(segment=seg,variant=variant,view=view,method=method,n_MC=row['n_MC'],
                        ratio=row['IQR_ratio'][j],CI64_low=b64['IQR_ratio_CI95'][j]['estimate95'][0],CI64_high=b64['IQR_ratio_CI95'][j]['estimate95'][1],
                        CI128_low=b128['IQR_ratio_CI95'][j]['estimate95'][0],CI128_high=b128['IQR_ratio_CI95'][j]['estimate95'][1],
                        paired_delta=row['paired_delta_ratio'][j],delta_low=b64['paired_delta_ratio_CI95'][j]['estimate95'][0],delta_high=b64['paired_delta_ratio_CI95'][j]['estimate95'][1]))
    with (OUT/'validation_table.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    last=report['checks']['last20']['variants'];nr=report['noise']['last20']
    new_widths=np.asarray([last[v]['reselected']['IQR_ratio'] for v in VARIANTS[1:]])
    rms=[nr[v]['pooled']['RMS_p10_p50_p90'][1] for v in VARIANTS[1:]]
    old=last['baseline']['reselected']['IQR_ratio']
    conclusion=(f"后段基线RMS中位数：实验 {nr['data']['RMS_p10_p50_p90'][1]:.3f} ADC，"
        f"原模型 {nr['baseline']['pooled']['RMS_p10_p50_p90'][1]:.3f}，三个新对照 {min(rms):.3f}–{max(rms):.3f}。"
        f"QDC带宽比由 {old[0]:.3f} 变为 {new_widths[:,0].min():.3f}–{new_widths[:,0].max():.3f}，"
        f"PCA由 {old[2]:.3f} 变为 {new_widths[:,2].min():.3f}–{new_widths[:,2].max():.3f}；"
        f"平均时间由 {old[1]:.3f} 变为 {new_widths[:,1].min():.3f}–{new_widths[:,1].max():.3f}。"
        "三个新对照的三种带宽配对变化95%区间均包含0；现有统计量尚不能确认改善。"
        "噪声漂移应作为采集条件记录，但未完成三种PID带宽的一致性验证。")
    delta_cis=[last[v]['reselected']['uncertainty_by_data_block']['64']['paired_delta_ratio_CI95'][j]['estimate95']
        for v in VARIANTS[1:] for j in range(3)]
    assert all(lo<=0<=hi for lo,hi in delta_cis),'Update interpretation if paired evidence changes'
    save(OUT/'decision.json',{'conclusion_zh':conclusion,'promoted_to_default':False,
        'all_predeclared_variants_reported':True,'last20_new_width_ratio_min':new_widths.min(0).tolist(),
        'last20_new_width_ratio_max':new_widths.max(0).tolist(),'new_independent_mothers':0,
        'all_last20_reselected_paired_delta_intervals_include_zero':True,
        'interpretation':'Conditional evidence from paired replays. No pure experimental particle labels, no unique noise mechanism, no updated alpha acceptance or gamma leakage.',
        'next_work':'Joint two-source first60 mean-time/tail constraints and slow-component bound sensitivity; retain waveform amplitude, F and PID checks. Freeze before increasing independent MC statistics.',
        'uncertainty':'Pointwise block64 and128 data / whole-mother MC intervals, with fits, fold assignments and all response parameters held fixed; not full calibration uncertainty.'})
    save(OUT/'plot_manifest.json',{'code_sha256':sha(Path(__file__)),'assessment_sha256':sha(OUT/'assessment.json'),
        'figures':['noise_state_closure','noise_pid_widths','noise_pid_paired_controls','noise_waveform_controls'],'table_rows':len(rows)})
    print('Saved four figure sets and',len(rows),'table rows.')


if __name__=='__main__':main()
