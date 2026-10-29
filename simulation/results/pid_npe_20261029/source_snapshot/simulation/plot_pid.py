"""Scientific PID figures: truth, common readout, finite-sample ROC and widths."""
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/sipm-pid-matplotlib')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from prepare_pid import OUT,PROJECT,STATES,read
from analyze_pid import combined,METHODS
from extract_pid import load,NAMES
from publication_stats import sha,save

COLORS={'run3':'#00806b','run12':'#d64c35','run18':'#8b5ba6','original_per_run':'#8c8c8c'}
LABELS={'run3':'Common run3 readout','run12':'Common run12 readout','run18':'Common run18 readout',
        'original_per_run':'Original per-run readouts'}
METHOD_LABELS=['Short / long charge','Negative mean time (µs)','Frozen PCA projection']
TRUTH_COLORS=['#d64c35','#2466a8','#d49a20','#696969','#bbbbbb']
TRUTH_LABELS=['α ancestry','γ ancestry','Decay-electron ancestry','Mixed / other','No deposit']


def main():
    protocol=read(OUT/'protocol.json');report=read(OUT/'pid.json');folder=OUT/'figures';folder.mkdir(exist_ok=True)
    datasets={case:combined(case,protocol) for case in list(STATES)+['original_per_run']}
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
        'figure.constrained_layout.use':True,'savefig.dpi':160})
    paths=[]
    def export(fig,stem):
        for ext in ['png','pdf','svg']:fig.savefig(folder/(stem+'.'+ext))
        paths.append(str((folder/(stem+'.png')).relative_to(PROJECT)));plt.close(fig)
    measured={f'run{run}_{segment}':load(OUT/'arrays'/f'measured_run{run}_{segment}.npz')
              for run in [3,12] for segment in ['first60','last20']}
    fig,axes=plt.subplots(2,2,figsize=(13,9),sharex=True,sharey=True)
    for ax,segment,title in [(axes[0,0],'first60','Measured training sources'),(axes[0,1],'last20','Measured later sources')]:
        for run,color,name in [(3,TRUTH_COLORS[1],'Co source / run3'),(12,TRUTH_COLORS[0],'Am source / run12')]:
            d=measured[f'run{run}_{segment}'];ax.scatter(d['q']/1000,d['scores'][:,0],s=8,alpha=.45,color=color,
                label=f'{name}, n={len(d["q"])}',rasterized=True)
        ax.set_title(title);ax.text(.02,.03,'Source tags; no measured particle truth',transform=ax.transAxes,fontsize=9)
    for ax,case,title in [(axes[1,0],'original_per_run','G4: original per-run response (diagnostic)'),
                          (axes[1,1],'run12','G4: common run12 response')]:
        d=datasets[case]
        for t,name in enumerate(TRUTH_LABELS):
            use=d['common']&(d['truth']==t)
            if np.any(use):ax.scatter(d['q'][use]/1000,d['scores'][use,0],s=15,color=TRUTH_COLORS[t],alpha=.7,
                            label=f'{name}, n={use.sum()}',rasterized=True)
        ax.set_title(title);ax.text(.02,.03,'≥90% deposited energy from one ancestry',transform=ax.transAxes,fontsize=9)
    threshold=read(OUT/'thresholds.json')['methods']['qdc']['threshold']
    for i,ax in enumerate(axes.flat):
        ax.axhline(threshold,color='#444444',ls='--',lw=1,label='Frozen training cut' if i==3 else None)
        ax.set(xlabel='Q (10³ ADC sample)',ylabel='Short / long charge');ax.legend(fontsize=8,loc='upper right');ax.grid(alpha=.15)
    export(fig,'pid_truth_and_measurement')

    fig,axes=plt.subplots(1,3,figsize=(15,4.8))
    centers=(np.array(protocol['q_edges'][:-1])+np.array(protocol['q_edges'][1:]))/2000
    exp=report['experimental_training_mixture_fits']['models']
    for j,(method,ax) in enumerate(zip(METHODS,axes)):
        points=sorted([x for x in exp if x['method']==method],key=lambda x:x['bin'])
        ax.plot(centers,[x['shape_fom_fwhm'] for x in points],'s-',color='#2466a8',label='Measured training cores',lw=1.7)
        for case in ['original_per_run',*STATES]:
            fits=report['cases'][case]['methods'][method]['source_mixture_fits'];valid=[x for x in fits if x['valid']]
            ax.plot([centers[x['bin']] for x in valid],[x['fit']['shape_fom_fwhm'] for x in valid],
                'o--' if case=='original_per_run' else 'o-',color=COLORS[case],label=LABELS[case],ms=4)
        ax.set(title=METHOD_LABELS[j],xlabel='Q (10³ ADC sample)',ylabel='Two-core FOM (FWHM)',xlim=(centers[0]-15,centers[-1]+15))
        ax.grid(alpha=.15)
    axes[0].legend(fontsize=8)
    fig.suptitle('Same two Gaussian cores + broad Cauchy model; MC points require ≥10 events per source',fontsize=11)
    export(fig,'pid_core_fom_vs_charge')

    fig,axes=plt.subplots(2,3,figsize=(15,9))
    ymin=1.
    for j,method in enumerate(METHODS):
        for case,style,marker in zip(STATES,['--','-',':'],['D','o','s']):
            curve=load(OUT/'curves'/f'{case}_{method}.npz');row=report['cases'][case]['methods'][method]
            a=row['operating']['alpha_acceptance'];g=row['operating']['gamma_leakage']
            axes[0,j].plot(curve['gamma_leakage'],curve['alpha_acceptance'],ls=style,color=COLORS[case],
                label=f'{LABELS[case]} (AUC {row["test_AUC"]:.3f})',lw=1.6)
            for ax in axes[:,j]:
                ax.errorbar(g['fraction'],a['fraction'],
                    xerr=np.array([[g['fraction']-g['exact_two_sided95'][0]],[g['exact_two_sided95'][1]-g['fraction']]]),
                    yerr=np.array([[a['fraction']-a['exact_two_sided95'][0]],[a['exact_two_sided95'][1]-a['fraction']]]),
                    fmt=marker,color=COLORS[case],ms=5,capsize=3,alpha=.8,
                    label=LABELS[case] if ax is axes[1,j] else None)
            ymin=min(ymin,a['exact_two_sided95'][0])
        axes[0,j].plot([0,1],[0,1],':',color='#bbbbbb',lw=.8)
        axes[0,j].set(title=METHOD_LABELS[j],xlim=(-.01,1.01),ylim=(-.01,1.02))
        axes[0,j].legend(fontsize=7,loc='lower right')
        axes[1,j].set(title='Frozen operating cut: zoom',xlim=(-.005,.09))
        axes[1,j].legend(fontsize=8,loc='lower right')
        for ax in axes[:,j]:ax.set(xlabel='Simulated γ leakage',ylabel='Simulated α acceptance');ax.grid(alpha=.15)
    for ax in axes[1]:ax.set_ylim(max(0,ymin-.025),1.015)
    fig.suptitle('Held-out mothers; error bars are exact two-sided 95% binomial intervals, not zero-error claims',fontsize=11)
    export(fig,'pid_roc_and_operating_points')

    fig,axes=plt.subplots(3,2,figsize=(13,11))
    for col,run in enumerate([3,12]):
        for j,method in enumerate(METHODS):
            ax=axes[j,col];comparison=report['measured'][f'run{run}_last20']['methods'][method]['comparisons']
            for case in STATES:
                valid=[c for c in comparison[case]['per_Q'] if 'IQR_ratio' in c]
                ax.plot([centers[c['bin']] for c in valid],[c['IQR_ratio'] for c in valid],
                    'o-',color=COLORS[case],label=LABELS[case],ms=4)
            ax.axhline(1.,color='#2466a8',ls='--',label='Measured IQR reference')
            ax.set(title=f'{METHOD_LABELS[j]} / measured run{run} last20',
                xlabel='Q (10³ ADC sample)',ylabel='MC IQR / measured IQR',xlim=(centers[0]-15,centers[-1]+15),ylim=(0,2.7))
            ax.grid(alpha=.15);ax.legend(fontsize=8)
    fig.suptitle('Band-width check within charge bins; source-tagged measured samples, no particle labels',fontsize=11)
    export(fig,'pid_bandwidth_check')

    record={'scope':__doc__,'code_sha256':sha(Path(__file__)),'PID_sha256':sha(OUT/'pid.json'),
        'thresholds_sha256':sha(OUT/'thresholds.json'),'figures':paths,
        'notes':['All selected common-Q points shown in PID scatter; no score-based cuts.',
          'FOM curves are descriptive fitted core separation, not empirical tail probabilities.',
          'ROC uses held-out mothers, same frozen primary-state cuts for all states.',
          'IQR is a distribution width, not an error bar. Missing high-Q Am points lack support.']}
    save(folder/'figure_record.json',record);print('Saved',len(paths),'PID figure groups.',flush=True)


if __name__=='__main__':main()
