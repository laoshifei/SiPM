"""Scientific figures for continuous RC and jointly constrained variance scatter."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/sipm-refined-rc-mpl')
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rc_refinement_common import *

SHOW=['frozen','coarse_spectrum','mean_only','inherited_CV_fit','spectrum_CV_fit']
LABELS=['Frozen response','Coarse RC + spectrum','Continuous RC only','Continuous RC + inherited','Continuous RC + spectrum']
COLORS=['#697986','#4e977d','#9b77b8','#247ab5','#d17b26']
FIGURES=['continuous_RC_calibration','variance_scatter_selection','refined_joint_checks','refined_native_waveforms','refined_PID_covariance']


def finish(fig,name,caption,legend=None,ncol=3):
    if legend is not None:
        h,l=legend.get_legend_handles_labels();fig.legend(h,l,loc='upper center',frameon=False,ncol=ncol,fontsize=9)
    fig.tight_layout(rect=(0,.06,1,.92 if legend is not None else 1))
    fig.text(.5,.015,caption,ha='center',fontsize=8)
    for ext in ['png','pdf','svg']:fig.savefig(OUT/'figures'/f'{name}.{ext}',dpi=170,bbox_inches='tight')
    plt.close(fig)


def point(ax,x,y,ci,color,label=None,marker='o'):
    ax.plot(x,y,marker,ms=4,color=color,label=label)
    if ci is not None:ax.vlines(x,*ci,color=color,lw=1)


def main(refresh=False):
    assert refresh or not (OUT/'figure_record.json').exists()
    p=read(OUT/'protocol.json');cal=read(OUT/'mean_calibration.json');select=read(OUT/'noise_selection.json');e=read(OUT/'evaluation.json')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(12,5.5))
    for col,(source,run) in enumerate(SOURCES):
        for j,seg in enumerate(SEGMENTS):
            for k,(B,ef) in enumerate([(64,0),(64,1),(128,0),(128,1)]):
                r=cal['groups'][f'run{run}_{seg}_B{B}_eval{ef}'];fit=read(PROJECT/r['fit']);b=np.array(fit['training_block_refits'])
                x=j+(k-1.5)*.13;point(axes[col],x,r['RC_ns'],np.quantile(b[:,0],[.025,.975]),COLORS[k+1],f'Block{B}, eval{ef}' if j==0 else None,marker='s' if B==64 else 'o')
                old=read(PARENT/'fits'/f'run{run}_{seg}_B{B}_eval{ef}.json')
                axes[col].plot(x,old['selected_RC_ns'],'_',ms=9,color=COLORS[k+1],alpha=.7)
        axes[col].set(title=f'{source}: continuous conditional RC',ylabel='Effective readout RC (ns)')
        axes[col].set_xticks(range(3),['First60','Middle20','Last20']);axes[col].grid(axis='y',alpha=.15)
    finish(fig,FIGURES[0],'Bars: 128 training-block refits with other fits and native offsets fixed. Horizontal marks: previous coarse-grid selections.',axes[0],2)

    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for ef in [0,1]:
        key=f'run12_last20_B128_eval{ef}';r=read(PROJECT/select['groups'][key]['fit'])
        for j,family in enumerate(FAMILIES):
            rows=r['families'][family]['choices'];x=[v['CV'] for v in rows];y=[v['loss'] for v in rows]
            ax=axes[0,ef];ax.plot(x,y,'o-',color=COLORS[j+3],label=['Inherited spectrum','Extended spectrum'][j])
            win=r['families'][family]['chosen_index'];ax.plot(x[win],y[win],'o',ms=10,mfc='none',mec=COLORS[j+3],mew=1.5)
            target=r['target']['F_sd'][0];axes[1,ef].plot(x,[v['native_summary']['F_sd'][0]/target for v in rows],'o-',color=COLORS[j+3])
        axes[0,ef].set(title=f'Am last20: train for evaluation fold{ef}',xlabel='Signal variance multiplier CV',ylabel='Joint training loss',yscale='log')
        axes[1,ef].set(title=f'Fold{ef}: F50 distribution constraint',xlabel='Signal variance multiplier CV',ylabel='MC / training-data F50 SD')
        axes[1,ef].axhline(1,color='k',ls='--',lw=.7)
        for ax in axes[:,ef]:ax.grid(alpha=.15)
    finish(fig,FIGURES[1],'Native CV grid, fixed moment-fitted powers. Rings mark training selections. No event gain or optical parameter is changed.',axes[0,0],2)

    fig,axes=plt.subplots(3,2,figsize=(12,11))
    for col,(source,run) in enumerate(SOURCES):
        for iv,(v,label) in enumerate(zip(SHOW,LABELS)):
            for j,seg in enumerate(SEGMENTS):
                r=e['groups'][f'run{run}_{seg}']['128']['variants'][v]
                for ir,name in enumerate(['mean_time_delta_ns','IQR_time_ratio']):
                    point(axes[ir,col],j+(iv-2)*.12,r['point'][name],r['CI95'][name]['CI95'],COLORS[iv],label if j==0 else None)
            r=e['groups'][f'run{run}_last20']['128']['variants'][v]
            for j,name in enumerate(['IQR_ratio_ratio','IQR_time_ratio','IQR_PCA_ratio']):
                point(axes[2,col],j+(iv-2)*.12,r['point'][name],r['CI95'][name]['CI95'],COLORS[iv])
        for ir in [0,1]:axes[ir,col].set_xticks(range(3),['First60','Middle20','Last20'])
        axes[2,col].set_xticks(range(3),['QDC','Mean time','PCA'])
        for ir,(title,ylabel,line) in enumerate([('time center','MC - data time (ns)',0),('time bandwidth','MC / data within-Q IQR',1),('last20 all PID widths','MC / data within-Q IQR',1)]):
            axes[ir,col].set(title=f'{source}: {title}',ylabel=ylabel);axes[ir,col].axhline(line,color='k',ls='--',lw=.7);axes[ir,col].grid(axis='y',alpha=.15)
    finish(fig,FIGURES[2],'512 paired original-mother / raw-block128 draws, conditional on all fitted choices. Original IDs/Q/PCA retained.',axes[0,0],3)

    fig,axes=plt.subplots(3,2,figsize=(12,10));time=np.arange(76)*40-20
    for col,(source,run) in enumerate(SOURCES):
        row=e['groups'][f'run{run}_last20']['128'];target=np.array(row['data_summary']['wave'])
        axes[0,col].plot(time,target,'k-',lw=2,label='Measured last20')
        axes[2,col].plot(time,100*np.sqrt(np.diag(row['data_shape_cov'])),'k-',lw=2)
        for iv in [0,1,3,4]:
            v=SHOW[iv];r=row['variants'][v];wave=np.array(r['native_summary']['wave'])
            axes[0,col].plot(time,wave,color=COLORS[iv],label=LABELS[iv]);axes[1,col].plot(time,wave-target,color=COLORS[iv])
            axes[2,col].plot(time,100*np.sqrt(np.diag(r['shape_cov'])),color=COLORS[iv])
        axes[0,col].set(title=f'{source}: absolute mean native ADC',ylabel='Baseline-subtracted ADC')
        axes[1,col].set(title=f'{source}: mean residual',ylabel='MC - data (ADC)');axes[1,col].axhline(0,color='k',lw=.7)
        axes[2,col].set(title=f'{source}: normalized gate fluctuations',ylabel='SD of 40 ns charge / Qlong (%)')
        for ax in axes[:,col]:ax.set(xlabel='Time from t10 anchor (ns)',xlim=(-40,3000));ax.grid(alpha=.15)
    finish(fig,FIGURES[3],'Conditional Q matching; generated ADC is never interpolated or scaled event by event. Normalized covariance is descriptive.',axes[0,0],3)

    fig,axes=plt.subplots(2,3,figsize=(12,7));selected=[1,3,4]
    for row,(source,run) in enumerate(SOURCES):
        r=e['groups'][f'run{run}_last20']['128'];dc=np.array(r['data_summary']['cov']);normal=np.sqrt(np.diag(dc)[:,None]*np.diag(dc)[None,:])
        for col,iv in enumerate(selected):
            v=SHOW[iv];res=(np.array(r['variants'][v]['native_summary']['cov'])-dc)/normal
            ax=axes[row,col];im=ax.imshow(res,vmin=-.6,vmax=.6,cmap='RdBu_r')
            for i in range(3):
                for j in range(3):ax.text(j,i,f'{res[i,j]:+.2f}',ha='center',va='center',fontsize=9)
            ax.set_xticks(range(3),['QDC','Time','PCA']);ax.set_yticks(range(3),['QDC','Time','PCA']);ax.set_title(source+'\n'+LABELS[iv],fontsize=9)
    fig.subplots_adjust(right=.9,wspace=.36,hspace=.38,bottom=.12)
    cb=fig.colorbar(im,cax=fig.add_axes([.92,.23,.018,.54]));cb.set_label('(MC covariance - data) / data SD products')
    fig.text(.5,.025,'Last20, block128. Jointly selected variance scatter; full within-Q covariance, including score correlations.',ha='center',fontsize=8)
    for ext in ['png','pdf','svg']:fig.savefig(OUT/'figures'/f'{FIGURES[4]}.{ext}',dpi=170,bbox_inches='tight')
    plt.close(fig)
    save(OUT/'figure_record.json',{'scope':__doc__,'code_sha256':sha(__file__),'figures':FIGURES,
        'inputs_sha256':{relative(OUT/n):sha(OUT/n) for n in ['protocol.json','mean_calibration.json','noise_selection.json','evaluation.json']}})


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--refresh',action='store_true');args=ap.parse_args();main(args.refresh)
