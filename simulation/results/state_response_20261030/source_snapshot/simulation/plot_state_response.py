"""Figures for conditional state calibration and rejected alignment shortcuts."""
import os
import argparse
os.environ.setdefault('MPLCONFIGDIR','/tmp/sipm-state-response-mpl')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from state_response_common import *
from evaluate_state_response import VARIANTS

COLORS=['#647480','#2a78b8','#d57b2a','#188d75']
LABELS=['Frozen response','Pretrigger electronics','RC + inherited spectrum','RC + extended spectrum']
FIGURES=['sampling_and_front_controls','conditional_RC_stability','state_native_joint_checks','state_native_waveforms','state_PID_covariance']


def finish(fig,name,caption,legend=None,ncol=4):
    if legend is not None:
        h,l=legend.get_legend_handles_labels();fig.legend(h,l,loc='upper center',frameon=False,ncol=ncol,fontsize=9)
    fig.tight_layout(rect=(0,.055,1,.94 if legend is not None else 1))
    fig.text(.5,.015,caption,ha='center',fontsize=8)
    for ext in ['png','pdf','svg']:fig.savefig(OUT/'figures'/f'{name}.{ext}',dpi=170,bbox_inches='tight')
    plt.close(fig)


def point(ax,x,y,ci,color,label=None):
    ax.plot(x,y,'o',ms=4,color=color,label=label)
    if ci is not None:ax.vlines(x,*ci,color=color,lw=1)


def main(refresh=False):
    assert refresh or not (OUT/'figure_record.json').exists()
    p=read(OUT/'protocol.json');align=read(OUT/'alignment.json');e=read(OUT/'evaluation.json')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for isrc,(source,run) in enumerate(SOURCES):
        phase=align['native_phase'][source];x=np.array([r['phase_ns'] for r in phase])
        axes[0,0].plot(x,[r['mean_time_change_ns'] for r in phase],'o-',label=source,color=COLORS[isrc+1])
        axes[0,1].plot(x,[r['IQR_ratio_to_phase0'][1] for r in phase],'o-',label=source,color=COLORS[isrc+1])
        offset=(isrc-.5)*.2
        for j,seg in enumerate(SEGMENTS):
            r=align['groups'][f'run{run}_{seg}']['128']
            axes[1,0].plot(j+offset,r['phase_regression_MSE_ratio'][1],'o',color=COLORS[isrc+1],label=source if j==0 else None)
            axes[1,1].plot(j+offset,r['front_data_boundary_fraction'],'o',color=COLORS[isrc+1],label=source+' data' if j==0 else None)
            axes[1,1].plot(j+offset,r['front_MC_boundary_fraction'],'x',color=COLORS[isrc+1],label=source+' MC' if j==0 else None)
    axes[0,0].set(title='Native aperture phase: mean time',xlabel='Aperture offset (ns)',ylabel='Change from phase0 (ns)')
    axes[0,1].set(title='Native aperture phase: time-score width',xlabel='Aperture offset (ns)',ylabel='IQR / phase0 IQR')
    axes[0,0].legend(frameon=False);axes[0,1].axhline(1,color='k',ls='--',lw=.7)
    axes[1,0].set(title='Measured t10 phase: blocked prediction',ylabel='MSE with phase / without phase');axes[1,0].axhline(1,color='k',ls='--',lw=.7)
    axes[1,1].set(title='Front-only lag fits hit the bounds',ylabel='Q-matched boundary fraction',ylim=(0,1));axes[1,1].legend(fontsize=8,frameon=False)
    for ax in axes[1]:ax.set_xticks(range(3),['First60','Middle20','Last20'])
    finish(fig,FIGURES[0],'Fixed native avalanches; observed t10 phase is only a noisy proxy. Front fits are diagnostics, not adopted timing corrections.')

    fig,axes=plt.subplots(3,2,figsize=(12,10))
    for col,(source,run) in enumerate(SOURCES):
        for row,seg in enumerate(SEGMENTS):
            ax=axes[row,col]
            for ef in [0,1]:
                r=read(OUT/'fits'/f'run{run}_{seg}_B128_eval{ef}.json');x=r['RC_grid_ns'];loss=np.array(r['grid_loss'])
                boot=np.array(r['training_block_refits']);counts=np.bincount(boot[:,0].astype(int),minlength=5)/len(boot)
                ax.plot(x,loss-loss.min(),'o-',color=COLORS[ef+1],label=f'Evaluate fold{ef}')
                for xx,y,prob in zip(x,loss-loss.min(),counts):
                    if prob>0:ax.annotate(f'{prob:.0%}',(xx,0),xycoords=('data','axes fraction'),
                        xytext=(0,12+ef*14),textcoords='offset points',ha='center',fontsize=8,color=COLORS[ef+1],
                        bbox={'facecolor':'white','edgecolor':'none','alpha':.92,'pad':.25})
            ax.set(title=f'{source} / {seg}: calibration grid',xlabel='Effective readout RC (ns)',ylabel='Mean-response loss - grid minimum')
            ax.grid(alpha=.15)
    finish(fig,FIGURES[1],'Labels: RC selection fractions over128 training-block refits. Quiet-grid proposals; electronics fits and MC held fixed.',axes[0,0],2)

    fig,axes=plt.subplots(3,2,figsize=(12,11))
    for col,(source,run) in enumerate(SOURCES):
        for j,seg in enumerate(SEGMENTS):
            row=e['groups'][f'run{run}_{seg}']['128']['variants']
            for iv,v in enumerate(VARIANTS):
                r=row[v];x=j+(iv-1.5)*.13
                for ir,name in enumerate(['mean_time_delta_ns','IQR_time_ratio','F202_ratio']):
                    point(axes[ir,col],x,r['point'][name],r['CI95'][name]['CI95'],COLORS[iv],LABELS[iv] if j==0 else None)
        for ir,(ylabel,line) in enumerate([('MC - data mean time (ns)',0),('MC / data time-score IQR',1),('MC / data F202',1)]):
            axes[ir,col].set(title=f'{source}: '+['time center','time bandwidth','202 ns fluctuations'][ir],ylabel=ylabel)
            axes[ir,col].set_xticks(range(3),['First60','Middle20','Last20']);axes[ir,col].axhline(line,color='k',ls='--',lw=.7);axes[ir,col].grid(axis='y',alpha=.15)
    finish(fig,FIGURES[2],'512 paired whole-mother / raw-block128 draws, conditional on fitted models. All experimental checks are internal.',axes[0,0],2)

    fig,axes=plt.subplots(3,2,figsize=(12,10));t=np.arange(76)*40-20
    for col,(source,run) in enumerate(SOURCES):
        r=e['groups'][f'run{run}_last20']['128'];target=np.array(r['data_moments']['wave'])
        axes[0,col].plot(t,target,'k-',lw=2,label='Measured last20')
        for iv,v in enumerate(VARIANTS):
            m=r['variants'][v];wave=np.array(m['native_moments']['wave'])
            axes[0,col].plot(t,wave,color=COLORS[iv],label=LABELS[iv]);axes[1,col].plot(t,wave-target,color=COLORS[iv])
            axes[2,col].plot(t,100*np.sqrt(np.diag(m['shape_cov'])),color=COLORS[iv])
        axes[2,col].plot(t,100*np.sqrt(np.diag(r['data_shape_cov'])),'k-',lw=2)
        axes[0,col].set(title=f'{source}: last20 absolute gate mean',ylabel='Baseline-subtracted ADC')
        axes[1,col].set(title=f'{source}: last20 mean residual',ylabel='MC - data (ADC)');axes[1,col].axhline(0,color='k',lw=.7)
        axes[2,col].set(title=f'{source}: normalized gate fluctuations',ylabel='SD of40 ns charge / Qlong (%)')
        for ax in axes[:,col]:ax.set(xlabel='Original-anchor time (ns)',xlim=(-40,3000));ax.grid(alpha=.15)
    finish(fig,FIGURES[3],'Q-matched original ADC means, without event amplitude scaling. Normalized shape covariance is descriptive, not intrinsic CsI variance.',axes[0,0],3)

    fig,axes=plt.subplots(2,4,figsize=(13,7));im=None
    for row,(source,run) in enumerate(SOURCES):
        r=e['groups'][f'run{run}_last20']['128'];dc=np.array(r['data_moments']['cov']);norm=np.sqrt(np.diag(dc)[:,None]*np.diag(dc)[None,:])
        for col,v in enumerate(VARIANTS):
            residual=(np.array(r['variants'][v]['native_moments']['cov'])-dc)/norm
            ax=axes[row,col];im=ax.imshow(residual,vmin=-.6,vmax=.6,cmap='RdBu_r')
            for i in range(3):
                for j in range(3):ax.text(j,i,f'{residual[i,j]:+.2f}',ha='center',va='center',fontsize=9)
            ax.set_xticks(range(3),['QDC','Time','PCA']);ax.set_yticks(range(3),['QDC','Time','PCA'])
            ax.set_title(source+'\n'+LABELS[col],fontsize=9)
    fig.subplots_adjust(right=.9,wspace=.42,hspace=.4,bottom=.12)
    cb=fig.colorbar(im,cax=fig.add_axes([.92,.23,.018,.54]));cb.set_label('(MC covariance - data) / data SD products')
    fig.text(.5,.025,'Last20, block128 response assignments; covariance includes within-Q correlations. Fixed effective model calibration.',ha='center',fontsize=8)
    for ext in ['png','pdf','svg']:fig.savefig(OUT/'figures'/f'{FIGURES[4]}.{ext}',dpi=170,bbox_inches='tight')
    plt.close(fig)
    save(OUT/'figure_record.json',{'scope':__doc__,'code_sha256':sha(__file__),'figures':FIGURES,
        'inputs_sha256':{relative(OUT/name):sha(OUT/name) for name in ['protocol.json','alignment.json','mean_calibration.json','signal_calibration.json','evaluation.json']}})


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--refresh',action='store_true');args=ap.parse_args();main(args.refresh)
