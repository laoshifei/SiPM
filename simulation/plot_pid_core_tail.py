"""Plot fixed-cohort core/tail evidence, retaining distribution and uncertainty limits."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/sipm-core-tail-mpl')
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pid_core_tail_common import *

SHOW=['frozen','coarse_spectrum','inherited_CV_fit','spectrum_CV_fit']
LABELS=['Frozen response','Coarse RC + spectrum','Fine RC + inherited spectrum','Fine RC + extended spectrum']
COLORS=['#77828c','#4d9273','#287eb6','#cb7a2b']
METHOD_LABELS=['QDC','Mean-time score (-t)','PCA']
FIGURES=['core_tail_quantile_widths','core_tail_mass_and_variance','core_tail_distributions','core_tail_waveform_covariance','core_tail_noise_associations']


def finish(fig,name,caption,legend=None,ncol=2):
    if legend is not None:
        h,l=legend.get_legend_handles_labels();fig.legend(h,l,loc='upper center',frameon=False,ncol=ncol,fontsize=9)
    fig.tight_layout(rect=(0,.06,1,.92 if legend is not None else 1))
    fig.text(.5,.015,caption,ha='center',fontsize=8)
    for ext in ['png','pdf','svg']:fig.savefig(OUT/'figures'/f'{name}.{ext}',dpi=170,bbox_inches='tight')
    plt.close(fig)


def point(ax,x,y,ci,color,label=None):
    ax.plot(x,y,'o',ms=4,color=color,label=label)
    if ci is not None:ax.vlines(x,ci[0],ci[1],color=color,lw=1)


def main(refresh=False):
    assert refresh or not (OUT/'figure_record.json').exists()
    e=read(OUT/'diagnostics.json')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(3,2,figsize=(12,11))
    for col,(source,run) in enumerate(SOURCES):
        row=e['groups'][f'run{run}_last20']['128']
        for j,m in enumerate(METHODS):
            ax=axes[j,col]
            for iv,v in enumerate(SHOW):
                r=row['variants'][v]
                for k,c in enumerate(COVERAGES):
                    name=f'width{int(c*100)}_{m}_ratio'
                    point(ax,k+(iv-1.5)*.11,r['point'][name],r['CI95'][name]['CI95'],COLORS[iv],LABELS[iv] if k==0 else None)
            ax.set(title=f'{source}: {METHOD_LABELS[j]}',ylabel='MC / data within-Q width')
            ax.set_xticks(range(4),['50% (IQR)','80%','90%','95%']);ax.axhline(1,color='k',ls='--',lw=.7);ax.grid(axis='y',alpha=.15)
    finish(fig,FIGURES[0],'Last20, block128. Central quantile spans; 512 paired conditional draws with all response parameters and diagnostic fences fixed.',axes[0,0])

    fig,axes=plt.subplots(2,2,figsize=(12,8.5));cohorts=['data',*SHOW];colors=['#181818',*COLORS]
    for col,(source,run) in enumerate(SOURCES):
        row=e['groups'][f'run{run}_last20']['128']
        for k,v in enumerate(cohorts):
            r=row['data'] if v=='data' else row['variants'][v]['native'];ci=row['data_tail_CI95'] if v=='data' else row['variants'][v]['tail_CI95']
            for ir,name in enumerate(['tail_probability','tail_variance_share']):
                for j in range(3):
                    interval0=ci[0][j][ir]['CI95'];interval0=None if interval0 is None else np.array(interval0)*100
                    point(axes[ir,col],j+(k-2)*.12,100*r[name][0][j],interval0,colors[k],('Measured' if k==0 else LABELS[k-1]) if j==0 else None)
        for ir,ylabel in enumerate(['Events beyond training fences (%)','Full within-Q variance in those events (%)']):
            axes[ir,col].set(title=f'{source}: |z| > 2',ylabel=ylabel);axes[ir,col].set_xticks(range(3),['QDC','Mean time','PCA']);axes[ir,col].grid(axis='y',alpha=.15)
    finish(fig,FIGURES[1],'Fences use opposite measured blocks: median +/- 2 x IQR/1.349. Variance contributions use each evaluation cohort own within-Q mean.',axes[0,0],3)

    fig,axes=plt.subplots(3,2,figsize=(12,10.5))
    for col,(source,run) in enumerate(SOURCES):
        row=e['groups'][f'run{run}_last20']['128']
        for j in range(3):
            ax=axes[j,col]
            for k,v in enumerate(cohorts):
                h=row['data']['histogram'] if v=='data' else row['variants'][v]['native']['histogram']
                edges=np.array(h['edges']);d=np.array(h['density'][j]);x=(edges[:-1]+edges[1:])/2
                ax.step(x,np.where(d>0,d,np.nan),where='mid',color=colors[k],lw=1.8 if k==0 else 1.1,label='Measured' if k==0 else LABELS[k-1])
            for bound in [-3,-2,2,3]:ax.axvline(bound,color='#666666',ls=':' if abs(bound)==3 else '--',lw=.6)
            ax.set(title=f'{source}: {METHOD_LABELS[j]}',xlabel='Score relative to training median / robust scale',ylabel='Weighted density',yscale='log',xlim=(-5,5));ax.grid(axis='y',alpha=.15)
    finish(fig,FIGURES[2],'Last20, block128. Fixed measured training reference for both data and MC. Empty bins omitted; no smoothing or renormalization of plotted range.',axes[0,0],3)

    fig,axes=plt.subplots(3,2,figsize=(12,10.5));time=np.arange(76)*40-20
    for col,(source,run) in enumerate(SOURCES):
        row=e['groups'][f'run{run}_last20']['128']
        for j in range(3):
            ax=axes[j,col]
            for v,color,label in [('data','#181818','Measured'),('inherited_CV_fit',COLORS[2],'Inherited spectrum'),('spectrum_CV_fit',COLORS[3],'Extended spectrum')]:
                r=row['data'] if v=='data' else row['variants'][v]['native']
                ax.plot(time,100*np.array(r['score_shape_per_SD'])[j],color=color,label=label+' total')
                ax.plot(time,100*np.array(r['tail_score_shape_per_SD'])[j],color=color,ls='--',label=label+' |z| > 2')
            ax.axhline(0,color='k',lw=.6);ax.set(title=f'{source}: {METHOD_LABELS[j]}',xlabel='Time from t10 anchor (ns)',ylabel='Cov(score, charge fraction) / SD (%)');ax.grid(alpha=.15)
    finish(fig,FIGURES[3],'Tail and core contributions sum exactly to full within-Q covariance. Shared waveform normalization and gate definitions preclude causal attribution.',axes[0,0],3)

    fig,axes=plt.subplots(2,3,figsize=(12,7))
    for ir,(source,run) in enumerate(SOURCES):
        row=e['groups'][f'run{run}_last20']['128']
        for col,(v,label) in enumerate([('data','Measured'),('inherited_CV_fit','Inherited spectrum'),('spectrum_CV_fit','Extended spectrum')]):
            r=row['data'] if v=='data' else row['variants'][v]['native'];c=np.array(r['auxiliary_correlations']);ax=axes[ir,col]
            im=ax.imshow(c,vmin=-1,vmax=1,cmap='RdBu_r',aspect='auto')
            for i in range(3):
                for j in range(4):ax.text(j,i,f'{c[i,j]:+.2f}',ha='center',va='center',fontsize=9,color='white' if abs(c[i,j])>.65 else 'black')
            ax.set_xticks(range(4),['log RMS','F50','F102','F202']);ax.set_yticks(range(3),['QDC','-time','PCA']);ax.set_title(source+'\n'+label)
    fig.subplots_adjust(right=.9,wspace=.4,hspace=.45,bottom=.12)
    fig.colorbar(im,cax=fig.add_axes([.92,.23,.018,.54])).set_label('Within-Q correlation')
    fig.text(.5,.025,'Last20, block128. Associations are descriptive; F and PID share signal samples. Correlation does not identify a noise mechanism.',ha='center',fontsize=8)
    for ext in ['png','pdf','svg']:fig.savefig(OUT/'figures'/f'{FIGURES[4]}.{ext}',dpi=170,bbox_inches='tight')
    plt.close(fig)
    save(OUT/'figure_record.json',{'scope':__doc__,'code_sha256':sha(__file__),'figures':FIGURES,
        'inputs_sha256':{relative(OUT/n):sha(OUT/n) for n in ['protocol.json','diagnostics.json']}})


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--refresh',action='store_true');args=ap.parse_args();main(args.refresh)
