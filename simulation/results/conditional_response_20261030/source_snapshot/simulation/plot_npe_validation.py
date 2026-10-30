"""Export scientific figures for the frozen photon-statistics candidate."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/sipm-npe-matplotlib')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from prepare_npe_profile import OUT,PARENT
from prepare_pid import read
from extract_pid import load

FIG=OUT/'figures'
LABELS=['QDC ratio','Negative mean time','Frozen PCA']
BLUE='#2876b4';ORANGE='#d47820';GREEN='#278365'


def save(fig,name):
    for ext in ['png','svg','pdf']:fig.savefig(FIG/f'{name}.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    selection=read(OUT/'selection.json');report=read(OUT/'validation/assessment.json')
    previous=read(PARENT/'validation/assessment.json')
    edges=np.asarray(read(OUT/'validation/protocol.json')['q_edges'])
    grid=selection['grid'];scales=[r['scale'] for r in grid]
    fig,axs=plt.subplots(1,3,figsize=(13,4.1))
    for key,label,color in [('joint_score','Combined',BLUE),('Co_score','Co / run3',ORANGE),('Am_score','Am / run12',GREEN)]:
        axs[0].semilogy(scales,[r[key] for r in grid],'o-',color=color,label=label)
    axs[0].set_ylabel('Joint covariance / F residual score');axs[0].legend(fontsize=8)
    for ax,key,title in zip(axs[1:],['Co_SD_ratio','Am_SD_ratio'],['Co / run3','Am / run12']):
        for j,(label,color) in enumerate(zip(LABELS,[BLUE,ORANGE,GREEN])):
            ax.plot(scales,[r[key][j] for r in grid],'o-',label=label,color=color)
        ax.axhline(1,color='k',ls=':',lw=1);ax.set_title(title);ax.set_ylabel('Predicted MC / robust data SD')
        ax.set_ylim(.8,1.35);ax.legend(fontsize=8)
    for ax in axs:
        ax.axvline(selection['chosen_scale'],color='#777777',lw=1,ls='--')
        ax.set_xlabel('Effective photon-count scale');ax.set_xticks(scales);ax.grid(alpha=.15)
    fig.suptitle('First60 calibration only: scale 2 selected; scale 3 has a similar score',y=1.02)
    fig.text(.5,-.025,'Scores are diagnostic objectives, not a likelihood or a confidence interval. PDE/coupling/yield are not separately measured.',ha='center',fontsize=9)
    fig.tight_layout();save(fig,'npe_training_profile')

    fig,axs=plt.subplots(3,3,figsize=(13,10),sharex=True)
    centers=(edges[:-1]+edges[1:])/2000
    for row,run in enumerate([3,12,18]):
        new=report['checks'][f'run{run}_last20'];old=previous['checks'][f'run{run}_last20']
        for j,ax in enumerate(axs[row]):
            for cells,version,color,offset,marker in [(old['per_Q'],'old',BLUE,-2,'o'),(new['per_Q'],'new',ORANGE,2,'s')]:
                xx=[];yy=[];lo=[];hi=[]
                for c in cells:
                    n=c['n_new'] if version=='old' else c['n_MC']
                    if not c['in_summary'] or n<20:continue
                    m=np.asarray(c['after_quartiles'] if version=='old' else c['MC_quartiles'],float)
                    d=np.asarray(c['data_quartiles'],float)
                    ci=c['after_ratio_CI95' if version=='old' else 'ratio_CI95'][j]['estimate95']
                    if ci is None:continue
                    val=(m[2,j]-m[0,j])/(d[2,j]-d[0,j])
                    xx.append(centers[c['bin']]+offset);yy.append(val);lo.append(max(0,val-ci[0]));hi.append(max(0,ci[1]-val))
                ax.errorbar(xx,yy,yerr=[lo,hi],fmt=marker+'-',color=color,capsize=3,
                    label='Previous noise-only candidate' if version=='old' else 'Photon + transfer candidate')
            ax.axhline(1,color='k',ls=':',lw=1);ax.set_ylim(bottom=0);ax.grid(alpha=.15)
            if row==0:ax.set_title(LABELS[j])
            if j==0:ax.set_ylabel(f'Run{run} / '+('Co' if run==3 else 'Am')+'\nMC / data IQR')
            if row==2:ax.set_xlabel('Q (1000 ADC samples)')
    axs[0,0].legend(fontsize=7.5)
    fig.suptitle('Full Geant4 at nominal PDE 0.3 vs experimental last20',y=1.01)
    fig.text(.5,-.012,'Pointwise 95% bootstrap intervals; frozen supported bins, at least 20 MC events. Experimental checks were previously explored.',ha='center',fontsize=9)
    fig.tight_layout();save(fig,'npe_bandwidth_validation')

    fig,axs=plt.subplots(3,2,figsize=(11.5,10))
    for row,run in enumerate([3,12,18]):
        a=load(OUT/'validation/means'/f'run{run}_last20.npz')
        for key,label,color in [('data','Measured last20','#222222'),('MC','New full Geant4',ORANGE)]:
            axs[row,0].plot(a['time_ns'],a[key+'_ADC'],color=color,label=label,lw=1.3)
            axs[row,1].plot([50,102,202],a[key+'_F'],'o-',color=color,label=label)
        axs[row,0].set_xlim(-100,3200);axs[row,0].set_ylabel(f'Run{run}\nBaseline-subtracted ADC')
        axs[row,1].set_ylabel('Excess variance / mean (ADC)');axs[row,1].grid(alpha=.15)
    axs[0,0].set_title('Aligned mean: no amplitude renormalization');axs[0,1].set_title('Short-scale fluctuation constraints')
    axs[0,0].legend(fontsize=9);axs[-1,0].set_xlabel('Time from t10 (ns)');axs[-1,1].set_xlabel('Box-filter width (ns)')
    fig.suptitle('Mean voltage and fluctuations after photon-statistics change',y=1.01)
    fig.text(.5,-.012,'Equal weight per frozen supported Q bin; both measured and simulated events pass the common Q selection.',ha='center',fontsize=9)
    fig.tight_layout();save(fig,'npe_waveform_validation')

    fig,axs=plt.subplots(1,3,figsize=(13,4.4),sharex=True)
    for ax,run in zip(axs,[3,12,18]):
        source='Co60' if run==3 else 'Am241';a=load(OUT/'validation/arrays'/f'run{run}_{source}.npz')
        d=load(PARENT/'arrays'/f'measured_run{run}_last20.npz');u=d['main_band']
        ax.scatter(d['q'][u]/1000,d['scores'][u,0],s=7,c='#777777',alpha=.28,rasterized=True,label='Measured last20 main band')
        ax.scatter(a['q']/1000,a['scores'][:,0],s=13,c=ORANGE,alpha=.6,rasterized=True,label='New MC; no PID cut')
        for b in report['checks'][f'run{run}_last20']['per_Q']:
            if not b['in_summary'] or b['n_MC']<20:continue
            x=centers[b['bin']]
            for shift,k,color in [(-2,'data_quartiles','#222222'),(2,'MC_quartiles',ORANGE)]:
                z=np.asarray(b[k],float)[:,0]
                ax.errorbar(x+shift,z[1],yerr=[[z[1]-z[0]],[z[2]-z[1]]],color=color,fmt='o',ms=4,lw=1.8,capsize=4)
        ax.set_title(f'Run{run} / {source}');ax.set_xlabel('Q (1000 ADC samples)');ax.grid(alpha=.15)
        ax.set_ylim((.40,.55) if run==3 else (.25,.40))
    axs[0].set_ylabel('QDC short / long ratio');axs[0].legend(fontsize=7.5,loc='upper right')
    fig.suptitle('QDC bands: medians and IQR bars in supported bins',y=1.01)
    fig.text(.5,-.015,'Main-band diagnostic is not experimental particle truth. Vertical limits focus on the main bands; all selected MC remains in numerical results.',ha='center',fontsize=8.5)
    fig.tight_layout();save(fig,'npe_QDC_bands')
    print('Saved 4 scientific figures in PNG / SVG / PDF.',flush=True)


if __name__=='__main__':main()
