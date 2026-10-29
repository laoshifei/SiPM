"""Scientific figures for exact noise decomposition and frozen native checks."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/sipm-width-matplotlib')
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from prepare_pid_bandwidth import OUT,PARENT
from prepare_pid import read
from extract_pid import load
from fit_pid_bandwidth import average_bins
from reference import PROJECT,TIME_NS

FIG=OUT/'figures';COLORS=['#3376b8','#e49635','#2c9b79','#9a7bb6','#777777']
LABELS=['QDC ratio','Negative mean time','Frozen PCA']


def save(fig,name):
    for ext in ['png','svg','pdf']:fig.savefig(FIG/f'{name}.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True);plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    diag=read(OUT/'diagnosis.json');fit=read(OUT/'covariance_fit/fit.json')
    report=read(OUT/'validation/assessment.json');edges=np.array(read(OUT/'protocol.json')['q_edges'])
    fig,axs=plt.subplots(1,3,figsize=(13.5,4.8),sharey=True)
    for j,ax in enumerate(axs):
        for pos,run in enumerate([3,12,18]):
            rows=diag['MC'][f'run{run}']['decomposition'];support=fit['states'][f'run{run}']['support_bins']
            covariance=np.mean([np.asarray(r['covariances'])[j] for r in rows if r['bin'] in support],axis=0)
            total=covariance.sum();v=np.r_[np.diag(covariance),total-np.trace(covariance)]/total
            plus=minus=0.
            for k,x in enumerate(v):
                bottom=plus if x>=0 else minus
                ax.bar(pos,x,bottom=bottom,color=COLORS[k],width=.65,label=['Native signal','Baseline electronics','Signal residual','Quantization','Cross terms'][k] if pos==0 else None)
                if x>=0:plus+=x
                else:minus+=x
        ax.axhline(1,color='k',lw=.8,ls=':');ax.axhline(0,color='k',lw=.6)
        ax.set_xticks(range(3),['Co / run3','Am / run12','Am / run18'],rotation=15);ax.set_title(LABELS[j])
    axs[0].set_ylabel('Contribution / total PID variance')
    handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,ncol=5,loc='lower center',bbox_to_anchor=(.5,.07),fontsize=9)
    fig.suptitle('Previous MC: exact fixed-event covariance decomposition',y=1.01)
    fig.text(.5,.025,'Signal includes photon/avalanche statistics, geometry and observed-anchor effects. Cross terms are retained.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.19,1,.97));save(fig,'pid_noise_contributions')

    fig,axs=plt.subplots(3,3,figsize=(13,10),sharex=True)
    centers=(edges[:-1]+edges[1:])/2000
    for row,run in enumerate([3,12,18]):
        r=report['checks'][f'run{run}_last20']
        for j,ax in enumerate(axs[row]):
            for name,color,offset,marker in [('before',COLORS[0],-2,'o'),('after',COLORS[1],2,'s')]:
                xx=[];yy=[];lo=[];hi=[]
                for cell in r['per_Q']:
                    n=cell['n_old' if name=='before' else 'n_new']
                    if n<20 or not cell['in_summary']:continue
                    d=np.array(cell['data_quartiles'],float);m=np.array(cell[name+'_quartiles'],float)
                    val=(m[2,j]-m[0,j])/(d[2,j]-d[0,j]);ci=cell[name+'_ratio_CI95'][j]['estimate95']
                    if ci is None:continue
                    xx.append(centers[cell['bin']]+offset);yy.append(val);lo.append(max(0,val-ci[0]));hi.append(max(0,ci[1]-val))
                ax.errorbar(xx,yy,yerr=[lo,hi],fmt=marker+'-',color=color,capsize=3,label='Previous response' if name=='before' else 'Joint covariance response')
            ax.axhline(1,color='k',ls=':',lw=1);ax.set_ylim(bottom=0)
            if row==0:ax.set_title(LABELS[j])
            if j==0:ax.set_ylabel(f'Run{run} / '+('Co' if run==3 else 'Am')+'\nMC / data IQR')
            if row==2:ax.set_xlabel('Q (1000 ADC samples)')
            ax.grid(alpha=.15)
    axs[0,0].legend(fontsize=8)
    fig.suptitle('Independent native seeds vs experimental last20: pointwise 95% intervals',y=1.01)
    fig.text(.5,-.015,'Experimental checks were previously explored. Only frozen supported Q bins with at least20 MC events are plotted.',ha='center',fontsize=9)
    fig.tight_layout();save(fig,'pid_bandwidth_validation')

    fig,axs=plt.subplots(3,2,figsize=(11.5,10))
    for row,(run,source) in enumerate([(3,'Co60'),(12,'Am241'),(18,'Am241')]):
        support=fit['states'][f'run{run}']['support_bins']
        path=(PARENT/'run18' if run==18 else PROJECT/'simulation/results/signal_fluctuations_20261028')/f'run{run}_first60.npz'
        d=load(path);mask=load(OUT/'arrays'/f'measured_run{run}_first60.npz')['main_band']
        shapes=[('Measured first60',average_bins(d['shapes'][mask],d['q'][mask],edges,support),'#222222')]
        F=[('Measured first60',average_bins(d['metrics'][mask,:,3],d['q'][mask],edges,support),'#222222')]
        for name,label,color in [('old','Previous response',COLORS[0]),('new','Joint covariance response',COLORS[1])]:
            a=load(OUT/'validation/arrays'/f'{name}_run{run}_{source}.npz')
            shapes.append((label,average_bins(a['shapes'],a['q'][a['common']],edges,support),color))
            F.append((label,average_bins(a['metrics'],a['q'][a['common']],edges,support),color))
        for label,y,color in shapes:axs[row,0].plot(TIME_NS,y*1000,color=color,lw=1.3,label=label)
        axs[row,0].set_xlim(-100,3200);axs[row,0].set_ylabel(f'Run{run}: {source}\nNormalized pulse (1/us)')
        for label,y,color in F:axs[row,1].plot([50,102,202],y,'o-',color=color,label=label)
        axs[row,1].set_ylabel('Excess variance / mean (ADC)');axs[row,1].grid(alpha=.15)
    axs[0,0].legend(fontsize=8);axs[0,0].set_title('Equal-Q-bin average waveform');axs[0,1].set_title('Short-scale fluctuation constraints')
    axs[-1,0].set_xlabel('Time from t10 (ns)');axs[-1,1].set_xlabel('Box-filter width (ns)')
    fig.suptitle('Waveform closure with new native seeds; parameters frozen before generation',y=1.01)
    fig.tight_layout();save(fig,'pid_width_waveform_closure')

    fig,axs=plt.subplots(1,3,figsize=(13,3.8))
    for ax,run in zip(axs,[3,12,18]):
        entry=read(OUT/'protocol.json')['entries'][f'run{run}']
        for path,label,color in [(PROJECT/entry['parent_config'],'Previous response',COLORS[0]),
                               (OUT/f'covariance_fit/run{run}_config.json','Joint covariance response',COLORS[1])]:
            c=read(path)['signal_dependent_noise'];h=np.array(c['coefficients']);f=np.fft.rfftfreq(16384,2)*1000
            power=c['variance_per_signal_adc']*abs(np.fft.rfft(h,n=16384))**2
            use=(f>=.06)&(f<=100);ax.loglog(f[use],power[use],color=color,label=label)
        ax.set_title(f'Run{run} empirical state');ax.set_xlabel('Frequency (MHz)');ax.grid(alpha=.15)
    axs[0].set_ylabel('Residual power per signal ADC');axs[0].legend(fontsize=8)
    fig.suptitle('Change of correlation spectrum, with baseline electronics unchanged',y=1.04)
    fig.tight_layout();save(fig,'pid_width_residual_spectra')
    print('Saved4 figures in PNG/SVG/PDF.',flush=True)


if __name__=='__main__':main()
