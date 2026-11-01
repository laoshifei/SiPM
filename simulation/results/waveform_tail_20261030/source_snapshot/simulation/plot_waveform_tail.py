"""Scientific figures for original ADC examples and frozen waveform diagnostics."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/sipm-wave-tail-mpl')
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from waveform_tail_common import *

FIGURES=['am_extreme_raw_ADC','co_extreme_raw_ADC','tail_estimator_width_controls','tail_baseline_anchor_observables','trained_waveform_modes']
METHOD_LABELS=['QDC','Mean-time score (-t)','PCA']


def finish(fig,name,caption,legend=None):
    if legend is not None:
        h,l=legend.get_legend_handles_labels();fig.legend(h,l,loc='upper center',frameon=False,ncol=2,fontsize=9)
    fig.tight_layout(rect=(0,.06,1,.94 if legend is not None else 1))
    fig.text(.5,.015,caption,ha='center',fontsize=8)
    for ext in ['png','pdf','svg']:fig.savefig(OUT/'figures'/f'{name}.{ext}',dpi=170,bbox_inches='tight')
    plt.close(fig)


def examples():
    record=read(OUT/'examples.json')['groups']
    for source,run,name in [('Am241',12,FIGURES[0]),('Co60',3,FIGURES[1])]:
        key=f'run{run}_last20';data=record[key+'_data'];mc=record[key+'_spectrum_CV_fit']
        dw=load(OUT/'arrays'/f'{key}_data_examples.npz')['waves'];mw=load(OUT/'arrays'/f'{key}_spectrum_CV_fit_examples.npz')['waves']
        lo=min(-2*r['anchor'] for r in data+mc)-20;hi=max(2*(len(dw[0])-1-r['anchor']) for r in data+mc)+20
        fig,axes=plt.subplots(3,3,figsize=(15,11))
        for i,(d,m) in enumerate(zip(data,mc)):
            ax=axes.flat[i]
            for label,r,w,color in [('Data',d,dw[i],'#183850'),('MC',m,mw[i],'#cb7a2b')]:
                t=2*(np.arange(len(w))-r['anchor']);y=w.astype(float)-r['baseline_ADC']
                ax.plot(t,y,color=color,lw=.7,alpha=.8,label=f"{label} #{r['event_id']}; Q={r['original_Q']/1000:.1f}k, z={r['actual_z']:+.2f}")
            ax.axvspan(-384,-128,color='#dddddd',alpha=.4);ax.axvline(0,color='k',ls=':',lw=.6)
            ax.set(title=f"{source}, {METHOD_LABELS[d['method_index']]}: {100*d['quantile']:g}%",xlabel='Time from native integer t10 anchor (ns)',ylabel='ADC - first128 baseline',xlim=(lo,hi))
            ax.legend(loc='upper right',frameon=False,fontsize=7);ax.grid(alpha=.12)
        finish(fig,name,'Last20, block128, fixed middle Q bin. Data/MC independently chosen by score quantile, not waveform resemblance. Raw ADC; no event amplitude scaling.')


def main(examples_only=False,refresh=False):
    assert refresh or not (OUT/'figure_record.json').exists()
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    examples()
    if examples_only:return
    p=read(OUT/'protocol.json');e=read(OUT/'analysis.json')
    fig,axes=plt.subplots(3,2,figsize=(12,10))
    for col,(source,run) in enumerate(SOURCES):
        r=e['groups'][f'run{run}_last20']['128']['variants']['spectrum_CV_fit']['controls']
        for j,m in enumerate(METHODS):
            ax=axes[j,col]
            for ic,(coverage,color) in enumerate([(50,'#287eb6'),(95,'#cb7a2b')]):
                name=f'width{coverage}_{m}_ratio'
                for i,control in enumerate(CONTROLS):
                    x=i+(ic-.5)*.12;point=r[control]['point'][name];ci=r[control]['CI95'][name]['CI95']
                    ax.plot(x,point,'o',ms=4,color=color,label=f'Central {coverage}% width' if i==0 else None)
                    if ci is not None:ax.vlines(x,ci[0],ci[1],color=color,lw=1)
            ax.axhline(1,color='k',ls='--',lw=.7);ax.set_xticks(range(4),['Original','Recent base','Smooth t10','Both'],rotation=12)
            ax.set(title=f'{source}: {METHOD_LABELS[j]}',ylabel='MC / data within-Q width');ax.grid(axis='y',alpha=.15)
    finish(fig,FIGURES[2],'Last20, block128, frozen extended-spectrum response. Apply each estimator to both data and MC; all integrals use unsmoothed ADC.',axes[0,0])

    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for col,(source,run) in enumerate(SOURCES):
        key=f'run{run}_last20'
        for cohort,color,marker,label in [('data','#183850','o','Data'),('spectrum_CV_fit','#cb7a2b','x','Extended-spectrum MC')]:
            a=load(OUT/'arrays'/(f'{key}_data.npz' if cohort=='data' else f'{key}_B128_{cohort}.npz'))
            z=load(PARENT/'arrays'/f'{key}_B128_{cohort}_z.npz')['z'];w=event_weights(a,p,source);use=w>0
            for row,field,xlabel in [(0,1,'Recent - first baseline / first128 RMS'),(1,3,'Smooth - raw integer anchor (ns)')]:
                ax=axes[row,col];ax.scatter(a['observables'][use,field],z[use,1],s=13,marker=marker,alpha=.38,color=color,label=label)
                ax.set(title=f'{source}: last20',xlabel=xlabel,ylabel='Original mean-time score z');ax.grid(alpha=.15)
                for bound in [-2,2]:ax.axhline(bound,color='k',ls='--',lw=.6)
    finish(fig,FIGURES[3],'Original events, fixed Q support and inherited opposite-block PID reference. Diagnostic associations do not identify a physical cause.',axes[0,0])

    fig,axes=plt.subplots(3,2,figsize=(12,10));t=np.arange(76)*40-20
    for col,(source,run) in enumerate(SOURCES):
        design=read(OUT/'designs'/f'run{run}_last20_B128.json')
        for j in range(3):
            ax=axes[j,col]
            for ef,color in [(0,'#287eb6'),(1,'#cb7a2b')]:
                d=design[str(ef)];v=np.array(d['vectors'])[:,j];scale=np.sqrt(d['eigenvalues'][j])
                ax.plot(t,100*scale*v,color=color,label=f'Train for evaluation fold {ef}')
            ax.axhline(0,color='k',lw=.7);ax.set(title=f'{source}: fixed-rank mode {j+1}',xlabel='Time from original t10 anchor (ns)',ylabel='Training SD x charge-fraction mode (%)');ax.grid(alpha=.15)
    finish(fig,FIGURES[4],'Opposite measured blocks only; top 3 modes fixed in advance. Signed by largest coefficient. Modes are descriptive and may rotate across folds.',axes[0,0])
    save(OUT/'figure_record.json',{'scope':__doc__,'code_sha256':sha(__file__),'figures':FIGURES,
        'inputs_sha256':{relative(OUT/n):sha(OUT/n) for n in ['protocol.json','examples.json','analysis.json']}})


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--examples-only',action='store_true');ap.add_argument('--refresh',action='store_true');args=ap.parse_args();main(args.examples_only,args.refresh)
