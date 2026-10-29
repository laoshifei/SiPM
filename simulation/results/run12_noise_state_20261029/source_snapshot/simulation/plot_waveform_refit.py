"""Standalone scientific figures for the notebook; fixed Q-ranked ADC examples."""
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/sipm-refit-matplotlib')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from refit_waveform_shape import PROJECT, OLD, OUT, T, arrays, sha, save
from prepare import wave_features

BLUE='#2466a8';RED='#d64c35';GREEN='#00806b';GRAY='#8b8b8b'


def ecdf(ax,x,label,color,style='-'):
    x=np.sort(x);ax.step(x,np.arange(1,len(x)+1)/len(x),where='post',label=label,color=color,ls=style,lw=1.5)


def feat(waves):
    f=wave_features(waves)
    return {'tpeak_ns':2*(f['peak_sample']-f['t10_sample']),'ratio':f['short_long_ratio'],
            'baseline_RMS_ADC':f['baseline_rms'],'baseline_ADC':f['baseline']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis',type=Path,required=True)
    parser.add_argument('--conditioned',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    report=json.loads((args.analysis/'validation.json').read_text())
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,
                         'savefig.dpi':170,'figure.constrained_layout.use':True})
    figures=[]
    def export(fig,stem):
        for ext in ['png','pdf','svg']:fig.savefig(args.out/(stem+'.'+ext))
        figures.append(str((args.out/(stem+'.png')).resolve().relative_to(PROJECT)));plt.close(fig)
    datasets={}
    for source,run in [('Co60',3),('Am241',12)]:
        datasets[source]={'data':arrays(OLD/f'run{run}_last20.npz'),
          'train':arrays(OLD/f'run{run}_first60.npz'),
          'old':arrays(OLD/'independent'/f'{source}_native.npz'),
          'model':arrays(args.analysis/f'{source}_native.npz')}
        if args.conditioned:datasets[source]['conditioned']=arrays(args.conditioned/f'{source}_native.npz')
    fig,axes=plt.subplots(3,2,figsize=(13,12))
    for col,(source,run) in enumerate([('Co60',3),('Am241',12)]):
        d=datasets[source]
        for key,label,color in [('data','Measured last20',BLUE),('old','Previous G4',GRAY),('model','Refitted G4',RED)]:
            y=d[key]['shapes'].mean(0)*1000
            axes[0,col].plot(T/1000,y,label=label,color=color,lw=1.5)
            axes[1,col].plot(T,y,label=label,color=color,lw=1.5)
            axes[2,col].plot([50,102,202],d[key]['metrics'][:,:,3].mean(0),'o-',label=label,color=color,lw=1.5,ms=4)
        axes[0,col].set(xlim=(-.1,3.5),xlabel='Time from t10 (µs)',ylabel='Area-normalized signal (µs⁻¹)',title=f'{source}: average shape')
        axes[1,col].set(xlim=(-20,600),xlabel='Time from t10 (ns)',ylabel='Area-normalized signal (µs⁻¹)',title='Early pulse and peak')
        axes[2,col].set(xlabel='Box-filter width (ns)',ylabel='Excess variance / mean (ADC)',title='Fluctuation means')
        interval=np.array(report['sources'][source]['comparisons'][f'run{run}_last20']['MC_F_interval95'])
        means=np.array(report['sources'][source]['F'])
        axes[2,col].errorbar([50,102,202],means,yerr=np.maximum(np.array([means-interval[0],interval[1]-means]),0),fmt='none',color=RED,capsize=3)
        for row in range(3):axes[row,col].legend(fontsize=9);axes[row,col].grid(alpha=.15)
    export(fig,'shape_and_fluctuation_comparison')
    fig,axes=plt.subplots(3,2,figsize=(13,12))
    for col,(source,run) in enumerate([('Co60',3),('Am241',12)]):
        d=datasets[source]
        sets=[('data','Measured last20',BLUE),('old','Previous G4',GRAY),('model','Refitted G4: training state',RED)]
        if 'conditioned' in d:sets.append(('conditioned','Refitted G4: pretrigger-conditioned',GREEN))
        for key,label,color in sets:
            f=feat(d[key]['waves'])
            for row,name in enumerate(['tpeak_ns','ratio','baseline_RMS_ADC']):ecdf(axes[row,col],f[name],label,color)
        for row,label in enumerate(['Peak time from t10 (ns)','Short / long integral','Pretrigger RMS (ADC)']):
            axes[row,col].set(xlabel=label,ylabel='Empirical cumulative fraction',title=source,ylim=(0,1.02))
            axes[row,col].legend(fontsize=8);axes[row,col].grid(alpha=.15)
    export(fig,'feature_distributions')
    examples=[];fig,axes=plt.subplots(2,2,figsize=(13,8))
    for col,(source,run) in enumerate([('Co60',3),('Am241',12)]):
        d=datasets[source];data=d['data'];model=d.get('conditioned',d['model']);order=np.argsort(data['q'])
        for row,fraction in enumerate([.25,.75]):
            i=order[int(fraction*(len(order)-1))];j=int(np.argmin(abs(model['q']-data['q'][i])))
            ax=axes[row,col]
            for a,k,label,color in [(data,i,f'Measured #{data["ids"][i]}',BLUE),(model,j,f'G4 #{model["ids"][j]}',RED)]:
                ax.plot((np.arange(a['waves'].shape[1])-a['t10'][k])*2,a['waves'][k].astype(float)-a['baseline'][k],color=color,label=label,lw=1.0)
            delta=float(model['q'][j]/data['q'][i]-1)
            ax.set(xlim=(-100,3000),xlabel='Time from t10 (ns)',ylabel='Baseline-subtracted ADC',title=f'{source}: measured Q{int(fraction*100)}, nearest MC Q ({100*delta:+.2f}%)')
            ax.legend(fontsize=10);ax.text(.98,.75,'No amplitude scaling',transform=ax.transAxes,ha='right',fontsize=9);ax.grid(alpha=.15)
            examples.append({'source':source,'measured_group':f'run{run}_last20','rank':fraction,
                'measured_id':int(data['ids'][i]),'MC_id':int(model['ids'][j]),'Q_relative_difference':delta,
                'model_analysis':str((args.conditioned or args.analysis).resolve().relative_to(PROJECT))})
    export(fig,'unscaled_individual_waveforms')
    fig,axes=plt.subplots(1,2,figsize=(13,4))
    edges=np.linspace(*np.array(report['q_edges'])[[0,-1]],19)
    for col,source in enumerate(['Co60','Am241']):
        d=datasets[source]
        for key,label,color in [('data','Measured last20',BLUE),('old','Previous G4',GRAY),('model','Refitted G4',RED)]:
            axes[col].hist(d[key]['q']/1000,bins=edges/1000,weights=np.ones(len(d[key]['q']))/len(d[key]['q']),histtype='step',label=label,color=color,lw=1.7)
        axes[col].set(xlabel='Q (10³ ADC sample)',ylabel='Fraction per bin',title=source+' / no spectral reweighting')
        axes[col].legend(fontsize=9)
    export(fig,'charge_spectrum_check')
    run18=OUT/'run18'
    if (run18/'validation.json').exists():
        d=arrays(run18/'run18_last20.npz')
        transfer=datasets['Am241']['model']
        frozen=arrays(run18/'first60/Am241_native.npz')
        conditioned=arrays(run18/'last20/Am241_native.npz')
        sets=[(d,'Measured run18 last20',BLUE),
              (transfer,'Run12 calibration transferred',GRAY),
              (frozen,'Run18 first60 calibration',RED),
              (conditioned,'Run18 + last20 pretrigger',GREEN)]
        fig,axes=plt.subplots(3,2,figsize=(13,12))
        for a,label,color in sets:
            axes[0,0].plot(T/1000,a['shapes'].mean(0)*1000,color=color,label=label)
            axes[0,1].plot([50,102,202],a['metrics'][:,:,3].mean(0),'o-',color=color,label=label,ms=4)
            ecdf(axes[1,0],a['metrics'][:,0,3],label,color)
            f=feat(a['waves'])
            ecdf(axes[1,1],f['baseline_RMS_ADC'],label,color)
            ecdf(axes[2,0],f['tpeak_ns'],label,color)
        axes[0,0].set(xlim=(-.1,3.5),xlabel='Time from t10 (µs)',ylabel='Area-normalized signal (µs⁻¹)',title='Run18 / shared optical parameters')
        axes[0,1].set(xlabel='Box-filter width (ns)',ylabel='Excess variance / mean (ADC)',title='Separate acquisition calibration')
        axes[1,0].set(xlabel='F50 (ADC)',ylabel='Empirical cumulative fraction',title='Event fluctuation distribution')
        axes[1,1].set(xlabel='Pretrigger RMS (ADC)',ylabel='Empirical cumulative fraction',title='Baseline noise')
        axes[2,0].set(xlabel='Peak time from t10 (ns)',ylabel='Empirical cumulative fraction',title='Peak timing: remaining discrepancy')
        i=np.argsort(d['q'])[int(.5*(len(d['q'])-1))]
        j=int(np.argmin(abs(conditioned['q']-d['q'][i])))
        for a,k,label,color in [(d,i,f'Measured #{d["ids"][i]}',BLUE),(conditioned,j,f'G4 #{conditioned["ids"][j]}',GREEN)]:
            axes[2,1].plot((np.arange(a['waves'].shape[1])-a['t10'][k])*2,
                a['waves'][k].astype(float)-a['baseline'][k],color=color,label=label,lw=1)
        delta=float(conditioned['q'][j]/d['q'][i]-1)
        axes[2,1].set(xlim=(-100,3000),xlabel='Time from t10 (ns)',ylabel='Baseline-subtracted ADC',
            title=f'Fixed measured Q50, nearest MC Q ({100*delta:+.2f}%)')
        axes[2,1].text(.98,.75,'No amplitude scaling',transform=axes[2,1].transAxes,ha='right',fontsize=9)
        for ax in axes.flat:ax.legend(fontsize=8);ax.grid(alpha=.15)
        examples.append({'source':'Am241','measured_group':'run18_last20','rank':.5,
            'measured_id':int(d['ids'][i]),'MC_id':int(conditioned['ids'][j]),'Q_relative_difference':delta,
            'model_analysis':str((run18/'last20').relative_to(PROJECT))})
        export(fig,'run18_separate_calibration')
    save(args.out/'figure_record.json',{'code_sha256':sha(Path(__file__)),
        'analysis_sha256':sha(args.analysis/'validation.json'),
        'conditioned_analysis_sha256':sha(args.conditioned/'validation.json') if args.conditioned else None,
        'run18_validation_sha256':sha(run18/'validation.json') if (run18/'validation.json').exists() else None,
        'figures':figures,'examples':examples,
        'scope':'Area normalization only for mean-shape plots. Raw ADC examples are baseline/t10 aligned, fixed measured Q ranks and nearest MC Q; no shape-based cherry-picking.'})
    print('Saved',len(figures),'figure groups.',flush=True)


if __name__=='__main__':main()
