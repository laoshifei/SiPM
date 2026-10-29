"""Exact fluctuation decomposition of saved native Geant4/SiPM waveforms.

No transport or waveform model is replaced. Stored signal_mV, analog_mV and ADC
allow signal, electronics, quantization and cross terms to be separated exactly
for the frozen event IDs and alignment. A stationary formula predicts moments;
its comparison with event medians is explicitly distinguished from a mean check.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import ROOT
from reference import PROJECT, wave_features
from diagnose_charge import sha
from diagnose_fluctuations import WINDOWS, fluctuation_metrics
from profile_signal_response import variance_integrals
from compare import BLUE, RED, GREEN, KEEP, panel, graph, label, save

HERE=Path(__file__).resolve().parent
COMPONENTS=('signal','electronics','quantization','signal_x_electronics','signal_x_quantization','electronics_x_quantization')


def decompose(signals, electronics, quantization, t10, mean_total):
    """Return excess variances / observed mean; cross entries include factor2."""
    values=[]
    for signal,noise,quant,start,means in zip(signals,electronics,quantization,t10,mean_total):
        y=np.array([signal,noise,quant]);cs=np.c_[np.zeros(3),np.cumsum(y,axis=1)]
        row=[]
        for k,width in enumerate(WINDOWS):
            half=width//2;res=y[:,half:-half]-(cs[:,width:]-cs[:,:-width])/width
            time=(np.arange(half,y.shape[1]-half)-start)*2
            pre=(time>=-600)&(time<-200);post=(time>=500)&(time<2500)
            if pre.sum()!=200 or post.sum()!=1000:raise ValueError('Incomplete time support')
            cov=np.cov(res[:,post],ddof=1)-np.cov(res[:,pre],ddof=1)
            row.append(np.r_[np.diag(cov),2*cov[0,1],2*cov[0,2],2*cov[1,2]]/means[k])
        values.append(row)
    return np.asarray(values)


def statistics(metric, mean_signal):
    """Expectation-like aggregates versus the published event-median diagnostic."""
    metric=np.asarray(metric);mean_signal=np.asarray(mean_signal)
    rng=np.random.default_rng(283330);indices=rng.integers(len(metric),size=(1000,len(metric)))
    boot=metric[indices];weights=mean_signal[indices]
    return {'n':len(metric),'event_median':float(np.median(metric)),
        'event_median_bootstrap_95':np.quantile(np.median(boot,axis=1),[.025,.975]).tolist(),
        'mean_event_ratio':float(metric.mean()),'mean_event_ratio_bootstrap_95':np.quantile(boot.mean(axis=1),[.025,.975]).tolist(),
        'ratio_of_summed_variance_to_summed_mean':float(np.sum(metric*mean_signal)/mean_signal.sum()),
        'ratio_of_sums_bootstrap_95':np.quantile(np.sum(boot*weights,axis=1)/weights.sum(axis=1),[.025,.975]).tolist()}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mc',type=Path,default=HERE/'results/colored_noise_20261028');p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    args.mc=args.mc.resolve();args.out.mkdir(parents=True,exist_ok=False)
    previous=args.mc/'fluctuations/metrics.npz'
    with np.load(previous) as z:old=dict(z)
    report={'scope':__doc__,'selection':'Exact event IDs and ADC alignment of prior common-Q fluctuation analysis; no re-selection, no resimulation.',
        'identity':'Excess variance(total) = sum of three component excess variances + twice the three pairwise excess covariances. All six components divide by the same observed mean. Their event means add; their separate medians generally do not.',
        'quantization':'ADC minus (analog_mV-adc_min_mV)/LSB, including any clipping; electronics = analog_mV-signal_mV-baseline_mV. Signal stored by native C++ digitizer.',
        'stationary_comparison':'Independent stationary-avalanche integral estimates an expectation-like variance/mean, not the median of a finite-window estimator. Both event means and intensity-weighted ratio-of-sums are reported with bootstrap intervals.',
        'limits':'Fixed selected sample; conditional selection, finite nonstationary emission and recovery remain. Bootstrap covers selected-event statistics only. Signal-only output is available only in MC, not recovered from measured data.',
        'inputs_sha256':{str(previous.relative_to(PROJECT)):sha(previous)},
        'code_sha256':{str(f.relative_to(PROJECT)):sha(f) for f in [Path(__file__),HERE/'diagnose_fluctuations.py',HERE/'profile_signal_response.py',HERE/'reference.py',PROJECT/'analysis/paper_baseline/prepare.py']},'sources':{}}
    arrays={}
    for source,name in [('Co60','co60'),('Am241','am241')]:
        path=args.mc/name/'events.root';file=ROOT.TFile.Open(str(path));tree=file.Get('Events');info=json.loads(str(file.Get('RunInfo').GetString().Data()))
        if not info['completed']:raise ValueError('Incomplete MC')
        tree.SetBranchStatus('*',0)
        for branch in ['adc','signal_mV','analog_mV','avalanche_t_ns','avalanche_q_C']:tree.SetBranchStatus(branch,1)
        cfg=info['config'];r=cfg['sipm']['readout'];s=cfg['sipm']['sensor'];lsb=(r['adc_max_mV']-r['adc_min_mV'])/2**r['adc_bits']
        ids=old[source+'_mc_ids'];adcs=[];signals=[];analog=[];charges=[];times=[]
        for index in ids:
            if tree.GetEntry(int(index))<=0:raise ValueError('Missing event')
            adcs.append(np.array(tree.adc,dtype=float));signals.append(np.array(tree.signal_mV)/lsb);analog.append(np.array(tree.analog_mV)/lsb)
            charges.append(np.array(tree.avalanche_q_C));times.append(np.array(tree.avalanche_t_ns))
        file.Close();adcs=np.array(adcs);signals=np.array(signals);analog=np.array(analog)
        f=wave_features(adcs);metrics=fluctuation_metrics(adcs,f['baseline'],f['t10_sample'])
        np.testing.assert_array_equal(metrics,old[source+'_mc_metrics'])
        electronics=analog-signals-r['baseline_mV']/lsb
        quant=adcs-(analog-r['adc_min_mV']/lsb)
        parts=decompose(signals,electronics,quant,f['t10_sample'],metrics[:,:,0])
        error=float(abs(parts.sum(axis=2)-metrics[:,:,3]).max())
        if error>1e-9:raise ValueError('Variance decomposition failed')
        signal_metrics=fluctuation_metrics(signals,np.zeros(len(ids)),f['t10_sample'])
        nominal_q=1.602176634e-19*s['gain_electrons'];nominal_area=nominal_q*r['transimpedance_ohm']*1e12/lsb
        stationary=nominal_area*variance_integrals([r['lowpass_ns']],rise=r['rise_ns'],decay=r['decay_ns'])[0]
        q_factors=[];counts=[]
        for ts,qs,start in zip(times,charges,f['t10_sample']):
            keep=(ts>=start*2+500)&(ts<start*2+2500);q=qs[keep]
            q_factors.append(float(np.sum(q*q)/np.sum(q)/nominal_q) if len(q) else None);counts.append(int(keep.sum()))
        row={'n':len(ids),'max_absolute_identity_error_ADC':error,'original_metrics_exact':True,
            'avalanche_count_in_signal_window_quantiles':np.quantile(counts,[.1,.5,.9]).tolist(),
            'avalanche_q2_over_q_over_nominal_quantiles':np.quantile([x for x in q_factors if x is not None],[.1,.5,.9]).tolist(),'filters':{}}
        for j,w in enumerate(WINDOWS):
            full=statistics(metrics[:,j,3],metrics[:,j,0]);pure=statistics(signal_metrics[:,j,3],signal_metrics[:,j,0])
            row['filters'][str(w*2)+'ns']={'observed_ADC':full,'native_signal_only':pure,'mean_component_contributions_ADC':dict(zip(COMPONENTS,parts[:,j,:].mean(axis=0).tolist())),
                'stationary_expectation_ADC':float(stationary[j]),
                'stationary_over_full_median_minus1':float(stationary[j]/full['event_median']-1),
                'stationary_over_signal_mean_minus1':float(stationary[j]/pure['mean_event_ratio']-1),
                'stationary_over_signal_ratio_of_sums_minus1':float(stationary[j]/pure['ratio_of_summed_variance_to_summed_mean']-1)}
        report['sources'][source]=row;report['inputs_sha256'][str(path.relative_to(PROJECT))]=sha(path)
        for suffix,value in {'ids':ids,'full_metrics':metrics,'signal_metrics':signal_metrics,'components':parts,'avalanche_q_factors':q_factors,'avalanche_counts':counts}.items():arrays[source+'_'+suffix]=np.asarray(value)
        print(source,json.dumps(row,indent=2),flush=True)
    np.savez_compressed(args.out/'components.npz',**arrays)
    (args.out/'components.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    ROOT.gROOT.SetBatch(True);ROOT.gStyle.SetOptStat(0);ROOT.gStyle.SetTitleFontSize(.039);ROOT.gStyle.SetTitleSize(.036,'XYZ');ROOT.gStyle.SetLabelSize(.032,'XYZ')
    c=ROOT.TCanvas('mc_components','Native Geant4 waveform moments',1450,1100);c.Divide(2,2,.003,.003)
    x=np.array(WINDOWS,dtype=float)*2
    for col,source in enumerate(['Co60','Am241']):
        rows=report['sources'][source]['filters'];series=[('observed_ADC','event_median',BLUE,'ADC event median'),('observed_ADC','mean_event_ratio',RED,'ADC mean of event ratios'),('native_signal_only','mean_event_ratio',GREEN,'Native signal-only mean')]
        upper=max(max(r[g][stat] for r in rows.values()) for g,stat,_,_ in series)*1.6
        panel(c,col+1,source+' / selected full MC;Filter width (ns);Excess variance / mean (ADC)',30.,0.,220.,upper)
        for i,(group,stat,color,text) in enumerate(series):graph(x,np.array([r[group][stat] for r in rows.values()]),color);label(text,.20,.86-i*.055,color,.028)
        graph(x,np.array([r['stationary_expectation_ADC'] for r in rows.values()]),1,2);label('Stationary expectation',.20,.695,size=.028)
        contributions=np.array([list(r['mean_component_contributions_ADC'].values()) for r in rows.values()])
        ymax=max(.01,contributions.max()*1.6);ymin=min(-.01,contributions.min()*1.5)
        panel(c,col+3,source+' / exact ADC decomposition;Filter width (ns);Mean component contribution (ADC)',30.,ymin,220.,ymax)
        for values,color,text,y in [(contributions[:,0],BLUE,'Signal',.86),(contributions[:,1]+contributions[:,2],RED,'Electronics + quantization',.80),(contributions[:,3:].sum(axis=1),GREEN,'All cross terms',.74)]:
            graph(x,values,color);label(text,.21,y,color,.029)
        label('Component means add; component medians do not',.21,.66,size=.026)
    save(c,args.out,'mc_component_check')


if __name__=='__main__':main()
