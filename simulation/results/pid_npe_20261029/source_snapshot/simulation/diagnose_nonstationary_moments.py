"""Finite-window avalanche self terms using the original native C++ readout.

Actual Geant4 avalanche times/charges are retained. Native BuildReadout replays
each stored signal and generates the single-avalanche kernels. No scintillation,
transport, recovery, gain, noise or source parameter is refitted. The diagonal
self term is a conditional diagnostic; correlations and smooth mean curvature
are not asserted to vanish for a selected finite event population.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import ROOT
from reference import PROJECT, wave_features
from diagnose_charge import sha
from diagnose_fluctuations import WINDOWS, fluctuation_metrics
from diagnose_mc_components import statistics
from profile_signal_response import variance_integrals
from compare import BLUE, RED, GREEN, KEEP, panel, graph, label, save

HERE=Path(__file__).resolve().parent


def native_readout(config,times,charges,duration):
    result=ROOT.sipm.Result()
    for value in times:result.avalanche_time_ns.push_back(float(value))
    for value in charges:result.avalanche_charge_C.push_back(float(value))
    result.charge_total_C=float(np.sum(charges))
    ROOT.sipm.BuildReadout(config,float(duration),result)
    return np.array(result.signal_mV,dtype=float)


def prefix_kernels(impulses,width):
    filt=-np.ones(width)/width;filt[width//2]+=1
    residual=np.array([np.convolve(wave,filt,mode='full') for wave in impulses])
    return np.c_[np.zeros(len(residual)),np.cumsum(residual,axis=1)],np.c_[np.zeros(len(residual)),np.cumsum(residual**2,axis=1)]


def window_variance(prefix,prefix2,phases,shifts,start,stop,half):
    n=stop-start
    if n<=1:raise ValueError('Need a finite variance window')
    lo=np.clip(start-shifts+half,0,prefix.shape[1]-1)
    hi=np.clip(stop-shifts+half,0,prefix.shape[1]-1)
    total=prefix[phases,hi]-prefix[phases,lo]
    square=prefix2[phases,hi]-prefix2[phases,lo]
    return (square-total**2/n)/(n-1)


def self_moments(times,charges,t10,kernels,nominal_charge,period=2.):
    nphase=len(kernels[0][0]);delta=period/nphase
    ticks=np.rint(np.asarray(times)/delta).astype(np.int64)
    shifts=ticks//nphase;phase=ticks%nphase;weight=(np.asarray(charges)/nominal_charge)**2
    before=(int(np.ceil(t10-300)),int(np.ceil(t10-100)))
    after=(int(np.ceil(t10+250)),int(np.ceil(t10+1250)))
    result=[]
    for width,(prefix,prefix2) in zip(WINDOWS,kernels):
        post=window_variance(prefix,prefix2,phase,shifts,*after,width//2)
        pre=window_variance(prefix,prefix2,phase,shifts,*before,width//2)
        result.append(float(np.sum(weight*(post-pre))))
    return np.array(result)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mc',type=Path,default=HERE/'results/colored_noise_20261028');p.add_argument('--components',type=Path,default=HERE/'results/mc_components_20261028');p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    args.mc=args.mc.resolve();args.components=args.components.resolve();args.out.mkdir(parents=True,exist_ok=False)
    ROOT.gInterpreter.AddIncludePath(str(HERE/'vendor'))
    if not ROOT.gInterpreter.Declare('#include "SiPMResponse.cc"'):raise RuntimeError('Cannot load original native readout')
    with np.load(args.components/'components.npz') as z:previous=dict(z)
    report={'scope':__doc__,
        'formula':'Sum_i (q_i/q_nominal)^2 * [sampleVar(F h_i in signal window)-sampleVar(F h_i in pre window)], divided by the actual native signal mean. F is the same centered box residual. Uses real avalanche timing; retains finite-window mean subtraction.',
        'interpretation':'This is the sum of each individual avalanche impulse variance, not the full sample variance of their sum. For independent arrivals it estimates the diagonal shot-noise covariance contribution; pair terms, conditioning, rate uncertainty and deterministic mean curvature remain. No physical parameter inferred from agreement.',
        'native_kernel':'PyROOT invokes the unchanged vendor SiPMResponse.cc BuildReadout implementation. All selected signal_mV arrays are replayed from saved avalanche_t_ns / avalanche_q_C.64 arrival phases per2ns sample, with32-phase convergence check; no empirical pulse template.',
        'inputs_sha256':{str(p.relative_to(PROJECT)):sha(p) for p in [args.components/'components.json',args.components/'components.npz']},
        'code_sha256':{str(p.relative_to(PROJECT)):sha(p) for p in [Path(__file__),HERE/'vendor/SiPMResponse.cc',HERE/'vendor/SiPMResponse.hh',HERE/'diagnose_mc_components.py',HERE/'diagnose_fluctuations.py',HERE/'profile_signal_response.py']},'sources':{}}
    arrays={}
    for source,name in [('Co60','co60'),('Am241','am241')]:
        path=args.mc/name/'events.root';file=ROOT.TFile.Open(str(path));tree=file.Get('Events');info=json.loads(str(file.Get('RunInfo').GetString().Data()))
        if not info['completed']:raise ValueError('Incomplete MC')
        cfg=info['config']['sipm'];r=cfg['readout'];s=cfg['sensor'];native=ROOT.sipm.Config.FromJSON(json.dumps(cfg))
        nominal=1.602176634e-19*s['gain_electrons'];lsb=(r['adc_max_mV']-r['adc_min_mV'])/2**r['adc_bits']
        tree.SetBranchStatus('*',0)
        for branch in ['adc','signal_mV','avalanche_t_ns','avalanche_q_C']:tree.SetBranchStatus(branch,1)
        ids=previous[source+'_ids'];duration=None;max_error=0.;numerators=[];coarse=[];means=[];observed=[];kernels=None
        for j,index in enumerate(ids):
            if tree.GetEntry(int(index))<=0:raise ValueError('Missing event')
            adc=np.array(tree.adc,dtype=float);signal=np.array(tree.signal_mV);ts=np.array(tree.avalanche_t_ns);qs=np.array(tree.avalanche_q_C)
            if duration is None:
                duration=len(adc)*r['sample_period_ns']
                impulses=np.array([native_readout(native,[phase*r['sample_period_ns']/64],[nominal],duration)/lsb for phase in range(64)])
                kernels=[prefix_kernels(impulses,width) for width in WINDOWS]
                coarse_kernels=[prefix_kernels(impulses[::2],width) for width in WINDOWS]
                arrays[source+'_native_impulses_ADC']=impulses
            replay=native_readout(native,ts,qs,duration)
            max_error=max(max_error,float(abs(replay-signal).max()))
            if not np.allclose(replay,signal,rtol=1e-11,atol=1e-10):raise ValueError('Native signal replay failed')
            f=wave_features(adc[None,:]);t10=f['t10_sample'][0]
            metrics=fluctuation_metrics(signal[None,:]/lsb,[0.],[t10])[0]
            np.testing.assert_array_equal(metrics,previous[source+'_signal_metrics'][j])
            numerators.append(self_moments(ts,qs,t10,kernels,nominal,r['sample_period_ns']))
            coarse.append(self_moments(ts,qs,t10,coarse_kernels,nominal,r['sample_period_ns']))
            means.append(metrics[:,0]);observed.append(metrics[:,3])
        file.Close();means=np.array(means);observed=np.array(observed);prediction=np.array(numerators)/means;coarse=np.array(coarse)/means
        convergence=float(np.max(abs(coarse-prediction)/np.maximum(abs(prediction),1e-12)))
        if convergence>1e-3:raise ValueError('Arrival phase grid has not converged')
        stationary=nominal*r['transimpedance_ohm']*1e12/lsb*variance_integrals([r['lowpass_ns']],rise=r['rise_ns'],decay=r['decay_ns'])[0]
        row={'n':len(ids),'max_native_signal_replay_error_mV':max_error,'max_relative_32_vs_64_phase_difference':convergence,'filters':{}}
        for k,width in enumerate(WINDOWS):
            actual=statistics(observed[:,k],means[:,k]);pred=statistics(prediction[:,k],means[:,k]);delta=statistics(observed[:,k]-prediction[:,k],means[:,k])
            row['filters'][str(2*width)+'ns']={'native_signal':actual,'finite_window_self_term':pred,'observed_minus_self_term':delta,
                'stationary_expectation_ADC':float(stationary[k]),'self_term_mean_relative_to_native_minus1':float(pred['mean_event_ratio']/actual['mean_event_ratio']-1),
                'stationary_mean_relative_to_native_minus1':float(stationary[k]/actual['mean_event_ratio']-1)}
        report['sources'][source]=row;report['inputs_sha256'][str(path.relative_to(PROJECT))]=sha(path)
        for suffix,value in {'ids':ids,'native_metric':observed,'self_metric':prediction,'coarse_self_metric':coarse,'mean_signal':means}.items():arrays[source+'_'+suffix]=value
        print(source,json.dumps(row,indent=2),flush=True)
    np.savez_compressed(args.out/'moments.npz',**arrays)
    (args.out/'moments.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    ROOT.gROOT.SetBatch(True);ROOT.gStyle.SetOptStat(0);ROOT.gStyle.SetTitleFontSize(.04);ROOT.gStyle.SetTitleSize(.039,'XYZ');ROOT.gStyle.SetLabelSize(.034,'XYZ')
    c=ROOT.TCanvas('native_moments','Native finite-window moment checks',1400,650);c.Divide(2,1,.003,.003)
    x=np.array(WINDOWS,dtype=float)*2
    for col,source in enumerate(['Co60','Am241']):
        rows=report['sources'][source]['filters']
        panel(c,col+1,source+' / fixed actual avalanche records;Filter width (ns);Signal variance / mean (ADC)',30.,0.,220.,.43)
        for group,color,text,y in [('native_signal',BLUE,'Native signal mean + event-bootstrap95%',.86),('finite_window_self_term',RED,'Actual-timing single-avalanche self sum',.80)]:
            values=np.array([v[group]['mean_event_ratio'] for v in rows.values()]);limits=np.array([v[group]['mean_event_ratio_bootstrap_95'] for v in rows.values()])
            g=ROOT.TGraphAsymmErrors(3,x,values,np.zeros(3),np.zeros(3),values-limits[:,0],limits[:,1]-values)
            g.SetMarkerStyle(20);g.SetMarkerColor(color);g.SetLineColor(color);g.SetLineWidth(2);g.Draw('LP SAME');KEEP.append(g);label(text,.19,y,color,.028)
        graph(x,np.array([v['stationary_expectation_ADC'] for v in rows.values()]),GREEN,2);label('Stationary expectation',.19,.74,GREEN,.029)
        label('No refit; finite-window conditional MC diagnostic',.19,.65,size=.026)
    save(c,args.out,'native_moments')


if __name__=='__main__':main()
