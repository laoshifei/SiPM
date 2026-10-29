"""Check the pretrigger Gaussian FIR model and its separate pulse-fluctuation limit."""
import argparse,json
from pathlib import Path
import numpy as np
import ROOT
from characterize_readout import detrend,covariance,periodogram,summary
from diagnose_charge import sha
from compare import BLUE,RED,KEEP,panel,graph,label,save
from reference import PROJECT

HERE=Path(__file__).resolve().parent


def load_pre(path):
    f=ROOT.TFile.Open(str(path));info=json.loads(str(f.Get('RunInfo').GetString().Data()));tree=f.Get('Events')
    if not info.get('completed'):raise ValueError('Incomplete MC')
    tree.SetBranchStatus('*',0);tree.SetBranchStatus('adc',1)
    pre=np.array([np.asarray(e.adc,dtype=np.uint16)[:256].copy() for e in tree]);f.Close()
    return pre,info


def describe(waves):
    cov=covariance(waves).mean(axis=0);freq,power=periodogram(waves)
    return {'n':len(waves),'detrended_rms_adc':summary(detrend(waves).std(axis=1)),
            'mean_detrended_variance_adc2':float(cov[0]),'lag1_correlation':float(cov[1]/cov[0])},cov,power.mean(axis=0),freq


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--new',type=Path,required=True)
    p.add_argument('--old',type=Path,default=HERE/'results/joint_optical_20261028')
    p.add_argument('--noise',type=Path,default=HERE/'results/readout_noise_20261028');p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    args.new=args.new.resolve();args.old=args.old.resolve();args.noise=args.noise.resolve();args.out.mkdir(parents=True,exist_ok=False)
    with np.load(args.noise/'samples.npz') as z:saved=dict(z)
    oldf_path=HERE/'results/fluctuations_20261028/diagnostics.json';newf_path=args.new/'fluctuations/diagnostics.json'
    oldf=json.loads(oldf_path.read_text());newf=json.loads(newf_path.read_text())
    report={'scope':__doc__,'baseline_selection':'Measured basic+t10 support+primary pretrigger QC, first256 raw samples. Full MC first256 samples are pre-decay baseline for every event, independent of whether a source interaction occurs. No pulse/noise template copied.',
        'interpretation':'Training average baseline variance was used for calibration; agreement there is not independent validation. Later recorded groups and covariance/spectral shape expose limits. Fixed Gaussian covariance cannot represent changing noise states or weak pulses.',
        'inputs_sha256':{str(p.relative_to(PROJECT)):sha(p) for p in (args.noise/'samples.npz',oldf_path,newf_path)},
        'code_sha256':{str(p.relative_to(PROJECT)):sha(p) for p in (Path(__file__),HERE/'characterize_readout.py')},'sources':{}}
    arrays={};ROOT.gROOT.SetBatch(True);ROOT.gStyle.SetOptStat(0);ROOT.gStyle.SetTitleFontSize(.041);ROOT.gStyle.SetLabelSize(.033,'XYZ');ROOT.gStyle.SetTitleSize(.037,'XYZ')
    gray=ROOT.TColor.GetColor('#888888');styles=[('measured',BLUE,'Measured training'),('old',gray,'Previous independent noise'),('new',RED,'Gaussian FIR noise')]
    c=ROOT.TCanvas('noise_update','Electronic noise update',1400,1450);c.Divide(2,3,.003,.003)
    for col,(source,name,run) in enumerate([('Co60','co60',3),('Am241','am241',12)]):
        phase=PROJECT/f'analysis/paper_baseline/results/phase2/run{run:04d}_features.npz'
        with np.load(phase) as z:primary=z['eligible_primary']
        report['inputs_sha256'][str(phase.relative_to(PROJECT))]=sha(phase)
        key=f'run{run}_train';data={'measured':saved[key+'_pretrigger_adc'][primary[saved[key+'_ids']]]}
        row={'baseline':{},'temporal_checks':{},'fluctuations':{}}
        for group,folder in [('old',args.old),('new',args.new)]:
            path=folder/name/'events.root';data[group],info=load_pre(path);report['inputs_sha256'][str(path.relative_to(PROJECT))]=sha(path)
            if group=='new' and info['electronics_noise']['model']!='gaussian_fir':raise ValueError('Wrong noise model')
        for group,waves in data.items():
            entry,cov,psd,freq=describe(waves);row['baseline'][group]=entry
            arrays[source+'_'+group+'_cov']=cov;arrays[source+'_'+group+'_psd']=psd;arrays['frequency_mhz']=freq
        for suffix in ('middle','late'):
            k=f'run{run}_{suffix}';waves=saved[k+'_pretrigger_adc'][primary[saved[k+'_ids']]]
            row['temporal_checks'][suffix]=describe(waves)[0]
        panel(c,col+1,source+' / first512ns, linear detrend;Lag (ns);Normalized covariance',0,-.18,128,1.25)
        for i,(group,color,text) in enumerate(styles):
            v=arrays[source+'_'+group+'_cov'];graph(np.arange(65)*2.,v/v[0],color,2 if group=='old' else 1);label(text,.45,.84-i*.055,color,.029)
        label('Mean variance calibrated; covariance shape checked',.22,.62,size=.026)
        c.cd(col+3).SetLogy();maximum=max(arrays[source+'_'+g+'_psd'].max() for g,_,_ in styles)
        panel(c,col+3,source+' / baseline spectrum;Frequency (MHz);PSD (ADC^{2}/MHz)',0,.0001,250,float(maximum*8))
        for i,(group,color,text) in enumerate(styles):
            graph(freq[1:],arrays[source+'_'+group+'_psd'][1:],color,2 if group=='old' else 1);label(text,.46,.84-i*.055,color,.029)
        for width in ('50ns','102ns','202ns'):
            row['fluctuations'][width]={
                'measured':newf['sources'][source]['groups']['measured']['metrics'][width]['excess_variance_over_mean'],
                'old':oldf['sources'][source]['groups']['mc']['metrics'][width]['excess_variance_over_mean'],
                'new':newf['sources'][source]['groups']['mc']['metrics'][width]['excess_variance_over_mean']}
        ymax=max(v[g]['p10_p50_p90'][1] for v in row['fluctuations'].values() for g,_,_ in styles)*1.6
        panel(c,col+5,source+' / pulse fluctuation metric;Box-filter width (ns);Excess variance / mean (ADC)',30,0,225,ymax)
        for i,(group,color,text) in enumerate(styles):
            records=[row['fluctuations'][w][group] for w in ('50ns','102ns','202ns')]
            x=np.array([50.,102.,202.]);y=np.array([r['p10_p50_p90'][1] for r in records]);lo=np.array([r['median_event_bootstrap_95'][0] for r in records]);hi=np.array([r['median_event_bootstrap_95'][1] for r in records])
            g=ROOT.TGraphAsymmErrors(3,x,y,np.zeros(3),np.zeros(3),y-lo,hi-y);g.SetMarkerStyle(20+i);g.SetLineColor(color);g.SetMarkerColor(color);g.SetLineWidth(2);g.Draw('LP SAME');KEEP.append(g)
            label(text+f", n={records[0]['n']}",.23,.84-i*.055,color,.029)
        label('Same frozen Q; measured bit0x8000 clear',.23,.62,size=.026)
        report['sources'][source]=row
        print(source,'baseline',row['baseline'],'50ns',row['fluctuations']['50ns'],flush=True)
    save(c,args.out,'noise_and_signal')
    c=ROOT.TCanvas('noise_stability','Recorded noise changes within runs',1300,600);c.Divide(2,1,.003,.003)
    for col,source in enumerate(('Co60','Am241')):
        row=report['sources'][source]
        values=[row['baseline']['measured']['detrended_rms_adc']['p10_p50_p90']]+[row['temporal_checks'][s]['detrended_rms_adc']['p10_p50_p90'] for s in ('middle','late')]
        values=np.array(values);x=np.array([.3,.7,.9]);y=values[:,1].copy()
        panel(c,col+1,source+' / recorded groups;Position in original acquisition fraction;Detrended pretrigger RMS (ADC)',0,0,1,float(values.max()*1.5))
        g=ROOT.TGraphAsymmErrors(3,x,y,np.zeros(3),np.zeros(3),y-values[:,0],values[:,2]-y);g.SetMarkerStyle(20);g.SetLineColor(BLUE);g.SetMarkerColor(BLUE);g.Draw('LP SAME');KEEP.append(g)
        predicted=row['baseline']['new']['detrended_rms_adc']['p10_p50_p90'][1];graph(np.array([0.,1.]),np.array([predicted,predicted]),RED,2)
        label('Measured: median with 10-90% population range',.20,.85,BLUE,.030)
        label('Fixed training-calibrated Gaussian FIR',.20,.78,RED,.030)
        label('Later groups were previously explored; no refit',.20,.66,size=.028)
    save(c,args.out,'noise_stability')
    np.savez_compressed(args.out/'curves.npz',**arrays)
    (args.out/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
