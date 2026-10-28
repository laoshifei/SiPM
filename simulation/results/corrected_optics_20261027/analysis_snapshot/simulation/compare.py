"""Compare independently generated ADC waveforms with frozen measured references.

No measured waveform template is inserted into simulated signals, no per-source
amplitude rescaling is applied, and the experimental bands are not truth labels.
"""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import ROOT
from reference import PROJECT,TIME_NS,aligned_shapes,wave_features
from calibrate import metrics
from refine_waveforms import advanced_features

HERE=Path(__file__).resolve().parent
KEEP=[]
BLUE=ROOT.TColor.GetColor('#215A8C');RED=ROOT.TColor.GetColor('#CB593A');GREEN=ROOT.TColor.GetColor('#258275')

def load_mc(paths):
    blocks=[];truth=[];trigger=[];event_ids=[];infos=[]
    for file_index,path in enumerate(paths):
        file=ROOT.TFile.Open(str(path));assert file and not file.IsZombie()
        tree=file.Get('Events');info=json.loads(str(file.Get('RunInfo').GetString().Data()))
        assert info.get('completed',True),'Refuse an incomplete Geant4 run'
        assert tree.GetEntries()==info['events_requested']
        infos.append(info);tree.SetBranchStatus('*',0)
        for name in ('adc','edep_MeV','edep_alpha_MeV','approximate_trigger','event_id'):tree.SetBranchStatus(name,1)
        for event in tree:
            blocks.append(np.array(event.adc,dtype=float));truth.append([event.edep_MeV,event.edep_alpha_MeV])
            trigger.append(event.approximate_trigger);event_ids.append([file_index,event.event_id])
        file.Close()
    assert len({x['source'] for x in infos})==1
    assert all(x['config']==infos[0]['config'] for x in infos),'Do not merge different configurations'
    assert all(x.get('schema_version',1)==infos[0].get('schema_version',1)
               and x.get('sensor_entry_policy')==infos[0].get('sensor_entry_policy') for x in infos),'Do not merge different optical scoring models'
    waves=np.array(blocks);f=wave_features(waves);truth=np.array(truth)
    a=advanced_features(waves,f['t10_sample'],f['peak_sample'],f['amplitude'],f['baseline'],float(np.median(f['baseline_rms'])))
    # The measured reference additionally required 4 us support. Simulation
    # decay is injected at 1000 ns (not a hardware-trigger model), so its full
    # 3 us comparison gate and plotted support are checked explicitly instead.
    good=(f['diagnostic_flags']==0)&~a['pretrigger_flag_primary']
    qlo,qhi=json.loads((PROJECT/'analysis/paper_baseline/results/confirmed_cohort.json').read_text())['common_charge_range_adc_samples']
    common=good&(f['charge_long']>=qlo)&(f['charge_long']<=qhi)
    result={'source':infos[0]['source'],'files':[str(p) for p in paths],
        'file_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        'config':infos[0]['config'],'schema_version':infos[0].get('schema_version',1),
        'sensor_entry_policy':infos[0].get('sensor_entry_policy','Legacy post-volume scoring'),
        'parent_decays':len(waves),'positive_deposit_events':int(np.sum(truth[:,0]>0)),
        'approximate_trigger_events':int(np.sum(trigger)),'quality_events':int(good.sum()),'common_charge_events':int(common.sum()),
        'common_charge_range_adc_samples':[qlo,qhi],'groups':{}}
    curves={};samples={}
    for name,mask in [('quality',good),('common',common),('common_alpha_truth',common&(truth[:,1]>0)),('common_no_alpha_truth',common&(truth[:,1]==0))]:
        idx=np.flatnonzero(mask)
        if not len(idx):continue
        shapes,selected=aligned_shapes(waves[idx],{k:v[idx] for k,v in f.items()});idx=idx[selected]
        if not len(idx):continue
        curves[name+'_mean']=shapes.mean(axis=0);curves[name+'_sem']=shapes.std(axis=0,ddof=1)/np.sqrt(len(idx)) if len(idx)>1 else np.full(len(TIME_NS),np.nan)
        if name=='common':curves[name+'_samples']=shapes
        samples[name+'_charge']=f['charge_long'][idx];samples[name+'_ratio']=f['short_long_ratio'][idx]
        result['groups'][name]={'n':len(idx),'event_ids':np.array(event_ids)[idx].tolist(),
            'alpha_truth_fraction':float(np.mean(truth[idx,1]>0)),
            'charge_adc_samples_p10_p50_p90':np.quantile(f['charge_long'][idx],[.1,.5,.9]).tolist(),
            'ratio_p10_p50_p90':np.quantile(f['short_long_ratio'][idx],[.1,.5,.9]).tolist(),
            'baseline_rms_adc_median':float(np.median(f['baseline_rms'][idx])),
            'amplitude_adc_median':float(np.median(f['amplitude'][idx]))}
    return result,curves,samples

def label(text,x,y,color=1,size=.033):
    t=ROOT.TLatex(x,y,text);t.SetNDC();t.SetTextFont(42);t.SetTextSize(size);t.SetTextColor(color);t.Draw();KEEP.append(t)

def graph(x,y,color,style=1):
    x=np.ascontiguousarray(x,dtype='float64');y=np.ascontiguousarray(y,dtype='float64')
    g=ROOT.TGraph(len(x),x,y);g.SetLineColor(color);g.SetLineStyle(style);g.SetLineWidth(2);g.Draw('L SAME');KEEP.append(g)

def panel(c,i,title,xmin,ymin,xmax,ymax):
    p=c.cd(i);p.SetLeftMargin(.15);p.SetRightMargin(.04);p.SetBottomMargin(.16);p.SetTopMargin(.09);p.SetTicks(1,1)
    h=p.DrawFrame(xmin,ymin,xmax,ymax,title);h.GetXaxis().SetTitleOffset(1.2);h.GetYaxis().SetTitleOffset(1.55);KEEP.append(h);return p

def save(c,out,name):
    c.Modified();c.Update()
    for ext in ('svg','png'):c.SaveAs(str(out/f'{name}.{ext}'))

def waveform_figure(out,refs,predictions,mc,report):
    c=ROOT.TCanvas('waveform_validation','CsI source waveform comparison',1450,950);c.Divide(3,2,.003,.003)
    for i,(key,source,title) in enumerate([('run3_late','Co60','Co-60 / run 3 late'),('run12_late','Am241','Am-241 / run 12 late'),('run18_check','Am241','Am-241 / run 18')]):
        obs=refs[key+'_mean'];analytic=predictions['Co60' if source=='Co60' else 'Am241_dominant_hypothesis'];sim=mc[source]['common_mean']
        panel(c,i+1,title+';Time from t_{10} (#mus);Normalized signal (#mus^{-1})',-.2,0,3.8,float(max(obs.max(),sim.max())*1200))
        graph(TIME_NS/1000,obs*1000,BLUE);graph(TIME_NS/1000,sim*1000,RED);graph(TIME_NS/1000,analytic*1000,GREEN,2)
        label('Measured mean',.53,.84,BLUE);label('Geant4 + SiPM ADC',.53,.79,RED);label('Fast model fit',.53,.74,GREEN)
        label(f"MC n = {report['mc'][source]['groups']['common']['n']}",.53,.67)
        nrmse=report['comparisons'][key]['mc_mean']['nrmse_peak']*100
        label(f'Peak NRMSE = {nrmse:.2f}%',.53,.62)
        delta=(sim-obs)/obs.max()*100;analytic_delta=(analytic-obs)/obs.max()*100
        ymax=max(10,float(np.max(np.abs(delta[(TIME_NS>=0)&(TIME_NS<=3500)])))*1.15)
        panel(c,i+4,'Residual / measured peak;Time from t_{10} (#mus);Residual (%)',-.2,-ymax,3.8,ymax)
        graph(TIME_NS/1000,np.zeros(len(TIME_NS)),ROOT.kGray+1,2);graph(TIME_NS/1000,delta,RED);graph(TIME_NS/1000,analytic_delta,GREEN,2)
        label('Same charge gate; event-area normalization',.18,.84,size=.029)
        label('Previously explored check; not a blind test',.18,.79,size=.029)
    save(c,out,'waveform_comparison')

def distribution_figure(out,samples):
    c=ROOT.TCanvas('charge_ratio','Charge and shape diagnostic',1300,850);c.Divide(2,2,.003,.003)
    for col,(source,run) in enumerate([('Co60',3),('Am241',12)]):
        f=np.load(PROJECT/f'analysis/paper_baseline/results/pha_run{run:04d}_features.npz')
        a=np.load(PROJECT/f'analysis/paper_baseline/results/phase2/run{run:04d}_features.npz')
        # Training portion only, so this spectrum panel is a calibration diagnostic.
        mask=a['eligible_primary']&(a['split']==0);common=mask&a['common_charge']
        for row,key in enumerate(('charge','ratio')):
            obs=f['charge_long'][mask]/1000 if key=='charge' else f['short_long_ratio'][common]
            sim=samples[source]['quality_charge']/1000 if key=='charge' else samples[source]['common_ratio']
            bins=np.linspace(0,1200,49) if key=='charge' else np.linspace(.15,.65,51)
            curves=[]
            for values in (obs,sim):
                counts,_=np.histogram(values,bins);curves.append(counts/max(1,len(values)))
            panel(c,1+col+2*row,f'{source} / run {run} training;'+('Charge (10^{3} ADC samples);Fraction per bin' if key=='charge' else 'Short / long charge;Fraction per bin'),float(bins[0]),0,float(bins[-1]),float(max(map(np.max,curves))*1.35))
            x=(bins[1:]+bins[:-1])/2
            graph(x,curves[0],BLUE);graph(x,curves[1],RED)
            label('Measured',.65,.83,BLUE);label('Geant4 + SiPM',.65,.78,RED)
            label('No amplitude rescaling',.42,.70,size=.031)
            label('Trigger/capsule/gain uncalibrated',.19,.87,size=.028)
    save(c,out,'charge_shape_diagnostics')

def distance_figure(out,path):
    data=json.loads(path.read_text());r=data['runs'];x=np.array([a['source_gap_mm'] for a in r]);prob=np.array([a['alpha_deposit_fraction'] for a in r]);energy=np.array([a['alpha_deposit_median_MeV'] or 0 for a in r])
    c=ROOT.TCanvas('distance','Air gap sensitivity',1300,550);c.Divide(2,1,.003,.003)
    panel(c,1,'Point Am-241 source + 5 #mum Mylar;Air gap (mm);Fraction of parent decays with #alpha deposit',0,0,21,.5)
    low=np.array([a['alpha_deposit_fraction_wilson95'][0] for a in r]);high=np.array([a['alpha_deposit_fraction_wilson95'][1] for a in r])
    g=ROOT.TGraphAsymmErrors(len(x),x,prob,np.zeros(len(x)),np.zeros(len(x)),prob-low,high-prob);g.SetMarkerStyle(20);g.SetLineColor(BLUE);g.SetMarkerColor(BLUE);g.Draw('LP SAME');KEEP.append(g)
    label('95% binomial intervals; no hardware trigger',.20,.84,size=.031)
    panel(c,2,'Deposited alpha energy;Air gap (mm);Median E_{dep,#alpha} (MeV)',0,0,21,5)
    graph(x,energy,RED);label('Conditional on #alpha reaching CsI',.20,.84,size=.033)
    label('No source capsule/backing modeled',.20,.78,size=.033)
    label('Sensitivity scan, not a measured distance',.20,.35,size=.033)
    save(c,out,'distance_sensitivity')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--co',type=Path,nargs='+',required=True);p.add_argument('--am',type=Path,nargs='+',required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--distance',type=Path)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    ROOT.gROOT.SetBatch(True);ROOT.gStyle.SetOptStat(0);ROOT.gStyle.SetTitleFontSize(.042)
    ROOT.gStyle.SetLabelSize(.035,'XYZ');ROOT.gStyle.SetTitleSize(.039,'XYZ')
    refs=dict(np.load(HERE/'results/reference/references.npz'));pred=dict(np.load(HERE/'results/calibration/expectations.npz'))
    report={'scope':'Pilot full optical transport + SiPM ADC, with mean-shape parameters frozen from training. No source-specific amplitude rescaling.',
        'selection':'MC basic QC, primary pretrigger veto, full 3 us integral and aligned [-200,3800] ns support; common Q uses frozen measured interval. Measured reference additionally required 4 us support. No ratio cut or MC truth cut for primary comparison.',
        'uncertainties':'Finite MC sample; no calibrated source capsule, absolute gain/light yield/PDE, or hardware trigger. Event distribution agreement is not established by mean-shape agreement.',
        'analysis_code_sha256':{str(path.relative_to(PROJECT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in [HERE/'compare.py',HERE/'calibrate.py',HERE/'reference.py',PROJECT/'analysis/paper_baseline/prepare.py',PROJECT/'analysis/paper_baseline/refine_waveforms.py']},
        'reference_sha256':hashlib.sha256((HERE/'results/reference/references.npz').read_bytes()).hexdigest(),
        'fit_summary_sha256':hashlib.sha256((HERE/'results/calibration/fit.json').read_bytes()).hexdigest(),
        'mc':{},'comparisons':{}};curves={};samples={};saved={'time_ns':TIME_NS}
    for source,paths in [('Co60',args.co),('Am241',args.am)]:
        report['mc'][source],curves[source],samples[source]=load_mc(paths)
        if 'common_mean' not in curves[source]:raise ValueError(f'{source}: no events in common charge; do not extrapolate a waveform comparison')
        for k,v in {**curves[source],**samples[source]}.items():saved[source+'_'+k]=v
    for key in ('run3_train','run3_middle','run3_late','run12_train','run12_middle','run12_late','run18_check'):
        source='Co60' if key.startswith('run3_') else 'Am241';obs=refs[key+'_mean']
        report['comparisons'][key]={'mc_mean':metrics(obs,curves[source]['common_mean']),
            'analytic_mean':metrics(obs,pred['Co60' if source=='Co60' else 'Am241_dominant_hypothesis'])}
        shapes=curves[source]['common_samples'];rng=np.random.default_rng(271999)
        boot=[metrics(obs,shapes[rng.integers(0,len(shapes),len(shapes))].mean(axis=0))['nrmse_peak'] for _ in range(1000)]
        report['comparisons'][key]['mc_resampling_nrmse_95']=np.quantile(boot,[.025,.975]).tolist()
        report['comparisons'][key]['mc_resampling_scope']='1000 whole-event bootstrap replicates, seed 271999, measured mean and calibrated parameters fixed; not total model uncertainty'
    np.savez_compressed(args.out/'curves.npz',**saved);(args.out/'comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    waveform_figure(args.out,refs,pred,curves,report);distribution_figure(args.out,samples)
    if args.distance:distance_figure(args.out,args.distance)
    print(json.dumps({k:v for k,v in report['comparisons'].items()},indent=2))

if __name__=='__main__':main()
