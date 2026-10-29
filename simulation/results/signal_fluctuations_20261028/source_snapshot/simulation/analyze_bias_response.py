"""Existing-bias-run checks of noise, pulse fluctuations and response identifiability.

Source identity is not particle truth. Bias runs do not establish equal deposited
energy, identical geometry, temperature, ADC scale or a single-photoelectron gain.
No Geant4 or SiPM parameter is fitted or changed by this extraction.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import ROOT
from reference import PROJECT, bin_layout, TIME_NS, aligned_shapes
from diagnose_charge import sha
from diagnose_fluctuations import fluctuation_metrics, WINDOWS
from characterize_readout import detrend, periodogram, smooth, candidate_indices
from compare import BLUE, RED, GREEN, KEEP, panel, graph, label, save

HERE = Path(__file__).resolve().parent
RUNS = (3, 7, 10, 11, 12, 13, 14, 18)
MAX_EVENTS = 3000
FREQUENCY_BANDS = ((5., 20.), (20., 80.), (110., 140.))
COLORS = [ROOT.TColor.GetColor(x) for x in ['#205E9D', '#777777', '#7654A3', '#198274', '#DB4927', '#B58119', '#348D38', '#CF6293']]


def evenly_spaced(ids, maximum=MAX_EVENTS):
    ids = np.asarray(ids, dtype=np.int64)
    return ids if len(ids) <= maximum else ids[np.linspace(0, len(ids)-1, maximum, dtype=int)]


def acquisition_segments(count):
    """Use original event positions, never compacted selected-event ranks."""
    i = np.arange(count)
    return np.where(i < int(count*.6), 0, np.where(i < int(count*.8), 1, 2))


def summarize(x, bootstrap=False):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    result = {'n': len(x)}
    if len(x):
        result['p10_p50_p90'] = np.quantile(x, [.1, .5, .9]).tolist()
        if bootstrap:
            rng = np.random.default_rng(283221)
            medians = [np.median(x[rng.integers(len(x), size=len(x))]) for _ in range(500)]
            result['median_event_bootstrap_95'] = np.quantile(medians, [.025, .975]).tolist()
    return result


def pulse_spectra(waves, baseline, t10):
    """Equal 512 ns detrended Hann windows; no fractional interpolation.

    Three nonoverlapping windows start >=500 ns after t10. Local smooth pulse
    curvature remains a low-frequency contaminant. Negative excess is retained.
    """
    waves = np.asarray(waves, dtype=float)
    anchor = np.ceil(t10).astype(int)
    index = np.arange(256)
    freq, pre = periodogram(waves[:, :256])
    power = []; means = []; bands = []
    for offset in (250, 506, 762):
        positions = anchor[:, None] + offset + index
        if np.any(positions >= waves.shape[1]): raise ValueError('Incomplete spectral window')
        values = np.take_along_axis(waves, positions, axis=1)
        mu = values.mean(axis=1)-baseline
        if np.any(mu <= 0): raise ValueError('Nonpositive signal mean in spectral window')
        _, psd = periodogram(values)
        excess = (psd-pre)/mu[:, None]
        power.append(excess); means.append(mu)
        bands.append(np.column_stack([excess[:, (freq >= low) & (freq < high)].sum(axis=1)*(freq[1]-freq[0]) for low, high in FREQUENCY_BANDS]))
    return freq, pre.mean(axis=0), np.mean(power, axis=(0, 1)), np.stack(means, axis=1), np.stack(bands, axis=1)


def make_plots(out, report, arrays):
    ROOT.gROOT.SetBatch(True); ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetTitleFontSize(.040); ROOT.gStyle.SetTitleSize(.038, 'XYZ'); ROOT.gStyle.SetLabelSize(.032, 'XYZ')
    c = ROOT.TCanvas('bias_overview', 'Bias-run comparison', 1450, 1100); c.Divide(2, 2, .003, .003)
    fields = [('q_all', 'Recorded pulse charge / not a gain calibration;Bias voltage (V);Median Q (ADC sample)', 0., 1.25e6),
              ('f50', 'Frozen Q / bit 0x8000 clear;Bias voltage (V);50 ns excess variance / mean (ADC)', 0., .37),
              ('ratio202_50', 'Fluctuation spectral shape;Bias voltage (V);Median 202 ns / median 50 ns', 2.5, 5.5),
              ('baseline', 'Common-Q cohort baseline;Bias voltage (V);Median detrended RMS (ADC)', 0., 7.)]
    for col, (field, title, low, high) in enumerate(fields):
        panel(c, col+1, title, 52.2, low, 54.2, high)
        for run, color in zip(RUNS, COLORS):
            row = report['runs'][str(run)]; group = row['groups']['first60']; values = group['pulse']
            if field == 'q_all': value = group['all_primary_bit_clear_charge']['p10_p50_p90'][1]
            elif field == 'baseline': value = values['baseline_rms']['p10_p50_p90'][1]
            elif field == 'f50': value = values['fluctuations']['50ns']['p10_p50_p90'][1]
            else: value = values['fluctuations']['202ns']['p10_p50_p90'][1]/values['fluctuations']['50ns']['p10_p50_p90'][1]
            x = np.array([row['bias_V']]); y = np.array([value])
            if field=='f50':
                lo,hi=values['fluctuations']['50ns']['median_event_bootstrap_95']
                g=ROOT.TGraphAsymmErrors(1,x,y,np.zeros(1),np.zeros(1),np.array([value-lo]),np.array([hi-value]))
            else:g = ROOT.TGraph(1, x, y)
            g.SetLineColor(color);g.SetMarkerStyle(24 if run in (3, 7) else 20); g.SetMarkerSize(1.2); g.SetMarkerColor(color); g.Draw('P SAME'); KEEP.append(g)
            y_label=value+(high-low)*.015 if field=='q_all' and run==3 else value
            if field=='q_all' and run==18:y_label=value-(high-low)*.015
            text = ROOT.TLatex(float(x[0]+.025), float(y_label), str(run)); text.SetTextSize(.029); text.SetTextColor(color); text.Draw(); KEEP.append(text)
        label('First 60% of each run; source/geometry confounding remains', .18, .86, size=.027)
        label('Open: Co; filled: Am; numbers are run IDs', .18, .80, size=.027)
    save(c, out, 'bias_overview')
    c = ROOT.TCanvas('bias_time', 'Temporal checks within bias runs', 1450, 1100); c.Divide(2, 2, .003, .003)
    for col, (metric, title, low, high) in enumerate([
        ('f50', 'Frozen Q interval / pulse fluctuations;Original acquisition fraction;50 ns excess variance / mean (ADC)', 0., .38),
        ('baseline', 'Frozen Q interval / baseline stability;Original acquisition fraction;Detrended pretrigger RMS (ADC)', 0., 9.),
        ('ratio', 'Frozen Q interval / pulse-shape stability;Original acquisition fraction;Short / long charge median', .27, .56),
        ('band', 'Pulse spectrum / 20-80 MHz;Original acquisition fraction;Excess band power / mean signal (ADC)', -.02, .18),
    ]):
        panel(c, col+1, title, 0., low, 1., high)
        for i, (run, color) in enumerate(zip(RUNS, COLORS)):
            ys=[]
            for group in report['runs'][str(run)]['groups'].values():
                row=group['pulse']
                if metric=='f50':value=row['fluctuations']['50ns']['p10_p50_p90'][1]
                elif metric=='baseline':value=row['baseline_rms']['p10_p50_p90'][1]
                elif metric=='ratio':value=row['short_long_ratio']['p10_p50_p90'][1]
                else:value=row['spectral_bands']['20-80MHz']['p10_p50_p90'][1]
                ys.append(value)
            graph(np.array([.3,.7,.9]),np.array(ys),color)
            label(f'Run {run}',.20+(i%4)*.18,.87-(i//4)*.05,color,.028)
        label('Previous checks are not blind validation',.20,.23 if metric=='ratio' else .73,size=.026)
    save(c,out,'bias_time_checks')
    c=ROOT.TCanvas('bias_shapes','Charge-conditioned pulse shapes',1450,1100);c.Divide(2,2,.003,.003)
    for i, (first,last) in enumerate([(-100.,800.),(0.,3500.)]):
        panel(c,i+1,'First 60% / common Q;Time from t10 (ns);Area-normalized ADC (1/ns)',first,0.,last,.00165)
        for j,(run,color) in enumerate(zip(RUNS,COLORS)):
            graph(TIME_NS,arrays[f'run{run}_first60_shape'],color)
            label(f'Run {run}',.49+(j%2)*.21,.86-(j//2)*.05,color,.027)
    for col, runs in enumerate([(3,7),(10,11,12,13,14,18)]):
        panel(c,col+3,'First 60% / per-charge-bin medians;Q (10^{3} ADC sample);50 ns excess variance / mean (ADC)',report['q_edges'][0]/1000,0.,report['q_edges'][-1]/1000,.40)
        for j,run in enumerate(runs):
            row=report['runs'][str(run)]['groups']['first60'];valid=[b for b in row['pulse']['charge_bins'] if b['n']>=20]
            color=COLORS[RUNS.index(run)]
            if valid:graph(np.array([(b['low']+b['high'])/2000 for b in valid]),np.array([b['fluctuation_medians'][0] for b in valid]),color)
            label(f'Run {run}',.21+(j%3)*.23,.85-(j//3)*.05,color,.029)
        label('Only bins with at least 20 sampled events shown',.20,.69,size=.026)
    save(c,out,'bias_charge_shapes')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--replot',action='store_true',help='Regenerate figures from existing numbers, without repeating extraction');args=p.parse_args()
    if args.replot:
        report=json.loads((args.out/'bias_response.json').read_text())
        with np.load(args.out/'samples.npz') as z:arrays=dict(z)
        make_plots(args.out,report,arrays)
        record={'scope':'Plot-only refresh; extraction source and numeric results unchanged','plot_code_sha256':sha(Path(__file__)),
            'inputs_sha256':{p.name:sha(p) for p in [args.out/'bias_response.json',args.out/'samples.npz']}}
        (args.out/'plot_record.json').write_text(json.dumps(record,indent=2)+'\n')
        return
    args.out.mkdir(parents=True,exist_ok=False)
    cohort=PROJECT/'analysis/paper_baseline/results/confirmed_cohort.json';provenance=PROJECT/'analysis/paper_baseline/user_provenance.json'
    qlo,qhi=json.loads(cohort.read_text())['common_charge_range_adc_samples'];edges=np.linspace(qlo,qhi,7)
    labels={int(r['run']):r for r in json.loads(provenance.read_text())['confirmed_runs']}
    report={'scope':__doc__,'q_edges':edges.tolist(),
        'selection':'Original primary QC, amplitude>=200 ADC, hardware bit0x8000 clear, frozen common Q, t10>=384 samples. First60/next20/last20 by original indices for every run. Up to3000 evenly spaced selected events per run/segment.',
        'noise_selection':'Basic diagnostic_flags==0 and t10>=384, independent of Q/amplitude/hardware bit. Up to3000 evenly spaced per segment; positive/negative candidate threshold uses the fixed first60 pooled robust scale.',
        'roles':'Run3/12 first60 retain training roles. All other groups are exploratory diagnostics, not additional calibrated or blind validation samples. Existing feature split arrays are never modified.',
        'fluctuation_method':'Existing 50/102/202ns centered-box residual metrics, sample variance in [-600,-200)ns versus [500,2500)ns relative to t10, divided by mean signal. Negative excess retained.',
        'spectral_method':'First256 samples versus three successive256-sample windows starting ceil(t10)+250,+506,+762. Each window linearly detrended and Hann tapered, identical PSD normalization. Per-event pulse-minus-pre PSD divided by local mean signal, averaged across three windows. Bandpower uses frequency-bin rectangles at fixed bands. Smooth pulse curvature can contaminate low frequencies.',
        'uncertainty':'500 event-bootstrap medians seed283221; descriptive recorded-event uncertainty only, not model or setup uncertainty. Deterministic subsampling retained.',
        'limits':['No known equal deposited energy or pure alpha population across bias runs.','Only runs3/12/18 have explicit same crystal/SiPM/wiring confirmation.','Current settings XML is not proven to describe each historical BIN.','Q matching is ADC-charge matching, not deposited-energy or particle matching.','Weak-pulse candidate area is not SPE gain; triggered intervals do not measure dark rate.'],
        'inputs_sha256':{str(f.relative_to(PROJECT)):sha(f) for f in [cohort,provenance]},
        'code_sha256':{str(f.relative_to(PROJECT)):sha(f) for f in [Path(__file__),HERE/'characterize_readout.py',HERE/'diagnose_fluctuations.py',HERE/'reference.py',PROJECT/'analysis/paper_baseline/prepare.py']},'runs':{}}
    arrays={}
    for run in RUNS:
        paths=[PROJECT/f'analysis/paper_baseline/results/pha_run{run:04d}_features.npz',PROJECT/f'analysis/paper_baseline/results/phase2/run{run:04d}_features.npz',PROJECT/f'data/DT5730_DPP_PHA/DAQ/run_{run}/UNFILTERED/Data_run_{run}.BIN']
        with np.load(paths[0]) as z:f=dict(z)
        with np.load(paths[1]) as z:a={k:z[k] for k in ['eligible_primary','split']}
        _,dtype,count=bin_layout(paths[2]);raw=np.memmap(paths[2],dtype=dtype,offset=2,mode='r',shape=(count,))
        if count!=len(f['amplitude']):raise ValueError('Feature/raw event mismatch')
        segment=acquisition_segments(count)
        basic=(f['diagnostic_flags']==0)&(f['t10_sample']>=384)
        primary=a['eligible_primary']&(f['amplitude']>=200)&((f['flags']&0x8000)==0)&basic
        common=primary&(f['charge_long']>=qlo)&(f['charge_long']<=qhi)
        row={'source':labels[run]['source_or_background'],'bias_V':float(labels[run]['bias_voltage_V']),'events_recorded':count,'groups':{}}
        scale=None
        for part,name in enumerate(['first60','middle20','last20']):
            key=f'run{run}_{name}';mask=segment==part
            ids=evenly_spaced(np.flatnonzero(mask&common))
            if len(ids)<20:raise ValueError(f'Insufficient selected events: {key}: {len(ids)}')
            waves=raw['data'][ids].astype(float);baseline=f['baseline'][ids];t10=f['t10_sample'][ids]
            metrics=fluctuation_metrics(waves,baseline,t10)
            freq,pre_psd,excess_psd,means,bands=pulse_spectra(waves,baseline,t10)
            residual=detrend(waves[:,:256]);q=f['charge_long'][ids]
            shapes,keep=aligned_shapes(waves,{k:v[ids] for k,v in f.items()})
            if len(keep)!=len(ids):raise ValueError('Unexpected shape-selection change')
            pulse={'n_available':int(np.sum(mask&common)),'n_used':len(ids),'charge':summarize(q),'short_long_ratio':summarize(f['short_long_ratio'][ids]),'baseline_rms':summarize(residual.std(axis=1)),
                'fluctuations':{str(2*w)+'ns':summarize(metrics[:,j,3],True) for j,w in enumerate(WINDOWS)},
                'spectral_bands':{f'{low:g}-{high:g}MHz':summarize(bands.mean(axis=1)[:,j],True) for j,(low,high) in enumerate(FREQUENCY_BANDS)},'charge_bins':[]}
            for j,(low,high) in enumerate(zip(edges[:-1],edges[1:])):
                sel=(q>=low)&((q<high)|((j==len(edges)-2)&(q==high)))
                pulse['charge_bins'].append({'low':float(low),'high':float(high),'n':int(sel.sum()),'fluctuation_medians':np.median(metrics[sel,:,3],axis=0).tolist() if sel.any() else None})
            enough=all(b['n']>=20 for b in pulse['charge_bins'])
            pulse['equal_bin_mean_of_medians']=np.mean([b['fluctuation_medians'] for b in pulse['charge_bins']],axis=0).tolist() if enough else None
            for suffix,value in {'ids':ids,'q':q,'metrics':metrics,'baseline_rms':residual.std(axis=1),'ratios':f['short_long_ratio'][ids],'pre_psd':pre_psd,'excess_psd':excess_psd,'spectral_means':means,'spectral_bands':bands,'shape':shapes.mean(axis=0)}.items():arrays[key+'_'+suffix]=value
            noise_ids=evenly_spaced(np.flatnonzero(mask&basic));noise_waves=raw['data'][noise_ids,:256].astype(float);noise=detrend(noise_waves)
            if scale is None:
                z=smooth(noise);scale=float(np.median(abs(z-np.median(z)))/.6744897501960817)
            noise_row={'n_available':int(np.sum(mask&basic)),'n_used':len(noise_ids),'rms':summarize(noise.std(axis=1)),'candidate_scale_adc':scale,'weak_candidates':{}}
            for threshold in (4.,5.,6.):
                entries={}
                for sign,sign_name in [(1,'positive'),(-1,'negative')]:
                    ii,peak,amp=candidate_indices(noise_waves,scale,threshold,sign)
                    snippets=sign*noise[ii[:,None],peak[:,None]+np.arange(-50,101)]
                    area=snippets[:,40:100].sum(axis=1)*2
                    entries[sign_name]={'n':len(ii),'area_adc_ns':summarize(area),'amplitude_adc':summarize(amp)}
                    if threshold==5.:
                        arrays[key+'_'+sign_name+'_candidate_ids']=noise_ids[ii];arrays[key+'_'+sign_name+'_candidate_area']=area
                        if len(ii):arrays[key+'_'+sign_name+'_candidate_mean']=snippets.mean(axis=0)
                noise_row['weak_candidates'][str(threshold)]=entries
            arrays[key+'_noise_ids']=noise_ids
            row['groups'][name]={'original_event_range':[0 if part==0 else int(count*(.6 if part==1 else .8)),int(count*(.6 if part==0 else .8)) if part<2 else count],
                'primary_bit_clear_n':int(np.sum(mask&primary)),'all_primary_bit_clear_charge':summarize(f['charge_long'][mask&primary]),'pulse':pulse,'noise':noise_row}
            report['runs'][str(run)]=row
            (args.out/'progress.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
            print(key,'available/used',pulse['n_available'],len(ids),'f50',pulse['fluctuations']['50ns']['p10_p50_p90'][1],'rms',pulse['baseline_rms']['p10_p50_p90'][1],flush=True)
        for path in paths:report['inputs_sha256'][str(path.relative_to(PROJECT))]=sha(path)
    arrays['frequency_mhz']=freq;arrays['time_ns']=TIME_NS
    np.savez_compressed(args.out/'samples.npz',**arrays)
    (args.out/'bias_response.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    make_plots(args.out,report,arrays)


if __name__=='__main__':main()
