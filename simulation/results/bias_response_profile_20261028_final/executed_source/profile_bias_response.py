"""Conditional impulse-response identifiability checks, not a SiPM gain fit.

The independent stationary-avalanche approximation is profiled over a two-pole
shape and a free effective ADC area. Positive model discrepancies cannot be
assigned uniquely to gain, excess noise, impedance or unmodeled signal noise.
The resulting diagnostic profiles must not be exported as detector calibrations.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import ROOT
from reference import PROJECT
from diagnose_charge import sha
from profile_signal_response import box_transfer, variance_integrals, WIDTHS
from compare import BLUE, RED, GREEN, KEEP, panel, graph, label, save

HERE=Path(__file__).resolve().parent


def kernel_grid(decays, rcs, bins=8000):
    """The same aperture and finite-window integral as profile_signal_response.

    Trapezoidal weights permit a matrix product instead of repeated integrals.
    The two free pole factors commute: their physical allocation is unidentifiable.
    """
    f=np.geomspace(1e-6,5.,bins)
    df=np.r_[f[1]-f[0],f[2:]-f[:-2],f[-1]-f[-2]]/2
    finite=(1000/999)*(1-box_transfer(f,1000)**2)
    weights=np.array([2*(1-box_transfer(f,n))**2*finite*np.sinc(2*f)**2*df/(1+(2*np.pi*f*2)**2) for n in WIDTHS])
    d=1/(1+(2*np.pi*np.asarray(decays)[:,None]*f)**2)
    r=1/(1+(2*np.pi*np.asarray(rcs)[:,None]*f)**2)
    return np.stack([(d*weight)@r.T for weight in weights],axis=2)


def profile(observed, kernels):
    observed=np.asarray(observed,dtype=float)
    if observed.shape!=(3,) or np.any(observed<=0):raise ValueError('Three positive filter metrics required')
    area=np.exp(np.mean(np.log(observed/kernels),axis=2))
    prediction=kernels*area[:,:,None]
    loss=np.mean(np.log(prediction/observed)**2,axis=2)
    return area,prediction,loss


def summarize_profile(observed,kernels,decays,rcs):
    area,prediction,loss=profile(observed,kernels);i,j=np.unravel_index(loss.argmin(),loss.shape)
    regions={}
    for tolerance in (.05,.10,.20):
        mask=np.all(abs(prediction/observed-1)<=tolerance,axis=2)
        ii,jj=np.where(mask)
        regions[str(tolerance)]={'grid_points':int(mask.sum()),
            'decay_ns_min_max':[float(decays[ii].min()),float(decays[ii].max())] if len(ii) else None,
            'RC_ns_min_max':[float(rcs[jj].min()),float(rcs[jj].max())] if len(ii) else None,
            'effective_ADC_area_ns_min_max':[float(area[mask].min()),float(area[mask].max())] if len(ii) else None,
            'touches_decay_upper_grid_edge':bool(np.any(ii==len(decays)-1)),
            'touches_RC_upper_grid_edge':bool(np.any(jj==len(rcs)-1))}
    return {'observed_filter_medians':list(observed),'best_grid':{'SPE_decay_ns':float(decays[i]),'RC_ns':float(rcs[j]),'effective_area_ADC_ns':float(area[i,j]),
        'predictions':prediction[i,j].tolist(),'fractional_residuals':(prediction[i,j]/observed-1).tolist(),'rms_log_residual':float(np.sqrt(loss[i,j]))},
        'descriptive_relative_tolerance_regions':regions},loss


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bias',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    args.bias=args.bias.resolve();args.out.mkdir(parents=True,exist_ok=False)
    input_path=args.bias/'bias_response.json';data=json.loads(input_path.read_text())
    decays=np.geomspace(2.5,200.,101);rcs=np.r_[0.,np.geomspace(.1,300.,120)];kernels=kernel_grid(decays,rcs)
    cfg_path=HERE/'results/colored_noise_20261028/co60/config.json';cfg=json.loads(cfg_path.read_text());r=cfg['sipm']['readout'];s=cfg['sipm']['sensor']
    nominal_area=1.602176634e-19*s['gain_electrons']*r['transimpedance_ohm']*1e12/((r['adc_max_mV']-r['adc_min_mV'])/2**r['adc_bits'])
    report={'scope':__doc__,'assumptions':{'fixed_rise_ns':2.,'SPE_decay_grid_ns':decays.tolist(),'readout_RC_grid_ns':rcs.tolist(),'nominal_avalanche_area_ADC_ns':nominal_area,
        'objective':'Unweighted mean squared logarithmic relative discrepancy for medians at50/102/202ns. Area minimized analytically in log space at each kernel. Not chi-square or likelihood.',
        'tolerance_regions':'Grid points where all3 relative discrepancies <=5/10/20%, after log-optimal area scaling. Descriptive and grid-limited, NOT confidence or credible intervals. A region reaching a grid edge is not bounded by the data.',
        'degeneracy':'The Fourier factors for SPE decay and additional RC commute exactly; this statistic alone cannot assign the two poles to physical components.'},
        'limits':['Locally stationary independent avalanche rate; finite decaying pulses, recovery, mixed signals, signal-dependent noise and deterministic curvature are not fully represented.',
        'Effective area may combine gain, impedance, excess-noise factors and model error; not gain_electrons or an SPE measurement.',
        'Voltage alone does not establish equal source geometry, deposited energy, historical ADC conversion or temperature.','No fitted profile is applied to Geant4 configuration.'],
        'inputs_sha256':{str(f.relative_to(PROJECT)):sha(f) for f in [input_path,cfg_path]},
        'code_sha256':{str(f.relative_to(PROJECT)):sha(f) for f in [Path(__file__),HERE/'profile_signal_response.py']},'runs':{}}
    arrays={'decays_ns':decays,'rcs_ns':rcs,'kernels':kernels};observations={}
    for run,row in data['runs'].items():
        report['runs'][run]={'bias_V':row['bias_V'],'groups':{}}
        for name,group in row['groups'].items():
            observed=np.array([group['pulse']['fluctuations'][str(2*w)+'ns']['p10_p50_p90'][1] for w in WIDTHS])
            result,loss=summarize_profile(observed,kernels,decays,rcs)
            report['runs'][run]['groups'][name]=result
            if name=='first60':arrays[f'run{run}_log_loss']=loss;observations[run]=observed
    # The same hardware at 53.2 V must not be assigned source-specific gain just
    # to remove the discrepancy. Compare that constraint to an explicit relaxation.
    for names in [('3','12'),('12','18'),('3','12','18')]:
        obs=np.array([observations[run] for run in names]);key='_'.join(names)
        common_area=np.exp(np.mean(np.log(obs[:,None,None,:]/kernels[None,:,:,:]),axis=(0,3)))
        prediction=kernels*common_area[:,:,None]
        loss=np.mean(np.log(prediction[None,:,:,:]/obs[:,None,None,:])**2,axis=(0,3));i,j=np.unravel_index(loss.argmin(),loss.shape)
        relaxed=[profile(o,kernels) for o in obs];relaxed_loss=np.mean([v[2] for v in relaxed],axis=0);ii,jj=np.unravel_index(relaxed_loss.argmin(),loss.shape)
        report.setdefault('shared_53p2V_checks',{})[key]={'runs':list(names),'one_kernel_one_area':{'SPE_decay_ns':float(decays[i]),'RC_ns':float(rcs[j]),'effective_area_ADC_ns':float(common_area[i,j]),
            'prediction':prediction[i,j].tolist(),'fractional_residuals_per_run':(prediction[i,j]/obs-1).tolist(),'rms_log_residual':float(np.sqrt(loss[i,j]))},
            'one_kernel_separate_diagnostic_areas':{'SPE_decay_ns':float(decays[ii]),'RC_ns':float(rcs[jj]),'areas_ADC_ns':[float(v[0][ii,jj]) for v in relaxed],
            'fractional_residuals_per_run':[(v[1][ii,jj]/o-1).tolist() for v,o in zip(relaxed,obs)],'rms_log_residual':float(np.sqrt(relaxed_loss[ii,jj]))},
            'meaning':'Conditional conflict of stationary common-response approximation, not proof of different physical gains or particle-dependent electronics.'}
    anchor=nominal_area*variance_integrals([r['lowpass_ns']],rise=r['rise_ns'],decay=r['decay_ns'])[0]
    for folder in ['colored_noise_20261028','joint_optical_20261028']:
        path=HERE/'results'/folder/'fluctuations/diagnostics.json'
        if not path.exists():continue
        mc=json.loads(path.read_text());report['inputs_sha256'][str(path.relative_to(PROJECT))]=sha(path)
        report.setdefault('known_full_MC_anchor',{})[folder]={source:{'analytic':anchor.tolist(),
            'full_MC_medians':[row['groups']['mc']['metrics'][str(2*w)+'ns']['excess_variance_over_mean']['p10_p50_p90'][1] for w in WIDTHS]} for source,row in mc['sources'].items()}
    np.savez_compressed(args.out/'profiles.npz',**arrays)
    (args.out/'profiles.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    ROOT.gROOT.SetBatch(True);ROOT.gStyle.SetOptStat(0);ROOT.gStyle.SetTitleFontSize(.039);ROOT.gStyle.SetLabelSize(.031,'XYZ');ROOT.gStyle.SetTitleSize(.036,'XYZ')
    c=ROOT.TCanvas('response_profiles','Conditional common-response check',1450,1100);c.Divide(2,2,.003,.003)
    for col,run in enumerate(['3','12']):
        pad=c.cd(col+1);pad.SetRightMargin(.16);pad.SetLeftMargin(.14);pad.SetBottomMargin(.13)
        # Linear display grid is interpolated only for plotting; all numbers use
        # the original grid and retain the interchange symmetry.
        h=ROOT.TH2D('profile_'+run,f'Run {run} / free effective area;SPE decay pole (ns);Extra RC pole (ns);RMS log discrepancy',80,2.5,162.5,80,0.,160.)
        loss=np.sqrt(arrays[f'run{run}_log_loss'])
        for ix in range(1,81):
            x=h.GetXaxis().GetBinCenter(ix);curve=np.array([np.interp(x,decays,loss[:,j]) for j in range(len(rcs))])
            for iy in range(1,81):h.SetBinContent(ix,iy,float(np.interp(h.GetYaxis().GetBinCenter(iy),rcs,curve)))
        h.SetMinimum(0.);h.SetMaximum(1.5);h.Draw('COLZ');KEEP.append(h)
        label('Descriptive profile; not confidence contours',.18,.88,size=.027)
    pair=report['shared_53p2V_checks']['3_12'];single=pair['one_kernel_one_area']['prediction']
    panel(c,3,'53.2 V / one kernel and one area;Filter width (ns);Excess variance / mean (ADC)',30.,0.,220.,1.1)
    for run,color in [('3',BLUE),('12',RED)]:graph(np.array(WIDTHS)*2.,observations[run],color);label(f'Measured run {run}',.21,.86-(run=='12')*.055,color,.029)
    graph(np.array(WIDTHS)*2.,np.array(single),GREEN,2);label('Best common stationary response',.21,.74,GREEN,.028)
    area_limits=[nominal_area]
    for row in report['runs'].values():
        fit=row['groups']['first60'];area_limits.append(fit['best_grid']['effective_area_ADC_ns'])
        bounds=fit['descriptive_relative_tolerance_regions']['0.1']['effective_ADC_area_ns_min_max']
        if bounds:area_limits.append(bounds[1])
    c.cd(4).SetLogy()
    panel(c,4,'Separate areas / diagnostic relaxation;Bias voltage (V);Conditional effective area (ADC ns)',52.2,10.,54.2,max(area_limits)*7.)
    for run,row in report['runs'].items():
        fit=row['groups']['first60'];region=fit['descriptive_relative_tolerance_regions']['0.1'];value=fit['best_grid']['effective_area_ADC_ns']
        g=ROOT.TGraph(1,np.array([row['bias_V']]),np.array([value]));g.SetMarkerStyle(24 if run in ('3','7') else 20);g.SetMarkerColor(BLUE);g.Draw('P SAME');KEEP.append(g)
        if region['grid_points']:
            lo,hi=region['effective_ADC_area_ns_min_max'];line=ROOT.TLine(row['bias_V'],lo,row['bias_V'],hi);line.SetLineColor(BLUE);line.Draw();KEEP.append(line)
        y_label=value*1.10 if run=='12' else value*.88 if run=='18' else value
        text=ROOT.TLatex(row['bias_V']+.025,y_label,str(run));text.SetTextSize(.028);text.Draw();KEEP.append(text)
    label('Range: all metrics within 10%; grid-limited, not a gain CI',.19,.85,size=.026)
    label(f'Nominal MC avalanche area: {nominal_area:.1f} ADC ns',.19,.79,size=.027)
    label('Run labels; temperature / setup remain confounded',.19,.73,size=.026)
    save(c,args.out,'response_profiles')
    print(json.dumps(report['shared_53p2V_checks'],indent=2),flush=True)


if __name__=='__main__':main()
