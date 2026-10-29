"""Conditional stationary Poisson-shot-noise check of the assumed signal kernel.

For a unit-area impulse h and avalanche ADC area A, Var(filtered signal)/mean
signal = A * integral |H(f) F(f)|^2 df. This is an analytic diagnostic only,
not a replacement for Geant4, a likelihood fit or a measured SiPM time constant.
"""
import argparse,json
from pathlib import Path
import numpy as np
import ROOT
from reference import PROJECT
from diagnose_charge import sha
from compare import BLUE,RED,GREEN,KEEP,panel,graph,label,save

HERE=Path(__file__).resolve().parent
WIDTHS=(25,51,101)


def box_transfer(frequency, length):
    x=2*frequency;den=np.sin(np.pi*x)
    return np.divide(np.sin(length*np.pi*x),length*den,out=np.ones_like(x),where=abs(den)>1e-10)


def variance_integrals(rc, rise=2., decay=29., bins=16000):
    f=np.geomspace(1e-6,5.,bins)  # 1/ns; includes aliases of the 2ns sample filter
    finite_sample=(1000/999)*(1-box_transfer(f,1000)**2)
    weights=np.array([2*(1-box_transfer(f,n))**2*finite_sample*np.sinc(2*f)**2 for n in WIDTHS])
    base=1/(1+(2*np.pi*f*rise)**2)/(1+(2*np.pi*f*decay)**2)
    return np.array([np.trapezoid(weights*base/(1+(2*np.pi*f*r)**2),f,axis=1) for r in np.atleast_1d(rc)])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    old_path=HERE/'results/fluctuations_20261028/diagnostics.json';control_path=HERE/'results/response_factorization_20261028/fluctuations/diagnostics.json'
    old=json.loads(old_path.read_text());control=json.loads(control_path.read_text())
    cfg_path=HERE/'results/joint_optical_20261028/co60/config.json';cfg=json.loads(cfg_path.read_text());r=cfg['sipm']['readout'];s=cfg['sipm']['sensor']
    lsb=(r['adc_max_mV']-r['adc_min_mV'])/2**r['adc_bits'];area=1.602176634e-19*s['gain_electrons']*r['transimpedance_ohm']*1e12/lsb
    rc=np.r_[0.,np.geomspace(.1,300.,400)];curves=area*variance_integrals(rc)
    anchors=area*variance_integrals([0.,r['lowpass_ns']]);report={'scope':__doc__,
        'assumptions':{'SPE_rise_ns':2.,'SPE_decay_ns':29.,'avalanche_area_adc_ns':area,'var_scale':'Constant ADC area per independent avalanche, locally stationary rate, no correlated avalanches. 2ns aperture and discrete centered box filters; finite 1000-sample variance correction.'},
        'limits':'Pulse rate changes over the finite window, current recovery, finite MC, signal/noise cross terms and deterministic curvature are not fully represented. Metric-specific RC crossings are conditional diagnostic values, not measured circuit constants. Do not export this as a calibrated detector configuration.',
        'inputs_sha256':{str(p.relative_to(PROJECT)):sha(p) for p in (old_path,control_path,cfg_path)},
        'code_sha256':{str(Path(__file__).relative_to(PROJECT)):sha(Path(__file__))},'sources':{}}
    ROOT.gROOT.SetBatch(True);ROOT.gStyle.SetOptStat(0);ROOT.gStyle.SetTitleFontSize(.043)
    ROOT.gStyle.SetLabelSize(.034,'XYZ');ROOT.gStyle.SetTitleSize(.04,'XYZ')
    canvas=ROOT.TCanvas('signal_constraint','Multiscale signal-response consistency',1350,650);canvas.Divide(2,1,.003,.003)
    for col,source in enumerate(('Co60','Am241')):
        observed=np.array([old['sources'][source]['groups']['measured']['metrics'][str(w*2)+'ns']['excess_variance_over_mean']['p10_p50_p90'][1] for w in WIDTHS])
        crossings=[float(np.interp(v,curves[::-1,i],rc[::-1])) for i,v in enumerate(observed)]
        losses=np.mean(np.log(curves/observed)**2,axis=1);best=int(losses.argmin())
        row={'observed_medians':observed.tolist(),'metric_specific_RC_crossings_ns':crossings,'best_shared_RC_for_three_metrics_ns':float(rc[best]),
            'best_predictions':curves[best].tolist(),'best_fractional_residuals':(curves[best]/observed-1).tolist(),'frozen_MC_proxy_checks':{}}
        for j,(text,d) in enumerate([('zero_RC_finite_emission_rise',control),('previous_RC94',old)]):
            actual=np.array([d['sources'][source]['groups']['mc']['metrics'][str(w*2)+'ns']['excess_variance_over_mean']['p10_p50_p90'][1] for w in WIDTHS])
            row['frozen_MC_proxy_checks'][text]={'analytic':anchors[j].tolist(),'full_MC_median':actual.tolist(),'relative_difference':(anchors[j]/actual-1).tolist()}
        report['sources'][source]=row
        canvas.cd(col+1).SetLogy();panel(canvas,col+1,source+' / fixed SPE and gain assumptions;Extra readout RC (ns);Excess variance / mean signal (ADC)',0,.004,160,8.)
        for i,(color,w) in enumerate(zip((BLUE,RED,GREEN),WIDTHS)):
            graph(rc,curves[:,i],color);graph(np.array([0.,160.]),np.array([observed[i]]*2),color,2)
            g=ROOT.TGraph(1,np.array([crossings[i]]),np.array([observed[i]]));g.SetMarkerStyle(20);g.SetMarkerColor(color);g.Draw('P SAME');KEEP.append(g)
            label(f'{2*w} ns filter: crossing ~ {crossings[i]:.0f} ns',.40,.84-i*.055,color,.030)
        label('Solid: stationary proxy; dashed: measured median',.22,.64,size=.027)
        label('Crossings are not measured RC constants',.27,.25,size=.028)
    save(canvas,args.out,'signal_constraints')
    np.savez_compressed(args.out/'curves.npz',rc_ns=rc,predictions_adc=curves)
    (args.out/'constraints.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['sources'],indent=2))


if __name__=='__main__':main()
