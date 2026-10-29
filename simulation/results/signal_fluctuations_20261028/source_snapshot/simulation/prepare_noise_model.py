"""Freeze Gaussian FIR electronic-noise candidates using only pretrigger training.

This estimates an effective noise covariance, not the single-avalanche signal
response. Photon yield, gain, geometry, emission times and signal filtering stay
at the previous joint candidate. No recorded trace is copied into simulated ADC.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from characterize_readout import covariance, periodogram
from diagnose_charge import sha
from reference import PROJECT

HERE = Path(__file__).resolve().parent


def make_kernel(waves, length=257):
    frequency, power = periodogram(waves); psd = power.mean(axis=0)
    analog_psd = np.maximum(psd-(1/12)/250.,1e-9)
    analog_psd[0] = analog_psd[1]  # declared flat extrapolation below 1.953125 MHz
    freq = np.fft.rfftfreq(4096,.002)
    amplitude = np.sqrt(np.interp(freq,frequency,analog_psd))
    impulse = np.fft.fftshift(np.fft.irfft(amplitude,4096))
    half=length//2; kernel=impulse[2048-half:2048+half+1]*np.hanning(length)
    kernel /= np.linalg.norm(kernel)
    ac=np.correlate(kernel,kernel,'full')[length-1:]
    lag=np.abs(np.arange(256)[:,None]-np.arange(256)[None,:]); matrix=ac[lag]
    x=np.arange(256,dtype=float);x-=x.mean()
    projected_variance=(np.trace(matrix)-matrix.sum()/256-x@matrix@x/(x@x))/256
    target=float(covariance(waves)[:,0].mean())
    sigma=float(np.sqrt((target-(1/12)*(254/256))/projected_variance))
    return kernel,sigma,frequency,psd,{'target_detrended_variance_adc2':target,'unit_FIR_projected_variance':float(projected_variance),'unprojected_sigma_adc':sigma}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--noise',type=Path,default=HERE/'results/readout_noise_20261028')
    p.add_argument('--base',type=Path,default=HERE/'results/joint_optical_20261028')
    p.add_argument('--out',type=Path,required=True);args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    with np.load(args.noise/'samples.npz') as z: arrays=dict(z)
    report={'scope':__doc__,'comparison_scope':'Pretrigger-calibrated Gaussian FIR noise update; all optical, avalanche and signal-response parameters inherited without refitting.',
        'method':'Hann PSD of linearly detrended first256 samples, basic QC + t10>=384 + primary pretrigger QC, first60% only. Subtract approximate ADC quantization PSD. Flat extrapolation below first positive frequency bin; zero-phase square-root spectrum,257-tap Hann truncation,unit squared norm. RMS chosen to reproduce the training detrended mean variance including1/12 ADC^2 quantization.',
        'limits':'Finite-window stationary Gaussian approximation; does not reproduce transient weak pulses, non-Gaussian tails, run12 late noise changes, or infer the single-PE signal response.',
        'source_code_sha256':{str(Path(__file__).relative_to(PROJECT)):sha(Path(__file__)),str((HERE/'characterize_readout.py').relative_to(PROJECT)):sha(HERE/'characterize_readout.py')},
        'input_sha256':{str((args.noise/'samples.npz').resolve().relative_to(PROJECT)):sha(args.noise/'samples.npz')},'sources':{}}
    saved={}
    for source,name,run,events,seed in [('Co60','co60',3,2000,283103),('Am241','am241',12,12000,283112)]:
        folder=args.out/name;folder.mkdir();key=f'run{run}_train';ids=arrays[key+'_ids']
        phase=PROJECT/f'analysis/paper_baseline/results/phase2/run{run:04d}_features.npz'
        with np.load(phase) as z:primary=z['eligible_primary'][ids]
        waves=arrays[key+'_pretrigger_adc'][primary]
        kernel,sigma,frequency,psd,diagnostic=make_kernel(waves)
        cfg=json.loads((args.base/name/'config.json').read_text());readout=cfg['sipm']['readout']
        lsb=(readout['adc_max_mV']-readout['adc_min_mV'])/2**readout['adc_bits']
        cfg['electronics_noise']={'model':'gaussian_fir','coefficients':kernel.tolist(),
            'sample_period_ns':2.,'provenance':f'Run{run} first60% pretrigger-only effective covariance; not a signal impulse response',
            'training_source_sha256':sha(phase)}
        readout['noise_rms_mV']=sigma*lsb
        cfg['evidence']['noise_model_update']={'scope':report['comparison_scope'],'limits':report['limits']}
        path=folder/'config.json';path.write_text(json.dumps(cfg,indent=2)+'\n')
        (folder/'run.mac').write_text(f'/csi/config {path.resolve()}\n/csi/source {source}\n/csi/seed {seed}\n/csi/output {(folder/"events.root").resolve()}\n/run/initialize\n/run/beamOn {events}\n')
        report['sources'][source]={'run':run,'n_training_noise_traces':len(waves),'seed':seed,'events_requested':events,'config_sha256':sha(path),
            'base_config_sha256':sha(args.base/name/'config.json'),'changes_to_numeric_config':['electronics_noise','sipm.readout.noise_rms_mV'],**diagnostic}
        report['input_sha256'][str(phase.relative_to(PROJECT))]=sha(phase)
        saved[key+'_frequency_mhz']=frequency;saved[key+'_psd']=psd;saved[key+'_kernel']=kernel;saved[key+'_ids']=ids[primary]
    np.savez_compressed(args.out/'noise_calibration.npz',**saved)
    (args.out/'design.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report['sources'],indent=2))


if __name__=='__main__':main()
