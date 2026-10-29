"""Training-only joint PID-covariance / waveform-fluctuation response proposal.

Positive colored-noise powers act on ADC waveforms through the existing native
SignalDependentNoise class. No PID-score perturbations or measured templates.
This is a conditional response test, not a microscopic mechanism determination.
"""
import copy
from pathlib import Path
import numpy as np
from scipy.optimize import nnls
from reference import PROJECT, wave_features
from prepare_pid import read
from prepare_pid_bandwidth import OUT, PARENT
from diagnose_pid_bandwidth import weights, linear_scores
from extract_pid import load
from diagnose_fluctuations import fluctuation_metrics
from fit_signal_fluctuations import colored
from refit_final_parameters import combine_kernels
from publication_stats import save, sha


def conditional_cov(scores,q,edges,support,robust=False):
    bins=np.clip(np.searchsorted(edges,q,'right')-1,0,5);cov=[]
    for b in support:
        x=scores[bins==b]
        if robust:
            med=np.median(x,axis=0);sd=np.diff(np.quantile(x,[.25,.75],axis=0),axis=0)[0]/1.348979500392
            x=np.clip(x,med-3*sd,med+3*sd)
        cov.append(np.cov(x,rowvar=False))
    return np.mean(cov,axis=0)


def average_bins(x,q,edges,support):
    bins=np.clip(np.searchsorted(edges,q,'right')-1,0,5)
    return np.mean([x[bins==b].mean(0) for b in support],axis=0)


def projected_cov(signal,anchor,scores,q,h,pca):
    w,c=weights(pca)
    gate=signal[np.arange(len(q))[:,None],anchor[:,None]+np.arange(-20,1500)]
    k=(w[None,:,:]-(scores+c)[:,:,None])*np.sqrt(np.maximum(gate,0))[:,None,:]/q[:,None,None]
    nfft=4096
    proj=np.fft.irfft(np.fft.rfft(k,n=nfft)*np.fft.rfft(h[::-1],n=nfft),n=nfft)
    return np.einsum('nit,njt->nij',proj,proj)


def main():
    dest=OUT/'covariance_fit';dest.mkdir(exist_ok=False)
    protocol=read(OUT/'protocol.json');edges=np.asarray(protocol['q_edges'])
    pca=read(PROJECT/'analysis/paper_baseline/results/publication_20261028/point.json')['windows']['3000']['pca']
    report={'scope':__doc__,'code_sha256':sha(Path(__file__)),
      'experimental_training':'first60 only; original main-band ratio0.4 diagnostic, no event particle labels.',
      'MC_role':'Reused former validation histories now used as calibration support; cannot call them independent validation for this fit.',
      'targets':'Equal-weight supported Q bins; winsorized at median +/-3 IQR/1.349 for data covariance, native base covariance untrimmed; three short-scale fluctuation means. Positive PSD components, nonnegative least squares.',
      'approximation':'PID perturbation linearized at fixed original ADC anchor, exact FIR covariance; F moments use16 independent noise replicas and fixed base mean. Actual own-anchor native ADC replay must check proposal.',
      'states':{}}
    for run in [3,12,18]:
        state=f'run{run}';entry=protocol['entries'][state];cfg=read(PROJECT/entry['parent_config'])
        d=load(OUT/'arrays'/f'measured_{state}_first60.npz')
        m=load(OUT/'arrays'/f'{state}_components.npz')
        measured_path=(PARENT/'run18' if run==18 else PROJECT/'simulation/results/signal_fluctuations_20261028')/f'{state}_first60.npz'
        original=load(measured_path)
        use=d['main_band'];data_cov_scores=d['scores'][use];data_q=d['q'][use]
        bins=np.clip(np.searchsorted(edges,m['q'],'right')-1,0,5)
        support=[b for b in range(6) if np.sum(bins==b)>=10 and np.sum((data_q>=edges[b])&(data_q<edges[b+1]))>=30]
        target_cov=conditional_cov(data_cov_scores,data_q,edges,support,robust=True)
        target_F=average_bins(original['metrics'][use,:,3],data_q,edges,support)
        base=np.floor(m['base_analog']);f=wave_features(base)
        base_scores,q,_=linear_scores(base,m['anchor'],f['baseline'],pca)
        base_cov=conditional_cov(base_scores,m['q'],edges,support)
        metrics=fluctuation_metrics(base,f['baseline'],m['anchor']-.5)
        base_F=average_bins(metrics[:,:,3],m['q'],edges,support)
        kernels=[np.asarray(cfg['signal_dependent_noise']['coefficients']),np.array([1.])]
        names=['previous_spectrum','white']
        for tau in [20.,60.,150.,400.,1000.,2500.]:
            h=np.exp(-np.arange(1025)*2/tau);h/=np.linalg.norm(h)
            kernels.append(h);names.append(f'exponential_{tau:g}ns')
        cov_components=[];F_components=[]
        rng=np.random.default_rng(291029810+run);white=rng.normal(size=(16,len(q),2496+1024))
        sqrt_signal=np.sqrt(np.maximum(m['signal'],0))
        for h in kernels:
            cov_components.append(average_bins(projected_cov(m['signal'],m['anchor'],base_scores,q,h,pca),m['q'],edges,support))
            values=[]
            for z in white:
                noise=colored(h,z[:,:2496+len(h)-1])*sqrt_signal
                fm=fluctuation_metrics(noise,np.zeros(len(q)),m['anchor']-.5)
                values.append(average_bins((fm[:,:,2]-fm[:,:,1])/metrics[:,:,0],m['q'],edges,support))
            F_components.append(np.mean(values,axis=0))
        C=np.asarray(cov_components);F=np.asarray(F_components)
        i,j=np.triu_indices(3)
        scale_cov=np.sqrt(np.diag(target_cov)[i]*np.diag(target_cov)[j])
        # Matching the complete covariance (not just three widths) prevents
        # unrelated random broadening from being counted as sufficient closure.
        A=np.r_[(C[:,i,j]/scale_cov).T/.25,(F/target_F).T/.10]
        target=np.r_[(target_cov-base_cov)[i,j]/scale_cov/.25,(target_F-base_F)/target_F/.10]
        v,loss=nnls(A,target)
        h,strength=combine_kernels(kernels,v)
        combined_cov=average_bins(projected_cov(m['signal'],m['anchor'],base_scores,q,h,pca),m['q'],edges,support)*strength
        actual_F=[]
        for z in white:
            noise=colored(h,z)*sqrt_signal*np.sqrt(strength)
            fm=fluctuation_metrics(noise,np.zeros(len(q)),m['anchor']-.5)
            actual_F.append(average_bins((fm[:,:,2]-fm[:,:,1])/metrics[:,:,0],m['q'],edges,support))
        final_F=base_F+np.mean(actual_F,axis=0)
        final_cov=base_cov+combined_cov
        candidate=copy.deepcopy(cfg)
        candidate['signal_dependent_noise'].update(coefficients=h.tolist(),variance_per_signal_adc=strength,
           provenance=str((dest/'fit.json').relative_to(PROJECT)),
           spectral_components='Training-only positive PSD combination; joint PID covariance and short-scale fluctuation constraints')
        candidate['status']='PID bandwidth response proposal; unvalidated, acquisition-specific empirical noise'
        path=dest/f'{state}_config.json';save(path,candidate)
        report['states'][state]={'support_bins':support,'training_path':str(measured_path.relative_to(PROJECT)),
          'training_sha256':sha(measured_path),'components':names,'component_variances':v.tolist(),
          'residual_norm':float(loss),'target_cov':target_cov.tolist(),'base_cov':base_cov.tolist(),
          'predicted_cov':final_cov.tolist(),'predicted_SD_ratio':np.sqrt(np.diag(final_cov)/np.diag(target_cov)).tolist(),
          'target_F':target_F.tolist(),'base_F':base_F.tolist(),'predicted_F':final_F.tolist(),
          'predicted_F_ratio':(final_F/target_F).tolist(),'variance_per_signal_adc':strength,
          'config':str(path.relative_to(PROJECT)),'config_sha256':sha(path),
          'component_covariances':C.tolist(),'component_F':F.tolist()}
        np.savez_compressed(dest/f'{state}_kernels.npz',**{n:h for n,h in zip(names,kernels)})
        print(state,'powers',dict(zip(names,np.round(v,5))),'SD_ratio',np.sqrt(np.diag(final_cov)/np.diag(target_cov)),
            'F_ratio',final_F/target_F,flush=True)
    save(dest/'fit.json',report)


if __name__=='__main__':main()
