"""Joint training-only photon-statistics profile with nonnegative waveform noise."""
import argparse,copy
from pathlib import Path
import numpy as np
from scipy.optimize import nnls
from reference import PROJECT,wave_features,TIME_NS
from prepare_npe_profile import OUT,PARENT,WAVE,SCALES
from prepare_pid import read
from extract_pid import load
from fit_pid_bandwidth import conditional_cov,average_bins,projected_cov
from fit_signal_fluctuations import colored
from diagnose_fluctuations import fluctuation_metrics
from refit_final_parameters import combine_kernels
from publication_stats import sha,save


def fit_one(entry,protocol,pca):
    key=entry['key'];run=entry['run'];dest=OUT/'fit';cfg=read(PROJECT/entry['config'])
    m=load(OUT/'profile/arrays'/f'{key}.npz')
    path=(WAVE/'run18' if run==18 else PROJECT/'simulation/results/signal_fluctuations_20261028')/f'run{run}_first60.npz'
    data=load(path);d=load(PARENT/'arrays'/f'measured_run{run}_first60.npz');use=d['main_band']
    edges=np.asarray(protocol['q_edges']);support=read(PARENT/'covariance_fit/fit.json')['states'][f'run{run}']['support_bins']
    bins=np.clip(np.searchsorted(edges,m['q'],'right')-1,0,5)
    assert all(np.sum(bins==b)>=10 for b in support)
    target_cov=conditional_cov(d['scores'][use],d['q'][use],edges,support,robust=True)
    target_F=average_bins(data['metrics'][use,:,3],d['q'][use],edges,support)
    base_cov=conditional_cov(m['scores'],m['q'],edges,support)
    base_F=average_bins(m['metrics'],m['q'],edges,support)
    mf=wave_features(m['waves']);metrics=fluctuation_metrics(m['waves'],mf['baseline'],m['anchor']-.5)
    np.testing.assert_allclose(metrics[:,:,3],m['metrics'],atol=1e-12,rtol=1e-12)
    kernels=[np.asarray(cfg['signal_dependent_noise']['coefficients']),np.array([1.])]
    names=['inherited_spectrum','white']
    for tau in [20.,60.,150.,400.,1000.,2500.]:
        h=np.exp(-np.arange(1025)*2/tau);h/=np.linalg.norm(h);kernels.append(h);names.append(f'exp_{tau:g}ns')
    rng=np.random.default_rng(2910291100+run)
    white=rng.normal(size=(16,len(m['q']),2496+1024));sqrt_signal=np.sqrt(np.maximum(m['signal'],0))
    def contributions(h):
        c=average_bins(projected_cov(m['signal'],m['anchor'],m['scores'],m['q'],h,pca),m['q'],edges,support)
        fs=[]
        for z in white:
            noise=colored(h,z[:,:2496+len(h)-1])*sqrt_signal
            fm=fluctuation_metrics(noise,np.zeros(len(m['q'])),m['anchor']-.5)
            fs.append(average_bins((fm[:,:,2]-fm[:,:,1])/metrics[:,:,0],m['q'],edges,support))
        return c,np.mean(fs,axis=0)
    C,F=map(np.asarray,zip(*(contributions(h) for h in kernels)))
    i,j=np.triu_indices(3);normal=np.sqrt(np.diag(target_cov)[i]*np.diag(target_cov)[j])
    def vector(c,f):return np.r_[c[i,j]/normal/.25,f/target_F/.10]
    A=np.column_stack([vector(c,f) for c,f in zip(C,F)])
    target=vector(target_cov-base_cov,target_F-base_F)
    variances,linear_loss=nnls(A,target)
    h,combined_strength=combine_kernels(kernels,variances)
    c,f=contributions(h);v=vector(c,f)
    strength=max(0.,float(v@target/(v@v)))
    predicted_cov=base_cov+strength*c;predicted_F=base_F+strength*f
    loss=float(np.sum((vector(predicted_cov-target_cov,predicted_F-target_F))**2))
    candidate=copy.deepcopy(cfg)
    candidate['signal_dependent_noise'].update(coefficients=h.tolist(),variance_per_signal_adc=strength,
         provenance=str((dest/f'{key}.json').relative_to(PROJECT)),
         spectral_components='Positive joint PID covariance and F fit, profiled effective photon statistics')
    candidate['status']='NUMERICAL PDE PROFILE WITH FITTED NOISE; requires nominal-PDE full-Geant4 verification before physical interpretation'
    cfgpath=dest/f'{key}_config.json';save(cfgpath,candidate)
    shape=average_bins(m['shapes'],m['q'],edges,support)
    target_shape=average_bins(data['shapes'][use],d['q'][use],edges,support)
    region=(TIME_NS>=-40)&(TIME_NS<=3000)
    report={'scope':__doc__,'key':key,'run':run,'scale':entry['scale'],'source':entry['source'],
      'code_sha256':sha(Path(__file__)),'protocol_sha256':sha(OUT/'protocol.json'),
      'training_path':str(path.relative_to(PROJECT)),'training_sha256':sha(path),'support_bins':support,
      'n_MC':len(m['q']),'common_Q_counts':np.bincount(bins,minlength=6).tolist(),
      'target_cov':target_cov.tolist(),'base_cov':base_cov.tolist(),'predicted_cov':predicted_cov.tolist(),
      'target_F':target_F.tolist(),'base_F':base_F.tolist(),'predicted_F':predicted_F.tolist(),
      'predicted_SD_ratio':np.sqrt(np.diag(predicted_cov)/np.diag(target_cov)).tolist(),
      'predicted_F_ratio':(predicted_F/target_F).tolist(),'score':loss,
      'linear_loss_squared':float(linear_loss**2),'components':names,'component_variances':variances.tolist(),
      'component_covariances':C.tolist(),'component_F':F.tolist(),
      'combined_strength_initial':combined_strength,'variance_per_signal_adc':strength,
      'noise_mean_shape_RMSE_not_included':float(np.sqrt(np.mean((shape[region]-target_shape[region])**2))/target_shape.max()),
      'config':str(cfgpath.relative_to(PROJECT)),'config_sha256':sha(cfgpath),
      'limits':'Conditional training loss, not chi-square or a parameter confidence interval. PID linearized at fixed native anchor; combined FIR strength refitted to training moments. Actual ADC and new full transport must be checked.'}
    save(dest/f'{key}.json',report)
    print(key,'loss',round(loss,4),'SD',np.round(report['predicted_SD_ratio'],3),'F',np.round(report['predicted_F_ratio'],3),flush=True)


def select():
    rows=[]
    for scale in SCALES:
        records=[read(OUT/'fit'/f's{scale:g}_run{run}_{source}.json') for source,run in [('Co60',3),('Am241',12)]]
        rows.append({'scale':scale,'joint_score':sum(r['score'] for r in records),
          'Co_score':records[0]['score'],'Am_score':records[1]['score'],
          'Co_SD_ratio':records[0]['predicted_SD_ratio'],'Am_SD_ratio':records[1]['predicted_SD_ratio']})
    winner=min(rows,key=lambda r:r['joint_score'])
    save(OUT/'selection.json',{'scope':'Choose shared effective-photon scale using only experimental first60 joint covariance/F profiles.',
       'code_sha256':sha(Path(__file__)),'grid':rows,'chosen_scale':winner['scale'],
       'criterion':'Minimum sum of actual-combined-FIR training residual scores, equal run3/run12 weight; no later-segment selection.',
       'parameter_is_not_measured':'Yield/PDE/coupling remain degenerate; the chosen grid point is a conditional model candidate, not a measured SiPM PDE or intrinsic CsI yield.',
       'fresh_transport_required':True})
    print('Selected scale',winner['scale'],'joint score',winner['joint_score'],flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--keys',nargs='*');p.add_argument('--select',action='store_true');args=p.parse_args()
    (OUT/'fit').mkdir(exist_ok=True);protocol=read(OUT/'protocol.json')
    pca=read(PROJECT/'analysis/paper_baseline/results/publication_20261028/point.json')['windows']['3000']['pca']
    for key in args.keys or []:
        assert not (OUT/'fit'/f'{key}.json').exists(),'No overwrite'
        fit_one(protocol['entries'][key],protocol,pca)
    if args.select:
        assert not (OUT/'selection.json').exists(),'Selection already frozen'
        select()


if __name__=='__main__':main()
