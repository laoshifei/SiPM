"""Freeze estimator controls and acquisition-state diagnostics on existing ADC."""
import json
import subprocess
from pathlib import Path
import numpy as np
from scipy.ndimage import uniform_filter1d
from reference import PROJECT, wave_features, bin_layout
from publication_stats import save, sha

OUT=PROJECT/'simulation/results/acquisition_state_20261030'
PARENT=PROJECT/'simulation/results/conditional_response_20261030'
DATA=PROJECT/'simulation/results/signal_fluctuations_20261028'
WIDTH=PROJECT/'simulation/results/pid_bandwidth_20261029'
SEGMENTS=['first60','middle20','last20']
SOURCES=[('Co60',3),('Am241',12)]
CONTROLS=['original','recent_baseline','smooth_anchor','recent_and_smooth']
MC_VARIANTS=['identity','no_signal_residual']
LAGS=np.array([0,1,2,4,8,16,32,64])


def read(path):return json.loads(Path(path).read_text())
def load(path):
    with np.load(path) as z:return dict(z)
def relative(path):return str(path.relative_to(PROJECT))
def finite(x):
    if isinstance(x,dict):return {k:finite(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [finite(v) for v in x]
    if isinstance(x,(float,np.floating)):return float(x) if np.isfinite(x) else None
    if isinstance(x,np.integer):return int(x)
    return x


def score_gate(waves,anchor,baseline,pca):
    assert np.all(anchor>=20) and np.all(anchor+1500<=waves.shape[1])
    gate=waves[np.arange(len(waves))[:,None],anchor[:,None]+np.arange(-20,1500)].astype(float)-baseline[:,None]
    q=gate.sum(1);assert np.all(q>0)
    w=np.zeros((1520,3));w[:270,0]=1;w[:,1]=-np.arange(-20,1500)*.002
    w[:,2]=np.repeat(np.asarray(pca['vector'])/pca['score_scale'],20)
    offset=np.array([0,0,np.dot(pca['center'],pca['vector'])/pca['score_scale']])
    return gate@w/q[:,None]-offset,q,gate


def extract_features(waves,pca):
    f=wave_features(waves);anchor=np.floor(f['t10_sample']).astype(int)+1
    assert np.all(anchor>=384), 'The first256 samples must precede the pulse.'
    smooth=uniform_filter1d(waves.astype(float),size=11,axis=1,mode='nearest')
    sf=wave_features(smooth);sa=np.floor(sf['t10_sample']).astype(int)+1
    recent=waves[np.arange(len(waves))[:,None],anchor[:,None]+np.arange(-192,-64)].mean(1)
    scores=[];charges=[];gate=None
    for name in CONTROLS:
        a=sa if name in ['smooth_anchor','recent_and_smooth'] else anchor
        b=recent if name in ['recent_baseline','recent_and_smooth'] else f['baseline']
        score,q,g=score_gate(waves,a,b,pca);scores.append(score);charges.append(q)
        if name=='original':gate=g
    shape=gate.reshape(len(waves),76,20).sum(2)/charges[0][:,None]
    np.testing.assert_allclose(shape.sum(1),1.,atol=1e-12)
    pre=waves[:,:256].astype(float);pre-=pre.mean(1)[:,None]
    t=np.arange(256)-127.5;pre-=((pre@t)/(t@t))[:,None]*t
    freq=np.fft.rfftfreq(256,d=2.)*1000;power=abs(np.fft.rfft(pre*np.hanning(256),axis=1))**2
    nonzero=freq>0;low=(freq>0)&(freq<=20)
    low_fraction=power[:,low].sum(1)/power[:,nonzero].sum(1)
    state=np.c_[f['baseline'],np.log(f['baseline_rms']),low_fraction]
    pre_ac=np.array([np.mean(pre[:,:256-l]*pre[:,l:],axis=1)/np.mean(pre*pre,axis=1) for l in [1,8,32]]).T
    hp=[]
    for wave,baseline,t10 in zip(waves,f['baseline'],f['t10_sample']):
        y=wave.astype(float)-baseline;cum=np.r_[0.,np.cumsum(y)]
        signal=y[12:-12];residual=signal-(cum[25:]-cum[:-25])/25
        time=(np.arange(12,len(y)-12)-t10)*2
        prior=(time>=-600)&(time< -200);post=(time>=500)&(time<2500)
        assert prior.sum()==200 and post.sum()==1000
        mu=signal[post].mean();assert mu>0
        rpre=residual[prior];rpost=residual[post]
        rpre=rpre-rpre.mean();rpost=rpost-rpost.mean()
        cov=lambda z,l:np.dot(z[:len(z)-l],z[l:])/(len(z)-l-1)
        hp.append([(cov(rpost,int(l))-cov(rpre,int(l)))/mu for l in LAGS])
    return {'scores':np.stack(scores,axis=1),'control_q':np.stack(charges,axis=1),'shape':shape,
        'state':state,'pre_ac':pre_ac,'hp_ac':np.array(hp),'anchor':anchor,'smooth_anchor':sa,
        't10':f['t10_sample'],'smooth_t10':sf['t10_sample'],'baseline':f['baseline'],
        'baseline_rms':f['baseline_rms'],'recent_baseline':recent,'q_recomputed':charges[0]}


def main():
    OUT.mkdir(exist_ok=False)
    for name in ['arrays','figures','logs','source_snapshot']:(OUT/name).mkdir()
    parent=read(PARENT/'protocol.json');pca_path=PROJECT/'analysis/paper_baseline/results/publication_20261028/point.json'
    assert sha(PARENT/'record.json')=='0e903fe73eca0bd9600a50fa72da811efe167d53e1d788b2c6d09948453606e6'
    inputs=[PARENT/'record.json',PARENT/'protocol.json',PARENT/'conditional.json',pca_path]
    for source,run in SOURCES:
        for seg in SEGMENTS:
            inputs.extend([DATA/f'run{run}_{seg}.npz',WIDTH/'arrays'/f'measured_run{run}_{seg}.npz',PARENT/'arrays'/f'data_run{run}_{seg}.npz'])
        for v in MC_VARIANTS:inputs.append(PARENT/'arrays'/f'run{run}_{source}__{v}.npz')
        inputs.append(PROJECT/f'data/DT5730_DPP_PHA/DAQ/run_{run}/UNFILTERED/Data_run_{run}.BIN')
    protocol={'scope':__doc__,'checkpoint_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT).decode().strip(),
        'code_sha256':sha(Path(__file__)),'parent_record_sha256':sha(PARENT/'record.json'),
        'inputs_sha256':{relative(f):sha(f) for f in inputs},'q_edges':parent['q_edges'],
        'support_bins':parent['support_bins'],'conditional_cuts':parent['conditional_cuts'],
        'controls':CONTROLS,'MC_variants':MC_VARIANTS,'highpass_lags_ns':(2*LAGS).tolist(),
        'state_features':['baseline_first128_ADC','log_RMS_first128','first256_detrended_Hann_power_fraction_0_to_20MHz'],
        'selection':'Existing experimental main-band IDs and original Q; native original parent IDs/Q. No reselection after estimator changes. Source is not particle truth.',
        'estimator_controls':'Original first128 baseline / raw t10; recent baseline in original anchor[-192,-64); t10 from centered11-sample box smoothing with nearest boundary. Smooth waveform ONLY for locating the anchor; all PID gates integrate original ADC. The recent baseline window is frozen to the original anchor in both recent variants.',
        'state':'First128 baseline and log RMS; first256 linear detrend + Hann FFT low-power fraction. No quantization-noise subtraction. Event index is acquisition order; timestamp ticks audited without assuming time units.',
        'regression':'Q-bin intercepts and within-bin linear Q, optionally three pretrigger state features, raw event fraction, or both. Weighted ordinary least squares; first60 only; six contiguous10%-of-run folds for internal crossvalidation. No outcome-driven model selection. Full-first60 coefficients applied to later segments only as exploratory checks; report marginal1%-99% and Mahalanobis99% state support. Time extrapolates after60%.',
        'covariance':'Exact within-original-Q decomposition into between-Q-half, between acquisition deciles within Q-half, and remaining within-cell covariance. Between-decile means include sampling noise, not causal drift variance. Reference null permutes whole raw64/128 blocks within each segment, preserving block order internally and last partial-block length.',
        'shape':'76 original-anchor40ns gate charge fractions, sum1; pulse covariance is descriptive, not intrinsic scintillation covariance. Original 50ns box-highpass autocovariance subtracts local pretrigger covariance, divides by mean signal; lag0 closes to frozen F50.',
        'uncertainty':'512 paired whole-original-mother and original-index experimental block64/128 draws for estimator comparisons; first60 Q2 targets/PCA fixed. 512 whole-block permutations per segment for temporal covariance reference. Regression crossvalidation reports held-out losses and fold variability, not a claim of independent experimental validation.',
        'new_independent_mothers':0,'new_native_replay_entries':0,'default_configuration_changed':False}
    save(OUT/'protocol.json',protocol);pca=read(pca_path)['windows']['3000']['pca'];extraction={}
    for source,run in SOURCES:
        raw_path=PROJECT/f'data/DT5730_DPP_PHA/DAQ/run_{run}/UNFILTERED/Data_run_{run}.BIN'
        _,dtype,n=bin_layout(raw_path);raw=np.memmap(raw_path,dtype=dtype,offset=2,mode='r',shape=(n,))
        stamps=np.asarray(raw['timestamp']);time_audit={'backsteps':int(np.sum(stamps[1:]<stamps[:-1])),
            'duplicates':int(np.sum(stamps[1:]==stamps[:-1])),'first_last_ticks':[int(stamps[0]),int(stamps[-1])],'units':'Unconverted stored ticks'}
        for seg in SEGMENTS:
            key=f'run{run}_{seg}';original=load(DATA/f'{key}.npz');d=load(PARENT/'arrays'/f'data_{key}.npz')
            use=load(WIDTH/'arrays'/f'measured_{key}.npz')['main_band'];ids=original['ids'][use];waves=original['waves'][use]
            np.testing.assert_array_equal(ids,d['ids']);np.testing.assert_array_equal(waves,raw['data'][ids]);assert int(d['n_raw'])==n
            a=extract_features(waves,pca);np.testing.assert_allclose(a['scores'][:,0],d['scores'],rtol=1e-10,atol=1e-10)
            np.testing.assert_allclose(a['q_recomputed'],d['q'],rtol=1e-12,atol=1e-7)
            np.testing.assert_allclose(a['hp_ac'][:,0],original['metrics'][use,0,3],rtol=1e-10,atol=1e-10)
            np.savez_compressed(OUT/'arrays'/f'{key}.npz',**a,ids=ids,q=d['q'],span=d['span'],n_raw=n,
                F=original['metrics'][use,:,3],timestamp=stamps[ids])
            extraction[key]={'n':len(ids),'timestamp':time_audit,'source':source,'run':run,'original_scores_and_F50_close':True,
                'anchor_shift_ns_p01_p50_p99':np.quantile(2*(a['smooth_anchor']-a['anchor']),[.01,.5,.99]).tolist(),
                'maximum_shape_sum_error':float(abs(a['shape'].sum(1)-1).max())}
            print(key,'prepared',len(ids),flush=True)
        for variant in MC_VARIANTS:
            key=f'run{run}_{source}__{variant}';old=load(PARENT/'arrays'/f'{key}.npz');use=old['fixed']
            a=extract_features(old['waves'][use],pca)
            np.testing.assert_allclose(a['scores'][:,0],old['scores'][use],rtol=1e-10,atol=1e-10)
            np.testing.assert_allclose(a['hp_ac'][:,0],old['metrics'][use,0],rtol=1e-10,atol=1e-10)
            np.savez_compressed(OUT/'arrays'/f'{key}.npz',**a,ids=old['ids'][use],q=old['original_q'][use],n_parent=old['n_parent'],F=old['metrics'][use])
            extraction[key]={'n':int(use.sum()),'n_parent':int(old['n_parent']),'source':source,'run':run,
                'original_scores_and_F50_close':True,'maximum_shape_sum_error':float(abs(a['shape'].sum(1)-1).max())}
            print(key,'prepared',int(use.sum()),flush=True)
    save(OUT/'extraction.json',extraction)


if __name__=='__main__':main()
