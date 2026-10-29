"""Fixed-event PID width, exact covariance and pretrigger-baseline diagnostics."""
import json
from pathlib import Path
import numpy as np
import ROOT
from reference import PROJECT, wave_features
from publication_stats import sha, save
from prepare_pid_bandwidth import OUT, PARENT, PID
from prepare_pid import read
from extract_pid import load, score_waves

METHODS=('qdc','negative_mean_time','pca_projection')
PARTS=('native_signal','baseline_electronics','signal_residual','quantization')


def weights(pca):
    w=np.zeros((3,1520));w[0,:270]=1
    w[1]=-np.arange(-20,1500)*.002
    w[2]=np.repeat(np.asarray(pca['vector'])/pca['score_scale'],20)
    c=np.array([0,0,np.dot(pca['center'],pca['vector'])/pca['score_scale']])
    return w,c


def linear_scores(waves,anchor,baseline,pca):
    w,c=weights(pca)
    gate=np.asarray(waves,float)[np.arange(len(waves))[:,None],anchor[:,None]+np.arange(-20,1500)]-baseline[:,None]
    q=gate.sum(1)
    return gate@w.T/q[:,None]-c,q,gate


def root_rows(path,ids):
    f=ROOT.TFile.Open(str(path));t=f.Get('Events')
    info=json.loads(str(f.Get('RunInfo').GetString().Data()))
    t.SetBranchStatus('*',0)
    keys=['adc','signal_mV','analog_mV']
    for k in keys:t.SetBranchStatus(k,1)
    rows={k:[] for k in keys}
    for i in ids:
        assert t.GetEntry(int(i))>0
        for k in keys:rows[k].append(np.array(getattr(t,k)))
    f.Close()
    return {k:np.asarray(v) for k,v in rows.items()},info


def stats(x):
    return {'n':len(x),'median':np.median(x,axis=0).tolist(),
            'IQR':np.diff(np.quantile(x,[.25,.75],axis=0),axis=0)[0].tolist(),
            'sd':np.std(x,axis=0,ddof=1).tolist(),'cov':np.cov(x,rowvar=False).tolist()}


def cells(q,variants,edges,mask=None):
    bins=np.clip(np.searchsorted(edges,q,'right')-1,0,5)
    mask=np.ones(len(q),bool) if mask is None else mask
    rows=[]
    for b in range(6):
        use=mask&(bins==b);row={'bin':b,'n':int(use.sum())}
        if use.sum()>=5:row['variants']={k:stats(v[use]) for k,v in variants.items()}
        rows.append(row)
    return rows


def recent(waves,anchor):
    ix=anchor[:,None]+np.arange(-192,-64)
    return np.asarray(waves,float)[np.arange(len(waves))[:,None],ix].mean(1)


def main():
    protocol=read(OUT/'protocol.json');edges=np.asarray(protocol['q_edges'])
    pca=read(PROJECT/'analysis/paper_baseline/results/publication_20261028/point.json')['windows']['3000']['pca']
    report={'scope':__doc__,'code_sha256':sha(Path(__file__)),'protocol_sha256':sha(OUT/'protocol.json'),
      'methods':METHODS,'components':PARTS,'MC':{},'measured':{},
      'identity':'U_c=(L_c-(r0+offset)*Q_c)/Q_total; sum U_c=score-r0. Covariance sum includes all off-diagonal terms. Components use the observed full-waveform anchor and identical event selection.',
      'causal_limit':'Signal component includes finite photoelectron/avalanche statistics, geometry and noise-dependent alignment; covariance attribution is algebraic, not a unique physical cause.',
      'selection':'Original fixed event IDs and original Q bins; no re-selection after removal or baseline change. Main-band ratio0.4 diagnostic is not a particle-truth label.',
      'baseline_sensitivity':'Recent mean in [anchor-192,anchor-64); original first128 mean; fixed full-waveform anchor. Alternative is diagnostic only.'}
    (OUT/'arrays').mkdir(exist_ok=True)
    for state,entry in protocol['entries'].items():
        run=entry['run'];source=entry['source']
        old=load(PID/'arrays'/f'{state}_{source}.npz');ids=np.flatnonzero(old['common'])
        full,info=root_rows(PROJECT/entry['parent_root'],ids)
        base,bi=root_rows(PROJECT/entry['base_root'],ids)
        assert bi['response_replay']['equal_original_avalanche_events']==len(old['ids'])
        np.testing.assert_array_equal(full['signal_mV'],base['signal_mV'])
        readout=info['config']['sipm']['readout'];lsb=(readout['adc_max_mV']-readout['adc_min_mV'])/2**readout['adc_bits']
        f=wave_features(full['adc']);anchor=np.floor(f['t10_sample']).astype(int)+1
        score,q,gate=linear_scores(full['adc'],anchor,f['baseline'],pca)
        np.testing.assert_allclose(score,old['scores'][ids],atol=1e-10,rtol=1e-10)
        signal=full['signal_mV']/lsb
        components=np.stack([signal,(base['analog_mV']-full['signal_mV'])/lsb,
            (full['analog_mV']-base['analog_mV'])/lsb,
            full['adc']-full['analog_mV']/lsb],axis=1)
        components-=components[:,:,:128].mean(2)[:,:,None]
        np.testing.assert_allclose(components.sum(1),full['adc']-f['baseline'][:,None],atol=1e-10,rtol=1e-10)
        cg=components[np.arange(len(ids))[:,None,None],np.arange(4)[None,:,None],anchor[:,None,None]+np.arange(-20,1500)]
        w,c=weights(pca);num=cg@w.T;cq=cg.sum(2)
        variants={'full':score}
        variants['recent_baseline']=linear_scores(full['adc'],anchor,recent(full['adc'],anchor),pca)[0]
        for name,x in [('no_signal_residual',base['adc']),('signal_only',signal)]:
            xf=wave_features(x)
            variants[name]=score_waves(x,pca,xf)
            variants[name+'_fixed_anchor']=linear_scores(x,anchor,xf['baseline'],pca)[0]
        bins=np.clip(np.searchsorted(edges,q,'right')-1,0,5)
        decomposition=[];projections=np.empty((len(ids),4,3));maxerr=0.
        for b in range(6):
            use=bins==b
            if use.sum()<2:continue
            r0=score[use].mean(0)
            u=(num[use]-(r0+c)[None,None,:]*cq[use,:,None])/q[use,None,None]
            maxerr=max(maxerr,float(abs(u.sum(1)-(score[use]-r0)).max()))
            projections[use]=u
            cov=np.stack([np.cov(u[:,:,j],rowvar=False) for j in range(3)])
            np.testing.assert_allclose(cov.sum((1,2)),np.var(score[use],axis=0,ddof=1),rtol=1e-9,atol=1e-16)
            decomposition.append({'bin':b,'n':int(use.sum()),'covariances':cov.tolist(),
                'total_variances':cov.sum((1,2)).tolist(),
                'diagonal_fraction':(np.diagonal(cov,axis1=1,axis2=2)/cov.sum((1,2))[:,None]).tolist(),
                'cross_fraction':((cov.sum((1,2))-np.trace(cov,axis1=1,axis2=2))/cov.sum((1,2))).tolist()})
        np.savez_compressed(OUT/'arrays'/f'{state}_components.npz',ids=ids,q=q,anchor=anchor,
           signal=signal,base_analog=base['analog_mV']/lsb,full_adc=full['adc'],
           projections=projections,**variants)
        report['MC'][state]={'n':len(ids),'root':entry['parent_root'],'base_root_sha256':sha(PROJECT/entry['base_root']),
            'same_signal':True,'equal_avalanche_events':bi['response_replay']['equal_original_avalanche_events'],
            'maximum_score_identity_error':maxerr,'per_Q':cells(q,variants,edges),'decomposition':decomposition,
            'baseline_RMS_median':float(np.median(f['baseline_rms'])),
            'recent_minus_original_baseline_sd':float(np.std(recent(full['adc'],anchor)-f['baseline'],ddof=1))}
        print(state,'component fractions QDC',np.round(np.array([r['diagonal_fraction'][0] for r in decomposition]),3).tolist(),flush=True)
    for run in [3,12,18]:
        for segment in ['first60','middle20','last20']:
            key=f'run{run}_{segment}'
            path=(PARENT/'run18' if run==18 else PROJECT/'simulation/results/signal_fluctuations_20261028')/(key+'.npz')
            data=load(path);waves=data['waves'];f=wave_features(waves);anchor=np.floor(f['t10_sample']).astype(int)+1
            scores,q,_=linear_scores(waves,anchor,f['baseline'],pca)
            frozen=load(PID/'arrays'/f'measured_{key}.npz')
            np.testing.assert_allclose(scores,frozen['scores'],rtol=1e-10,atol=1e-10)
            delta=recent(waves,anchor)-f['baseline']
            alternative=linear_scores(waves,anchor,recent(waves,anchor),pca)[0]
            main=scores[:,0]>=.4 if run==3 else scores[:,0]<.4
            variants={'original':scores,'recent_baseline':alternative}
            np.savez_compressed(OUT/'arrays'/f'measured_{key}.npz',ids=data['ids'],q=q,
                scores=scores,recent_baseline=alternative,baseline_delta=delta,main_band=main)
            report['measured'][key]={'n':len(q),'main_band_n':int(main.sum()),'outside_main_band_n':int((~main).sum()),
                'input':str(path.relative_to(PROJECT)),'input_sha256':sha(path),'per_Q':cells(q,variants,edges),
                'main_band_per_Q':cells(q,variants,edges,main),'baseline_RMS_median':float(np.median(f['baseline_rms'])),
                'recent_minus_original_baseline_sd':float(np.std(delta,ddof=1))}
            print(key,'baseline RMS/delta',report['measured'][key]['baseline_RMS_median'],report['measured'][key]['recent_minus_original_baseline_sd'],flush=True)
    save(OUT/'diagnosis.json',report)


if __name__=='__main__':main()
