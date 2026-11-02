"""Propose bounded spectral-state parameters without tuning to native outcomes."""
import argparse
import concurrent.futures
import copy
from scipy.optimize import lsq_linear
from spectral_split_common import *
from mc_statistics_common import assemble,pool
from fit_signal_fluctuations import colored


def one(source):
    p=read(OUT/'protocol.json');run=dict(SOURCES)[source];context=f'run{run}_last20_B128'
    dest=OUT/'fits'/f'{source}.json';assert not dest.exists()
    data=load(REPLAY/'point_arrays'/f'{context}_data.npz');original=load(REPLAY/'point_arrays'/f'{context}_current.npz')
    # Compact point arrays omit timing. The frozen complete-pool zero-CV ADC
    # extraction retains the exact same r0 anchors for every original mother.
    zero_t10=np.concatenate([load(FULL/'arrays'/f'b{b:02d}_run{run}_zero_cv_r0.npz')['t10'] for b in range(4)])[original['ids']]
    repeats={v:load(REPLAY/'repeat_arrays'/f'{source}_{v}_all.npz') for v in ['current','zero_cv']}
    parent=read(PARENT/'protocol.json')
    signal=pool([assemble(b,run,'last20',128,'spectrum_CV_fit',data,parent,source) for b in range(4)])['signal']
    assert signal.shape==(len(original['ids']),2496)
    records={}
    for ef in [0,1]:
        use=original['fold']==ef;data_mask=(data['fold']!=ef).astype(int)
        target=covariance_summary(data['X'][:,3:6],data,p,source,data_mask)
        baseline={v:{key:np.mean([covariance_summary(a['X'][:,r,3:6],original,p,source,use.astype(int))[key] for r in range(16)],0) for key in ['mean','cov']} for v,a in repeats.items()}
        cp=PROJECT/parent['responses'][f'{context}_eval{ef}_spectrum']['config'];cfg=read(cp)
        model=cfg['signal_dependent_noise'];h=np.asarray(model['coefficients']);strength=model['variance_per_signal_adc'];oldcv=model['event_variance_cv']
        weight=covariance_weights(original,p,source,use);ids=np.flatnonzero(weight>0)
        G=np.zeros((16,4,3,3));rng=np.random.default_rng(p['proposal_seed']+run+100*ef)
        mean_F=np.zeros((16,4,3))
        for start in range(0,len(ids),64):
            ix=ids[start:start+64];s=signal[ix];t=zero_t10[ix]
            post=np.ceil(t+250).astype(int)[:,None]+np.arange(1000)
            mu=s[np.arange(len(ix))[:,None],post].mean(1);assert np.all(mu>0)
            envelope=np.sqrt(strength*np.maximum(s,0));length=2496+len(h)-1;edge=max(DELAYS)
            for r in range(16):
                w=rng.normal(size=(len(ix),length+2*edge));aux=rng.normal(size=w.shape)
                innovations=[w[:,edge:edge+length]]+[high_innovations(w,aux,d,edge) for d in DELAYS]
                for k,z in enumerate(innovations):
                    noise=colored(h,z)*envelope;f=F_metrics(noise,np.zeros(len(ix)),t)
                    q=(f[:,:,2]-f[:,:,1])/mu[:,None]
                    G[r,k]+=np.einsum('n,ni,nj->ij',weight[ix],q,q)
                    mean_F[r,k]+=weight[ix]@q
        sd=np.sqrt(np.diag(target['cov']));normal=sd[:,None]*sd[None,:]
        tri=np.triu_indices(3);targ=((target['cov']-baseline['zero_cv']['cov'])/normal)[tri]
        matrices=G.mean(0);candidates=[]
        for k,delay in enumerate(DELAYS,1):
            A=np.column_stack([(matrices[j]/normal)[tri] for j in [0,k]])
            if oldcv==0:
                b=float(np.clip(A[:,1]@targ/(A[:,1]@A[:,1]),0,4));a=0.;active=[True,b in [0.,4.]]
            else:
                fit=lsq_linear(A,targ,bounds=([0.,0.],[oldcv**2,4.]),tol=1e-12)
                assert fit.success;a,b=fit.x;active=(fit.active_mask!=0).tolist()
            pred=baseline['zero_cv']['cov']+a*matrices[0]+b*matrices[k]
            sensitivities=[]
            for sl in [slice(0,8),slice(8,16)]:
                gm=G[sl].mean(0);pc=baseline['zero_cv']['cov']+a*gm[0]+b*gm[k]
                sensitivities.append(np.sqrt(np.diag(pc)/np.diag(target['cov'])))
            candidates.append({'delay_samples':delay,'common_variance_cv':float(np.sqrt(a)),
                'high_variance_cv':float(np.sqrt(b/(1+a))),'a':a,'b':b,'active_bounds':active,
                'objective':float(np.sum((A@np.array([a,b])-targ)**2)),
                'predicted_cov':pred,'predicted_SD_ratio':np.sqrt(np.diag(pred)/np.diag(target['cov'])),
                'first8_last8_predicted_SD_ratio':sensitivities})
        chosen=int(np.argmin([x['objective'] for x in candidates]));candidate=candidates[chosen]
        output=copy.deepcopy(cfg);output['signal_dependent_noise'].update(event_variance_cv=candidate['common_variance_cv'],
            spectral_split={'model':'complementary_delay_variance','delay_samples':candidate['delay_samples'],'high_variance_cv':candidate['high_variance_cv']})
        output['status']='Frozen complementary spectral-state proposal; full-pool native evaluation pending; no default promotion.'
        config_path=OUT/'configs'/f'{context}_eval{ef}_split.json';save(config_path,output)
        np.savez_compressed(OUT/'fits'/f'{context}_eval{ef}_quadratic.npz',G=G,mean_F=mean_F,ids=original['ids'][ids],weights=weight[ids])
        records[str(ef)]={'config':relative(config_path),'config_sha256':sha(config_path),'source_config':relative(cp),
            'training_data_ids':data['ids'][data_mask>0],'evaluation_data_ids':data['ids'][data_mask==0],
            'original_mother_ids':original['ids'][ids],'target':target,'baseline':baseline,
            'quadratic_matrices':matrices,'candidates':candidates,'chosen_index':chosen,
            'zero_CV_SD_floor':np.sqrt(np.diag(baseline['zero_cv']['cov'])/np.diag(target['cov'])),
            'legacy_current_prediction':baseline['zero_cv']['cov']+oldcv**2*matrices[0],
            'legacy_current_actual':baseline['current']['cov']}
        print(source,'fold',ef,'delay',candidate['delay_samples'],'CV',candidate['common_variance_cv'],candidate['high_variance_cv'],'predicted SD',np.round(candidate['predicted_SD_ratio'],3),flush=True)
    save(dest,serialize({'source':source,'context':context,'folds':records,'protocol_sha256':sha(OUT/'protocol.json'),'code_sha256':sha(__file__)}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=2);args=parser.parse_args();p=read(OUT/'protocol.json')
    for name,h in {**p['code_sha256'],**p['inherited_code_sha256']}.items():assert sha(PROJECT/name)==h,name
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        for result in executor.map(one,[s for s,_ in SOURCES]):pass
    save(OUT/'fit.json',{s:read(OUT/'fits'/f'{s}.json') for s,_ in SOURCES})
