"""Paired core/tail diagnostics with frozen response parameters and training fences."""
from pid_core_tail_common import *
from publication_stats import block_counts


def main():
    assert not (OUT/'diagnostics.json').exists()
    p=read(OUT/'protocol.json');prior=read(PARENT/'evaluation.json');draws={}
    report={'scope':__doc__,'code_sha256':sha(__file__),'protocol_sha256':sha(OUT/'protocol.json'),
        'common_code_sha256':sha(PROJECT/'simulation/pid_core_tail_common.py'),'metrics':NAMES,'groups':{},
        'new_independent_mothers':0,'new_native_replays':0,'default_changed':False}
    for source,run in SOURCES:
        truth=load(OUT/'arrays'/f'run{run}_ancestry.npz')
        for si,seg in enumerate(SEGMENTS):
            key=f'run{run}_{seg}';data=load(OUT/'arrays'/f'{key}_data.npz');report['groups'][key]={}
            for B in p['block_sizes']:
                reference=read(OUT/'references'/f'{key}_B{B}.json')
                assignment=load(PROJECT/'simulation/results/state_response_20261030/arrays'/f'{key}_alignment_B{B}.npz')['mother_assignment']
                mc={v:native(run,seg,B,v) for v in VARIANTS}
                for a in mc.values():np.testing.assert_array_equal(a['ids'],truth['event_id'])
                dz=standardized(data,p,source,reference,(data['ids']//B)%2)
                mz={v:standardized(a,p,source,reference,assignment) for v,a in mc.items()}
                dp=diagnostics(data,p,source,dz,full=True);mp={v:diagnostics(a,p,source,mz[v],full=True) for v,a in mc.items()}
                points={v:contrasts(m,dp) for v,m in mp.items()}
                for v in VARIANTS:
                    old=prior['groups'][key][str(B)]['variants'][v]['point']
                    np.testing.assert_allclose(points[v][:3],[old[n] for n in ['IQR_ratio_ratio','IQR_time_ratio','IQR_PCA_ratio']],atol=1e-12)
                    np.testing.assert_allclose(points[v][12:15],[old[n] for n in ['SD_ratio_ratio','SD_time_ratio','SD_PCA_ratio']],atol=1e-12)
                rng=np.random.default_rng(610031900+run+100*si+B);n=int(mc['frozen']['n_parent'])
                boot=np.full((512,len(VARIANTS),len(NAMES)),np.nan)
                absolute=np.full((512,1+len(VARIANTS),2,3,2),np.nan)
                for ib in range(512):
                    dc=block_counts(data['ids'],tuple(data['span']),B,rng);db=diagnostics(data,p,source,dz,dc)
                    if db is None:continue
                    counts=np.bincount(rng.integers(n,size=n),minlength=n)
                    absolute[ib,0]=np.stack([db['tail_probability'],db['tail_variance_share']],axis=-1)
                    for iv,v in enumerate(VARIANTS):
                        a=mc[v];mb=diagnostics(a,p,source,mz[v],counts[a['ids']])
                        if mb is None:continue
                        boot[ib,iv]=contrasts(mb,db)
                        absolute[ib,iv+1]=np.stack([mb['tail_probability'],mb['tail_variance_share']],axis=-1)
                row={'data':dp,'data_tail_CI95':[[[interval(absolute[:,0,k,j,l]) for l in range(2)] for j in range(3)] for k in range(2)],'variants':{}}
                for iv,v in enumerate(VARIANTS):
                    r={'point':dict(zip(NAMES,points[v])),'native':mp[v],
                        'CI95':{name:interval(boot[:,iv,j]) for j,name in enumerate(NAMES)},
                        'paired_from_coarse':{name:interval(boot[:,iv,j]-boot[:,VARIANTS.index('coarse_spectrum'),j]) for j,name in enumerate(NAMES)},
                        'tail_CI95':[[[interval(absolute[:,iv+1,k,j,l]) for l in range(2)] for j in range(3)] for k in range(2)],
                        'ancestry':truth_parts(mc[v],p,source,mz[v],truth)}
                    row['variants'][v]=r
                    np.savez_compressed(OUT/'arrays'/f'{key}_B{B}_{v}_z.npz',ids=mc[v]['ids'],z=mz[v],fold=assignment)
                report['groups'][key][str(B)]=row
                np.savez_compressed(OUT/'arrays'/f'{key}_B{B}_data_z.npz',ids=data['ids'],z=dz,fold=(data['ids']//B)%2)
                draws[f'{key}_B{B}']=boot;draws[f'{key}_B{B}_absolute']=absolute
                print(key,B,'width50/SD shape ratios',[(v,np.round(points[v][15:18],3).tolist()) for v in VARIANTS],flush=True)
    save(OUT/'diagnostics.json',serialize(report));np.savez_compressed(OUT/'bootstrap.npz',**draws)


if __name__=='__main__':main()
