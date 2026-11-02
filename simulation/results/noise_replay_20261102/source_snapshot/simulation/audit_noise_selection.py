"""Quantify the existing 0.4 QDC band cut on fixed experimental and MC cohorts."""
import argparse
import concurrent.futures
from noise_replay_common import *
from publication_stats import block_counts


def one(group,B):
    p=read(OUT/'selection_protocol.json');dest=OUT/'selection/evaluation'/f'{group}_B{B}.json';assert not dest.exists()
    run=int(group.split('_')[0][3:]);seg=group.split('_')[1];source=dict((r,s) for s,r in SOURCES)[run]
    old=load(DATA/f'{group}.npz');raw=features(old['waves'],old['ids'],old['q']);raw['fold']=(raw['ids']//B)%2
    ref=read(QDC/'references'/f'{group}_B{B}.json');data=compact(raw,p,source,ref)
    prior=load(QDC/'arrays'/f'{group}_B{B}_data.npz');data['span']=prior['span'];data['n_raw']=prior['n_raw']
    mask=band(data['X'],run);oldband=load(WIDTH/'arrays'/f'measured_{group}.npz')
    np.testing.assert_array_equal(data['ids'],oldband['ids']);np.testing.assert_array_equal(mask,oldband['main_band'])
    np.testing.assert_array_equal(data['ids'][mask],prior['ids'])
    np.testing.assert_allclose(data['X'][mask],old_compact(prior)['X'],rtol=1e-9,atol=1e-9)
    path=OUT/'selection/arrays'/f'{group}_B{B}_data.npz';np.savez_compressed(path,**data,main_band=mask)
    mc={v:old_compact(load(QDC/'arrays'/f'{group}_B{B}_{v}.npz')) for v in VARIANTS}
    frozen_mask=band(mc['frozen']['X'],run)
    masks={v:{'all':np.ones(len(a['ids']),bool),'own_band':band(a['X'],run),'fixed_band':frozen_mask} for v,a in mc.items()}
    ds={k:summary(data,p,source,mask=m) for k,m in [('all',None),('band',mask)]}
    ms={v:{k:summary(a,p,source,mask=m) for k,m in masks[v].items()} for v,a in mc.items()}
    pairing=[('all','all'),('band','all'),('band','own_band'),('band','fixed_band')]
    point=np.array([[contrast(ms[v][mk],ds[dk]) for dk,mk in pairing] for v in VARIANTS])
    draws=np.full((512,3,4,len(METRICS)),np.nan)
    previous=load(QDC/'evaluation'/f'{group}_B{B}_bootstrap.npz')['joint']
    oldmc=load(RC/'arrays'/f'{group}_B{B}_frozen_evaluation.npz')
    rng=np.random.default_rng(731031900+run+100*SEGMENTS.index(seg)+B)
    for ib in range(512):
        dc=block_counts(data['ids'],tuple(data['span']),B,rng)
        counts=mother_counts(mc['frozen'],rng,[EVENTS[source]]*4)
        mother_counts(oldmc,rng,[int(oldmc['n_parent'])])
        db={'all':summary(data,p,source,dc),'band':summary(data,p,source,dc,mask)}
        for iv,v in enumerate(VARIANTS):
            memo={};mb={}
            for k,m in masks[v].items():
                identity=m.tobytes()
                if identity not in memo:memo[identity]=summary(mc[v],p,source,counts,m)
                mb[k]=memo[identity]
            for ic,(dk,mk) in enumerate(pairing):draws[ib,iv,ic]=contrast(mb[mk],db[dk])
    mappings=[(2,2),(3,3),(4,4),(8,5),(9,6),(10,7),(14,0),(17,1)]
    for j,k in mappings:np.testing.assert_allclose(draws[:,:,1,j],previous[:,:,k],rtol=1e-8,atol=1e-10)
    bp=OUT/'selection/evaluation'/f'{group}_B{B}_bootstrap.npz';np.savez_compressed(bp,contrasts=draws)
    survival={'data':{'all_events':len(mask),'band_events':int(mask.sum()),'band_weight_mass':float(event_weights(data,p,source)@mask),
                     'nonpositive_pulse_mean':int(np.sum(data['mu']<=0)), 'per_Q':ds['all']['per_Q']}}
    comparisons={}
    for iv,v in enumerate(VARIANTS):
        w=event_weights(mc[v],p,source)
        survival[v]={k:{'events':int(m.sum()),'weighted_mass':float(w@m)} for k,m in masks[v].items()}
        comparisons[v]={}
        for ic,case in enumerate(CASES):
            comparisons[v][case]={'point':dict(zip(METRICS,point[iv,ic])),
                'CI95':{n:interval(draws[:,iv,ic,j]) for j,n in enumerate(METRICS)},
                'change_from_inherited':dict(zip(METRICS,point[iv,ic]-point[iv,1])),
                'change_CI95':{n:interval(draws[:,iv,ic,j]-draws[:,iv,1,j]) for j,n in enumerate(METRICS)}}
    save(dest,serialize({'group':group,'block':B,'source':source,'protocol_sha256':sha(OUT/'selection_protocol.json'),
        'data_array_sha256':sha(path),'bootstrap_sha256':sha(bp),'data_summaries':ds,'MC_summaries':ms,
        'survival':survival,'comparisons':comparisons,'parent_scalar_checks':512*3*8}))
    print('Audited',group,B,'same observable band and all parent512 draws.',flush=True)


def main(workers):
    p=read(OUT/'selection_protocol.json');assert not (OUT/'selection_analysis.json').exists()
    for k,v in p['code_sha256'].items():assert sha(PROJECT/k)==v,k
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool0:
        jobs=[pool0.submit(one,f'run{run}_{seg}',B) for _,run in SOURCES for seg in SEGMENTS for B in p['block_sizes']]
        for j in concurrent.futures.as_completed(jobs):j.result()
    save(OUT/'selection_analysis.json',{'protocol_sha256':sha(OUT/'selection_protocol.json'),'groups':{
        f'run{run}_{seg}':{str(B):read(OUT/'selection/evaluation'/f'run{run}_{seg}_B{B}.json') for B in p['block_sizes']}
        for _,run in SOURCES for seg in SEGMENTS}})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=2);args=parser.parse_args();main(args.workers)
