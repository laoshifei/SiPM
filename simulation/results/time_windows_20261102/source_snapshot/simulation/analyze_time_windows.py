"""Exact temporal variance attribution with the inherited paired uncertainty."""
import argparse
import concurrent.futures
from time_window_common import *


def one(source):
    p=read(OUT/'protocol.json');old=load(SPLIT/'evaluation'/f'{source}_counts.npz');previous=read(SPLIT/'analysis.json')[source]
    arrays={v:load(OUT/'arrays'/f'{source}_{v}.npz') for v in ['data',*VARIANTS]};summaries={};details={}
    for v,a in arrays.items():
        count=old['data_counts'] if v=='data' else old['MC_counts']
        if v!='data':np.testing.assert_array_equal(a['ids'],old['ids'])
        summaries[v],details[v]=moments(a,p,source,np.vstack([np.ones((1,len(a['ids'])),int),count]))
        actual=summaries[v]['cov'][0].sum((-1,-2));expected=previous['data_summary'] if v=='data' else previous['summaries'][v]
        np.testing.assert_allclose(actual,np.diag(expected['cov'])[3:6],rtol=1e-9,atol=1e-10)
        np.testing.assert_allclose(summaries[v]['mean'][0].sum(1),np.asarray(expected['mean'])[3:6],rtol=1e-10,atol=1e-10)
    old_draw=load(SPLIT/'evaluation'/f'{source}_bootstrap.npz')['contrasts'];report={'source':source,'variants':{},'summaries':{},'cells':details,'fold_checks':{}}
    all_draws=[]
    for v in ['data',*VARIANTS]:report['summaries'][v]={k:value[0] for k,value in summaries[v].items()}
    for j,v in enumerate(VARIANTS):
        z=contrast(summaries[v],summaries['data']);all_draws.append(z[1:])
        for k,s in enumerate(SCALES):
            index=read(SPLIT/'analysis_protocol.json')['metrics'].index(f'F{s}_SD_ratio')
            np.testing.assert_allclose(z[1:,k,0],old_draw[:,[2,5][j],index],rtol=1e-8,atol=1e-9)
        report['variants'][v]={str(s):{'point':dict(zip(NAMES,z[0,k])),
            'CI95':{name:interval(z[1:,k,l]) for l,name in enumerate(NAMES)}} for k,s in enumerate(SCALES)}
    for ef in [0,1]:
        subset={v:moments(a,p,source,(a['fold']==ef).astype(int))[0] for v,a in arrays.items()}
        report['fold_checks'][str(ef)]={v:contrast(subset[v],subset['data'])[0] for v in VARIANTS}
    atomic_npz(OUT/'evaluation'/f'{source}_bootstrap.npz',contrasts=np.stack(all_draws,1))
    save(OUT/'evaluation'/f'{source}.json',serialize(report));print(source,'temporal attribution and all512 inherited F draws verified',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=2);args=parser.parse_args();p=read(OUT/'protocol.json')
    for name,h in {**p['code_sha256'],**p['inherited_code_sha256']}.items():assert sha(PROJECT/name)==h,name
    assert not (OUT/'analysis_protocol.json').exists()
    save(OUT/'analysis_protocol.json',{'code_sha256':sha(__file__),'arrays_sha256':{relative(x):sha(x) for x in (OUT/'arrays').glob('*.npz')},'protocol_sha256':sha(OUT/'protocol.json')})
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        for result in executor.map(one,[s for s,_ in SOURCES]):pass
    save(OUT/'analysis.json',{s:read(OUT/'evaluation'/f'{s}.json') for s,_ in SOURCES})
