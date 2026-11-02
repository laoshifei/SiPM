"""Paired complete-pool comparison against the already frozen spectral split."""
from lowfreq_common import *
from analyze_spectral_split import guards


def main():
    assert not (OUT/'analysis.json').exists();p=read(OUT/'protocol.json');previous=read(SPLIT/'analysis.json')[SOURCE]
    inputs=[OUT/'fit.json',OUT/'native_protocol.json',OUT/'extraction_protocol.json']+list((OUT/'arrays').glob('*.npz'))
    save(OUT/'analysis_protocol.json',{'code_sha256':sha(__file__),'inputs_sha256':{relative(x):sha(x) for x in inputs},
        'protocol_sha256':sha(OUT/'protocol.json'),'variants':VARIANTS,'metrics':METRICS})
    old=load(SPLIT/'evaluation/Am241_counts.npz');parts={v:[] for v in VARIANTS};selection={v:[] for v in VARIANTS}
    for b in range(4):
        key=f'b{b:02d}_run12';arrays={v:load((SPLIT if v.startswith('previous') else OUT)/'arrays'/f'{key}_split_r{v[-1]}.npz') for v in VARIANTS}
        union=np.isin(np.arange(EVENTS)+b*EVENTS,old['ids'])
        for a in arrays.values():union|=a['common']
        ix=np.flatnonzero(union)
        for v,a in arrays.items():
            ref=arrays['previous_r'+v[-1]]['supported'];use=a['supported']
            selection[v].append({'batch':b,'supported':int(use.sum()),'common':int(a['common'].sum()),
                'enter':int((use&~ref).sum()),'leave':int((ref&~use).sum())})
            fields=['q','qcalc','X','mean_wave','fold','selected','common','supported','old_common','truth']
            parts[v].append({**{k:a[k][ix] for k in fields},'ids':ix+b*EVENTS})
    arrays={v:merge(x) for v,x in parts.items()};ids=arrays[VARIANTS[0]]['ids'];data,ref,dp=read_data(p)
    packs={v:full_pack(a,p,SOURCE,ref) for v,a in arrays.items()};ds=statistics(dp,keep_Q=True);ms={v:statistics(a,keep_Q=True) for v,a in packs.items()}
    point=np.array([contrasts(ms[v],ds) for v in VARIANTS])
    for r in [0,1]:np.testing.assert_allclose(point[2*r],[previous['variants'][f'split_r{r}']['point'][n] for n in METRICS],rtol=1e-11,atol=1e-12)
    counts=extend_mother_counts(ids,old['ids'],old['MC_counts'],SOURCE,p['extension_seed'])
    np.savez_compressed(OUT/'evaluation/Am241_counts.npz',ids=ids,MC_counts=counts,data_counts=old['data_counts'],previous_ids=old['ids'])
    draws=np.full((512,4,len(METRICS)),np.nan)
    previous_draw=load(SPLIT/'evaluation/Am241_bootstrap.npz')['contrasts']
    reused=(OUT/'full_pool_reuse.json').exists()
    if reused:
        np.testing.assert_array_equal(ids,old['ids']);np.testing.assert_array_equal(counts,old['MC_counts'])
        for r in [0,1]:
            for field in arrays[f'previous_r{r}']:np.testing.assert_array_equal(arrays[f'previous_r{r}'][field],arrays[f'taper_r{r}'][field])
        draws=previous_draw[:,[2,2,5,5]].copy()
    else:
        for i in range(512):
            d=statistics(dp,old['data_counts'][i])
            for j,v in enumerate(VARIANTS):draws[i,j]=contrasts(statistics(packs[v],counts[i]),d)
            if (i+1)%128==0:print('Am241 low-frequency paired draws',i+1,flush=True)
    for r in [0,1]:np.testing.assert_allclose(draws[:,2*r],previous_draw[:,[2,5][r]],rtol=1e-9,atol=1e-10)
    np.savez_compressed(OUT/'evaluation/Am241_bootstrap.npz',contrasts=draws)
    result={'source':SOURCE,'context':'run12_last20_B128','selection':selection,'variants':{},'data_summary':ds,'summaries':ms,'fold_checks':{},
        'bootstrap_mode':'Reuse exact prior512 draws for byte-identical identity data and counts' if reused else 'Paired512 draws recomputed'}
    for j,v in enumerate(VARIANTS):
        base=2*(j//2);result['variants'][v]={'point':dict(zip(METRICS,point[j])),
            'CI95':{n:interval(draws[:,j,k]) for k,n in enumerate(METRICS)},
            'change_vs_previous':{'point':dict(zip(METRICS,point[j]-point[base])),
                'CI95':{n:interval(draws[:,j,k]-draws[:,base,k]) for k,n in enumerate(METRICS)}}}
        np.savez_compressed(OUT/'evaluation'/f'Am241_{v}.npz',**arrays[v])
    for ef in [0,1]:
        d=statistics(dp,(data['fold']==ef).astype(int));result['fold_checks'][str(ef)]={v:dict(zip(METRICS,contrasts(statistics(packs[v],(arrays[v]['fold']==ef).astype(int)),d))) for v in VARIANTS}
    result['joint_guards']={str(r):guards(result['variants'][f'taper_r{r}']['point'],result['variants'][f'previous_r{r}']['point'],p) for r in [0,1]}
    save(OUT/'analysis.json',serialize(result));print('Complete-pool low-frequency statistics and inherited512 draws verified.',flush=True)


if __name__=='__main__':main()
