"""Paired-support selection/baseline checks with the primary PCA kept fixed."""
import json
from pathlib import Path
import numpy as np
import publication_stats as p
from publication_transfer import scores,cells,metrics,read_groups,reference


def alternative(out,key,model,variant):
    run=int(key.split('_')[0][3:]);seg=p.GROUPS.index(key.split('_')[1])
    pool=p.load(out/f'pool_run{run}.npz')
    mask=pool['primary']&(pool['split']==seg)
    if variant=='include_hardware_bit':
        mask=pool['eligible_primary']&(pool['amplitude']>=200)&(pool['t10_sample']>=384)&(pool['split']==seg)
    if variant=='strict_pretrigger':mask&=pool['eligible_strict']
    if variant=='exclude_secondary_rise':mask&=~pool['secondary_rise_flag']
    if variant=='ratio_030_035':mask&=(pool['short_long_ratio']>=.30)&(pool['short_long_ratio']<.35)
    ix=np.flatnonzero(mask)
    if len(ix)>3000:ix=ix[np.linspace(0,len(ix)-1,3000,dtype=int)]
    d=p.subset(pool,ix)
    if variant=='recent_baseline':
        delta=d['recent_baseline_delta'];q0=d['long_charge'][:,2].copy();q=q0-1520*delta
        d['mean_time_us'][:,2]=(d['mean_time_us'][:,2]*q0-delta*1520*(1500-21)/2*.002)/q
        d['long_charge'][:,2]=q;d['short_charge'][:,2]-=270*delta
        d['block_charge']=d['block_charge'].astype(float)-20*delta[:,None]
    d['scores']=scores(d,model)
    return d


def main(out=p.OUT):
    primary=read_groups(out);model=json.loads((out/'point.json').read_text())['windows']['3000']['pca']
    report={'scope':__doc__,'matched_sha256':p.sha(out/'transfer_matched.json'),'code_sha256':p.sha(Path(__file__)),
      'note':'Use the intersection of supported original six Q bins for the primary and each alternative. No effect is inferred by comparing different supported-bin sets. Including hardware bit may also change the deterministic3000-event cap; fixed Q membership and gates are not reoptimized. No extra uncertainty intervals for these point sensitivity checks.',
      'groups':{}}
    for key in ('run12_middle20','run12_last20','run18_first60'):
        ref=reference(key);rows={}
        for variant in ('include_hardware_bit','strict_pretrigger','exclude_secondary_rise','ratio_030_035','recent_baseline'):
            a=alternative(out,key,model,variant);b=alternative(out,ref,model,variant)
            va=cells(a,b);vp=cells(primary[key],primary[ref]);shared=sorted({i for i,_,_ in va}&{i for i,_,_ in vp})
            value=lambda x,y,c:np.mean([metrics(x['scores'][i],y['scores'][j]) for k,i,j in c if k in shared],axis=0).tolist()
            rows[variant]={'supported_bins':shared,'events':[len(a['ids']),len(b['ids'])],
                'primary_same_support':value(primary[key],primary[ref],vp),'alternative_same_support':value(a,b,va)}
        report['groups'][key]=rows
    p.save(out/'transfer_selection_sensitivity.json',report)
    print('run12 middle strict, matched support',report['groups']['run12_middle20']['strict_pretrigger'],flush=True)


if __name__=='__main__':main()
