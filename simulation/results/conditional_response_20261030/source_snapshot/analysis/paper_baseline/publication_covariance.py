"""Q-matched cross-run covariance with group means re-estimated in every draw.

Only zero-lag local covariance is given an interval here. It includes shape,
alignment, electronics and nonstationarity; it is not a measured avalanche gain.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import publication_stats as p
sys.path.insert(0,str(p.SIM))
from analyze_residual_covariance import extract_windows, covariance_parts
from reference import bin_layout

VIEWS=('all','ratio_030_035')


def weighted_parts(z, weights):
    """Exact sample-variance identity for integer event multiplicities."""
    n=weights.sum()
    if n<10:raise ValueError('Fewer than10 effective events')
    flat=z.reshape(len(z),-1)
    mean=(weights@flat/n).reshape(z.shape[1:])
    raw=weights@np.mean(z*z,axis=-1).reshape(len(z),-1)/n
    raw=raw.reshape(z.shape[1:-1])
    across=n/(n-1)*(raw-np.mean(mean*mean,axis=-1))
    return raw,across


def select(out,run,variant='primary'):
    pool=p.load(out/f'pool_run{run}.npz');count=int(pool['original_count'])
    mask=pool['primary']&(pool['split']==0)
    if variant=='strict_pretrigger':mask&=pool['eligible_strict']
    if variant=='exclude_secondary_rise':mask&=~pool['secondary_rise_flag']
    if variant=='include_hardware_bit':
        mask=pool['eligible_primary']&(pool['amplitude']>=200)&(pool['t10_sample']>=384)&(pool['split']==0)
    ix=np.flatnonzero(mask)
    if len(ix)>3000:ix=ix[np.linspace(0,len(ix)-1,3000,dtype=int)]
    d=p.subset(pool,ix);del pool
    path=p.PROJECT/f'data/DT5730_DPP_PHA/DAQ/run_{run}/UNFILTERED/Data_run_{run}.BIN'
    _,dtype,actual=bin_layout(path)
    assert count==actual
    raw=np.memmap(path,dtype=dtype,offset=2,mode='r',shape=(count,))
    baseline=d['baseline'].astype(float)
    if variant=='recent_baseline':baseline=baseline+d['recent_baseline_delta']
    means,residual=extract_windows(raw['data'][d['ids']],baseline,d['t10_sample'])
    d['signal']=residual[:,:,1:]/np.sqrt(means)[:,None,:,None]
    d['pre']=np.broadcast_to(residual[:,:,:1],d['signal'].shape)/np.sqrt(means)[:,None,:,None]
    d['span']=(0,int(.6*count))
    # Both Am first60 cohorts are inside the same previously defined <.4 shape band.
    if not np.all(d['short_long_ratio']<.4):raise ValueError('Need explicit coarse shape strata')
    return d


def cell_estimates(d,weights,view):
    selected=np.ones(len(weights),bool)
    if view=='ratio_030_035':selected=(d['short_long_ratio']>=.30)&(d['short_long_ratio']<.35)
    estimates=[];ns=[]
    for b in range(6):
        ix=np.flatnonzero(selected&(d['charge_bin']==b));w=weights[ix]
        _,signal=weighted_parts(d['signal'][ix],w)
        _,pre=weighted_parts(d['pre'][ix],w)
        estimates.append(signal-pre);ns.append(int(w.sum()))
    return np.array(estimates),ns


def evaluate(data,counts):
    result=[]
    for run in (12,18):
        result.append([cell_estimates(data[run],counts[run],view)[0] for view in VIEWS])
    return np.array(result)


def interval_array(x):
    return np.quantile(x,[.025,.975],axis=0).tolist()


def run(out):
    folder=out/'covariance';folder.mkdir(exist_ok=True)
    data={r:select(out,r) for r in (12,18)}
    one={r:np.ones(len(data[r]['ids']),int) for r in data}
    point=evaluate(data,one)
    old=json.loads((p.SIM/'results/residual_covariance_20261028/covariance.json').read_text())
    for ir,r in enumerate((12,18)):
        for iv,view in enumerate(VIEWS):
            oldcells=old['groups'][f'run{r}_first60']['views'][view]['cells']
            oldvalues=np.array([cell['across_event'] for cell in oldcells])[...,0]
            np.testing.assert_allclose(point[ir,iv],oldvalues,rtol=2e-12,atol=2e-13)
    design={'scope':__doc__,'protocol_sha256':p.sha(out/'protocol.json'),
      'code_sha256':{str(path.relative_to(p.PROJECT)):p.sha(path) for path in [Path(__file__),p.SIM/'analyze_residual_covariance.py']},
      'statistic':'Equal1/6 weight of fixed common Q bins. Each cell: n/(n-1)*(weighted second moment - squared re-estimated weighted mean), pulse minus equal-length baseline, both divided by local mean. Integer multiplicities reproduce expanded samples exactly.',
      'bootstrap':'512 acquisition-index-block draws at64;128 each at32/128. Empty blocks retained. Same draws across shape views and all12 local/filter combinations. Conditional exploratory95% percentile intervals, no multiplicity correction.',
      'axes':['run12/run18','all/ratio_030_035','six Q bins','filter50/102/202ns','local0-400/500-900/1200-1600/2200-2600ns'],
      'point_cells':point.tolist(),'equal_q_point':point.mean(axis=2).tolist(),
      'cell_counts':{str(r):{v:cell_estimates(data[r],one[r],v)[1] for v in VIEWS} for r in data}}
    target=folder/'design.json'
    if target.exists():assert json.loads(target.read_text())==design
    else:p.save(target,design)
    results={}
    for size,draws in ((64,512),(32,128),(128,128)):
        saved=[]
        for i in range(draws):
            path=folder/f'b{size}_draw{i:04d}.npy'
            if path.exists():est=np.load(path)
            else:
                rng=np.random.default_rng(np.random.SeedSequence([28102852,size,i]))
                counts={r:p.block_counts(data[r]['ids'],data[r]['span'],size,rng) for r in data}
                est=evaluate(data,counts);np.save(path,est)
            saved.append(est)
            if i==0 or (i+1)%64==0:print('covariance block',size,'draw',i+1,flush=True)
        boot=np.array(saved);equal=boot.mean(axis=3) # draw,run,view,filter,window
        effect=equal[:,0]/equal[:,1]-1
        results[str(size)]={'draws':draws,'equal_q_interval95':interval_array(equal),
            'run12_over_run18_minus1_point':(point[0].mean(axis=1)/point[1].mean(axis=1)-1).tolist(),
            'relative_effect_interval95':interval_array(effect),
            'difference_interval95':interval_array(equal[:,0]-equal[:,1]),
            'cell_relative_effect_interval95':interval_array(boot[:,0]/boot[:,1]-1)}
    sensitivity={}
    for variant in ('include_hardware_bit','strict_pretrigger','exclude_secondary_rise','recent_baseline'):
        new={r:select(out,r,variant) for r in data}
        values=evaluate(new,{r:np.ones(len(new[r]['ids']),int) for r in new})
        sensitivity[variant]={'events':{str(r):len(new[r]['ids']) for r in new},
          'equal_q_point':values.mean(axis=2).tolist(),
          'run12_over_run18_minus1':(values[0].mean(axis=1)/values[1].mean(axis=1)-1).tolist()}
        del new
    report={**design,'block_sizes':results,'selection_sensitivity':sensitivity}
    p.save(out/'covariance_uncertainty.json',report)
    print('50ns500-900ns effects',results['64']['run12_over_run18_minus1_point'][0][0][1],
          results['64']['run12_over_run18_minus1_point'][1][0][1],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,default=p.OUT)
    run(parser.parse_args().out)
