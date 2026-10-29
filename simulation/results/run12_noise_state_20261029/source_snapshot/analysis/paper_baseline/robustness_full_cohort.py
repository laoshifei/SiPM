"""Uncapped-data and finer-Q checks of the primary local covariance contrast.

Post-publication-baseline sensitivity, not a replacement primary cohort. The
endpoint is fixed at the50ns filter and500--900ns window. Shared unsupported
shape strata are counted; no event is relabelled as a particle or a bad pulse.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import publication_stats as p
sys.path.insert(0,str(p.SIM))
from analyze_residual_covariance import extract_windows
from reference import bin_layout

OUT=p.HERE/'results/robustness_20261028'
SCOPES=('full','frozen3000','time_quarter0','time_quarter1','time_quarter2','time_quarter3')
VIEWS=('ratio_lt040','ratio_030_035','unstratified')


def focus_residuals(waves,baseline,t10):
    """Only the pre and500--900ns windows required by the declared endpoint."""
    waves=np.asarray(waves,float)-baseline[:,None]
    pos=np.ceil(t10).astype(int)[:,None,None]+np.array([-300,250])[None,:,None]+np.arange(200)
    if pos.min()<12 or pos.max()>=waves.shape[1]-12:raise ValueError('Unsupported50ns filter')
    local=np.take_along_axis(waves[:,None,:],pos,axis=2);mu=local[:,1].mean(axis=1)
    if np.any(mu<=0):raise ValueError('Non-positive primary local signal mean')
    c=np.c_[np.zeros(len(waves)),np.cumsum(waves,axis=1)]
    smooth=(np.take_along_axis(c[:,None,:],pos+13,axis=2)-np.take_along_axis(c[:,None,:],pos-12,axis=2))/25
    residual=local-smooth;residual-=residual.mean(axis=2,keepdims=True)
    return residual[:,[1,0]]/np.sqrt(mu)[:,None,None]


def add_moments(count,total,square,keys,z):
    """Accumulate count, vector sum and temporal mean-square by integer group."""
    if not len(keys):return
    order=np.argsort(keys,kind='stable');unique,starts,ns=np.unique(keys[order],return_index=True,return_counts=True)
    values=z[order]
    count[unique]+=ns
    total[unique]+=np.add.reduceat(values,starts,axis=0)
    square[unique]+=np.add.reduceat(np.mean(values*values,axis=-1),starts,axis=0)


def from_moments(n,total,square):
    """Unbiased sample covariance at zero lag for temporally centred vectors."""
    n=np.asarray(n,float)
    if np.any(n<2):raise ValueError('A covariance cell needs at least two events')
    mean=total/n[...,None,None]
    values=n[...,None]/(n[...,None]-1)*(square/n[...,None]-np.mean(mean*mean,axis=-1))
    return values[...,0]-values[...,1]


def coarsen_q(n,total,square,bins):
    if 24%bins:raise ValueError('Q cells must divide24')
    shape=n.shape[:-1]+(bins,24//bins)
    return n.reshape(shape).sum(axis=-1),total.reshape(shape+total.shape[-2:]).sum(axis=-3),square.reshape(shape+square.shape[-1:]).sum(axis=-2)


def combine_blocks(d,factor):
    result={}
    for name in ('block_n','block_total','block_square'):
        a=d[name];length=a.shape[2];pad=(-length)%factor
        if pad:
            widths=[(0,0)]*a.ndim;widths[2]=(0,pad);a=np.pad(a,widths)
        result[name]=a.reshape(a.shape[:2]+(-1,factor)+a.shape[3:]).sum(axis=3)
    return result


def prepare(out):
    out.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((p.OUT/'protocol.json').read_text())
    design={'scope':__doc__,'parent_record_sha256':p.sha(p.OUT/'record.json'),
      'parent_protocol_sha256':p.sha(p.OUT/'protocol.json'),'code_sha256':p.sha(Path(__file__)),
      'original_endpoint':'50ns centered box residual,500--900ns after ceil(t10),200 samples. Equal-length baseline[-600,-200)ns. Each divided by local pulse mean; conditional event means removed.',
      'selection':'Same eligibility,amplitude>=200,bit0x8000clear,t10>=384 and fixed Q as the parent. Original first60 only. Full data and the exact frozen IDs both retained. No redefinition of the primary cohort.',
      'shape_policy':'Primary comparison has ratio<.40, the original common populated shape stratum. Additional0.30<=ratio<0.35 and unstratified point views. Rare unsupported>=.40 events retained and counted, not called bad or assigned particle truth.',
      'point_sensitivity':'Six,twelve,twenty-four equal-width Q cells, each run needs>=10 events. Temporal quarters refer to acquisition-index fractions within first60, not equal wall time. All four quarters share their common supported six-Q-bin intersection.',
      'bootstrap':'512 original-index blocks64 and256 blocks128. Re-estimate each cell mean. Same run-specific draw for full/frozen and both shape views; point-six-Q support fixed. Percentile intervals conditional on recorded population,chosen model and cuts.',
      'axes':{'global':['scope','shape_view','24_Q_cells','signal_or_pre','200_samples'],
        'block':['full_or_frozen','first_two_shape_views','original_block64','6_Q_cells','signal_or_pre','200_samples']},
      'runs':{},'input_sha256':{}}
    for run in (12,18):
        source=p.OUT/f'pool_run{run}.npz';d=p.load(source);nraw=int(d['original_count']);span=int(.6*nraw)
        eligible=(d['split']==0)&d['eligible_primary']&(d['amplitude']>=200)&((d['flags']&0x8000)==0)&(d['t10_sample']>=384)
        data=p.subset(d,np.flatnonzero(eligible));ids=data['ids'];del d
        q=data['long_charge'][:,2];q24=np.minimum(np.searchsorted(np.linspace(protocol['q_edges'][0],protocol['q_edges'][-1],25),q,side='right')-1,23)
        np.testing.assert_array_equal(q24//4,data['charge_bin'])
        assert np.all((q24>=0)&(q24<24))
        ratio=data['short_long_ratio'];rare=ids[ratio>=.4]
        quarter=np.minimum(ids*4//span,3)
        nblocks=(span+63)//64
        stats={'n':np.zeros((6,3,24),np.int64),'total':np.zeros((6,3,24,2,200)),
          'square':np.zeros((6,3,24,2)), 'block_n':np.zeros((2,2,nblocks,6),np.int64),
          'block_total':np.zeros((2,2,nblocks,6,2,200)), 'block_square':np.zeros((2,2,nblocks,6,2)),
          'ids':ids,'q':q,'ratio':ratio,'original_span':np.array(span),'frozen_mask':data['primary']}
        target=out/f'full_run{run}.npz'
        if not target.exists():
            path=p.PROJECT/f'data/DT5730_DPP_PHA/DAQ/run_{run}/UNFILTERED/Data_run_{run}.BIN'
            _,dtype,count=bin_layout(path);assert count==nraw
            raw=np.memmap(path,dtype=dtype,offset=2,mode='r',shape=(count,))
            for start in range(0,len(ids),512):
                end=min(start+512,len(ids));ix=np.arange(start,end)
                z=focus_residuals(raw['data'][ids[ix]],data['baseline'][ix],data['t10_sample'][ix])
                views=[ratio[ix]<.4,(ratio[ix]>=.30)&(ratio[ix]<.35),np.ones(len(ix),bool)]
                scopes=[np.ones(len(ix),bool),data['primary'][ix]]+[(quarter[ix]==j) for j in range(4)]
                for si,scope in enumerate(scopes):
                    for vi,view in enumerate(views):
                        keep=scope&view
                        add_moments(stats['n'][si,vi],stats['total'][si,vi],stats['square'][si,vi],q24[ix][keep],z[keep])
                        if si<2 and vi<2:
                            keys=(ids[ix]//64)*6+q24[ix]//4
                            add_moments(stats['block_n'][si,vi].reshape(-1),stats['block_total'][si,vi].reshape(-1,2,200),
                                        stats['block_square'][si,vi].reshape(-1,2),keys[keep],z[keep])
                if start==0 or (start//512+1)%32==0:print('full covariance run',run,'events',end,'/',len(ids),flush=True)
            np.savez_compressed(target,**stats)
        else:
            old=p.load(target)
            np.testing.assert_array_equal(old['ids'],ids)
            np.testing.assert_array_equal(old['frozen_mask'],data['primary'])
        design['runs'][str(run)]={'original_n':nraw,'first60_span':span,'full_n':len(ids),'frozen_n':int(data['primary'].sum()),
            'ratio_ge040_n':len(rare),'ratio_ge040_ids':rare.tolist(),'primary_coarse_shape_n':int((ratio<.4).sum()),
            'narrow_shape_n':int(((ratio>=.30)&(ratio<.35)).sum())}
        design['input_sha256'][str(source.relative_to(p.PROJECT))]=p.sha(source)
    target=out/'full_cohort_design.json'
    if target.exists():assert json.loads(target.read_text())==design
    else:p.save(target,design)


def run(out):
    design=json.loads((out/'full_cohort_design.json').read_text());assert design['code_sha256']==p.sha(Path(__file__))
    for path,digest in design['input_sha256'].items():assert p.sha(p.PROJECT/path)==digest
    data={r:p.load(out/f'full_run{r}.npz') for r in (12,18)}
    previous=json.loads((p.OUT/'covariance_uncertainty.json').read_text())
    points={}
    for bins in (6,12,24):
        ns=[];values=[]
        for r in (12,18):
            a=data[r];n,s,ss=coarsen_q(a['n'],a['total'],a['square'],bins)
            values.append(from_moments(np.maximum(n,2),s,ss));ns.append(n)
        ns=np.array(ns);values=np.array(values);valid=np.all(ns>=10,axis=0)
        records={}
        for si,scope in enumerate(SCOPES[:2]):
            rows={}
            for vi,view in enumerate(VIEWS):
                selected=valid[si,vi]
                if not selected.any():raise ValueError('No common supported Q cells')
                a=values[:,si,vi,selected].mean(axis=1)
                rows[view]={'supported_bins':np.flatnonzero(selected).tolist(),'counts':ns[:,si,vi].tolist(),
                  'equal_q_values':a.tolist(),'run12_over18_minus1':float(a[0]/a[1]-1)}
                if bins==6 and si==1 and vi<2:
                    old=np.array(previous['equal_q_point'])[:,vi,0,1]
                    np.testing.assert_allclose(a,old,rtol=2e-12,atol=2e-13)
            records[scope]=rows
        if bins==6:
            quarters={}
            for vi,view in enumerate(VIEWS[:2]):
                common=np.all(valid[2:,vi],axis=0);qs=[]
                if not common.any():
                    quarters[view]={'shared_supported_bins':[],'points':[]};continue
                for si in range(2,6):
                    v=values[:,si,vi,common].mean(axis=1)
                    qs.append({'quarter':si-2,'equal_q_values':v.tolist(),'relative':float(v[0]/v[1]-1),
                        'counts':ns[:,si,vi].tolist()})
                quarters[view]={'shared_supported_bins':np.flatnonzero(common).tolist(),'points':qs}
        points[str(bins)]=records
    bootstraps={}
    for size,draws in ((64,512),(128,256)):
        estimates=[]
        for r in (12,18):
            a=combine_blocks(data[r],size//64);nblocks=a['block_n'].shape[2]
            rng=np.random.default_rng(np.random.SeedSequence([281028201,size,r]))
            multiplicities=np.array([np.bincount(rng.integers(nblocks,size=nblocks),minlength=nblocks) for _ in range(draws)],float)
            values=np.empty((draws,2,2,6))
            for si in range(2):
                for vi in range(2):
                    n=multiplicities@a['block_n'][si,vi]
                    s=(multiplicities@a['block_total'][si,vi].reshape(nblocks,-1)).reshape(draws,6,2,200)
                    ss=(multiplicities@a['block_square'][si,vi].reshape(nblocks,-1)).reshape(draws,6,2)
                    values[:,si,vi]=from_moments(n,s,ss)
            estimates.append(values.mean(axis=-1))
        estimates=np.array(estimates).transpose(1,0,2,3) # draw,run,scope,view
        relative=estimates[:,0]/estimates[:,1]-1
        bootstrap={'draws':draws,'axes':['full_or_frozen','ratio_lt040_or_narrow'],
          'equal_q_interval95':np.quantile(estimates,[.025,.975],axis=0).tolist(),
          'relative_interval95':np.quantile(relative,[.025,.975],axis=0).tolist(),
          'paired_frozen_minus_full_relative_interval95':np.quantile(relative[:,1]-relative[:,0],[.025,.975],axis=0).tolist()}
        bootstraps[str(size)]=bootstrap;np.savez_compressed(out/f'full_cohort_bootstrap_b{size}.npz',equal_q=estimates,relative=relative)
        print('full covariance bootstrap',size,draws,flush=True)
    report={**design,'points':points,'time_quarters':quarters,'block_sizes':bootstraps}
    p.save(out/'full_cohort_covariance.json',report)
    print('full6bin',points['6']['full'],'finer12/24',points['12']['full']['ratio_030_035']['run12_over18_minus1'],points['24']['full']['ratio_030_035']['run12_over18_minus1'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['prepare','run'])
    parser.add_argument('--out',type=Path,default=OUT);args=parser.parse_args()
    {'prepare':prepare,'run':run}[args.stage](args.out)
