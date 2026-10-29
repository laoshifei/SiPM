"""Charge-stratified transfer checks for publication-cohort PSD definitions.

PCA is learned only on run3/12 first60. Check data were previously explored.
IQR refers to the whole selected score distribution, not a particle-peak width.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import publication_stats as p

COMPARISONS=('run3_middle20','run3_last20','run7_first60','run12_middle20','run12_last20','run18_first60')


def scores(d,model):
    matrix=d['block_charge'][:,:model['blocks']].astype(float)/d['long_charge'][:,2,None]
    return np.column_stack((d['short_charge'][:,2]/d['long_charge'][:,2],-d['mean_time_us'][:,2],
        ((matrix-np.asarray(model['center']))@np.asarray(model['vector']))/model['score_scale']))


def distance(x,y):
    x=np.sort(x);y=np.sort(y);points=np.r_[x,y]
    return float(abs(np.searchsorted(x,points,'right')/len(x)-np.searchsorted(y,points,'right')/len(y)).max())


def read_groups(out):
    model=json.loads((out/'point.json').read_text())['windows']['3000']['pca'];result={}
    for run in p.RUNS:
        pool=p.load(out/f'pool_run{run}.npz');n=int(pool['original_count'])
        q=p.load(p.HERE/f'results/pha_run{run:04d}_features.npz')['charge_long']
        for s,group in enumerate(p.GROUPS):
            d=p.subset(pool,np.flatnonzero(pool['primary']&(pool['split']==s)))
            d['q']=q[d['ids']];d['scores']=scores(d,model)
            d['span']=[(0,int(.6*n)),(int(.6*n),int(.8*n)),(int(.8*n),n)][s]
            result[f'run{run}_{group}']=d
    return result


def reference(key):
    run=int(key.split('_')[0][3:]);return f'run{3 if run in (3,7) else 12}_first60'


def cells(a,b,nbins=6,edges=None,gate=None):
    if edges is None:
        ac=a['charge_bin'];bc=b['charge_bin']
    else:
        ac=np.minimum(np.searchsorted(edges,a['q'],side='right')-1,nbins-1)
        bc=np.minimum(np.searchsorted(edges,b['q'],side='right')-1,nbins-1)
    usea=np.ones(len(ac),bool);useb=np.ones(len(bc),bool)
    if gate=='strict_pretrigger':usea&=a['eligible_strict'];useb&=b['eligible_strict']
    if gate=='exclude_secondary_rise':usea&=~a['secondary_rise_flag'];useb&=~b['secondary_rise_flag']
    if gate=='ratio_030_035':
        usea&=(a['short_long_ratio']>=.30)&(a['short_long_ratio']<.35)
        useb&=(b['short_long_ratio']>=.30)&(b['short_long_ratio']<.35)
    return [(i,np.flatnonzero((ac==i)&usea),np.flatnonzero((bc==i)&useb)) for i in range(nbins)
            if min(np.sum((ac==i)&usea),np.sum((bc==i)&useb))>=20]


def metrics(x,y):
    a=np.quantile(x,[.25,.5,.75],axis=0);b=np.quantile(y,[.25,.5,.75],axis=0)
    width=b[2]-b[0]
    if np.any(width<=0):raise ValueError('Reference IQR is zero')
    return np.array([[distance(x[:,j],y[:,j]) for j in range(3)],(a[1]-b[1])/width,(a[2]-a[0])/width])


def matched(out):
    groups=read_groups(out);protocol=json.loads((out/'protocol.json').read_text());report={'scope':__doc__,
      'point_sha256':p.sha(out/'point.json'),'code_sha256':p.sha(Path(__file__)),
      'columns':['empirical KS','median shift/reference IQR','whole-distribution IQR ratio'],
      'method_order':p.METHODS,'metric_axes':['metric','method'],
      'equal_N':'256 repeats; within each supported Q bin, n=min(nref,ncheck,200) without replacement for observed check/reference. Two independent with-replacement samples of the same n from reference provide an IID conditional finite-N benchmark, not a time-aware null test or p-value. Same indices for all methods.',
      'matching':'Equal weight of supported Q bins (coarse matching, not exact continuous-Q matching). Sensitivity uses12 bins, stricter pretrigger, secondary-rise exclusion and overlapping Am shape gate. Each support count is explicit.',
      'groups':{},'matched':{}}
    for key,d in groups.items():
        ref=groups[reference(key)];valid=cells(d,ref)
        value=[metrics(d['scores'][a],ref['scores'][b]) for _,a,b in valid]
        report['groups'][key]={'reference':reference(key),'events':len(d['ids']),
          'supported_bins':[i for i,_,_ in valid], 'bin_counts':[[len(a),len(b)] for _,a,b in valid],
          'equal_bin_point':np.mean(value,axis=0).tolist() if value else None}
    for key in COMPARISONS:
        d=groups[key];ref=groups[reference(key)];valid=cells(d,ref);observed=[];null=[]
        rng=np.random.default_rng(28102871+int(key.split('_')[0][3:])+100*p.GROUPS.index(key.split('_')[1]))
        for _ in range(256):
            samples=[];references=[]
            for _,a,b in valid:
                n=min(len(a),len(b),200)
                samples.append(metrics(d['scores'][rng.choice(a,n,replace=False)],ref['scores'][rng.choice(b,n,replace=False)]))
                references.append(metrics(ref['scores'][rng.choice(b,n,replace=True)],ref['scores'][rng.choice(b,n,replace=True)]))
            observed.append(np.mean(samples,axis=0));null.append(np.mean(references,axis=0))
        sensitivity={}
        edges=np.linspace(protocol['q_edges'][0],protocol['q_edges'][-1],13)
        for name in ('12_Q_bins','strict_pretrigger','exclude_secondary_rise','ratio_030_035'):
            valid2=cells(d,ref,12,edges) if name=='12_Q_bins' else cells(d,ref,gate=name)
            values=[metrics(d['scores'][a],ref['scores'][b]) for _,a,b in valid2]
            sensitivity[name]={'supported_bins':[i for i,_,_ in valid2],
              'bin_counts':[[len(a),len(b)] for _,a,b in valid2],
              'equal_bin_point':np.mean(values,axis=0).tolist() if values else None}
        report['matched'][key]={'per_bin_equal_n':[min(len(a),len(b),200) for _,a,b in valid],
          'observed_median':np.median(observed,axis=0).tolist(),'observed_subsampling_range95':np.quantile(observed,[.025,.975],axis=0).tolist(),
          'reference_IID_median':np.median(null,axis=0).tolist(),'reference_IID_range95':np.quantile(null,[.025,.975],axis=0).tolist(),
          'sensitivity':sensitivity}
        print('matched',key,'KS',report['matched'][key]['observed_median'][0],flush=True)
    p.save(out/'transfer_matched.json',report)


def bootstrap(out):
    point=json.loads((out/'transfer_matched.json').read_text())
    assert point['code_sha256']==p.sha(Path(__file__))
    groups=read_groups(out);training=p.training(out)
    files=sorted((out/'bootstrap_b64').glob('draw*.npz'))
    if len(files)!=256:raise ValueError('Wait for all256 paired training draws')
    folder=out/'transfer_bootstrap';folder.mkdir(exist_ok=True);collected=[]
    for i,path in enumerate(files):
        checkpoint=folder/f'draw{i:04d}.npz'
        if checkpoint.exists():values=np.load(checkpoint)['values']
        else:
            draw=p.load(path);model={'blocks':76,'center':draw['pca_center_3000'],'vector':draw['pca_vector_3000'],
                'score_scale':float(draw['pca_scale_3000'])}
            replicas={key:scores(groups[key],model) for key in set(COMPARISONS)|{'run3_first60','run12_first60'}}
            rng=np.random.default_rng(np.random.SeedSequence([28102872,i]));values=[]
            for key in COMPARISONS:
                d=groups[key];rk=reference(key);ref=groups[rk];r=int(rk.split('_')[0][3:])
                wref=draw['counts'][training['run']==r]
                np.testing.assert_array_equal(ref['ids'],training['ids'][training['run']==r])
                wc=p.block_counts(d['ids'],d['span'],64,rng);means=[]
                for _,a,b in cells(d,ref):
                    x=np.repeat(a,wc[a]);y=np.repeat(b,wref[b])
                    means.append(metrics(replicas[key][x],replicas[rk][y]))
                values.append(np.mean(means,axis=0))
            values=np.array(values);np.savez_compressed(checkpoint,values=values)
        collected.append(values)
        if i==0 or (i+1)%64==0:print('transfer bootstrap',i+1,flush=True)
    boot=np.array(collected);report={'scope':__doc__,'matched_sha256':p.sha(out/'transfer_matched.json'),
      'uncertainty':'256 paired acquisition-index blocks of64, check segments separately resampled. Same training-draw PCA center/vector/scale and reference multiplicities as FOM; no training on check data. Fixed common Q support. Exploratory percentile intervals, no multiplicity correction. KS bootstrap is descriptive, not a null-test p-value.',
      'groups':{}}
    for k,key in enumerate(COMPARISONS):
        report['groups'][key]={'point':point['groups'][key]['equal_bin_point'],
          'interval95':np.quantile(boot[:,k],[.025,.975],axis=0).tolist()}
    p.save(out/'transfer_uncertainty.json',report)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['matched','bootstrap'])
    parser.add_argument('--out',type=Path,default=p.OUT);args=parser.parse_args()
    {'matched':matched,'bootstrap':bootstrap}[args.stage](args.out)
