"""Common-supported-Q reweighting of fixed native response candidates.

This checks whether different within-range charge populations explain the
recorded discrepancy. It is an exploratory sensitivity, not detector calibration,
new transport, a particle-label measurement or a selection-adjusted test.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import publication_stats as p
from publication_mc_uncertainty import weighted_ks,REGION
from robustness_full_cohort import OUT

CANDIDATES=('control','native_best','prompt_plus_ap15')
TARGETS=('equal_bins','measured_bin_fractions_fixed')


def bin_ids(q,edges):
    if np.any((q<edges[0])|(q>edges[-1])):raise ValueError('Q outside primary range')
    return np.minimum(np.searchsorted(edges,q,side='right')-1,5)


def cell_sums(data,counts):
    cells=data['bins'];n=np.bincount(cells,weights=counts,minlength=6)
    f=np.zeros((6,3));s=np.zeros((6,data['shapes'].shape[1]))
    for b in range(6):
        ix=cells==b
        if ix.any():
            f[b]=counts[ix]@data['fluctuations'][ix]
            s[b]=counts[ix]@data['shapes'][ix]
    return n,f,s


def matched_moments(cells,bins,targets):
    n,f,s=cells
    if np.any(n[bins]<=0):raise ValueError('Empty fixed support cell in this draw')
    return targets@(f[bins]/n[bins,None]),targets@(s[bins]/n[bins,None])


def event_weights(cells,counts,bins,targets):
    n=np.bincount(cells,weights=counts,minlength=6);mapping=np.zeros(6)
    if np.any(n[bins]<=0):raise ValueError('Empty fixed support cell')
    mapping[bins]=targets/n[bins]
    return counts*mapping[cells]


def discrepancy(model,data):
    f,shape=model;fm,target=data
    return np.r_[f/fm-1,np.sqrt(np.mean((shape[REGION]-target[REGION])**2))/target.max()]


def interval(values):
    values=np.asarray(values);good=np.isfinite(values).all(axis=1)
    return {'valid':int(good.sum()),'draws':len(values),
      'interval95':np.quantile(values[good],[.025,.975],axis=0).tolist() if good.sum()>=.8*len(values) else None}


def run(out):
    out.mkdir(parents=True,exist_ok=True)
    edges=np.array(json.loads((p.OUT/'protocol.json').read_text())['q_edges'])
    design={'scope':__doc__,'parent_record_sha256':p.sha(p.OUT/'record.json'),'code_sha256':p.sha(Path(__file__)),
      'q_edges':edges.tolist(),'common_support':'Same fixed six Q cells. A cell needs>=20 measured and>=10 events in EVERY candidate;>=5 MC is a declared sensitivity. Support intersection is source-specific and identical across candidates. Never extrapolate missing high-Q cells.',
      'targets':'Equal supported-bin weights (primary sensitivity); measured supported-bin fractions held at original values (secondary). Within-cell Q distributions may still differ. Point weighted KS and weighted mean Q differences reported.',
      'resampling':'512 full-parent multinomial draws shared across candidates; measured original-index blocks64 shared across all variants. Support and target weights remain fixed. Empty resample cells invalidate the relevant statistic, counted without redefining support.',
      'metric_order':['F50 relative residual','F102 relative residual','F202 relative residual','mean_shape_NRMSE'],
      'assumptions':'Fixed previously selected candidates. No candidate-selection correction or geometry/circuit/PDE/calibration uncertainty. Sampling intervals are not goodness-of-fit p-values.',
      'inputs_sha256':{},'sources':{}}
    arrays={}
    for source,run in (('Co60',3),('Am241',12)):
        data={}
        for key in ('measured',)+CANDIDATES:
            path=p.OUT/'mc'/(source+'_'+key+'.npz');data[key]=p.load(path)
            data[key]['bins']=bin_ids(data[key]['q'],edges)
            design['inputs_sha256'][str(path.relative_to(p.PROJECT))]=p.sha(path)
        sums={key:cell_sums(d,np.ones(len(d['ids']))) for key,d in data.items()}
        source_design={'counts_by_bin':{key:sums[key][0].astype(int).tolist() for key in data},'settings':{}}
        for minimum in (10,5):
            support=sums['measured'][0]>=20
            for c in CANDIDATES:support&=sums[c][0]>=minimum
            bins=np.flatnonzero(support)
            if not len(bins):
                source_design['settings'][str(minimum)]={'supported_bins':[]};continue
            original_counts={key:np.ones(len(d['ids'])) for key,d in data.items()}
            nselected={key:int(sums[key][0][bins].sum()) for key in data}
            retained={key:nselected[key]/len(data[key]['ids']) for key in data}
            setting={'supported_bins':bins.tolist(),'selected_n':nselected,'retained_fraction':retained,'targets':{},'unweighted_same_support':{}}
            unweighted={key:matched_moments(sums[key],bins,sums[key][0][bins]/sums[key][0][bins].sum()) for key in data}
            setting['unweighted_same_support']={'measured_F':unweighted['measured'][0].tolist(),
                **{c:discrepancy(unweighted[c],unweighted['measured']).tolist() for c in CANDIDATES}}
            for target in TARGETS:
                weights=np.ones(len(bins))/len(bins) if target=='equal_bins' else sums['measured'][0][bins]/sums['measured'][0][bins].sum()
                moments={key:matched_moments(sums[key],bins,weights) for key in data}
                ew={key:event_weights(data[key]['bins'],original_counts[key],bins,weights) for key in data}
                row={'bin_weights':weights.tolist(),'measured_F':moments['measured'][0].tolist(),'candidates':{}}
                for c in CANDIDATES:
                    row['candidates'][c]={'point':discrepancy(moments[c],moments['measured']).tolist(),
                      'MC_F':moments[c][0].tolist(),'weighted_Q_KS':weighted_ks(data[c]['q'],ew[c],data['measured']['q'],ew['measured']),
                      'weighted_mean_Q_difference_adc_samples':float(ew[c]@data[c]['q']-ew['measured']@data['measured']['q']),
                      'effective_weighted_n':float(1/(ew[c]@ew[c]))}
                setting['targets'][target]=row
            source_design['settings'][str(minimum)]=setting
        design['sources'][source]=source_design
        arrays[source]=data
    design_path=out/'mc_charge_design.json'
    if design_path.exists():assert json.loads(design_path.read_text())==design
    else:p.save(design_path,design)
    report={**design,'bootstrap':{}}
    for source,run in (('Co60',3),('Am241',12)):
        data=arrays[source];settings=design['sources'][source]['settings'];nparent=int(data['control']['nparent'])
        draws={str(m):{t:[] for t in TARGETS} for m in (10,5)}
        for draw in range(512):
            rng=np.random.default_rng(np.random.SeedSequence([281028202,run,draw]))
            parents=np.bincount(rng.integers(nparent,size=nparent),minlength=nparent)
            d=data['measured'];counts={'measured':p.block_counts(d['ids'],d['span'],64,rng)}
            counts.update({c:parents[data[c]['ids']] for c in CANDIDATES})
            sums={key:cell_sums(d,counts[key]) for key,d in data.items()}
            for m in (10,5):
                setting=settings[str(m)];bins=np.array(setting['supported_bins'],int)
                if not len(bins):continue
                for target in TARGETS:
                    weights=np.array(setting['targets'][target]['bin_weights']);row=np.full((3,4),np.nan)
                    try:
                        reference=matched_moments(sums['measured'],bins,weights)
                        for i,c in enumerate(CANDIDATES):
                            row[i]=discrepancy(matched_moments(sums[c],bins,weights),reference)
                    except ValueError:pass
                    draws[str(m)][target].append(row)
            if draw==0 or (draw+1)%128==0:print('Q-matched MC',source,'draw',draw+1,flush=True)
        report['bootstrap'][source]={};saved={}
        for m in (10,5):
            report['bootstrap'][source][str(m)]={}
            for target in TARGETS:
                values=np.array(draws[str(m)][target]);saved[f'min{m}_{target}']=values
                row={c:interval(values[:,i]) for i,c in enumerate(CANDIDATES)}
                row['paired_best_minus_control']=interval(values[:,1]-values[:,0])
                report['bootstrap'][source][str(m)][target]=row
        np.savez_compressed(out/(source+'_mc_charge_bootstrap.npz'),**saved)
        print(source,'equal-Q points',{m:{c:settings[str(m)]['targets']['equal_bins']['candidates'][c]['point'] for c in CANDIDATES} for m in (10,5)},flush=True)
    p.save(out/'mc_charge_matched.json',report)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,default=OUT)
    run(parser.parse_args().out)
