"""New-seed native ADC checks of a frozen bandwidth-response proposal."""
import argparse
import json
from pathlib import Path
import numpy as np
import ROOT
from reference import PROJECT,wave_features,aligned_shapes,TIME_NS
from compare import mc_wave_features
from prepare_pid import read
from prepare_pid_bandwidth import OUT,PARENT,PID
from diagnose_pid_bandwidth import METHODS
from extract_pid import load,score_waves,label_truth
from diagnose_fluctuations import fluctuation_metrics
from publication_stats import save,sha,block_counts
from fit_pid_bandwidth import average_bins
from pid_metrics import auc,operating_point

DEST=OUT/'validation'
SEGMENTS=['first60','middle20','last20']


def extract(entry,pca,edges):
    key=entry['key'];path=PROJECT/entry['path']
    froot=ROOT.TFile.Open(str(path));tree=froot.Get('Events')
    info=json.loads(str(froot.Get('RunInfo').GetString().Data()));n=int(tree.GetEntries())
    assert info['completed'] and n==entry['events'] and info['seed']==entry['seed']
    assert info['config']==read(PROJECT/entry['config'])
    if not entry['native']:
        assert info['response_replay']['equal_original_avalanche_events']==n
        assert info['response_replay']['max_relative_charge_balance']<1e-10
    tree.SetBranchStatus('*',0)
    for k in ['event_id','adc','edep_by_origin_MeV','edep_MeV']:tree.SetBranchStatus(k,1)
    waves=np.empty((n,2496),np.uint16);energy=np.empty((n,6));edep=np.empty(n)
    for i,e in enumerate(tree):
        assert e.event_id==i
        waves[i]=np.array(e.adc);energy[i]=np.array(e.edep_by_origin_MeV);edep[i]=e.edep_MeV
    froot.Close();np.testing.assert_allclose(energy.sum(1),edep,rtol=1e-10,atol=1e-12)
    f,quality=mc_wave_features(waves);selected=quality&(f['amplitude']>=200)&(f['t10_sample']>=384)
    common=selected&(f['charge_long']>=edges[0])&(f['charge_long']<=edges[-1])
    ids=np.flatnonzero(selected);cf={k:v[ids] for k,v in f.items()}
    scores=np.full((n,3),np.nan);scores[ids]=score_waves(waves[ids],pca,cf)
    cids=np.flatnonzero(common);cf={k:v[cids] for k,v in f.items()}
    metrics=fluctuation_metrics(waves[cids],cf['baseline'],cf['t10_sample'])[:,:,3]
    shapes,kept=aligned_shapes(waves[cids],cf);assert len(kept)==len(cids)
    labels,_=label_truth(energy)
    np.savez_compressed(DEST/'arrays'/f'{key}.npz',ids=np.arange(n),q=f['charge_long'],scores=scores,
        selected=selected,common=common,truth=labels,energy_by_origin=energy,
        amplitude=f['amplitude'],baseline_rms=f['baseline_rms'],t10=f['t10_sample'],
        common_ids=cids,metrics=metrics,shapes=shapes,common_waves=waves[cids])
    row={'key':key,'n_parent':n,'n_selected':int(selected.sum()),'n_common':len(cids),
       'root_sha256':sha(path),'replay':info.get('response_replay'),
       'truth_counts_common':np.bincount(labels[common],minlength=5).tolist(),
       'baseline_RMS_median':float(np.median(f['baseline_rms'][common]))}
    save(DEST/'extraction'/f'{key}.json',row);print(key,row['n_common'],row['truth_counts_common'],flush=True)


def quartiles(x):return np.quantile(x,[.25,.5,.75],axis=0)


def per_bin(q,scores,edges,counts=None):
    bins=np.clip(np.searchsorted(edges,q,'right')-1,0,5);out=np.full((6,3,3),np.nan)
    for b in range(6):
        use=bins==b;x=scores[use]
        if counts is not None:x=np.repeat(x,counts[use],axis=0)
        if len(x)>=5:out[b]=quartiles(x)
    return out


def interval(x):
    valid=np.isfinite(x);vals=x[valid]
    return {'estimate95':np.quantile(vals,[.025,.975]).tolist() if len(vals)>=128 else None,'valid_draws':len(vals)}


def widths(quarts):return quarts[...,2,:]-quarts[...,0,:]


def rms_width(iqr,support):return np.sqrt(np.mean(iqr[...,support,:]**2,axis=-2))


def mc_bootstrap(entries,edges):
    rng=np.random.default_rng(291029950);result={}
    for source,n in [('Co60',8000),('Am241',48000)]:
        group=[k for k,e in entries.items() if e['source']==source]
        arrays={k:load(DEST/'arrays'/f'{k}.npz') for k in group}
        draws={k:np.full((512,6,3,3),np.nan) for k in group}
        for b in range(512):
            c=np.bincount(rng.integers(n,size=n),minlength=n)
            for k,a in arrays.items():
                use=a['common'];draws[k][b]=per_bin(a['q'][use],a['scores'][use],edges,c[use])
        for k,a in arrays.items():
            use=a['common'];result[k]={'point':per_bin(a['q'][use],a['scores'][use],edges),'bootstrap':draws[k]}
            np.savez_compressed(DEST/'bootstrap'/f'{k}.npz',quartiles=draws[k])
    return result


def measured_boot(run,segment,edges):
    a=load(OUT/'arrays'/f'measured_run{run}_{segment}.npz')
    p=load(PID/'arrays'/f'measured_run{run}_{segment}.npz');n=int(p['original_count'])
    borders=[0,int(n*.6),int(n*.8),n];s=SEGMENTS.index(segment)
    rng=np.random.default_rng(291029960+run*10+s);boot=np.full((512,6,3,3),np.nan)
    use=a['main_band'];ids=a['ids'][use];q=a['q'][use];scores=a['scores'][use]
    for b in range(512):
        c=block_counts(ids,(borders[s],borders[s+1]),64,rng)
        boot[b]=per_bin(q,scores,edges,c)
    return {'point':per_bin(q,scores,edges),'bootstrap':boot,'n':len(q)}


def assess(protocol):
    entries=protocol['entries'];edges=np.asarray(protocol['q_edges'])
    fit=read(OUT/'covariance_fit/fit.json');mc=mc_bootstrap(entries,edges)
    report={'scope':__doc__,'code_sha256':sha(Path(__file__)),
      'protocol_sha256':sha(DEST/'protocol.json'),'methods':METHODS,'checks':{},'common_state_PID':{},
      'uncertainty':'512 paired whole-mother bootstrap for old/new MC, original-index block64 for each experimental segment; fixed parameters and independent MC vs data draws. Not full model uncertainty.',
      'width_summary':'RMS of three-method IQR separately over equal-weight frozen fit-supported Q bins; no mixing band means. Main-band0.4 experimental separator diagnostic only.',
      'interval_note':'Pointwise95% percentile intervals, no multiplicity correction. Sparse bootstrap cells are invalid rather than interpolated.'}
    for run,source in [(3,'Co60'),(12,'Am241'),(18,'Am241')]:
        support=fit['states'][f'run{run}']['support_bins']
        oldkey=f'old_run{run}_{source}';newkey=f'new_run{run}_{source}'
        old=load(DEST/'arrays'/f'{oldkey}.npz');new=load(DEST/'arrays'/f'{newkey}.npz')
        np.testing.assert_array_equal(old['energy_by_origin'],new['energy_by_origin'])
        for segment in SEGMENTS:
            key=f'run{run}_{segment}';data=measured_boot(run,segment,edges)
            np.savez_compressed(DEST/'bootstrap'/f'{key}.npz',quartiles=data['bootstrap'])
            path=(PARENT/'run18' if run==18 else PROJECT/'simulation/results/signal_fluctuations_20261028')/(key+'.npz')
            orig=load(path);da=load(OUT/'arrays'/f'measured_{key}.npz');main=da['main_band']
            data_F=average_bins(orig['metrics'][main,:,3],orig['q'][main],edges,support)
            data_shape=average_bins(orig['shapes'][main],orig['q'][main],edges,support)
            row={'experimental_n':data['n'],'support_bins':support,'before':{},'after':{},'per_Q':[],
                 'experimental_F_means':data_F.tolist()}
            for name,k,a in [('before',oldkey,old),('after',newkey,new)]:
                iw=widths(mc[k]['point']);dw=widths(data['point'])
                br=rms_width(widths(mc[k]['bootstrap']),support)/rms_width(widths(data['bootstrap']),support)
                ratio=rms_width(iw,support)/rms_width(dw,support)
                F=average_bins(a['metrics'],a['q'][a['common']],edges,support)
                shape=average_bins(a['shapes'],a['q'][a['common']],edges,support)
                region=(TIME_NS>=-40)&(TIME_NS<=3000)
                row[name]={'n_MC':int(a['common'].sum()),'IQR_ratio':ratio.tolist(),
                    'IQR_ratio_CI95':[interval(br[:,j]) for j in range(3)],
                    'MC_F_means':F.tolist(),'F_ratio':(F/data_F).tolist(),
                    'mean_shape_normalized_RMSE':float(np.sqrt(np.mean((shape[region]-data_shape[region])**2))/data_shape.max()),
                    'baseline_RMS_median':float(np.median(a['baseline_rms'][a['common']]))}
            for b in range(6):
                cell={'bin':b,'in_summary':b in support,'data_quartiles':data['point'][b].tolist(),
                      'before_quartiles':mc[oldkey]['point'][b].tolist(),'after_quartiles':mc[newkey]['point'][b].tolist(),
                      'n_old':int(np.sum(old['common']&(old['q']>=edges[b])&(old['q']<=edges[b+1]))),
                      'n_new':int(np.sum(new['common']&(new['q']>=edges[b])&(new['q']<=edges[b+1])))}
                for name,k in [('before',oldkey),('after',newkey)]:
                    br=widths(mc[k]['bootstrap'])[:,b]/widths(data['bootstrap'])[:,b]
                    cell[name+'_ratio_CI95']=[interval(br[:,j]) for j in range(3)]
                row['per_Q'].append(cell)
            # Same old-selected IDs: only events with defined scores in both versions.
            same=old['common']&new['selected'];assert same.sum()>=.95*old['common'].sum()
            op=per_bin(old['q'][same],old['scores'][same],edges)
            npair=per_bin(old['q'][same],new['scores'][same],edges)
            row['fixed_old_ids']={'n':int(same.sum()),'old_n':int(old['common'].sum()),
                'new_over_old_RMS_IQR':(rms_width(widths(npair),support)/rms_width(widths(op),support)).tolist(),
                'no_Q_reselection':True}
            report['checks'][key]=row
            print(key,'old/new IQR ratios',np.round(row['before']['IQR_ratio'],3),np.round(row['after']['IQR_ratio'],3),
                'new F',np.round(row['after']['F_ratio'],3),flush=True)
    alpha=load(DEST/'arrays/new_run12_Am241.npz');gamma=load(DEST/'arrays/new_run12_Co60.npz')
    ca=read(PROJECT/entries['new_run12_Am241']['config']);cg=read(PROJECT/entries['new_run12_Co60']['config'])
    for k in ['sipm','electronics_noise','signal_dependent_noise','readout_response']:assert ca[k]==cg[k]
    sa=alpha['scores'][alpha['common']&(alpha['truth']==0)]
    sg=gamma['scores'][gamma['common']&(gamma['truth']==1)]
    thresholds=read(PID/'thresholds.json')
    for j,method in enumerate(METHODS):
        cut=thresholds['methods'][method]['threshold']
        report['common_state_PID'][method]={'AUC':auc(sa[:,j],sg[:,j]),
           'fixed_previous_threshold':cut,'operating':operating_point(sa[:,j],sg[:,j],cut)}
    report['common_state_note']='Same run12 response for both sources; independent native seed and fixed old operating thresholds. Still conditional simulation performance, not experimental efficiency.'
    # Unavailable sparse-bin quartiles are explicitly null in JSON.
    def finite(obj):
        if isinstance(obj,float) and not np.isfinite(obj):return None
        if isinstance(obj,dict):return {k:finite(v) for k,v in obj.items()}
        if isinstance(obj,(tuple,list)):return [finite(x) for x in obj]
        return obj
    save(DEST/'assessment.json',finite(report))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--extract',nargs='*');parser.add_argument('--assess',action='store_true')
    args=parser.parse_args();protocol=read(DEST/'protocol.json')
    for d in ['arrays','extraction','bootstrap']:(DEST/d).mkdir(exist_ok=True)
    if args.extract is not None:
        pca=read(PROJECT/'analysis/paper_baseline/results/publication_20261028/point.json')['windows']['3000']['pca']
        for key in args.extract:
            assert not (DEST/'extraction'/f'{key}.json').exists(),'No overwrite'
            extract(protocol['entries'][key],pca,protocol['q_edges'])
    if args.assess:
        assert not (DEST/'assessment.json').exists(),'No overwrite'
        assess(protocol)


if __name__=='__main__':main()
