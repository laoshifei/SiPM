"""Paired parent-decay bootstrap of existing native Geant4 response candidates.

Replays share parent histories. Their resamples share parent multiplicities,
including no-signal parents; they are not independent extra transport events.
All intervals are conditional on geometry, source/circuit assumptions and the
adaptively screened candidates. They are not parameter or selection-adjusted CIs.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import publication_stats as p
sys.path.insert(0,str(p.SIM))
from screen_response_candidates import load,info_of,measured_references,ks
from diagnose_fluctuations import fluctuation_metrics
from compare import mc_wave_features
from reference import aligned_shapes,TIME_NS,bin_layout

SOURCES=(('Co60','co60',3),('Am241','am241',12))
CANDIDATES=('control','native_best','prompt_plus_ap15')
REGION=(TIME_NS>=-100)&(TIME_NS<=3500)


def wmean(values,weights):
    if weights.sum()<=0:raise ValueError('Empty selected bootstrap sample')
    return np.tensordot(weights,values,axes=(0,0))/weights.sum()


def weighted_ks(x,wx,y,wy):
    if min(wx.sum(),wy.sum())<=0:raise ValueError('Empty KS resample')
    a=np.argsort(x);b=np.argsort(y);xx=x[a];yy=y[b];points=np.r_[xx,yy]
    cx=np.r_[0,np.cumsum(wx[a])]/wx.sum();cy=np.r_[0,np.cumsum(wy[b])]/wy.sum()
    return float(np.max(abs(cx[np.searchsorted(xx,points,'right')]-cy[np.searchsorted(yy,points,'right')])))


def measured(out,run):
    pool=p.load(out/f'pool_run{run}.npz');n=int(pool['original_count'])
    d=p.subset(pool,np.flatnonzero(pool['primary']&(pool['split']==0)));ids=d['ids']
    f=p.load(p.HERE/f'results/pha_run{run:04d}_features.npz')
    path=p.PROJECT/f'data/DT5730_DPP_PHA/DAQ/run_{run}/UNFILTERED/Data_run_{run}.BIN'
    _,dtype,count=bin_layout(path);assert count==n
    raw=np.memmap(path,dtype=dtype,offset=2,mode='r',shape=(count,));waves=raw['data'][ids]
    features={k:v[ids] for k,v in f.items() if v.ndim and len(v)==n}
    shapes,kept=aligned_shapes(waves,features);np.testing.assert_array_equal(kept,np.arange(len(ids)))
    fluctuations=fluctuation_metrics(waves,features['baseline'],features['t10_sample'])[:,:,3]
    return {'ids':ids,'fluctuations':fluctuations,'shapes':shapes,'q':features['charge_long'],'span':(0,int(.6*n))}


def prepare(out):
    folder=out/'mc';folder.mkdir(exist_ok=True)
    root=p.SIM/'results/response_candidates_20261028'
    old=json.loads((root/'screen.json').read_text());confirmation=json.loads((root/'confirmation_final/confirmation.json').read_text())
    refs,edges=measured_references();report={'scope':__doc__,'sources':{},
      'protocol_sha256':p.sha(out/'protocol.json'),'code_sha256':p.sha(Path(__file__)),
      'metrics':'Common-Q fluctuation means and area-normalized mean-shape NRMSE on the same publication training IDs; charge KS inside common-Q (new primary diagnostic) and broad selected charge KS (legacy diagnostic). No particle labels, no energy calibration.',
      'uncertainty':'512 independent parent-decay multinomial draws per source, shared across candidates. Measured training events use original-index blocks64 and are shared across candidates. Report both MC-only and joint recorded-event+MC conditional percentile intervals. No new transport or source-distance calibration.',
      'metric_order':['F50','F102','F202','mean_shape_NRMSE','common_Q_KS','broad_selected_Q_KS'],
      'limits':['Only2000 Co and12000 Am parent histories; selected common-Q events are much fewer.','Source/circuit/PDE/ADC assumptions and adaptive candidate selection are not covered by the intervals.','Broad selected-Q diagnostic uses the earlier broader measured selection; its data-reference uncertainty is held fixed.','Common-Q spectra are not reweighted between MC and data; differences may affect fluctuation and shape comparisons.']}
    for source,name,run in SOURCES:
        d=measured(out,run);np.savez_compressed(folder/(source+'_measured.npz'),**d)
        np.testing.assert_allclose(d['fluctuations'].mean(axis=0),refs[source]['fluctuations'],rtol=1e-12)
        np.testing.assert_allclose(d['shapes'].mean(axis=0),refs[source]['mean'],rtol=1e-10,atol=1e-14)
        row={'measured_n':len(d['ids']),'measured_F':d['fluctuations'].mean(axis=0).tolist(),'candidates':{}}
        for candidate in CANDIDATES:
            path=(root/'control'/name/'events.root') if candidate=='control' else root/'confirmation_final'/candidate/name/'events.root'
            waves=load(path)['adc'];info=info_of(path);f,good=mc_wave_features(waves)
            assert len(waves)==info['events_requested']
            selected=good&(f['amplitude']>=200)
            common=selected&(f['charge_long']>=edges[0])&(f['charge_long']<=edges[-1])&(f['t10_sample']>=384)
            ids=np.flatnonzero(common);selected_ids=np.flatnonzero(selected)
            shapes,keep=aligned_shapes(waves[ids],{k:v[ids] for k,v in f.items()})
            np.testing.assert_array_equal(keep,np.arange(len(ids)))
            fluct=fluctuation_metrics(waves[ids],f['baseline'][ids],f['t10_sample'][ids])[:,:,3]
            arrays={'nparent':np.array(len(waves)),'ids':ids,'selected_ids':selected_ids,'q':f['charge_long'][ids],
              'selected_q':f['charge_long'][selected_ids],'shapes':shapes,'fluctuations':fluct}
            oldrow=old['candidates']['prompt_0.000']['sources'][source] if candidate=='control' else confirmation['results'][candidate]['sources'][source]
            np.testing.assert_array_equal(ids,oldrow['common_ids']);np.testing.assert_allclose(fluct.mean(axis=0),oldrow['fluctuation_means'],rtol=1e-12)
            mean=shapes.mean(axis=0);target=d['shapes'].mean(axis=0)
            shape_error=np.sqrt(np.mean((mean[REGION]-target[REGION])**2))/target.max()
            metrics=np.r_[fluct.mean(axis=0),shape_error,ks(arrays['q'],d['q']),ks(arrays['selected_q'],refs[source]['q'])]
            np.testing.assert_allclose(metrics[[3,5]],[oldrow['shape_nrmse'],oldrow['charge_ks']],rtol=1e-10)
            np.savez_compressed(folder/(source+'_'+candidate+'.npz'),**arrays)
            row['candidates'][candidate]={'nparent':len(waves),'common_n':len(ids),'selected_n':int(selected.sum()),'point':metrics.tolist(),
                'root':str(path.relative_to(p.PROJECT)),
                'root_sha256':p.sha(path)}
            print('MC prepared',source,candidate,'common',len(ids),'of',len(waves),flush=True)
            del waves
        report['sources'][source]=row
    p.save(folder/'design.json',report)


def bootstrap(out):
    folder=out/'mc';design=json.loads((folder/'design.json').read_text())
    assert design['code_sha256']==p.sha(Path(__file__))
    refs,_=measured_references();report={**design,'results':{}}
    for source,name,run in SOURCES:
        data=p.load(folder/(source+'_measured.npz'))
        models={c:p.load(folder/(source+'_'+c+'.npz')) for c in CANDIDATES}
        n=int(models['control']['nparent']);means=data['fluctuations'].mean(axis=0);target=data['shapes'].mean(axis=0)
        values=[];ratios=[];mc_only=[];parent_sizes=[]
        for draw in range(512):
            rng=np.random.default_rng(np.random.SeedSequence([28102883,run,draw]))
            weights=np.bincount(rng.integers(n,size=n),minlength=n)
            wm=p.block_counts(data['ids'],data['span'],64,rng)
            fm=wmean(data['fluctuations'],wm);sm=wmean(data['shapes'],wm)
            row=[];fixed=[];selected_n=[]
            for c in CANDIDATES:
                d=models[c];w=weights[d['ids']];ws=weights[d['selected_ids']]
                fm_mc=wmean(d['fluctuations'],w);shape=wmean(d['shapes'],w)
                broad=weighted_ks(d['selected_q'],ws,refs[source]['q'],np.ones(len(refs[source]['q'])))
                row.append(np.r_[fm_mc,np.sqrt(np.mean((shape[REGION]-sm[REGION])**2))/sm.max(),
                    weighted_ks(d['q'],w,data['q'],wm),broad])
                fixed.append(np.r_[fm_mc,np.sqrt(np.mean((shape[REGION]-target[REGION])**2))/target.max(),
                    weighted_ks(d['q'],w,data['q'],np.ones(len(wm))),broad])
                selected_n.append(w.sum())
            values.append(row);mc_only.append(fixed);ratios.append(np.array(row)[:,:3]/fm-1);parent_sizes.append(selected_n)
        boot=np.array(values);ratios=np.array(ratios);fixed=np.array(mc_only)
        result={}
        for i,c in enumerate(CANDIDATES):
            point=np.array(design['sources'][source]['candidates'][c]['point'])
            result[c]={'point':point.tolist(),'joint_interval95':np.quantile(boot[:,i],[.025,.975],axis=0).tolist(),
              'mc_only_interval95':np.quantile(fixed[:,i],[.025,.975],axis=0).tolist(),
              'fluctuation_relative_residual':(point[:3]/means-1).tolist(),
              'fluctuation_relative_residual_joint_interval95':np.quantile(ratios[:,i],[.025,.975],axis=0).tolist(),
              'selected_common_n_interval95':np.quantile(np.array(parent_sizes)[:,i],[.025,.975]).tolist()}
        result['paired_native_best_minus_control']={'point':(np.array(result['native_best']['point'])-result['control']['point']).tolist(),
          'joint_interval95':np.quantile(boot[:,1]-boot[:,0],[.025,.975],axis=0).tolist()}
        result['measured_F_interval95']=np.quantile(np.array([wmean(data['fluctuations'],p.block_counts(data['ids'],data['span'],64,np.random.default_rng(np.random.SeedSequence([28102884,run,i])))) for i in range(512)]),[.025,.975],axis=0).tolist()
        np.savez_compressed(folder/(source+'_bootstrap.npz'),joint=boot,mc_only=fixed,relative=ratios)
        report['results'][source]=result
        print(source,'F50 relative intervals',[(c,result[c]['fluctuation_relative_residual_joint_interval95'][0][0],result[c]['fluctuation_relative_residual_joint_interval95'][1][0]) for c in CANDIDATES],flush=True)
    p.save(out/'mc_uncertainty.json',report)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['prepare','bootstrap'])
    parser.add_argument('--out',type=Path,default=p.OUT);args=parser.parse_args()
    {'prepare':prepare,'bootstrap':bootstrap}[args.stage](args.out)
