"""Validate a refit with native Geant4 ADC and unchanged measured selections."""
import argparse
import json
from pathlib import Path
import numpy as np
import ROOT
from reference import PROJECT, TIME_NS, aligned_shapes, wave_features
from compare import mc_wave_features
from screen_response_candidates import load, info_of, ks
from diagnose_charge import sha
from diagnose_fluctuations import fluctuation_metrics
from calibrate_signal_fluctuations import diagnostic, save
from validate import check
from validate_signal_fluctuations import bootstrap, error, weights
from publication_mc_uncertainty import weighted_ks

OLD=PROJECT/'simulation/results/signal_fluctuations_20261028'


def arrays(path):
    with np.load(path) as z:return dict(z)


def features(waves):
    f=wave_features(waves)
    return {'peak_ADC':f['amplitude'],'peak_per_Q':f['amplitude']/f['charge_long'],
      'tpeak_minus_t10_ns':2*(f['peak_sample']-f['t10_sample']),
      'short_long_ratio':f['short_long_ratio'],'baseline_RMS_ADC':f['baseline_rms'],
      'baseline_ADC':f['baseline']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mc',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--role',default='Native candidate diagnostic; not a blind test')
    args=parser.parse_args();args.mc=args.mc.resolve();args.out=args.out.resolve()
    args.out.mkdir(parents=True,exist_ok=False)
    edges=np.array(json.loads((OLD/'design.json').read_text())['q_edges'])
    report={'scope':__doc__,'code_sha256':sha(Path(__file__)),'role':args.role,'q_edges':edges.tolist(),'sources':{},
      'selection':'Frozen measured IDs unchanged. Native ADC recomputed quality,amplitude>=200,common Q,t10>=384. No amplitude scaling.',
      'uncertainty':'512 mother-event multinomial and measured raw-index block64 draws, fixed fitted parameters. Does not propagate model/calibration selection.',
      'limits':'Later measured segments were previously explored. Pretrigger-conditioned baseline variants must not be called forecasts.'}
    for source,name,run in [('Co60','co60',3),('Am241','am241',12)]:
        path=args.mc/name/'events.root';info=info_of(path)
        validity=check(path);assert validity['passed']
        raw=load(path)['adc'];f,good=mc_wave_features(raw)
        selected=good&(f['amplitude']>=200)
        common=selected&(f['charge_long']>=edges[0])&(f['charge_long']<=edges[-1])&(f['t10_sample']>=384)
        ids=np.flatnonzero(common);waves=raw[ids];fm={k:v[ids] for k,v in f.items()}
        assert len(ids)>=30
        shapes,keep=aligned_shapes(waves,fm);np.testing.assert_array_equal(keep,np.arange(len(ids)))
        metrics=fluctuation_metrics(waves,fm['baseline'],fm['t10_sample']);d=diagnostic(waves,fm,edges)
        root=ROOT.TFile.Open(str(path));tree=root.Get('Events');tree.SetBranchStatus('*',0)
        for k in ('signal_mV','analog_mV'):tree.SetBranchStatus(k,1)
        extra={k:[] for k in ('signal_mV','analog_mV')}
        for i in ids:
            tree.GetEntry(int(i))
            for k in extra:extra[k].append(np.array(getattr(tree,k)))
        root.Close()
        model={'ids':ids,'waves':waves,'q':fm['charge_long'],'ratio':fm['short_long_ratio'],
          'baseline':fm['baseline'],'t10':fm['t10_sample'],'shapes':shapes,'metrics':metrics,
          'power':d['power'],'frequency_MHz':d['frequency_MHz'],'selected_q':f['charge_long'][selected],
          **{k:np.array(v) for k,v in extra.items()}}
        np.savez_compressed(args.out/f'{source}_native.npz',**model)
        native_features=features(waves)
        row={'ROOT_path':str(path.relative_to(PROJECT)),'root_sha256':sha(path),'config':info['config'],
          'seed':info['seed'],'nparent':info['events_requested'],'common_n':len(ids),'selected_n':int(selected.sum()),
          'F':metrics[:,:,3].mean(0).tolist(),'charge_balance':validity,
          'is_replay':'response_replay' in info,'comparisons':{}}
        if 'response_replay' in info:row['replay']=info['response_replay']
        groups=[(run,split,label) for split,label in enumerate(('first60','middle20','last20'))]
        if run==12:groups.append((18,0,'first60'))
        for measured_run,split,label in groups:
            key=f'run{measured_run}_{label}';data=arrays(OLD/f'{key}.npz')
            fd=data['metrics'][:,:,3].mean(0);mf=np.array(row['F'])
            interval,draws=bootstrap(model,data,measured_run,split,row['nparent'])
            c={'n':len(data['ids']),'measured_F':fd.tolist(),'relative':(mf/fd-1).tolist(),
               'shape_nrmse':error(shapes.mean(0),data['shapes'].mean(0)),
               'common_Q_KS':ks(model['q'],data['q']),
               'F_distribution_KS':[ks(metrics[:,j,3],data['metrics'][:,j,3]) for j in range(3)],
               'F_IQR_ratios':(np.diff(np.quantile(metrics[:,:,3],[.25,.75],axis=0),axis=0)[0]/np.diff(np.quantile(data['metrics'][:,:,3],[.25,.75],axis=0),axis=0)[0]).tolist(),
               'features':{},**interval}
            measured_features=features(data['waves'])
            for feature,x in native_features.items():
                y=measured_features[feature]
                c['features'][feature]={'MC_quantiles':np.quantile(x,[.1,.25,.5,.75,.9]).tolist(),
                  'data_quantiles':np.quantile(y,[.1,.25,.5,.75,.9]).tolist(),'KS':ks(x,y)}
            counts=np.array([np.histogram(a['q'],edges)[0] for a in (data,model)])
            support=np.flatnonzero((counts[0]>=20)&(counts[1]>=10))
            c['matched_Q']={'support_zero_based':support.tolist(),'counts':counts.tolist()}
            if len(support):
                wd=weights(data['q'],edges,support);wm=weights(model['q'],edges,support)
                c['matched_Q'].update(shape_nrmse=error(wm@shapes,wd@data['shapes']),
                   relative=(wm@metrics[:,:,3]/(wd@data['metrics'][:,:,3])-1).tolist(),
                   weighted_Q_KS=weighted_ks(model['q'],wm,data['q'],wd),
                   feature_KS={key_:weighted_ks(x,wm,measured_features[key_],wd) for key_,x in native_features.items()})
            row['comparisons'][key]=c
            np.save(args.out/f'{source}_{key}_relative_draws.npy',draws)
            print(source,key,'n',len(ids),'shape',c['shape_nrmse'],'Frel',c['relative'],
                  'peak medians',c['features']['tpeak_minus_t10_ns']['MC_quantiles'][2],c['features']['tpeak_minus_t10_ns']['data_quantiles'][2],
                  'ratioKS',c['features']['short_long_ratio']['KS'],flush=True)
        report['sources'][source]=row
    save(args.out/'validation.json',report)


if __name__=='__main__':main()
