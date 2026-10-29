"""Check separately calibrated run18 states on common frozen waveform cohorts."""
import json
from pathlib import Path
import numpy as np
import ROOT
from reference import PROJECT, aligned_shapes
from compare import mc_wave_features
from diagnose_charge import sha
from diagnose_fluctuations import fluctuation_metrics
from screen_response_candidates import info_of, load, ks
from validate_signal_fluctuations import bootstrap, error, weights
from refit_native_analysis import arrays, features
from publication_mc_uncertainty import weighted_ks
from calibrate_signal_fluctuations import save

BASE=PROJECT/'simulation/results/waveform_refit_20261029'
OUT=BASE/'run18'


def main():
    parent=json.loads((BASE/'independent/analysis/validation.json').read_text())['sources']['Am241']
    assert parent['charge_balance']['passed']
    edges=np.array(json.loads((BASE/'independent/analysis/validation.json').read_text())['q_edges'])
    record={'scope':__doc__,'code_sha256':sha(Path(__file__)),'fit_sha256':sha(OUT/'fit.json'),
      'parent_validation_sha256':sha(BASE/'independent/analysis/validation.json'),'variants':{},
      'limits':'Separate run18 first60 residual calibration. Fixed-state middle/last checks exclude their pulse samples from fitting. Conditioned variant uses last20 pretrigger only; not a forecast. Same Am241 mother histories are reused, not new transport.',
      'q_edges':edges.tolist()}
    for variant in ['first60','last20']:
        folder=OUT/variant;path=folder/'events.root';info=info_of(path)
        assert info['response_replay']['equal_original_avalanche_events']==parent['nparent']
        assert info['events_requested']==parent['nparent'] and info['seed']==parent['seed']
        assert info['config']==json.loads((OUT/f'Am241_run18_{variant}.json').read_text())
        raw=load(path)['adc'];f,good=mc_wave_features(raw)
        selected=good&(f['amplitude']>=200)&(f['charge_long']>=edges[0])&(f['charge_long']<=edges[-1])&(f['t10_sample']>=384)
        ids=np.flatnonzero(selected);waves=raw[ids];mf={k:v[ids] for k,v in f.items()}
        shapes,keep=aligned_shapes(waves,mf);np.testing.assert_array_equal(keep,np.arange(len(ids)))
        metrics=fluctuation_metrics(waves,mf['baseline'],mf['t10_sample'])
        model={'ids':ids,'waves':waves,'q':mf['charge_long'],'baseline':mf['baseline'],'t10':mf['t10_sample'],
          'shapes':shapes,'metrics':metrics,'ratio':mf['short_long_ratio']}
        np.savez_compressed(folder/'Am241_native.npz',**model)
        fm=features(waves);row={'common_n':len(ids),'nparent':parent['nparent'],'seed':parent['seed'],
          'root_sha256':sha(path),'F':metrics[:,:,3].mean(0).tolist(),'replay':info['response_replay'],
          'physical_checks':'Inherited completed and validated independent Am241 transport; native replay reports exact avalanche identity for every mother event.',
          'comparisons':{}}
        for split,segment in enumerate(['first60','middle20','last20']):
            key=f'run18_{segment}';data=arrays(OUT/f'{key}.npz');fd=features(data['waves'])
            interval,draws=bootstrap(model,data,18,split,parent['nparent'])
            target=data['metrics'][:,:,3].mean(0);mc=metrics[:,:,3].mean(0)
            result={'n':len(data['ids']),'measured_F':target.tolist(),'relative':(mc/target-1).tolist(),
              'shape_nrmse':error(shapes.mean(0),data['shapes'].mean(0)),
              'common_Q_KS':ks(model['q'],data['q']),
              'F_distribution_KS':[ks(metrics[:,j,3],data['metrics'][:,j,3]) for j in range(3)],
              'features':{},**interval}
            for feature,x in fm.items():
                result['features'][feature]={'MC_quantiles':np.quantile(x,[.1,.25,.5,.75,.9]).tolist(),
                  'data_quantiles':np.quantile(fd[feature],[.1,.25,.5,.75,.9]).tolist(),'KS':ks(x,fd[feature])}
            counts=np.array([np.histogram(a['q'],edges)[0] for a in [data,model]])
            support=np.flatnonzero((counts[0]>=20)&(counts[1]>=10));matched={'support_zero_based':support.tolist(),'counts':counts.tolist()}
            if len(support):
                wd=weights(data['q'],edges,support);wm=weights(model['q'],edges,support)
                matched.update(shape_nrmse=error(wm@shapes,wd@data['shapes']),
                  relative=(wm@metrics[:,:,3]/(wd@data['metrics'][:,:,3])-1).tolist(),
                  weighted_Q_KS=weighted_ks(model['q'],wm,data['q'],wd))
            result['matched_Q']=matched;row['comparisons'][key]=result
            np.save(folder/f'Am241_{key}_relative_draws.npy',draws)
            print(variant,key,'mean shape',result['shape_nrmse'],'Frel',result['relative'],flush=True)
        record['variants'][variant]=row
    save(OUT/'validation.json',record)


if __name__=='__main__':main()
