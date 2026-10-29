"""Extract native ADC and latent signal for the photon-statistics profile."""
import argparse,json
from pathlib import Path
import numpy as np
import ROOT
from reference import PROJECT,wave_features,aligned_shapes
from compare import mc_wave_features
from prepare_pid import read
from prepare_npe_profile import OUT,PARENT
from extract_pid import score_waves,label_truth
from diagnose_fluctuations import fluctuation_metrics
from publication_stats import sha,save


def extract(entry,dest,edges,pca):
    path=PROJECT/entry['path'];froot=ROOT.TFile.Open(str(path));tree=froot.Get('Events')
    info=json.loads(str(froot.Get('RunInfo').GetString().Data()));n=int(tree.GetEntries())
    assert n==entry['events'] and info['completed'] and info['seed']==entry['seed']
    assert info['config']==read(PROJECT/entry['config'])
    tree.SetBranchStatus('*',0)
    for b in ['event_id','adc','edep_by_origin_MeV']:tree.SetBranchStatus(b,1)
    waves=np.empty((n,2496),np.uint16);energy=np.empty((n,6))
    for i,e in enumerate(tree):
        assert e.event_id==i
        waves[i]=np.array(e.adc);energy[i]=np.array(e.edep_by_origin_MeV)
    features,quality=mc_wave_features(waves)
    selected=quality&(features['amplitude']>=200)&(features['t10_sample']>=384)
    common=selected&(features['charge_long']>=edges[0])&(features['charge_long']<=edges[-1])
    ids=np.flatnonzero(common);cf={k:v[ids] for k,v in features.items()};scores=score_waves(waves[ids],pca,cf)
    shapes,kept=aligned_shapes(waves[ids],cf);assert len(kept)==len(ids)
    tree.SetBranchStatus('*',0)
    for b in ['signal_mV','analog_mV','avalanche_q_C']:tree.SetBranchStatus(b,1)
    signal=[];analog=[];nav=[];charge=[]
    for i in ids:
        tree.GetEntry(int(i));signal.append(np.array(tree.signal_mV));analog.append(np.array(tree.analog_mV))
        nav.append(len(tree.avalanche_q_C));charge.append(sum(tree.avalanche_q_C))
    froot.Close();r=info['config']['sipm']['readout'];lsb=(r['adc_max_mV']-r['adc_min_mV'])/2**r['adc_bits']
    np.savez_compressed(dest/'arrays'/f"{entry['key']}.npz",ids=ids,q=cf['charge_long'],scores=scores,
        signal=np.asarray(signal)/lsb,base_analog=np.asarray(analog)/lsb,waves=waves[ids],
        anchor=np.floor(cf['t10_sample']).astype(int)+1,baseline=cf['baseline'],baseline_rms=cf['baseline_rms'],
        metrics=fluctuation_metrics(waves[ids],cf['baseline'],cf['t10_sample'])[:,:,3],shapes=shapes,
        n_avalanches=nav,charge_C=charge,energy=energy[ids],truth=label_truth(energy[ids])[0],
        all_q=features['charge_long'],all_selected=selected,all_common=common,all_energy=energy,
        all_amplitude=features['amplitude'],n_parent=n)
    meta={'key':entry['key'],'n_parent':n,'n_selected':int(selected.sum()),'n_common':len(ids),
      'root_sha256':sha(path),'replay':info.get('response_replay'),
      'source':entry['source'],'common_avalanches_p10_p50_p90':np.quantile(nav,[.1,.5,.9]).tolist(),
      'common_truth_counts':np.bincount(label_truth(energy[ids])[0],minlength=5).tolist(),
      'common_Q_bins':np.histogram(cf['charge_long'],edges)[0].tolist(),
      'code_sha256':sha(Path(__file__))}
    save(dest/'extraction'/f"{entry['key']}.json",meta)
    print(entry['key'],'common',len(ids),'avalanches',meta['common_avalanches_p10_p50_p90'],flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--keys',nargs='+',required=True)
    p.add_argument('--stage',default='profile',choices=['profile','validation','calibration_check']);args=p.parse_args()
    protocol=read(OUT/('protocol.json' if args.stage=='profile' else args.stage+'/protocol.json'))
    dest=OUT/args.stage
    for folder in ['arrays','extraction']:(dest/folder).mkdir(exist_ok=True)
    pca=read(PROJECT/'analysis/paper_baseline/results/publication_20261028/point.json')['windows']['3000']['pca']
    for key in args.keys:
        assert not (dest/'extraction'/f'{key}.json').exists(),'Refuse overwrite'
        extract(protocol['entries'][key],dest,protocol['q_edges'],pca)


if __name__=='__main__':main()
