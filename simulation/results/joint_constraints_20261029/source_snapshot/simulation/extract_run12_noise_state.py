"""Extract reselected and fixed-mother ADC diagnostics for noise-state checks."""
import argparse,json
from pathlib import Path
import numpy as np
import ROOT
from prepare_run12_noise_state import OUT,PARENT,PROJECT
from prepare_pid import read
from extract_pid import load,score_waves,label_truth
from reference import wave_features,aligned_shapes
from compare import mc_wave_features
from diagnose_fluctuations import fluctuation_metrics
from diagnose_pid_bandwidth import linear_scores
from analyze_npe_validation import raw_shapes
from publication_stats import save,sha


def extract(key):
    assert not (OUT/'extraction'/f'{key}.json').exists(),'Refuse overwrite'
    protocol=read(OUT/'protocol.json');e=protocol['entries'][key];edges=np.asarray(protocol['q_edges'])
    pca=read(PROJECT/'analysis/paper_baseline/results/publication_20261028/point.json')['windows']['3000']['pca']
    original=load(PARENT/'validation/arrays/run12_Am241.npz');parent_ids=original['ids']
    root=ROOT.TFile.Open(str(PROJECT/e['path']));t=root.Get('Events');info=json.loads(str(root.Get('RunInfo').GetString().Data()))
    n=int(t.GetEntries());assert n==e['events'] and info['completed'] and info['seed']==e['seed']
    assert info['config']==read(PROJECT/e['config'])
    if not e['parent_existing']:
        assert info['response_replay']['equal_original_avalanche_events']==n
        assert info['response_replay']['max_relative_charge_balance']<1e-9
    t.SetBranchStatus('*',0)
    for b in ['event_id','adc','edep_by_origin_MeV']:t.SetBranchStatus(b,1)
    waves=np.empty((n,2496),np.uint16);energy=np.empty((n,6))
    for i,event in enumerate(t):
        assert event.event_id==i;waves[i]=np.asarray(event.adc);energy[i]=np.asarray(event.edep_by_origin_MeV)
    np.testing.assert_array_equal(energy,original['all_energy'])
    f,quality=mc_wave_features(waves)
    selected=quality&(f['amplitude']>=200)&(f['t10_sample']>=384)
    common=selected&(f['charge_long']>=edges[0])&(f['charge_long']<=edges[-1])
    fixed=np.zeros(n,bool);fixed[parent_ids]=True;ids=np.flatnonzero(common|fixed)
    cf={k:v[ids] for k,v in f.items()};anchor=np.floor(cf['t10_sample']).astype(int)+1
    assert np.all((anchor>=20)&(anchor+1500<=2496))
    scores=score_waves(waves[ids],pca,cf)
    shapes,keep=aligned_shapes(waves[ids],cf);assert len(keep)==len(ids)
    metrics=fluctuation_metrics(waves[ids],cf['baseline'],cf['t10_sample'])[:,:,3]
    fixed_anchor_scores=linear_scores(waves[parent_ids],original['anchor'],f['baseline'][parent_ids],pca)[0]
    ro=info['config']['sipm']['readout'];lsb=(ro['adc_max_mV']-ro['adc_min_mV'])/2**ro['adc_bits']
    t.SetBranchStatus('*',0);t.SetBranchStatus('signal_mV',1)
    for j,i in enumerate(parent_ids):
        t.GetEntry(int(i));np.testing.assert_array_equal(np.asarray(t.signal_mV)/lsb,original['signal'][j])
    root.Close()
    np.savez_compressed(OUT/'arrays'/f'{key}.npz',ids=ids,waves=waves[ids],q=cf['charge_long'],scores=scores,
        common=common[ids],fixed=fixed[ids],original_q=original['all_q'][ids],anchor=anchor,
        baseline=cf['baseline'],baseline_rms=cf['baseline_rms'],metrics=metrics,shapes=shapes,raw_shapes=raw_shapes(waves[ids]),
        fixed_ids=parent_ids,fixed_anchor_scores=fixed_anchor_scores,
        all_q=f['charge_long'],all_common=common,all_selected=selected,all_energy=energy,
        all_baseline=f['baseline'],all_baseline_rms=f['baseline_rms'],truth=label_truth(energy[ids])[0])
    meta={'key':key,'n_parent':n,'common_n_all_mothers':int(common.sum()),'fixed_n':len(parent_ids),
        'newly_selected':int(np.sum(common&~fixed)),'lost_from_original':int(np.sum(fixed&~common)),
        'all_energy_equal_parent':True,'all_replay_avalanches_equal_parent':e['parent_existing'] or info['response_replay']['equal_original_avalanche_events']==n,
        'exact_signal_preserved_checked_parent_ids':len(parent_ids),'replay':info.get('response_replay'),
        'root_sha256':sha(PROJECT/e['path']),'config_sha256':sha(PROJECT/e['config']),
        'code_sha256':sha(Path(__file__)),'protocol_sha256':sha(OUT/'protocol.json')}
    save(OUT/'extraction'/f'{key}.json',meta)
    print(key,'common',common.sum(),'gained/lost',meta['newly_selected'],meta['lost_from_original'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--key',required=True);extract(p.parse_args().key)
