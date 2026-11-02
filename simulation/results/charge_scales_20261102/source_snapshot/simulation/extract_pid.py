"""Extract fixed ADC PID scores and ancestry labels; never infer measured truth."""
import argparse
import json
from pathlib import Path
import numpy as np
import ROOT
from reference import PROJECT, wave_features
from compare import mc_wave_features
from refine_waveforms import advanced_features
from publication_stats import sha, save
from prepare_pid import OUT, PARENT, SOURCES, STATES, read

NAMES=('alpha_family','gamma_family','decay_electron_family','mixed_other','no_deposit')


def load(path):
    with np.load(path) as z:return dict(z)


def label_truth(contributions,dominance=.9):
    x=np.asarray(contributions,float)
    if x.ndim!=2 or x.shape[1]!=6 or not .5<dominance<=1:raise ValueError('Invalid truth definition')
    total=x.sum(1);fractions=np.divide(x,total[:,None],out=np.zeros_like(x),where=total[:,None]>0)
    labels=np.full(len(x),3,dtype=np.int8)
    for category,origin in enumerate([1,2,3]):labels[fractions[:,origin]>=dominance]=category
    labels[total<=0]=4
    return labels,fractions


def score_waves(waves,pca,f=None):
    if f is None:f=wave_features(waves)
    advanced=advanced_features(waves,f['t10_sample'],f['peak_sample'],f['amplitude'],f['baseline'],
        float(np.median(f['baseline_rms'])))
    q=advanced['long_charge'][:,2];ratio=advanced['short_charge'][:,2]/q
    np.testing.assert_allclose(q,f['charge_long'],rtol=1e-12,atol=1e-7)
    np.testing.assert_allclose(ratio,f['short_long_ratio'],rtol=1e-12,atol=1e-12)
    blocks=advanced['block_charge'][:,:pca['blocks']].astype(float)
    scores=np.column_stack([ratio,-advanced['mean_time_us'][:,2],
        ((blocks/q[:,None]-np.array(pca['center']))@np.array(pca['vector']))/pca['score_scale']])
    assert np.isfinite(scores).all()
    return scores


def profile(config):
    return [config[k] for k in ['sipm','electronics_noise','signal_dependent_noise','readout_response']]


def extract_mc(key,entry,protocol,pca,splits):
    path=PROJECT/entry['path'];froot=ROOT.TFile.Open(str(path));tree=froot.Get('Events')
    info=json.loads(str(froot.Get('RunInfo').GetString().Data()))
    count=int(tree.GetEntries());source=info['source']
    assert info['completed'] and info['schema_version']==9 and count==info['events_requested']
    assert count==len(splits[source])
    config=read(PROJECT/entry['scenario_config'])
    assert sha(PROJECT/entry['scenario_config'])==entry['scenario_config_sha256']
    assert profile(info['config'])==profile(config)
    for field in ['optical','geometry','acquisition']:assert info['config'][field]==config[field]
    if 'response_replay' in info:
        assert info['response_replay']['equal_original_avalanche_events']==count
        assert info['response_replay']['max_relative_charge_balance']<1e-10
    tree.SetBranchStatus('*',0)
    branches=['event_id','adc','edep_MeV','edep_by_origin_MeV','charge_by_origin_C']
    for branch in branches:tree.SetBranchStatus(branch,1)
    waves=np.empty((count,info['config']['acquisition']['samples']),dtype=np.uint16)
    energy=np.empty((count,6));charge=np.empty((count,6));edep=np.empty(count)
    for i,event in enumerate(tree):
        assert event.event_id==i
        waves[i]=np.array(event.adc);energy[i]=np.array(event.edep_by_origin_MeV)
        charge[i]=np.array(event.charge_by_origin_C);edep[i]=event.edep_MeV
    froot.Close()
    np.testing.assert_allclose(energy.sum(1),edep,rtol=1e-10,atol=1e-12)
    f,quality=mc_wave_features(waves)
    selected=quality&(f['amplitude']>=200)&(f['t10_sample']>=384)
    ids=np.flatnonzero(selected);sub={k:v[ids] for k,v in f.items()}
    scores=np.full((count,3),np.nan);scores[ids]=score_waves(waves[ids],pca,sub)
    edges=np.array(protocol['q_edges']);common=selected&(f['charge_long']>=edges[0])&(f['charge_long']<=edges[-1])
    labels,fraction=label_truth(energy);charge_labels,charge_fraction=label_truth(charge)
    result={'ids':np.arange(count),'q':f['charge_long'],'scores':scores,'amplitude':f['amplitude'],
        'baseline':f['baseline'],'baseline_rms':f['baseline_rms'],'t10':f['t10_sample'],
        'selected':selected,'common':common,'quality':quality,'split':splits[source],
        'truth':labels,'truth_charge':charge_labels,'energy_by_origin':energy,
        'charge_by_origin':charge,'energy_fractions':fraction,'charge_fractions':charge_fraction}
    np.savez_compressed(OUT/'arrays'/f'{key}.npz',**result)
    counts={}
    for name,mask in [('all_mothers',np.ones(count,bool)),('selected',selected),('common_Q',common),
          ('common_train',common&(splits[source]==0)),('common_test',common&(splits[source]==1))]:
        counts[name]={'n':int(mask.sum()),'truth_counts':dict(zip(NAMES,np.bincount(labels[mask],minlength=5).tolist()))}
    row={'source':source,'state':key.split('_')[0],'ROOT_path':entry['path'],'ROOT_sha256':sha(path),
        'nparent':count,'seed':info['seed'],'kind':entry['kind'],'counts':counts,
        'energy_charge_label_disagreement_common':int(np.sum(common&(labels!=charge_labels))),
        'replay_checks':info.get('response_replay'),
        'scope':'Transport/charge conservation inherited from verified parent; all-ancestry energy sum rechecked; exact avalanche identity required for each replay.'}
    print(key,counts['common_Q'],flush=True)
    return row


def extract_measured(protocol,pca):
    old=PROJECT/'simulation/results/signal_fluctuations_20261028';record={}
    for run in [3,12,18]:
        poolpath=PROJECT/f'analysis/paper_baseline/results/publication_20261028/pool_run{run}.npz'
        with np.load(poolpath) as z:pool={k:z[k] for k in ['ids','primary','split','original_count']}
        for split,segment in enumerate(['first60','middle20','last20']):
            key=f'run{run}_{segment}'
            path=(PARENT/'run18' if run==18 else old)/(key+'.npz');d=load(path)
            np.testing.assert_array_equal(d['ids'],pool['ids'][pool['primary']&(pool['split']==split)])
            f=wave_features(d['waves']);np.testing.assert_allclose(f['charge_long'],d['q'],rtol=0,atol=1e-7)
            scores=score_waves(d['waves'],pca,f)
            np.savez_compressed(OUT/'arrays'/f'measured_{key}.npz',ids=d['ids'],q=d['q'],scores=scores,
                amplitude=f['amplitude'],baseline_rms=f['baseline_rms'],original_count=pool['original_count'])
            record[key]={'n':len(d['ids']),'input':str(path.relative_to(PROJECT)),'input_sha256':sha(path),
                'pool_sha256':sha(poolpath),'particle_truth':'Not available; source provenance only.'}
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--keys',nargs='*')
    args=parser.parse_args();protocol=read(OUT/'protocol.json')
    (OUT/'arrays').mkdir(exist_ok=True);(OUT/'extraction').mkdir(exist_ok=True)
    point=PROJECT/'analysis/paper_baseline/results/publication_20261028/point.json'
    assert sha(point)==protocol['inputs_sha256'][str(point.relative_to(PROJECT))]
    pca=read(point)['windows']['3000']['pca'];splits=load(OUT/'mother_splits.npz')
    for key,entry in protocol['source_rows'].items():
        if args.keys is not None and key not in args.keys:continue
        dest=OUT/'extraction'/f'{key}.json'
        assert not dest.exists(),f'Refuse to overwrite extracted input {key}'
        row=extract_mc(key,entry,protocol,pca,splits)
        row.update(code_sha256=sha(Path(__file__)),protocol_sha256=sha(OUT/'protocol.json'))
        save(dest,row)
    if args.keys is None:
        save(OUT/'extraction/measured.json',{'code_sha256':sha(Path(__file__)),'groups':extract_measured(protocol,pca),
            'PCA_path':str(point.relative_to(PROJECT)),'PCA_sha256':sha(point),'PCA_refit':False})


if __name__=='__main__':main()
