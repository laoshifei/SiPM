"""Reapply the inherited software cuts to every full-pool ADC record."""
import argparse
import concurrent.futures
import json
import ROOT
from full_selection_common import *
from compare import mc_wave_features
from reference import wave_features
from refine_waveforms import advanced_features
from extract_pid import label_truth


def quality(waves,fold):
    parts={}
    for start in range(0,len(waves),512):
        for k,v in wave_features(waves[start:start+512]).items():parts.setdefault(k,[]).append(v)
    f={k:np.concatenate(v) for k,v in parts.items()};reference=float(np.median(f['baseline_rms']))
    fold_ref=np.array([np.median(f['baseline_rms'][fold==j]) for j in [0,1]])
    pre=[];alternate=[]
    for start in range(0,len(waves),512):
        sl=slice(start,start+512);args=(waves[sl],f['t10_sample'][sl],f['peak_sample'][sl],f['amplitude'][sl],f['baseline'][sl])
        pre.append(advanced_features(*args,reference)['pretrigger_flag_primary'])
        alternate.append(advanced_features(*args,fold_ref[fold[sl]])['pretrigger_flag_primary'])
    good=(f['diagnostic_flags']==0)&~np.concatenate(pre)
    alternate=(f['diagnostic_flags']==0)&~np.concatenate(alternate)
    return f,good,alternate,reference,fold_ref


def one(key):
    p=read(OUT/'protocol.json');e=p['entries'][key];dest=OUT/'logs'/f'{key}_extraction.json'
    assert not dest.exists(),'Refuse extraction overwrite'
    native=read(OUT/'logs'/f'{key}_native.json');assert sha(PROJECT/e['output'])==native['output_sha256']
    froot=ROOT.TFile.Open(str(PROJECT/e['output']));tree=froot.Get('Readout');n=e['events']
    assert tree.GetEntries()==n
    tree.SetBranchStatus('*',0)
    for name in ['event_id','fold','edep_by_origin_MeV']:tree.SetBranchStatus(name,1)
    folds=[];energy=[]
    for i,row in enumerate(tree):
        assert row.event_id==i;folds.append(row.fold);energy.append(np.asarray(row.edep_by_origin_MeV).copy())
    fold=np.asarray(folds,dtype=np.int8);energy=np.asarray(energy)
    np.testing.assert_array_equal(fold,read(PROJECT/e['plan'])['folds'])
    truth=label_truth(energy)[0]
    old=load(PARENT/'transport/arrays'/f'{key}.npz');old_ids=old['ids'];reports={};old_common=np.zeros(n,bool);old_common[old_ids]=True
    for variant in ['original',*VARIANTS]:
        branch=f'adc_{variant}';tree.SetBranchStatus('*',0);tree.SetBranchStatus(branch,1)
        waves=np.empty((n,2496),np.uint16)
        for i,row in enumerate(tree):waves[i]=np.asarray(getattr(row,branch))
        f,good,alternate,ref,fold_ref=quality(waves,fold)
        selected=good&(f['amplitude']>=200)&(f['t10_sample']>=384)
        selected_fold=alternate&(f['amplitude']>=200)&(f['t10_sample']>=384)
        bins=charge_bin(f['charge_long'],p)
        common=selected&(bins>=1)&(bins<=6)&np.isfinite(f['charge_long'])
        supported=common&np.isin(bins,np.asarray(p['support_bins'][e['source']])+1)
        if variant=='original':
            np.testing.assert_array_equal(f['charge_long'],old['all_q'])
            np.testing.assert_array_equal(selected,old['all_selected'])
            np.testing.assert_array_equal(common,old['all_common'])
            np.testing.assert_array_equal(energy,old['all_energy'])
        # Preserve old cohort even when current QC fails; no accidental cleanup
        # of the very selection effect being audited. Invalid features stay NaN.
        wanted=old_common|common
        eligible=np.isfinite(f['t10_sample'])&(f['t10_sample']>=384)&(np.floor(f['t10_sample'])+1501<=2496)&(f['charge_long']>0)
        feature_ids=np.flatnonzero(wanted&eligible)
        X=np.full((n,6),np.nan);profile=np.full((n,76),np.nan)
        for start in range(0,len(feature_ids),512):
            ix=feature_ids[start:start+512];a=features(waves[ix],ix,f['charge_long'][ix])
            np.testing.assert_array_equal(a['qcalc'],f['charge_long'][ix])
            X[ix]=np.c_[a['scores'],a['F']];profile[ix]=a['mean_wave']
        path=OUT/'arrays'/f'{key}_{variant}.npz'
        np.savez_compressed(path,ids=np.arange(n),fold=fold,energy=energy,truth=truth,
            q=old['all_q'],qcalc=f['charge_long'],X=X,mean_wave=profile,
            quality=good,selected=selected,selected_fold=selected_fold,common=common,supported=supported,
            old_common=old_common,feature_valid=eligible,baseline=f['baseline'],baseline_rms=f['baseline_rms'],
            amplitude=f['amplitude'],t10=f['t10_sample'],flags=f['diagnostic_flags'],n_parent=n)
        reports[variant]={'array_sha256':sha(path),'all_mothers':n,'quality':int(good.sum()),
            'selected':int(selected.sum()),'common':int(common.sum()),'supported':int(supported.sum()),
            'global_baseline_RMS':ref,'fold_baseline_RMS':fold_ref.tolist(),
            'fold_QC_disagreement':int(np.count_nonzero(selected!=selected_fold)),
            'invalid_wanted_features':int(np.count_nonzero(wanted&~eligible)),
            'common_truth_counts':np.bincount(truth[common],minlength=5).tolist(),
            'supported_truth_counts':np.bincount(truth[supported],minlength=5).tolist(),
            'common_bins':np.bincount(bins[common],minlength=8).tolist()}
        print(key,variant,'complete selection extraction',flush=True)
    froot.Close();save(dest,{'entry':key,'variants':reports,'native_sha256':native['output_sha256'],
        'code_sha256':sha(__file__),'original_selection_closure':True})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=2);parser.add_argument('--keys',nargs='+')
    args=parser.parse_args();p=read(OUT/'protocol.json')
    for name,h in {**p['code_sha256'],**p['inherited_code_sha256']}.items():assert sha(PROJECT/name)==h,name
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        for result in executor.map(one,args.keys or list(p['entries'])):pass
