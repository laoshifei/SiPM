"""Extract current full-pool selection for native spectral-state candidates."""
import argparse
import concurrent.futures
import ROOT
from spectral_split_common import *
from extract_full_selection import quality


def one(key):
    p=read(OUT/'protocol.json');e=read(OUT/'native_protocol.json')['entries'][key]
    dest=OUT/'logs'/f'{key}_extraction.json';assert not dest.exists()
    record=read(OUT/'logs'/f'{key}_native.json');assert sha(PROJECT/e['output'])==record['output_sha256']
    froot=ROOT.TFile.Open(str(PROJECT/e['output']));tree=froot.Get('Readout');n=e['events'];assert tree.GetEntries()==n
    old=load(FULL/'arrays'/f'{key}_original.npz');fold=old['fold'];reports={}
    for variant in ['split_r0','split_r1']:
        tree.SetBranchStatus('*',0)
        for name in ['event_id','fold','adc_'+variant]:tree.SetBranchStatus(name,1)
        waves=np.empty((n,2496),np.uint16)
        for i,row in enumerate(tree):
            assert row.event_id==i and row.fold==fold[i];waves[i]=np.asarray(getattr(row,'adc_'+variant))
        f,good,alternate,ref,fold_ref=quality(waves,fold)
        selected=good&(f['amplitude']>=200)&(f['t10_sample']>=384)
        selected_fold=alternate&(f['amplitude']>=200)&(f['t10_sample']>=384)
        q=f['charge_long'];common=selected&(q>=p['q_edges'][0])&(q<=p['q_edges'][-1])
        from charge_scale_common import charge_bin
        supported=common&np.isin(charge_bin(q,p),np.asarray(p['support_bins'][e['source']])+1)
        ids=np.flatnonzero(common);X=np.full((n,6),np.nan);profile=np.full((n,76),np.nan)
        for start in range(0,len(ids),512):
            ix=ids[start:start+512];a=features(waves[ix],ix,q[ix]);np.testing.assert_array_equal(a['qcalc'],q[ix])
            X[ix]=np.c_[a['scores'],a['F']];profile[ix]=a['mean_wave']
        path=OUT/'arrays'/f'{key}_{variant}.npz'
        np.savez_compressed(path,ids=np.arange(n),fold=fold,q=old['q'],qcalc=q,X=X,mean_wave=profile,
            selected=selected,common=common,supported=supported,old_common=old['old_common'],truth=old['truth'],
            baseline=f['baseline'],baseline_rms=f['baseline_rms'],t10=f['t10_sample'],n_parent=n)
        reports[variant]={'array_sha256':sha(path),'mothers':n,'common':int(common.sum()),'supported':int(supported.sum()),
            'global_baseline_RMS':ref,'fold_baseline_RMS':fold_ref,'fold_QC_disagreement':int(np.count_nonzero(selected!=selected_fold)),
            'nonfinite_selected_features':int(np.count_nonzero(~np.isfinite(X[common]).all(1)))}
        print(key,variant,'current full selection extracted',flush=True)
    froot.Close();save(dest,serialize({'variants':reports,'native_sha256':record['output_sha256'],'code_sha256':sha(__file__)}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=2);parser.add_argument('--keys',nargs='+');args=parser.parse_args()
    p=read(OUT/'protocol.json')
    for n,h in {**p['code_sha256'],**p['inherited_code_sha256']}.items():assert sha(PROJECT/n)==h,n
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        for result in executor.map(one,args.keys or list(p['entries'])):pass
