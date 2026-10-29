"""Conditional uncertainty for native-vs-measured waveform feature medians."""
import argparse
import json
from pathlib import Path
import numpy as np
from refit_native_analysis import OLD, arrays, features
from reference import PROJECT
from diagnose_charge import sha
from calibrate_signal_fluctuations import save
import publication_stats as p


def weighted_median_sorted(values,counts):
    counts=np.asarray(counts,dtype=np.int64)
    total=int(counts.sum());assert total>0
    cumulative=counts.cumsum()
    i=np.searchsorted(cumulative,(total-1)//2+1)
    j=np.searchsorted(cumulative,total//2+1)
    return .5*(values[i]+values[j])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis',type=Path,required=True)
    args=parser.parse_args();folder=args.analysis.resolve();report=json.loads((folder/'validation.json').read_text())
    result={'scope':__doc__,'code_sha256':sha(Path(__file__)),'validation_sha256':sha(folder/'validation.json'),
      'method':'512 full-mother multinomial and measured raw-index block64 resamples, same seeds as F intervals; median of replicated event multiset. All fitted parameters held fixed.',
      'limits':'Conditional intervals exclude fitting and model selection; later-segment pretrigger calibration, where used, is also held fixed. No simultaneous multiplicity adjustment.',
      'sources':{}}
    for source,info in report['sources'].items():
        model=arrays(folder/f'{source}_native.npz');fm=features(model['waves']);mo={k:np.argsort(v) for k,v in fm.items()}
        groups={}
        for group,c in info['comparisons'].items():
            run=int(group.split('_')[0][3:]);segment=group.split('_')[1];split=['first60','middle20','last20'].index(segment)
            data=arrays(OLD/f'{group}.npz');fd=features(data['waves']);do={k:np.argsort(v) for k,v in fd.items()}
            nraw={3:4776,12:7516,18:156966}[run]
            lo,hi=[(0,int(.6*nraw)),(int(.6*nraw),int(.8*nraw)),(int(.8*nraw),nraw)][split]
            draws={k:[] for k in fm}
            for iteration in range(512):
                rng=np.random.default_rng(np.random.SeedSequence([281028321,run,split,iteration]))
                mother=np.bincount(rng.integers(info['nparent'],size=info['nparent']),minlength=info['nparent'])
                wm=mother[model['ids']];wd=p.block_counts(data['ids'],(lo,hi),64,rng)
                for key in fm:
                    a=weighted_median_sorted(fm[key][mo[key]],wm[mo[key]])
                    b=weighted_median_sorted(fd[key][do[key]],wd[do[key]])
                    draws[key].append(a-b)
            groups[group]={key:{'MC_median':float(np.median(fm[key])),'data_median':float(np.median(fd[key])),
                'median_difference':float(np.median(fm[key])-np.median(fd[key])),
                'difference_interval95':np.quantile(values,[.025,.975]).tolist()} for key,values in draws.items()}
            np.savez_compressed(folder/f'{source}_{group}_feature_draws.npz',**draws)
        result['sources'][source]=groups
    save(folder/'feature_intervals.json',result)
    print('Saved conditional median intervals.',flush=True)


if __name__=='__main__':main()
