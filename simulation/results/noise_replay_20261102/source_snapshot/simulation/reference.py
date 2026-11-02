"""Freeze measured waveform references using the existing phase-2 selection.

Only run 3/12 first 60% are calibration data. Later portions and run 18 remain
checks. These data have been explored previously, so checks are not blind tests.
"""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np

PROJECT=Path(__file__).resolve().parents[1]
BASE=PROJECT/'analysis/paper_baseline'
sys.path.insert(0,str(BASE))
from prepare import bin_layout, wave_features

TIME_NS=np.arange(-200.,3800.+1,2.)

def aligned_shapes(waves, features=None):
    f=wave_features(waves) if features is None else features
    result=[];keep=[]
    for i,w in enumerate(waves):
        if f['diagnostic_flags'][i] or not np.isfinite(f['t10_sample'][i]):continue
        t=(np.arange(len(w))-f['t10_sample'][i])*2
        if t[0]>TIME_NS[0] or t[-1]<TIME_NS[-1]:continue
        y=np.interp(TIME_NS,t,w-f['baseline'][i])
        integral=y[(TIME_NS>=-40)&(TIME_NS<3000)].sum()*2
        if integral<=0:continue
        result.append(y/integral);keep.append(i)
    return np.array(result),np.array(keep,dtype=int)

def main():
    output=PROJECT/'simulation/results/reference';output.mkdir(parents=True,exist_ok=True)
    arrays={'time_ns':TIME_NS};metadata={'normalization':'Each event divided by aligned integral [-40,3000) ns; units 1/ns',
        'selection':'Existing phase2 eligible_primary and frozen common charge; deterministic evenly spaced sample up to 1200 per group',
        'particle_labels':'Source names only; dominant Am shape is not per-event alpha truth','groups':{},'inputs_sha256':{}}
    for run in (3,12,18):
        file=PROJECT/f'data/DT5730_DPP_PHA/DAQ/run_{run}/UNFILTERED/Data_run_{run}.BIN'
        header,dtype,count=bin_layout(file)
        raw=np.memmap(file,dtype=dtype,offset=2,mode='r',shape=(count,))
        with np.load(BASE/f'results/phase2/run{run:04d}_features.npz') as z:phase={k:z[k] for k in z.files}
        with np.load(BASE/f'results/pha_run{run:04d}_features.npz') as z:features={k:z[k] for k in z.files}
        groups=[('train',0),('middle',1),('late',2)] if run!=18 else [('check',3)]
        for name,split in groups:
            mask=phase['eligible_primary']&phase['common_charge']&(phase['split']==split)
            idx=np.flatnonzero(mask)
            if len(idx)>1200:idx=idx[np.linspace(0,len(idx)-1,1200,dtype=int)]
            waves=np.asarray(raw['data'][idx],dtype=float)
            shapes,good=aligned_shapes(waves,{k:v[idx] for k,v in features.items()})
            key=f'run{run}_{name}';idx=idx[good];waves=waves[good]
            arrays[key+'_mean']=shapes.mean(axis=0)
            arrays[key+'_std']=shapes.std(axis=0,ddof=1)
            arrays[key+'_q16']=np.quantile(shapes,.16,axis=0)
            arrays[key+'_q84']=np.quantile(shapes,.84,axis=0)
            arrays[key+'_indices']=idx
            arrays[key+'_noise']=waves[:,:128]-waves[:,:128].mean(axis=1,keepdims=True)
            metadata['groups'][key]={'source':'Co60' if run==3 else 'Am241','n_available':int(mask.sum()),'n_used':len(shapes),
                'baseline_adc':float(np.median(features['baseline'][idx])),'baseline_rms_adc':float(np.median(features['baseline_rms'][idx])),
                'charge_adc_samples_median':float(np.median(features['charge_long'][idx])),
                'ratio_median':float(np.median(features['short_long_ratio'][idx])),
                'role':'calibration' if name=='train' else 'previously explored check'}
        source_record=next(r for r in json.loads((BASE/'results/phase2/source_sha256.json').read_text()) if r['run']==run)
        metadata['inputs_sha256'][str(file.relative_to(PROJECT))]=source_record['sha256']
    np.savez_compressed(output/'references.npz',**arrays)
    (output/'references.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(metadata['groups'],ensure_ascii=False,indent=2))

if __name__=='__main__':main()
