"""Training-only calibration of an effective signal-dependent readout residual.

This models missing waveform covariance, not an identified avalanche mechanism.
No measured trace is copied into MC. Each acquisition calibration is separate;
source identity is not consumed by the native stochastic response module.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import ROOT
from reference import PROJECT, bin_layout, wave_features, aligned_shapes, TIME_NS
from diagnose_charge import sha
from diagnose_fluctuations import fluctuation_metrics
from screen_response_candidates import info_of, load
from compare import mc_wave_features

BASE=PROJECT/'analysis/paper_baseline/results/publication_20261028'
HERE=Path(__file__).resolve().parent
OUT=HERE/'results/signal_fluctuations_20261028'
PARENT=HERE/'results/response_candidates_20261028/control'
OFFSETS=np.array([-300,250,600,1100])
N=200


def save(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def diagnostic(waves,f,edges):
    anchor=np.ceil(f['t10_sample']).astype(int)
    index=anchor[:,None,None]+OFFSETS[None,:,None]+np.arange(N)
    signal=waves.astype(float)-f['baseline'][:,None]
    windows=np.take_along_axis(signal[:,None,:],index,axis=2)
    means=windows[:,1:].mean(axis=-1)
    if np.any(means<=0):raise ValueError('Nonpositive local signal')
    bins=np.minimum(np.searchsorted(edges,f['charge_long'],side='right')-1,5)
    cells=2*bins+(f['short_long_ratio']>=.4)
    taper=np.hanning(N);taper/=np.sqrt(np.mean(taper*taper))
    # Hann periodogram; subtract per-Q/shape conditional event mean in FFT space.
    centred=windows-windows.mean(axis=-1,keepdims=True)
    fft=np.fft.rfft(centred*taper,axis=-1)/np.sqrt(N)
    spectra=[];sizes=[];cell_rows=[]
    for cell in np.unique(cells):
        use=cells==cell;n=int(use.sum())
        if n<10:continue
        z=fft[use,1:]/np.sqrt(means[use])[:,:,None]
        pre=fft[use,:1]/np.sqrt(means[use])[:,:,None]
        ps=(np.abs(z-z.mean(axis=0))**2-np.abs(pre-pre.mean(axis=0))**2).mean(axis=0)*n/(n-1)
        spectra.append(ps);sizes.append(n)
        m=fluctuation_metrics(waves[use],f['baseline'][use],f['t10_sample'][use])
        cell_rows.append({'cell':int(cell),'n':n,'mean_signal_adc':float(m[:,0,0].mean()),
                          'F_mean':m[:,:,3].mean(axis=0).tolist()})
    return {'power':np.average(spectra,axis=0,weights=sizes),
            'frequency_MHz':np.fft.rfftfreq(N,2.)*1000,'cell_rows':cell_rows,
            'local_means':means.mean(axis=0).tolist(),'used_n':sum(sizes)}


def prepare(out):
    out.mkdir(parents=True,exist_ok=True)
    edges=np.array(json.loads((BASE/'protocol.json').read_text())['q_edges'])
    design={'scope':__doc__,'parent_publication_record_sha256':sha(BASE/'record.json'),
        'parent_robustness_record_sha256':sha(BASE.parent/'robustness_20261028/record.json'),
        'training':'Exactly frozen run3 and run12 original first60 IDs. Middle/last20 and run18 are previously explored internal checks, excluded from calibration.',
        'intent':'Calibrate extra stochastic covariance only; fixed Geant4 geometry, photons, optical emission, PDE, gain and average response. Native output remains in ADC counts; no measured waveform templates or charge-spectrum reweighting.',
        'model_status':'Effective acquisition-dependent residual; not evidence of source-dependent SiPM gain or a unique microscopic origin.',
        'targets':'Train frequency-dependent residual covariance, compare filter50/102/202ns, held-segment distributions, local windows, normalized means and unscaled charge spectrum. No claim that arbitrary measured events have a unique identical MC counterpart.',
        'code_sha256':sha(Path(__file__)),'q_edges':edges.tolist(),'inputs_sha256':{},'groups':{}}
    for run in (3,12,18):
        poolpath=BASE/f'pool_run{run}.npz'
        with np.load(poolpath) as z:pool=dict(z)
        fp=BASE.parent/f'pha_run{run:04d}_features.npz'
        with np.load(fp) as z:features=dict(z)
        rawpath=PROJECT/f'data/DT5730_DPP_PHA/DAQ/run_{run}/UNFILTERED/Data_run_{run}.BIN'
        _,dt,count=bin_layout(rawpath);raw=np.memmap(rawpath,dtype=dt,offset=2,mode='r',shape=(count,))
        for split,label in enumerate(('first60','middle20','last20')):
            if run==18 and split>0:continue
            ids=pool['ids'][pool['primary']&(pool['split']==split)]
            waves=np.array(raw['data'][ids]);f={k:v[ids] for k,v in features.items() if v.ndim and len(v)==count}
            shapes,kept=aligned_shapes(waves,f);np.testing.assert_array_equal(kept,np.arange(len(ids)))
            metrics=fluctuation_metrics(waves,f['baseline'],f['t10_sample'])
            d=diagnostic(waves,f,edges);power=d.pop('power');freq=d.pop('frequency_MHz')
            key=f'run{run}_{label}'
            np.savez_compressed(out/(key+'.npz'),ids=ids,waves=waves,baseline=f['baseline'],t10=f['t10_sample'],
              q=f['charge_long'],ratio=f['short_long_ratio'],shapes=shapes,metrics=metrics,power=power,frequency_MHz=freq)
            design['groups'][key]={'n':len(ids),'F':metrics[:,:,3].mean(axis=0).tolist(),**d}
            print(key,len(ids),'F',design['groups'][key]['F'],flush=True)
        design['inputs_sha256'][str(poolpath.relative_to(PROJECT))]=sha(poolpath)
        design['inputs_sha256'][str(fp.relative_to(PROJECT))]=sha(fp)
    for source,name in [('Co60','co60'),('Am241','am241')]:
        path=PARENT/name/'events.root';info=info_of(path)
        file=ROOT.TFile.Open(str(path));tree=file.Get('Events')
        tree.SetBranchStatus('*',0)
        for key in ('adc','signal_mV','analog_mV'):tree.SetBranchStatus(key,1)
        with np.load(BASE/'mc'/f'{source}_control.npz') as z:ids=z['ids']
        values={key:[] for key in ('adc','signal_mV','analog_mV')}
        for i in ids:
            assert tree.GetEntry(int(i))>0
            for key in values:values[key].append(np.array(getattr(tree,key),dtype=np.uint16 if key=='adc' else float))
        file.Close();a={k:np.array(v) for k,v in values.items()};f=wave_features(a['adc'])
        d=diagnostic(a['adc'],f,edges);power=d.pop('power');freq=d.pop('frequency_MHz')
        metrics=fluctuation_metrics(a['adc'],f['baseline'],f['t10_sample'])
        np.savez_compressed(out/(source+'_parent.npz'),ids=ids,**a,baseline=f['baseline'],t10=f['t10_sample'],q=f['charge_long'],metrics=metrics,power=power,frequency_MHz=freq)
        design['groups'][source+'_parent']={'n':len(ids),'F':metrics[:,:,3].mean(axis=0).tolist(),**d}
        design['inputs_sha256'][str(path.relative_to(PROJECT))]=sha(path)
        save(out/(source+'_base_config.json'),info['config'])
        print(source,'parent',len(ids),flush=True)
    save(out/'design.json',design)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['prepare'])
    parser.add_argument('--out',type=Path,default=OUT);args=parser.parse_args()
    prepare(args.out)
