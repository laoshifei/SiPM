"""Read the existing actual ADC without generating or rescaling any waveform."""
import argparse
import concurrent.futures
import ROOT
from time_window_common import *
from reference import bin_layout,wave_features


def attach(a,p,source):
    q=a.get('qcalc',a['q']);bins=charge_bin(q,p)-1
    active=np.isin(bins,p['support_bins'][source])&a.get('selected',np.ones(len(q),bool))
    run=dict(SOURCES)[source];ref=read(QDC/'references'/f'run{run}_last20_B128.json');mu0=np.full(len(q),np.nan)
    for ef in [0,1]:
        for b in p['support_bins'][source]:mu0[(bins==b)&(a['fold']==ef)]=ref[str(ef)]['Q_bins'][str(b)]['pulse_mean']
    return {**{k:a[k] for k in ['ids','fold']},'qcalc':q,'active':active,'mu_reference':mu0,
        'T':np.full((len(q),3,7),np.nan),'local_F':np.full((len(q),3,4),np.nan),'mu':np.full(len(q),np.nan),
        'window_mu':np.full((len(q),4),np.nan),'F':np.full((len(q),3),np.nan)}


def fill(out,ix,waves,old):
    f=wave_features(waves);d=decompose(waves,f['baseline'],f['t10_sample'],out['mu_reference'][ix])
    np.testing.assert_allclose(d['F'],old['X'][ix,3:6],rtol=1e-9,atol=1e-9)
    for name,value in d.items():out[name][ix]=value


def one(source):
    p=read(OUT/'protocol.json');run=dict(SOURCES)[source];records={}
    old=load(REPLAY/'point_arrays'/f'run{run}_last20_B128_data.npz');data=attach(old,p,source)
    path=PROJECT/f'data/DT5730_DPP_PHA/DAQ/run_{run}/UNFILTERED/Data_run_{run}.BIN';_,dtype,n=bin_layout(path)
    raw=np.memmap(path,dtype=dtype,offset=2,mode='r',shape=(n,));ids=np.flatnonzero(data['active'])
    for start in range(0,len(ids),128):
        ix=ids[start:start+128];fill(data,ix,np.array(raw['data'][old['ids'][ix]]),old)
    dest=OUT/'arrays'/f'{source}_data.npz';atomic_npz(dest,**data);records['data']={'array_sha256':sha(dest),'active':int(data['active'].sum()),'nonpositive_local_mu':int(np.count_nonzero(data['window_mu'][data['active']]<=0))}
    for variant in VARIANTS:
        old=load(SPLIT/'evaluation'/f'{source}_{variant}.npz');out=attach(old,p,source)
        for b in range(4):
            ix=np.flatnonzero(out['active']&(out['ids']//EVENTS[source]==b));root=ROOT.TFile.Open(str(SPLIT/'native'/f'b{b:02d}_run{run}.root'));tree=root.Get('Readout')
            tree.SetBranchStatus('*',0);tree.SetBranchStatus('adc_'+variant,1)
            for start in range(0,len(ix),128):
                at=ix[start:start+128];waves=[]
                for index in at:
                    assert tree.GetEntry(int(out['ids'][index]%EVENTS[source]))>0;waves.append(np.asarray(getattr(tree,'adc_'+variant)).copy())
                fill(out,at,np.asarray(waves,np.uint16),old)
            root.Close()
        dest=OUT/'arrays'/f'{source}_{variant}.npz';atomic_npz(dest,**out)
        records[variant]={'array_sha256':sha(dest),'active':int(out['active'].sum()),'nonpositive_local_mu':int(np.count_nonzero(out['window_mu'][out['active']]<=0))}
    save(OUT/'evaluation'/f'{source}_extraction.json',records);print(source,'actual ADC window decomposition complete',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=2);args=parser.parse_args();p=read(OUT/'protocol.json')
    for name,h in {**p['code_sha256'],**p['inherited_code_sha256']}.items():assert sha(PROJECT/name)==h,name
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        for result in executor.map(one,[s for s,_ in SOURCES]):pass
