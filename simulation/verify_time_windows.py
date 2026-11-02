"""Independent raw convolution, weighted moments and provenance checks."""
import argparse
import csv
import shutil
import sys
from pathlib import Path
import nbformat
import ROOT
from time_window_common import *
from reference import bin_layout,wave_features


def direct(wave,mu0):
    f=wave_features(wave[None,:]);y=wave.astype(float)-f['baseline'][0];anchor=f['t10_sample'][0]
    pre=int(np.ceil(anchor-300));post=int(np.ceil(anchor+250));window_mu=y[post:post+1000].reshape(4,250).mean(1);mu=window_mu.mean()
    result=[];local=[]
    for width in [25,51,101]:
        kernel=np.full(width,-1/width);kernel[width//2]+=1;filtered=np.convolve(y,kernel,mode='same')
        pieces=[filtered[post+250*j:post+250*(j+1)] for j in range(4)]
        means=np.array([np.mean(z) for z in pieces]);variances=np.array([np.var(z,ddof=1) for z in pieces]);before=np.var(filtered[pre:pre+200],ddof=1)
        between=250*np.sum((means-means.mean())**2)/999
        fixed=np.r_[249*variances/999,between,-before]/mu0
        whole=(np.var(filtered[post:post+1000],ddof=1)-before)/mu
        result.append(np.r_[fixed,whole-fixed.sum()]);local.append((variances-before)/window_mu)
    return np.array(result),np.array(local)


def verify():
    assert not (OUT/'verification.json').exists();p=read(OUT/'protocol.json');ap=read(OUT/'analysis_protocol.json');analysis=read(OUT/'analysis.json')
    for path,h in {**p['inputs_sha256'],**p['code_sha256'],**p['inherited_code_sha256'],**ap['arrays_sha256']}.items():assert sha(PROJECT/path)==h,path
    assert ap['code_sha256']==sha(PROJECT/'simulation/analyze_time_windows.py')
    raw_checks=events=weighted=0;valid={}
    for source,run in SOURCES:
        previous=read(SPLIT/'analysis.json')[source];counts=load(SPLIT/'evaluation'/f'{source}_counts.npz');group=analysis[source]
        receipt=read(OUT/'evaluation'/f'{source}_extraction.json');old_draws=load(SPLIT/'evaluation'/f'{source}_bootstrap.npz')['contrasts'];draws=load(OUT/'evaluation'/f'{source}_bootstrap.npz')['contrasts'];valid[source]={}
        for vi,v in enumerate(['data',*VARIANTS]):
            path=OUT/'arrays'/f'{source}_{v}.npz';a=load(path);assert sha(path)==receipt[v]['array_sha256'];active=a['active'];ix=np.flatnonzero(active);events+=len(ix)
            old=load((REPLAY/'point_arrays'/f'run{run}_last20_B128_data.npz') if v=='data' else SPLIT/'evaluation'/f'{source}_{v}.npz')
            np.testing.assert_array_equal(a['ids'],old['ids']);np.testing.assert_allclose(a['T'][active].sum(2),old['X'][active,3:6],rtol=1e-9,atol=1e-9)
            assert np.isfinite(a['local_F'][active]).all() and receipt[v]['nonpositive_local_mu']==0
            # Independent raw ADC check: direct convolution, not cumulative-sum filtering.
            rng=np.random.default_rng(731103900+run+100*vi);sample=np.unique(np.r_[ix[:12],rng.choice(ix,min(64,len(ix)),replace=False)])
            waves={}
            if v=='data':
                rawpath=PROJECT/f'data/DT5730_DPP_PHA/DAQ/run_{run}/UNFILTERED/Data_run_{run}.BIN';_,dtype,n=bin_layout(rawpath);raw=np.memmap(rawpath,dtype=dtype,offset=2,mode='r',shape=(n,))
                waves={int(i):np.array(raw['data'][a['ids'][i]]) for i in sample}
            else:
                for batch in range(4):
                    take=sample[a['ids'][sample]//EVENTS[source]==batch];root=ROOT.TFile.Open(str(SPLIT/'native'/f'b{batch:02d}_run{run}.root'));tree=root.Get('Readout');tree.SetBranchStatus('*',0);tree.SetBranchStatus('adc_'+v,1)
                    for i in take:assert tree.GetEntry(int(a['ids'][i]%EVENTS[source]))>0;waves[int(i)]=np.asarray(getattr(tree,'adc_'+v)).copy()
                    root.Close()
            for i,wave in waves.items():
                T,local=direct(wave,a['mu_reference'][i]);np.testing.assert_allclose(T,a['T'][i],rtol=1e-8,atol=1e-9);np.testing.assert_allclose(local,a['local_F'][i],rtol=1e-8,atol=1e-9);raw_checks+=1
            # Independent centered covariance, including explicit bootstrap replicas.
            inherited=counts['data_counts'] if v=='data' else counts['MC_counts'];multiplicities=np.vstack([np.ones((1,len(a['ids'])),int),inherited[[0,127,511]]])
            check,_=moments(a,p,source,multiplicities)
            for d,mult in enumerate(multiplicities):
                means=[];covs=[];lmeans=[];lcovs=[]
                for _,at,h,target in cells(a,p,source):
                    inds=np.repeat(at,mult[at]);half=np.repeat(h,mult[at]);w=(target/np.bincount(half,minlength=2))[half]
                    t=a['T'][inds];mean=np.einsum('n,nki->ki',w,t);center=t-mean
                    cov=np.einsum('n,nki,nkj->kij',w,center,center);local=a['local_F'][inds];lm=np.einsum('n,nki->ki',w,local);lr=local-lm
                    means.append(mean);covs.append(cov);lmeans.append(lm);lcovs.append(np.einsum('n,nki,nkj->kij',w,lr,lr));weighted+=1
                for k,values in [('mean',means),('cov',covs),('local_mean',lmeans),('local_cov',lcovs)]:np.testing.assert_allclose(np.mean(values,0),check[k][d],rtol=1e-9,atol=1e-10)
            for k in ['mean','cov','local_mean','local_cov']:np.testing.assert_allclose(check[k][0],group['summaries'][v][k],rtol=1e-12,atol=1e-12)
        for j,v in enumerate(VARIANTS):
            valid[source][v]={}
            for k,scale in enumerate(SCALES):
                previous_index=read(SPLIT/'analysis_protocol.json')['metrics'].index(f'F{scale}_SD_ratio');np.testing.assert_allclose(draws[:,j,k,0],old_draws[:,[2,5][j],previous_index],rtol=1e-8,atol=1e-9)
                assert {n:interval(draws[:,j,k,i]) for i,n in enumerate(NAMES)}==group['variants'][v][str(scale)]['CI95']
                valid[source][v][str(scale)]=int(np.isfinite(draws[:,j,k]).all(1).sum())
        print(source,'independent raw filters, temporal moments and inherited draws verified',flush=True)
    nbpath=PROJECT/'jupyter/codex_0001_csi_time_resolved_fluctuations.ipynb';nb=nbformat.read(nbpath,as_version=4);nbformat.validate(nb);code=[c for c in nb.cells if c.cell_type=='code']
    assert all(c.execution_count is not None for c in code) and all(o.output_type!='error' for c in code for o in c.outputs)
    images=sum('image/png' in o.get('data',{}) for c in code for o in c.outputs);assert images==5
    tables={name:len(list(csv.DictReader((OUT/name).open()))) for name in ['attribution_table.csv','local_window_table.csv']};assert list(tables.values())==[12,48]
    assert read(OUT/'decision.json')['code_sha256']==sha(PROJECT/'simulation/report_time_windows.py')
    save(OUT/'verification.json',{'inputs_verified':len(p['inputs_sha256']),'events_exact_F_reconstructed':events,'independent_raw_ADC_events':raw_checks,
        'explicit_weighted_moment_cells':weighted,'bootstrap_valid_draws':valid,'table_rows':tables,'executed_code_cells':len(code),'embedded_PNG':images,
        'notebook_sha256':sha(nbpath),'code_sha256':sha(__file__),'new_native_readouts':0,'new_independent_mothers':0,'default_changed':False,'official_PID_changed':False})


def archive():
    assert (OUT/'verification.json').exists() and not (OUT/'record.json').exists();p=read(OUT/'protocol.json');names=set(p['code_sha256'])|set(p['inherited_code_sha256'])
    names.update(['simulation/report_time_windows.py','simulation/verify_time_windows.py','simulation/README.md','jupyter/codex_0001_build_time_window_notebook.py'])
    for module in list(sys.modules.values()):
        name=getattr(module,'__file__',None)
        if name:
            path=Path(name).resolve()
            if path.is_relative_to(PROJECT) and path.suffix=='.py':names.add(relative(path))
    sources={}
    for name in sorted(names):
        path=OUT/'source_snapshot'/name;path.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(PROJECT/name,path);sources[name]=sha(path)
    outputs={relative(x):sha(x) for x in sorted(OUT.rglob('*')) if x.is_file() and 'source_snapshot' not in x.parts and x.name!='record.json'}
    nb=PROJECT/'jupyter/codex_0001_csi_time_resolved_fluctuations.ipynb';save(OUT/'record.json',{'outputs_sha256':outputs,'sources_snapshot_sha256':sources,'notebook':relative(nb),'notebook_sha256':sha(nb),
        'new_native_readouts':0,'new_independent_mothers':0,'default_changed':False,'official_PID_changed':False});print('Archived',len(outputs),'outputs and',len(sources),'source snapshots.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--archive',action='store_true');args=parser.parse_args();archive() if args.archive else verify()
