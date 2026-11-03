"""Independently check intensity conditioning with explicit replicas and weighted least squares."""
import argparse
import csv
import shutil
import sys
import nbformat
from intensity_condition_common import *


def direct(a,p,source,multiplicity):
    records=[];crossfit_valid=True
    for _,at,h,target in cells(a,p,source):
        ii=np.repeat(at,multiplicity[at]);hh=np.repeat(h,multiplicity[at]);fold=a['fold'][ii]
        number=np.bincount(hh,minlength=2);assert len(ii)>=5 and np.all((number>0)|(target==0))
        w=(target/number)[hh];x,y=design(a,ii);my=np.einsum('n,nsi->si',w,y);yc=y-my;mx=w@x
        c=np.einsum('n,nsi,nsj->sij',w,yc,yc);vx=w@((x-mx)**2)
        linear=[];residual=[];slopes=[];cfs=[];cfmeans=[]
        valid=True
        for ef in [0,1]:
            f=fold==ef;n=np.bincount(hh[f],minlength=2)
            valid &= f.sum()>=5 and bool(np.all((n>0)|(target==0)))
        crossfit_valid &= valid
        for j in range(5):
            A=np.c_[np.ones(len(x)),x[:,j]];beta=np.linalg.lstsq(np.sqrt(w)[:,None]*A,np.sqrt(w)[:,None]*y.reshape(len(y),21),rcond=None)[0].reshape(2,3,7)
            slopes.append(beta[1]);pred=beta[0]+x[:,j,None,None]*beta[1];pr=pred-np.einsum('n,nsi->si',w,pred)
            r=y-pred;r-=np.einsum('n,nsi->si',w,r)
            linear.append(np.einsum('n,nsi,nsj->sij',w,pr,pr));residual.append(np.einsum('n,nsi,nsj->sij',w,r,r))
            if valid:
                pred=np.empty_like(y)
                for ef in [0,1]:
                    train=fold!=ef;test=~train;number=np.bincount(hh[train],minlength=2);tw=(target/number)[hh[train]]
                    b=np.linalg.lstsq(np.sqrt(tw)[:,None]*A[train],np.sqrt(tw)[:,None]*y[train].reshape(train.sum(),21),rcond=None)[0].reshape(2,3,7)
                    pred[test]=b[0]+x[test,j,None,None]*b[1]
                r=y-pred;rm=np.einsum('n,nsi->si',w,r);z=r-rm
                cfs.append(np.einsum('n,nsi,nsj->sij',w,z,z));cfmeans.append(rm)
            else:cfs.append(np.full((3,7,7),np.nan));cfmeans.append(np.full((3,7),np.nan))
        records.append({'mean':my,'x_mean':mx,'x_var':vx,'cov':c,'slope':np.array(slopes),
            'linear_cov':np.array(linear),'residual_cov':np.array(residual),'crossfit_cov':np.array(cfs),'crossfit_mean':np.array(cfmeans)})
    return {k:np.mean([r[k] for r in records],0) for k in records[0]},crossfit_valid,len(records)*5


def verify():
    assert not (OUT/'verification.json').exists();p=read(OUT/'protocol.json');a=read(OUT/'analysis.json')
    for path,h in {**p['inputs_sha256'],**p['code_sha256'],**p['inherited_code_sha256']}.items():assert sha(PROJECT/path)==h,path
    checks=0;draw_counts={}
    for source,_ in SOURCES:
        counts=load(SPLIT/'evaluation'/f'{source}_counts.npz');old=load(TIME/'evaluation'/f'{source}_bootstrap.npz')['contrasts']
        boot=load(OUT/'evaluation'/f'{source}_bootstrap.npz');draw_counts[source]={}
        for v in ['data',*VARIANTS]:
            arr=load(TIME/'arrays'/f'{source}_{v}.npz');c=counts['data_counts'] if v=='data' else counts['MC_counts']
            multiplicities=np.vstack([np.ones((1,len(arr['ids'])),int),c[[0,127,511]]]);s,_,valid=summarize(arr,p,source,multiplicities)
            for i,m in enumerate(multiplicities):
                expected,cf,n=direct(arr,p,source,m);checks+=n
                assert cf==bool(valid['crossfit'][i])
                for key,x in expected.items():np.testing.assert_allclose(x,s[key][i],rtol=2e-8,atol=2e-9,err_msg=f'{source}/{v}/{i}/{key}')
            for key in s:np.testing.assert_allclose(s[key][0],a[source]['summaries'][v][key],rtol=1e-12,atol=1e-12)
            for covkey in ['cov','linear_cov','residual_cov','crossfit_cov']:
                eig=np.linalg.eigvalsh(s[covkey][0]);assert eig.min()>-1e-9,(source,v,covkey,eig.min())
        for vi,v in enumerate(VARIANTS):
            draw_counts[source][v]={}
            for kind,names,preds in [('global',GLOBAL_NAMES,PREDICTORS),('local',LOCAL_NAMES,PREDICTORS[:4])]:
                b=boot[kind+'_contrasts'];assert b.shape==(512,2,len(preds),3,len(names))
                for j,pred in enumerate(preds):
                    for k,scale in enumerate(SCALES):
                        ci={n:interval(b[:,vi,j,k,i]) for i,n in enumerate(names)}
                        assert ci==a[source]['variants'][v][kind][pred][str(scale)]['CI95']
                        if kind=='global':np.testing.assert_allclose(b[:,vi,j,k,0],old[:,vi,k,0],rtol=1e-8,atol=1e-9)
                draw_counts[source][v][kind]={'original':int(np.isfinite(b[:,vi,...,0]).all((1,2)).sum()),
                    'linear':int(np.isfinite(b[:,vi,...,1]).all((1,2)).sum()),'crossfit':int(np.isfinite(b[:,vi,...,2]).all((1,2)).sum())}
        print(source,'explicit weighted least-squares replicas and all intervals verified',flush=True)
    nbpath=PROJECT/'jupyter/codex_0001_csi_intensity_conditioning.ipynb';nb=nbformat.read(nbpath,as_version=4);nbformat.validate(nb)
    code=[c for c in nb.cells if c.cell_type=='code'];assert all(c.execution_count is not None for c in code)
    assert all(o.output_type!='error' for c in code for o in c.outputs)
    images=sum('image/png' in o.get('data',{}) for c in code for o in c.outputs);assert images==5
    tables={k:len(list(csv.DictReader((OUT/(k+'_table.csv')).open()))) for k in ['global','local']};assert tables=={'global':60,'local':48}
    assert read(OUT/'decision.json')['code_sha256']==sha(PROJECT/'simulation/report_intensity_condition.py')
    for path,h in read(OUT/'figure_record.json')['figures_sha256'].items():assert sha(PROJECT/path)==h,path
    save(OUT/'verification.json',{'input_files_verified':len(p['inputs_sha256']),'direct_weighted_regression_checks':checks,
        'sampled_draws':['point',0,127,511],'bootstrap_draws':draw_counts,'table_rows':tables,'executed_code_cells':len(code),'embedded_PNG':images,
        'notebook_sha256':sha(nbpath),'code_sha256':sha(__file__),'new_native_readouts':0,'default_changed':False,'official_PID_changed':False})


def archive():
    assert (OUT/'verification.json').exists() and not (OUT/'record.json').exists();p=read(OUT/'protocol.json')
    names=set(p['code_sha256'])|set(p['inherited_code_sha256'])
    names.update(['simulation/report_intensity_condition.py','simulation/verify_intensity_condition.py','simulation/README.md','jupyter/codex_0001_build_intensity_condition_notebook.py'])
    for module in list(sys.modules.values()):
        name=getattr(module,'__file__',None)
        if name:
            path=Path(name).resolve()
            if path.is_relative_to(PROJECT) and path.suffix=='.py':names.add(relative(path))
    sources={}
    for name in sorted(names):
        path=OUT/'source_snapshot'/name;path.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(PROJECT/name,path);sources[name]=sha(path)
    outputs={relative(x):sha(x) for x in sorted(OUT.rglob('*')) if x.is_file() and 'source_snapshot' not in x.parts and x.name!='record.json'}
    nb=PROJECT/'jupyter/codex_0001_csi_intensity_conditioning.ipynb'
    save(OUT/'record.json',{'outputs_sha256':outputs,'sources_snapshot_sha256':sources,'notebook':relative(nb),'notebook_sha256':sha(nb),
        'new_native_readouts':0,'new_independent_mothers':0,'default_changed':False,'official_PID_changed':False})
    print('Archived',len(outputs),'outputs and',len(sources),'source snapshots.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--archive',action='store_true');args=parser.parse_args()
    archive() if args.archive else verify()
