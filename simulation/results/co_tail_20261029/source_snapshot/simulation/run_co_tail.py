"""Run the frozen two-source Geant4 tail candidate and check every mother."""
import argparse,concurrent.futures,os,subprocess,time
from pathlib import Path
from fit_co_tail import OUT,PROJECT
from prepare_pid import read
from publication_stats import save,sha

PYTHON='/home/xc/software/jupyter/.venv/bin/python'


def one(entry):
    key=entry['key'];start=time.time();steps=[]
    assert sha(PROJECT/entry['config'])==entry['config_sha256']
    root=PROJECT/entry['path']
    jobs=[('native',entry['command'],root),
          ('validate',[PYTHON,'-u','simulation/validate.py',str(root)],root.with_name(root.stem+'_validation.json')),
          ('extract',[PYTHON,'-u','simulation/run_co_tail.py','--extract',key],OUT/'extraction'/f'{key}.json')]
    for stage,command,target in jobs:
        if target.exists():
            steps.append({'stage':stage,'existing':True});continue
        log=OUT/'logs'/f'{key}_{stage}.log'
        with log.open('w') as f:
            result=subprocess.run(command,cwd=PROJECT,stdout=f,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError(f'{key} {stage} exit{result.returncode}; see {log}')
        steps.append({'stage':stage,'command':command,'exit':result.returncode,'log':str(log.relative_to(PROJECT))})
        print(key,stage,'completed',flush=True)
    return {'key':key,'seconds':time.time()-start,'steps':steps}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--extract');args=p.parse_args()
    protocol=read(OUT/'native_protocol.json')
    if args.extract:
        from extract_npe_profile import extract
        assert not (OUT/'extraction'/f'{args.extract}.json').exists(),'Refuse overwrite'
        pca=read(PROJECT/'analysis/paper_baseline/results/publication_20261028/point.json')['windows']['3000']['pca']
        extract(protocol['entries'][args.extract],OUT,protocol['q_edges'],pca);return
    assert sha(PROJECT/protocol['binary'])==protocol['binary_sha256']
    os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
    rows=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(one,e) for e in protocol['entries'].values()]
        for f in concurrent.futures.as_completed(futures):rows.append(f.result())
    save(OUT/'execution.json',{'scope':__doc__,'code_sha256':sha(Path(__file__)),
        'protocol_sha256':sha(OUT/'native_protocol.json'),'jobs':rows,'new_independent_mothers':sum(e['events'] for e in protocol['entries'].values())})
    print('Both native sources completed, validated and extracted.',flush=True)


if __name__=='__main__':main()
