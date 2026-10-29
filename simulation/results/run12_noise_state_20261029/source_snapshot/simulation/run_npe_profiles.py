"""Run independent native profile/extraction/fit jobs with bounded concurrency."""
import argparse,concurrent.futures,json,os,subprocess,time
from pathlib import Path
from prepare_npe_profile import OUT,PROJECT
from prepare_pid import read
from publication_stats import sha,save

PYTHON='/home/xc/software/jupyter/.venv/bin/python'


def one(entry):
    key=entry['key'];started=time.time();steps=[]
    assert sha(PROJECT/entry['config'])==entry['config_sha256']
    jobs=[('replay',entry['command'],PROJECT/entry['path']),
          ('extract',[PYTHON,'-u','simulation/extract_npe_profile.py','--keys',key],OUT/'profile/extraction'/f'{key}.json'),
          ('fit',[PYTHON,'-u','simulation/fit_npe_profile.py','--keys',key],OUT/'fit'/f'{key}.json')]
    for name,command,target in jobs:
        if target.exists():
            steps.append({'stage':name,'existing':str(target.relative_to(PROJECT))});continue
        log=OUT/'logs'/f'{key}_{name}.log'
        with log.open('w') as handle:
            result=subprocess.run(command,cwd=PROJECT,stdout=handle,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError(f'{key} {name} exit{result.returncode}; see {log}')
        steps.append({'stage':name,'command':command,'log':str(log.relative_to(PROJECT)),'exit':result.returncode})
        print(key,name,'completed',flush=True)
    return {'key':key,'steps':steps,'seconds':time.time()-started}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workers',type=int,default=3);args=p.parse_args()
    assert 1<=args.workers<=4
    os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
    (OUT/'logs').mkdir(exist_ok=True)
    protocol=read(OUT/'protocol.json');rows=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(one,e) for e in protocol['entries'].values()]
        for f in concurrent.futures.as_completed(futures):rows.append(f.result())
    save(OUT/'execution.json',{'scope':__doc__,'code_sha256':sha(Path(__file__)),'workers':args.workers,'jobs':rows})
    print('All8 profiles extracted and fitted.',flush=True)


if __name__=='__main__':main()
