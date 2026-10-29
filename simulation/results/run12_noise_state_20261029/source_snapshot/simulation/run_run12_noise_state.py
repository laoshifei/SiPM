"""Run frozen native noise replays with bounded process concurrency."""
import argparse,concurrent.futures,os,subprocess,time
from pathlib import Path
from prepare_run12_noise_state import OUT,PROJECT
from prepare_pid import read
from publication_stats import save,sha

PYTHON='/home/xc/software/jupyter/.venv/bin/python'


def one(e):
    key=e['key'];start=time.time();stages=[]
    assert sha(PROJECT/e['config'])==e['config_sha256']
    jobs=[]
    if e['command']:jobs.append(('replay',e['command'],PROJECT/e['path']))
    jobs.append(('extract',[PYTHON,'-u','simulation/extract_run12_noise_state.py','--key',key],OUT/'extraction'/f'{key}.json'))
    for stage,command,target in jobs:
        if target.exists():
            stages.append({'stage':stage,'existing':True});continue
        log=OUT/'logs'/f'{key}_{stage}.log'
        with log.open('w') as f:result=subprocess.run(command,cwd=PROJECT,stdout=f,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError(f'{key} {stage} failed; see {log}')
        stages.append({'stage':stage,'command':command,'exit':result.returncode,'log':str(log.relative_to(PROJECT))})
        print(key,stage,'completed',flush=True)
    return {'key':key,'seconds':time.time()-start,'stages':stages}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workers',type=int,default=3);a=p.parse_args()
    assert 1<=a.workers<=4
    os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
    protocol=read(OUT/'protocol.json');assert sha(PROJECT/protocol['binary'])==protocol['binary_sha256']
    entries=sorted(protocol['entries'].values(),key=lambda e:({'last20':1,'middle20':2,'first60':3,None:0}[e['segment']],e['key']))
    rows=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures=[pool.submit(one,e) for e in entries]
        for f in concurrent.futures.as_completed(futures):rows.append(f.result())
    save(OUT/'execution.json',{'scope':__doc__,'code_sha256':sha(Path(__file__)),'workers':a.workers,'jobs':rows})
    print('All frozen noise-state jobs completed.',flush=True)


if __name__=='__main__':main()
