"""Run a frozen local configuration and save code hashes, log and ROOT together."""
import argparse
from pathlib import Path
import hashlib
import json
import os
import platform
import subprocess
import time

HERE=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=HERE/'configs/csi_tl_initial.json')
    p.add_argument('--source',choices=['Am241','Co60'],required=True)
    p.add_argument('--events',type=int,default=100)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--gap-mm',type=float)
    p.add_argument('--transport-only',action='store_true',help='Disable scintillation ONLY for fast source/window energy-loss scans; not a waveform result')
    args=p.parse_args()
    if args.out.exists():p.error(f'Output directory already exists: {args.out}')
    # Hashes must describe the code actually executed. A failed rebuild can
    # leave an older executable in place, so never silently reuse that binary.
    build=subprocess.run(['cmake','--build',str(HERE/'build'),'-j4'],capture_output=True,text=True)
    if build.returncode:
        raise SystemExit('Build failed; no simulation started:\n'+build.stdout+build.stderr)
    args.out.mkdir(parents=True,exist_ok=False)
    (args.out/'build.log').write_text(build.stdout+build.stderr)
    cfg=json.loads(args.config.read_text())
    if args.gap_mm is not None:cfg['geometry']['source_gap_mm']=args.gap_mm
    if args.transport_only:
        cfg['optical']['yield_photons_per_MeV']=0.
        cfg['status']='Energy-loss transport diagnostic; scintillation disabled, waveforms not physical'
    local=args.out/'config.json';local.write_text(json.dumps(cfg,ensure_ascii=False,indent=2)+'\n')
    sources=[HERE/'main.cc',HERE/'vendor/SiPMResponse.cc',HERE/'vendor/SiPMResponse.hh',HERE/'CMakeLists.txt']
    sources+=sorted((HERE/'include').glob('*.hh'))+sorted((HERE/'src').glob('*.cc'))
    record={'source':args.source,'events':args.events,'seed':args.seed,'config_sha256':hashlib.sha256(local.read_bytes()).hexdigest(),
        'source_sha256':{str(f.relative_to(HERE)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sources},
        'platform':platform.platform(),'start_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'mode':'transport-only' if args.transport_only else 'optical-and-SiPM',
        'executable_sha256':hashlib.sha256((HERE/'build/csi_sources').read_bytes()).hexdigest()}
    env=dict(os.environ)
    # Installed Geant4 uses its own data defaults; record the actual configured environment.
    record['geant4_data_dir']=env.get('GEANT4_DATA_DIR','Geant4 compiled-in data location')
    record['status']='running'
    (args.out/'run.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    cmd=[str(HERE/'build/csi_sources'),str(local.resolve()),args.source,str(args.events),str(args.seed),str((args.out/'events.root').resolve())]
    started=time.monotonic()
    with (args.out/'run.log').open('w') as log:
        result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=env)
    record['exit_code']=result.returncode;record['elapsed_seconds']=time.monotonic()-started
    record['status']='process completed; integrity check pending' if result.returncode==0 else 'failed'
    (args.out/'run.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(record,ensure_ascii=False))
    raise SystemExit(result.returncode)

if __name__=='__main__':main()
