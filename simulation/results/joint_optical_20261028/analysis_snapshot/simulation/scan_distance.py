"""Source gap sensitivity, with scintillation disabled. Never infer a fitted gap."""
import argparse,json,subprocess,sys
from pathlib import Path
from validate import check

HERE=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--events',type=int,default=1000)
    p.add_argument('--config',type=Path,default=HERE/'configs/csi_tl_initial.json')
    p.add_argument('--resume',action='store_true',help='Reuse only completed, validated runs; refuse partial directories')
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=args.resume)
    reports=[]
    for i,gap in enumerate((.5,1.,2.,5.,10.,20.)):
        folder=args.out/f'Am241_{gap:g}mm'
        if folder.exists():
            record=json.loads((folder/'run.json').read_text())
            expected=json.loads(args.config.read_text());expected['geometry']['source_gap_mm']=gap
            expected['optical']['yield_photons_per_MeV']=0.
            # Legacy scans saved noise-only ADC arrays; schema 5 explicitly
            # skips digitization for zero-light scans. Preserve legacy resume.
            saved=json.loads((folder/'config.json').read_text())
            if 'digitize' in saved['acquisition']:expected['acquisition']['digitize']=False
            expected['status']='Energy-loss transport diagnostic; scintillation disabled, waveforms not physical'
            assert args.resume and record.get('exit_code')==0 and record['events']==args.events and record['source']=='Am241' and record['seed']==272000+i,'Partial or incompatible run: use a new output directory'
            assert json.loads((folder/'config.json').read_text())==expected,'Resume configuration mismatch'
        else:
            subprocess.run([sys.executable,str(HERE/'run.py'),'--config',str(args.config),'--source','Am241',
                '--events',str(args.events),'--seed',str(272000+i),'--gap-mm',str(gap),'--transport-only','--out',str(folder)],check=True,stdout=subprocess.DEVNULL)
        report=check(folder/'events.root');report['run_folder']=str(folder)
        n=report['events'];k=report['alpha_deposit_events'];p0=k/n
        report['alpha_deposit_fraction']=p0
        # Wilson 95% binomial interval; denominator is emitted parent decays,
        # not trigger counts or experimental exposure.
        z=1.95996398454;den=1+z*z/n
        mid=(p0+z*z/(2*n))/den;half=z*((p0*(1-p0)/n+z*z/(4*n*n))**.5)/den
        report['alpha_deposit_fraction_wilson95']=[mid-half,mid+half]
        reports.append(report);(folder/'events_validation.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report),flush=True)
    (args.out/'summary.json').write_text(json.dumps({'scope':'Conditional point-source air-gap sensitivity with 5 um Mylar. No capsule/backing; no absolute activity, trigger efficiency or fitted experimental distance.',
        'runs':reports},indent=2)+'\n')

if __name__=='__main__':main()
