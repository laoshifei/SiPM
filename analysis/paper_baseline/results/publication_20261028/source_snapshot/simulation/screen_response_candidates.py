"""Screen native single-avalanche response/AP candidates on frozen G4 photons.

The linear two-path scan mixes native noiseless responses and requantizes with
the identical electronic-noise realization. Candidate cuts are recomputed. This
is conditional model screening, not new Geant4 transport or a parameter fit CI.
"""
import argparse
import json
import subprocess
from pathlib import Path
import numpy as np
import ROOT
from reference import PROJECT, TIME_NS, aligned_shapes
from diagnose_charge import sha
from diagnose_fluctuations import fluctuation_metrics
from compare import mc_wave_features, BLUE, RED, GREEN, KEEP, panel, graph, label, save
from analyze_residual_covariance import analyze

HERE=Path(__file__).resolve().parent
FRACTIONS=(0., .025, .05, .075, .1, .125, .15, .2, .3)
SOURCES=(('Co60','co60',3),('Am241','am241',12))


def info_of(path):
    f=ROOT.TFile.Open(str(path));info=json.loads(str(f.Get('RunInfo').GetString().Data()));f.Close()
    if not info.get('completed'):raise ValueError('Incomplete ROOT input')
    return info


def load(path, signals=False):
    f=ROOT.TFile.Open(str(path));t=f.Get('Events');t.SetBranchStatus('*',0)
    names=['adc']+(['signal_mV','analog_mV'] if signals else [])
    for name in names:t.SetBranchStatus(name,1)
    values={name:[] for name in names}
    for event in t:
        for name in names:values[name].append(np.array(getattr(event,name),dtype=np.uint16 if name=='adc' else float))
    f.Close();return {name:np.array(v) for name,v in values.items()}


def ks(a,b):
    a=np.sort(a);b=np.sort(b);x=np.r_[a,b]
    return float(abs(np.searchsorted(a,x,'right')/len(a)-np.searchsorted(b,x,'right')/len(b)).max())


def prepare(args):
    args.out.mkdir(parents=True,exist_ok=True)
    design={'scope':__doc__,'fractions':FRACTIONS,'prompt_decay_ns':5.,'prompt_lowpass_ns':0.,
      'afterpulse_probabilities':[.15,.30],
      'fixed':'Original geometry, optical parameters, gain, PDE, slow response, noise and seeds. Same response/AP parameters for both sources. AP delay inherited60ns, hypothetical. XT stays disabled because flattened mapping lacks physical neighbors.',
      'screen':'Equal source weight; sum squared log-relative residuals for3 fluctuation means + (shape NRMSE/.03)^2 + (charge KS/.15)^2. Descriptive, not likelihood. Screening flags: each fluctuation within20%, shape NRMSE<=3%, charge KS<=.15. Not an uncertainty-qualified acceptance test.',
      'roles':'Run3/12 first60 only for screening; other run/segment comparisons are previously explored checks, not blind.',
      'sources':{},'code_sha256':{str(p.relative_to(PROJECT)):sha(p) for p in [Path(__file__),HERE/'replay_response.cc',HERE/'src/ReadoutResponse.cc',HERE/'include/ReadoutResponse.hh',HERE/'build/replay_response']}}
    for source,name,run in SOURCES:
        parent=args.mc/name/'events.root';info=info_of(parent);entries={}
        for candidate in ['control','prompt_basis','ap15','ap30']:
            folder=args.out/candidate/name;folder.mkdir(parents=True,exist_ok=True)
            cfg=json.loads(json.dumps(info['config']))
            if candidate=='prompt_basis':cfg['readout_response']={'model':'parallel_paths','prompt_charge_fraction':1.,'prompt_decay_ns':5.,'prompt_lowpass_ns':0.}
            if candidate.startswith('ap'):cfg['sipm']['sensor']['afterpulse_probability']=int(candidate[2:])/100
            cfg['response_candidate_status']='Hypothetical controlled test; not calibrated parameters'
            config=folder/'config.json'
            if config.exists() and json.loads(config.read_text())!=cfg:raise ValueError('Existing candidate differs')
            config.write_text(json.dumps(cfg,indent=2)+'\n')
            entries[candidate]={'config_sha256':sha(config),'path':str(folder.relative_to(PROJECT))}
        design['sources'][source]={'parent':str(parent),'parent_sha256':sha(parent),'events':info['events_requested'],'candidates':entries}
    target=args.out/'design.json'
    if target.exists() and json.loads(target.read_text())!=json.loads(json.dumps(design)):raise ValueError('Frozen design changed')
    target.write_text(json.dumps(design,indent=2)+'\n')


def replay(args):
    design=json.loads((args.out/'design.json').read_text())
    if sha(HERE/'build/replay_response')!=design['code_sha256'][str((HERE/'build/replay_response').relative_to(PROJECT))]:raise ValueError('Replay binary changed after design')
    for source,name,run in SOURCES:
        row=design['sources'][source]
        for candidate,entry in row['candidates'].items():
            folder=PROJECT/entry['path'];path=folder/'events.root'
            if path.exists():info=info_of(path)
            else:
                with (folder/'replay.log').open('w') as log:
                    subprocess.run([str(HERE/'build/replay_response'),row['parent'],str(folder/'config.json'),str(path)],check=True,stdout=log,stderr=subprocess.STDOUT)
                info=info_of(path)
            if candidate=='control':
                if info['response_replay']['equal_original_adc_events']!=row['events']:raise ValueError('Original ADC replay differs')
                if info['response_replay']['equal_original_avalanche_events']!=row['events']:raise ValueError('Original avalanche replay differs')
            print(source,candidate,info['response_replay'],flush=True)


def measured_references():
    with np.load(HERE/'results/bias_response_20261028/samples.npz') as z:bias=dict(z)
    cov=json.loads((HERE/'results/residual_covariance_20261028/covariance.json').read_text())
    result={}
    for source,name,run in SOURCES:
        key=f'run{run}_first60'
        with np.load(PROJECT/f'analysis/paper_baseline/results/pha_run{run:04d}_features.npz') as z:f=dict(z)
        with np.load(PROJECT/f'analysis/paper_baseline/results/phase2/run{run:04d}_features.npz') as z:a={k:z[k] for k in ['eligible_primary','split']}
        mask=a['eligible_primary']&(a['split']==0)&(f['amplitude']>=200)&((f['flags']&0x8000)==0)
        result[source]={'q':f['charge_long'][mask],'mean':bias[key+'_shape'],'fluctuations':bias[key+'_metrics'][:,:,3].mean(axis=0),
                        'covariance':np.array(cov['groups'][key]['views']['all']['across_event'])}
    return result,np.array(cov['q_edges'])


def evaluate(waves, target, edges):
    f,good=mc_wave_features(waves)
    selected=good&(f['amplitude']>=200)
    common=selected&(f['charge_long']>=edges[0])&(f['charge_long']<=edges[-1])&(f['t10_sample']>=384)
    ids=np.flatnonzero(common)
    if len(ids)<20:raise ValueError('Too few common-charge events')
    shapes,keep=aligned_shapes(waves[ids],{k:v[ids] for k,v in f.items()});shape=shapes.mean(axis=0)
    fluc=fluctuation_metrics(waves[ids],f['baseline'][ids],f['t10_sample'][ids])[:,:,3].mean(axis=0)
    region=(TIME_NS>=-100)&(TIME_NS<=3500)
    nrmse=float(np.sqrt(np.mean((shape[region]-target['mean'][region])**2))/target['mean'].max())
    distance=ks(f['charge_long'][selected],target['q'])
    cov,_=analyze(waves[ids].astype(float),f['baseline'][ids],f['t10_sample'][ids],f['charge_long'][ids],f['short_long_ratio'][ids],ids,edges)
    rel=fluc/target['fluctuations']-1
    row={'selected_n':int(selected.sum()),'common_n':len(ids),'common_ids':ids.tolist(),'shape_n':len(keep),
         'fluctuation_means':fluc.tolist(),'fluctuation_relative_residual':rel.tolist(),
         'shape_nrmse':nrmse,'charge_ks':distance,'charge_quantiles':np.quantile(f['charge_long'][selected],[.1,.5,.9]).tolist(),
         'covariance':cov['views']['all'],
         'descriptive_score':float(np.sum(np.log(fluc/target['fluctuations'])**2)+(nrmse/.03)**2+(distance/.15)**2),
         'within_screening_tolerances':bool(np.max(abs(rel))<=.2 and nrmse<=.03 and distance<=.15)}
    return row,shape


def screen(args):
    refs,edges=measured_references();rows={};arrays={'time_ns':TIME_NS}
    for source,name,run in SOURCES:
        base=load(args.out/'control'/name/'events.root',True)
        fast=load(args.out/'prompt_basis'/name/'events.root',True)
        cfg=info_of(args.out/'control'/name/'events.root')['config']['sipm']['readout']
        levels=2**cfg['adc_bits'];lsb=(cfg['adc_max_mV']-cfg['adc_min_mV'])/levels
        delta=fast['signal_mV']-base['signal_mV']
        for fraction in FRACTIONS:
            key=f'prompt_{fraction:.3f}';folder=args.out/'screen';folder.mkdir(exist_ok=True)
            checkpoint=folder/f'{source}_{key}.json';curve=folder/f'{source}_{key}.npy'
            if checkpoint.exists():row=json.loads(checkpoint.read_text());shape=np.load(curve)
            else:
                waves=np.clip(np.floor((base['analog_mV']+fraction*delta-cfg['adc_min_mV'])/lsb),0,levels-1).astype(np.uint16)
                if fraction==0:np.testing.assert_array_equal(waves,base['adc'])
                row,shape=evaluate(waves,refs[source],edges)
                checkpoint.write_text(json.dumps(row,indent=2,allow_nan=False)+'\n');np.save(curve,shape)
            rows.setdefault(key,{'prompt_fraction':fraction,'afterpulse_probability':0.,'sources':{}})['sources'][source]=row
            arrays[source+'_'+key+'_shape']=shape
            print(source,key,row['fluctuation_means'],row['shape_nrmse'],flush=True)
        del base,fast,delta
        for key in ['ap15','ap30']:
            checkpoint=args.out/'screen'/f'{source}_{key}.json';curve=checkpoint.with_suffix('.npy')
            if checkpoint.exists():row=json.loads(checkpoint.read_text());shape=np.load(curve)
            else:
                waves=load(args.out/key/name/'events.root')['adc'];row,shape=evaluate(waves,refs[source],edges)
                checkpoint.write_text(json.dumps(row,indent=2,allow_nan=False)+'\n');np.save(curve,shape)
            rows.setdefault(key,{'prompt_fraction':0.,'afterpulse_probability':int(key[2:])/100,'sources':{}})['sources'][source]=row
            arrays[source+'_'+key+'_shape']=shape
    for row in rows.values():
        row['joint_score']=sum(r['descriptive_score'] for r in row['sources'].values())
        row['within_screening_tolerances']=all(r['within_screening_tolerances'] for r in row['sources'].values())
    best=min(rows,key=lambda k:rows[k]['joint_score'])
    report={'scope':__doc__,'design_sha256':sha(args.out/'design.json'),'candidates':rows,'best_descriptive_candidate':best,
            'any_within_screening_tolerances':any(r['within_screening_tolerances'] for r in rows.values()),
            'measured_fluctuation_targets':{s:refs[s]['fluctuations'].tolist() for s,_,_ in SOURCES}}
    (args.out/'screen.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    np.savez_compressed(args.out/'curves.npz',**arrays)
    plots(args.out,report)


def plots(out,report):
    ROOT.gROOT.SetBatch(True);ROOT.gStyle.SetOptStat(0)
    c=ROOT.TCanvas('response_candidates','Shared native response candidates',1450,1050);c.Divide(2,2,.003,.003)
    for col,(source,name,run) in enumerate(SOURCES):
        rows=[report['candidates'][f'prompt_{f:.3f}']['sources'][source] for f in FRACTIONS]
        panel(c,col+1,source+' / shared prompt-charge fraction;Prompt charge fraction;Mean excess variance / mean (ADC)',0,0,.3,2.5)
        for k,(color,title) in enumerate([(BLUE,'50 ns'),(RED,'102 ns'),(GREEN,'202 ns')]):
            graph(np.array(FRACTIONS),np.array([r['fluctuation_means'][k] for r in rows]),color)
            graph(np.array([0.,.3]),np.full(2,report['measured_fluctuation_targets'][source][k]),color,2)
            label(title,.22+k*.22,.85,color,.03)
        label('Solid: native response; dashed: measured training',.18,.77,size=.026)
        panel(c,col+3,source+' / other validation metrics;Prompt charge fraction;Discrepancy',0,0,.3,.65)
        graph(np.array(FRACTIONS),np.array([r['charge_ks'] for r in rows]),RED)
        graph(np.array(FRACTIONS),np.array([r['shape_nrmse'] for r in rows]),BLUE)
        label('Charge KS distance',.22,.85,RED);label('Mean-shape NRMSE',.22,.79,BLUE)
        label('Selection recomputed; fixed gain/PDE/geometry',.18,.71,size=.026)
    save(c,out,'response_candidates')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','replay','screen','plot'])
    p.add_argument('--out',type=Path,required=True);p.add_argument('--mc',type=Path,default=HERE/'results/colored_noise_20261028')
    args=p.parse_args();args.out=args.out.resolve();args.mc=args.mc.resolve()
    if args.stage=='plot':plots(args.out,json.loads((args.out/'screen.json').read_text()))
    else:{'prepare':prepare,'replay':replay,'screen':screen}[args.stage](args)


if __name__=='__main__':main()
