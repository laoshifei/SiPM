"""Continuous training-only RC proposals and fixed native noisy-bias corrections."""
import copy
import argparse
import sys
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize_scalar
from rc_refinement_common import *
from publication_stats import block_counts


class ResponseGrid:
    def __init__(self,p,source):
        self.rc=np.array([r['RC_ns'] for r in p['grid'][source]])
        ms=[summary(load(PROJECT/r['array']),p,source) for r in p['grid'][source]]
        self.functions={k:PchipInterpolator(self.rc,np.array([m[k] for m in ms]),axis=0,extrapolate=False)
            for k in ['wave','mean','tail']}
        self.bounds=p['RC_bounds_ns'];self.gain_bounds=p['transmission_bounds']
    def prediction(self,rc,gain,offset=None):
        z={k:np.asarray(fn(rc)) for k,fn in self.functions.items()};z['wave']*=gain/.5
        if offset is not None:
            z={k:v+offset[k] for k,v in z.items()}
        return z
    def solve(self,target,offset=None):
        if offset is None:offset={k:np.zeros_like(v(120.)) for k,v in self.functions.items()}
        def value(rc):
            wave=self.functions['wave'](rc)
            gain=float(np.clip(.5*np.dot(wave,target['wave']-offset['wave'])/np.dot(wave,wave),*self.gain_bounds))
            return mean_loss(self.prediction(rc,gain,offset),target),gain
        grid=np.arange(self.bounds[0],self.bounds[1]+.001,.25)
        losses=np.array([value(rc)[0] for rc in grid]);i=int(np.argmin(losses));lo=grid[max(0,i-1)];hi=grid[min(len(grid)-1,i+1)]
        candidates=[(losses[i],float(grid[i]),value(grid[i])[1])]
        if hi>lo:
            opt=minimize_scalar(lambda rc:value(rc)[0],bounds=(lo,hi),method='bounded',options={'xatol':1e-5})
            candidates.append((float(opt.fun),float(opt.x),value(opt.x)[1]))
        loss,rc,gain=min(candidates)
        return {'RC_ns':rc,'transmission':gain,'surrogate_loss':loss,'scan_RC_ns':grid,'scan_loss':losses}


def main(source_filter):
    destination=OUT/f'mean_calibration_{source_filter}.json'
    assert not destination.exists()
    p=read(OUT/'protocol.json');final={};initial_grids={}
    for source,run in SOURCES:
        if source!=source_filter:continue
        grid=ResponseGrid(p,source);contexts={};native_rounds={}
        for key,e in p['entries'].items():
            if e['source']!=source:continue
            data,_=measured(run,e['segment']);train=(data['ids']//e['block'])%2!=e['evaluation_fold']
            target=summary(data,p,source,train.astype(int));assert target is not None
            contexts[key]={'entry':e,'data':data,'train':train,'target':target,'rounds':[]}
        for iteration in range(3):
            jobs=[]
            for key,c in contexts.items():
                offset=None
                if iteration:
                    previous=c['rounds'][-1];a=load(OUT/'arrays'/f"{previous['native_key']}.npz")
                    actual=summary(a,p,source);quiet=grid.prediction(previous['RC_ns'],previous['transmission'])
                    offset={k:actual[k]-quiet[k] for k in quiet}
                fit=grid.solve(c['target'],offset);nk=f'{key}_mean{iteration}'
                cfg=read(PARENT/'configs'/f'{key}_electronics.json')
                cfg['sipm']['readout']['lowpass_ns']=fit['RC_ns'];cfg['readout_response']['signal_transmission']=fit['transmission']
                cfg['rc_refinement']={'context':key,'stage':'mean','iteration':iteration,
                    'training_fold':1-c['entry']['evaluation_fold'],'block':c['entry']['block']}
                jobs.append(job(nk,cfg));c['rounds'].append({**fit,'native_key':nk,'bias_offset':offset})
            execute(source,jobs,f'mean_round{iteration}')
            for key,c in contexts.items():
                r=c['rounds'][-1];a=load(OUT/'arrays'/f"{r['native_key']}.npz");m=summary(a,p,source)
                r['native_training_mean_loss']=mean_loss(m,c['target']);r['native_moments']=m
                print(key,'round',iteration,'RC',round(r['RC_ns'],3),'gain',round(r['transmission'],5),
                    'train time delta',round(-1000*(m['mean'][1]-c['target']['mean'][1]),3),flush=True)
        base_jobs=[]
        for key,c in contexts.items():
            e=c['entry'];chosen=min(range(3),key=lambda i:c['rounds'][i]['native_training_mean_loss'])
            fit=c['rounds'][chosen];cfg=read(OUT/'configs'/f"{fit['native_key']}.json")
            cfg['signal_dependent_noise']['variance_per_signal_adc']=0.;cfg['rc_refinement']['stage']='base'
            base_jobs.append(job(key+'_base',cfg))
            rng=np.random.default_rng(710030900+run+100*SEGMENTS.index(e['segment'])+e['block']+10*e['evaluation_fold'])
            stability=[]
            for ib in range(128):
                count=block_counts(c['data']['ids'],tuple(c['data']['span']),e['block'],rng)*c['train']
                target=summary(c['data'],p,source,count)
                if target is None:continue
                refit=grid.solve(target,fit['bias_offset']);stability.append([refit['RC_ns'],refit['transmission']])
            report={'scope':__doc__,'code_sha256':sha(__file__),'source':source,'context':key,'entry':e,
                'calibration_ids':c['data']['ids'][c['train']],'evaluation_ids':c['data']['ids'][~c['train']],
                'target':c['target'],'rounds':c['rounds'],'chosen_round':chosen,'selected':fit,
                'training_block_refits':stability,'limits':'Aggregate-summary interpolation proposes RC only; all checked ADC from native C++. Bias offsets and electronics/signal-noise law fixed during conditional RC bootstrap; no full nested uncertainty.'}
            save(OUT/'fits'/f'{key}_mean.json',serialize(report))
            final[key]={'entry':e,'RC_ns':fit['RC_ns'],'transmission':fit['transmission'],'selected_round':chosen,
                'native_key':fit['native_key'],'fit':relative(OUT/'fits'/f'{key}_mean.json'),'fit_sha256':sha(OUT/'fits'/f'{key}_mean.json')}
        execute(source,base_jobs,'base')
    save(destination,{'scope':__doc__,'code_sha256':sha(__file__),
        'protocol_sha256':sha(OUT/'protocol.json'),'groups':final})


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--source',choices=['Co60','Am241']);args=ap.parse_args()
    if args.source:main(args.source)
    else:
        assert not (OUT/'mean_calibration.json').exists()
        processes=[subprocess.Popen([sys.executable,'-u',str(Path(__file__)), '--source',source]) for source,_ in SOURCES]
        codes=[process.wait() for process in processes];assert all(code==0 for code in codes),codes
        groups={}
        for source,_ in SOURCES:groups.update(read(OUT/f'mean_calibration_{source}.json')['groups'])
        save(OUT/'mean_calibration.json',{'scope':__doc__,'code_sha256':sha(__file__),
            'protocol_sha256':sha(OUT/'protocol.json'),'groups':groups})
