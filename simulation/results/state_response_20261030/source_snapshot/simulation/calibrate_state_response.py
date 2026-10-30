"""Opposite-block pretrigger calibration and bounded native mean-response proposal."""
import copy
import numpy as np
from state_response_common import *
from prepare_noise_model import make_kernel
from prepare_run12_noise_state import fit_states,apply_states
from publication_stats import block_counts


def mean_proposal(target,grid,p):
    loss=[];trans=[]
    for m in grid:
        gain=float(np.clip(.5*np.dot(target['wave'],m['wave'])/np.dot(m['wave'],m['wave']),*p['transmission_bounds']))
        l=np.mean((m['wave']*gain/.5-target['wave'])**2)/(.03*max(target['wave']))**2
        l+=((m['mean'][1]-target['mean'][1])/.003)**2
        l+=((m['mean'][0]-target['mean'][0])/.003)**2
        l+=((m['tail']-target['tail'])/.005)**2
        trans.append(gain);loss.append(l)
    j=int(np.argmin(loss));return j,trans[j],loss,trans


def main():
    assert not (OUT/'mean_calibration.json').exists()
    p=read(OUT/'protocol.json');report={'scope':__doc__,'groups':{},'protocol_sha256':sha(OUT/'protocol.json'),
        'fit_design_sha256':sha(OUT/'fit_design.json'),'code_sha256':sha(__file__)}
    for source,run in SOURCES:
        base=read(PROJECT/p['native'][source]['config']);ro=base['sipm']['readout']
        lsb=(ro['adc_max_mV']-ro['adc_min_mV'])/2**ro['adc_bits']
        grid=[moments(load(OUT/'arrays'/f'run{run}_quietRC{j}.npz'),p,source) for j in range(5)];jobs=[]
        for si,seg in enumerate(SEGMENTS):
            a,raw=measured(run,seg)
            for B in p['block_sizes']:
                folds=(a['ids']//B)%2;rawfolds=(raw['ids']//B)%2
                for ef in [0,1]:
                    key=f'run{run}_{seg}_B{B}_eval{ef}';train=folds!=ef;rt=rawfolds!=ef
                    assert not np.intersect1d(a['ids'][train],a['ids'][~train]).size
                    pre=raw['waves'][rt,:256].astype(float)
                    h,_,freq,power,diag=make_kernel(pre);states,state_fit=fit_states(pre[:,:128],h,lsb)
                    provenance={'record':relative(OUT/'fits'/f'{key}.json'),'training_fold':1-ef,'evaluation_fold':ef,
                        'raw_block_size':B,'segment':seg,'input_region':'First128 for baseline/RMS states; detrended first256 for FIR, original eligible waveform pool before main-band selection.'}
                    e=apply_states(base,states,h,provenance)
                    jobs.append(create_job(key+'_electronics',e))
                    target=moments(a,p,source,train.astype(int));assert target is not None
                    j,trans,loss,gains=mean_proposal(target,grid,p)
                    response=copy.deepcopy(e);response['sipm']['readout']['lowpass_ns']=p['RC_grid_ns'][j]
                    response['readout_response']['signal_transmission']=trans
                    response['signal_dependent_noise']['variance_per_signal_adc']=0.
                    response['state_response_study']={'context':key,'stage':'base_without_signal_residual',**provenance}
                    jobs.append(create_job(key+'_base',response))
                    rng=np.random.default_rng(610030900+run+100*si+B+10*ef);stability=[]
                    for b in range(128):
                        count=block_counts(a['ids'],tuple(a['span']),B,rng)*train
                        tt=moments(a,p,source,count)
                        if tt is None:continue
                        jj,g,_,_=mean_proposal(tt,grid,p);stability.append([jj,g])
                    row={'source':source,'run':run,'segment':seg,'block':B,'evaluation_fold':ef,
                        'calibration_ids':a['ids'][train],'evaluation_ids':a['ids'][~train],
                        'pretrigger_calibration_ids':raw['ids'][rt],'n_pretrigger':int(rt.sum()),
                        'spectrum_diagnostic':diag,'state_fit':state_fit,'training_target':target,
                        'RC_grid_ns':p['RC_grid_ns'],'grid_loss':loss,'grid_transmission':gains,
                        'selected_RC_index':j,'selected_RC_ns':p['RC_grid_ns'][j],'selected_transmission':trans,
                        'training_block_refits':stability,'noise_state_count':len(states),
                        'refit_limit':'Training raw-block resampling of mean objective only; fixed original MC/Q2 targets, noise spectrum and mixture held fixed. Not a full parameter/model uncertainty interval.'}
                    save(OUT/'fits'/f'{key}.json',serialize(row));np.savez_compressed(OUT/'fits'/f'{key}_spectrum.npz',
                        frequency_mhz=freq,power=power,kernel=h,training_ids=raw['ids'][rt])
                    report['groups'][key]={'source':source,'run':run,'segment':seg,'block':B,'evaluation_fold':ef,
                        'RC_ns':p['RC_grid_ns'][j],'transmission':trans,'noise_states':len(states),
                        'fit':relative(OUT/'fits'/f'{key}.json'),'fit_sha256':sha(OUT/'fits'/f'{key}.json')}
                    print(key,'RC',round(p['RC_grid_ns'][j],3),'trans',round(trans,4),'states',len(states),flush=True)
        execute_jobs(source,jobs,'state_base')
    report['default_changed']=False;save(OUT/'mean_calibration.json',serialize(report))


if __name__=='__main__':main()
