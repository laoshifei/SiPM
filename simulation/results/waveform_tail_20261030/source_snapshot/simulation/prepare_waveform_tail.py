"""Freeze raw-waveform inspection, estimator controls and rank-three shape diagnostics."""
import subprocess
from waveform_tail_common import *
from evaluate_refined_rc import selected_arrays
from state_response_common import measured as original_measured
from reference import bin_layout


def main():
    assert not OUT.exists();OUT.mkdir()
    for name in ['arrays','designs','figures','source_snapshot']:(OUT/name).mkdir()
    parent=read(PARENT/'protocol.json');pr=read(PARENT/'record.json');rcp=read(RC/'protocol.json');rcr=read(RC/'record.json')
    mean=read(RC/'mean_calibration.json');noise=read(RC/'noise_selection.json');needed=[PARENT/n for n in ['protocol.json','record.json','diagnostics.json','decision.json']]
    needed += [RC/n for n in ['protocol.json','record.json','mean_calibration.json','noise_selection.json']]
    needed.append(PROJECT/'analysis/paper_baseline/results/publication_20261028/point.json')
    for source,run in SOURCES:
        needed += [PROJECT/f'data/DT5730_DPP_PHA/DAQ/run_{run}/UNFILTERED/Data_run_{run}.BIN',STATE/'arrays'/f'run{run}_phase00.npz']
        for seg in SEGMENTS:
            key=f'run{run}_{seg}';needed.append(PARENT/'arrays'/f'{key}_data.npz')
            for B in parent['block_sizes']:
                needed.extend([PARENT/'references'/f'{key}_B{B}.json',STATE/'arrays'/f'{key}_alignment_B{B}.npz'])
                for v in ['data',*VARIANTS]:needed.append(PARENT/'arrays'/f'{key}_B{B}_{v}_z.npz')
                for ef in [0,1]:
                    for family in ['inherited','spectrum']:
                        k=noise['groups'][f'{key}_B{B}_eval{ef}']['families'][family]['chosen_native_key'];needed.append(RC/'arrays'/f'{k}.npz')
    hashes={relative(f):sha(f) for f in sorted(set(needed))}
    for path,value in hashes.items():
        expected=pr['outputs_sha256'].get(path,rcr['outputs_sha256'].get(path,rcp['inherited_inputs_sha256'].get(path)))
        if expected is not None:assert value==expected,path
    p={'scope':__doc__,'code_sha256':sha(__file__),'common_code_sha256':sha(PROJECT/'simulation/waveform_tail_common.py'),
        'checkpoint_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT).decode().strip(),
        'parent_record_sha256':sha(PARENT/'record.json'),'inputs_sha256':hashes,
        'q_edges':parent['q_edges'],'support_bins':parent['support_bins'],'conditional_cuts':parent['conditional_cuts'],
        'block_sizes':parent['block_sizes'],'groups':parent['groups'],'variants':VARIANTS,'controls':CONTROLS,
        'observables':OBSERVABLES,'flags':FLAGS,'coverages':COVERAGES,'shape_rank':3,'example_quantiles':EXAMPLE_QUANTILES,
        'fixed_cohort':'Original IDs/Q/targets/PCA and mother folds, including all diagnostic tails. No simulation refit, event rejection, new Geant4 mothers or ADC replay.',
        'ADC':'Experimental waves must equal original BIN rows at original IDs. MC waves are assembled from the previously frozen native responses, one response per mother and context. Recompute all controls from original ADC.',
        'controls_definition':'Original first128 baseline/raw t10; recent baseline at original anchor[-192,-64); centered11-sample smoothing only to find t10; both. All PID integrals use unsmoothed ADC. Report50/80/90/95-percent widths, SD and centers. The diagnostic controls do not replace official PID.',
        'observables_definition':'Recent minus first128 baseline in ADC and units of first128 RMS (not a standard error); linear first256 slope in ADC/us; integer-anchor and fractional-t10 shifts in ns; original-gate charge fractions500-1500 and2000-3000ns; positive maximum adjacent40ns-mean step from600 to2800ns divided by original peak amplitude. The step is not a pileup classifier.',
        'flag_definition':'Opposite measured blocks, within each original Q bin:99th percentile of absolute baseline_delta_RMS, absolute integer-anchor shift and late positive step; late fraction outside1st-99th percentiles. Same flags applied to data/MC. No causal classification or removal. Compare all events and inherited joint PID tail abs(z)>2.',
        'shape_modes':'Opposite measured blocks only: weighted within-Q covariance of76 original-gate charge fractions. Retain fixed top3 eigenvectors, each signed so largest absolute coefficient is positive. Center each evaluation Q-bin/fold on its own mean when decomposing variance into mode and orthogonal parts; separately report mean-square offset from training templates. Rank-ordered modes may rotate across folds; report subspace singular values. No waveform synthesis or score correction from these modes.',
        'example_rule':'Last20,block128 only. Middle supported Q bin (lower middle if even number); for each score choose the event nearest its own weighted2.5/50/97.5-percentile standardized score, ties by original ID. Data and extended-spectrum MC selected independently under identical rule. Show unsmoothed baseline-subtracted native ADC without event amplitude rescaling; examples are not evidence of population agreement.',
        'uncertainty':'512 paired original-index block64/128 and whole-original-mother draws. Responses, original tail fences, flag thresholds, modes, Q targets/PCA fixed. Report paired estimator changes and mode variance ratios. Threshold/mode/refit uncertainty and multiple comparisons not included; all experiments already explored.',
        'new_independent_mothers':0,'new_native_replays':0,'new_experimental_events':0,'default_changed':False}
    save(OUT/'protocol.json',p);extraction={};examples={}
    for source,run in SOURCES:
        path=PROJECT/f'data/DT5730_DPP_PHA/DAQ/run_{run}/UNFILTERED/Data_run_{run}.BIN';_,dtype,n=bin_layout(path)
        raw=np.memmap(path,dtype=dtype,offset=2,mode='r',shape=(n,))
        for seg in SEGMENTS:
            key=f'run{run}_{seg}';old,_=original_measured(run,seg);waves=old['waves']
            np.testing.assert_array_equal(waves,raw['data'][old['ids']]);a=extract(waves,old['ids'],old['q'])
            a.update(span=old['span'],n_raw=old['n_raw']);np.testing.assert_allclose(a['scores'][:,0],old['scores'],atol=1e-12)
            np.testing.assert_allclose(a['shape'],old['shape'],atol=1e-13)
            np.savez_compressed(OUT/'arrays'/f'{key}_data.npz',**a)
            extraction[key]={'n':len(a['ids']),'exact_original_BIN_ADC':True,'original_PID_shape_exact':True}
            for B in p['block_sizes']:
                design=training_design(a,p,source,B);save(OUT/'designs'/f'{key}_B{B}.json',design)
                if seg=='last20' and B==128:
                    z=load(PARENT/'arrays'/f'{key}_B{B}_data_z.npz')['z'];er=select_examples(a,p,source,z)
                    examples[f'{key}_data']=er;np.savez_compressed(OUT/'arrays'/f'{key}_data_examples.npz',waves=waves[[x['index'] for x in er]],ids=[x['event_id'] for x in er])
                assignment=load(STATE/'arrays'/f'{key}_alignment_B{B}.npz')['mother_assignment']
                for v in VARIANTS:
                    z0=selected_arrays(run,seg,B,v,mean,noise);z=extract(z0['waves'],z0['ids'],z0['q']);z.update(fold=assignment,n_parent=z0['n_parent'])
                    np.testing.assert_allclose(z['scores'][:,0],z0['scores'],atol=1e-12);np.testing.assert_allclose(z['shape'],z0['shape'],atol=1e-13)
                    nk=f'{key}_B{B}_{v}';np.savez_compressed(OUT/'arrays'/f'{nk}.npz',**z)
                    extraction[nk]={'n':len(z['ids']),'original_PID_shape_exact':True,'new_independent_mothers':0,'new_native_replays':0}
                    if seg=='last20' and B==128 and v=='spectrum_CV_fit':
                        oldz=load(PARENT/'arrays'/f'{key}_B{B}_{v}_z.npz')['z'];er=select_examples(z,p,source,oldz)
                        examples[f'{key}_{v}']=er;np.savez_compressed(OUT/'arrays'/f'{key}_{v}_examples.npz',waves=z0['waves'][[x['index'] for x in er]],ids=[x['event_id'] for x in er])
            print(key,'original ADC controls, designs and examples saved',flush=True)
        del raw
    save(OUT/'extraction.json',{'scope':__doc__,'code_sha256':sha(__file__),'groups':extraction})
    save(OUT/'examples.json',{'rule':p['example_rule'],'groups':examples})


if __name__=='__main__':main()
