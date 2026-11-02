"""Freeze a small low-frequency transfer grid and native calibration jobs."""
import copy
import subprocess
from lowfreq_common import *


def main():
    assert not OUT.exists(),'Refuse an existing study'
    for name in ['configs','grid_native','grid_arrays','native','arrays','logs','evaluation','figures','source_snapshot']:(OUT/name).mkdir(parents=True,exist_ok=True)
    parent=read(SPLIT/'protocol.json');record=read(SPLIT/'record.json');entries={k:e for k,e in parent['entries'].items() if e['source']==SOURCE}
    configs={};jobs={};inputs=[SPLIT/n for n in ['record.json','protocol.json','analysis.json','fit.json','verification.json']]
    inputs += [SPLIT/'evaluation'/f'Am241_{n}.npz' for n in ['counts','bootstrap']]
    inputs += [REPLAY/'point_arrays/run12_last20_B128_data.npz',SCALES/'references/run12_last20_B128.json']
    for ef in [0,1]:
        original=PROJECT/read(SPLIT/'fit.json')[SOURCE]['folds'][str(ef)]['config'];base=read(original);inputs.append(original)
        configs[str(ef)]={}
        for label,corner,dc in GRID:
            cfg=copy.deepcopy(base);m=cfg['signal_dependent_noise'];h,lost=tapered_coefficients(m['coefficients'],corner,dc)
            power=float(h@h)
            if label!='identity':
                m['coefficients']=(h/np.sqrt(power)).tolist();m['variance_per_signal_adc']*=power
                cfg['status']='Frozen low-frequency transfer sensitivity; candidate only, no default promotion.'
            cp=OUT/'configs'/f'eval{ef}_{label}.json';save(cp,cfg)
            configs[str(ef)][label]={'path':relative(cp),'sha256':sha(cp),'corner_MHz':corner,'dc_amplitude':dc,
                'kernel_power_ratio':power,'outside_original_FIR_power_fraction':lost,'source_config':relative(original)}
    for key,e in entries.items():
        old=load(PARENT/'arrays'/f'{key}_frozen.npz');fold=np.asarray(read(FULL/'configs'/f'{key}.json')['folds'])
        inputs += [PROJECT/e['path'],PARENT/'arrays'/f'{key}_frozen.npz',FULL/'configs'/f'{key}.json',FULL/'arrays'/f'{key}_original.npz',
            SPLIT/'native'/f'{key}.root',SPLIT/'logs'/f'{key}_extraction.json']
        inputs += [SPLIT/'arrays'/f'{key}_split_r{r}.npz' for r in [0,1]]
        for ef in [0,1]:
            name=f'{key}_eval{ef}';ids=old['ids'][fold[old['ids']]==ef];cases=[]
            for label,_,_ in GRID:
                for r in [0,1]:
                    tag=f'{name}_{label}_r{r}';cases.append({'key':tag,'config':str(PROJECT/configs[str(ef)][label]['path']),
                        'output':str(OUT/'grid_native'/f'{tag}.root'),'phase_ns':0.,'noise_replica':r,'noise_scope':'all_readout','candidate':label})
            plan=OUT/'configs'/f'{name}_jobs.json';save(plan,{'ids':ids.tolist(),'jobs':cases})
            jobs[name]={'key':key,'batch':e['batch'],'fold':ef,'path':relative(plan),'sha256':sha(plan),'parent':e['path'],
                'parent_sha256':e['full_parent_sha256'],'mothers':len(ids)}
    code=['simulation/lowfreq_common.py','simulation/prepare_lowfreq.py','simulation/run_lowfreq_grid.py','simulation/fit_lowfreq.py']
    inherited=['simulation/spectral_split_common.py','simulation/full_selection_common.py','simulation/charge_scale_common.py',
        'simulation/state_response_common.py','simulation/extract_full_selection.py','simulation/analyze_spectral_split.py',
        'analysis/paper_baseline/prepare.py','analysis/paper_baseline/refine_waveforms.py','simulation/prepare_acquisition_state.py']
    for path in inputs:
        if relative(path) in record['outputs_sha256']:assert sha(path)==record['outputs_sha256'][relative(path)]
    save(OUT/'protocol.json',{'checkpoint_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'scope':'Am241 low-frequency native sensitivity after the spectral split; all existing data are explored, not blind validation.',
        **{k:parent[k] for k in ['q_edges','support_bins','conditional_cuts','joint_guard']},'entries':entries,'configs':configs,'grid_jobs':jobs,
        'inputs_sha256':{relative(p):sha(p) for p in sorted(set(inputs))},'code_sha256':{p:sha(PROJECT/p) for p in code},
        'inherited_code_sha256':{p:sha(PROJECT/p) for p in inherited},
        'binary_sha256':{relative(BUILD/n):sha(BUILD/n) for n in ['replay_readout_cached','replay_spectral_pool']},
        'model':'Multiply inherited FIR transfer by sqrt((f^2+(rho*fc)^2)/(f^2+fc^2)), retaining phase; use16384-point factorization and crop to original1025 taps. Store unit-norm FIR and reduce variance strength by actual kernel norm. Identity is byte-exact. No additional amplitude fit, event normalization or waveform injection.',
        'grid':GRID,'calibration':'Assigned original selected MC mothers, two fixed native readout streams0/1, own ADC charge/quality with frozen prior full-batch RMS. Use opposite measured fold only. Fit all six F covariance terms, six PID covariance terms, three F means, six PID widths and mean ADC. Final full-pool selection required.',
        'objective':'Equal sum of mean squared normalized F covariance residuals/.2, PID covariance residuals/.2, F relative mean residuals/.1, PID relative width residuals/.15, and mean ADC relative L2/.03. Average per-stream covariance/width squared/profile before objective. First minimum wins; identity included. No grid expansion/refit after full-pool results.',
        'fixed_parameters':'Complementary delay, high/common variance CV, optical physics, avalanche histories, mean readout and stationary electronic noise all fixed to previous candidate. Source crystal/SiPM geometry unchanged.',
        'native_validation':'Chosen fold configs evaluated over all192000 original Am241 mothers with current/zero-CV legacy ADC control. Prior split r0/r1 are the comparison baseline. All source Q bins and half-bin targets required.',
        'uncertainty':'Extend prior split512 whole-mother counts conditionally for entrants; retain every experimental block count. Fit and realized noise fixed. Two readout streams are not a converged expectation.',
        'extension_seed':731102990,'new_independent_mothers':0,'new_experimental_events':0,'default_changed':False,'official_PID_changed':False})
    print('Frozen ten finite-FIR candidates, opposite-fold joint loss and full-pool validation.',flush=True)


if __name__=='__main__':main()
