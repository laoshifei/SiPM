"""Freeze the core/tail diagnostic before inspecting its numerical results."""
import subprocess
import json
import ROOT
from pid_core_tail_common import *
from extract_pid import label_truth


def main():
    assert not OUT.exists();OUT.mkdir()
    for name in ['arrays','references','figures','source_snapshot']:(OUT/name).mkdir()
    old=read(PARENT/'protocol.json');record=read(PARENT/'record.json');acqr=read(ACQ/'record.json')
    needed=[PARENT/n for n in ['protocol.json','record.json','decision.json','evaluation.json']]+[ACQ/'record.json']
    for source,run in SOURCES:
        needed.append(PROJECT/old['native'][source]['path'])
        for seg in SEGMENTS:
            needed.append(ACQ/'arrays'/f'run{run}_{seg}.npz')
            for B in old['block_sizes']:
                needed.append(PROJECT/'simulation/results/state_response_20261030/arrays'/f'run{run}_{seg}_alignment_B{B}.npz')
                for v in VARIANTS:needed.append(PARENT/'arrays'/f'run{run}_{seg}_B{B}_{v}_evaluation.npz')
    hashes={relative(f):sha(f) for f in needed}
    for path,value in hashes.items():
        expected=record['outputs_sha256'].get(path,acqr['outputs_sha256'].get(path,old['inherited_inputs_sha256'].get(path)))
        if expected is not None:assert value==expected,path
    p={'scope':__doc__,'code_sha256':sha(__file__),'common_code_sha256':sha(PROJECT/'simulation/pid_core_tail_common.py'),
        'checkpoint_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT).decode().strip(),
        'inputs_sha256':hashes,'parent_record_sha256':sha(PARENT/'record.json'),
        'q_edges':old['q_edges'],'support_bins':old['support_bins'],'conditional_cuts':old['conditional_cuts'],
        'groups':old['groups'],'native':old['native'],'block_sizes':old['block_sizes'],'variants':VARIANTS,
        'coverages':COVERAGES,'fences_robust_sigma':FENCES,'normal_IQR_constant':NORMAL_IQR,
        'selection':'All original measured main-band and MC IDs retained; original Q, first60 Q-half targets, PCA, RC/noise candidates and mother folds fixed. No post-diagnostic event rejection or parameter fitting.',
        'reference':'For each original Q bin and evaluation fold, use opposite measured raw-ID blocks to estimate weighted median and IQR/1.3489795 for each score. Apply the same reference to data and MC evaluation fold. Diagnostic tails use abs(z)>2 and >3; these are not Gaussian probabilities or particle selections.',
        'widths':'Root-mean-square of within-original-Q central50/80/90/95-percent weighted quantile widths; same endpoint convention as prior IQR. SD uses full untrimmed within-Q covariance. Compare width50/SD between MC and data.',
        'variance':'Use deviations from each evaluation cohort own Q-bin mean; partition exact full within-Q second moments by fixed per-score tail masks and joint-any-score masks. No covariance term removed.',
        'waveforms':'Per-score cross-covariance with76 normalized40ns gate fractions, split exactly into abs(z)>2 and complement, divided by whole-cohort within-Q score SD. Descriptive correlation, not unique noise-source identification.',
        'auxiliary':'Within-Q correlations of three PID scores with log first128 baseline RMS and F50/102/202; shared samples/normalization can create correlation, not causal evidence.',
        'truth':'Stored original parent energy and avalanche-charge ancestry with90% dominance, energy95% sensitivity. Exact within-Q covariance decomposition by90% energy class. No ancestry labels assigned to data; Am contains alpha/gamma/decay-electron products.',
        'uncertainty':'512 paired raw-block/whole-original-mother draws; all fitted response parameters and diagnostic fences held fixed. Block128 primary,64 sensitivity. Full-pipeline exploration and threshold-fitting uncertainty excluded. Zero observed tail events can produce degenerate empirical-bootstrap intervals and do not prove zero tail probability; report raw counts and effective support.',
        'new_independent_mothers':0,'new_native_replays':0,'new_experimental_events':0,'default_changed':False}
    save(OUT/'protocol.json',p)
    for source,run in SOURCES:
        entry=old['native'][source];ids=np.array(entry['selected_ids']);f=ROOT.TFile.Open(str(PROJECT/entry['path']))
        tree=f.Get('Events');info=json.loads(str(f.Get('RunInfo').GetString().Data()))
        assert info['completed'] and info['source']==source and info['seed']==entry['seed'] and tree.GetEntries()==entry['events']
        tree.SetBranchStatus('*',0)
        fields=['event_id','edep_MeV','charge_C','edep_by_origin_MeV','charge_by_origin_C','edep_origin_species_MeV','edep_mylar_MeV','edep_air_MeV']
        for name in fields:assert tree.GetBranch(name),name;tree.SetBranchStatus(name,1)
        values={name:[] for name in fields};seen=[];wanted=set(ids.tolist())
        for i in range(tree.GetEntries()):
            tree.GetBranch('event_id').GetEntry(i)
            if int(tree.event_id) not in wanted:continue
            tree.GetEntry(i)
            for name in fields:
                x=getattr(tree,name);values[name].append(np.array(x) if 'by_origin' in name or 'origin_species' in name else x)
        f.Close();values={k:np.asarray(v) for k,v in values.items()};np.testing.assert_array_equal(values['event_id'],ids)
        np.testing.assert_allclose(values['edep_by_origin_MeV'].sum(1),values['edep_MeV'],rtol=1e-10,atol=1e-12)
        np.testing.assert_allclose(values['edep_origin_species_MeV'].sum(1),values['edep_MeV'],rtol=1e-10,atol=1e-12)
        np.testing.assert_allclose(values['charge_by_origin_C'].sum(1),values['charge_C'],rtol=1e-10,atol=1e-20)
        for name,key,dominance in [('label_energy90','edep_by_origin_MeV',.9),('label_energy95','edep_by_origin_MeV',.95),('label_charge90','charge_by_origin_C',.9)]:
            values[name],values[name+'_fractions']=label_truth(values[key],dominance)
        np.savez_compressed(OUT/'arrays'/f'run{run}_ancestry.npz',**values)
        for seg in SEGMENTS:
            data=measured(run,seg);np.savez_compressed(OUT/'arrays'/f'run{run}_{seg}_data.npz',**data)
            for B in p['block_sizes']:
                reference=fit_reference(data,p,source,B);save(OUT/'references'/f'run{run}_{seg}_B{B}.json',reference)
        print(source,'original ancestry and fixed diagnostic references saved',flush=True)


if __name__=='__main__':main()
