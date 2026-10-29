"""Native confirmation of a screened response and one exploratory AP combination."""
import argparse
import json
import subprocess
from pathlib import Path
import numpy as np
import ROOT
from screen_response_candidates import HERE, PROJECT, SOURCES, info_of, load, measured_references, evaluate
from diagnose_charge import sha
from validate import check


def compare_linear_adc(control, prompt, waves, fraction, readout):
    """Compare all samples without retaining two full floating-point runs."""
    files = [ROOT.TFile.Open(str(path)) for path in (control, prompt)]
    trees = [f.Get('Events') for f in files]
    for tree, names in zip(trees, [('analog_mV', 'signal_mV'), ('signal_mV',)]):
        tree.SetBranchStatus('*', 0)
        for name in names:
            tree.SetBranchStatus(name, 1)
        if tree.GetEntries() != len(waves):
            raise ValueError('Linear replay event count differs')
    levels = 2 ** readout['adc_bits']
    lsb = (readout['adc_max_mV'] - readout['adc_min_mV']) / levels
    maximum = differing = 0
    try:
        for i in range(len(waves)):
            for tree in trees:
                tree.GetEntry(i)
            base_signal = np.array(trees[0].signal_mV)
            fast_signal = np.array(trees[1].signal_mV)
            analog = np.array(trees[0].analog_mV)
            predicted = np.clip(np.floor((analog + fraction * (fast_signal - base_signal)
                - readout['adc_min_mV']) / lsb), 0, levels - 1).astype(np.uint16)
            delta = waves[i].astype(np.int32) - predicted.astype(np.int32)
            maximum = max(maximum, int(abs(delta).max()))
            differing += int(np.count_nonzero(delta))
    finally:
        for f in files:
            f.Close()
    if differing:
        raise AssertionError(f'Linear screen differs in {differing} samples, maximum {maximum} ADC')
    return maximum, differing


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--screen',type=Path,required=True)
    p.add_argument('--out',type=Path);args=p.parse_args()
    args.screen=args.screen.resolve();report=json.loads((args.screen/'screen.json').read_text())
    best=report['best_descriptive_candidate'];candidate=report['candidates'][best]
    if candidate['afterpulse_probability']!=0:raise ValueError('This confirmation expects a prompt-path candidate')
    fraction=candidate['prompt_fraction'];refs,edges=measured_references()
    output=(args.out or args.screen/'confirmation').resolve();output.mkdir(parents=True,exist_ok=True)
    design={'scope':__doc__,'parent_screen_sha256':sha(args.screen/'screen.json'),
            'candidate':best,'prompt_fraction':fraction,'additional_combined_AP_probability':.15,
            'roles':'Adaptive exploratory follow-up after inspecting the first screen. No new transport histories or blind validation. No parameters promoted to calibrated status.',
            'code_sha256':{str(p.relative_to(PROJECT)):sha(p) for p in [Path(__file__),HERE/'validate.py',HERE/'screen_response_candidates.py',HERE/'replay_response.cc',HERE/'src/ReadoutResponse.cc',HERE/'build/replay_response']}}
    design_path=output/'design.json'
    if design_path.exists() and json.loads(design_path.read_text())!=design:raise ValueError('Confirmation design changed')
    design_path.write_text(json.dumps(design,indent=2)+'\n')
    results={};curves={}
    for source,name,run in SOURCES:
        parent=HERE/'results/colored_noise_20261028'/name/'events.root';parent_info=info_of(parent)
        for variant,prob in [('native_best',0.),('prompt_plus_ap15',.15)]:
            folder=output/variant/name;folder.mkdir(parents=True,exist_ok=True)
            config=json.loads(json.dumps(parent_info['config']))
            config['readout_response']={'model':'parallel_paths','prompt_charge_fraction':fraction,'prompt_decay_ns':5.,'prompt_lowpass_ns':0.}
            config['sipm']['sensor']['afterpulse_probability']=prob
            path=folder/'events.root';config_path=folder/'config.json'
            if config_path.exists() and json.loads(config_path.read_text())!=config:
                raise ValueError('Existing confirmation configuration differs')
            config_path.write_text(json.dumps(config,indent=2)+'\n')
            if not path.exists():
                with (folder/'replay.log').open('w') as log:
                    subprocess.run([str(HERE/'build/replay_response'),str(parent),str(config_path),str(path)],check=True,stdout=log,stderr=subprocess.STDOUT)
            checkpoint=folder/'validation.json';curve=folder/'shape.npy'
            if checkpoint.exists() and curve.exists():
                row=json.loads(checkpoint.read_text())
                if row['root_sha256']!=sha(path):raise ValueError('Completed response file changed')
                results.setdefault(variant,{'sources':{}})['sources'][source]=row
                curves[source+'_'+variant+'_shape']=np.load(curve)
                print(source,variant,'reused complete confirmation',flush=True)
                continue
            integrity_path=folder/'integrity.json';root_hash=sha(path)
            if integrity_path.exists():
                saved=json.loads(integrity_path.read_text())
                if saved['root_sha256']!=root_hash:raise ValueError('Validated response file changed')
                integrity=saved['report']
            else:
                print(source,variant,'checking native output',flush=True)
                integrity=check(path)
                integrity_path.write_text(json.dumps({'root_sha256':root_hash,'report':integrity},indent=2)+'\n')
            if not integrity['passed']:raise ValueError('Native response integrity failed')
            waves=load(path)['adc'];row,shape=evaluate(waves,refs[source],edges)
            row['integrity']=integrity;row['config_sha256']=sha(config_path);row['root_sha256']=root_hash
            if prob==0:
                maximum,differing=compare_linear_adc(args.screen/'control'/name/'events.root',
                    args.screen/'prompt_basis'/name/'events.root',waves,fraction,config['sipm']['readout'])
                row['linear_screen_max_adc_difference']=maximum
                row['linear_screen_differing_samples']=differing
                np.testing.assert_array_equal(row['common_ids'],candidate['sources'][source]['common_ids'])
            results.setdefault(variant,{'sources':{}})['sources'][source]=row
            curves[source+'_'+variant+'_shape']=shape
            checkpoint.write_text(json.dumps(row,indent=2,allow_nan=False)+'\n');np.save(curve,shape)
            print(source,variant,row['fluctuation_means'],row['shape_nrmse'],row['charge_ks'],flush=True)
    for row in results.values():
        row['joint_score']=sum(r['descriptive_score'] for r in row['sources'].values())
        row['within_screening_tolerances']=all(r['within_screening_tolerances'] for r in row['sources'].values())
    (output/'confirmation.json').write_text(json.dumps({'design':design,'results':results},indent=2,allow_nan=False)+'\n')
    np.savez_compressed(output/'curves.npz',**curves)


if __name__=='__main__':main()
