"""Freeze inputs for PID width diagnostics; leave earlier calibrated studies intact."""
import copy
from pathlib import Path
from reference import PROJECT
from publication_stats import sha,save
from prepare_pid import read

OUT=PROJECT/'simulation/results/pid_bandwidth_20261029'
PARENT=PROJECT/'simulation/results/waveform_refit_20261029'
PID=PROJECT/'simulation/results/pid_20261029'


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    entries={}
    for run,source,root,cfg in [
      (3,'Co60','independent/co60/events.root','independent/Co60_calibrated.json'),
      (12,'Am241','independent/am241/events.root','independent/Am241_calibrated.json'),
      (18,'Am241','run18/first60/events.root','run18/Am241_run18_first60.json')]:
        config=read(PARENT/cfg);candidate=copy.deepcopy(config)
        candidate['signal_dependent_noise']['variance_per_signal_adc']=0.
        candidate['status']='PID width diagnostic: exactly same optical histories and baseline noise; signal residual disabled'
        folder=OUT/f'run{run}/without_signal_residual';folder.mkdir(parents=True)
        config_path=folder/'config.json';save(config_path,candidate)
        entries[f'run{run}']={'run':run,'source':source,'parent_root':str((PARENT/root).relative_to(PROJECT)),
          'parent_sha256':sha(PARENT/root),'parent_config':str((PARENT/cfg).relative_to(PROJECT)),
          'base_config':str(config_path.relative_to(PROJECT)),'base_config_sha256':sha(config_path),
          'base_root':str((folder/'events.root').relative_to(PROJECT))}
    protocol={'scope':__doc__,'code_sha256':sha(Path(__file__)),
      'parent_waveform_record_sha256':sha(PARENT/'record.json'),'parent_PID_record_sha256':sha(PID/'record.json'),
      'binary_sha256':sha(PROJECT/'simulation/build/replay_response'),'entries':entries,
      'q_edges':read(PID/'protocol.json')['q_edges'],
      'training':'Any new response fit uses only frozen experimental first60; later segments excluded, previously explored checks not blind.',
      'diagnosis':'Separate native noiseless signal, baseline electronics, signal-dependent residual and ADC quantization on identical mothers. Fixed-anchor exact ratio covariance decomposition plus own-anchor removal comparisons; account for cross covariances.',
      'measured_sensitivity':'Original baseline versus recent strictly pretrigger baseline, on identical frozen event IDs; no new charge/shape selection. Dominant-band diagnostics retain the pre-existing0.4 ratio separator and report removed tails separately.',
      'goals':['Explain which contributors are compatible with observed PID widths.',
        'If evidence supports a change, recalibrate waveform-level response on training data, retaining mean-shape, baseline and short-timescale fluctuation checks.',
        'Do not add random numbers directly to PID scores or claim a unique physical mechanism from a conditional fit.'],
      'limits':['Native noiseless signal still includes finite photon/avalanche statistics and source geometry.',
        'Observed-anchor component covariance is an algebraic decomposition, not a unique causal attribution.',
        'Original common-Q MC samples are small; assess conditional uncertainty and support before interpreting width ratios.']}
    save(OUT/'protocol.json',protocol);print('Prepared',len(entries),'noise-removal replays.',flush=True)


if __name__=='__main__':main()
