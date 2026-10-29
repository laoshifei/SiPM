"""Freeze new-seed native validation and paired previous-response controls."""
import copy
from pathlib import Path
from prepare_pid_bandwidth import OUT, PARENT
from prepare_pid import read
from reference import PROJECT
from publication_stats import save,sha


def main():
    dest=OUT/'validation';dest.mkdir(exist_ok=False)
    fitted={r:read(OUT/f'covariance_fit/run{r}_config.json') for r in [3,12,18]}
    before={3:read(PARENT/'independent/Co60_calibrated.json'),
            12:read(PARENT/'independent/Am241_calibrated.json'),
            18:read(PARENT/'run18/Am241_run18_first60.json')}
    base={'Co60':before[3],'Am241':before[12]}
    entries={};native=[];replays=[]
    spec=[('new',3,'Co60'),('new',12,'Am241'),('old',3,'Co60'),('old',12,'Am241'),
          ('new',18,'Am241'),('old',18,'Am241'),('new',12,'Co60')]
    counts={'Co60':8000,'Am241':48000};seeds={'Co60':291029903,'Am241':291029912}
    for version,run,source in spec:
        key=f'{version}_run{run}_{source}';cfg=copy.deepcopy(base[source]);profile=(fitted if version=='new' else before)[run]
        for k in ['sipm','readout_response','electronics_noise','signal_dependent_noise']:
            cfg[k]=copy.deepcopy(profile[k])
        cfg['status']='Frozen PID bandwidth validation: '+key
        cfg['bandwidth_validation']={'record':str((dest/'protocol.json').relative_to(PROJECT)),
           'role':'new independent transport seed; no measured later-segment tuning','response':version,'state':run}
        folder=dest/key;folder.mkdir();cfgpath=folder/'config.json';rootpath=folder/'events.root'
        save(cfgpath,cfg);isnative=(version,run,source) in spec[:2]
        if isnative:
            command=['simulation/build/csi_sources',str(cfgpath.relative_to(PROJECT)),source,str(counts[source]),str(seeds[source]),str(rootpath.relative_to(PROJECT))]
        else:
            parent=dest/('new_run3_Co60' if source=='Co60' else 'new_run12_Am241')/'events.root'
            command=['simulation/build/replay_response',str(parent.relative_to(PROJECT)),str(cfgpath.relative_to(PROJECT)),str(rootpath.relative_to(PROJECT))]
        row={'key':key,'version':version,'run':run,'source':source,'native':isnative,'events':counts[source],
          'seed':seeds[source],'config':str(cfgpath.relative_to(PROJECT)),'config_sha256':sha(cfgpath),
          'path':str(rootpath.relative_to(PROJECT)),'command':command}
        entries[key]=row;(native if isnative else replays).append(row)
    protocol={'scope':__doc__,'code_sha256':sha(Path(__file__)),'fit_sha256':sha(OUT/'covariance_fit/fit.json'),
      'native_binary_sha256':sha(PROJECT/'simulation/build/csi_sources'),'replay_binary_sha256':sha(PROJECT/'simulation/build/replay_response'),
      'entries':entries,'native_jobs':native,'replay_jobs':replays,
      'q_edges':read(OUT/'protocol.json')['q_edges'],
      'selection':'Identical waveform-quality/primary-pretrigger, amplitude>=200 ADC, complete3us and t10>=384; no PID-score cut. Recompute own selection; also retain paired fixed old-selected IDs for sensitivity.',
      'targets_frozen':True,'new_transport':56000,
      'comparison':'first60 training closure plus middle20/last20 experimental checks; already explored experiments, not blind. New native seeds independent of calibration MC.',
      'width':'Per-Q IQR with paired whole-mother bootstrap for MC and original-index block64 resampling for experiment. Equal-bin RMS IQR across frozen fit-supported bins as summary, no pooling different Q-band centers.',
      'other_checks':['Mean normalized waveforms and all three short-scale F means','Baseline RMS','Original threshold/PCA unchanged','Same run12 response applied to both sources; no source ID input'],
      'decision':'A successful noise-only fit must preserve waveform means/F and improve widths on new native histories; a remaining discrepancy must be reported, not tuned using these checks.',
      'limits':['Long-correlated signal residual is phenomenological; no unique noise/intrinsic-shape identification.',
                'Old run3/run12 source-specific responses are for acquisition-state matching; common run12 state is the PID control.',
                'Counts remain insufficient for rare leakage claims.']}
    save(dest/'protocol.json',protocol)
    print('Frozen',len(native),'native jobs and',len(replays),'paired replays.',flush=True)


if __name__=='__main__':main()
