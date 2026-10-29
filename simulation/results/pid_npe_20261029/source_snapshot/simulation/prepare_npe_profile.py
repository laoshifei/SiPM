"""Freeze a native photon-statistics / passive voltage-transfer profile."""
import copy
from pathlib import Path
from reference import PROJECT
from prepare_pid import read
from publication_stats import save,sha

OUT=PROJECT/'simulation/results/pid_npe_20261029'
PARENT=PROJECT/'simulation/results/pid_bandwidth_20261029'
WAVE=PROJECT/'simulation/results/waveform_refit_20261029'
BUILD=PROJECT/'simulation/build/npe_20261029'
SCALES=(1.,1.5,2.,3.)


def main():
    OUT.mkdir(parents=True,exist_ok=False);entries={}
    parent=read(PARENT/'validation/protocol.json')
    for source,run in [('Co60',3),('Am241',12)]:
        original=parent['entries'][f'new_run{run}_{source}'];base=read(PROJECT/original['config'])
        parent_hash=sha(PROJECT/original['path'])
        for scale in SCALES:
            key=f's{scale:g}_run{run}_{source}';cfg=copy.deepcopy(base)
            cfg['sipm']['sensor']['constant_pde']=base['sipm']['sensor']['constant_pde']*scale
            cfg['readout_response']['signal_transmission']=1/scale
            cfg['signal_dependent_noise']['variance_per_signal_adc']=0.
            cfg['status']='NUMERICAL PROFILE ONLY: PDE thinning proxy for effective detected-photon statistics, not a measured or recommended device PDE'
            cfg['npe_profile']={'scale':scale,'source_identity_not_readout_input':True,
                'role':'Proposal on existing optical histories. A selected physical candidate must restore nominal PDE0.3, multiply scintillation yield by scale, retain the same passive voltage transmission, and undergo new full optical transport.',
                'transmission':'Effective passive ADC signal transfer; not a change of physical SiPM avalanche gain or nominal50ohm digitizer input. Unknown summing/coupling cannot be uniquely identified from these data.'}
            folder=OUT/'profile'/key;folder.mkdir(parents=True)
            cfgpath=folder/'config.json';root=folder/'events.root';save(cfgpath,cfg)
            entries[key]={'key':key,'source':source,'run':run,'scale':scale,'events':original['events'],
               'seed':original['seed'],'parent':original['path'],'parent_sha256':parent_hash,
               'config':str(cfgpath.relative_to(PROJECT)),'config_sha256':sha(cfgpath),'path':str(root.relative_to(PROJECT)),
               'command':[str(BUILD/'replay_response'),str(PROJECT/original['path']),str(cfgpath),str(root)]}
    protocol={'scope':__doc__,'code_sha256':sha(Path(__file__)),
      'checkpoint_commit':'db3dcff3e189cb4c73bde1db5af8e9cca6e68c76','parent_record_sha256':sha(PARENT/'record.json'),
      'q_edges':read(PARENT/'protocol.json')['q_edges'],'scales':SCALES,'entries':entries,
      'binary_sha256':{name:sha(BUILD/name) for name in ['csi_sources','replay_response']},
      'source_sha256':{str(p.relative_to(PROJECT)):sha(p) for folder in ['include','src','vendor','tests']
          for p in sorted((PROJECT/'simulation'/folder).glob('*')) if p.suffix in ['.cc','.hh','.py','.json']},
      'experimental_training':'Frozen first60 of each acquisition only; no change to experimental IDs, Q bins or PCA. Main-band0.4 separator only for robust diagnostic covariance.',
      'MC_training':'Former bandwidth-validation histories are explicitly reused as proposal/calibration support. They are not independent validation for this phase.',
      'profile':'Shared effective photoelectron scale; inverse passive voltage transmission approximately preserves mean waveform amplitude. At each scale recalibrate only nonnegative signal-residual PSD from complete3x3 PID covariance and3 short-scale F constraints.',
      'PDE_proxy_boundary':'PDE values0.3..0.9 in these replay files are mathematical proposal settings only. They are not an inference of S13361 efficiency. Final claimed results require new native Geant4 transport at nominalPDE0.3 and scaled optical yield.',
      'selection':'Same native ADC quality/pretrigger, amplitude>=200, t10>=384, commonQ; no PID-score selection. Recompute for each response.',
      'decision':'Select a shared scale using run3/run12 first60 fit only; report entire profile. Freeze final physical configs before new-seed full transport. Report remaining discrepancies; never tune on later experimental checks.',
      'identifiability':'Only an effective photon-count/ADC-conversion family can be constrained. Photon yield, PDE, collection and electrical coupling are not individually measured.'}
    save(OUT/'protocol.json',protocol)
    print('Prepared',len(entries),'native response profiles; no new transport yet.',flush=True)


if __name__=='__main__':main()
