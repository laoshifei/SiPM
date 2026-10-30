"""Condition run18 on first60, then freeze nominal-PDE full-transport checks."""
import argparse,copy,subprocess
from pathlib import Path
from prepare_npe_profile import OUT,PARENT,BUILD,PROJECT
from prepare_pid import read
from publication_stats import sha,save


def run18():
    from extract_npe_profile import extract
    from fit_npe_profile import fit_one
    scale=read(OUT/'selection.json')['chosen_scale'];key=f's{scale:g}_run18_Am241'
    parent=read(PARENT/'validation/protocol.json')['entries']['new_run18_Am241']
    cfg=read(PROJECT/parent['config'])
    cfg['sipm']['sensor']['constant_pde']*=scale
    cfg['readout_response']['signal_transmission']=1/scale
    cfg['signal_dependent_noise']['variance_per_signal_adc']=0.
    cfg['status']='NUMERICAL PROFILE ONLY: run18 first60 acquisition state at the shared selected effective-photon scale'
    folder=OUT/'profile'/key;folder.mkdir(parents=True,exist_ok=False)
    cfgpath=folder/'config.json';path=folder/'events.root';save(cfgpath,cfg)
    entry={'key':key,'source':'Am241','run':18,'scale':scale,'events':parent['events'],'seed':parent['seed'],
       'parent':parent['path'],'parent_sha256':sha(PROJECT/parent['path']),
       'config':str(cfgpath.relative_to(PROJECT)),'config_sha256':sha(cfgpath),'path':str(path.relative_to(PROJECT)),
       'command':[str(BUILD/'replay_response'),str(PROJECT/parent['path']),str(cfgpath),str(path)]}
    protocol=read(OUT/'protocol.json')
    save(OUT/'run18_protocol.json',{'scope':'Run18 first60 state extension at the frozen shared scale; no reselecting scale.',
       'code_sha256':sha(Path(__file__)),'parent_profile_protocol_sha256':sha(OUT/'protocol.json'),
       'selection_sha256':sha(OUT/'selection.json'),'entry':entry,'q_edges':protocol['q_edges']})
    with (OUT/'logs'/f'{key}_replay.log').open('w') as log:subprocess.run(entry['command'],check=True,stdout=log,stderr=subprocess.STDOUT)
    pca=read(PROJECT/'analysis/paper_baseline/results/publication_20261028/point.json')['windows']['3000']['pca']
    extract(entry,OUT/'profile',protocol['q_edges'],pca)
    fit_one(entry,protocol,pca)
    save(OUT/'run18_extension.json',{'scope':'Additional first60-only state fit; this file links the standard fit output to its separately frozen run18 inputs.',
       'protocol_sha256':sha(OUT/'run18_protocol.json'),'fit_sha256':sha(OUT/'fit'/f'{key}.json'),
       'extraction_sha256':sha(OUT/'profile/extraction'/f'{key}.json'),'config_sha256':sha(OUT/'fit'/f'{key}_config.json')})


def freeze():
    scale=read(OUT/'selection.json')['chosen_scale'];dest=OUT/'validation';dest.mkdir(exist_ok=False)
    physical={}
    for run,source in [(3,'Co60'),(12,'Am241'),(18,'Am241')]:
        key=f's{scale:g}_run{run}_{source}';cfg=read(OUT/'fit'/f'{key}_config.json')
        cfg['sipm']['sensor']['constant_pde']=.3
        cfg['optical']['yield_photons_per_MeV']*=scale
        cfg['status']='Frozen full-Geant4 candidate: nominalPDE0.3, shared scaled optical yield and passive signal transfer; parameters remain effective hypotheses'
        cfg['npe_profile']={'chosen_scale':scale,'selection':str((OUT/'selection.json').relative_to(PROJECT)),
            'no_numerical_high_PDE_in_this_config':True,
            'physical_interpretation':'Effective yield / electrical transfer family, not an independent measurement of scintillation yield, PDE or coupling. SiPM physical gain and nominal50ohm input unchanged.'}
        physical[run]=cfg
    entries={};native=[];replays=[]
    for run,source,n,seed,isnative in [(3,'Co60',8000,301029203,True),(12,'Am241',24000,301029212,True),
                                      (18,'Am241',24000,301029212,False),(12,'Co60',8000,301029203,False)]:
        key=f'run{run}_{source}';cfg=copy.deepcopy(physical[3 if source=='Co60' else 12])
        for k in ['sipm','readout_response','electronics_noise','signal_dependent_noise']:cfg[k]=copy.deepcopy(physical[run][k])
        cfg['validation_case']=key
        folder=dest/key;folder.mkdir();cfgpath=folder/'config.json';root=folder/'events.root';save(cfgpath,cfg)
        if isnative:command=[str(BUILD/'csi_sources'),str(cfgpath),source,str(n),str(seed),str(root)]
        else:command=[str(BUILD/'replay_response'),str(dest/('run3_Co60' if source=='Co60' else 'run12_Am241')/'events.root'),str(cfgpath),str(root)]
        entry={'key':key,'run':run,'source':source,'events':n,'seed':seed,'native':isnative,
            'config':str(cfgpath.relative_to(PROJECT)),'config_sha256':sha(cfgpath),'path':str(root.relative_to(PROJECT)),'command':command}
        entries[key]=entry;(native if isnative else replays).append(entry)
    protocol={'scope':__doc__,'code_sha256':sha(Path(__file__)),'selection_sha256':sha(OUT/'selection.json'),
      'run18_extension_sha256':sha(OUT/'run18_extension.json'),'q_edges':read(OUT/'protocol.json')['q_edges'],
      'chosen_scale':scale,'entries':entries,'native_jobs':native,'replay_jobs':replays,
      'binary_sha256':{name:sha(BUILD/name) for name in ['csi_sources','replay_response']},
      'independent_new_mothers':32000,'detector_PDE':.3,
      'validation':'Fresh optical and readout seeds. Frozen measured first60 closure and middle20/last20 checks, previously explored experimental data. No validation-driven refit.',
      'selection':'Same native ADC criteria and frozen experimental IDs, Q bins and PCA. No PID-score cut in MC.',
      'uncertainty':'512 whole-mother MC bootstrap and original-index block64 experimental bootstrap; fixed fitted model. Parameter degeneracy remains.',
      'check_scope':'Native optical yield changed, PDE restored to0.3; this full transport tests the replay approximation. Compare PID widths, full covariance, average waveforms, short-scale F and baseline.',
      'earlier_results':'Previous bandwidth archive is preserved; earlier MC may now serve calibration and cannot count as independent validation for the new profile.'}
    save(dest/'protocol.json',protocol)
    print('Frozen native validation at scale',scale,'yield',physical[3]['optical']['yield_photons_per_MeV'],
          'transmission',physical[3]['readout_response']['signal_transmission'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run18',action='store_true');p.add_argument('--freeze',action='store_true');a=p.parse_args()
    if a.run18:run18()
    if a.freeze:freeze()
