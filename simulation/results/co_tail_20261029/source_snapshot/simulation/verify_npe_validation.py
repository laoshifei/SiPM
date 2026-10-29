"""Verify and archive the photon-statistics/passive-transfer candidate."""
import argparse
import base64
import functools
import json
import platform
import shutil
from pathlib import Path
import numpy as np
import scipy
import ROOT
import nbformat
from prepare_npe_profile import OUT,PARENT,BUILD,PROJECT
from prepare_pid import read
from extract_pid import load,label_truth
from verify_pid import independent_scores,independent_auc
from pid_metrics import operating_point
from publication_stats import sha,save


@functools.lru_cache(None)
def digest(path):return sha(Path(path))


def mapping_hashes(mapping,base=PROJECT):
    for rel,expected in mapping.items():assert digest(base/rel)==expected,rel
    return len(mapping)


def same_signal(parent,child,ids):
    files=[ROOT.TFile.Open(str(p)) for p in [parent,child]]
    trees=[f.Get('Events') for f in files]
    for t in trees:t.SetBranchStatus('*',0);t.SetBranchStatus('signal_mV',1)
    for i in ids:
        for t in trees:t.GetEntry(int(i))
        np.testing.assert_array_equal(np.asarray(trees[0].signal_mV),np.asarray(trees[1].signal_mV))
    for f in files:f.Close()
    return len(ids)


def main(archive=False):
    if (OUT/'record.json').exists():
        assert not archive,'Archive already exists'
        record=read(OUT/'record.json')
        print('Verified frozen outputs:',mapping_hashes(record['outputs_sha256']),flush=True)
        print('Verified source snapshots:',mapping_hashes(record['source_sha256'],OUT/'source_snapshot'),flush=True)
        return
    protocol=read(OUT/'protocol.json');vp=read(OUT/'validation/protocol.json')
    report=read(OUT/'validation/assessment.json');selection=read(OUT/'selection.json')
    assert digest(PARENT/'record.json')==protocol['parent_record_sha256']
    parent=read(PARENT/'record.json')
    checks={'scope':__doc__,'parent_archive':{
        'record_sha256':protocol['parent_record_sha256'],
        'unchanged_outputs':mapping_hashes(parent['outputs_sha256']),
        'unchanged_source_snapshots':mapping_hashes(parent['source_sha256'],PARENT/'source_snapshot')}}
    print('Previous bandwidth archive unchanged.',flush=True)
    checks['frozen_native_source_files']=mapping_hashes(protocol['source_sha256'])
    for name,h in protocol['binary_sha256'].items():assert h==vp['binary_sha256'][name]==digest(BUILD/name)
    for record,name in [(protocol,'prepare_npe_profile.py'),(vp,'prepare_npe_validation.py'),
        (selection,'fit_npe_profile.py'),(report,'analyze_npe_validation.py')]:
        assert record['code_sha256']==digest(PROJECT/'simulation'/name),name
    assert vp['selection_sha256']==digest(OUT/'selection.json')
    assert vp['run18_extension_sha256']==digest(OUT/'run18_extension.json')
    assert report['protocol_sha256']==digest(OUT/'validation/protocol.json')
    assert selection['chosen_scale']==vp['chosen_scale']==2.
    pca=read(PROJECT/'analysis/paper_baseline/results/publication_20261028/point.json')['windows']['3000']['pca']
    entries=dict(protocol['entries']);extension=read(OUT/'run18_protocol.json')
    entries[extension['entry']['key']]=extension['entry']
    assert extension['parent_profile_protocol_sha256']==digest(OUT/'protocol.json')
    assert extension['selection_sha256']==digest(OUT/'selection.json')
    checks['profiles']={}
    for key,e in entries.items():
        assert digest(PROJECT/e['config'])==e['config_sha256']
        assert digest(PROJECT/e['parent'])==e['parent_sha256']
        meta=read(OUT/'profile/extraction'/f'{key}.json');fit=read(OUT/'fit'/f'{key}.json')
        assert meta['root_sha256']==digest(PROJECT/e['path'])
        assert meta['code_sha256']==digest(PROJECT/'simulation/extract_npe_profile.py')
        assert fit['code_sha256']==digest(PROJECT/'simulation/fit_npe_profile.py')
        assert fit['protocol_sha256']==digest(OUT/'protocol.json')
        assert fit['training_sha256']==digest(PROJECT/fit['training_path'])
        assert fit['config_sha256']==digest(PROJECT/fit['config'])
        old=load(PARENT/'validation/arrays'/f'new_run{e["run"]}_{e["source"]}.npz')
        new=load(OUT/'profile/arrays'/f'{key}.npz')
        np.testing.assert_array_equal(old['energy_by_origin'],new['all_energy'])
        np.testing.assert_allclose(independent_scores(new['waves'],pca),new['scores'],rtol=1e-10,atol=1e-10)
        row={'mothers':e['events'],'same_energy_deposits':True,'common_events':len(new['ids'])}
        if e['scale']==1:
            assert meta['replay']['equal_original_avalanche_events']==e['events']
            ids=np.flatnonzero(new['all_energy'].sum(axis=1)>0)
            row['legacy_exact_signal_deposited_events']=same_signal(PROJECT/e['parent'],PROJECT/e['path'],ids)
            row['scope']='Every nonzero-energy-deposit mother checked for bit-exact signal; all mothers checked for identical avalanche histories by native replay metadata.'
        checks['profiles'][key]=row
    print('Profile deposit histories and scale-one legacy signals verified.',flush=True)
    arrays={};configs={};checks['new_transport']={}
    for key,e in vp['entries'].items():
        assert digest(PROJECT/e['config'])==e['config_sha256']
        cfg=read(PROJECT/e['config']);configs[key]=cfg
        assert cfg['sipm']['sensor']['constant_pde']==.3
        assert cfg['readout_response']['signal_transmission']==.5
        oldcfg=read(PARENT/'validation'/f'new_run{e["run"]}_{e["source"]}'/'config.json')
        assert cfg['sipm']==oldcfg['sipm']
        for k in ['geometry','acquisition','electronics_noise']:assert cfg[k]==oldcfg[k],(key,k)
        for k,v in oldcfg['readout_response'].items():assert cfg['readout_response'][k]==v,(key,k)
        for k,v in oldcfg['optical'].items():
            assert cfg['optical'][k]==(v*2 if k=='yield_photons_per_MeV' else v),(key,k)
        h=np.asarray(cfg['signal_dependent_noise']['coefficients']);assert len(h)==1025
        np.testing.assert_allclose(h@h,1,atol=1e-12,rtol=0)
        meta=read(OUT/'validation/extraction'/f'{key}.json')
        assert meta['root_sha256']==digest(PROJECT/e['path'])
        assert meta['code_sha256']==digest(PROJECT/'simulation/extract_npe_profile.py')
        a=load(OUT/'validation/arrays'/f'{key}.npz');arrays[key]=a
        assert a['n_parent']==e['events']
        np.testing.assert_array_equal(a['ids'],np.flatnonzero(a['all_common']))
        np.testing.assert_array_equal(a['energy'],a['all_energy'][a['ids']])
        np.testing.assert_allclose(independent_scores(a['waves'],pca),a['scores'],rtol=1e-10,atol=1e-10)
        np.testing.assert_array_equal(label_truth(a['energy'])[0],a['truth'])
        if not e['native']:assert meta['replay']['equal_original_avalanche_events']==e['events']
        else:
            v=read((PROJECT/e['path']).with_name('events_validation.json'))
            assert v['passed'] and v['events']==e['events'] and v['ancestry_checks']
            assert v['max_relative_charge_balance_error']<1e-9
        checks['new_transport'][key]={'mothers':e['events'],'common':len(a['ids']),
            'independent_ADC_scores':True,'native_full_source_ancestry_charge_check':bool(e['native'])}
    for k in ['sipm','readout_response','electronics_noise','signal_dependent_noise']:
        assert configs['run12_Am241'][k]==configs['run12_Co60'][k]
    np.testing.assert_array_equal(arrays['run3_Co60']['all_energy'],arrays['run12_Co60']['all_energy'])
    np.testing.assert_array_equal(arrays['run12_Am241']['all_energy'],arrays['run18_Am241']['all_energy'])
    A=arrays['run12_Am241'];G=arrays['run12_Co60'];sa=A['scores'][A['truth']==0];sg=G['scores'][G['truth']==1]
    cuts=read(PROJECT/'simulation/results/pid_20261029/thresholds.json')
    for j,m in enumerate(report['methods']):
        r=report['common_run12_PID'][m]
        assert abs(independent_auc(sa[:,j],sg[:,j])-r['AUC'])<1e-12
        assert r['fixed_previous_threshold']==cuts['methods'][m]['threshold']
        assert r['operating']==operating_point(sa[:,j],sg[:,j],r['fixed_previous_threshold'])
    for row in report['checks'].values():
        for ci in row['IQR_ratio_CI95']:assert ci['valid_draws']>=480
    checks['common_response_PID_independently_reproduced']=True
    checks['new_independent_mothers']=sum(e['events'] for e in vp['entries'].values() if e['native'])
    assert checks['new_independent_mothers']==32000
    nbpath=PROJECT/'jupyter/codex_0001_csi_pid_photon_statistics.ipynb'
    nb=nbformat.read(nbpath,as_version=4);nbformat.validate(nb)
    code=[c for c in nb.cells if c.cell_type=='code'];images=0
    for c in code:
        assert c.execution_count is not None and all(o.output_type!='error' for o in c.outputs)
        for o in c.outputs:
            if 'data' in o and 'image/png' in o.data:
                assert base64.b64decode(o.data['image/png']).startswith(b'\x89PNG');images+=1
    assert images==4
    for f in (OUT/'figures').glob('*'):
        if f.suffix=='.png':assert f.read_bytes().startswith(b'\x89PNG')
        if f.suffix=='.pdf':assert f.read_bytes().startswith(b'%PDF-')
        if f.suffix=='.svg':assert '<svg' in f.read_text()[:3000]
    assert '100% tests passed' in (OUT/'logs/ctest.log').read_text()
    assert 'Ran 3 tests' in (OUT/'logs/replay_tests.log').read_text() and (OUT/'logs/replay_tests.log').read_text().rstrip().endswith('OK')
    checks['tests']={'ctest_targets':5,'response_replay_integration_tests':3,'passed':True}
    checks['notebook']={'executed_code_cells':len(code),'embedded_images':images,'valid':True}
    checks['versions']={'Python':platform.python_version(),'NumPy':np.__version__,'SciPy':scipy.__version__,'ROOT':ROOT.gROOT.GetVersion()}
    checks['caveat']='Engineering/analysis consistency checks do not establish unique physical parameters or full experimental bandwidth agreement; see decision.json.'
    save(OUT/'verification.json',checks)
    if archive:
        sources=set((PROJECT/'simulation').glob('*.py'))
        for folder in ['include','src','vendor','tests']:
            sources.update(p for p in (PROJECT/'simulation'/folder).rglob('*') if p.suffix in ['.py','.cc','.hh','.json'])
        sources.update([PROJECT/'simulation/main.cc',PROJECT/'simulation/replay_response.cc',PROJECT/'simulation/CMakeLists.txt',
            PROJECT/'simulation/README.md',PROJECT/'jupyter/codex_0001_build_pid_photon_statistics_notebook.py'])
        sources.update((PROJECT/'analysis/paper_baseline').glob('*.py'))
        source_hashes={};snapshot=OUT/'source_snapshot'
        for p in sorted(sources):
            rel=p.relative_to(PROJECT);target=snapshot/rel;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(p,target);source_hashes[str(rel)]=sha(target)
        outputs={str(p.relative_to(PROJECT)):digest(p) for p in sorted(OUT.rglob('*')) if p.is_file() and 'source_snapshot' not in p.parts}
        outputs[str(nbpath.relative_to(PROJECT))]=digest(nbpath)
        save(OUT/'record.json',{'scope':__doc__,'outputs_sha256':outputs,'source_sha256':source_hashes,
            'checkpoint_commit':protocol['checkpoint_commit'],'parent_record_sha256':protocol['parent_record_sha256'],
            'verification_sha256':digest(OUT/'verification.json'),'result':'See decision.json: conditional improvement and remaining limits, not a unique detector calibration.'})
        print('Archived',len(outputs),'outputs and',len(source_hashes),'source snapshots.',flush=True)
    print('All photon-statistics verification checks passed.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive',action='store_true');args=p.parse_args();main(args.archive)
