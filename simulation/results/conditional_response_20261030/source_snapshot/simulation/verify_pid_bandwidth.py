"""Verify and freeze the PID bandwidth response study without replacing parents."""
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
from reference import PROJECT
from prepare_pid import read
from prepare_pid_bandwidth import OUT,PARENT,PID
from publication_stats import save,sha
from extract_pid import load,label_truth
from verify_pid import independent_scores,independent_auc
from pid_metrics import operating_point

@functools.lru_cache(None)
def digest(path):return sha(Path(path))


def mapping_hashes(mapping,base=PROJECT):
    for rel,expected in mapping.items():assert digest(base/rel)==expected,rel
    return len(mapping)


def main(archive=False):
    if (OUT/'record.json').exists():
        assert not archive,'Archive already exists'
        record=read(OUT/'record.json')
        print('Verified frozen outputs:',mapping_hashes(record['outputs_sha256']),flush=True)
        print('Verified source snapshots:',mapping_hashes(record['source_sha256'],OUT/'source_snapshot'),flush=True)
        return
    protocol=read(OUT/'protocol.json');vprotocol=read(OUT/'validation/protocol.json')
    report=read(OUT/'validation/assessment.json');diag=read(OUT/'diagnosis.json');fit=read(OUT/'covariance_fit/fit.json')
    checks={'scope':__doc__,'parents':{}}
    for path,key in [(PARENT,'parent_waveform_record_sha256'),(PID,'parent_PID_record_sha256')]:
        assert digest(path/'record.json')==protocol[key]
        record=read(path/'record.json')
        checks['parents'][path.name]={'unchanged_outputs':mapping_hashes(record['outputs_sha256']),
             'unchanged_snapshots':mapping_hashes(record['source_sha256'],path/'source_snapshot')}
        print('Parent archive unchanged:',path.name,flush=True)
    for record,path in [(protocol,'simulation/prepare_pid_bandwidth.py'),(vprotocol,'simulation/prepare_bandwidth_validation.py'),
        (report,'simulation/analyze_bandwidth_validation.py'),(diag,'simulation/diagnose_pid_bandwidth.py'),(fit,'simulation/fit_pid_bandwidth.py')]:
        assert record['code_sha256']==digest(PROJECT/path),path
    assert vprotocol['fit_sha256']==digest(OUT/'covariance_fit/fit.json')
    assert protocol['binary_sha256']==vprotocol['replay_binary_sha256']==digest(PROJECT/'simulation/build/replay_response')
    assert vprotocol['native_binary_sha256']==digest(PROJECT/'simulation/build/csi_sources')
    for state,entry in protocol['entries'].items():
        assert digest(PROJECT/entry['parent_root'])==entry['parent_sha256']
        assert digest(PROJECT/entry['base_root'])==diag['MC'][state]['base_root_sha256']
        assert digest(PROJECT/entry['base_config'])==entry['base_config_sha256']
        assert diag['MC'][state]['same_signal'] and diag['MC'][state]['maximum_score_identity_error']<1e-10
        old=read(PROJECT/entry['parent_config']);new=read(OUT/f'covariance_fit/{state}_config.json')
        for k in ['sipm','optical','geometry','readout_response','electronics_noise','acquisition']:assert old[k]==new[k],(state,k)
        assert old['signal_dependent_noise']['event_variance_cv']==new['signal_dependent_noise']['event_variance_cv']
        h=np.asarray(new['signal_dependent_noise']['coefficients']);assert len(h)==1025
        np.testing.assert_allclose(h@h,1,rtol=0,atol=1e-12)
    checks['only_signal_residual_spectrum_changed']=True
    for key,entry in diag['measured'].items():assert digest(PROJECT/entry['input'])==entry['input_sha256']
    pca=read(PROJECT/'analysis/paper_baseline/results/publication_20261028/point.json')['windows']['3000']['pca']
    checks['native']={}
    for key,entry in vprotocol['entries'].items():
        assert digest(PROJECT/entry['config'])==entry['config_sha256']
        extraction=read(OUT/'validation/extraction'/f'{key}.json')
        assert digest(PROJECT/entry['path'])==extraction['root_sha256']
        a=load(OUT/'validation/arrays'/f'{key}.npz')
        np.testing.assert_array_equal(a['common_ids'],np.flatnonzero(a['common']))
        np.testing.assert_allclose(independent_scores(a['common_waves'],pca),a['scores'][a['common']],rtol=1e-10,atol=1e-10)
        labels,_=label_truth(a['energy_by_origin']);np.testing.assert_array_equal(labels,a['truth'])
        assert len(a['ids'])==entry['events']
        if not entry['native']:assert extraction['replay']['equal_original_avalanche_events']==entry['events']
        checks['native'][key]={'mothers':len(a['ids']),'common':len(a['common_ids']),'scores_independently_reproduced':True}
    A=load(OUT/'validation/arrays/new_run12_Am241.npz');G=load(OUT/'validation/arrays/new_run12_Co60.npz')
    sa=A['scores'][A['common']&(A['truth']==0)];sg=G['scores'][G['common']&(G['truth']==1)]
    oldcuts=read(PID/'thresholds.json')
    for j,method in enumerate(report['methods']):
        r=report['common_state_PID'][method]
        assert abs(independent_auc(sa[:,j],sg[:,j])-r['AUC'])<1e-12
        assert r['fixed_previous_threshold']==oldcuts['methods'][method]['threshold']
        assert operating_point(sa[:,j],sg[:,j],r['fixed_previous_threshold'])==r['operating']
    for run in [3,12,18]:
        source='Co60' if run==3 else 'Am241'
        old=load(OUT/'validation/arrays'/f'old_run{run}_{source}.npz')
        new=load(OUT/'validation/arrays'/f'new_run{run}_{source}.npz')
        np.testing.assert_array_equal(old['energy_by_origin'],new['energy_by_origin'])
        for seg in ['first60','middle20','last20']:
            row=report['checks'][f'run{run}_{seg}'];assert row['support_bins']==fit['states'][f'run{run}']['support_bins']
            for version in ['before','after']:
                for ci in row[version]['IQR_ratio_CI95']:assert ci['valid_draws']>=480
    nbpath=PROJECT/'jupyter/codex_0001_csi_pid_bandwidth.ipynb'
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
    checks['notebook']={'executed_code_cells':len(code),'embedded_images':images,'valid':True}
    checks['python']=platform.python_version();checks['numpy']=np.__version__;checks['scipy']=scipy.__version__;checks['ROOT']=ROOT.gROOT.GetVersion()
    checks['native_source_changed']=False;checks['unit_tests']='tests/test_pid_bandwidth.py:2 independent waveform/covariance checks passed; log archived'
    save(OUT/'verification.json',checks)
    if archive:
        sources=set((PROJECT/'simulation').glob('*.py'))
        for folder in ['include','src','vendor','tests']:
            sources.update(p for p in (PROJECT/'simulation'/folder).rglob('*') if p.suffix in ['.py','.cc','.hh','.json'])
        sources.update([PROJECT/'simulation/main.cc',PROJECT/'simulation/replay_response.cc',PROJECT/'simulation/CMakeLists.txt',PROJECT/'simulation/README.md',
          PROJECT/'jupyter/codex_0001_build_pid_bandwidth_notebook.py'])
        sources.update((PROJECT/'analysis/paper_baseline').glob('*.py'))
        source_hashes={};snapshot=OUT/'source_snapshot'
        for path in sorted(sources):
            rel=path.relative_to(PROJECT);target=snapshot/rel;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,target);source_hashes[str(rel)]=sha(target)
        outputs={str(p.relative_to(PROJECT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and 'source_snapshot' not in p.parts}
        outputs[str(nbpath.relative_to(PROJECT))]=sha(nbpath)
        save(OUT/'record.json',{'scope':__doc__,'outputs_sha256':outputs,'source_sha256':source_hashes,
          'parent_waveform_record_sha256':protocol['parent_waveform_record_sha256'],
          'parent_PID_record_sha256':protocol['parent_PID_record_sha256'],
          'verification_sha256':sha(OUT/'verification.json'),'result':'See decision.json for partial closure and unresolved discrepancies.'})
        print('Archived',len(outputs),'outputs and',len(source_hashes),'source snapshots.',flush=True)
    print('All PID bandwidth verification checks passed.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive',action='store_true');args=p.parse_args();main(args.archive)
