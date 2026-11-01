"""Audit inherited results, opposite-block fences and exact core/tail moment partitions."""
import argparse
import base64
import shutil
import nbformat
from pid_core_tail_common import *
from verify_acquisition_state import check_interval


def sources():
    return [PROJECT/'simulation'/n for n in ['pid_core_tail_common.py','prepare_pid_core_tail.py','analyze_pid_core_tail.py',
        'plot_pid_core_tail.py','summarize_pid_core_tail.py','verify_pid_core_tail.py','rc_refinement_common.py',
        'state_response_common.py','analyze_acquisition_state.py','prepare_acquisition_state.py','prepare_state_response.py',
        'extract_pid.py','reference.py','README.md','src/RootIO.cc','include/TrackInformation.hh']]+[
        PROJECT/'analysis/paper_baseline/publication_stats.py',PROJECT/'jupyter/codex_0001_build_pid_core_tail_notebook.py']


def check_hashes(mapping):
    for path,value in mapping.items():assert sha(PROJECT/path)==value,path
    return len(mapping)


def archive():
    assert not (OUT/'record.json').exists();v=read(OUT/'verification.json')
    check_hashes(v['outputs_sha256']);check_hashes(v['sources_sha256']);snap={}
    for f in sources():
        dest=OUT/'source_snapshot'/f.relative_to(PROJECT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dest);snap[relative(f)]=sha(dest)
    files={relative(f):sha(f) for f in sorted(OUT.rglob('*')) if f.is_file() and 'source_snapshot' not in f.parts}
    nb=PROJECT/'jupyter/codex_0001_csi_pid_core_tail.ipynb';files[relative(nb)]=sha(nb)
    save(OUT/'record.json',{'scope':__doc__,'outputs_sha256':files,'sources_sha256':snap,
        'checkpoint_commit':read(OUT/'protocol.json')['checkpoint_commit'],'parent_record_sha256':sha(PARENT/'record.json'),
        'unique_original_mothers':470,'supported_original_mothers':456,'new_independent_mothers':0,
        'new_native_replays':0,'default_changed':False})
    print('Archived',len(files),'outputs and',len(snap),'sources.',flush=True)


def main():
    assert not (OUT/'verification.json').exists()
    p=read(OUT/'protocol.json');d=read(OUT/'diagnostics.json');prior=read(PARENT/'evaluation.json');boot=load(OUT/'bootstrap.npz')
    report={'scope':__doc__,'inputs_checked':check_hashes(p['inputs_sha256'])}
    assert p['code_sha256']==sha(PROJECT/'simulation/prepare_pid_core_tail.py')
    assert p['common_code_sha256']==sha(PROJECT/'simulation/pid_core_tail_common.py')==d['common_code_sha256']
    assert d['code_sha256']==sha(PROJECT/'simulation/analyze_pid_core_tail.py') and d['protocol_sha256']==sha(OUT/'protocol.json')
    minimum=512;fences=0;partitions=0
    for source,run in SOURCES:
        truth=load(OUT/'arrays'/f'run{run}_ancestry.npz')
        np.testing.assert_allclose(truth['edep_by_origin_MeV'].sum(1),truth['edep_MeV'],rtol=1e-10,atol=1e-12)
        np.testing.assert_allclose(truth['charge_by_origin_C'].sum(1),truth['charge_C'],rtol=1e-10,atol=1e-20)
        for name,field,cut in [('label_energy90','edep_by_origin_MeV',.9),('label_energy95','edep_by_origin_MeV',.95),('label_charge90','charge_by_origin_C',.9)]:
            labels=[]
            for energy in truth[field]:
                total=float(sum(energy));label=4 if total<=0 else 3
                if total>0:
                    for category,origin in enumerate([1,2,3]):
                        if float(energy[origin])/total>=cut:label=category
                labels.append(label)
            np.testing.assert_array_equal(labels,truth[name])
        for seg in SEGMENTS:
            key=f'run{run}_{seg}';a=load(OUT/'arrays'/f'{key}_data.npz')
            for B in p['block_sizes']:
                ref=read(OUT/'references'/f'{key}_B{B}.json');computed=fit_reference(a,p,source,B);assert ref==computed
                for ef in [0,1]:
                    x=ref[str(ef)];train=np.array(x['training_ids']);test=np.array(x['evaluation_ids'])
                    assert not set(train)&set(test) and np.all((train//B)%2!=ef) and np.all((test//B)%2==ef)
                    fences+=1
                row=d['groups'][key][str(B)];bs=boot[f'{key}_B{B}'];absolute=boot[f'{key}_B{B}_absolute']
                for iv,v in enumerate(['data',*VARIANTS]):
                    cohort=a if v=='data' else native(run,seg,B,v)
                    zv=load(OUT/'arrays'/f'{key}_B{B}_{v}_z.npz');np.testing.assert_array_equal(cohort['ids'],zv['ids'])
                    z=standardized(cohort,p,source,ref,zv['fold']);np.testing.assert_array_equal(z,zv['z'])
                    measured0=row['data'] if v=='data' else row['variants'][v]['native']
                    computed0=diagnostics(cohort,p,source,z,full=True)
                    for n in ['widths','cov','tail_probability','tail_variance_share','joint_tail_covariance','score_shape_covariance','tail_score_shape_covariance']:
                        np.testing.assert_allclose(computed0[n],measured0[n],atol=1e-12,rtol=1e-12)
                    widths=np.array(measured0['widths']);assert np.all(np.diff(widths,axis=0)>=-1e-14)
                    cv=np.array(measured0['cov']);np.testing.assert_allclose(cv,cv.T,atol=1e-14);assert np.linalg.eigvalsh(cv).min()>-1e-14
                    for tc in measured0['joint_tail_covariance']:
                        assert np.linalg.eigvalsh(np.array(tc)).min()>-1e-14 and np.linalg.eigvalsh(cv-tc).min()>-1e-14
                    np.testing.assert_allclose(np.array(measured0['core_score_shape_covariance'])+measured0['tail_score_shape_covariance'],measured0['score_shape_covariance'],atol=1e-15)
                    # Direct event summation, independent of diagnostics' einsum tail aggregation.
                    rr=rows(cohort,p,source);mass=np.zeros((2,3));tailv=np.zeros((2,3));whole=np.zeros(3)
                    for b,ix,w,_ in rr:
                        xx=cohort['scores'][ix];mu=np.average(xx,axis=0,weights=w)
                        for j in range(3):
                            whole[j]+=sum(float(ww*(vv-mu[j])**2) for ww,vv in zip(w,xx[:,j]))
                            for k,bound in enumerate(FENCES):
                                use=np.abs(z[ix,j])>bound;mass[k,j]+=sum(float(ww) for ww,ok in zip(w,use) if ok)
                                tailv[k,j]+=sum(float(ww*(vv-mu[j])**2) for ww,vv,ok in zip(w,xx[:,j],use) if ok)
                    np.testing.assert_allclose(mass/len(rr),measured0['tail_probability'],atol=1e-13)
                    np.testing.assert_allclose(tailv/whole,measured0['tail_variance_share'],atol=1e-12);partitions+=1
                    intervals=row['data_tail_CI95'] if v=='data' else row['variants'][v]['tail_CI95']
                    for k in range(2):
                        for j in range(3):
                            for l in range(2):check_interval(absolute[:,iv,k,j,l],intervals[k][j][l])
                    if v=='data':continue
                    r=row['variants'][v];old=prior['groups'][key][str(B)]['variants'][v]
                    for newname,oldname in [(f'width50_{m}_ratio',n) for m,n in zip(METHODS,['IQR_ratio_ratio','IQR_time_ratio','IQR_PCA_ratio'])]+[(f'SD_{m}_ratio',n) for m,n in zip(METHODS,['SD_ratio_ratio','SD_time_ratio','SD_PCA_ratio'])]:
                        np.testing.assert_allclose(r['point'][newname],old['point'][oldname],atol=1e-12)
                        np.testing.assert_allclose(r['CI95'][newname]['CI95'],old['CI95'][oldname]['CI95'],atol=1e-12)
                    for j,n in enumerate(NAMES):
                        check_interval(bs[:,iv-1,j],r['CI95'][n]);check_interval(bs[:,iv-1,j]-bs[:,VARIANTS.index('coarse_spectrum'),j],r['paired_from_coarse'][n])
                        minimum=min(minimum,r['CI95'][n]['valid_draws'])
                    tp=truth_parts(cohort,p,source,z,truth)
                    for n in ['covariance_total','between_truth_mean_covariance','within_truth_covariance']:np.testing.assert_allclose(tp[n],r['ancestry'][n],atol=1e-14)
                print('Verified',key,B,'fences, widths, tails and paired intervals',flush=True)
    assert minimum>=128
    nbpath=PROJECT/'jupyter/codex_0001_csi_pid_core_tail.ipynb';nb=nbformat.read(nbpath,as_version=4);nbformat.validate(nb)
    code=[c for c in nb.cells if c.cell_type=='code'];assert all(c.execution_count for c in code)
    assert not any(o.output_type=='error' for c in code for o in c.outputs)
    images=[base64.b64decode(o.data['image/png']) for c in code for o in c.outputs if 'data' in o and 'image/png' in o.data]
    figs=read(OUT/'figure_record.json');assert len(images)==len(figs['figures'])==5
    for blob,name in zip(images,figs['figures']):assert blob==(OUT/'figures'/f'{name}.png').read_bytes()
    check_hashes(figs['inputs_sha256']);assert figs['code_sha256']==sha(PROJECT/'simulation/plot_pid_core_tail.py')
    decision=read(OUT/'decision.json');check_hashes(decision['inputs_sha256'])
    assert decision['code_sha256']==sha(PROJECT/'simulation/summarize_pid_core_tail.py')
    assert decision['table_sha256']==sha(OUT/'validation_table.csv') and not decision['default_changed']
    report.update(opposite_block_references=fences,direct_event_partitions=partitions,minimum_valid_bootstrap_draws=minimum,
        inherited_IQR_SD_points_and_intervals_exact=True,ancestry_energy_charge_conservation=True,
        notebook={'cells':len(nb.cells),'executed_cells':len(code),'exact_embedded_pngs':len(images)},
        new_native_replays=0,new_independent_mothers=0,default_changed=False)
    report['outputs_sha256']={relative(f):sha(f) for f in sorted(OUT.rglob('*')) if f.is_file() and 'source_snapshot' not in f.parts}
    report['outputs_sha256'][relative(nbpath)]=sha(nbpath);report['sources_sha256']={relative(f):sha(f) for f in sources()}
    save(OUT/'verification.json',report);print('All core/tail, ancestry, uncertainty and notebook checks passed.',flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--archive',action='store_true');args=ap.parse_args()
    archive() if args.archive else main()
