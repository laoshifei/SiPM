"""Recheck cohorts, summarize comparisons and archive this existing-data stage."""
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

BASE=Path(__file__).resolve().parent
PROJECT=BASE.parents[2]
SIM=PROJECT/'simulation'
PROFILE=SIM/'results/bias_response_profile_20261028_final'
MEAN_PROFILE=SIM/'results/bias_response_mean_profile_20261028'
COMPONENTS=SIM/'results/mc_components_20261028'
MOMENTS=SIM/'results/nonstationary_moments_20261028'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())
def rel(path):return str(path.relative_to(PROJECT))


report=read(BASE/'bias_response.json')
with np.load(BASE/'samples.npz') as z:arrays=dict(z)
checks={'scope':'Cohort reproducibility and descriptive cross-run checks; no response parameters exported','cohorts':{},'previous_metric_regression':{},'shape_comparisons':{},'pairs':{}}
for run,row in report['runs'].items():
    root=PROJECT/'analysis/paper_baseline/results'
    fpath=root/f'pha_run{int(run):04d}_features.npz';apath=root/f'phase2/run{int(run):04d}_features.npz'
    assert sha(fpath)==report['inputs_sha256'][rel(fpath)]
    assert sha(apath)==report['inputs_sha256'][rel(apath)]
    with np.load(fpath) as z:f=dict(z)
    with np.load(apath) as z:a={'eligible_primary':z['eligible_primary']}
    for group,row2 in row['groups'].items():
        key=f'run{run}_{group}';lo,hi=row2['original_event_range'];idx=f['event_index']
        basic=(f['diagnostic_flags']==0)&(f['t10_sample']>=384)&(idx>=lo)&(idx<hi)
        selected=basic&a['eligible_primary']&(f['amplitude']>=200)&((f['flags']&0x8000)==0)&(f['charge_long']>=report['q_edges'][0])&(f['charge_long']<=report['q_edges'][-1])
        available=np.flatnonzero(selected);expected=available if len(available)<=3000 else available[np.linspace(0,len(available)-1,3000,dtype=int)]
        np.testing.assert_array_equal(arrays[key+'_ids'],expected)
        noise=np.flatnonzero(basic);noise=noise if len(noise)<=3000 else noise[np.linspace(0,len(noise)-1,3000,dtype=int)]
        np.testing.assert_array_equal(arrays[key+'_noise_ids'],noise)
        assert row2['pulse']['n_available']==len(available)
        checks['cohorts'][key]={'pulse_n_available':len(available),'pulse_n_used':len(expected),'noise_n_used':len(noise),'event_ids_and_selection_match':True}

old_path=SIM/'results/fluctuations_20261028/metrics.npz'
with np.load(old_path) as old:
    for run,source in [(3,'Co60'),(12,'Am241')]:
        ids=arrays[f'run{run}_first60_ids'];old_ids=old[source+'_measured_ids'];mapping={int(event):j for j,event in enumerate(old_ids)}
        positions=[mapping[int(event)] for event in ids]
        np.testing.assert_array_equal(arrays[f'run{run}_first60_metrics'],old[source+'_measured_metrics'][positions])
        removed=np.setdiff1d(old_ids,ids)
        with np.load(PROJECT/f'analysis/paper_baseline/results/pha_run{run:04d}_features.npz') as z:removed_t=z['t10_sample'][removed]
        assert np.all(removed_t<384)
        checks['previous_metric_regression'][str(run)]={'original_n':len(old_ids),'current_n':len(ids),'exact_metric_match_on_shared_events':True,'removed_indices':removed.tolist(),'removed_t10_samples':removed_t.tolist(),'only_change':'Additional fixed pretrigger-spectrum support requirement t10>=384 samples'}

reference=arrays['run12_first60_shape'];mask=(arrays['time_ns']>=-100)&(arrays['time_ns']<=3500)
for run in [10,11,13,14,18]:
    wave=arrays[f'run{run}_first60_shape']
    checks['shape_comparisons'][f'{run}_versus_12']={'scope':'Measured versus measured, not MC accuracy','time_range_ns':[-100,3500],'area_normalized_peak_NRMSE':float(np.sqrt(np.mean((wave[mask]-reference[mask])**2))/reference.max())}
for first,second in [(12,18),(12,13),(12,3)]:
    one=report['runs'][str(first)]['groups']['first60']['pulse'];two=report['runs'][str(second)]['groups']['first60']['pulse']
    key=f'{first}_over_{second}'
    pair={'raw_median_50ns_ratio':one['fluctuations']['50ns']['p10_p50_p90'][1]/two['fluctuations']['50ns']['p10_p50_p90'][1],
          'spectral_20_80MHz_median_ratio':one['spectral_bands']['20-80MHz']['p10_p50_p90'][1]/two['spectral_bands']['20-80MHz']['p10_p50_p90'][1],
          'per_Q_bin_median_ratios':[]}
    for a,b in zip(one['charge_bins'],two['charge_bins']):
        if min(a['n'],b['n'])>=20:pair['per_Q_bin_median_ratios'].append({'q_low':a['low'],'q_high':a['high'],'n':[a['n'],b['n']],'ratios':(np.array(a['fluctuation_medians'])/b['fluctuation_medians']).tolist()})
    if one['equal_bin_mean_of_medians'] is not None and two['equal_bin_mean_of_medians'] is not None:
        pair['equal_six_bin_mean_of_medians_ratio']=(np.array(one['equal_bin_mean_of_medians'])/two['equal_bin_mean_of_medians']).tolist()
    checks['pairs'][key]=pair
checks['inputs_sha256']={rel(p):sha(p) for p in [BASE/'bias_response.json',BASE/'samples.npz',old_path]}
(BASE/'cohort_checks.json').write_text(json.dumps(checks,indent=2,allow_nan=False)+'\n')

build=read(SIM/'results/colored_noise_20261028/build_record.json')
assert all(sha(PROJECT/path)==expected for path,expected in build['source_sha256'].items())
assert sha(SIM/'build/csi_sources')==build['binary_sha256']
source=BASE/'executed_source/analyze_bias_response.py'
assert sha(source)==report['code_sha256']['simulation/analyze_bias_response.py']
for path,expected in report['code_sha256'].items():
    if path!='simulation/analyze_bias_response.py':assert sha(PROJECT/path)==expected
profile=read(PROFILE/'profiles.json')
for path,expected in profile['code_sha256'].items():
    actual=PROFILE/'executed_source/profile_bias_response.py' if path=='simulation/profile_bias_response.py' else PROJECT/path
    assert sha(actual)==expected
for path in [MEAN_PROFILE/'profiles.json',COMPONENTS/'components.json',MOMENTS/'moments.json']:
    data=read(path)
    for source,expected in data['code_sha256'].items():assert sha(PROJECT/source)==expected
assert read(MEAN_PROFILE/'profiles.json')['statistic']=='mean'
components=read(COMPONENTS/'components.json');moments=read(MOMENTS/'moments.json')
for source in ['Co60','Am241']:
    assert components['sources'][source]['original_metrics_exact']
    assert components['sources'][source]['max_absolute_identity_error_ADC']<1e-9
    assert moments['sources'][source]['max_native_signal_replay_error_mV']==0.
    assert moments['sources'][source]['max_relative_32_vs_64_phase_difference']<1e-3
plot=read(BASE/'plot_record.json')
assert sha(SIM/'analyze_bias_response.py')==plot['plot_code_sha256']
for path,expected in plot['inputs_sha256'].items():assert sha(BASE/path)==expected
log=SIM/'results/bias_response_tests_20261028_final.log'
assert 'Ran 6 tests' in log.read_text() and '\nOK\n' in log.read_text()
for name in ['mc_components_tests_20261028.log','nonstationary_moments_tests_20261028.log']:
    log_text=(SIM/'results'/name).read_text();assert 'Ran 2 tests' in log_text and '\nOK\n' in log_text

sources=[PROJECT/'README.md',SIM/'README.md',SIM/'RESULTS.md']
sources+=[SIM/name for name in ['analyze_bias_response.py','profile_bias_response.py','characterize_readout.py','diagnose_fluctuations.py','diagnose_charge.py','reference.py','profile_signal_response.py','compare.py','compare_absolute.py','validate_joint.py','screen_joint.py','calibrate.py','tests/test_bias_response.py']]
sources+=[PROJECT/'analysis/paper_baseline'/name for name in ['prepare.py','refine_waveforms.py']]
sources+=[SIM/name for name in ['diagnose_mc_components.py','diagnose_nonstationary_moments.py','tests/test_mc_components.py','tests/test_nonstationary_moments.py']]
for path in sources:
    target=BASE/'source_snapshot'/path.relative_to(PROJECT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
    assert sha(path)==sha(target)
outputs=[]
for folder in [BASE,PROFILE,MEAN_PROFILE,COMPONENTS,MOMENTS]:
    outputs += [p for p in folder.rglob('*') if p.is_file() and 'source_snapshot' not in p.parts and p.name not in ['record.json','finalize.log']]
outputs += [SIM/'results'/name for name in ['bias_response_20261028.log','bias_response_20261028_replot.log','bias_response_profile_20261028_final.log','bias_response_tests_20261028_final.log']]
outputs += [SIM/'results'/name for name in ['bias_response_mean_profile_20261028.log','mc_components_20261028.log','nonstationary_moments_20261028.log','mc_components_tests_20261028.log','nonstationary_moments_tests_20261028.log']]
record={'stage':'Existing bias series, conditional response profiles, exact native decomposition/replay and finite-window avalanche self terms',
    'archived_utc':datetime.now(timezone.utc).isoformat(),
    'result':'Mean pulse shapes are close across several Am runs, but short-time fluctuations remain run dependent at common ADC charge. Do not export a voltage-gain calibration or per-source SiPM gain from these profiles.',
    'verification':{'analysis_checks_passed':10,'cohorts_checked':len(checks['cohorts']),'pulse_waveforms_used':sum(v['pulse_n_used'] for v in checks['cohorts'].values()),'baseline_waveforms_used':sum(v['noise_n_used'] for v in checks['cohorts'].values()),
        'original_metric_regression_exact':True,'native_Geant4_source_and_executable_unchanged':True,'new_Geant4_runs':0,
        'native_signal_replays':158,'maximum_native_signal_replay_difference_mV':0.,'maximum_variance_decomposition_error_ADC':max(v['max_absolute_identity_error_ADC'] for v in components['sources'].values()),
        'maximum_phase_grid_relative_change':max(v['max_relative_32_vs_64_phase_difference'] for v in moments['sources'].values())},
    'numeric_checks':rel(BASE/'cohort_checks.json'),'native_build_reference':rel(SIM/'results/colored_noise_20261028/build_record.json'),
    'notes':['Only runs3/12 first60 retain training roles. Bias groups remain exploratory checks; existing selections/split arrays are unchanged.',
        'Extraction executed source preserved and hash matched. Final plot-only refresh adds error bars and improves labels without changing numeric results.',
        'The preserved median profile is bias_response_profile_20261028_final, with exact executed source. Mean is now the default statistic and its separate result is bias_response_mean_profile_20261028. The earliest profile directory is an intermediate plot/layout version.',
        'Five/ten/twenty-percent profile regions are descriptive, grid-limited and not confidence intervals. Effective ADC area is not measured SPE gain.',
        'Stationary expectation differs by7-39% from full-ADC event medians, by7-17% from native signal means. The actual-time finite-window self-term diagnostic differs by0.2-2.6% from native group means; paired bootstrap intervals contain zero. This is internal MC consistency, not measured-detector validation.',
        'No new detector calibration configuration, Git commit or push created in this stage.'],
    'source_sha256':{rel(p):sha(p) for p in sources},'output_sha256':{rel(p):sha(p) for p in sorted(set(outputs))}}
(BASE/'record.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
print(json.dumps({'verification':record['verification'],'source_files':len(sources),'output_files':len(record['output_sha256']),'record':rel(BASE/'record.json')},indent=2))
