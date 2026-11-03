"""Freeze and evaluate linear intensity conditioning without modifying any ADC."""
import concurrent.futures
import argparse
import subprocess
from intensity_condition_common import *
from test_intensity_condition import run as algebra_test


def prepare():
    assert not OUT.exists()
    tests=algebra_test();old=read(TIME/'protocol.json');record=read(TIME/'record.json')
    paths=[TIME/'record.json',TIME/'protocol.json',TIME/'analysis.json',TIME/'verification.json',
        SPLIT/'analysis_protocol.json']
    for source,_ in SOURCES:
        paths += [TIME/'arrays'/f'{source}_{v}.npz' for v in ['data',*VARIANTS]]
        paths += [TIME/'evaluation'/f'{source}_bootstrap.npz',SPLIT/'evaluation'/f'{source}_counts.npz']
    for path in paths:
        if relative(path) in record['outputs_sha256']:assert sha(path)==record['outputs_sha256'][relative(path)],path
    for name in ['evaluation','figures','source_snapshot']:(OUT/name).mkdir(parents=True,exist_ok=True)
    code=['simulation/intensity_condition_common.py','simulation/analyze_intensity_condition.py','simulation/test_intensity_condition.py']
    inherited=['simulation/time_window_common.py','simulation/charge_scale_common.py','simulation/spectral_split_common.py']
    save(OUT/'protocol.json',{'checkpoint_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        **{k:old[k] for k in ['q_edges','support_bins','conditional_cuts']},'predictors':PREDICTORS,'scales_ns':SCALES,
        'global_metrics':GLOBAL_NAMES,'local_metrics':LOCAL_NAMES,'code_sha256':{p:sha(PROJECT/p) for p in code},
        'inherited_code_sha256':{p:sha(PROJECT/p) for p in inherited},'inputs_sha256':{relative(p):sha(p) for p in sorted(set(paths))},
        'scope':'Describe how existing temporal variance relates to measured pulse intensity; no fit to a native noise model and no ADC, default, PID, or selection modification.',
        'design':'Five separate one-dimensional linear regressions, one per four local pulse means and one whole post-window mean. All divided by the inherited opposite-fold per-Q mu0; no new amplitude scale fitted. Every regression predicts the same seven exact temporal terms at all three smoothing scales.',
        'weights':'Fit within each original current-Q bin, retaining original half-bin masses. Each data/MC sample has its own descriptive coefficients. Aggregate the within-Q covariances with equal bin weight. No intensity cuts or intensity reweighting.',
        'partition':'For each predictor x, c=Cov(T,x), vx=Var(x); beta=c/vx; Clinear=c*c.T/vx and Cres=C-Clinear. This in-sample algebraic partition is exact but does not identify physical noise.',
        'crossfit':'Also fit intercept and slope on the opposite original fold within each Q bin and data/MC sample; apply to the evaluation fold. Recombine residuals under the original full-bin half weights, then center within Q. Between-fold residual mean differences remain included. No threshold tuning or parameter selection.',
        'local':'For window k, report diagonal variance of Tk conditioned on its own local mean. Tk=(249/999)*Vwindow/mu0, not local F and not the sum of all original F terms.',
        'validity':'Each full bin and each training fold needs at least5 multiplicity-weighted events and positive support in every target half. All predictor variances must exceed1e-14. Mark whole draws invalid rather than drop Q bins or predictors.',
        'uncertainty':'Reuse all512 paired block/mother multiplicities, re-estimating the new descriptive regressions within every draw. Previous physical/model fits, references, selections and realized noise remain fixed. No new independent data or blind-validation claim.',
        'overlap':'Report point MC mass inside each measured weighted5th–95th predictor interval per Q bin, using cumulative-weight interpolation. This does not remove or reweight events.',
        'limits':'Intensity and variance come from the same noisy ADC, and source mixtures, finite pulse curvature and reference uncertainty remain. Linear residuals are diagnostic statistics, not corrected ADC or independent noise measurements. Out-of-fold statistics are internal checks on already explored events; intervals lack multiplicity/systematics corrections.',
        'new_native_readouts':0,'new_independent_mothers':0,'new_experimental_events':0,'default_changed':False,'official_PID_changed':False})
    save(OUT/'algebra_tests.json',tests)
    print('Frozen five one-dimensional intensity controls, opposite-fold checks and512 paired draws.',flush=True)


def one(source):
    p=read(OUT/'protocol.json');counts=load(SPLIT/'evaluation'/f'{source}_counts.npz')
    arrays={v:load(TIME/'arrays'/f'{source}_{v}.npz') for v in ['data',*VARIANTS]}
    previous=read(TIME/'analysis.json')[source];previous_draws=load(TIME/'evaluation'/f'{source}_bootstrap.npz')['contrasts']
    summaries={};report={'source':source,'summaries':{},'cells':{},'variants':{},'overlap':{},'valid_draws':{}}
    for v,a in arrays.items():
        c=counts['data_counts'] if v=='data' else counts['MC_counts']
        if v!='data':np.testing.assert_array_equal(a['ids'],counts['ids'])
        s,details,valid=summarize(a,p,source,np.vstack([np.ones((1,len(a['ids'])),int),c]),True)
        assert all(mask[0] for mask in valid.values());summaries[v]=s;report['summaries'][v]={k:value[0] for k,value in s.items()};report['cells'][v]=details
        report['valid_draws'][v]={k:int(mask[1:].sum()) for k,mask in valid.items()}
        np.testing.assert_allclose(s['cov'][0],previous['summaries'][v]['cov'],rtol=1e-9,atol=1e-10)
        np.testing.assert_allclose(s['mean'][0],previous['summaries'][v]['mean'],rtol=1e-10,atol=1e-10)
    gs=[];ls=[]
    for vi,v in enumerate(VARIANTS):
        g,l=contrast(summaries[v],summaries['data']);gs.append(g[1:]);ls.append(l[1:])
        for k in range(len(PREDICTORS)):
            np.testing.assert_allclose(g[1:,k,:,0],previous_draws[:,vi,:,0],rtol=1e-8,atol=1e-9)
        report['variants'][v]={}
        for name,z,metrics,labels in [('global',g,GLOBAL_NAMES,PREDICTORS),('local',l,LOCAL_NAMES,PREDICTORS[:4])]:
            report['variants'][v][name]={label:{str(scale):{'point':dict(zip(metrics,z[0,j,k])),
                'CI95':{n:interval(z[1:,j,k,i]) for i,n in enumerate(metrics)}} for k,scale in enumerate(SCALES)} for j,label in enumerate(labels)}
        report['overlap'][v]=range_overlap(arrays[v],arrays['data'],p,source)
    atomic_npz(OUT/'evaluation'/f'{source}_bootstrap.npz',global_contrasts=np.stack(gs,1),local_contrasts=np.stack(ls,1))
    save(OUT/'evaluation'/f'{source}.json',serialize(report));print(source,'intensity regressions and all inherited global SD draws verified',report['valid_draws'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--resume-export-fix',action='store_true');args=parser.parse_args()
    if args.resume_export_fix:
        assert (OUT/'pre_export_fix_protocol.json').exists() and not (OUT/'export_fix_record.json').exists()
        assert not list((OUT/'evaluation').glob('*'))
        old=read(OUT/'protocol.json');p=dict(old)
        p['validity']='Each original full bin needs at least5 multiplicity-weighted events and positive support in every target half. Linear fits additionally require all predictor variances above1e-14; cross-fold fits additionally require the same count/support conditions in each training fold. Invalidate the entire affected estimator draw, never drop bins. Preserve original unconditioned moments even when a new cross-fold fit is invalid.'
        p['code_sha256']={name:sha(PROJECT/name) for name in old['code_sha256']}
        save(OUT/'protocol.json',p)
        save(OUT/'export_fix_record.json',{'reason':'The initial closure check rejected NaNs introduced by sparse training-fold bootstrap counts. Separate validity masks retain all original moments and explicitly mark unavailable new regressions. No cuts, regressions, thresholds, multiplicities, or model parameters changed.',
            'prior_protocol_sha256':sha(OUT/'pre_export_fix_protocol.json'),'revised_protocol_sha256':sha(OUT/'protocol.json'),
            'prior_code_sha256':old['code_sha256'],'revised_code_sha256':p['code_sha256'],'prior_statistical_outputs':0})
        save(OUT/'algebra_tests_after_export_fix.json',algebra_test())
    else:prepare()
    p=read(OUT/'protocol.json')
    for path,h in {**p['code_sha256'],**p['inherited_code_sha256']}.items():assert sha(PROJECT/path)==h,path
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        for result in executor.map(one,[s for s,_ in SOURCES]):pass
    save(OUT/'analysis.json',{s:read(OUT/'evaluation'/f'{s}.json') for s,_ in SOURCES})
