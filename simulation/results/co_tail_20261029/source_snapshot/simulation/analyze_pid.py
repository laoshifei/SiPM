"""Truth-aware PID, held-out operating points and common-readout sensitivity."""
import csv
import json
from pathlib import Path
import numpy as np
from prepare_pid import OUT,PARENT,STATES,SOURCES,PROJECT,read
from extract_pid import load,label_truth,NAMES
from publication_stats import sha,save
from shape_mixture import fit_mixture
from pid_metrics import auc,roc,operating_point,gaussian_equivalent_fom

METHODS=('qdc','negative_mean_time','pca_projection')
CASES=STATES+('original_per_run',)


def combined(case,protocol):
    parts=[]
    for s,source in enumerate(SOURCES):
        state=case if case!='original_per_run' else ('run3' if s==0 else 'run12')
        d=load(OUT/'arrays'/f'{state}_{source}.npz')
        d['source']=np.full(len(d['ids']),s,dtype=np.int8)
        d['mother']=d['ids']+(0 if s==0 else 4000)
        d['bin']=np.clip(np.searchsorted(protocol['q_edges'],d['q'],'right')-1,0,5)
        parts.append(d)
    return {k:np.concatenate([p[k] for p in parts]) for k in parts[0]}


def ci(values):
    x=np.asarray(values,float);ok=np.isfinite(x)
    return {'interval95':np.quantile(x[ok],[.025,.975]).tolist() if ok.sum()>=20 else None,
            'valid_draws':int(ok.sum()),'total_draws':len(x)}


def ks(a,b):
    a=np.sort(a);b=np.sort(b);x=np.r_[a,b]
    return float(np.max(abs(np.searchsorted(a,x,'right')/len(a)-np.searchsorted(b,x,'right')/len(b))))


def freeze_thresholds(primary,protocol):
    mask=primary['common']&(primary['split']==0)&(primary['truth']==0)
    result={'scope':'Primary common run12 response; empirical95th percentile (higher order statistic) of training-alpha scores.',
        'protocol_sha256':sha(OUT/'protocol.json'),'code_sha256':sha(Path(__file__)),
        'split_sha256':sha(OUT/'mother_splits.npz'),'alpha_mothers':primary['mother'][mask].tolist(),
        'used_test_mothers':False,'methods':{}}
    for j,method in enumerate(METHODS):
        x=primary['scores'][mask,j];threshold=float(np.quantile(x,.95,method='higher'))
        result['methods'][method]={'threshold':threshold,'n_training_alpha':len(x),
            'training_retained_fraction':float(np.mean(x<=threshold)),'direction':'alpha-like if score<=threshold'}
    save(OUT/'thresholds.json',result)
    return result


def bootstrap_counts():
    rng=np.random.default_rng(291029702);parts=[]
    for n in [4000,24000]:
        parts.append(np.array([np.bincount(rng.integers(n,size=n),minlength=n) for _ in range(512)],dtype=np.uint16))
    counts=np.concatenate(parts,axis=1)
    np.savez_compressed(OUT/'bootstrap_mother_counts.npz',counts=counts)
    return counts


def analyze_case(case,d,thresholds,counts,protocol):
    common=d['common'];test=common&(d['split']==1);truth=d['truth']
    row={'scope':'Common response for both sources' if case!='original_per_run' else 'Diagnostic only: original different per-run responses, confounded with source.',
         'n_common':int(common.sum()),'n_test':int(test.sum()),'methods':{},'truth_sensitivity':{}}
    draws={'auc':np.full((512,3),np.nan),'fom':np.full((512,3),np.nan),
        'binned_auc':np.full((512,3),np.nan),'fom_per_bin':np.full((512,3,6),np.nan)}
    testA=np.flatnonzero(test&(truth==0));testG=np.flatnonzero(test&(truth==1))
    allA=np.flatnonzero(common&(truth==0));allG=np.flatnonzero(common&(truth==1))
    row['truth_counts_test']=dict(zip(NAMES,np.bincount(truth[test],minlength=5).tolist()))
    bin_indices=[]
    for b in range(6):
        bin_indices.append((testA[d['bin'][testA]==b],testG[d['bin'][testG]==b],
                            allA[d['bin'][allA]==b],allG[d['bin'][allG]==b]))
    support=[b for b,(a,g,_,_) in enumerate(bin_indices) if min(len(a),len(g))>=5]
    row['within_bin_AUC_support']=support
    for j,method in enumerate(METHODS):
        score=d['scores'][:,j];threshold=thresholds['methods'][method]['threshold']
        r={'test_AUC':auc(score[testA],score[testG]),'operating':operating_point(score[testA],score[testG],threshold),
           'descriptive_all_common_truth_FOM':gaussian_equivalent_fom(score[allA],score[allG]),
           'per_Q':[],'test_confusion_by_truth':{},'source_mixture_fits':[]}
        curve=roc(score[testA],score[testG])
        np.savez_compressed(OUT/'curves'/f'{case}_{method}.npz',**curve)
        r['ROC_note']='Curve includes implicit initial threshold -infinity; remaining thresholds are finite distinct scores. Lower score means alpha-like.'
        for t,name in enumerate(NAMES):
            x=score[test&(truth==t)]
            r['test_confusion_by_truth'][name]={'alpha_like':int(np.sum(x<=threshold)),'other':int(np.sum(x>threshold)),'n':len(x)}
        for b,(a,g,aa,gg) in enumerate(bin_indices):
            cell={'bin':b,'test_alpha':len(a),'test_gamma':len(g),'all_alpha':len(aa),'all_gamma':len(gg)}
            if min(len(a),len(g))>=5:
                cell.update(test_AUC=auc(score[a],score[g]),operating=operating_point(score[a],score[g],threshold))
            if min(len(aa),len(gg))>=10:cell['truth_FOM']=gaussian_equivalent_fom(score[aa],score[gg])
            r['per_Q'].append(cell)
            ix=common&(d['bin']==b)
            ns=np.bincount(d['source'][ix],minlength=2)
            fitrow={'bin':b,'source_counts':ns.tolist(),'valid':False}
            if ns.min()>=10:
                fit=fit_mixture(score[ix],d['source'][ix],max_iter=600,tolerance=1e-7)
                fitrow.update(fit=fit,valid=bool(fit['converged'] and not fit['at_sigma_floor']))
            r['source_mixture_fits'].append(fitrow)
        r['within_bin_equal_mean_AUC']=float(np.mean([r['per_Q'][b]['test_AUC'] for b in support])) if support else None
        for k,c in enumerate(counts):
            wa=c[d['mother'][testA]];wg=c[d['mother'][testG]]
            if min(wa.sum(),wg.sum())>0:draws['auc'][k,j]=auc(score[testA],score[testG],wa,wg)
            wa=c[d['mother'][allA]];wg=c[d['mother'][allG]]
            if min(wa.sum(),wg.sum())>1:draws['fom'][k,j]=gaussian_equivalent_fom(score[allA],score[allG],wa,wg)
            bin_auc=[]
            for b,(a,g,aa,gg) in enumerate(bin_indices):
                if b in support:
                    wa=c[d['mother'][a]];wg=c[d['mother'][g]]
                    bin_auc.append(auc(score[a],score[g],wa,wg) if min(wa.sum(),wg.sum())>0 else np.nan)
                if min(len(aa),len(gg))>=10:
                    wa=c[d['mother'][aa]];wg=c[d['mother'][gg]]
                    if min(wa.sum(),wg.sum())>1:
                        draws['fom_per_bin'][k,j,b]=gaussian_equivalent_fom(score[aa],score[gg],wa,wg)
            if bin_auc:draws['binned_auc'][k,j]=float(np.mean(bin_auc))
        r['test_AUC_uncertainty']=ci(draws['auc'][:,j]);r['truth_FOM_uncertainty']=ci(draws['fom'][:,j])
        r['binned_AUC_uncertainty']=ci(draws['binned_auc'][:,j])
        for b,c in enumerate(r['per_Q']):
            if 'truth_FOM' in c:c['truth_FOM_uncertainty']=ci(draws['fom_per_bin'][:,j,b])
        row['methods'][method]=r
    for desc,labels in [('energy80',label_truth(d['energy_by_origin'],.8)[0]),
                         ('energy95',label_truth(d['energy_by_origin'],.95)[0]),('charge90',d['truth_charge'])]:
        a=np.flatnonzero(test&(labels==0));g=np.flatnonzero(test&(labels==1))
        values={'test_counts':dict(zip(NAMES,np.bincount(labels[test],minlength=5).tolist())),'methods':{}}
        if min(len(a),len(g))>0:
            for j,method in enumerate(METHODS):
                score=d['scores'][:,j]
                values['methods'][method]={'test_AUC':auc(score[a],score[g]),
                    'operating':operating_point(score[a],score[g],thresholds['methods'][method]['threshold'])}
        row['truth_sensitivity'][desc]=values
    np.savez_compressed(OUT/'curves'/f'{case}_bootstrap.npz',**draws)
    print(case,'held-out counts',row['truth_counts_test'],
          {m:row['methods'][m]['operating'] for m in METHODS},flush=True)
    return row,draws


def experimental_comparison(datasets,thresholds,protocol):
    groups={}
    for run in [3,12,18]:
        for segment in ['first60','middle20','last20']:
            key=f'run{run}_{segment}';data=load(OUT/'arrays'/f'measured_{key}.npz')
            db=np.clip(np.searchsorted(protocol['q_edges'],data['q'],'right')-1,0,5)
            source=0 if run==3 else 1;result={'n':len(data['q']),'truth':'Unavailable','methods':{}}
            for j,method in enumerate(METHODS):
                obs=data['scores'][:,j];threshold=thresholds['methods'][method]['threshold']
                r={'observed_low_score_fraction':float(np.mean(obs<=threshold)),
                   'interpretation':'Selected source-run low-score retention, not alpha efficiency, gamma leakage or purity.',
                   'observed_quantiles':np.quantile(obs,[.1,.25,.5,.75,.9]).tolist(),'comparisons':{}}
                for case,d in datasets.items():
                    mask=d['common']&(d['source']==source);values=d['scores'][mask,j]
                    cells=[]
                    for b in range(6):
                        x=obs[db==b];y=d['scores'][mask&(d['bin']==b),j]
                        cell={'bin':b,'n_measured':len(x),'n_MC':len(y)}
                        if min(len(x),len(y))>=10:
                            oq=np.quantile(x,[.25,.5,.75]);mq=np.quantile(y,[.25,.5,.75])
                            cell.update(data_quartiles=oq.tolist(),MC_quartiles=mq.tolist(),
                              IQR_ratio=float((mq[2]-mq[0])/(oq[2]-oq[0])),KS=ks(x,y))
                        cells.append(cell)
                    r['comparisons'][case]={'source_only_no_truth_cut':True,'n_MC':len(values),
                        'MC_quantiles':np.quantile(values,[.1,.25,.5,.75,.9]).tolist(),
                        'KS':ks(obs,values),'per_Q':cells}
                result['methods'][method]=r
            groups[key]=result
    return groups


def export_csv(report):
    rows=[]
    for case,r in report['cases'].items():
        for method,m in r['methods'].items():
            a=m['operating']['alpha_acceptance'];g=m['operating']['gamma_leakage']
            rows.append({'case':case,'method':method,'threshold':m['operating']['threshold'],
                'test_AUC':m['test_AUC'],'within_Q_bin_AUC':m['within_bin_equal_mean_AUC'],
                'n_alpha':a['total'],'alpha_passed':a['passed'],'alpha_acceptance':a['fraction'],
                'alpha_CI95_low':a['exact_two_sided95'][0],'alpha_CI95_high':a['exact_two_sided95'][1],
                'n_gamma':g['total'],'gamma_passed_as_alpha':g['passed'],'gamma_leakage':g['fraction'],
                'gamma_CI95_low':g['exact_two_sided95'][0],'gamma_CI95_high':g['exact_two_sided95'][1],
                'gamma_upper_one_sided95':g['one_sided95_upper'],
                'descriptive_truth_FOM':m['descriptive_all_common_truth_FOM']})
    with (OUT/'operating_points.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    rows=[]
    for case,r in report['cases'].items():
        for method,m in r['methods'].items():
            for cell,fit in zip(m['per_Q'],m['source_mixture_fits']):
                rows.append({'case':case,'method':method,'bin':cell['bin'],
                    'test_alpha':cell['test_alpha'],'test_gamma':cell['test_gamma'],
                    'all_alpha':cell['all_alpha'],'all_gamma':cell['all_gamma'],
                    'truth_FOM':cell.get('truth_FOM'),'test_AUC':cell.get('test_AUC'),
                    'mixture_FOM_valid':fit['valid'],
                    'mixture_FOM':fit['fit']['shape_fom_fwhm'] if fit['valid'] else None})
    with (OUT/'per_Q_PID.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    assert not (OUT/'pid.json').exists(),'Refuse to replace frozen PID results'
    protocol=read(OUT/'protocol.json');(OUT/'curves').mkdir(exist_ok=True)
    datasets={case:combined(case,protocol) for case in CASES}
    for source in SOURCES:
        parent=load(OUT/'arrays'/f'run12_{source}.npz')
        for state in STATES:
            d=load(OUT/'arrays'/f'{state}_{source}.npz')
            for key in ['ids','energy_by_origin','charge_by_origin','truth','split']:
                np.testing.assert_array_equal(d[key],parent[key])
    thresholds=freeze_thresholds(datasets['run12'],protocol)
    counts=bootstrap_counts()
    report={'scope':__doc__,'code_sha256':sha(Path(__file__)),'metrics_code_sha256':sha(Path(__file__).with_name('pid_metrics.py')),
        'protocol_sha256':sha(OUT/'protocol.json'),'thresholds_sha256':sha(OUT/'thresholds.json'),
        'all_response_states_share_truth_and_mother_split':True,'cases':{},'paired_state_differences':{},
        'experimental_truth_available':False,'interval_scope':'Fixed model/PCA/threshold; mother-event bootstrap. AUC bootstrap at 1 may degenerate; does not establish zero rare leakage. Use finite binomial limits.',
        'original_per_run_role':'Diagnostic comparison; run3 Co + run12 Am use different readout states and cannot alone establish particle discrimination under a common detector response.'}
    draws={}
    for case in CASES:report['cases'][case],draws[case]=analyze_case(case,datasets[case],thresholds,counts,protocol)
    for case in CASES:
        if case=='run12':continue
        report['paired_state_differences'][case]={}
        for j,method in enumerate(METHODS):
            values={metric:draws[case][metric][:,j]-draws['run12'][metric][:,j] for metric in ['auc','fom','binned_auc']}
            report['paired_state_differences'][case][method]={metric:ci(x) for metric,x in values.items()}
            np.savez_compressed(OUT/'curves'/f'{case}_minus_run12_{method}.npz',**values)
    report['measured']=experimental_comparison(datasets,thresholds,protocol)
    point=PROJECT/'analysis/paper_baseline/results/publication_20261028/point.json'
    report['experimental_training_mixture_fits']={'path':str(point.relative_to(PROJECT)),'sha256':sha(point),
        'window_ns':3000,'models':read(point)['windows']['3000']['fits'],
        'note':'Frozen source-constrained statistical cores; no experimental particle truth. Uncertainty/assumptions remain those of the original publication analysis.'}
    save(OUT/'pid.json',report);export_csv(report)
    print('PID statistics and tables saved.',flush=True)


if __name__=='__main__':main()
