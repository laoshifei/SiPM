"""Conditional empirical PID distributions and paired native mechanism effects."""
import argparse
import csv
import numpy as np
from prepare_conditional_response import OUT, PARENT, PROJECT, SOURCES, SEGMENTS, VARIANTS, relative
from prepare_pid import read
from extract_pid import load
from publication_stats import save, sha, block_counts
from analyze_co_tail import finite
from analyze_npe_validation import raw_shapes
from reference import TIME_NS

NBOOT=512
MODES=['original','q2','q4','q2r2']


def quantile(x,w):
    """Endpoint-normalized weighted midpoint CDF; equal weights give type 7.

    Integer bootstrap multiplicities are expanded BEFORE weighting, so the
    unweighted branch is exactly ordinary resampled NumPy quartiles.
    """
    order=np.argsort(x);x=x[order];w=w[order]
    pos=np.cumsum(w)-.5*w
    if len(x)<2 or pos[-1]<=pos[0]:return np.full(3,np.nan)
    pos=(pos-pos[0])/(pos[-1]-pos[0])
    return np.interp([.25,.5,.75],pos,x)


def bin_weights(a,design,mode,b,counts=None):
    edges=np.asarray(design['edges']); bins=np.clip(np.searchsorted(edges,a['q'],'right')-1,0,5)
    ix=np.flatnonzero(bins==b)
    if counts is not None:ix=np.repeat(ix,counts[ix])
    if len(ix)<5:return None
    c=design['cuts'][str(b)]; q=a['q'][ix]
    if mode=='original':cell=np.zeros(len(ix),int);target=np.ones(1)
    elif mode=='q2':cell=(q>=c['q_quartiles'][1]).astype(int);target=np.asarray(c['q2_target'])
    elif mode=='q4':cell=np.searchsorted(c['q_quartiles'],q,'right');target=np.asarray(c['q4_target'])
    elif mode=='q2r2':
        cell=2*(q>=c['q_quartiles'][1]).astype(int)+(a['baseline_rms'][ix]>=c['rms_median'])
        target=np.asarray(c['q2r2_target'])
    else:raise ValueError(mode)
    number=np.bincount(cell,minlength=len(target))
    if np.any((target>0)&(number==0)):return None
    weights=np.divide(target,number,out=np.zeros_like(target),where=number>0)[cell]
    assert np.isclose(weights.sum(),1.)
    return ix,weights,cell,target


def summary(a,design,mode,counts=None,detail=False):
    means=[];iqrs=[];covs=[];betweens=[];details=[]
    for b in design['support']:
        out=bin_weights(a,design,mode,b,counts)
        if out is None:
            return (np.full(24,np.nan),{'valid':False,'missing_bin':b}) if detail else np.full(24,np.nan)
        ix,w,c,t=out;x=a['scores'][ix];mu=w@x;r=x-mu
        cov=r.T@(w[:,None]*r);between=np.zeros((3,3))
        for j,prob in enumerate(t):
            if prob<=0:continue
            sub=c==j;m=np.average(x[sub],axis=0,weights=w[sub]);delta=m-mu
            between+=prob*np.outer(delta,delta)
        quarts=np.array([quantile(x[:,j],w) for j in range(3)]).T
        means.append(mu);iqrs.append(quarts[2]-quarts[0]);covs.append(cov);betweens.append(between)
        if detail:
            details.append({'bin':b,'n':len(ix),'ESS':float(1/(w@w)),'cell_counts':np.bincount(c,minlength=len(t)).tolist(),
                'cell_mass':np.bincount(c,weights=w,minlength=len(t)).tolist(),'max_normalized_event_weight':float(w.max()),
                'mean':mu.tolist(),'IQR':(quarts[2]-quarts[0]).tolist(),'covariance_population':cov.tolist(),
                'between_cell_covariance':between.tolist()})
    result=np.r_[np.mean(means,axis=0),np.sqrt(np.mean(np.asarray(iqrs)**2,axis=0)),
                 np.mean(covs,axis=0).ravel(),np.mean(betweens,axis=0).ravel()]
    return (result,{'valid':True,'bins':details,'minimum_ESS':min(r['ESS'] for r in details)}) if detail else result


def interval(a):
    x=np.asarray(a);u=np.isfinite(x)
    return {'CI95':np.quantile(x[u],[.025,.975]).tolist() if u.sum()>=128 else None,'valid_draws':int(u.sum())}


def contrasts(m,d):
    # mean time is stored with a negative sign in the frozen score definition.
    return np.r_[-1000*(m[1]-d[1]),m[3:6]/d[3:6]]


def design_for(p,source):
    return {'edges':p['q_edges'],'support':p['support_bins'][source],'cuts':p['conditional_cuts'][source]}


def conditional():
    assert not (OUT/'conditional.json').exists()
    p=read(OUT/'protocol.json');report={'scope':__doc__,'code_sha256':sha(__file__),'protocol_sha256':sha(OUT/'protocol.json'),'groups':{},
        'estimator':'Mean and population within-original-Q-bin covariance, then equal bin average; RMS of per-bin IQR. Weighted midpoint CDF normalized to endpoints equals NumPy type7 for equal weights. Bootstrap integer multiplicities expanded before matching. Fixed first60 targets; no trimming or clipping.',
        'limits':'Reweighting conditions on observed Q and baseline RMS, not latent energy or particle truth. Sparse state support is not repaired by weight clipping. Intervals conditional on original model, cuts and targets; later data internal checks only.'}
    draws={};points={}
    for source,run in SOURCES:
        design=design_for(p,source);m=load(PARENT/'validation/arrays'/f'run{run}_{source}.npz')
        mb={mode:np.empty((NBOOT,24)) for mode in MODES}
        rng=np.random.default_rng(410030500+run)
        for ib in range(NBOOT):
            counts=np.bincount(rng.integers(int(m['n_parent']),size=int(m['n_parent'])),minlength=int(m['n_parent']))[m['ids']]
            for mode in MODES:mb[mode][ib]=summary(m,design,mode,counts)
        mp={mode:summary(m,design,mode,detail=True) for mode in MODES}
        for mode in MODES:draws[f'run{run}__MC__{mode}']=mb[mode]
        for si,segment in enumerate(SEGMENTS):
            group=f'run{run}_{segment}';d=load(OUT/'arrays'/f'data_{group}.npz');row={'source':source,'modes':{}}
            db={}
            for block in [64,128]:
                rng=np.random.default_rng(410030700+run*100+si*1000+block)
                db[block]={mode:np.empty((NBOOT,24)) for mode in MODES}
                for ib in range(NBOOT):
                    counts=block_counts(d['ids'],tuple(d['span']),block,rng)
                    for mode in MODES:db[block][mode][ib]=summary(d,design,mode,counts)
                for mode in MODES:draws[f'{group}__block{block}__{mode}']=db[block][mode]
            for mode in MODES:
                dp,dd=summary(d,design,mode,detail=True);mv,md=mp[mode];v=contrasts(mv,dp)
                item={'MC':mv.tolist(),'data':dp.tolist(),'MC_support':md,'data_support':dd,
                    'mean_time_difference_ns':float(v[0]),'IQR_ratio':v[1:].tolist(),'uncertainty':{}}
                points[f'{group}__{mode}']=v
                for block in [64,128]:
                    B=np.array([contrasts(x,y) for x,y in zip(mb[mode],db[block][mode])])
                    B0=np.array([contrasts(x,y) for x,y in zip(mb['original'],db[block]['original'])])
                    item['uncertainty'][str(block)]={'mean_time':interval(B[:,0]),'IQR_ratio':[interval(B[:,j]) for j in range(1,4)],
                        'paired_change_from_original':[interval((B-B0)[:,j]) for j in range(4)]}
                row['modes'][mode]=finite(item)
            report['groups'][group]=row
            print(group,{mode:np.round(points[f'{group}__{mode}'],3).tolist() for mode in MODES},flush=True)
    # Exact closure of the unweighted estimator against the frozen publication-stage diagnostic.
    old=read(PARENT/'assessment.json')
    for group,row in report['groups'].items():
        ref=old['checks'][group]['models']['joint'];a=row['modes']['original']
        np.testing.assert_allclose(a['IQR_ratio'],ref['IQR_ratio'],rtol=1e-11,atol=1e-11)
        np.testing.assert_allclose(a['mean_time_difference_ns'],ref['tail_difference'][0],atol=1e-8)
    report['original_estimator_closure']=True
    np.savez_compressed(OUT/'conditional_bootstrap.npz',**draws)
    save(OUT/'conditional.json',finite(report))


def average_value(a,values,design,mode='q2'):
    result=[]
    for b in design['support']:
        out=bin_weights(a,design,mode,b)
        if out is None:return np.full(values.shape[1:],np.nan)
        ix,w,_,_=out;result.append(np.tensordot(w,values[ix],axes=1))
    return np.mean(result,axis=0)


def cohort(a,scheme):
    use=a['common'] if scheme=='reselected' else a['fixed']
    out={k:a[k][use] for k in ['ids','q','scores','baseline_rms']}
    if scheme!='reselected':out['q']=a['original_q'][use]
    if scheme=='fixed_anchor':
        np.testing.assert_array_equal(out['ids'],a['fixed_ids']);out['scores']=a['fixed_anchor_scores']
    return out


def mechanisms():
    assert not (OUT/'mechanisms.json').exists()
    p=read(OUT/'protocol.json');execution=read(OUT/'execution.json')
    assert len(execution['jobs'])==12
    cb=load(OUT/'conditional_bootstrap.npz');conditional_report=read(OUT/'conditional.json')
    report={'scope':__doc__,'code_sha256':sha(__file__),'protocol_sha256':sha(OUT/'protocol.json'),'sources':{},
        'uncertainty':'512 paired whole-original-mother draws across all response variants, original-index data block64/128. Q2 target from first60 fixed. Pointwise intervals; no model/fit uncertainty, no mechanism selected on later data.',
        'comparators':'Physical mechanism changes are paired against no_signal_residual. Identity retains empirical signal noise. Contrasts are not tuned candidates; no gain compensation and no additional experimental truth labels.'}
    draws={};table=[]
    for source,run in SOURCES:
        design=design_for(p,source);all_arrays={v:load(OUT/'arrays'/f'run{run}_{source}__{v}.npz') for v in VARIANTS}
        cohorts={(v,s):cohort(a,s) for v,a in all_arrays.items() for s in ['reselected','fixed','fixed_anchor']}
        point={k:summary(a,design,'q2',detail=True) for k,a in cohorts.items()}
        boots={k:np.empty((NBOOT,24)) for k in cohorts};rng=np.random.default_rng(410030500+run)
        n=int(all_arrays['identity']['n_parent'])
        for ib in range(NBOOT):
            counts=np.bincount(rng.integers(n,size=n),minlength=n)
            for key,a in cohorts.items():boots[key][ib]=summary(a,design,'q2',counts[a['ids']])
        source_row={'variants':{},'support_bins':design['support']}
        for variant,a in all_arrays.items():
            er=read(OUT/'extraction'/f'run{run}_{source}__{variant}.json')
            row={'extraction':er,'schemes':{}}
            for scheme in ['reselected','fixed','fixed_anchor']:
                key=(variant,scheme);mp,md=point[key];B=boots[key];base=boots[('no_signal_residual',scheme)]
                bp=point[('no_signal_residual',scheme)][0]
                delta=np.r_[-1000*(mp[1]-bp[1]),mp[3:6]/bp[3:6]]
                effect=np.c_[-1000*(B[:,1]-base[:,1]),B[:,3:6]/base[:,3:6]]
                item={'point':mp.tolist(),'support':md,'change_from_no_residual':delta.tolist(),
                    'paired_effect_CI95':[interval(effect[:,j]) for j in range(4)],'data_checks':{}}
                for segment in SEGMENTS:
                    group=f'run{run}_{segment}';dpoint=np.asarray(conditional_report['groups'][group]['modes']['q2']['data'],float)
                    value=contrasts(mp,dpoint);uncertainty={}
                    for block in [64,128]:
                        db=cb[f'{group}__block{block}__q2'];diff=np.array([contrasts(x,y) for x,y in zip(B,db)])
                        uncertainty[str(block)]=[interval(diff[:,j]) for j in range(4)]
                    item['data_checks'][segment]={'mean_time_difference_ns':float(value[0]),'IQR_ratio':value[1:].tolist(),'uncertainty':uncertainty}
                if scheme=='fixed':
                    use=a['fixed'];x=cohorts[key]
                    raw=raw_shapes(a['waves'][use]);avg=average_value(x,raw,design)
                    avg0=average_value(cohorts[('no_signal_residual','fixed')],raw_shapes(all_arrays['no_signal_residual']['waves'][all_arrays['no_signal_residual']['fixed']]),design)
                    region=(TIME_NS>=-40)&(TIME_NS<3000)
                    item['mean_ADC_change_NRMSE_vs_no_residual']=float(np.sqrt(np.mean((avg[region]-avg0[region])**2))/avg0.max())
                    item['F']=average_value(x,a['metrics'][use],design).tolist()
                    item['observed_Q_ratio_to_original']=float(average_value(x,a['q'][use]/a['original_q'][use],design))
                    cov=mp[6:15].reshape(3,3);cov0=bp[6:15].reshape(3,3)
                    dc=np.asarray(conditional_report['groups'][f'run{run}_first60']['modes']['q2']['data'])[6:15].reshape(3,3)
                    item['covariance_increment_normalized_data_SD_products']=((cov-cov0)/np.sqrt(np.outer(np.diag(dc),np.diag(dc)))).tolist()
                    table.append({'source':source,'variant':variant,'n_common':er['n_common'],'n_fixed':er['n_fixed'],
                        'mean_time_shift_from_no_residual_ns':delta[0],
                        'QDC_width_ratio_to_no_residual':delta[1],'mean_time_width_ratio_to_no_residual':delta[2],'PCA_width_ratio_to_no_residual':delta[3],
                        'mean_time_width_effect_CI95_low':item['paired_effect_CI95'][2]['CI95'][0],
                        'mean_time_width_effect_CI95_high':item['paired_effect_CI95'][2]['CI95'][1],
                        'afterpulse_fraction':er['fixed_afterpulse_avalanche_fraction'],'mean_recovery':er['fixed_mean_recovery'],
                        'ADC_NRMSE_change_percent':100*item['mean_ADC_change_NRMSE_vs_no_residual'],
                        'observed_Q_ratio_to_original':item['observed_Q_ratio_to_original'],
                        'first60_mean_time_difference_ns':item['data_checks']['first60']['mean_time_difference_ns'],
                        'last20_mean_time_difference_ns':item['data_checks']['last20']['mean_time_difference_ns']})
                row['schemes'][scheme]=finite(item);draws[f'run{run}__{variant}__{scheme}']=B
            source_row['variants'][variant]=row
            print(source,variant,'fixed delta/time,width',np.round(row['schemes']['fixed']['change_from_no_residual'],4),flush=True)
        report['sources'][source]=source_row
    np.savez_compressed(OUT/'mechanism_bootstrap.npz',**draws)
    with (OUT/'mechanism_table.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=table[0]);w.writeheader();w.writerows(table)
    save(OUT/'mechanisms.json',finite(report))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['conditional','mechanisms']);args=ap.parse_args()
    conditional() if args.stage=='conditional' else mechanisms()
