"""Export traceable supplementary CSV tables without changing frozen results."""
import csv
import json
from pathlib import Path
import zipfile
import numpy as np
import publication_stats as p
from robustness_full_cohort import OUT,SCOPES,VIEWS,coarsen_q,from_moments


def run(out=OUT):
    folder=out/'tables';folder.mkdir(exist_ok=True);sources={};tables=[]
    def read(name,new=False):
        path=(out if new else p.OUT)/name;sources[str(path.relative_to(p.PROJECT))]=p.sha(path)
        return json.loads(path.read_text())
    def write(name,rows,description,units,uncertainty):
        if not rows:raise ValueError('Empty requested table')
        fields=list(dict.fromkeys(k for row in rows for k in row))
        with (folder/name).open('w',encoding='utf-8-sig',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
        tables.append({'file':name,'rows':len(rows),'columns':fields,'description':description,
            'units':units,'uncertainty':uncertainty,'sha256':p.sha(folder/name)})
    protocol=read('protocol.json');point=read('point.json');stats=read('paired_uncertainty.json')
    write('S01_cohort_flow.csv',protocol['flow'],'Sequential event flow in every original acquisition-index segment','counts', 'No efficiency or live-time denominator inferred')
    rows=[]
    for block,r in stats['block_sizes'].items():
        for window,a in r['windows'].items():
            for i,method in enumerate(p.METHODS):
                interval=a['median_fom'][method]
                rows.append({'block_events':int(block),'window_end_ns':int(window),'method':method,
                  'median_FOM':a['point_median_fom'][i],'interval95_low':interval['interval95'][0],'interval95_high':interval['interval95'][1],
                  'valid_draws':interval['n'],'total_draws':interval['total']})
    write('S02_window_method_FOM.csv',rows,'Median of six fixed-Q-bin FWHM FOM values','dimensionless','Paired block percentile; PCA and mixture nuisance refit; unadjusted exploratory intervals')
    rows=[]
    for block,r in stats['block_sizes'].items():
        for window,a in r['windows'].items():
            for pair,ci in a['paired_differences'].items():
                rows.append({'block_events':int(block),'window_end_ns':int(window),'comparison':pair,'summary':'difference_of_medians','Q_bin_zero_based':'',
                  'point':ci['point'],'interval95_low':ci['interval95'][0],'interval95_high':ci['interval95'][1],'valid_draws':ci['n'],'total_draws':ci['total']})
                left,right=pair.split(' minus ');li=p.METHODS.index(left);ri=p.METHODS.index(right)
                f=np.array(point['windows'][window]['fom_by_method_bin'])
                for b,ci in enumerate(a['per_bin_differences'][pair]):
                    rows.append({'block_events':int(block),'window_end_ns':int(window),'comparison':pair,'summary':'within_Q_bin','Q_bin_zero_based':b,
                      'point':f[li,b]-f[ri,b],'interval95_low':ci['interval95'][0],'interval95_high':ci['interval95'][1],'valid_draws':ci['n'],'total_draws':ci['total']})
    write('S03_paired_method_differences.csv',rows,'Paired differences; median and per-bin summaries are distinct','dimensionless','Conditional percentile intervals; no multiplicity correction')
    model=read('core_model_sensitivity.json');rows=[]
    for name,a in {'primary':{'events':3997,'fom_by_method_bin':point['windows']['3000']['fom_by_method_bin']},**point['sensitivity']}.items():
        medians=np.median(a['fom_by_method_bin'],axis=1)
        for i,m in enumerate(p.METHODS):rows.append({'selection':name,'core_family':'Gaussian','fixed_dof':'','method':m,'training_n':a['events'],'median_FWHM_FOM':medians[i]})
    for dof,a in model['core_families'].items():
        for i,m in enumerate(p.METHODS):rows.append({'selection':'primary','core_family':'Student_t','fixed_dof':int(dof),'method':m,'training_n':3997,'median_FWHM_FOM':a['medians'][i]})
    write('S04_selection_and_core_models.csv',rows,'3us point sensitivity; Student-t checks added after Gaussian baseline','dimensionless','Point estimates only; t scale is not a standard deviation')
    transfer=read('transfer_uncertainty.json');matched=read('transfer_matched.json');rows=[]
    metrics=['empirical_KS','median_shift_over_reference_IQR','whole_score_IQR_ratio']
    for key,a in transfer['groups'].items():
        parent=matched['groups'][key]
        for mi,metric in enumerate(metrics):
            for i,method in enumerate(p.METHODS):rows.append({'check_group':key,'reference':parent['reference'],'supported_Q_bins_zero_based':';'.join(map(str,parent['supported_bins'])),
              'method':method,'metric':metric,'point':a['point'][mi][i],'interval95_low':a['interval95'][0][mi][i],'interval95_high':a['interval95'][1][mi][i],'draws':256})
    write('S05_cross_run_effects.csv',rows,'Equal supported-Q-bin transfer effects with training uncertainty','dimensionless','Conditional block intervals; KS is not a p-value; IQR is not a particle-peak width')
    sens=read('transfer_selection_sensitivity.json');rows=[]
    for key,variants in sens['groups'].items():
        for variant,a in variants.items():
            for mi,metric in enumerate(metrics):
                for i,method in enumerate(p.METHODS):rows.append({'check_group':key,'variant':variant,'supported_Q_bins_zero_based':';'.join(map(str,a['supported_bins'])),
                  'check_n':a['events'][0],'reference_n':a['events'][1],'method':method,'metric':metric,
                  'primary_same_support':a['primary_same_support'][mi][i],'alternative_same_support':a['alternative_same_support'][mi][i]})
    write('S06_transfer_selection.csv',rows,'Primary and alternative evaluated on identical supported Q cells','dimensionless','Point sensitivity; original primary PCA fixed')
    cov=read('covariance_uncertainty.json');rows=[]
    for block,a in cov['block_sizes'].items():
        ci=np.array(a['equal_q_interval95'])
        for r,run in enumerate((12,18)):
            for vi,view in enumerate(('all','ratio_030_035')):
                for f,width in enumerate((50,102,202)):
                    for w,start in enumerate((0,500,1200,2200)):
                        rows.append({'run':run,'view':view,'block_events':int(block),'filter_ns':width,'window_start_ns':start,'window_end_ns':start+400,
                          'equal_Q_point_ADC':cov['equal_q_point'][r][vi][f][w],'interval95_low':ci[0,r,vi,f,w],'interval95_high':ci[1,r,vi,f,w],'draws':a['draws']})
    write('S07_primary_local_covariance.csv',rows,'Frozen-cohort zero-lag excess covariance, six Q bins equally weighted','ADC','Each cell mean recomputed per block draw; not a gain measurement')
    full=read('full_cohort_covariance.json',True);rows=[]
    for nq,scopes in full['points'].items():
        for scope,views in scopes.items():
            for view,a in views.items():
                ci=['','']
                if nq=='6' and view in VIEWS[:2]:
                    si=('full','frozen3000').index(scope);vi=VIEWS.index(view)
                    ci=[full['block_sizes']['64']['relative_interval95'][k][si][vi] for k in (0,1)]
                rows.append({'scope':scope,'shape_view':view,'Q_bins':int(nq),'time_quarter_zero_based':'',
                  'supported_Q_bins_zero_based':';'.join(map(str,a['supported_bins'])),'run12_ADC':a['equal_q_values'][0],'run18_ADC':a['equal_q_values'][1],
                  'run12_over18_minus1':a['run12_over18_minus1'],'interval95_low':ci[0],'interval95_high':ci[1]})
    for view,a in full['time_quarters'].items():
        for v in a['points']:rows.append({'scope':'full_time_quarter','shape_view':view,'Q_bins':6,'time_quarter_zero_based':v['quarter'],
          'supported_Q_bins_zero_based':';'.join(map(str,a['shared_supported_bins'])),'run12_ADC':v['equal_q_values'][0],'run18_ADC':v['equal_q_values'][1],
          'run12_over18_minus1':v['relative'],'interval95_low':'','interval95_high':''})
    write('S08_full_cohort_covariance.csv',rows,'50ns filter,500--900ns local endpoint only','ADC for means; fraction for relative contrast','Only six-bin full/frozen first-two-view rows have512 block64 intervals; all other rows point-only')
    oldmc=read('mc_uncertainty.json');rows=[]
    for source,r in oldmc['results'].items():
        for candidate in ('control','native_best','prompt_plus_ap15'):
            a=r[candidate];n=oldmc['sources'][source]['candidates'][candidate]
            for k,metric in enumerate(oldmc['metric_order']):rows.append({'source':source,'candidate':candidate,'metric':metric,'measured_n':oldmc['sources'][source]['measured_n'],
              'parent_decays':n['nparent'],'common_Q_MC_n':n['common_n'],'point':a['point'][k],'joint_interval95_low':a['joint_interval95'][0][k],'joint_interval95_high':a['joint_interval95'][1][k],
              'MC_only_interval95_low':a['mc_only_interval95'][0][k],'MC_only_interval95_high':a['mc_only_interval95'][1][k]})
    write('S09_MC_original_metrics.csv',rows,'Original fixed-range MC comparison, before Q-population reweighting','F:ADC; other metrics:dimensionless','512 shared-parent draws; broad-selected Q reference held fixed; geometry/circuit uncertainty excluded')
    mc=read('mc_charge_matched.json',True);rows=[];support_rows=[]
    for source,r in mc['sources'].items():
        for dataset,counts in r['counts_by_bin'].items():
            for b,n in enumerate(counts):support_rows.append({'source':source,'dataset':dataset,'Q_bin_zero_based':b,'Q_low_ADC_samples':mc['q_edges'][b],
              'Q_high_ADC_samples':mc['q_edges'][b+1],'n':n,'supported_all_candidates_min10':b in r['settings']['10']['supported_bins'],'supported_all_candidates_min5':b in r['settings']['5']['supported_bins']})
        for minimum,s in r['settings'].items():
            for target,a in s['targets'].items():
                for candidate,v in a['candidates'].items():
                    ci=mc['bootstrap'][source][minimum][target][candidate]
                    for k,metric in enumerate(mc['metric_order']):rows.append({'source':source,'min_MC_n_per_bin':int(minimum),'weight_target':target,'candidate':candidate,
                      'supported_Q_bins_zero_based':';'.join(map(str,s['supported_bins'])),'bin_weights':';'.join(map(str,a['bin_weights'])),
                      'measured_n':s['selected_n']['measured'],'MC_n':s['selected_n'][candidate],'MC_effective_weighted_n':v['effective_weighted_n'],
                      'metric':metric,'point':v['point'][k],'interval95_low':ci['interval95'][0][k],'interval95_high':ci['interval95'][1][k],
                      'valid_draws':ci['valid'],'total_draws':ci['draws'],'weighted_Q_KS_point':v['weighted_Q_KS']})
    write('S10_MC_Q_matched.csv',rows,'Fixed supported cells and shared Q-bin weights for data and all candidates','relative residuals and NRMSE:fractions; KS:dimensionless','512 paired-parent and recorded-block draws; empty cells counted as invalid; source-specific support')
    write('S11_MC_Q_support.csv',support_rows,'Counts and missing Q coverage before choosing supported cells','counts; Q:ADC samples','No extrapolation into missing cells; bin numbers are zero-based')
    rows=[]
    for run in (12,18):
        path=out/f'full_run{run}.npz';sources[str(path.relative_to(p.PROJECT))]=p.sha(path);d=p.load(path)
        for bins in (6,12,24):
            n,s,ss=coarsen_q(d['n'][:2],d['total'][:2],d['square'][:2],bins)
            values=from_moments(np.maximum(n,2),s,ss)
            edges=np.linspace(protocol['q_edges'][0],protocol['q_edges'][-1],bins+1)
            for si,scope in enumerate(SCOPES[:2]):
                for vi,view in enumerate(VIEWS):
                    for b in range(bins):rows.append({'run':run,'scope':scope,'shape_view':view,'number_Q_bins':bins,'Q_bin_zero_based':b,
                      'Q_low_ADC_samples':edges[b],'Q_high_ADC_samples':edges[b+1],'n':int(n[si,vi,b]),'eligible_n_ge10':bool(n[si,vi,b]>=10),
                      'excess_covariance_ADC':float(values[si,vi,b]) if n[si,vi,b]>=10 else ''})
    write('S12_full_cohort_Q_cells.csv',rows,'Per-Q-cell sufficient-statistic covariance for the declared local endpoint','ADC','Point estimates only; cells below10 events not estimated')
    manifest={'code_sha256':p.sha(Path(__file__)),'tables':tables,'inputs_sha256':sources,
        'conventions':'UTF-8 BOM CSV; empty interval fields mean not estimated, never zero. Relative effects are fractions, not percent. Q-bin/time-quarter numbers are zero-based. All methods and gates preserve source-vs-particle distinction.'}
    p.save(folder/'manifest.json',manifest)
    with zipfile.ZipFile(out/'supplementary_tables.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(folder.iterdir()):
            if path.suffix in ('.csv','.json'):archive.write(path,path.name)
    print('Exported',len(tables),'tables;',sum(t['rows'] for t in tables),'data rows',flush=True)


if __name__=='__main__':run()
