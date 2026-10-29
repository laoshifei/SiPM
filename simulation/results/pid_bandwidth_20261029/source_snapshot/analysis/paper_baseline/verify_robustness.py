"""Verify and archive the supplementary phase without rewriting its parent."""
import csv
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from urllib.parse import unquote
import zipfile
import numpy as np
import ROOT
import publication_stats as p
import robustness_full_cohort as full
import robustness_mc_charge as mc


def read(path):
    return json.loads(path.read_text())


def check_hashes(mapping, base=p.PROJECT):
    for name, digest in mapping.items():
        assert p.sha(base/name) == digest, name
    return len(mapping)


def verify_full(out, report):
    assert report['code_sha256'] == p.sha(p.HERE/'robustness_full_cohort.py')
    check_hashes(report['input_sha256'])
    assert report['parent_protocol_sha256'] == p.sha(p.OUT/'protocol.json')
    protocol = read(p.OUT/'protocol.json')
    old = read(p.OUT/'covariance_uncertainty.json')
    arrays = {}
    for run in (12, 18):
        d = p.load(p.OUT/f'pool_run{run}.npz')
        keep = ((d['split'] == 0) & d['eligible_primary'] & (d['amplitude'] >= 200)
                & ((d['flags'] & 0x8000) == 0) & (d['t10_sample'] >= 384))
        a = p.load(out/f'full_run{run}.npz')
        arrays[run] = a
        np.testing.assert_array_equal(a['ids'], d['ids'][keep])
        np.testing.assert_array_equal(a['frozen_mask'], d['primary'][keep])
        np.testing.assert_array_equal(a['ids'][a['frozen_mask']],
                                      d['ids'][d['primary'] & (d['split'] == 0)])
        np.testing.assert_array_equal(a['q'], d['long_charge'][keep, 2])
        np.testing.assert_array_equal(a['ratio'], d['short_long_ratio'][keep])
        assert len(np.unique(a['ids'])) == len(a['ids'])
        info = report['runs'][str(run)]
        assert len(a['ids']) == info['full_n'] == {12:1547, 18:65034}[run]
        assert int(a['frozen_mask'].sum()) == info['frozen_n'] == {12:1547, 18:3000}[run]
        assert int(a['original_span']) == info['first60_span'] == int(.6*int(d['original_count']))
        rare = a['ids'][a['ratio'] >= .4]
        np.testing.assert_array_equal(rare, info['ratio_ge040_ids'])
        assert len(rare) == info['ratio_ge040_n'] == {12:0, 18:2}[run]
        if run == 18:
            np.testing.assert_array_equal(rare, [45356, 79618])
        q24 = np.minimum(np.searchsorted(np.linspace(protocol['q_edges'][0],
                          protocol['q_edges'][-1], 25), a['q'], side='right')-1, 23)
        quarter = np.minimum(a['ids']*4//int(a['original_span']), 3)
        scopes = [np.ones(len(a['ids']), bool), a['frozen_mask']]
        scopes += [quarter == i for i in range(4)]
        views = [a['ratio'] < .4, (a['ratio'] >= .30) & (a['ratio'] < .35),
                 np.ones(len(a['ids']), bool)]
        for si, scope in enumerate(scopes):
            for vi, view in enumerate(views):
                counts = np.bincount(q24[scope & view], minlength=24)
                np.testing.assert_array_equal(counts, a['n'][si, vi])
        assert a['n'][0, 0].sum() == info['primary_coarse_shape_n']
        assert a['n'][0, 1].sum() == info['narrow_shape_n']
        for name in ('n', 'total', 'square', 'block_n', 'block_total', 'block_square'):
            assert np.isfinite(a[name]).all(), name
        n, s, ss = full.coarsen_q(a['n'][:2, :2], a['total'][:2, :2], a['square'][:2, :2], 6)
        np.testing.assert_array_equal(a['block_n'].sum(axis=2), n)
        np.testing.assert_allclose(a['block_total'].sum(axis=2), s, rtol=1e-10, atol=2e-10)
        np.testing.assert_allclose(a['block_square'].sum(axis=2), ss, rtol=1e-12, atol=1e-10)
        # Frozen six-bin values also reproduce the historical raw-waveform endpoint.
        vals = full.from_moments(n[1], s[1], ss[1]).mean(axis=-1)
        np.testing.assert_allclose(vals, np.array(old['equal_q_point'])[(12,18).index(run), :, 0, 1],
                                   rtol=2e-12, atol=2e-13)
    for bins in (6, 12, 24):
        ns, values = [], []
        for run in (12, 18):
            a = arrays[run]
            n, s, ss = full.coarsen_q(a['n'], a['total'], a['square'], bins)
            ns.append(n)
            values.append(full.from_moments(np.maximum(n, 2), s, ss))
        ns, values = np.array(ns), np.array(values)
        valid = np.all(ns >= 10, axis=0)
        for si, scope in enumerate(full.SCOPES[:2]):
            for vi, view in enumerate(full.VIEWS):
                row = report['points'][str(bins)][scope][view]
                support = np.flatnonzero(valid[si, vi])
                np.testing.assert_array_equal(support, row['supported_bins'])
                np.testing.assert_array_equal(ns[:, si, vi], row['counts'])
                v = values[:, si, vi, support].mean(axis=1)
                np.testing.assert_array_equal(v, row['equal_q_values'])
                assert v[0]/v[1]-1 == row['run12_over18_minus1']
        if bins == 6:
            for vi, view in enumerate(full.VIEWS[:2]):
                row = report['time_quarters'][view]
                support = np.flatnonzero(np.all(valid[2:, vi], axis=0))
                np.testing.assert_array_equal(support, row['shared_supported_bins'])
                for qi, item in enumerate(row['points']):
                    v = values[:, qi+2, vi, support].mean(axis=1)
                    np.testing.assert_array_equal(v, item['equal_q_values'])
                    np.testing.assert_array_equal(ns[:, qi+2, vi], item['counts'])
                    assert item['quarter'] == qi and item['relative'] == v[0]/v[1]-1
    for size, draws in ((64, 512), (128, 256)):
        d = p.load(out/f'full_cohort_bootstrap_b{size}.npz')
        assert d['equal_q'].shape == (draws, 2, 2, 2)
        assert np.isfinite(d['equal_q']).all()
        relative = d['equal_q'][:, 0]/d['equal_q'][:, 1]-1
        np.testing.assert_array_equal(relative, d['relative'])
        row = report['block_sizes'][str(size)]
        assert row['draws'] == draws
        for key, values in [('equal_q_interval95', d['equal_q']),
                            ('relative_interval95', relative),
                            ('paired_frozen_minus_full_relative_interval95', relative[:,1]-relative[:,0])]:
            np.testing.assert_array_equal(np.quantile(values, [.025,.975], axis=0), row[key])
        # Audit paired block resampling from saved sufficient statistics, not just the quantiles.
        for ri, run in enumerate((12,18)):
            a = full.combine_blocks(arrays[run], size//64)
            nb = a['block_n'].shape[2]
            rng = np.random.default_rng(np.random.SeedSequence([281028201, size, run]))
            for draw in range(3):
                weights = np.bincount(rng.integers(nb, size=nb), minlength=nb).astype(float)
                n = np.einsum('b,svbq->svq', weights, a['block_n'])
                total = np.einsum('b,svbqwt->svqwt', weights, a['block_total'])
                square = np.einsum('b,svbqw->svqw', weights, a['block_square'])
                v = full.from_moments(n, total, square).mean(axis=-1)
                np.testing.assert_allclose(v, d['equal_q'][draw,ri], rtol=2e-11, atol=2e-12)
    return {'full_event_counts':report['runs'], 'point_Q_bins':[6,12,24],
            'paired_block_draws':{'64':512,'128':256},
            'raw_reselection_and_frozen_ID_subset':True,
            'moment_block_and_frozen_endpoint_identities':True,
            'all_interval_quantiles_recomputed':True,
            'sampled_block_draws_reconstructed':12}


def direct_matched(data, counts, support, target):
    """Independent event-weight expression used to audit cell-sum estimates."""
    n = np.bincount(data['bins'], weights=counts, minlength=6)
    if np.any(n[support] == 0):
        return None
    mapping = np.zeros(6)
    mapping[support] = target/n[support]
    w = counts*mapping[data['bins']]
    np.testing.assert_allclose(w.sum(), 1, atol=1e-13)
    np.testing.assert_allclose(np.bincount(data['bins'], weights=w, minlength=6)[support], target)
    return w@data['fluctuations'], w@data['shapes'], w


def verify_mc(out, report):
    assert report['code_sha256'] == p.sha(p.HERE/'robustness_mc_charge.py')
    check_hashes(report['inputs_sha256'])
    checked = 0
    for source, run in (('Co60',3), ('Am241',12)):
        data = {}
        for name in ('measured',)+mc.CANDIDATES:
            a = p.load(p.OUT/'mc'/f'{source}_{name}.npz')
            a['bins'] = mc.bin_ids(a['q'], np.array(report['q_edges']))
            data[name] = a
            np.testing.assert_array_equal(np.bincount(a['bins'], minlength=6),
                                          report['sources'][source]['counts_by_bin'][name])
        boot = p.load(out/f'{source}_mc_charge_bootstrap.npz')
        nparent = int(data['control']['nparent'])
        assert nparent == {'Co60':2000,'Am241':12000}[source]
        for name in mc.CANDIDATES:
            ids = data[name]['ids']
            assert int(data[name]['nparent']) == nparent
            assert len(np.unique(ids)) == len(ids) and np.all((ids >= 0) & (ids < nparent))
        for minimum in (10,5):
            counts = report['sources'][source]['counts_by_bin']
            eligible = np.array(counts['measured']) >= 20
            for name in mc.CANDIDATES:
                eligible &= np.array(counts[name]) >= minimum
            support = np.flatnonzero(eligible)
            setting = report['sources'][source]['settings'][str(minimum)]
            np.testing.assert_array_equal(support, setting['supported_bins'])
            for name, a in data.items():
                selected = int(np.isin(a['bins'],support).sum())
                assert selected == setting['selected_n'][name]
                assert selected/len(a['ids']) == setting['retained_fraction'][name]
            for target_name in mc.TARGETS:
                row = setting['targets'][target_name]
                target = np.array(row['bin_weights'])
                expected = (np.ones(len(support))/len(support) if target_name == 'equal_bins'
                            else np.array(counts['measured'])[support]/setting['selected_n']['measured'])
                np.testing.assert_array_equal(target, expected)
                moments = {name:direct_matched(a,np.ones(len(a['ids'])),support,target)
                           for name,a in data.items()}
                np.testing.assert_allclose(moments['measured'][0], row['measured_F'], rtol=1e-13)
                values = boot[f'min{minimum}_{target_name}']
                assert values.shape == (512,3,4) and not np.isinf(values).any()
                for i,name in enumerate(mc.CANDIDATES):
                    item = row['candidates'][name]
                    v = mc.discrepancy(moments[name][:2],moments['measured'][:2])
                    np.testing.assert_allclose(v,item['point'],rtol=2e-12,atol=2e-14)
                    np.testing.assert_allclose(moments[name][0],item['MC_F'],rtol=1e-13)
                    w = moments[name][2]
                    np.testing.assert_allclose(1/(w@w),item['effective_weighted_n'],rtol=1e-13)
                    ks = mc.weighted_ks(data[name]['q'],w,data['measured']['q'],moments['measured'][2])
                    assert ks == item['weighted_Q_KS']
                    np.testing.assert_allclose(w@data[name]['q']-moments['measured'][2]@data['measured']['q'],
                                               item['weighted_mean_Q_difference_adc_samples'],atol=1e-8)
                    interval = mc.interval(values[:,i])
                    assert interval == report['bootstrap'][source][str(minimum)][target_name][name]
                    assert interval['valid'] == (510 if source == 'Am241' and minimum == 5 else 512)
                    checked += 1
                assert mc.interval(values[:,1]-values[:,0]) == report['bootstrap'][source][str(minimum)][target_name]['paired_best_minus_control']
                # Reconstruct several actual draws, including any empty-cell draw.
                invalid = np.flatnonzero(~np.isfinite(values).all(axis=(1,2)))
                audit_draws = sorted(set([0,1,511]+invalid.tolist()))
                for draw in audit_draws:
                    rng = np.random.default_rng(np.random.SeedSequence([281028202,run,draw]))
                    parents = np.bincount(rng.integers(nparent,size=nparent),minlength=nparent)
                    counts_draw = {'measured':p.block_counts(data['measured']['ids'],data['measured']['span'],64,rng)}
                    counts_draw.update({name:parents[data[name]['ids']] for name in mc.CANDIDATES})
                    reference = direct_matched(data['measured'],counts_draw['measured'],support,target)
                    reconstructed = np.full((3,4),np.nan)
                    if reference is not None:
                        for i,name in enumerate(mc.CANDIDATES):
                            model = direct_matched(data[name],counts_draw[name],support,target)
                            if model is None:
                                break
                            reconstructed[i] = mc.discrepancy(model[:2],reference[:2])
                    np.testing.assert_allclose(reconstructed,values[draw],rtol=2e-11,atol=2e-13,equal_nan=True)
    return {'candidate_weight_settings_checked':checked,'draws_per_source':512,
            'conditional_parent_counts':{'Co60':2000,'Am241':12000},
            'all_point_weights_and_bootstrap_quantiles_recomputed':True,
            'sampled_draws_and_all_empty_draws_reconstructed':True,
            'new_independent_Geant4_transport':0}


def verify_tables(out, full_report, mc_report):
    manifest = read(out/'tables/manifest.json')
    assert manifest['code_sha256'] == p.sha(p.HERE/'export_publication_tables.py')
    check_hashes(manifest['inputs_sha256'])
    tables = {}
    for item in manifest['tables']:
        path = out/'tables'/item['file']
        assert p.sha(path) == item['sha256']
        with path.open(encoding='utf-8-sig',newline='') as stream:
            reader = csv.DictReader(stream)
            assert reader.fieldnames == item['columns']
            rows = list(reader)
        assert len(rows) == item['rows'] and all(None not in row for row in rows)
        for row in rows:
            for value in row.values():
                assert value.lower() not in ('nan','inf','-inf','none','null')
        assert item['units'] and item['uncertainty'] and item['description']
        tables[item['file']] = rows
    assert len(tables) == 12 and sum(map(len,tables.values())) == 1238
    with zipfile.ZipFile(out/'supplementary_tables.zip') as archive:
        assert archive.testzip() is None
        assert set(archive.namelist()) == set(tables)|{'manifest.json'}
        for name in archive.namelist():
            assert archive.read(name) == (out/'tables'/name).read_bytes()
    for row in tables['S08_full_cohort_covariance.csv']:
        if row['scope'] == 'full_time_quarter':
            src = full_report['time_quarters'][row['shape_view']]['points'][int(row['time_quarter_zero_based'])]
            expected = src['relative']
        else:
            src = full_report['points'][row['Q_bins']][row['scope']][row['shape_view']]
            expected = src['run12_over18_minus1']
        assert float(row['run12_over18_minus1']) == expected
        np.testing.assert_array_equal([float(row['run12_ADC']),float(row['run18_ADC'])],src['equal_q_values'])
        has_interval = row['scope'] in full.SCOPES[:2] and row['Q_bins'] == '6' and row['shape_view'] in full.VIEWS[:2]
        assert bool(row['interval95_low']) == bool(row['interval95_high']) == has_interval
        if has_interval:
            si = full.SCOPES.index(row['scope']); vi = full.VIEWS.index(row['shape_view'])
            ci = np.array(full_report['block_sizes']['64']['relative_interval95'])[:,si,vi]
            np.testing.assert_array_equal(ci,[float(row['interval95_low']),float(row['interval95_high'])])
    for row in tables['S10_MC_Q_matched.csv']:
        setting = mc_report['sources'][row['source']]['settings'][row['min_MC_n_per_bin']]
        value = setting['targets'][row['weight_target']]['candidates'][row['candidate']]
        ci = mc_report['bootstrap'][row['source']][row['min_MC_n_per_bin']][row['weight_target']][row['candidate']]
        k = mc_report['metric_order'].index(row['metric'])
        assert float(row['point']) == value['point'][k]
        assert int(row['measured_n']) == setting['selected_n']['measured']
        assert int(row['MC_n']) == setting['selected_n'][row['candidate']]
        assert int(row['valid_draws']) == ci['valid'] and int(row['total_draws']) == ci['draws']
        np.testing.assert_array_equal([float(row['interval95_low']),float(row['interval95_high'])],np.array(ci['interval95'])[:,k])
    paired = read(p.OUT/'paired_uncertainty.json')
    for row in tables['S03_paired_method_differences.csv']:
        if row['summary'] == 'difference_of_medians':
            src = paired['block_sizes'][row['block_events']]['windows'][row['window_end_ns']]['paired_differences'][row['comparison']]
            assert float(row['point']) == src['point']
            np.testing.assert_array_equal([float(row['interval95_low']),float(row['interval95_high'])],src['interval95'])
    return {'CSV_tables':12,'rows':1238,'archive_bytes_equal_to_tables':True,
            'source_and_table_hashes_checked':True,'key_numbers_and_missing_intervals_checked':True}


def main(out=full.OUT):
    parent_path = p.OUT/'record.json'
    parent_digest = p.sha(parent_path)
    parent = read(parent_path)
    reports = [read(out/name) for name in ('full_cohort_covariance.json','mc_charge_matched.json')]
    assert all(r['parent_record_sha256'] == parent_digest for r in reports)
    checks = {'parent_artifacts_unchanged':check_hashes(parent['outputs_sha256'])}
    checks['parent_source_snapshots_unchanged'] = check_hashes(parent['source_sha256'],p.OUT/'source_snapshot')
    print('Parent archive preserved',flush=True)
    logs = out/'logs'; logs.mkdir(exist_ok=True)
    result = subprocess.run([sys.executable,str(p.HERE/'test_robustness.py')],capture_output=True,text=True)
    (logs/'test_robustness.py.log').write_text(result.stdout+result.stderr)
    assert result.returncode == 0, result.stdout+result.stderr
    assert 'Ran 7 tests' in result.stderr
    checks['unit_tests'] = {'passed':True,'count':7}
    checks['full_cohort'] = verify_full(out,reports[0])
    print('Full/frozen IDs, sufficient statistics and paired intervals verified',flush=True)
    checks['charge_matched_MC'] = verify_mc(out,reports[1])
    checks['supplementary_tables'] = verify_tables(out,*reports)
    print('MC support, weights, intervals and12 supplementary tables verified',flush=True)
    fig = read(out/'figures/manifest.json')
    assert fig['code_sha256'] == p.sha(p.HERE/'plot_robustness.py')
    check_hashes(fig['inputs_sha256'],out)
    assert fig['stems'] == ['figS01_full_cohort','figS02_charge_matched_mc']
    for stem in fig['stems']:
        for ext,magic in [('png',b'\x89PNG'),('pdf',b'%PDF')]:
            with (out/'figures'/f'{stem}.{ext}').open('rb') as stream:
                assert stream.read(len(magic)) == magic
        assert '<svg' in (out/'figures'/f'{stem}.svg').read_text()
    checks['figures'] = {'count':2,'formats':['png','svg','pdf'],
        'visual_review':'Both final PNG canvases inspected. S02 label overlaps and shaded-axis redraw corrected; final plots legible. PDF/SVG share the ROOT canvases; PDF magic checked, not separately rasterized.'}
    documents = [p.PROJECT/'README.md',p.PROJECT/'EXPERIMENT_INFO.md',p.SIM/'README.md',p.SIM/'RESULTS.md',
                 p.HERE/'README.md',p.HERE/'PHASE2_RESULTS.md',p.HERE/'MANUSCRIPT_DRAFT.md',p.HERE/'LOCAL_EVIDENCE.md']
    linked = 0
    for doc in documents:
        for match in re.finditer(r'\]\((<[^>]+>|[^)\n]+)\)',doc.read_text()):
            target = match.group(1).strip('<>')
            if target.startswith(('http://','https://','#')):
                continue
            target = re.sub(r':\d+$','',unquote(target.split('#')[0]))
            path = (doc.parent/target).resolve() if not target.startswith('/') else Path(target)
            if path == out/'record.json':
                continue
            assert path.exists(), f'Broken local link in {doc}: {target}'
            linked += 1
    checks['existing_local_document_targets'] = linked
    snapshots = out/'source_snapshot'; snapshots.mkdir(exist_ok=True)
    sources = sorted(set(p.HERE.glob('*.py'))|set(p.HERE.glob('publication_*.cc'))|set(p.SIM.glob('*.py')))
    hashes = {}
    for source in sources+documents:
        rel = source.relative_to(p.PROJECT); target = snapshots/rel
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target)
        hashes[str(rel)] = p.sha(source)
        assert p.sha(target) == hashes[str(rel)]
    for name in ['sipm_robustness_full_prepare','sipm_robustness_full','sipm_robustness_mc_charge',
                 'sipm_robustness_tests','sipm_robustness_figures','sipm_robustness_tables']:
        source = Path('/tmp')/(name+'.log')
        if source.exists():
            shutil.copyfile(source,logs/source.name)
    assert p.sha(parent_path) == parent_digest
    outputs = {str(path.relative_to(p.PROJECT)):p.sha(path) for path in sorted(out.rglob('*'))
               if path.is_file() and path.name != 'record.json' and 'source_snapshot' not in path.parts}
    record = {'date':'2026-10-28','scope':'Supplementary uncapped local endpoint and common-supported-Q MC sensitivity; primary publication record unchanged.',
        'checks':checks,'environment':{'python':sys.version,'numpy':np.__version__,'ROOT':ROOT.gROOT.GetVersion()},
        'parent_record':{'path':str(parent_path.relative_to(p.PROJECT)),'sha256':parent_digest},
        'source_sha256':hashes,'outputs_sha256':outputs,
        'raw_data_provenance':'Raw run12/18 waveforms were reread for the focused endpoint; event IDs and frozen endpoint verified. Raw BIN hashes inherited from parent audits, not recomputed for all BIN files.',
        'literature':'Reused previously checked local evidence; no new literature retrieval or shared-library writes.',
        'limits':['Supplementary analyses reuse previously explored data; not independent validation or new transport.',
                  'Full-record bootstrap is for50ns,500--900ns endpoint only; finer-Q and acquisition-quarter curves are point sensitivities.',
                  'Q-bin matching is source-specific with finite MC; continuous within-cell Q may differ. Empty resample cells counted.',
                  'Conditional intervals omit candidate-selection, unknown circuit, geometry, calibration and full nonstationarity uncertainty.'],
        'repository_actions':'No commit or push; unrelated working-tree edits preserved.'}
    p.save(out/'record.json',record)
    print(f'PASS:7 tests;full65034 run18 events;paired intervals;2 figures;12 tables/1238 rows;{linked} links;{len(outputs)} artifact hashes',flush=True)


if __name__ == '__main__':
    main()
