"""Fixed-event algebraic diagnostics of QDC tails and fluctuation dispersion."""
import numpy as np
from mc_statistics_common import (PROJECT, OUT as PARENT, WAVE, CORE, RC, SOURCES,
    SEGMENTS, VARIANTS, BATCHES, EVENTS, assemble, pool, mother_counts,
    read, load, save, sha, relative, serialize, measured, rows, event_weights)
from state_response_common import F_metrics, features
from reference import bin_layout
from pid_core_tail_common import standardized
from analyze_acquisition_state import weighted_quantile, interval

OUT = PROJECT/'simulation/results/qdc_fluctuation_20261101'
SCALES = [50, 102, 202]
TERMS = ['post_window', 'pre_window', 'normalization']
NATIVE_TERMS = ['native_signal', 'additive_residual', 'cross_term']
METRICS = ['width50_QDC_ratio', 'width95_QDC_ratio'] + [
    f'F{w}_{s}ratio' for s in ['', 'SD_'] for w in SCALES] + [
    'F202_width50_ratio', 'F202_width95_ratio', 'F202_fixed_denominator_SD_ratio',
    'F202_kurtosis_ratio', 'QDC_F202_correlation_delta', 'QDC_squared_F202_correlation_delta',
    'F202_outer_mass_delta_pp', 'QDC_tail_mass_delta_pp', 'joint_tail_mass_delta_pp',
    'F202_variance_in_QDC_tail_delta_pp', 'QDC_variance_in_F202_outer_delta_pp'] + [
    f'F202_{t}_excess_over_data_variance' for t in TERMS] + [
    'F202_Qhalf_between_fraction_delta_pp', 'F202_fold_between_fraction_delta_pp',
    'F202_within_cell_SD_ratio', 'F202_post_term_SD_ratio', 'F202_pre_term_SD_ratio',
    'F202_normalization_term_SD_ratio', 'pulse_mean_SD_ratio',
    'F202_lower_mass_delta_pp', 'F202_upper_mass_delta_pp']


def data_waves(run, seg):
    old = load(WAVE/'arrays'/f'run{run}_{seg}_data.npz')
    path = PROJECT/f'data/DT5730_DPP_PHA/DAQ/run_{run}/UNFILTERED/Data_run_{run}.BIN'
    _, dtype, n = bin_layout(path)
    raw = np.memmap(path, dtype=dtype, offset=2, mode='r', shape=(n,))
    a = features(np.array(raw['data'][old['ids']]), old['ids'], old['q'])
    np.testing.assert_allclose(a['scores'], old['scores'][:, 0], rtol=1e-12, atol=1e-12)
    for name in ['baseline', 't10', 'shape']:
        np.testing.assert_allclose(a[name], old[name], rtol=1e-12, atol=1e-12)
    a.update(span=old['span'], n_raw=old['n_raw'])
    return a


def extract(a):
    """Recover exact measured F terms and, for MC only, native signal/residual terms."""
    keep = ['ids', 'q', 'scores', 'fold', 'n_parent', 'batch', 'span', 'n_raw']
    result = {k: a[k] for k in keep if k in a}
    details = []; native = []
    for start in range(0, len(a['ids']), 256):
        stop = min(start+256, len(a['ids']))
        waves = a['waves'][start:stop]; base = a['baseline'][start:stop]; t10 = a['t10'][start:stop]
        detail = F_metrics(waves, base, t10); details.append(detail)
        if 'signal' not in a: continue
        y = waves.astype(float)-base[:, None]; sig = a['signal'][start:stop]
        cy = np.c_[np.zeros(len(y)), y.cumsum(1)]; cs = np.c_[np.zeros(len(sig)), sig.cumsum(1)]
        pre = np.ceil(t10-300).astype(int)[:, None]+np.arange(200)
        post = np.ceil(t10+250).astype(int)[:, None]+np.arange(1000)
        ix = np.arange(len(y))[:, None]; parts = []
        for k, width in enumerate([25, 51, 101]):
            half = width//2
            ry = y[:, half:-half]-(cy[:, width:]-cy[:, :-width])/width
            rs = sig[:, half:-half]-(cs[:, width:]-cs[:, :-width])/width
            variances = []
            for at in [pre, post]:
                s = rs[ix, at-half]; n = ry[ix, at-half]-s
                s = s-s.mean(1)[:, None]; n = n-n.mean(1)[:, None]
                variances.append(np.c_[np.sum(s*s, 1), np.sum(n*n, 1), 2*np.sum(s*n, 1)]/(s.shape[1]-1))
            parts.append((variances[1]-variances[0])/detail[:, k, 0, None])
        native.append(np.stack(parts, 1))
    result['detail'] = np.concatenate(details)
    result['F'] = result['detail'][:, :, 3]
    np.testing.assert_allclose(result['F'], a['F'], rtol=1e-10, atol=1e-10)
    assert np.isfinite(result['detail']).all() and np.all(result['detail'][:, :, 0] > 0)
    if native:
        result['native_terms'] = np.concatenate(native)
        np.testing.assert_allclose(result['native_terms'].sum(2), result['F'], rtol=1e-9, atol=1e-9)
    return result


def reference(data, p, source):
    result = {}
    for ef in [0, 1]:
        use = data['fold'] != ef; rr = rows(data, p, source, use.astype(int)); assert rr is not None
        result[str(ef)] = {'training_ids': data['ids'][use].tolist(), 'Q_bins': {}}
        for b, ix, w, _ in rr:
            result[str(ef)]['Q_bins'][str(b)] = {
                'pulse_mean': float(w@data['detail'][ix, 2, 0]),
                'F202_quantiles': weighted_quantile(data['F'][ix, 2], w, [.05, .5, .95]).tolist()}
    return result


def attach(a, p, source, ref, pidref):
    bins = np.clip(np.searchsorted(p['q_edges'], a['q'], 'right')-1, 0, 5)
    mu0 = np.ones(len(bins)); edges = np.full((len(bins), 2), np.nan)
    for ef in [0, 1]:
        for b in p['support_bins'][source]:
            use = (bins == b)&(a['fold'] == ef); r = ref[str(ef)]['Q_bins'][str(b)]
            mu0[use] = r['pulse_mean']; edges[use] = np.array(r['F202_quantiles'])[[0, 2]]
    mu, pre, post = (a['detail'][:, :, i] for i in range(3))
    a['terms'] = np.stack([post/mu0[:, None], -pre/mu0[:, None],
                          (post-pre)*(1/mu-1/mu0[:, None])], 2)
    np.testing.assert_allclose(a['terms'].sum(2), a['F'], rtol=1e-12, atol=1e-12)
    a['z'] = standardized(a, p, source, pidref, a['fold'])
    a['QDC_tail'] = np.abs(a['z'][:, 0]) > 2
    a['F_lower'] = a['F'][:, 2] < edges[:, 0]
    a['F_upper'] = a['F'][:, 2] > edges[:, 1]
    a['F_outer'] = a['F_lower']|a['F_upper']
    a['tail_category'] = a['QDC_tail'].astype(int)+2*a['F_outer'].astype(int)
    a['mu_reference'] = mu0
    return a


def term_cov(x, w):
    r = x-np.einsum('n,nki->ki', w, x)
    return np.einsum('n,nki,nkj->kij', w, r, r)


def summarize(a, p, source, counts=None):
    rr = rows(a, p, source, counts)
    if rr is None: return None
    items = []; raw_counts = []
    for b, ix, w, half in rr:
        x = np.c_[a['scores'][ix, 0], a['F'][ix]]; mu = w@x; r = x-mu
        cov = r.T@(w[:, None]*r); parts = np.zeros((3, 4, 4))
        for h in [0, 1]:
            sel = half == h; mass = w[sel].sum()
            if mass == 0: continue
            hm = (w[sel]@x[sel])/mass; dh = hm-mu
            parts[0] += mass*np.outer(dh, dh)
            for fold in [0, 1]:
                take = sel&(a['fold'][ix] == fold); fw = w[take]; fm = fw.sum()
                if fm == 0: continue
                mean = fw@x[take]/fm; df = mean-hm; res = x[take]-mean
                parts[1] += fm*np.outer(df, df); parts[2] += res.T@(fw[:, None]*res)
        q = np.array([weighted_quantile(x[:, j], w, [.025, .25, .75, .975]) for j in range(4)])
        tcov = term_cov(a['terms'][ix], w)
        muvar = float(w@((a['detail'][ix, 2, 0]-w@a['detail'][ix, 2, 0])**2))
        q2 = r[:, 0]**2; q2r = q2-w@q2
        category = a['tail_category'][ix]
        mass = np.array([w@(category == j) for j in range(4)])
        qvar = np.array([(w*(category == j))@(r[:, 0]**2) for j in range(4)])
        fvar = np.array([(w*(category == j))@(r[:, 3]**2) for j in range(4)])
        item = {'Q_bin': b, 'mean': mu, 'cov': cov, 'cell_cov': parts, 'terms_cov': tcov,
                'widths': np.stack([q[:, 2]-q[:, 1], q[:, 3]-q[:, 0]]),
                'F202_fourth': float(w@(r[:, 3]**4)), 'pulse_mean_variance': muvar,
                'QDC_squared_cov_F202': float(w@(q2r*r[:, 3])), 'QDC_squared_variance': float(w@(q2r*q2r)),
                'category_mass': mass, 'category_QDC_variance': qvar, 'category_F202_variance': fvar,
                'F_lower_mass': float(w@a['F_lower'][ix]), 'F_upper_mass': float(w@a['F_upper'][ix]),
                'events': len(ix), 'ESS': float(1/(w@w)),
                'category_counts': np.bincount(category, minlength=4)}
        if 'native_terms' in a: item['native_terms_cov'] = term_cov(a['native_terms'][ix], w)
        items.append(item); raw_counts.append(item['category_counts'])
    keys = ['mean', 'cov', 'cell_cov', 'terms_cov', 'F202_fourth', 'pulse_mean_variance',
            'QDC_squared_cov_F202', 'QDC_squared_variance', 'category_mass', 'category_QDC_variance',
            'category_F202_variance', 'F_lower_mass', 'F_upper_mass']
    out = {k: np.mean([r[k] for r in items], axis=0) for k in keys}
    out['widths'] = np.sqrt(np.mean([r['widths']**2 for r in items], axis=0))
    out['category_counts'] = np.sum(raw_counts, axis=0)
    out['per_Q'] = items
    if 'native_terms' in a: out['native_terms_cov'] = np.mean([r['native_terms_cov'] for r in items], axis=0)
    return out


def vectors(s):
    c = s['cov']; vf = c[3, 3]; vq = c[0, 0]; tc = s['terms_cov'][2]
    mass = s['category_mass']; hierarchy = s['cell_cov'][:, 3, 3]
    return {
        'F202_kurtosis': s['F202_fourth']/vf**2,
        'QDC_F202_correlation': c[0, 3]/np.sqrt(vq*vf),
        'QDC_squared_F202_correlation': s['QDC_squared_cov_F202']/np.sqrt(s['QDC_squared_variance']*vf),
        'F202_outer_mass': mass[2]+mass[3], 'QDC_tail_mass': mass[1]+mass[3], 'joint_tail_mass': mass[3],
        'F202_variance_in_QDC_tail': np.sum(s['category_F202_variance'][[1, 3]])/vf,
        'QDC_variance_in_F202_outer': np.sum(s['category_QDC_variance'][[2, 3]])/vq,
        'component_contribution': tc.sum(1), 'hierarchy_fraction': hierarchy/vf,
        'fixed_denominator_variance': tc[:2, :2].sum()}


def compare(m, d):
    mv = vectors(m); dv = vectors(d); mc = m['cov']; dc = d['cov']
    values = [m['widths'][i, 0]/d['widths'][i, 0] for i in [0, 1]]
    values += list(m['mean'][1:]/d['mean'][1:])+list(np.sqrt(np.diag(mc)[1:]/np.diag(dc)[1:]))
    values += list(m['widths'][:, 3]/d['widths'][:, 3])
    values += [np.sqrt(mv['fixed_denominator_variance']/dv['fixed_denominator_variance']),
               mv['F202_kurtosis']/dv['F202_kurtosis']]
    for name in ['QDC_F202_correlation', 'QDC_squared_F202_correlation']:
        values.append(mv[name]-dv[name])
    for name in ['F202_outer_mass', 'QDC_tail_mass', 'joint_tail_mass',
                 'F202_variance_in_QDC_tail', 'QDC_variance_in_F202_outer']:
        values.append(100*(mv[name]-dv[name]))
    values += list((mv['component_contribution']-dv['component_contribution'])/dc[3, 3])
    values += list(100*(mv['hierarchy_fraction']-dv['hierarchy_fraction'])[:2])
    values.append(np.sqrt(m['cell_cov'][2, 3, 3]/d['cell_cov'][2, 3, 3]))
    values += list(np.sqrt(np.diag(m['terms_cov'][2])/np.diag(d['terms_cov'][2])))
    values.append(np.sqrt(m['pulse_mean_variance']/d['pulse_mean_variance']))
    values += [100*(m[k]-d[k]) for k in ['F_lower_mass', 'F_upper_mass']]
    assert len(values) == len(METRICS)
    return np.asarray(values)


def closure(s):
    np.testing.assert_allclose(s['cell_cov'].sum(0), s['cov'], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(s['terms_cov'].sum((1, 2)), np.diag(s['cov'])[1:], rtol=1e-10, atol=1e-10)
    if 'native_terms_cov' in s:
        np.testing.assert_allclose(s['native_terms_cov'].sum((1, 2)), np.diag(s['cov'])[1:], rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(s['category_mass'].sum(), 1., atol=1e-12)
    np.testing.assert_allclose(s['category_QDC_variance'].sum(), s['cov'][0, 0], rtol=1e-12)
    np.testing.assert_allclose(s['category_F202_variance'].sum(), s['cov'][3, 3], rtol=1e-12)
