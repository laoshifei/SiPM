"""Finite-event Gaussian quadratic-form closure of the frozen ADC fluctuations."""
import numpy as np
from qdc_fluctuation_common import (PROJECT, OUT as QDC, PARENT, RC, SOURCES,
    SEGMENTS, VARIANTS, BATCHES, EVENTS, assemble, pool, mother_counts,
    read, load, save, sha, relative, serialize, data_waves)

OUT = PROJECT/'simulation/results/gaussian_closure_20261102'
SCALES = [50, 102, 202]
NORMS = ['event_mu', 'opposite_fold_mu0']
CONDITIONS = ['Q_half', 'Q_half_fold']
STATS = ['observed', 'gaussian_U', 'gaussian_plugin', 'between',
         'within_observed', 'within_gaussian_U', 'fourth_remainder', 'plugin_bias']
METRICS = ['observed_SD_ratio', 'gaussian_SD_ratio', 'observed_excess_over_data',
           'gaussian_excess_over_data', 'remainder_excess_over_data',
           'data_remainder_fraction', 'MC_remainder_fraction',
           'data_plugin_bias_fraction', 'MC_plugin_bias_fraction']
LAGS = [0, 1, 2, 5, 10, 25, 51, 101, 200]


def quadratic_vectors(raw):
    """Post and pre are separately centered; signed norm exactly recovers F*mu."""
    vectors = []
    for start in range(0, len(raw['ids']), 128):
        y = raw['waves'][start:start+128].astype(float)-raw['baseline'][start:start+128, None]
        t = raw['t10'][start:start+128]
        pre = np.ceil(t-300).astype(int)[:, None]+np.arange(200)
        post = np.ceil(t+250).astype(int)[:, None]+np.arange(1000)
        ix = np.arange(len(y))[:, None]; c = np.c_[np.zeros(len(y)), y.cumsum(1)]
        parts = []
        for width in [25, 51, 101]:
            h = width//2; r = y[:, h:-h]-(c[:, width:]-c[:, :-width])/width
            s = r[ix, post-h]; b = r[ix, pre-h]
            parts.append(np.c_[(s-s.mean(1)[:, None])/np.sqrt(999),
                               (b-b.mean(1)[:, None])/np.sqrt(199)])
        vectors.append(np.stack(parts, 1))
    x = np.concatenate(vectors)
    q = np.sum(x[:, :, :1000]**2, 2)-np.sum(x[:, :, 1000:]**2, 2)
    return x, q


def weighted_u(K, counts):
    """Distinct-original-ID pair/quadruple U moments, including block multipliers.

    A bootstrap repeat must never turn K_ii into a cross-event pair. The weights
    enter products over distinct original IDs, not distinct bootstrap replicas.
    This is a weighted U diagnostic with the parent's resampling multipliers.
    """
    c = np.asarray(counts, dtype=float)
    if c.ndim == 1: c = c[:, None]
    n = c.sum(0); c2 = c*c; p2 = c2.sum(0); p3 = (c2*c).sum(0); p4 = (c2*c2).sum(0)
    k = K.copy(); np.fill_diagonal(k, 0.)
    kc = k@c; s = np.sum(c*kc, 0)
    s2 = np.sum(c*((k*k)@c), 0)
    r2 = np.sum(c2*kc*kc, 0)
    double = np.sum(c2*((k*k)@c2), 0)
    d2 = n*n-p2; d4 = n**4-6*n*n*p2+3*p2*p2+8*n*p3-6*p4
    ok = ((c > 0).sum(0) >= 4)&(d2 > 0)&(d4 > 0)
    pair = np.divide(s2, d2, out=np.full_like(s2, np.nan), where=d2 > 0)
    quad = np.divide(s*s-4*r2+2*double, d4, out=np.full_like(s, np.nan), where=ok)
    return 2*(pair-quad)


def cell_statistics(K, counts):
    c = np.asarray(counts, dtype=float); n = c.sum(0)
    safe = np.where(n > 0, n, 1.)
    q = np.diag(K); mean = (q@c)/safe
    var = (q*q)@c/safe-mean*mean
    kc = K@c
    plugin = 2*(np.sum(c*((K*K)@c), 0)/(safe*safe)
                -(np.sum(c*kc, 0)/(safe*safe))**2)
    gu = weighted_u(K, c)*(safe-1)/safe
    return mean, var, gu, plugin*(safe-1)/safe


def build_kernels(raw, compact, p, source):
    for key in ['ids', 'q', 'fold']:
        np.testing.assert_array_equal(raw[key], compact[key])
    x, raw_q = quadratic_vectors(raw)
    mu = compact['detail'][:, 0, 0]
    np.testing.assert_allclose(raw_q/mu[:, None], compact['F'], rtol=1e-9, atol=1e-9)
    bins = np.clip(np.searchsorted(p['q_edges'], compact['q'], 'right')-1, 0, 5)
    kernels = {}; cells = []
    for b in p['support_bins'][source]:
        cut = p['conditional_cuts'][source][str(b)]; half = compact['q'] >= cut['q_quartiles'][1]
        for h in [0, 1]:
            ix = np.flatnonzero((bins == b)&(half == h)); key = f'b{b}_h{h}'
            z = x[ix]; gram = np.stack([a[:, :1000]@a[:, :1000].T-a[:, 1000:]@a[:, 1000:].T
                                      for a in z.transpose(1, 0, 2)])
            kernels[key+'_gram'] = gram; kernels[key+'_indices'] = ix
            norms = np.stack([mu[ix], compact['mu_reference'][ix]])
            kernels[key+'_denominators'] = norms; kernels[key+'_fold'] = compact['fold'][ix]
            lag = np.empty((2, 3, len(LAGS))); trace = np.empty((2, 3))
            for j, den in enumerate(norms):
                for k in range(3):
                    a = z[:, k, :1000]/np.sqrt(den[:, None]); a -= a.mean(0)
                    trace[j, k] = np.mean(a*a)
                    lag[j, k] = [np.mean(a[:, :1000-l]*a[:, l:])/trace[j, k] for l in LAGS]
            kernels[key+'_lag'] = lag; kernels[key+'_trace'] = trace
            cells.append({'Q_bin': b, 'half': h, 'key': key, 'events': len(ix),
                          'fold_counts': np.bincount(compact['fold'][ix], minlength=2).tolist(),
                          'target': cut['q2_target'][h]})
    return kernels, cells


def aggregate(kernels, cells, counts, p, source):
    """All draws at once; original Q-half masses and within-Q definition retained."""
    qbins = p['support_bins'][source]; nd = counts.shape[1]
    result = np.zeros((2, 2, 3, len(qbins), nd, len(STATS)))
    for ib, b in enumerate(qbins):
        subset = [cell for cell in cells if cell['Q_bin'] == b]
        total = sum(counts[kernels[cell['key']+'_indices']].sum(0) for cell in subset)
        for ic, cond in enumerate(CONDITIONS):
            entries = []
            valid = total >= 5
            for cell in subset:
                key = cell['key']; ix = kernels[key+'_indices']; cc = counts[ix]
                nh = cc.sum(0); valid &= (nh > 0)|(cell['target'] == 0)
                for fold in ([None] if cond == 'Q_half' else [0, 1]):
                    use = np.ones(len(ix), bool) if fold is None else kernels[key+'_fold'] == fold
                    c = cc[use]; n = c.sum(0)
                    mass = cell['target']*np.divide(n, nh, out=np.zeros(nd), where=nh > 0)
                    entries.append((key, use, c, mass))
            for im in range(2):
                for k in range(3):
                    terms = []
                    for key, use, c, mass in entries:
                        d = kernels[key+'_denominators'][im, use]
                        K = kernels[key+'_gram'][k][np.ix_(use, use)]/np.sqrt(d[:, None]*d[None, :])
                        mean, var, gu, plugin = cell_statistics(K, c)
                        terms.append((mass, mean, var, gu, plugin))
                    mean = sum(mass*m for mass, m, *_ in terms)
                    within = sum(mass*var for mass, _, var, *_ in terms)
                    between = sum(mass*(m-mean)**2 for mass, m, *_ in terms)
                    U = sum(np.where(mass > 0, mass*gu, 0.) for mass, _, _, gu, _ in terms)
                    plugin = sum(mass*g for mass, _, _, _, g in terms)
                    obs = within+between; pred = U+between
                    values = np.stack([obs, pred, plugin+between, between, within, U,
                                       obs-pred, plugin-U], 1)
                    values[~valid] = np.nan
                    result[ic, im, k, ib] = values
    return result.mean(3), result


def contrast(mc, data):
    """Array axes are arbitrary, with STATS last; never clip negative U estimates."""
    mv = mc[..., 0]; dv = data[..., 0]; mg = mc[..., 1]; dg = data[..., 1]
    with np.errstate(invalid='ignore', divide='ignore'):
        g_ratio = np.where((mg > 0)&(dg > 0), np.sqrt(mg/dg), np.nan)
        return np.stack([np.sqrt(mv/dv), g_ratio, (mv-dv)/dv, (mg-dg)/dv,
            ((mv-mg)-(dv-dg))/dv, (dv-dg)/dv, (mv-mg)/mv,
            data[..., 7]/dv, mc[..., 7]/mv], -1)


def point_rows(values):
    return {cond: {norm: {str(scale): dict(zip(STATS, values[ic, im, k]))
                for k, scale in enumerate(SCALES)} for im, norm in enumerate(NORMS)}
                for ic, cond in enumerate(CONDITIONS)}
