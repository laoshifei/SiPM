"""Frozen-response, independent-transport statistics with original Q support."""
import numpy as np
from state_response_common import (PROJECT, SOURCES, SEGMENTS, PCA, features, measured,
                                   read, load, save, sha, relative, serialize, rows, event_weights)
from pid_core_tail_common import (diagnostics, standardized, contrasts as core_contrasts,
                                 NAMES as CORE_NAMES, METHODS, COVERAGES)
from rc_refinement_common import summary
from evaluate_refined_rc import contrasts as joint_contrasts, NAMES as JOINT_NAMES
from waveform_tail_common import mode_statistics, mode_contrasts, MODE_NAMES

OUT = PROJECT/'simulation/results/mc_statistics_20261030'
WAVE = PROJECT/'simulation/results/waveform_tail_20261030'
CORE = PROJECT/'simulation/results/pid_core_tail_20261030'
RC = PROJECT/'simulation/results/rc_refinement_20261030'
JOINT = PROJECT/'simulation/results/joint_constraints_20261029'
NATIVE = PROJECT/'simulation/build/npe_20261029/csi_sources'
REPLAY = PROJECT/'simulation/build/rc_refinement_20261030/replay_readout_cached'
PYTHON = '/home/xc/software/jupyter/.venv/bin/python'
VARIANTS = ['frozen', 'inherited_CV_fit', 'spectrum_CV_fit']
BATCHES = list(range(4))
EVENTS = {'Co60': 16000, 'Am241': 48000}
EXTRA = [0, 1, 2, 9, 10, 11, 12, 13, 14, 15, 16, 17]
METRICS = CORE_NAMES + [JOINT_NAMES[i] for i in EXTRA] + MODE_NAMES


def batch_key(batch, run):
    return f'b{batch:02d}_run{run}'


def projection(a, p, source, design, fold):
    bins = np.clip(np.searchsorted(p['q_edges'], a['q'], 'right')-1, 0, 5)
    z = np.full((len(bins), 3), np.nan)
    residual = np.full((len(bins), 76), np.nan)
    for ef in [0, 1]:
        d = design[str(ef)]; basis = np.array(d['vectors'])
        for b in p['support_bins'][source]:
            use = (bins == b)&(fold == ef)
            r = a['shape'][use]-np.array(d['templates'][str(b)])
            z[use] = r@basis
            residual[use] = r-z[use]@basis.T
    return {'projection': z, 'residual': residual, 'fold': fold}


def stats(a, p, source, z, modes, counts=None):
    core = diagnostics(a, p, source, z, counts)
    if core is None: return None
    return {'core': core, 'joint': summary(a, p, source, counts),
            'mode': mode_statistics(a, p, source, modes, counts)}


def contrast(m, d):
    return np.r_[core_contrasts(m['core'], d['core']),
                 joint_contrasts(m['joint'], d['joint'])[EXTRA], mode_contrasts(m['mode'], d['mode'])]


def assign(a, data, p, source, run, si, B, batch):
    prob = float(event_weights(data, p, source)@((data['ids']//B)%2 == 0))
    rng = np.random.default_rng(731030700+1000*batch+100*si+run+B)
    return (rng.random(int(a['n_parent'])) >= prob)[a['ids']].astype(int)


def assemble(batch, run, seg, B, variant, data, p, source):
    key = batch_key(batch, run); base = load(OUT/'arrays'/f'{key}_frozen.npz')
    fold = assign(base, data, p, source, run, SEGMENTS.index(seg), B, batch)
    if variant == 'frozen':
        return {**base, 'fold': fold}
    family = variant.split('_')[0]
    choices = [load(OUT/'arrays'/f'{key}_{seg}_B{B}_eval{ef}_{family}.npz') for ef in [0, 1]]
    for a in choices:
        for name in ['ids', 'q']: np.testing.assert_array_equal(a[name], base[name])
    result = {}
    for name, value in choices[0].items():
        if value.ndim and len(value) == len(fold):
            value = value.copy(); value[fold == 1] = choices[1][name][fold == 1]
        result[name] = value
    result['fold'] = fold
    return result


def pool(cohorts):
    result = {}
    n = [int(a['n_parent']) for a in cohorts]
    for name in cohorts[0]:
        if name == 'n_parent': result[name] = np.array(sum(n))
        elif name == 'ids':
            offsets = np.cumsum([0, *n[:-1]])
            result[name] = np.concatenate([a['ids']+off for a, off in zip(cohorts, offsets)])
        else: result[name] = np.concatenate([a[name] for a in cohorts])
    result['batch'] = np.concatenate([np.full(len(a['ids']), i, int) for i, a in enumerate(cohorts)])
    return result


def mother_counts(a, rng, batch_sizes):
    """Exact selected-category marginal of whole-mother multinomial bootstrap."""
    counts = np.empty(len(a['ids']), int)
    offset = 0
    for n in batch_sizes:
        use = (a['ids'] >= offset)&(a['ids'] < offset+n)
        k = int(use.sum())
        probs = np.r_[np.full(k, 1/n), (n-k)/n]
        counts[use] = rng.multinomial(n, probs)[:-1]
        offset += n
    assert offset == int(a['n_parent'])
    return counts
