"""Full available mother-pool selection, with frozen readouts and references."""
import numpy as np
from charge_scale_common import (PROJECT, PARENT, REPLAY, OUT as SCALES, SOURCES, EVENTS,
    read, load, save, sha, relative, serialize, charge_bin, cohort, statistics, contrasts,
    reference, METRICS, interval, ROTATION, FEATURES)
from state_response_common import features, event_weights

OUT=PROJECT/'simulation/results/full_selection_20261102'
BUILD=PROJECT/'simulation/build/full_selection_20261102'
PYTHON='/home/xc/software/jupyter/.venv/bin/python'
VARIANTS=['current_r0','zero_cv_r0','current_r1','zero_cv_r1']
CONDITIONS=['old_cohort_original_Q','old_cohort_current_Q',
            'old_cohort_current_selection','full_pool_current_selection']


def full_folds(n,data,p,source,run,batch):
    prob=float(event_weights(data,p,source)@(data['fold']==0))
    rng=np.random.default_rng(731030700+1000*batch+100*2+run+128)
    return (rng.random(n)>=prob).astype(np.int8)


def extend_mother_counts(ids,old_ids,old_counts,source,seed):
    """Exact conditional full-pool multinomial extension of inherited counts.

    Given the old selected-category counts, the remaining draws are uniform
    over the old excluded mothers. Aggregate never-used mothers into one cell.
    This preserves every inherited bootstrap comparison and couples entrants.
    """
    assert np.all(np.diff(ids)>0) and np.all(np.diff(old_ids)>0)
    lookup=np.searchsorted(ids,old_ids);np.testing.assert_array_equal(ids[lookup],old_ids)
    result=np.zeros((len(old_counts),len(ids)),np.int32);result[:,lookup]=old_counts
    rng=np.random.default_rng(seed);n=EVENTS[source]
    for batch in range(4):
        old=(old_ids>=batch*n)&(old_ids<(batch+1)*n)
        take=(ids>=batch*n)&(ids<(batch+1)*n)&~np.isin(ids,old_ids)
        indices=np.flatnonzero(take);unselected=n-int(old.sum());k=len(indices)
        assert k<=unselected
        if k==0:continue
        probabilities=np.r_[np.full(k,1/unselected),(unselected-k)/unselected]
        for draw in range(len(result)):
            remain=n-int(old_counts[draw,old].sum());assert remain>=0
            result[draw,indices]=rng.multinomial(remain,probabilities)[:-1]
    np.testing.assert_array_equal(result[:,lookup],old_counts)
    return result


def prepare_pack(a,p,source,ref,condition):
    old=a['old_common'].astype(bool);selected=a['selected'].astype(bool)
    allowed=old if condition in CONDITIONS[:2] else old&selected if condition==CONDITIONS[2] else selected
    q=a['q'] if condition==CONDITIONS[0] else a['qcalc']
    bins=charge_bin(q,p)-1;Y=np.full((len(q),len(FEATURES)),np.nan);Y[:,:6]=a['X'][:,:6]
    # Unselected mothers may have no valid integration window. Do not require
    # a fabricated profile/charge for them or silently discard a selected NaN.
    for fold in [0,1]:
        for b in p['support_bins'][source]:
            take=allowed&(a['fold']==fold)&(bins==b)
            z=a['X'][take,3:6]/np.asarray(ref[str(fold)][str(b)]['mean_F'])-1
            Y[take,6:]=z@ROTATION.T
    cells=[]
    for b in p['support_bins'][source]:
        ix=np.flatnonzero(allowed&(bins==b));cut=p['conditional_cuts'][source][str(b)]
        cells.append((b,ix,(q[ix]>=cut['q_quartiles'][1]).astype(int),np.asarray(cut['q2_target'])))
    return {'Y':Y,'mean_wave':a['mean_wave'],'cells':cells,'qcalc':a['qcalc'],'q':q,
        'ids':a['ids'],'source':source,'condition':condition}
