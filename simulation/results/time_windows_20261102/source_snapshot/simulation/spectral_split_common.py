"""Frozen complementary spectral-state proposal and complete-pool checks."""
from pathlib import Path
import numpy as np
from full_selection_common import (PROJECT,PARENT,REPLAY,SCALES,OUT as FULL,SOURCES,EVENTS,
    read,load,save,sha,relative,serialize,statistics,cohort,prepare_pack,extend_mother_counts,
    reference,contrasts,METRICS,interval)
from state_response_common import F_metrics,features,event_weights,rows

OUT=PROJECT/'simulation/results/spectral_split_20261102'
BUILD=PROJECT/'simulation/build/spectral_split_20261102'
PYTHON='/home/xc/software/jupyter/.venv/bin/python'
DELAYS=[1,4,16]
VARIANTS=['current_r0','zero_cv_r0','split_r0','current_r1','zero_cv_r1','split_r1']


def high_innovations(white,auxiliary,delay,max_delay=16):
    n=white.shape[1]-2*max_delay;lo=max_delay-delay;hi=max_delay+delay
    return (-white[:,lo:lo+n]+2*white[:,max_delay:max_delay+n]-white[:,hi:hi+n]
            -auxiliary[:,lo:lo+n]+auxiliary[:,hi:hi+n])*.25


def covariance_summary(X,a,p,source,counts=None):
    rr=rows(a,p,source,counts)
    if rr is None:raise ValueError('Missing required calibration cell')
    means=[];covs=[]
    for _,ix,w,_ in rr:
        x=X[ix];m=w@x;r=x-m;means.append(m);covs.append(r.T@(w[:,None]*r))
    return {'mean':np.mean(means,0),'cov':np.mean(covs,0)}


def covariance_weights(a,p,source,mask):
    rr=rows(a,p,source,mask.astype(int));weight=np.zeros(len(a['ids']))
    for _,ix,w,_ in rr:weight[ix]=w*(1-w)/len(rr)
    return weight


def full_pack(a,p,source,ref):
    # Current charge and quality on the complete mother pool; no original-Q
    # restriction and no inferred experimental particle label.
    return prepare_pack(a,p,source,ref,'full_pool_current_selection')
