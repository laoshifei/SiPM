"""Bounded native low-frequency sensitivity after complementary variance splitting."""
from pathlib import Path
import numpy as np
from spectral_split_common import (PROJECT,PARENT,REPLAY,SCALES,FULL,OUT as SPLIT,BUILD,
    read,load,save,sha,relative,serialize,statistics,cohort,full_pack,contrasts,METRICS,interval,
    extend_mother_counts,features)

OUT=PROJECT/'simulation/results/lowfreq_taper_20261102'
SOURCE='Am241'
RUN=12
EVENTS=48000
GRID=[('identity',0.,1.)]+[(f'f{int(f*100):03d}_r{int(r*100):03d}',f,r)
    for f in [.1,.5,2.] for r in [.5,.75,.9]]
VARIANTS=['previous_r0','taper_r0','previous_r1','taper_r1']


def tapered_coefficients(coefficients,corner_MHz,dc_amplitude):
    """Apply a fixed smooth transfer, then retain the original finite FIR length.

    Return the actual unnormalized kernel. Its norm changes the configured
    noise strength; normalizing the stored coefficients must not restore the
    removed low-frequency power. There is no fitted amplitude or event scaling.
    """
    h=np.asarray(coefficients,float)
    if corner_MHz==0 or dc_amplitude==1:return h.copy(),0.
    n=16384;center=n//2;half=len(h)//2;buffer=np.zeros(n)
    buffer[center-half:center+half+1]=h
    f=np.fft.rfftfreq(n,2.)*1000
    transfer=np.sqrt((f*f+(corner_MHz*dc_amplitude)**2)/(f*f+corner_MHz**2))
    full=np.fft.irfft(np.fft.rfft(buffer)*transfer,n=n)
    kept=full[center-half:center+half+1].copy()
    lost=1-float(kept@kept)/(full@full)
    return kept,lost


def average(values):
    assert all(x is not None for x in values),'A required Q cell is missing'
    return {k:np.mean([x[k] for x in values],axis=0) for k in ['mean','cov','width2','profile']}


def objective(m,d):
    """Prespecified joint fit; all F and PID covariance entries are retained."""
    terms={}
    for label,sl in [('F',slice(3,6)),('PID',slice(0,3))]:
        target=d['cov'][sl,sl];sd=np.sqrt(np.diag(target))
        residual=(m['cov'][sl,sl]-target)/(sd[:,None]*sd[None,:])
        terms[label+'_covariance']=float(np.mean((residual[np.triu_indices(3)]/.2)**2))
    terms['F_mean']=float(np.mean(((m['mean'][3:6]/d['mean'][3:6]-1)/.1)**2))
    terms['PID_widths']=float(np.mean(((np.sqrt(m['width2']/d['width2'])-1)/.15)**2))
    terms['mean_ADC']=float((np.linalg.norm(m['profile']-d['profile'])/np.linalg.norm(d['profile'])/.03)**2)
    return {'total':sum(terms.values()),'terms':terms}


def read_data(p):
    a=load(REPLAY/'point_arrays/run12_last20_B128_data.npz')
    ref=read(SCALES/'references/run12_last20_B128.json')
    return a,ref,cohort(a,p,SOURCE,ref,'original_Q')


def merge(arrays):
    return {k:np.concatenate([a[k] for a in arrays]) for k in arrays[0]}
