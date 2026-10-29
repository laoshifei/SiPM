"""Small statistical primitives for lower-score-is-alpha conditional PID."""
import numpy as np
from scipy.stats import beta


def valid_weights(x,w=None):
    x=np.asarray(x,float);w=np.ones(len(x)) if w is None else np.asarray(w,float)
    if x.ndim!=1 or w.shape!=x.shape or not np.isfinite(x).all() or not np.isfinite(w).all():
        raise ValueError('Finite one-dimensional scores/weights required')
    if np.any(w<0) or w.sum()<=0:raise ValueError('Nonnegative weights with positive sum required')
    return x,w


def auc(alpha,gamma,wa=None,wg=None):
    a,wa=valid_weights(alpha,wa);g,wg=valid_weights(gamma,wg)
    order=np.argsort(g);g=g[order];wg=wg[order];c=np.r_[0.,wg.cumsum()]
    left=np.searchsorted(g,a,'left');right=np.searchsorted(g,a,'right')
    wins=wg.sum()-c[right]+.5*(c[right]-c[left])
    return float(np.dot(wa,wins)/(wa.sum()*wg.sum()))


def roc(alpha,gamma):
    a,_=valid_weights(alpha);g,_=valid_weights(gamma)
    thresholds=np.unique(np.r_[a,g])
    return {'thresholds':thresholds,'alpha_acceptance':np.r_[0,np.searchsorted(np.sort(a),thresholds,'right')/len(a)],
            'gamma_leakage':np.r_[0,np.searchsorted(np.sort(g),thresholds,'right')/len(g)]}


def binomial(k,n):
    if int(n)!=n or int(k)!=k or n<=0 or k<0 or k>n:raise ValueError('Require integers 0<=k<=n and n>0')
    low=0. if k==0 else float(beta.ppf(.025,k,n-k+1))
    high=1. if k==n else float(beta.ppf(.975,k+1,n-k))
    upper=1. if k==n else float(beta.ppf(.95,k+1,n-k))
    return {'passed':int(k),'total':int(n),'fraction':float(k/n),
            'exact_two_sided95':[low,high],'one_sided95_upper':upper}


def gaussian_equivalent_fom(alpha,gamma,wa=None,wg=None):
    a,wa=valid_weights(alpha,wa);g,wg=valid_weights(gamma,wg)
    if wa.sum()<=1 or wg.sum()<=1:raise ValueError('Need at least two replicated observations per class')
    ma=np.dot(a,wa)/wa.sum();mg=np.dot(g,wg)/wg.sum()
    sa=np.sqrt(np.dot((a-ma)**2,wa)/(wa.sum()-1));sg=np.sqrt(np.dot((g-mg)**2,wg)/(wg.sum()-1))
    if sa+sg<=0:raise ValueError('Zero total spread')
    return float(abs(ma-mg)/(2*np.sqrt(2*np.log(2))*(sa+sg)))


def operating_point(alpha,gamma,threshold):
    a,_=valid_weights(alpha);g,_=valid_weights(gamma)
    return {'threshold':float(threshold),'alpha_acceptance':binomial(int(np.sum(a<=threshold)),len(a)),
            'gamma_leakage':binomial(int(np.sum(g<=threshold)),len(g))}
