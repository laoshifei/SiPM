"""Numerically verified compiled execution of the unchanged publication protocol.

Only the EM inner loop is accelerated. Quantiles, initial seeds, all equations,
convergence criteria, PCA and the random draws are unchanged. This execution
extension is recorded separately from the already frozen statistical protocol.
"""
import argparse
import ctypes
import json
import math
import subprocess
from pathlib import Path
import numpy as np
import publication_stats as p

LIBRARY = None


def build(out):
    global LIBRARY
    source = p.HERE/'publication_em.cc'
    library = out/('publication_em_'+p.sha(source)[:16]+'.so')
    if not library.exists():
        subprocess.run(['g++','-std=c++17','-O2','-shared','-fPIC',str(source),'-o',str(library)], check=True)
    LIBRARY = ctypes.CDLL(str(library))
    LIBRARY.publication_em.argtypes = [np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS'),ctypes.c_int,
        np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS'),ctypes.c_double,ctypes.c_double,
        ctypes.c_double,ctypes.c_int,ctypes.c_double,np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')]
    LIBRARY.publication_em.restype = None
    return library


def fit(x, group, cores=2, initial=None, background_scale_factor=1., max_iter=400, tolerance=1e-7):
    if cores != 2 or initial is not None: raise ValueError('Accelerated path only implements fresh two-core fits')
    x = np.ascontiguousarray(x, dtype=np.float64)
    groups, gi = np.unique(group, return_inverse=True)
    gi = np.ascontiguousarray(gi,dtype=np.int32)
    if len(groups)!=2 or not np.isfinite(x).all() or len(x)<20 or np.ptp(x)<=0:
        raise ValueError('Need two groups and20 finite nonconstant events')
    q05,q95 = np.quantile(x,[.05,.95]); scale=max(float(q95-q05),float(np.std(x))*.05,1e-8)
    floor=scale*.01; center=float((q05+q95)/2)
    seeds=[np.sort([np.median(x[gi==j]) for j in range(2)]),np.quantile(x,[.2,.8]),np.quantile(x,[.35,.65])]
    best=None
    for means in seeds:
        output=np.empty(14);seed=np.r_[means,scale]
        LIBRARY.publication_em(x,gi,len(x),seed,center,scale*background_scale_factor,floor,max_iter,tolerance,output)
        mu=output[:2];sd=output[2:4];ll=float(output[10])
        model={'means':mu.tolist(),'sigmas':sd.tolist(),'weights':output[4:10].reshape(2,3).tolist(),
          'background_center':center,'background_scale':scale*background_scale_factor,'sigma_floor':floor,
          'groups':groups.astype(int).tolist(),'log_likelihood':ll,'bic':8*math.log(len(x))-2*ll,
          'free_parameters':8,'n':len(x),'converged':bool(output[11]),'iterations':int(output[12]),
          'at_sigma_floor':bool(np.any(sd<=floor*1.01)), 'history_min_step':float(output[13]),
          'background_policy':'Fixed broad Cauchy nuisance; center/scale estimated once from training data',
          'shape_fom_fwhm':float((mu[1]-mu[0])/(2*np.sqrt(2*np.log(2))*sum(sd)))}
        if best is None or ll>best['log_likelihood']:best=model
    return best


def verify(out, library):
    report=json.loads((out/'point.json').read_text());d=p.training(out);errors=[]
    for w,end in enumerate(p.ENDS):
        scores,_=p.pca_scores(d,w,np.ones(len(d['ids']),int))
        for reference in report['windows'][str(end*2)]['fits']:
            method=p.METHODS.index(reference['method']);ix=d['charge_bin']==reference['bin']
            candidate=fit(scores[ix,method],d['run'][ix],max_iter=600,tolerance=1e-7)
            for key in ['means','sigmas','weights','background_center','background_scale','shape_fom_fwhm','log_likelihood']:
                np.testing.assert_allclose(candidate[key],reference[key],rtol=2e-7,atol=2e-9)
            assert candidate['converged']==reference['converged']
            assert candidate['at_sigma_floor']==reference['at_sigma_floor']
            errors.append(abs(candidate['shape_fom_fwhm']-reference['shape_fom_fwhm']))
    # Altered sample weights and nuisance widths must also agree with the Python reference.
    rng=np.random.default_rng(28102899)
    for factor in (.5,1.,2.):
        ix=np.flatnonzero(d['charge_bin']==0)
        repeated=np.repeat(ix,rng.integers(0,4,len(ix)))
        a=fit(scores[repeated,2],d['run'][repeated],background_scale_factor=factor,max_iter=600)
        b=p.fit_mixture(scores[repeated,2],d['run'][repeated],background_scale_factor=factor,max_iter=600)
        np.testing.assert_allclose([a['shape_fom_fwhm'],a['log_likelihood']],
                                   [b['shape_fom_fwhm'],b['log_likelihood']],rtol=2e-7,atol=2e-9)
    record={'scope':__doc__,'protocol_sha256':p.sha(out/'protocol.json'),
      'point_fit_comparisons':len(errors),'resampled_background_comparisons':3,'max_absolute_fom_error':max(errors),
      'compiler':subprocess.check_output(['g++','--version'],text=True).splitlines()[0],
      'flags':['-std=c++17','-O2','-shared','-fPIC'],'library_sha256':p.sha(library),
      'code_sha256':{str(path.relative_to(p.PROJECT)):p.sha(path) for path in [Path(__file__),p.HERE/'publication_em.cc']}}
    p.save(out/'computation.json',record);print('Numerical equivalence:',len(errors),'point fits; max FOM difference',max(errors),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['verify','bootstrap']);parser.add_argument('--out',type=Path,default=p.OUT)
    parser.add_argument('--block-size',type=int,default=64);parser.add_argument('--draws',type=int,default=256)
    args=parser.parse_args();library=build(args.out)
    if args.stage=='verify':verify(args.out,library);return
    record=json.loads((args.out/'computation.json').read_text())
    assert p.sha(args.out/'protocol.json')==record['protocol_sha256']
    assert p.sha(library)==record['library_sha256']
    for path,digest in record['code_sha256'].items():assert p.sha(p.PROJECT/path)==digest
    p.fit_mixture=fit
    p.bootstrap(args.out,args.block_size,args.draws)


if __name__=='__main__':main()
