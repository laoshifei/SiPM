"""Post-evaluation descriptive audit of between-response-fold PID covariance."""
import numpy as np
from state_response_common import *
from evaluate_state_response import assemble,VARIANTS


def decomposition(a,p,source,labels):
    parts=np.zeros((3,3,3));rr=rows(a,p,source)
    for _,ix,w,_ in rr:
        x=a['scores'][ix];mu=w@x;delta=x-mu;parts[0]+=delta.T@(w[:,None]*delta)
        for fold in [0,1]:
            use=labels[ix]==fold;mass=w[use].sum()
            if not mass:continue
            mean=w[use]@x[use]/mass;parts[1]+=mass*np.outer(mean-mu,mean-mu)
            residual=x[use]-mean;parts[2]+=residual.T@(w[use,None]*residual)
    parts/=len(rr);np.testing.assert_allclose(parts[0],parts[1]+parts[2],rtol=1e-12,atol=1e-15)
    return parts


def main():
    assert not (OUT/'fold_mixture_diagnostic.json').exists()
    p=read(OUT/'protocol.json');result={}
    for source,run in SOURCES:
        for seg in SEGMENTS:
            a,_=measured(run,seg);key=f'run{run}_{seg}';result[key]={}
            for B in p['block_sizes']:
                d=decomposition(a,p,source,(a['ids']//B)%2)
                assigned=load(OUT/'arrays'/f'{key}_alignment_B{B}.npz')['mother_assignment']
                row={'data_parts':d,'data_between_fold_fraction':np.diag(d[1])/np.diag(d[0]),'variants':{}}
                for v in VARIANTS:
                    m=assemble(run,seg,B,v);parts=decomposition(m,p,source,assigned)
                    row['variants'][v]={'parts':parts,'between_fold_fraction':np.diag(parts[1])/np.diag(parts[0]),
                        'within_fold_SD_ratio':np.sqrt(np.diag(parts[2])/np.diag(d[2]))}
                result[key][str(B)]=row
    save(OUT/'fold_mixture_diagnostic.json',serialize({'scope':__doc__,'code_sha256':sha(__file__),
        'registered_after_evaluation_sha256':sha(OUT/'evaluation.json'),'groups':result,
        'limits':'Post-evaluation diagnostic only; no refit, cohort change or candidate selection. Between-fold term includes finite-sample mean noise and deterministic differences between fitted response functions, not identified physical drift. Decomposition of weighted within-Q covariance is exact.'}))
    for B in p['block_sizes']:
        row=result['run12_last20'][str(B)]
        print('Am last20',B,'data between-fold time fraction',row['data_between_fold_fraction'][1],flush=True)
        for v,r in row['variants'].items():print(v,r['between_fold_fraction'][1],r['within_fold_SD_ratio'][1],flush=True)


if __name__=='__main__':main()
