"""Meaningful selection compatibility and conditional-bootstrap checks."""
from full_selection_common import *
from extract_full_selection import quality
from compare import mc_wave_features


def main():
    p=read(OUT/'protocol.json');reports={}
    for source,run in SOURCES:
        old=load(PARENT/'arrays'/f'b00_run{run}_frozen.npz')
        # More than one processing chunk, with distinctly different noise
        # populations, tests that the global reference is not per-chunk.
        waves=np.concatenate([old['waves'][:300],old['waves'][-300:]])
        fold=np.arange(len(waves))%2
        f,g,_,_,_=quality(waves,fold);expected,eg=mc_wave_features(waves,batch_size=73)
        np.testing.assert_array_equal(g,eg)
        for k in f:np.testing.assert_array_equal(f[k],expected[k])
        data=load(REPLAY/'point_arrays'/f'run{run}_last20_B128_data.npz')
        ref=reference(data,p,source)
        a=load(REPLAY/'point_arrays'/f'run{run}_last20_B128_current.npz')
        a['qcalc']=np.rint(a['mean_wave'].sum(1)*160)/8
        a['old_common']=np.ones(len(a['ids']),bool);a['selected']=np.ones(len(a['ids']),bool)
        for c,oldc in zip(CONDITIONS[:2],['original_Q','current_Q']):
            left=statistics(prepare_pack(a,p,source,ref,c));right=statistics(cohort(a,p,source,ref,oldc))
            for k in ['mean','cov','width2','profile']:np.testing.assert_array_equal(left[k],right[k])
        reports[source]={'QC_events':len(waves),'original_and_current_charge_statistics_identical':True}
    # Exact conditional multinomial moments can be checked against closed form.
    # Include an old count of zero and a batch with all remaining draws assigned.
    source='Co60';n=EVENTS[source];oldids=np.array([0,3,n+2]);ids=np.array([0,1,3,n+1,n+2])
    counts=np.tile([2,1,0],(20000,1));got=extend_mother_counts(ids,oldids,counts,source,89021)
    np.testing.assert_array_equal(got[:,[0,2,4]],counts)
    for col,remaining,unselected in [(1,n-3,n-2),(3,n,n-1)]:
        mu=remaining/unselected;var=mu*(1-1/unselected)
        assert abs(got[:,col].mean()-mu)<.04 and abs(got[:,col].var()-var)<.07
    assert np.all(got[:,:3].sum(1)<=n) and np.all(got[:,3:].sum(1)<=n)
    # An excluded parent with NaN charge/profile must not invalidate valid cells.
    a={k:np.concatenate([v,v[:1]],axis=0) if v.ndim and len(v)==len(a['ids']) else v for k,v in a.items()}
    a['old_common'][-1]=False;a['selected'][-1]=False;a['mean_wave'][-1]=np.nan;a['qcalc'][-1]=np.nan
    assert statistics(prepare_pack(a,p,source='Am241',ref=ref,condition=CONDITIONS[-1])) is not None
    save(OUT/'algebra_tests.json',{'comparisons':reports,'conditional_multinomial_draws':20000,
        'excluded_invalid_waveform_not_used':True,'code_sha256':sha(__file__)})
    print('QC/global-reference, original-statistic, excluded-NaN and conditional-bootstrap checks passed.')


if __name__=='__main__':main()
