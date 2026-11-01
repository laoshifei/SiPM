"""Evaluate original-ADC controls and opposite-block waveform modes without refitting MC."""
from waveform_tail_common import *
from publication_stats import block_counts


def main():
    assert not (OUT/'analysis.json').exists()
    p=read(OUT/'protocol.json');previous=read(PARENT/'diagnostics.json');report={'scope':__doc__,
        'code_sha256':sha(__file__),'protocol_sha256':sha(OUT/'protocol.json'),
        'common_code_sha256':sha(PROJECT/'simulation/waveform_tail_common.py'),
        'control_metrics':CONTROL_NAMES,'mode_metrics':MODE_NAMES,'groups':{},
        'new_independent_mothers':0,'new_native_replays':0,'default_changed':False};draws={}
    for source,run in SOURCES:
        for si,seg in enumerate(SEGMENTS):
            key=f'run{run}_{seg}';data=load(OUT/'arrays'/f'{key}_data.npz');old=load(PARENT/'arrays'/f'{key}_data.npz')
            for name in ['ids','q']:np.testing.assert_array_equal(data[name],old[name])
            np.testing.assert_allclose(data['scores'][:,0],old['scores'],atol=1e-12)
            report['groups'][key]={}
            for B in p['block_sizes']:
                design=read(OUT/'designs'/f'{key}_B{B}.json');df=(data['ids']//B)%2
                mc={v:load(OUT/'arrays'/f'{key}_B{B}_{v}.npz') for v in VARIANTS}
                applied={'data':apply_design(data,p,source,design,df)}
                for v,a in mc.items():applied[v]=apply_design(a,p,source,design,a['fold'])
                ds=control_statistics(data,p,source);ms={v:control_statistics(a,p,source) for v,a in mc.items()}
                dm=mode_statistics(data,p,source,applied['data']);mm={v:mode_statistics(a,p,source,applied[v]) for v,a in mc.items()}
                points={v:control_contrasts(m,ds) for v,m in ms.items()};mp={v:mode_contrasts(m,dm) for v,m in mm.items()}
                for v in VARIANTS:
                    oldpoint=previous['groups'][key][str(B)]['variants'][v]['point']
                    for j,n in enumerate(CONTROL_NAMES):
                        if n!='mean_time_delta_ns':np.testing.assert_allclose(points[v][0,j],oldpoint[n],atol=1e-12)
                boot=np.full((512,len(VARIANTS),4,len(CONTROL_NAMES)),np.nan);mb=np.full((512,len(VARIANTS),len(MODE_NAMES)),np.nan)
                rng=np.random.default_rng(610031900+run+100*si+B);n=int(mc['frozen']['n_parent'])
                for ib in range(512):
                    dc=block_counts(data['ids'],tuple(data['span']),B,rng);db=control_statistics(data,p,source,dc)
                    if db is None:continue
                    dmode=mode_statistics(data,p,source,applied['data'],dc)
                    counts=np.bincount(rng.integers(n,size=n),minlength=n)
                    for iv,v in enumerate(VARIANTS):
                        a=mc[v];c=counts[a['ids']];m=control_statistics(a,p,source,c)
                        if m is None:continue
                        boot[ib,iv]=control_contrasts(m,db);mb[ib,iv]=mode_contrasts(mode_statistics(a,p,source,applied[v],c),dmode)
                u0=np.array(design['0']['vectors']);u1=np.array(design['1']['vectors'])
                row={'data_controls':ds,'data_modes':dm,'variants':{},
                    'fold_subspace_singular_values':np.linalg.svd(u0.T@u1,compute_uv=False),
                    'training_rank3_fractions':[design[str(i)]['training_rank3_variance_fraction'] for i in [0,1]],'feature_audits':{}}
                for v,a in [('data',data),*mc.items()]:
                    z=load(PARENT/'arrays'/f'{key}_B{B}_{v}_z.npz');np.testing.assert_array_equal(z['ids'],a['ids'])
                    row['feature_audits'][v]=feature_audit(a,p,source,applied[v],z['z'])
                    np.savez_compressed(OUT/'arrays'/f'{key}_B{B}_{v}_modes.npz',**applied[v],ids=a['ids'])
                for iv,v in enumerate(VARIANTS):
                    row['variants'][v]={'controls':{},'native_controls':ms[v],'native_modes':mm[v],
                        'mode_point':dict(zip(MODE_NAMES,mp[v])),
                        'mode_CI95':{name:interval(mb[:,iv,j]) for j,name in enumerate(MODE_NAMES)}}
                    for c,name in enumerate(CONTROLS):
                        row['variants'][v]['controls'][name]={'point':dict(zip(CONTROL_NAMES,points[v][c])),
                            'CI95':{nm:interval(boot[:,iv,c,j]) for j,nm in enumerate(CONTROL_NAMES)},
                            'paired_from_original':{nm:interval(boot[:,iv,c,j]-boot[:,iv,0,j]) for j,nm in enumerate(CONTROL_NAMES)}}
                report['groups'][key][str(B)]=row;draws[f'{key}_B{B}_controls']=boot;draws[f'{key}_B{B}_modes']=mb
                print(key,B,'extended-spectrum time width95',np.round(points['spectrum_CV_fit'][:,CONTROL_NAMES.index('width95_negative_mean_time_ratio')],3).tolist(),
                    'mode variance ratios',np.round(mp['spectrum_CV_fit'][:6],3).tolist(),flush=True)
    save(OUT/'analysis.json',serialize(report));np.savez_compressed(OUT/'bootstrap.npz',**draws)


if __name__=='__main__':main()
