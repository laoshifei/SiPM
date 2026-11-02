"""Reuse the sealed full pool only when both selected configs are exact identities."""
import shutil
from lowfreq_common import *


def main():
    assert not (OUT/'native_protocol.json').exists();p=read(OUT/'protocol.json');fit=read(OUT/'fit.json')
    assert all(f['candidate']=='identity' for f in fit['folds'].values()),'Nonidentity candidates require new native validation'
    old_native=read(SPLIT/'native_protocol.json');old_record=read(SPLIT/'record.json');entries={};reused={};copied={}
    for key in p['entries']:
        old=old_native['entries'][key];plan=read(PROJECT/old['plan'])
        for index in [2,3]:
            for ef in [0,1]:
                chosen=PROJECT/fit['folds'][str(ef)]['config'];assert read(chosen)==read(plan['cases'][index]['configs'][ef])
        for field in ['output','plan']:
            path=PROJECT/old[field];assert sha(path)==old_record['outputs_sha256'][relative(path)];reused[relative(path)]=sha(path)
        entries[key]=old
        for r in [0,1]:
            source=SPLIT/'arrays'/f'{key}_split_r{r}.npz';dest=OUT/'arrays'/source.name
            assert sha(source)==old_record['outputs_sha256'][relative(source)];shutil.copyfile(source,dest)
            assert sha(dest)==sha(source);copied[relative(dest)]={'source':relative(source),'sha256':sha(dest)}
        for kind in ['native','extraction']:
            source=SPLIT/'logs'/f'{key}_{kind}.json';assert sha(source)==old_record['outputs_sha256'][relative(source)]
            record=read(source);record['reused_from']=relative(source);record['new_native_execution']=False
            if kind=='native':record['seconds']=0.
            save(OUT/'logs'/source.name,record);reused[relative(source)]=sha(source)
    save(OUT/'native_protocol.json',{'entries':entries,'fit_sha256':sha(OUT/'fit.json'),'protocol_sha256':sha(OUT/'protocol.json'),
        'binary_sha256':old_native['binary_sha256'],'code_sha256':sha(__file__),'new_native_execution':False,
        'scope':'Both selected identity configurations exactly equal the sealed full-pool split configurations, including every numerical parameter. Reuse their existing full-pool ADC and feature arrays; no duplicate native execution.'})
    save(OUT/'extraction_protocol.json',{'adapter_sha256':sha(__file__),'extractor_sha256':sha(PROJECT/'simulation/extract_spectral_split.py'),
        'native_protocol_sha256':sha(OUT/'native_protocol.json'),'mode':'Byte-identical copies of previously verified full-pool extraction; no new extraction.'})
    save(OUT/'full_pool_reuse.json',{'reason':'Both opposite-fold minima are identity; no new parameter candidate needs repeated native simulation.',
        'reused_inputs_sha256':reused,'copied_arrays':copied,'fit_sha256':sha(OUT/'fit.json'),'new_full_pool_readouts':0,
        'referenced_parent_events':192000,'referenced_logical_readouts':768000,'code_sha256':sha(__file__)})
    print('Selected configurations are exact identities; reused verified192000-mother full pool without new readouts.',flush=True)


if __name__=='__main__':main()
