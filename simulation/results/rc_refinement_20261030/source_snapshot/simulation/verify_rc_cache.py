"""Compare cached and legacy C++ readout bitwise across noise and cache invalidations."""
import copy
import tempfile
from rc_refinement_common import *


def main():
    assert not (OUT/'cache_verification.json').exists()
    binary=PROJECT/'simulation/build/rc_refinement_20261030/replay_readout_cached'
    p=read(OUT/'protocol.json');tests=[];checked=0
    for source,run in SOURCES:
        e=p['native'][source];base=read(PROJECT/e['config']);allids=np.array(e['selected_ids'])
        ids=allids[np.linspace(0,len(allids)-1,12,dtype=int)].tolist()
        with tempfile.TemporaryDirectory(prefix='sipm-readout-cache-') as folder:
            temp=Path(folder);cases=[]
            for key in ['identity','cv_half','cv_two','white','zero_residual','RC_changed','return_original','phase_changed','electronics_changed']:
                c=copy.deepcopy(base);phase=0.
                if key=='cv_half':c['signal_dependent_noise']['event_variance_cv']=.5
                elif key=='cv_two':c['signal_dependent_noise']['event_variance_cv']=2.
                elif key=='white':c['signal_dependent_noise'].update(coefficients=[1.],event_variance_cv=0.)
                elif key=='zero_residual':c['signal_dependent_noise']['variance_per_signal_adc']=0.
                elif key=='RC_changed':c['sipm']['readout']['lowpass_ns']+=2.
                elif key=='phase_changed':phase=1.
                elif key=='electronics_changed':c['electronics_noise']['states'][0]['noise_rms_mV']*=1.1
                cp=temp/f'{key}.json';save(cp,c);cases.append({'key':key,'config':str(cp),'phase_ns':phase,'require_identity':key in ['identity','return_original']})
            outputs={}
            for mode,exe in [('legacy',BUILD/'replay_readout'),('cached',binary)]:
                jobs=[{**j,'output':str(temp/f"{mode}_{j['key']}.root")} for j in cases]
                plan=temp/f'{mode}.json';save(plan,{'ids':ids,'jobs':jobs})
                result=subprocess.run([str(exe),str(PROJECT/e['path']),str(plan)],capture_output=True,text=True)
                assert result.returncode==0,result.stderr;outputs[mode]=jobs
            for j,case in enumerate(cases):
                values={};infos={}
                for mode in outputs:
                    f=ROOT.TFile.Open(outputs[mode][j]['output']);tree=f.Get('Readout');infos[mode]=json.loads(str(f.Get('RunInfo').GetString().Data()))
                    values[mode]=[(int(r.event_id),int(r.parent_entry),np.array(r.adc),np.array(r.signal_mV),np.array(r.analog_mV)) for r in tree];f.Close()
                assert infos['legacy']['config']==infos['cached']['config'] and len(values['legacy'])==len(ids)
                for old,new in zip(values['legacy'],values['cached']):
                    assert old[:2]==new[:2]
                    for x,y in zip(old[2:],new[2:]):np.testing.assert_array_equal(x,y)
                generation=infos['cached']['base_response_cache']['generation'];expected=[1,1,1,1,1,2,3,4,5][j]
                assert generation==expected,(generation,expected)
                checked+=len(ids);tests.append({'source':source,'case':case['key'],'events':len(ids),
                    'base_cache_generation':generation,'bitwise_ADC_signal_analog_equal':True})
    record={'scope':__doc__,'passed':True,'code_sha256':sha(__file__),'protocol_sha256':sha(OUT/'protocol.json'),
        'cached_binary':relative(binary),'cached_binary_sha256':sha(binary),'legacy_binary':relative(BUILD/'replay_readout'),
        'legacy_binary_sha256':sha(BUILD/'replay_readout'),'source_sha256':sha(PROJECT/'simulation/replay_readout.cc'),
        'test_cases':tests,'paired_event_cases':checked,'engineering_native_jobs':36,'new_independent_mothers':0,
        'implementation_change':'Performance-only amendment before noise grid: cache the last fixed-avalanche deterministic+electronics result; regenerate signal residual with original event seed. Geometry, optics, sensor, fitting choices and all physics parameters unchanged. 216 event-cases equal legacy ADC/signal/analog bitwise, including cache invalidation.'}
    save(OUT/'cache_verification.json',record);print('Cached readout verified:',checked,'paired event cases; exact ADC, signal and analog.',flush=True)


if __name__=='__main__':main()
