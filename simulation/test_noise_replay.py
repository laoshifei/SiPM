"""Independent finite-repeat moment and native ROOT replay checks."""
import copy
import subprocess
import ROOT
from noise_replay_common import *


def algebra():
    rng=np.random.default_rng(2026110207);w=np.array([.1,.15,.2,.25,.3]);mu=np.arange(10).reshape(5,2)*.1
    sigma=np.array([.3,.5,.7,.9,1.1]);center=mu-w@mu
    truth_B=center.T@(w[:,None]*center);truth_W=np.eye(2)*np.sum(w*sigma*sigma)
    truth_one=truth_B+np.eye(2)*np.sum(w*(1-w)*sigma*sigma)
    values=[]
    for _ in range(2000):
        X=mu[:,None,:]+rng.normal(size=(5,8,2))*sigma[:,None,None]
        s=nested_moments(X,w)
        flat=X.reshape(-1,2);weights=np.repeat(w/8,8)
        np.testing.assert_allclose(s['flat_covariance'],np.cov(flat.T,aweights=weights,ddof=0),rtol=1e-10,atol=1e-10)
        values.append([s['between_corrected'],s['within'],s['one_readout_expected_covariance']])
    estimated=np.mean(values,0)
    for x,y in zip(estimated,[truth_B,truth_W,truth_one]):np.testing.assert_allclose(x,y,rtol=.07,atol=.012)
    a=np.tile(np.array([-1.,1.])[None,:,None],(5,1,1))
    negative=nested_moments(a,w)['between_corrected'][0,0];assert negative<0
    return {'trials':2000,'independent_flat_covariance_checks':2000,'expected':np.stack([truth_B,truth_W,truth_one]),
            'estimated':estimated,'negative_between_correction_retained':negative}


def root_arrays(path):
    f=ROOT.TFile.Open(str(path));assert f and not f.IsZombie();t=f.Get('Readout')
    info=__import__('json').loads(str(f.Get('RunInfo').GetString().Data()))
    keys=['event_id','adc','signal_mV','analog_mV','baseline_readout_seed','signal_readout_seed']
    result={k:[] for k in keys}
    for row in t:
        for k in keys:
            v=getattr(row,k);result[k].append(np.array(v) if k in ['adc','signal_mV','analog_mV'] else int(v))
    f.Close();return {k:np.asarray(v) for k,v in result.items()},info


def main():
    dest=OUT/'replay_tests.json';assert not dest.exists()
    p=read(PARENT/'protocol.json');tests=[]
    for source,run in SOURCES:
        entry=p['entries'][f'b00_run{run}'];a=load(PARENT/'arrays'/f'b00_run{run}_frozen.npz')
        take=np.unique(np.linspace(0,len(a['ids'])-1,6).astype(int));ids=a['ids'][take].tolist()
        directory=OUT/'logs'/f'api_test_run{run}';directory.mkdir(exist_ok=False)
        jobs=[]
        for scope in ['all_readout','signal_only','electronics_only']:
            for replica in [0,1,2]:
                jobs.append({'key':f'{scope}_{replica}','config':str(PROJECT/entry['config']),
                    'phase_ns':0.,'noise_scope':scope,'noise_replica':replica,'require_identity':replica==0})
        jobs.append({**jobs[1],'key':'repeat_one'})
        collections={}
        for cached in [False,True]:
            label='cached' if cached else 'plain';binary=BUILD/('replay_readout_cached' if cached else 'replay_readout')
            plan={'ids':ids,'jobs':[{**j,'output':str(directory/f'{label}_{j["key"]}.root')} for j in jobs]}
            path=directory/f'{label}.json';save(path,plan)
            with (directory/f'{label}.log').open('w') as log:
                subprocess.run([str(binary),str(PROJECT/entry['path']),str(path)],cwd=PROJECT,stdout=log,stderr=subprocess.STDOUT,check=True)
            collections[label]={}
            for job in plan['jobs']:
                arr,info=root_arrays(job['output']);collections[label][job['key']]=arr
                assert info['new_independent_mothers']==0 and info['noise_replica']==job['noise_replica']
                np.testing.assert_array_equal(arr['event_id'],ids)
                if job['noise_replica']==0:
                    np.testing.assert_array_equal(arr['adc'],a['waves'][take]);assert info['equal_parent_ADC']==len(ids)
        for key in collections['plain']:
            x=collections['plain'][key];y=collections['cached'][key]
            for field in x:np.testing.assert_array_equal(x[field],y[field])
            np.testing.assert_array_equal(x['signal_mV'],collections['plain']['all_readout_0']['signal_mV'])
        for field in collections['plain']['repeat_one']:
            np.testing.assert_array_equal(collections['plain']['repeat_one'][field],collections['plain']['all_readout_1'][field])
        z=collections['plain'];assert not np.array_equal(z['all_readout_1']['adc'],z['all_readout_2']['adc'])
        np.testing.assert_array_equal(z['all_readout_1']['baseline_readout_seed'],z['electronics_only_1']['baseline_readout_seed'])
        np.testing.assert_array_equal(z['all_readout_1']['signal_readout_seed'],z['signal_only_1']['signal_readout_seed'])
        np.testing.assert_array_equal(z['all_readout_0']['baseline_readout_seed'],z['signal_only_1']['baseline_readout_seed'])
        np.testing.assert_array_equal(z['all_readout_0']['signal_readout_seed'],z['electronics_only_1']['signal_readout_seed'])
        tests.append({'source':source,'mothers':len(ids),'cases':len(jobs),'cached_plain_bitwise':True,'zero_identity':True,'reproducibility':True,'unchanged_signal':True})
    ctest=BUILD/'Testing/Temporary/LastTest.log';text=ctest.read_text();assert text.count('Test Passed.')==6
    save(dest,serialize({'native_tests':tests,'algebra':algebra(),'ctest_sha256':sha(ctest),'ctest_passed':6,
        'binary_sha256':{relative(BUILD/n):sha(BUILD/n) for n in ['replay_readout','replay_readout_cached']},
        'code_sha256':sha(__file__)}))
    print('Native replay API, all six CTests, and finite-repeat moment checks passed.',flush=True)


if __name__=='__main__':main()
