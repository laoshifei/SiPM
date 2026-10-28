"""Check decay content, finite ADC output, charge conservation and saved counts."""
import argparse,json
from pathlib import Path
import numpy as np
import ROOT

def check_ancestry(event):
    deposits=np.array(event.edep_by_origin_MeV);generated=np.array(event.scintillation_by_origin)
    charge=np.array(event.charge_by_origin_C)
    assert len(deposits)==len(generated)==len(charge)==6
    assert np.all(deposits>=0) and np.all(generated>=0) and np.all(charge>=0)
    assert np.isclose(deposits.sum(),event.edep_MeV,rtol=1e-10,atol=1e-12),'Energy ancestry not conserved'
    assert int(generated.sum())==event.n_scintillation,'Scintillation ancestry not conserved'
    assert np.isclose(charge.sum(),event.charge_C,rtol=1e-10,atol=1e-25),'Charge ancestry not conserved'
    origins=np.array(event.photon_origin);emitters=np.array(event.photon_emitter_pdg)
    assert len(origins)==len(emitters)==len(event.photon_t_ns)
    assert np.all((origins>=0)&(origins<6)) and not np.any(origins==0),'Unattributed optical hit'
    if hasattr(event,'photon_boundary_status'):
        boundary=np.array(event.photon_boundary_status)
        assert len(boundary)==len(origins)
        assert np.all(np.isin(boundary,[1,2,12])),'Non-transmitted photon counted as SiPM entrance'
    avalanche=np.array(event.avalanche_origin);emitter=np.array(event.avalanche_emitter_pdg)
    assert len(avalanche)==len(emitter)==len(event.avalanche_q_C)
    assert np.all((avalanche>=0)&(avalanche<6))
    for origin in range(6):
        assert np.isclose(np.array(event.avalanche_q_C)[avalanche==origin].sum(),charge[origin],rtol=1e-10,atol=1e-25)

def check(path):
    file=ROOT.TFile.Open(str(path));assert file and not file.IsZombie(),path
    info=json.loads(str(file.Get('RunInfo').GetString().Data()));tree=file.Get('Events')
    assert info.get('completed',True),'Run did not complete'
    assert tree.GetEntries()==info['events_requested'],'Incomplete run'
    cfg=info['config'];samples=cfg['acquisition']['samples'];dt=cfg['acquisition']['sample_period_ns']
    pdgs={};gammas=[];deposits=[];alpha=[];zeros=0;balances=[];first_times=[];scintillation=[]
    for event in tree:
        if info.get('schema_version',1)>=3:check_ancestry(event)
        adc=np.array(event.adc);assert len(adc)==samples and np.all(adc<16384)
        assert len(event.signal_mV)==samples and np.isfinite(np.array(event.signal_mV)).all()
        analog=np.array(event.analog_mV);assert np.isfinite(analog).all()
        q=np.array(event.avalanche_q_C);t=np.array(event.avalanche_t_ns)
        assert np.all(q>=0) and np.all(np.diff(t)>=0) and np.all((t>=0)&(t<samples*dt))
        rel=abs(event.charge_balance_C)/max(event.charge_C,1e-30);assert rel<1e-9
        balances.append(rel);deposits.append(float(event.edep_MeV));alpha.append(float(event.edep_alpha_MeV));zeros+=event.edep_MeV==0
        scintillation.append(int(event.n_scintillation))
        first_times.extend(list(event.decay_time_ns))
        for code,e in zip(event.decay_pdg,event.decay_energy_MeV):
            pdgs[str(code)]=pdgs.get(str(code),0)+1
            if code==22:gammas.append(float(e))
        if event.edep_MeV>0:
            assert event.edep_MeV<6.0 if info['source']=='Am241' else event.edep_MeV<3.0
    assert len(first_times)>0 and max(first_times)<samples*dt,'Parent clock conditioning failed'
    if info['source']=='Am241':
        assert pdgs.get('1000020040',0)>0,'Am alpha decays absent'
        if tree.GetEntries()>=100:assert any(abs(e-.05954)<.001 for e in gammas),'Am 59.54 keV gamma absent'
    else:
        assert pdgs.get('11',0)>0,'Co beta decay absent'
        assert any(abs(e-1.173)<.005 for e in gammas) and any(abs(e-1.332)<.005 for e in gammas),'Co cascade not present'
    a=np.array(alpha);d=np.array(deposits)
    report={'passed':True,'source':info['source'],'events':int(tree.GetEntries()),'zero_deposit_events':int(zeros),
        'deposited_events':int(np.sum(d>0)),'alpha_deposit_events':int(np.sum(a>0)),
        'alpha_deposit_median_MeV':float(np.median(a[a>0])) if np.any(a>0) else None,
        'energy_deposit_p05_p50_p95_MeV':np.quantile(d[d>0],[.05,.5,.95]).tolist() if np.any(d>0) else [],
        'decay_product_counts':pdgs,'max_relative_charge_balance_error':float(max(balances)),
        'prompt_gamma_lines_counts':{str(energy):int(np.sum(np.abs(np.array(gammas)-energy)<.001)) for energy in (.05954,1.17323,1.33249)},
        'ancestry_checks':info.get('schema_version',1)>=3,
        'scope':'Numerical and source-physics checks, not validation against experimental waveform',
        'source_gap_mm':cfg['geometry']['source_gap_mm'],'transport_only':cfg['optical']['yield_photons_per_MeV']==0}
    if report['transport_only']:assert max(scintillation)==0,'Scintillation unexpectedly enabled'
    file.Close();return report

def main():
    p=argparse.ArgumentParser();p.add_argument('files',type=Path,nargs='+');args=p.parse_args()
    for path in args.files:
        report=check(path);(path.parent/(path.stem+'_validation.json')).write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report))
if __name__=='__main__':main()
