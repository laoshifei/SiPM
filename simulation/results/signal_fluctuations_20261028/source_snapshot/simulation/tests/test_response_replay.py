"""Re-digitization reproduces the native G4Digitizer, and preserves transport."""
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
import numpy as np
import ROOT
import sys
HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from validate import check


class ReplayTests(unittest.TestCase):
    def test_signal_dependent_native_digitizer_and_replay(self):
        with tempfile.TemporaryDirectory(prefix='csi-signal-noise-') as directory:
            directory=Path(directory)
            for source in ['Co60','Am241']:
                base=json.loads((HERE/'configs/csi_tl_initial.json').read_text())
                base['electronics_noise']={'model':'gaussian_fir','coefficients':[.8,.6]}
                files=[]
                for enabled in (False,True):
                    config=json.loads(json.dumps(base))
                    if enabled:
                        config['signal_dependent_noise']={'model':'gaussian_fir_signal_variance',
                          'coefficients':[.8,.6],'variance_per_signal_adc':.5,'sample_period_ns':2.,'event_variance_cv':.25}
                    cfg=directory/f'{source}_{enabled}.json';cfg.write_text(json.dumps(config))
                    path=directory/f'{source}_{enabled}.root'
                    subprocess.run([str(HERE/'build/csi_sources'),str(cfg),source,'6','281103',str(path)],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
                    self.assertTrue(check(path)['passed']);files.append(path)
                original=ROOT.TFile.Open(str(files[0]));modified=ROOT.TFile.Open(str(files[1]))
                a=original.Get('Events');b=modified.Get('Events');changed=0
                info=json.loads(str(modified.Get('RunInfo').GetString().Data()))
                self.assertEqual(info['schema_version'],9)
                self.assertEqual(info['signal_dependent_noise']['model'],'gaussian_fir_signal_variance')
                cfg.write_text(json.dumps(info['config']));output=directory/f'{source}_replayed.root'
                subprocess.run([str(HERE/'build/replay_response'),str(files[1]),str(cfg),str(output)],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
                replay=ROOT.TFile.Open(str(output));c=replay.Get('Events')
                for i in range(6):
                    a.GetEntry(i);b.GetEntry(i);c.GetEntry(i)
                    for name in ['photon_t_ns','photon_x_mm','photon_y_mm','photon_e_eV','avalanche_t_ns','avalanche_q_C','signal_mV']:
                        np.testing.assert_array_equal(np.array(getattr(a,name)),np.array(getattr(b,name)))
                    for name in ['adc','analog_mV','signal_mV']:
                        np.testing.assert_array_equal(np.array(getattr(b,name)),np.array(getattr(c,name)))
                    no_signal=np.array(a.signal_mV)<=0
                    np.testing.assert_array_equal(np.array(a.adc)[no_signal],np.array(b.adc)[no_signal])
                    changed+=not np.array_equal(np.array(a.adc),np.array(b.adc))
                self.assertGreater(changed,0)
                replay.Close();modified.Close();original.Close()

    def test_original_and_alternative_response_for_both_sources(self):
        with tempfile.TemporaryDirectory(prefix='csi-response-replay-') as directory:
            directory=Path(directory)
            for source in ['Co60','Am241']:
                base=json.loads((HERE/'configs/csi_tl_initial.json').read_text())
                base['electronics_noise']={'model':'gaussian_fir','coefficients':[.8,.6]}
                cfg=directory/(source+'.json');cfg.write_text(json.dumps(base))
                parent=directory/(source+'.root')
                subprocess.run([str(HERE/'build/csi_sources'),str(cfg),source,'6','281103',str(parent)],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
                original=ROOT.TFile.Open(str(parent));tree=original.Get('Events')
                info=json.loads(str(original.Get('RunInfo').GetString().Data()))
                for name in ['same','zero','fast','afterpulse']:
                    config=json.loads(json.dumps(info['config']))
                    if name in ('zero','fast'):
                        config['readout_response']={'model':'parallel_paths','prompt_charge_fraction':.12 if name=='fast' else 0.,'prompt_decay_ns':5.,'prompt_lowpass_ns':0.}
                    if name=='afterpulse': config['sipm']['sensor']['afterpulse_probability']=.15
                    cfg.write_text(json.dumps(config));out=directory/(source+'_'+name+'.root')
                    subprocess.run([str(HERE/'build/replay_response'),str(parent),str(cfg),str(out)],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
                    report=check(out);self.assertTrue(report['passed'])
                    replay=ROOT.TFile.Open(str(out));other=replay.Get('Events');changed=0
                    for i in range(6):
                        tree.GetEntry(i);other.GetEntry(i)
                        for branch in ['photon_t_ns','photon_x_mm','photon_y_mm','photon_e_eV','photon_origin','edep_by_origin_MeV','decay_pdg']:
                            np.testing.assert_array_equal(np.array(getattr(tree,branch)),np.array(getattr(other,branch)))
                        if name!='afterpulse':
                            for branch in ['avalanche_t_ns','avalanche_q_C','avalanche_origin']:
                                np.testing.assert_array_equal(np.array(getattr(tree,branch)),np.array(getattr(other,branch)))
                        if name in ['same','zero']:
                            for branch in ['adc','signal_mV','analog_mV']:
                                np.testing.assert_array_equal(np.array(getattr(tree,branch)),np.array(getattr(other,branch)))
                            self.assertEqual(tree.approximate_trigger,other.approximate_trigger)
                        changed+=not np.array_equal(np.array(tree.adc),np.array(other.adc))
                    if name in ['fast','afterpulse']:self.assertGreater(changed,0)
                    replay.Close()
                original.Close()
                config['geometry']['source_gap_mm']+=1.;cfg.write_text(json.dumps(config))
                bad=directory/(source+'_invalid.root')
                r=subprocess.run([str(HERE/'build/replay_response'),str(parent),str(cfg),str(bad)],capture_output=True)
                self.assertNotEqual(r.returncode,0);self.assertFalse(bad.exists())


if __name__=='__main__': unittest.main(verbosity=2)
