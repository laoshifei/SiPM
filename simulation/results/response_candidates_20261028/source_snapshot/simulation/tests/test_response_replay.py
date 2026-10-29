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
