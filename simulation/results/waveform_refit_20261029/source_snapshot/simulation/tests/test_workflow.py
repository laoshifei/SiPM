"""Integration checks for independent seeds, overwrite guards and data isolation."""
import json,subprocess,sys,tempfile,unittest
from pathlib import Path
import numpy as np
import ROOT

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from reference import TIME_NS,aligned_shapes
from calibrate import expectation
from validate import check_ancestry

class WorkflowChecks(unittest.TestCase):
    def test_training_split_isolation(self):
        path=HERE/'results/reference/references.npz'
        if not path.exists():self.skipTest('Measured references not generated')
        with np.load(path) as data:
            for run,n in ((3,4776),(12,7516)):
                train=data[f'run{run}_train_indices'];middle=data[f'run{run}_middle_indices'];late=data[f'run{run}_late_indices']
                self.assertTrue(np.all(train<int(n*.6)))
                self.assertTrue(np.all((middle>=int(n*.6))&(middle<int(n*.8))))
                self.assertTrue(np.all(late>=int(n*.8)))
                self.assertFalse(set(train)&set(middle));self.assertFalse(set(train)&set(late))

    def test_alignment_preserves_shape_under_adc_scaling_and_offset(self):
        t=np.arange(2496)*2.;y=np.maximum(t-820,0);pulse=np.where(t>=820,np.exp(-y/1400)-np.exp(-y/90),0)
        waves=np.array([3000+800*pulse,3070+400*pulse])
        shapes,indices=aligned_shapes(waves)
        self.assertEqual(list(indices),[0,1]);np.testing.assert_allclose(shapes[0],shapes[1],rtol=0,atol=2e-17)
        for shape in shapes:self.assertAlmostEqual(shape[(TIME_NS>=-40)&(TIME_NS<3000)].sum()*2,1.,places=12)

    def test_effective_forward_model_area_and_causality(self):
        for fraction in (0.,.2,.8,1.):
            y=expectation(480,1370,94,fraction)
            self.assertTrue(np.isfinite(y).all());self.assertTrue((y>=-1e-14).all())
            self.assertAlmostEqual(y[(TIME_NS>=-40)&(TIME_NS<3000)].sum()*2,1.,places=12)
            self.assertLess(np.max(np.abs(y[TIME_NS<-100])),1e-13)

    def test_geant4_seed_replay_and_output_guards(self):
        binary=HERE/'build/csi_sources'
        if not binary.exists():self.skipTest('Geant4 build required')
        config=HERE/'results/calibration/csi_tl_effective.json'
        if not config.exists():config=HERE/'configs/csi_tl_initial.json'
        with tempfile.TemporaryDirectory(prefix='csi-integration-') as directory:
            paths=[Path(directory)/f'replay{i}.root' for i in (0,1,2)]
            cmd=[str(binary),str(config),'Co60','6','271103']
            for path in paths[:2]:subprocess.run(cmd+[str(path)],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
            macro=Path(directory)/'replay.mac'
            macro.write_text(f'/csi/config {config}\n/csi/source Co60\n/csi/seed 271103\n/csi/output {paths[2]}\n/run/initialize\n/run/beamOn 6\n')
            subprocess.run([str(binary),str(macro)],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
            files=[ROOT.TFile.Open(str(p)) for p in paths];trees=[f.Get('Events') for f in files];detected=0
            for file in files:
                info=json.loads(str(file.Get('RunInfo').GetString().Data()))
                self.assertTrue(info['completed']);self.assertEqual(info['events_requested'],6)
                self.assertEqual(info['events_completed'],6);self.assertEqual(info['seed'],271103)
            for i in range(6):
                for tree in trees:tree.GetEntry(i)
                for tree in trees:
                    check_ancestry(tree)
                    self.assertEqual(tree.edep_by_origin_MeV[1],0.,'Co event incorrectly inherits alpha ancestry')
                for branch in ('adc','photon_t_ns','avalanche_t_ns','avalanche_q_C','decay_pdg','decay_energy_MeV'):
                    np.testing.assert_array_equal(np.array(getattr(trees[0],branch)),np.array(getattr(trees[1],branch)))
                    np.testing.assert_array_equal(np.array(getattr(trees[0],branch)),np.array(getattr(trees[2],branch)))
                detected+=len(trees[0].photon_t_ns)
            self.assertGreater(detected,0,'Replay must exercise a real optical signal')
            for file in files:file.Close()
            old=paths[0].read_bytes();r=subprocess.run(cmd+[str(paths[0])],capture_output=True)
            self.assertNotEqual(r.returncode,0);self.assertIn(b'Refusing to overwrite',r.stderr);self.assertEqual(paths[0].read_bytes(),old)
            bad=Path(directory)/'invalid.root';r=subprocess.run([str(binary),str(config),'Cs137','1','1',str(bad)],capture_output=True)
            self.assertNotEqual(r.returncode,0);self.assertFalse(bad.exists())

    def test_am_transport_replay_and_boundary(self):
        binary=HERE/'build/csi_sources';config=HERE/'results/calibration/csi_tl_effective.json'
        if not binary.exists() or not config.exists():self.skipTest('Built simulation and effective config required')
        with tempfile.TemporaryDirectory(prefix='csi-am-replay-') as directory:
            paths=[Path(directory)/f'am{i}.root' for i in range(2)]
            for path in paths:
                subprocess.run([str(binary),str(config),'Am241','4','271112',str(path)],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
            files=[ROOT.TFile.Open(str(path)) for path in paths];trees=[f.Get('Events') for f in files];photons=0
            for i in range(4):
                for tree in trees:
                    tree.GetEntry(i);check_ancestry(tree)
                for branch in ('adc','photon_t_ns','photon_boundary_status','avalanche_q_C','edep_by_origin_MeV'):
                    np.testing.assert_array_equal(np.array(getattr(trees[0],branch)),np.array(getattr(trees[1],branch)))
                photons+=len(trees[0].photon_t_ns)
            self.assertGreater(photons,0)
            for f in files:f.Close()

    def test_initialized_geometry_export(self):
        binary=HERE/'build/csi_sources'
        if not binary.exists():self.skipTest('Geant4 build required')
        with tempfile.TemporaryDirectory(prefix='csi-geometry-') as directory:
            path=Path(directory)/'geometry.json';macro=Path(directory)/'geometry.mac'
            macro.write_text(f'/run/initialize\n/csi/exportGeometry {path}\n')
            subprocess.run([str(binary),str(macro)],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
            data=json.loads(path.read_text());volumes=data['volumes']
            self.assertEqual(len(volumes),9)
            crystal=next(v for v in volumes if v['name']=='CsI')
            mylar=next(v for v in volumes if v['name']=='Mylar')
            self.assertEqual(crystal['size_mm'],[10.,10.,10.]);self.assertAlmostEqual(mylar['size_mm'][2],.005)
            self.assertEqual(sum(v['name']=='PTFE' for v in volumes),4)
            self.assertAlmostEqual(data['source']['center_mm'][2],-6.005)
            old=path.read_bytes();result=subprocess.run([str(binary),str(macro)],capture_output=True)
            self.assertNotEqual(result.returncode,0);self.assertEqual(path.read_bytes(),old)

    def test_macro_errors_and_post_initialization_config_lock(self):
        binary=HERE/'build/csi_sources'
        if not binary.exists():self.skipTest('Geant4 build required')
        with tempfile.TemporaryDirectory(prefix='csi-macro-guards-') as directory:
            path=Path(directory)/'forbidden.root';macro=Path(directory)/'bad.mac'
            macro.write_text(f'/csi/output {path}\n/run/initialize\n/csi/source Co60\n/run/beamOn 1\n')
            result=subprocess.run([str(binary),str(macro)],capture_output=True)
            self.assertNotEqual(result.returncode,0);self.assertFalse(path.exists())
            macro.write_text('/csi/unknown_command 1\n')
            result=subprocess.run([str(binary),str(macro)],capture_output=True)
            self.assertNotEqual(result.returncode,0)

    def test_transport_only_energy_matrix_and_exit_foil(self):
        binary=HERE/'build/csi_sources'
        if not binary.exists():self.skipTest('Geant4 build required')
        from validate import check
        with tempfile.TemporaryDirectory(prefix='csi-transport-') as directory:
            folder=Path(directory);config=json.loads((HERE/'configs/csi_tl_initial.json').read_text())
            config['optical']['yield_photons_per_MeV']=0.;config['acquisition']['digitize']=False
            config['geometry']['source_exit_foil']={'material':'G4_Al','thickness_mm':.25,'distance_from_source_mm':.01,'width_mm':20.}
            path=folder/'config.json';path.write_text(json.dumps(config));root=folder/'transport.root'
            macro=folder/'transport.mac';geometry=folder/'geometry.json'
            macro.write_text(f'/csi/config {path}\n/csi/source Co60\n/csi/seed 281000\n/csi/output {root}\n/run/initialize\n/csi/exportGeometry {geometry}\n/run/beamOn 30\n')
            subprocess.run([str(binary),str(macro)],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
            report=check(root);self.assertTrue(report['passed']);self.assertFalse(report['digitization_enabled'])
            data=json.loads(geometry.read_text());foil=next(v for v in data['volumes'] if v['name']=='SourceExitFoil')
            self.assertEqual(foil['size_mm'],[20.,20.,.25]);self.assertEqual(foil['material'],'G4_Al')
            config['geometry']['source_exit_foil']['thickness_mm']=2.;path.write_text(json.dumps(config))
            result=subprocess.run([str(binary),str(path),'Co60','1','1',str(folder/'invalid.root')],capture_output=True)
            self.assertNotEqual(result.returncode,0);self.assertIn(b'foil must lie strictly',result.stderr)

    def test_finite_rise_macro_config_and_zero_default_replay(self):
        binary=HERE/'build/csi_sources'
        if not binary.exists():self.skipTest('Geant4 build required')
        with tempfile.TemporaryDirectory(prefix='csi-rise-') as directory:
            folder=Path(directory);base=json.loads((HERE/'configs/csi_tl_initial.json').read_text())
            files=[]
            for name,rise in [('default',None),('zero',[0.,0.]),('finite',[120.,100.])]:
                config=json.loads(json.dumps(base))
                if rise is not None:config['optical']['rise_time_ns']=rise
                path=folder/(name+'.json');path.write_text(json.dumps(config));root=folder/(name+'.root')
                macro=folder/(name+'.mac')
                macro.write_text(f'/csi/config {path}\n/csi/source Co60\n/csi/seed 281103\n/csi/output {root}\n/run/initialize\n/run/beamOn 8\n')
                subprocess.run([str(binary),str(macro)],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
                file=ROOT.TFile.Open(str(root));files.append(file)
                info=json.loads(str(file.Get('RunInfo').GetString().Data()))
                self.assertTrue(info['completed']);self.assertEqual(info['schema_version'],9)
                self.assertEqual(info['scintillation_time_model']['finite_rise_enabled'],name=='finite')
                self.assertEqual(info['scintillation_time_model']['rise_time_ns'],rise or [0.,0.])
            trees=[f.Get('Events') for f in files];photons=0
            for i in range(8):
                for tree in trees:tree.GetEntry(i);check_ancestry(tree)
                for branch in ('adc','photon_t_ns','avalanche_t_ns','avalanche_q_C'):
                    np.testing.assert_array_equal(np.array(getattr(trees[0],branch)),np.array(getattr(trees[1],branch)))
                photons+=len(trees[2].photon_t_ns)
            self.assertGreater(photons,0,'Finite-rise test must exercise optical transport')
            for f in files:f.Close()
            for bad in ([-1.,0.],[0.],1.):
                base['optical']['rise_time_ns']=bad;path.write_text(json.dumps(base))
                invalid=folder/'invalid.root'
                result=subprocess.run([str(binary),str(path),'Co60','1','1',str(invalid)],capture_output=True)
                self.assertNotEqual(result.returncode,0);self.assertFalse(invalid.exists())

    def test_colored_noise_preserves_transport_and_avalanche_stream(self):
        binary=HERE/'build/csi_sources'
        if not binary.exists():self.skipTest('Geant4 build required')
        with tempfile.TemporaryDirectory(prefix='csi-noise-') as directory:
            folder=Path(directory);config=json.loads((HERE/'configs/csi_tl_initial.json').read_text());files=[]
            for i in range(2):
                if i:config['electronics_noise']={'model':'gaussian_fir','coefficients':[float(np.sqrt(.5))]*2}
                path=folder/f'config{i}.json';path.write_text(json.dumps(config));root=folder/f'noise{i}.root'
                subprocess.run([str(binary),str(path),'Co60','6','281103',str(root)],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
                files.append(ROOT.TFile.Open(str(root)))
            trees=[f.Get('Events') for f in files];different=0;photons=0
            for i in range(6):
                for tree in trees:tree.GetEntry(i)
                for branch in ('photon_t_ns','avalanche_t_ns','avalanche_q_C','signal_mV','charge_by_origin_C'):
                    np.testing.assert_array_equal(np.array(getattr(trees[0],branch)),np.array(getattr(trees[1],branch)))
                different+=not np.array_equal(np.array(trees[0].adc),np.array(trees[1].adc));photons+=len(trees[0].photon_t_ns)
            self.assertGreater(photons,0);self.assertEqual(different,6)
            for f in files:f.Close()

if __name__=='__main__':unittest.main(verbosity=2)
