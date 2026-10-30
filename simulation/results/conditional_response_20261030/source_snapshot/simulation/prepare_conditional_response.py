"""Freeze conditional-PID diagnostics and native afterpulse/recovery contrasts.

No optical retuning, score smearing or new independent transport histories.
"""
import copy
from pathlib import Path
import numpy as np
from reference import PROJECT, wave_features
from prepare_pid import read
from extract_pid import load
from publication_stats import save, sha

OUT = PROJECT/'simulation/results/conditional_response_20261030'
PARENT = PROJECT/'simulation/results/joint_constraints_20261029'
DATA = PROJECT/'simulation/results/signal_fluctuations_20261028'
WIDTH = PROJECT/'simulation/results/pid_bandwidth_20261029'
BUILD = PROJECT/'simulation/build/conditional_response_20261030'
SEGMENTS = ['first60', 'middle20', 'last20']
SOURCES = [('Co60', 3), ('Am241', 12)]
VARIANTS = {
    'identity': {},
    'no_signal_residual': {},
    'ap05_60ns': {'afterpulse_probability': .05, 'afterpulse_delay_ns': 60.},
    'ap05_300ns': {'afterpulse_probability': .05, 'afterpulse_delay_ns': 300.},
    'ap15_300ns': {'afterpulse_probability': .15, 'afterpulse_delay_ns': 300.},
    'recovery60ns': {'recovery_ns': 60.},
}


def relative(path):
    return str(path.relative_to(PROJECT))


def main():
    OUT.mkdir(exist_ok=False)
    for name in ['configs', 'roots', 'logs', 'arrays', 'extraction', 'figures', 'source_snapshot']:
        (OUT/name).mkdir()
    p = read(PARENT/'validation/protocol.json')
    assert sha(PARENT/'record.json') == 'c45c438386fe91e158d7d85fbc6d191a7604c4eabb9e539a8b5fa7e2cabf3b10'
    edges = np.asarray(p['q_edges']); cuts = {}; entries = {}; inputs = {}
    for source, run in SOURCES:
        key = f'run{run}_{source}'; parent = p['entries'][key]
        original = load(PARENT/'validation/arrays'/f'{key}.npz')
        inputs[relative(PARENT/'validation/arrays'/f'{key}.npz')] = sha(PARENT/'validation/arrays'/f'{key}.npz')
        nraw = int(load(PROJECT/f'simulation/results/pid_20261029/arrays/measured_run{run}_first60.npz')['original_count'])
        borders = [0, int(.6*nraw), int(.8*nraw), nraw]
        for si, segment in enumerate(SEGMENTS):
            group = f'run{run}_{segment}'; dp = DATA/f'{group}.npz'; sp = WIDTH/'arrays'/f'measured_{group}.npz'
            d = load(dp); s = load(sp); np.testing.assert_array_equal(d['ids'], s['ids'])
            u = s['main_band']; f = wave_features(d['waves'][u])
            np.testing.assert_allclose(f['charge_long'], s['q'][u], atol=1e-7, rtol=1e-12)
            np.savez_compressed(OUT/'arrays'/f'data_{group}.npz', ids=s['ids'][u], q=s['q'][u],
                scores=s['scores'][u], baseline_rms=f['baseline_rms'], baseline=f['baseline'],
                baseline_delta=s['baseline_delta'][u], span=borders[si:si+2], n_raw=nraw)
            for path in [dp, sp]: inputs[relative(path)] = sha(path)
            if segment == 'first60':
                bins=np.clip(np.searchsorted(edges,s['q'][u],'right')-1,0,5); row={}
                for b in p['support_bins'][source]:
                    q=s['q'][u][bins==b]; rms=f['baseline_rms'][bins==b]
                    qs=np.quantile(q,[.25,.5,.75]); rs=float(np.median(rms))
                    cell=2*(q>=qs[1]).astype(int)+(rms>=rs)
                    row[str(b)]={'q_quartiles':qs.tolist(),'rms_median':rs,
                        'q2_target':[float(np.mean(q<qs[1])),float(np.mean(q>=qs[1]))],
                        'q4_target':(np.bincount(np.searchsorted(qs,q,'right'),minlength=4)/len(q)).tolist(),
                        'q2r2_target':(np.bincount(cell,minlength=4)/len(q)).tolist(), 'training_n':len(q)}
                cuts[source]=row
        root = PROJECT/parent['path']; parent_hash=sha(root)
        for variant, changes in VARIANTS.items():
            c=copy.deepcopy(read(PROJECT/parent['config']))
            if variant != 'identity':
                c['signal_dependent_noise']['variance_per_signal_adc']=0.
                c['sipm']['sensor'].update(changes)
                c['status']='Conditional response mechanism contrast; not a fitted replacement or measured sensor parameters'
                c['conditional_response']={'variant':variant,'parent_record_sha256':sha(PARENT/'record.json'),
                    'signal_residual_disabled':True,'gain_compensation':False,
                    'parameter_role':'Predeclared sensitivity/stress values, not measured S13361 values or literature constants'}
            name=f'{key}__{variant}'; cp=OUT/'configs'/f'{name}.json'; rp=OUT/'roots'/f'{name}.root'
            save(cp,c)
            entries[name]={'key':name,'source':source,'run':run,'variant':variant,'events':parent['events'],
                'seed':parent['seed'],'parent':parent['path'],'parent_sha256':parent_hash,
                'config':relative(cp),'config_sha256':sha(cp),'path':relative(rp),
                'command':[str(BUILD/'replay_response'),str(root),str(cp),str(rp)]}
    protocol={'scope':__doc__,'code_sha256':sha(Path(__file__)),'parent_record_sha256':sha(PARENT/'record.json'),
        'literature_note':'jupyter/codex_0001_csi_pid_bandwidth_literature_web.ipynb',
        'literature_note_sha256':sha(PROJECT/'jupyter/codex_0001_csi_pid_bandwidth_literature_web.ipynb'),
        'q_edges':edges.tolist(),'support_bins':p['support_bins'],'conditional_cuts':cuts,
        'entries':entries,'inputs_sha256':inputs,'binary_sha256':sha(BUILD/'replay_response'),
        'source_sha256':{relative(f):sha(f) for f in [Path(__file__),PROJECT/'simulation/replay_response.cc',
            PROJECT/'simulation/vendor/SiPMResponse.cc',PROJECT/'simulation/src/SignalDependentNoise.cc']},
        'conditional_modes':['original','q2','q4','q2r2'],
        'conditioning':'Cuts and target masses from first60 only, separately inside each original Q bin. Match to first60 Q sub-bin distribution or joint Q/baseline-RMS cells. Equal weight original supported bins. No event deletion, tail extrapolation or weight clipping. Report ESS/empty cells; sparse state matching not causal correction.',
        'uncertainty':'512 paired original-mother bootstrap, data original-index block64/128. Conditioning target/cuts, old fit and mechanism values fixed; later segments previously explored. Empty target cells invalidate a draw; >=128 valid draws required for interval.',
        'native_contrasts':'Identity with original residual; all mechanism contrasts disable that residual to avoid double counting. AP probability .05 at60/300ns, .15 at300ns stress; recovery60ns vs nominal29ns. Constant gain/PDE/readout/geometry/optics. Do not promote a best case or infer actual AP rates.',
        'spatial_crosstalk':'Remains zero: flattened cell map does not preserve 2D adjacency.',
        'MC_scope':'The previous 32000 validation mothers are now reused for diagnostic exploration. 192000 replay entries are not new independent statistics; no new transport and no new experiment.',
        'new_independent_mothers':0,'unique_reused_mothers':32000,'replay_entries':192000,
        'selection':'Report full reselected own-Q results and paired original-ID/Q results, with own-anchor and fixed-parent-anchor controls. Experimental IDs, original Q, PCA and main-band0.4 diagnostic remain frozen; source labels are not particle truth.'}
    save(OUT/'protocol.json',protocol)
    print('Frozen 12 native response contrasts and first60 conditional targets.',flush=True)


if __name__=='__main__':main()
