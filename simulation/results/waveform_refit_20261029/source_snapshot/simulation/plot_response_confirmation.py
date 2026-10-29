"""Plot recorded means/local covariance against confirmed native candidates."""
import argparse
import json
from pathlib import Path
import numpy as np
import ROOT
from reference import PROJECT, TIME_NS
from diagnose_charge import sha
from compare import BLUE, RED, GREEN, panel, graph, label, save


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--screen', type=Path, required=True)
    p.add_argument('--confirmation', type=Path, required=True)
    args = p.parse_args()
    paths = [args.screen/'screen.json', args.screen/'curves.npz',
             args.confirmation/'confirmation.json', args.confirmation/'curves.npz',
             PROJECT/'simulation/results/bias_response_20261028/samples.npz',
             PROJECT/'simulation/results/residual_covariance_20261028/covariance.json']
    screen = json.loads(paths[0].read_text())
    with np.load(paths[1]) as z:
        old_curves = dict(z)
    confirmation = json.loads(paths[2].read_text())
    with np.load(paths[3]) as z:
        curves = dict(z)
    with np.load(paths[4]) as z:
        measured = {k: z[k] for k in ('run3_first60_shape', 'run12_first60_shape')}
    covariance = json.loads(paths[5].read_text())
    ROOT.gROOT.SetBatch(True)
    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetTitleFontSize(.037)
    c = ROOT.TCanvas('native_confirmation', 'Native response confirmation', 1450, 1050)
    c.Divide(2, 2, .003, .003)
    records = {}
    for col, (source, run) in enumerate((('Co60', 3), ('Am241', 12))):
        rows = [screen['candidates']['prompt_0.000']['sources'][source],
                confirmation['results']['native_best']['sources'][source],
                confirmation['results']['prompt_plus_ap15']['sources'][source]]
        shapes = [measured[f'run{run}_first60_shape'], old_curves[source+'_prompt_0.000_shape'],
                  curves[source+'_native_best_shape'], curves[source+'_prompt_plus_ap15_shape']]
        colors = [ROOT.kBlack, BLUE, RED, GREEN]
        titles = ['Measured first60', 'Original response', 'Prompt fraction 0.10', 'Prompt 0.10 + AP 0.15']
        panel(c, col+1, source+' / normalized mean;Time from t_{10} (#mus);Normalized signal (#mus^{-1})',
              -.1, -.01, 3.5, max(s.max() for s in shapes)*1250)
        for i, (shape, color, title) in enumerate(zip(shapes, colors, titles)):
            graph(TIME_NS/1000, shape*1000, color, 1 if i == 0 else i+1)
            label(title, .53, .84-i*.055, color, .029)
        label('Same charge interval; normalized for shape only', .20, .20, size=.026)
        observed = covariance['groups'][f'run{run}_first60']['views']['all']
        values = [np.array(observed['across_event'])[0, :, 0]]
        values += [np.array(r['covariance']['across_event'])[0, :, 0] for r in rows]
        panel(c, col+3, source+' / 50 ns local residual;Window centre from t_{10} (ns);Excess covariance / local mean (ADC)',
              0, 0, 2800, max(v.max() for v in values)*1.3)
        for i, (v, color) in enumerate(zip(values, colors)):
            graph(np.array([200., 700., 1400., 2400.]), v, color, 1 if i == 0 else i+1)
        label('400 ns windows; reproducible mean structure removed', .18, .86, size=.026)
        label('Charge/shape cells with >=10 events; counts differ', .18, .80, size=.026)
        records[source] = {'measured_n': observed['n_used'],
            'original_prompt_combined_n': [r['covariance']['n_used'] for r in rows],
            'covariance50_measured_original_prompt_combined': [v.tolist() for v in values]}
    save(c, args.confirmation, 'native_response_comparison')
    record = {'scope': __doc__, 'analysis_changed': False, 'code_sha256': sha(Path(__file__)),
              'inputs_sha256': {str(p.resolve().relative_to(PROJECT)): sha(p) for p in paths}, 'curves': records}
    (args.confirmation/'plot_record.json').write_text(json.dumps(record, indent=2)+'\n')


if __name__ == '__main__':
    main()
