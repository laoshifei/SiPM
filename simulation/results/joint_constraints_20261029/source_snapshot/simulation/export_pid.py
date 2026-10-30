"""Export per-event PID regions with explicit measured-vs-simulated truth status."""
import csv
from pathlib import Path
import numpy as np
from prepare_pid import OUT,PROJECT,STATES,read
from analyze_pid import combined,METHODS
from extract_pid import load,NAMES
from publication_stats import sha,save


def main():
    protocol=read(OUT/'protocol.json');thresholds=read(OUT/'thresholds.json')
    cuts=np.array([thresholds['methods'][m]['threshold'] for m in METHODS])
    exported={}
    for kind in ['measured','MC']:
        path=OUT/('measured_pid_regions.csv' if kind=='measured' else 'MC_pid_events.csv');rows=[]
        if kind=='measured':
            for run in [3,12,18]:
                for segment in ['first60','middle20','last20']:
                    d=load(OUT/'arrays'/f'measured_run{run}_{segment}.npz')
                    for i,event in enumerate(d['ids']):
                        r={'run':run,'segment':segment,'event_index':int(event),'Q_ADC_sample':float(d['q'][i]),'particle_truth':'unavailable'}
                        for j,m in enumerate(METHODS):r[m]=float(d['scores'][i,j]);r[m+'_region']='low_score' if d['scores'][i,j]<=cuts[j] else 'high_score'
                        rows.append(r)
        else:
            for state in STATES:
                d=combined(state,protocol)
                for i in np.flatnonzero(d['common']):
                    r={'readout_state':state,'source':'Co60' if d['source'][i]==0 else 'Am241',
                        'mother_event_id':int(d['ids'][i]),'split':'training' if d['split'][i]==0 else 'test',
                        'Q_ADC_sample':float(d['q'][i]),'particle_truth':NAMES[d['truth'][i]]}
                    for j,m in enumerate(METHODS):r[m]=float(d['scores'][i,j]);r[m+'_region']='low_score' if d['scores'][i,j]<=cuts[j] else 'high_score'
                    rows.append(r)
        with path.open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        exported[kind]={'path':str(path.relative_to(PROJECT)),'rows':len(rows),'sha256':sha(path)}
    point=PROJECT/'analysis/paper_baseline/results/publication_20261028/point.json'
    classifier={'scope':'Fixed conditional PID-score regions, not measured particle truth or universal efficiency.',
      'calibration_state':'common run12 response','thresholds':{m:thresholds['methods'][m]['threshold'] for m in METHODS},
      'low_score_definition':'score<=threshold; alpha-like under the simulated model, not a ground-truth label for measured events.',
      'q_edges':protocol['q_edges'],'amplitude_min_ADC':200,'t10_min_sample':384,
      'sample_period_ns':2,'baseline_samples':128,'anchor':'floor(t10)+1',
      'short_offsets_samples':[-20,250],'long_offsets_samples':[-20,1500],
      'PCA':read(point)['windows']['3000']['pca'],
      'required_quality':'Follow the existing primary pretrigger/baseline and complete-window checks; event regions are conditional on the frozen selection.',
      'threshold_record_sha256':sha(OUT/'thresholds.json'),'PCA_record_sha256':sha(point),
      'code_entry':'simulation/extract_pid.py:score_waves'}
    save(OUT/'pid_classifier.json',classifier)
    save(OUT/'exports.json',{'scope':__doc__,'code_sha256':sha(Path(__file__)),'tables':exported,
        'classifier_sha256':sha(OUT/'pid_classifier.json'),
        'limits':'Three MC states repeat the same mothers; measured low/high score regions are predictions, never experimental truth labels.'})
    print({k:v['rows'] for k,v in exported.items()},flush=True)


if __name__=='__main__':main()
