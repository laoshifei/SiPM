"""Apply the existing 3 us QDC/time/PCA definitions to all frozen bias cohorts.

No weights, gates, labels or mixture peaks are retrained. Distribution shifts
are conditional on recorded events and charge bins, not particle error rates.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import ROOT
from reference import PROJECT, bin_layout
from diagnose_charge import sha
from compare import BLUE, RED, GREEN, panel, graph, label, save

HERE=Path(__file__).resolve().parent
RUNS=(3,7,10,11,12,13,14,18)
GROUPS=('first60','middle20','last20')
METHODS=('qdc','negative_mean_time','pca_projection')


def frozen_scores(data,pca):
    window=2
    q=data['long_charge'][:,window]
    matrix=np.divide(data['block_charge'][:,:pca['blocks']],q[:,None],out=np.full((len(q),pca['blocks']),np.nan),where=q[:,None]>0)
    return {'qdc':np.divide(data['short_charge'][:,2],q,out=np.full(len(q),np.nan),where=q>0),
            'negative_mean_time':-data['mean_time_us'][:,window],
            'pca_projection':((matrix-np.asarray(pca['center']))@np.asarray(pca['vector']))/pca['score_scale']}


def distance(x,y):
    x=np.sort(x);y=np.sort(y);points=np.r_[x,y]
    return float(abs(np.searchsorted(x,points,'right')/len(x)-np.searchsorted(y,points,'right')/len(y)).max())


def selected_blocks(waves, baseline, anchor, blocks):
    y=np.asarray(waves,float)-baseline[:,None]
    c=np.c_[np.zeros(len(y)),np.cumsum(y,axis=1)]
    boundaries=anchor[:,None]-20+20*np.arange(blocks+1)
    if boundaries.min()<128 or boundaries.max()>waves.shape[1]:raise ValueError('Unsupported PCA window')
    return np.diff(np.take_along_axis(c,boundaries,axis=1),axis=1).astype(np.float32)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--resume',action='store_true');args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=args.resume)
    bias=HERE/'results/bias_response_20261028';phase=PROJECT/'analysis/paper_baseline/results/phase2'
    pca_path=phase/'pca_models.json';pca=next(x for x in json.loads(pca_path.read_text()) if x['end_samples']==1500)
    with np.load(bias/'samples.npz') as z:cohort=dict(z)
    prior=json.loads((bias/'bias_response.json').read_text());edges=np.array(prior['q_edges'])
    report={'scope':__doc__,'q_edges':edges.tolist(),
            'selection':'Same capped frozen bias cohorts, including amplitude>=200, hardware bit clear and t10 support; this differs explicitly from the earlier phase2 FOM cohort. No old FOM is recomputed or replaced.',
            'reference':'Run3 first60 for Co; run12 first60 for Am. Compare in six frozen charge bins with at least20 events in BOTH groups. Equal-bin descriptive summaries; bins missing support are counted.',
            'metrics':'Empirical KS distance (no p-value), median shift divided by reference IQR, IQR ratio. No particle purity/efficiency or FOM inferred. Same-source comparison still allows source-geometry/mixture differences.',
            'roles':'All runs previously explored. PCA center, vector and score scale frozen from original first60 run3/12 training.',
            'raw_identity_policy':'For runs without cached PCA blocks, only frozen selected waveforms are read. Raw hashes inherited from bias_response.json are not recomputed; feature hashes are checked here.',
            'inputs_sha256':{str(x.relative_to(PROJECT)):sha(x) for x in [pca_path,bias/'samples.npz',bias/'bias_response.json']},
            'code_sha256':sha(Path(__file__)),'groups':{}}
    scores={};arrays={}
    for run in RUNS:
        path=phase/f'run{run:04d}_features.npz'
        with np.load(path) as z:data={k:z[k] for k in ['long_charge','short_charge','mean_time_us','anchor_sample']+(['block_charge'] if 'block_charge' in z.files else [])}
        raw=None
        if 'block_charge' not in data:
            raw_path=PROJECT/f'data/DT5730_DPP_PHA/DAQ/run_{run}/UNFILTERED/Data_run_{run}.BIN'
            _,dtype,count=bin_layout(raw_path);raw=np.memmap(raw_path,dtype=dtype,offset=2,mode='r',shape=(count,))
            fp=PROJECT/f'analysis/paper_baseline/results/pha_run{run:04d}_features.npz'
            with np.load(fp) as z:baseline=z['baseline']
            report['inputs_sha256'][str(fp.relative_to(PROJECT))]=sha(fp)
            report['inputs_sha256'][str(raw_path.relative_to(PROJECT))]=prior['inputs_sha256'][str(raw_path.relative_to(PROJECT))]
        report['inputs_sha256'][str(path.relative_to(PROJECT))]=sha(path)
        for group in GROUPS:
            key=f'run{run}_{group}';ids=cohort[key+'_ids']
            checkpoint=args.out/(key+'.npz')
            if checkpoint.exists():
                with np.load(checkpoint) as z:scores[key]={m:z[m] for m in METHODS}
            else:
                sub={k:v[ids] for k,v in data.items()}
                if raw is not None:sub['block_charge']=selected_blocks(raw['data'][ids],baseline[ids],sub['anchor_sample'],pca['blocks'])
                scores[key]=frozen_scores(sub,pca)
                np.savez_compressed(checkpoint,**scores[key])
            arrays.update({key+'_'+m:scores[key][m] for m in METHODS})
            print(key,len(ids),flush=True)
    for run in RUNS:
        ref=f'run{3 if run in (3,7) else 12}_first60'
        for group in GROUPS:
            key=f'run{run}_{group}';q=cohort[key+'_q'];rq=cohort[ref+'_q'];rows={}
            for method in METHODS:
                cells=[]
                for b,(lo,hi) in enumerate(zip(edges[:-1],edges[1:])):
                    use=(q>=lo)&((q<hi)|((b==5)&(q==hi)))
                    ruse=(rq>=lo)&((rq<hi)|((b==5)&(rq==hi)))
                    x=scores[key][method][use];y=scores[ref][method][ruse]
                    x=x[np.isfinite(x)];y=y[np.isfinite(y)]
                    row={'bin':b,'n':len(x),'reference_n':len(y)}
                    if min(len(x),len(y))>=20:
                        a=np.quantile(x,[.25,.5,.75]);r=np.quantile(y,[.25,.5,.75]);iqr=r[2]-r[0]
                        row.update(ks=distance(x,y),median_shift_reference_iqr=float((a[1]-r[1])/iqr),iqr_ratio=float((a[2]-a[0])/iqr))
                    cells.append(row)
                valid=[x for x in cells if 'ks' in x]
                rows[method]={'bins':cells,'supported_bins':len(valid),
                              'equal_bin_mean_ks':float(np.mean([r['ks'] for r in valid])) if valid else None,
                              'equal_bin_mean_median_shift_reference_iqr':float(np.mean([r['median_shift_reference_iqr'] for r in valid])) if valid else None,
                              'equal_bin_mean_iqr_ratio':float(np.mean([r['iqr_ratio'] for r in valid])) if valid else None}
            report['groups'][key]={'n':len(q),'reference':ref,'methods':rows}
    np.savez_compressed(args.out/'scores.npz',**arrays)
    (args.out/'transfer.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    ROOT.gROOT.SetBatch(True);ROOT.gStyle.SetOptStat(0)
    c=ROOT.TCanvas('frozen_psd','Frozen PSD score transfer',1500,600);c.Divide(3,1,.003,.003)
    labels=['QDC ratio','Negative mean time','Frozen PCA projection']
    for i,method in enumerate(METHODS):
        panel(c,i+1,labels[i]+';Run ID;Equal-bin mean KS distance',2.,0.,19.,.65)
        for group,color,title in zip(GROUPS,[BLUE,RED,GREEN],['First60','Middle20','Last20']):
            vals=[report['groups'][f'run{r}_{group}']['methods'][method]['equal_bin_mean_ks'] for r in RUNS]
            valid=[j for j,v in enumerate(vals) if v is not None]
            xx=np.ascontiguousarray(np.array(RUNS)[valid],dtype=float);yy=np.ascontiguousarray(np.array(vals)[valid],dtype=float)
            g=ROOT.TGraph(len(xx),xx,yy);g.SetMarkerStyle(20+GROUPS.index(group));g.SetMarkerColor(color);g.SetMarkerSize(1.1);g.Draw('P SAME')
            from compare import KEEP
            KEEP.append(g)
            label(title,.20,.85-.06*GROUPS.index(group),color,.031)
        label('Co vs run3; Am vs run12; no retraining',.18,.68,size=.025)
        label('Different sample sizes; no significance claim',.18,.63,size=.024)
    save(c,args.out,'frozen_psd_transfer')
    print({key:{m:round(v['equal_bin_mean_ks'],4) if v['equal_bin_mean_ks'] is not None else None for m,v in row['methods'].items()} for key,row in report['groups'].items() if key.endswith('first60')})


if __name__=='__main__':main()
