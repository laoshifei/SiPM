"""Six publication-draft figures from recorded numerical results, using ROOT."""
import json
from pathlib import Path
import sys
import numpy as np
import ROOT
import publication_stats as p
sys.path.insert(0,str(p.SIM))
from compare import panel,graph,label,KEEP,BLUE,RED,GREEN
from reference import TIME_NS

COLORS=(BLUE,RED,GREEN)
NAMES=('QDC','Negative mean time','PCA')


def dots(x,y,color,style=20,size=1.):
    x=np.ascontiguousarray(x,float);y=np.ascontiguousarray(y,float)
    g=ROOT.TGraph(len(x),x,y);g.SetMarkerColor(color);g.SetMarkerStyle(style);g.SetMarkerSize(size)
    g.Draw('P SAME');KEEP.append(g)


def whiskers(x,y,lo,hi,color,style=20):
    # A percentile interval need not contain the original point estimate.
    for xx,a,b in zip(x,lo,hi):
        line=ROOT.TLine(float(xx),float(a),float(xx),float(b));line.SetLineColor(color);line.SetLineWidth(2);line.Draw();KEEP.append(line)
    dots(x,y,color,style)


def finish(c,folder,name,record):
    c.Modified();c.Update()
    for ext in ('png','svg','pdf'):c.SaveAs(str(folder/(name+'.'+ext)))
    record.append(name)


def run(out=p.OUT):
    folder=out/'figures';folder.mkdir(exist_ok=True);record=[]
    ROOT.gROOT.SetBatch(True);ROOT.gStyle.SetOptStat(0);ROOT.gStyle.SetTitleFontSize(.037)
    ROOT.gStyle.SetTitleSize(.038,'XYZ');ROOT.gStyle.SetLabelSize(.034,'XYZ')
    point=json.loads((out/'point.json').read_text());stats=json.loads((out/'paired_uncertainty.json').read_text())
    transfer=json.loads((out/'transfer_uncertainty.json').read_text());matched=json.loads((out/'transfer_matched.json').read_text())
    covariance=json.loads((out/'covariance_uncertainty.json').read_text());mc=json.loads((out/'mc_uncertainty.json').read_text())
    data={source:p.load(out/'mc'/(source+'_measured.npz')) for source in ('Co60','Am241')}
    d=p.training(out);s=p.load(out/'training_scores.npz')['3000']

    c=ROOT.TCanvas('pub_data','Recorded training waveforms',1400,620);c.Divide(2,1,.003,.003)
    ymax=max(v['shapes'].mean(axis=0).max() for v in data.values())*1200
    panel(c,1,'(a) Same frozen ADC-charge interval;Time from t_{10} (#mus);Area-normalized signal (#mus^{-1})',-.15,0,3.6,float(ymax))
    for i,(source,title) in enumerate((('Co60','^{60}Co / run3: n=2450'),('Am241','^{241}Am / run12: n=1547'))):
        graph(TIME_NS/1000,data[source]['shapes'].mean(axis=0)*1000,COLORS[i]);label(title,.43,.85-i*.055,COLORS[i],.030)
    label('Training segments; source labels are not event truth',.36,.69,size=.024)
    panel(c,2,'(b) Primary training scores;Reference Q (10^{3} ADC samples);Q_{short} / Q_{3#mus}',195,.24,425,.60)
    for i,run in enumerate((3,12)):
        ix=d['run']==run
        dots(data['Co60' if run==3 else 'Am241']['q']/1000,s[ix,0],COLORS[i],20,.50)
    label('Fixed six Q bins; no particle gate',.22,.84,size=.029)
    label('All methods use these same 3997 training events',.22,.78,size=.026)
    finish(c,folder,'fig01_primary_data',record)

    c=ROOT.TCanvas('pub_psd','Method and window comparisons',1400,620);c.Divide(2,1,.003,.003)
    main=stats['block_sizes']['64'];windows=('1000','2000','3000','4000');x=np.arange(1,5,dtype=float)
    panel(c,1,'(a) Median over six fixed Q bins;Integration endpoint (#mus);Shape-core FOM (FWHM)',.7,.9,4.3,3.2)
    for i,m in enumerate(p.METHODS):
        y=np.array([main['windows'][w]['point_median_fom'][i] for w in windows])
        intervals=np.array([main['windows'][w]['median_fom'][m]['interval95'] for w in windows])
        graph(x,y,COLORS[i]);whiskers(x+(i-1)*.025,y,*intervals.T,COLORS[i],20+i)
        label(NAMES[i],.20,.86-i*.055,COLORS[i],.03)
    label('Paired blocks 64; PCA + nuisance refit in 256 draws',.18,.19,size=.025)
    panel(c,2,'(b) Paired differences at 3 #mus;Comparison index;Difference of median FOM',.5,-.18,3.5,.4)
    keys=['negative_mean_time minus qdc','pca_projection minus qdc','pca_projection minus negative_mean_time']
    graph(np.array([.5,3.5]),np.zeros(2),ROOT.kGray+1,2)
    for i,(key,title) in enumerate(zip(keys,['1: time - QDC','2: PCA - QDC','3: PCA - time'])):
        row=main['windows']['3000']['paired_differences'][key]
        whiskers([i+1],[row['point']],[row['interval95'][0]],[row['interval95'][1]],COLORS[i])
        label(title,.52,.86-i*.055,COLORS[i],.03)
    label('Exploratory conditional 95% intervals',.20,.19,size=.029)
    finish(c,folder,'fig02_methods_windows',record)

    c=ROOT.TCanvas('pub_transfer','Charge-conditioned transfer',1450,640);c.Divide(2,1,.003,.003)
    keys=('run12_middle20','run12_last20','run18_first60');xs=np.arange(1,4,dtype=float)
    panel(c,1,'(a) Check / run12 first60;1:run12 middle  2:run12 late  3:run18 first;Whole-score IQR ratio',.5,.7,3.5,1.6)
    graph(np.array([.5,3.5]),np.ones(2),ROOT.kGray+1,2)
    for i,m in enumerate(p.METHODS):
        y=[transfer['groups'][k]['point'][2][i] for k in keys]
        lo=[transfer['groups'][k]['interval95'][0][2][i] for k in keys];hi=[transfer['groups'][k]['interval95'][1][2][i] for k in keys]
        whiskers(xs+(i-1)*.1,y,lo,hi,COLORS[i],20+i);label(NAMES[i],.20,.86-i*.05,COLORS[i],.028)
    label('Equal Q-bin weights; PCA training uncertainty included',.18,.21,size=.024)
    panel(c,2,'(b) Equal-N context for QDC;1:run12 middle  2:run12 late  3:run18 first;Mean within-bin KS distance',.5,.04,3.5,.26)
    for i,(prefix,title,color) in enumerate([('observed','Check / reference',BLUE),('reference_IID','Reference-only IID benchmark',ROOT.kGray+2)]):
        vals=[matched['matched'][k] for k in keys]
        range_key='observed_subsampling_range95' if prefix=='observed' else 'reference_IID_range95'
        whiskers(xs+(i-.5)*.12,[v[prefix+'_median'][0][0] for v in vals],
                 [v[range_key][0][0][0] for v in vals],[v[range_key][1][0][0] for v in vals],color,20+i)
        label(title,.46,.86-i*.055,color,.025)
    label('256 repeats; n<=200/bin; ranges, not null-test CIs',.18,.21,size=.025)
    finish(c,folder,'fig03_cross_run',record)

    c=ROOT.TCanvas('pub_covariance','Local residual covariance',1400,620);c.Divide(2,1,.003,.003)
    v=np.array(covariance['equal_q_point']);interval=np.array(covariance['block_sizes']['64']['equal_q_interval95'])
    x=np.array([200,700,1400,2400],float)
    for iv,title in enumerate(('(a) Common Q / all retained shapes','(b) Common Q / 0.30 <= ratio <0.35')):
        panel(c,iv+1,title+';400 ns window centre from t_{10} (ns);Excess covariance / local mean (ADC)',0,.08,2700,.38)
        for ir,(run,color) in enumerate(((12,RED),(18,BLUE))):
            y=v[ir,iv,0];graph(x,y,color);whiskers(x+(ir-.5)*18,y,interval[0,ir,iv,0],interval[1,ir,iv,0],color,20+ir)
            label(f'Run{run} first60',.51,.86-ir*.06,color,.030)
        label('50 ns filter; equal six-Q-bin weights',.20,.24,size=.027)
        label('Cell means refit; 512 block draws; not avalanche gain',.20,.19,size=.025)
    finish(c,folder,'fig04_local_covariance',record)

    c=ROOT.TCanvas('pub_mc','Native response test',1450,1080);c.Divide(2,2,.003,.003)
    for col,source in enumerate(('Co60','Am241')):
        measured=data[source]['shapes'].mean(axis=0)
        modeldata={name:p.load(out/'mc'/(source+'_'+name+'.npz')) for name in ('control','native_best','prompt_plus_ap15')}
        shape_max=max(measured.max(),modeldata['native_best']['shapes'].mean(axis=0).max())*1200
        panel(c,col+1,source+' / mean shape;Time from t_{10} (#mus);Area-normalized signal (#mus^{-1})',-.15,0,3.6,float(shape_max))
        graph(TIME_NS/1000,measured*1000,BLUE)
        graph(TIME_NS/1000,modeldata['native_best']['shapes'].mean(axis=0)*1000,RED,2)
        label('Measured training',.48,.85,BLUE,.030);label('Native prompt fraction 0.10',.48,.79,RED,.028)
        label('Conditional geometry/circuit; no refit here',.20,.19,size=.026)
        ymax=.48 if col==0 else 1.05
        panel(c,col+3,source+' / filter-dependent fluctuations;Box-filter width (ns);Excess variance / mean (ADC)',30,-.02,225,ymax)
        xx=np.array([50.,102.,202.]);meas=np.array(mc['sources'][source]['measured_F']);mi=np.array(mc['results'][source]['measured_F_interval95'])
        graph(xx,meas,BLUE);whiskers(xx-3,meas,mi[0],mi[1],BLUE)
        for i,(name,title,color) in enumerate([('control','Original response',ROOT.kGray+2),('native_best','Prompt fraction0.10',RED),('prompt_plus_ap15','Prompt + AP0.15',GREEN)]):
            row=mc['results'][source][name];y=np.array(row['point'][:3]);ci=np.array(row['mc_only_interval95'])[:,:3]
            graph(xx,y,color,2);whiskers(xx+i*2,y,ci[0],ci[1],color,21+i)
            label(title,.20,.85-i*.05,color,.028)
        label('Measured training',.20,.70,BLUE,.028)
        label('Data: block 95%; MC: paired-parent 95%',.20,.63,size=.026)
        label('Same parent histories across MC candidates',.20,.58,size=.024)
    finish(c,folder,'fig05_mc_shape_fluctuations',record)

    c=ROOT.TCanvas('pub_charge','Common-charge distributions',1400,620);c.Divide(2,1,.003,.003)
    edges=np.linspace(*[json.loads((out/'protocol.json').read_text())['q_edges'][i]/1000 for i in (0,-1)],19)
    centers=(edges[1:]+edges[:-1])/2
    for i,source in enumerate(('Co60','Am241')):
        panel(c,i+1,source+' / common-Q shape;Reference Q (10^{3} ADC samples);Fraction per bin',edges[0],0,edges[-1],.25)
        values=[('Measured',data[source]['q'],BLUE),('Original MC',p.load(out/'mc'/(source+'_control.npz'))['q'],ROOT.kGray+2),
                ('Prompt MC',p.load(out/'mc'/(source+'_native_best.npz'))['q'],RED)]
        for j,(title,q,color) in enumerate(values):
            hist=np.histogram(q/1000,edges)[0]/len(q);graph(centers,hist,color,1 if j==0 else 2)
            label(title+f' (n={len(q)})',.47,.85-j*.055,color,.029)
    finish(c,folder,'fig06_mc_charge',record)
    p.save(folder/'manifest.json',{'figure_stems':record,'code_sha256':p.sha(Path(__file__)),
      'inputs_sha256':{name:p.sha(out/name) for name in ['point.json','paired_uncertainty.json','transfer_matched.json','transfer_uncertainty.json','covariance_uncertainty.json','mc_uncertainty.json']},
      'captions':'See MANUSCRIPT_DRAFT.md; all source identities remain distinct from event particle truth.'})


if __name__=='__main__':run()
