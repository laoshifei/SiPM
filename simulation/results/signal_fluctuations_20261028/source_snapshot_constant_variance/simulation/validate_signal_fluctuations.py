"""Frozen-calibration validation of native ADC, including previously explored checks."""
import json
from pathlib import Path
import numpy as np
import ROOT
from calibrate_signal_fluctuations import OUT,BASE,PARENT,diagnostic,save
from reference import PROJECT,aligned_shapes,TIME_NS,wave_features
from diagnose_charge import sha
from diagnose_fluctuations import fluctuation_metrics
from screen_response_candidates import load,info_of,ks
from compare import mc_wave_features,BLUE,RED,GREEN,KEEP,panel,graph,label,save as save_canvas
from validate import check
import publication_stats as p
from publication_mc_uncertainty import weighted_ks

REGION=(TIME_NS>=-100)&(TIME_NS<=3500)


def arrays(path):
    with np.load(path) as z:return dict(z)


def weights(q,edges,support):
    cells=np.minimum(np.searchsorted(edges,q,side='right')-1,5)
    count=np.bincount(cells,minlength=6);w=np.zeros(len(q))
    for b in support:w[cells==b]=1/(len(support)*count[b])
    return w


def error(shape,reference):
    return float(np.sqrt(np.mean((shape[REGION]-reference[REGION])**2))/reference.max())


def bootstrap(model,data,run,segment,nparent,draws=512):
    nraw={3:4776,12:7516,18:156966}[run]
    lo,hi=[(0,int(.6*nraw)),(int(.6*nraw),int(.8*nraw)),(int(.8*nraw),nraw)][segment]
    relative=[];nrmse=[];means=[]
    for i in range(draws):
        rng=np.random.default_rng(np.random.SeedSequence([281028321,run,segment,i]))
        parent=np.bincount(rng.integers(nparent,size=nparent),minlength=nparent)
        wm=parent[model['ids']].astype(float);wm/=wm.sum()
        wd=p.block_counts(data['ids'],(lo,hi),64,rng).astype(float);wd/=wd.sum()
        fm=wm@model['metrics'][:,:,3];fd=wd@data['metrics'][:,:,3]
        relative.append(fm/fd-1);means.append(fm)
        nrmse.append(error(wm@model['shapes'],wd@data['shapes']))
    return {'relative_interval95':np.quantile(relative,[.025,.975],axis=0).tolist(),
            'MC_F_interval95':np.quantile(means,[.025,.975],axis=0).tolist(),
            'shape_interval95':np.quantile(nrmse,[.025,.975]).tolist(),
            'draws':draws},np.array(relative)


def validate():
    fit=json.loads((OUT/'fit.json').read_text());frozen=json.loads((OUT/'native_run.json').read_text())
    assert sha(OUT/'fit.json')==frozen['fit_sha256']
    design=json.loads((OUT/'design.json').read_text());edges=np.array(design['q_edges'])
    report={'scope':__doc__,'code_sha256':sha(Path(__file__)),
      'fit_sha256':sha(OUT/'fit.json'),'native_run_sha256':sha(OUT/'native_run.json'),
      'selection':'Recompute quality/amplitude>=200/common Q/t10>=384 on native ADC. Measured IDs are unchanged publication IDs. Comparisons do not rescale individual amplitudes or charge spectra.',
      'uncertainty':'512 original-parent multinomial resamples and measured original-index blocks64,conditional on the already fitted model. Does not propagate residual-spectrum calibration,RC selection,unrecorded apparatus or all nonstationarity.',
      'role':'Empirical per-acquisition calibration. Middle/last and run18 were explored before but excluded from this fit. No independent new transport and no physical cause inferred.',
      'q_edges':edges.tolist(),'sources':{}}
    for source,name,run in [('Co60','co60',3),('Am241','am241',12)]:
        cfg=OUT/f'{source}_calibrated.json';assert sha(cfg)==frozen['config_sha256'][source]
        path=OUT/name/'events.root';info=info_of(path)
        assert info['config']==json.loads(cfg.read_text())
        nparent=info['events_requested'];assert info['response_replay']['equal_original_avalanche_events']==nparent
        validity=check(path);assert validity['passed'];save(OUT/name/'validation.json',validity)
        raw=load(path)['adc'];f,good=mc_wave_features(raw)
        selected=good&(f['amplitude']>=200)
        common=selected&(f['charge_long']>=edges[0])&(f['charge_long']<=edges[-1])&(f['t10_sample']>=384)
        ids=np.flatnonzero(common);waves=raw[ids];features={k:v[ids] for k,v in f.items()}
        shapes,kept=aligned_shapes(waves,features);np.testing.assert_array_equal(kept,np.arange(len(ids)))
        metrics=fluctuation_metrics(waves,features['baseline'],features['t10_sample'])
        extra=diagnostic(waves,features,edges)
        model={'ids':ids,'waves':waves,'q':features['charge_long'],'baseline':features['baseline'],
              't10':features['t10_sample'],'shapes':shapes,'metrics':metrics,'power':extra['power'],
              'selected_q':f['charge_long'][selected]}
        np.savez_compressed(OUT/f'{source}_native.npz',**model)
        row={'root_sha256':sha(path),'nparent':nparent,'common_n':len(ids),'selected_n':int(selected.sum()),
             'F':metrics[:,:,3].mean(axis=0).tolist(),'config_sha256':sha(cfg),'comparisons':{},
             'signal_dependent_noise':info['signal_dependent_noise'],'charge_balance':validity,
             'avalanche_events_identical_to_parent':nparent}
        old=arrays(BASE/'mc'/f'{source}_native_best.npz')
        oldbase=arrays(BASE/'mc'/f'{source}_control.npz')
        groups=[(run,s,label) for s,label in enumerate(('first60','middle20','last20'))]
        if run==12:groups.append((18,0,'first60'))
        for measured_run,split,label_ in groups:
            key=f'run{measured_run}_{label_}';data=arrays(OUT/(key+'.npz'))
            reference=data['shapes'].mean(axis=0);fm=metrics[:,:,3].mean(axis=0);fd=data['metrics'][:,:,3].mean(axis=0)
            interval,draws=bootstrap(model,data,measured_run,split,nparent)
            counts=[np.histogram(d['q'],edges)[0] for d in (data,model,old,oldbase)]
            support=np.flatnonzero((counts[0]>=20)&np.all(np.array(counts[1:])>=10,axis=0))
            matched={}
            if len(support):
                wd=weights(data['q'],edges,support);df=wd@data['metrics'][:,:,3]
                for candidate,d in [('original',oldbase),('prompt010',old),('calibrated',model)]:
                    w=weights(d['q'],edges,support);ff=d['metrics'][:,:,3] if candidate=='calibrated' else d['fluctuations']
                    matched[candidate]={'n':int(np.isin(np.minimum(np.searchsorted(edges,d['q'],side='right')-1,5),support).sum()),
                      'F':(w@ff).tolist(),'relative':(w@ff/df-1).tolist(),
                      'shape_nrmse':error(w@d['shapes'],wd@data['shapes']),
                      'weighted_Q_KS':weighted_ks(d['q'],w,data['q'],wd)}
                matched['measured_F']=df.tolist()
            f50=data['metrics'][:,0,3];mf50=metrics[:,0,3]
            row['comparisons'][key]={'n':len(data['ids']),'measured_F':fd.tolist(),'relative':(fm/fd-1).tolist(),
                'shape_nrmse':error(shapes.mean(axis=0),reference),'common_Q_KS':ks(model['q'],data['q']),
                'F50_distribution_KS':ks(mf50,f50),'F50_IQR_ratio':float(np.diff(np.quantile(mf50,[.25,.75]))[0]/np.diff(np.quantile(f50,[.25,.75]))[0]),
                'pretrigger_50ns_variance_ratio':float(metrics[:,0,1].mean()/data['metrics'][:,0,1].mean()),
                'common_Q_bin_support_zero_based':support.tolist(),'matched_Q':matched,**interval}
            np.save(OUT/f'{source}_{key}_relative_draws.npy',draws)
            print(source,key,'rel',np.round(fm/fd-1,4),'shape',row['comparisons'][key]['shape_nrmse'],'F50 KS',row['comparisons'][key]['F50_distribution_KS'],flush=True)
        report['sources'][source]=row
    save(OUT/'validation.json',report)
    return report


def plot(report):
    ROOT.gROOT.SetBatch(True);ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetTitleFontSize(.038);ROOT.gStyle.SetTitleSize(.036,'XYZ');ROOT.gStyle.SetLabelSize(.032,'XYZ')
    c=ROOT.TCanvas('signal_calibration','Native waveform calibration',1500,1400);c.Divide(2,3,.004,.004)
    for col,(source,run) in enumerate([('Co60',3),('Am241',12)]):
        data=arrays(OUT/f'run{run}_first60.npz');late=arrays(OUT/f'run{run}_last20.npz');model=arrays(OUT/f'{source}_native.npz')
        old=arrays(BASE/'mc'/f'{source}_native_best.npz');row=report['sources'][source]
        curves=[(data['shapes'].mean(0),BLUE,'Training measured'),(late['shapes'].mean(0),GREEN,'Last20 measured'),
                (old['shapes'].mean(0),ROOT.kGray+2,'Previous prompt0.10'),(model['shapes'].mean(0),RED,'Calibrated native ADC')]
        ymax=max(v.max() for v,_,_ in curves)*1000*1.25
        panel(c,col+1,source+' / mean shape;Time from t10 (#mus);Area-normalized signal (#mus^{-1})',-.15,-.01,3.5,float(ymax))
        for i,(v,color,text) in enumerate(curves):
            graph(TIME_NS/1000,v*1000,color,2 if i==2 else 1);label(text,.49,.86-i*.055,color,.029)
        label('Shape comparison only; source tags are not truth',.19,.15,size=.023)
        xmax=210.;ymax=max(data['metrics'][:,:,3].mean(0).max(),model['metrics'][:,:,3].mean(0).max())*1.35
        panel(c,col+3,source+' / fluctuation means;Box-filter width (ns);Excess variance / mean (ADC)',35.,0.,xmax,float(ymax))
        x=np.array([50.,102.,202.])
        for i,(values,color,text) in enumerate([(data['metrics'][:,:,3].mean(0),BLUE,'Training measured'),(late['metrics'][:,:,3].mean(0),GREEN,'Last20 measured'),(old['fluctuations'].mean(0),ROOT.kGray+2,'Previous prompt0.10'),(model['metrics'][:,:,3].mean(0),RED,'Calibrated native ADC')]):
            graph(x,values,color,2 if i==2 else 1);label(text,.18,.87-i*.055,color,.028)
        a=row['comparisons'][f'run{run}_first60'];ci=np.array(a['MC_F_interval95']);vals=np.array(row['F'])
        g=ROOT.TGraphAsymmErrors(3,x,vals,np.zeros(3),np.zeros(3),np.maximum(vals-ci[0],0),np.maximum(ci[1]-vals,0))
        g.SetMarkerStyle(20);g.SetMarkerColor(RED);g.SetLineColor(RED);g.Draw('P SAME');KEEP.append(g)
        label('MC error: parent bootstrap; fit fixed',.18,.17,size=.025)
        qedges=np.linspace(report['q_edges'][0],report['q_edges'][-1],19)
        vals=[np.histogram(d['q'],qedges)[0]/len(d['q']) for d in (data,model)]
        panel(c,col+5,source+' / charge spectrum;Q (10^{3} ADC sample);Fraction per bin',qedges[0]/1000,0.,qedges[-1]/1000,float(max(v.max() for v in vals)*1.4))
        for v,color,text in zip(vals,[BLUE,RED],['Measured training','Calibrated native ADC']):
            graph((qedges[:-1]+qedges[1:])/2000,v,color)
        label('Measured training',.47,.86,BLUE,.030);label('Calibrated native ADC',.47,.80,RED,.030)
        label('No charge-spectrum fit or amplitude rescaling',.18,.17,size=.025)
    save_canvas(c,OUT,'native_calibration_summary');c.Print(str(OUT/'native_calibration_summary.pdf'))
    c=ROOT.TCanvas('individual_calibration','Unscaled ADC waveform examples',1500,1000);c.Divide(2,2,.004,.004)
    examples=[]
    for col,(source,run) in enumerate([('Co60',3),('Am241',12)]):
        data=arrays(OUT/f'run{run}_last20.npz');model=arrays(OUT/f'{source}_native.npz')
        order=np.argsort(data['q'])
        for rnk,fraction in enumerate((.25,.75)):
            i=order[int(fraction*(len(order)-1))];j=int(np.argmin(abs(model['q']-data['q'][i])))
            a=data['waves'][i].astype(float)-data['baseline'][i];b=model['waves'][j].astype(float)-model['baseline'][j]
            xa=(np.arange(len(a))-data['t10'][i])*2;xb=(np.arange(len(b))-model['t10'][j])*2
            panel(c,col+1+2*rnk,source+f' / last20 Q{int(fraction*100)};Time from t10 (ns);Baseline-subtracted ADC',-100.,-25.,3000.,float(max(a.max(),b.max())*1.25))
            graph(xa,a,BLUE);graph(xb,b,RED)
            label(f'Measured event {data["ids"][i]}',.49,.86,BLUE,.031)
            label(f'G4 parent {model["ids"][j]}',.49,.80,RED,.031)
            dq=model['q'][j]/data['q'][i]-1
            label(f'Nearest Q only: difference {100*dq:+.2f}%',.49,.73,size=.026)
            label('No amplitude scaling or shape-based selection',.18,.17,size=.025)
            examples.append({'source':source,'measured_group':f'run{run}_last20','Q_rank':fraction,
                'measured_id':int(data['ids'][i]),'MC_parent_id':int(model['ids'][j]),'relative_Q_difference':float(dq)})
    save_canvas(c,OUT,'individual_native_calibration');c.Print(str(OUT/'individual_native_calibration.pdf'))
    save(OUT/'figure_record.json',{'code_sha256':sha(Path(__file__)),'validation_sha256':sha(OUT/'validation.json'),
        'stems':['native_calibration_summary','individual_native_calibration'],'examples':examples})


if __name__=='__main__':
    plot(validate())
