"""Supplementary figures for full-cohort and Q-matched response checks."""
import json
from pathlib import Path
import numpy as np
import ROOT
import publication_stats as p
from robustness_full_cohort import OUT,VIEWS
from plot_publication import panel,graph,label,whiskers,dots,KEEP,BLUE,RED,GREEN


def save(c,folder,name):
    c.Modified();c.Update()
    for ext in ('png','svg','pdf'):c.SaveAs(str(folder/(name+'.'+ext)))


def run(out=OUT):
    folder=out/'figures';folder.mkdir(exist_ok=True)
    ROOT.gROOT.SetBatch(True);ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetTitleFontSize(.036);ROOT.gStyle.SetTitleSize(.037,'XYZ');ROOT.gStyle.SetLabelSize(.032,'XYZ')
    full=json.loads((out/'full_cohort_covariance.json').read_text())
    mc=json.loads((out/'mc_charge_matched.json').read_text())
    c=ROOT.TCanvas('uncapped_covariance','Full-record covariance sensitivity',1800,620);c.Divide(3,1,.003,.003)
    interval=np.array(full['block_sizes']['64']['relative_interval95'])*100
    panel(c,1,'(a) Same six Q bins;1: frozen subset     2: full records;Run12 / run18 excess (%)',.5,50,2.5,64)
    for vi,(view,color,title) in enumerate([(VIEWS[0],BLUE,'Ratio < 0.40'),(VIEWS[1],RED,'0.30 <= ratio < 0.35')]):
        points=[full['points']['6'][scope][view]['run12_over18_minus1']*100 for scope in ('frozen3000','full')]
        whiskers(np.array([1.,2.])+(vi-.5)*.10,points,interval[0,[1,0],vi],interval[1,[1,0],vi],color,20+vi)
        label(title,.20,.86-vi*.06,color,.030)
    label('512 paired block draws; cell means refit',.18,.24,size=.027)
    label('Run18: 3000 frozen / 65034 full eligible',.18,.19,size=.025)
    panel(c,2,'(b) Full-record Q granularity;Number of equal-width Q bins;Run12 / run18 excess (%)',3,55,27,62)
    for vi,(view,color,title) in enumerate([(VIEWS[0],BLUE,'Ratio < 0.40'),(VIEWS[1],RED,'0.30 <= ratio < 0.35')]):
        x=np.array([6.,12.,24.]);y=np.array([full['points'][str(int(b))]['full'][view]['run12_over18_minus1']*100 for b in x])
        graph(x,y,color);dots(x,y,color,20+vi)
        label(title,.20,.86-vi*.06,color,.030)
    label('Point sensitivities; all Q bins supported',.18,.20,size=.026)
    panel(c,3,'(c) Full-record time sensitivity;Acquisition-index quarter within first60;Run12 / run18 excess (%)',.65,50,4.35,67)
    for vi,(view,color,title) in enumerate([(VIEWS[0],BLUE,'Ratio < 0.40'),(VIEWS[1],RED,'0.30 <= ratio < 0.35')]):
        r=full['time_quarters'][view]
        if r['points']:
            x=np.arange(1,5,dtype=float);y=np.array([a['relative']*100 for a in r['points']])
            graph(x,y,color);dots(x,y,color,20+vi)
        label(title,.20,.86-vi*.06,color,.030)
    label('Point estimates; same six-Q-bin support',.18,.24,size=.026)
    label('Index quarters are not equal wall-time periods',.18,.19,size=.024)
    save(c,folder,'figS01_full_cohort')

    c=ROOT.TCanvas('charge_matched_mc','Common-supported-Q response comparison',1450,1080);c.Divide(2,2,.003,.003)
    colors=(ROOT.kGray+2,RED,GREEN);titles=('Original response','Prompt fraction 0.10','Prompt + AP 0.15')
    for col,source in enumerate(('Co60','Am241')):
        setting=mc['sources'][source]['settings']['10'];bins=setting['supported_bins']
        qlabel=','.join(str(b+1) for b in bins)
        panel(c,col+1,f'{source} / common Q bins {qlabel};Box-filter width (ns);F_{{MC}} / F_{{data}} - 1 (%)',30,-110,225,110)
        graph(np.array([30.,225.]),np.zeros(2),ROOT.kGray+1,2)
        for i,(name,color,title) in enumerate(zip(('control','native_best','prompt_plus_ap15'),colors,titles)):
            point=np.array(setting['targets']['equal_bins']['candidates'][name]['point'][:3])*100
            ci=np.array(mc['bootstrap'][source]['10']['equal_bins'][name]['interval95'])[:,:3]*100
            xx=np.array([50.,102.,202.]);graph(xx,point,color,1 if i==1 else 2)
            whiskers(xx+(i-1)*2,point,ci[0],ci[1],color,20+i)
            label(title,.50,.86-i*.055,color,.028)
        label('Equal bin weights; conditional 95%',.50,.69,size=.025)
        label(f'Measured n={setting["selected_n"]["measured"]}; prompt MC n={setting["selected_n"]["native_best"]}',.50,.64,size=.024)
        panel(c,col+3,source+' / original common-Q counts;Original Q-bin index (1--6);Selected MC events per Q bin',.5,0,6.5,52)
        for b in bins:
            box=ROOT.TBox(b+.5,0,b+1.5,52);box.SetFillColor(ROOT.TColor.GetColor('#eaf3f7'));box.SetLineColor(ROOT.kWhite);box.Draw();KEEP.append(box)
        for i,(name,color,title) in enumerate(zip(('control','native_best','prompt_plus_ap15'),colors,titles)):
            values=np.array(mc['sources'][source]['counts_by_bin'][name],float)
            xx=np.arange(1,7,dtype=float)+(i-1)*.055;graph(xx,values,color);dots(xx,values,color,20+i)
            label(title,.50,.86-i*.055,color,.027)
        graph(np.array([.5,6.5]),np.full(2,10.),ROOT.kGray+1,2)
        label('Shaded: all MC candidates have n >= 10',.55,.69,size=.023)
        label('Every measured bin has n >= 20',.55,.64,size=.024)
        ROOT.gPad.RedrawAxis()
    save(c,folder,'figS02_charge_matched_mc')
    p.save(folder/'manifest.json',{'stems':['figS01_full_cohort','figS02_charge_matched_mc'],
      'code_sha256':p.sha(Path(__file__)),'inputs_sha256':{name:p.sha(out/name) for name in ['full_cohort_covariance.json','mc_charge_matched.json']},
      'note':'Only panel(a) of S01 has intervals. S02 uses source-specific common support, never all six bins for both sources.'})


if __name__=='__main__':run()
