"""Report charge migration, scale correlations and descriptive moment-line limits."""
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from charge_scale_common import *

FIGURES=['charge_migration','charge_width_controls','charge_scale_joint_modes',
         'charge_variance_line','charge_segment_checks','charge_mean_waveforms']
COLORS=['#267a9b','#cf7432'];COND_LABELS=['Original Q','Retained / old Q','Current Q']
VAR_LABELS=['Current response','Zero signal-variance CV']


def finish(fig,name,note):
    fig.tight_layout(rect=[0,.12,1,.91]);fig.text(.5,.025,note,ha='center',va='bottom',fontsize=9)
    for ext in ['png','svg','pdf']:fig.savefig(OUT/'figures'/f'{name}.{ext}',dpi=180)
    plt.close(fig)


def mark(ax,x,r,n,color,label=None):
    y=r['point'][n];ci=r['CI95'][n]['CI95']
    if y is None:return
    ax.plot(x,y,'o',color=color,ms=5,label=label)
    if ci is not None:
        ax.plot([x,x],ci,color=color,lw=1)
        for v in ci:ax.plot([x-.03,x+.03],[v,v],color=color,lw=1)


def figures(points,repeats):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.15})
    fig,axes=plt.subplots(2,2,figsize=(12,9))
    for col,(source,_) in enumerate(SOURCES):
        for row,v in enumerate(VARIANTS):
            a=np.mean([x['matrix'] for x in repeats[source]['migration'][v]],axis=0)[1:7]
            total=a.sum(1)[:,None];frac=np.divide(a,total,out=np.zeros_like(a),where=total>0);ax=axes[row,col];ax.grid(False)
            ax.imshow(frac,vmin=0,vmax=1,cmap='Blues',aspect='auto')
            for i in range(6):
                if total[i,0]==0:ax.text(3.5,i,'No original events',ha='center',va='center',color='#6d7780',fontsize=8)
                for j in range(8):
                    if a[i,j]>.01:ax.text(j,i,f'{a[i,j]:.1f}',ha='center',va='center',color='white' if frac[i,j]>.55 else 'black',fontsize=8)
            ax.set_xticks(range(8),['Under',*map(str,range(6)),'Over']);ax.set_yticks(range(6));ax.set_xlabel('Current-charge bin');ax.set_ylabel('Original-charge bin')
            ax.set_title(f'{source}: {VAR_LABELS[row]}')
    finish(fig,FIGURES[0],'Average event counts per readout over16 fixed-mother repeats; shading is the fraction within each original row.\nUnder/overflow is explicit. Am comparison support remains bins0..2; original-cohort truncation cannot be repaired by rebinning.')
    fig,axes=plt.subplots(2,3,figsize=(13,8))
    for row,(source,_) in enumerate(SOURCES):
        for col,scale in enumerate([50,102,202]):
            ax=axes[row,col]
            for iv,v in enumerate(VARIANTS):
                for j,c in enumerate(CONDITIONS):mark(ax,j+(iv-.5)*.15,repeats[source]['conditions'][c]['variants'][v],f'F{scale}_SD_ratio',COLORS[iv],VAR_LABELS[iv] if j==0 else None)
            ax.set_xticks(range(3),COND_LABELS,rotation=15);ax.axhline(1,color='black',ls='--',lw=1);ax.set_title(f'{source}: F{scale}');ax.set_ylabel('MC / measured event SD')
    h,l=axes[0,0].get_legend_handles_labels();fig.legend(h,l,loc='upper center',ncol=2)
    finish(fig,FIGURES[1],'Last20 / block128. Covariance is evaluated and Q-reweighted separately for each of16 readouts, then averaged.\n512 paired original data-block / whole-mother draws; fixed references, no refit/systematic or equivalence inference.')
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for col,(source,_) in enumerate(SOURCES):
        g=repeats[source]['conditions']['current_Q'];models=[g['data_summary'],g['model_summaries']['current'],g['model_summaries']['zero_cv']]
        for iv,(s,color,label) in enumerate(zip(models,['#26343d',*COLORS],['Measured',*VAR_LABELS])):
            if s is None:continue
            rho=correlation(np.asarray(s['cov'])[3:6,3:6])[np.triu_indices(3,1)]
            axes[0,col].plot(np.arange(3)+(iv-1)*.15,rho,'o',color=color,label=label)
        axes[0,col].set_xticks(range(3),['F50 / F102','F50 / F202','F102 / F202']);axes[0,col].set_ylabel('Within-Q correlation (point only)');axes[0,col].set_ylim(-1.05,1.05);axes[0,col].set_title(source)
        for iv,v in enumerate(VARIANTS):
            for j,n in enumerate(['F_common','F_short_long','F_curvature']):mark(axes[1,col],j+(iv-.5)*.15,g['variants'][v],n+'_SD_ratio',COLORS[iv])
        axes[1,col].axhline(1,color='black',ls='--',lw=1);axes[1,col].set_xticks(range(3),['Common','Short - long','Curvature']);axes[1,col].set_ylabel('Normalized-feature mode SD ratio')
    h,l=axes[0,0].get_legend_handles_labels();fig.legend(h,l,loc='upper center',ncol=3)
    finish(fig,FIGURES[2],'Current-Q sensitivity, last20 / block128 /16 readouts. F normalized by frozen opposite-fold measured Q-bin means.\nThree orthonormal feature combinations are diagnostic axes, not independent spectral bands or identified noise mechanisms.')
    fig,axes=plt.subplots(1,2,figsize=(12,6))
    for col,(source,_) in enumerate(SOURCES):
        for ic,c in enumerate(CONDITIONS):
            r=repeats[source]['conditions'][c]['interpolation']
            for j,scale in enumerate([50,102,202]):mark(axes[col],j+(ic-1)*.17,r,f'lambda_F{scale}',['#6d7780','#a37e41','#267a9b'][ic],COND_LABELS[ic] if j==0 else None)
        axes[col].set_xticks(range(3),['F50','F102','F202']);axes[col].set_title(source);axes[col].set_ylabel('Required covariance-line fraction lambda')
        for y in [0,1]:axes[col].axhline(y,color='black',ls='--',lw=1)
    h,l=axes[0].get_legend_handles_labels();fig.legend(h,l,loc='upper center',ncol=3)
    finish(fig,FIGURES[3],'lambda=(measured variance - zero-CV variance)/(current variance - zero-CV variance); values outside[0,1] retained.\nMoment interpolation only: lambda0/1 are the two evaluated endpoints. No physical CV inference or assumed native interpolation law.')
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for col,(source,run) in enumerate(SOURCES):
        contexts=[f'run{run}_{seg}_B{B}' for seg in SEGMENTS for B in [64,128]]
        for row,scale in enumerate([50,202]):
            for ic,c in enumerate(['original_Q','current_Q']):
                for j,key in enumerate(contexts):mark(axes[row,col],j+(ic-.5)*.15,points[key]['conditions'][c]['variants']['current'],f'F{scale}_SD_ratio',COLORS[ic],['Original Q','Current Q'][ic] if j==0 else None)
            axes[row,col].axhline(1,color='black',ls='--',lw=1);axes[row,col].set_ylabel(f'F{scale} event SD ratio')
            axes[row,col].set_xticks(range(6),[f'{seg}\nB{B}' for seg in SEGMENTS for B in [64,128]],fontsize=8)
        axes[0,col].set_title(source)
    h,l=axes[0,0].get_legend_handles_labels();fig.legend(h,l,loc='upper center',ncol=2)
    finish(fig,FIGURES[4],'All original acquisition segments and block sizes; inherited replica0, current response.\nNo response refit after charge reassignment, and no removal of weak or unestimable support cells.')
    fig,axes=plt.subplots(2,2,figsize=(12,8));time=-20+40*np.arange(76)
    for col,(source,_) in enumerate(SOURCES):
        g=repeats[source]['conditions'];data=np.asarray(g['original_Q']['data_summary']['profile'])
        axes[0,col].plot(time,data,color='#26343d',label='Measured')
        for c,v,color,label in [('original_Q','current','#6d7780','Original Q: current'),('current_Q','current',COLORS[0],'Current Q: current'),('current_Q','zero_cv',COLORS[1],'Current Q: zero CV')]:
            s=g[c]['model_summaries'][v]
            if s is None:continue
            y=np.asarray(s['profile']);axes[0,col].plot(time,y,color=color,label=label);axes[1,col].plot(time,y-data,color=color)
        axes[0,col].set_title(source);axes[0,col].set_ylabel('Mean baseline-subtracted ADC');axes[1,col].set_ylabel('MC - measured (ADC)')
        axes[1,col].axhline(0,color='black',ls='--',lw=1)
        for ax in axes[:,col]:ax.set_xlabel('Time from integration anchor (ns)');ax.set_xlim(-40,3000)
    h,l=axes[0,0].get_legend_handles_labels();fig.legend(h,l,loc='upper center',ncol=2)
    finish(fig,FIGURES[5],'Last20 / block128 /16 readouts. Only cohort weighting and charge assignment change; original ADC is untouched.\nNo event amplitude rescaling. Current-Q comparison remains conditional on the originally truncated cohort.')


def numeric(r,names):
    out=dict(r['point'])
    for n in names:
        c=r['CI95'][n];out[n+'_valid']=c['valid_draws']
        for suffix,index in [('lo',0),('hi',1)]:out[n+'_'+suffix]=None if c['CI95'] is None else c['CI95'][index]
    return out


def csv_file(name,items):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(items[0]),lineterminator='\n');w.writeheader();w.writerows(items)


def main():
    assert not (OUT/'record.json').exists() and not (OUT/'verification.json').exists()
    points=read(OUT/'point_analysis.json');repeats=read(OUT/'repeat_analysis.json')
    csv_file('point_table.csv',[{'context':key,'condition':c,'variant':v,**numeric(g['conditions'][c]['variants'][v],METRICS)} for key,g in points.items() for c in CONDITIONS for v in VARIANTS])
    csv_file('repeat_table.csv',[{'source':source,'condition':c,'variant':v,**numeric(g['conditions'][c]['variants'][v],METRICS)} for source,g in repeats.items() for c in CONDITIONS for v in VARIANTS])
    csv_file('variance_line_table.csv',[{'stage':stage,'context':key,'condition':c,**numeric(g['conditions'][c]['interpolation'],LAMBDA_NAMES)} for stage,collection in [('replica0',points),('mean16',repeats)] for key,g in collection.items() for c in CONDITIONS])
    migration_rows=[]
    for key,g in points.items():
        for v,m in g['migration'].items():migration_rows.append({'stage':'replica0','context':key,'variant':v,'replica':0,**{n:m[n] for n in ['old_support','current_support','leave_support','enter_support','bin_changed','underflow','overflow','total_original_selected']}})
    for source,g in repeats.items():
        for v,items in g['migration'].items():
            for j,m in enumerate(items):migration_rows.append({'stage':'positive_repeat','context':source,'variant':v,'replica':j+1,**{n:m[n] for n in ['old_support','current_support','leave_support','enter_support','bin_changed','underflow','overflow','total_original_selected']}})
    csv_file('migration_table.csv',migration_rows);figures(points,repeats)
    findings=[];gaps={}
    def fmt(r,n):
        x=r['point'][n];c=r['CI95'][n]['CI95']
        return '不可估计' if x is None else f'{x:.3f}'+('（区间不可估计）' if c is None else f' [{c[0]:.3f}, {c[1]:.3f}]')
    for source,_ in SOURCES:
        g=repeats[source]['conditions'];old=g['original_Q']['variants']['current'];new=g['current_Q']['variants']['current'];zero=g['current_Q']['variants']['zero_cv']
        findings.append(f'**{source}，晚段块128、16次读出：**按当前电荷重新分箱后，当前响应F202 SD比由'+fmt(old,'F202_SD_ratio')+'变为'+fmt(new,'F202_SD_ratio')+'；F50由'+fmt(old,'F50_SD_ratio')+'变为'+fmt(new,'F50_SD_ratio')+'。')
        findings.append(f'**{source}，当前电荷条件下零CV：**F202 SD比为'+fmt(zero,'F202_SD_ratio')+'，F50 SD比为'+fmt(zero,'F50_SD_ratio')+'。')
        r=g['current_Q']['interpolation'];findings.append(f'**{source}，方差直线诊断：**λ50='+fmt(r,'lambda_F50')+'，λ202='+fmt(r,'lambda_F202')+'；配对差λ50−λ202='+fmt(r,'lambda_F50_minus_F202')+'。λ只描述两个已算对照的方差端点，不是器件CV。')
        gaps[source]=r['CI95']['lambda_F50_minus_F202']['CI95']
    incompatible={s:(c is not None and (c[0]>0 or c[1]<0)) for s,c in gaps.items()}
    conclusion='已完成原电荷与当前电荷的配对敏感性、多尺度协方差和方差直线诊断；没有修改波形、参数或正式PID。'
    if any(incompatible.values()):conclusion+='至少一个源的短、长尺度要求在该条件方差直线上存在明确差异，不能用这条单方向插值直接选定统一响应。'
    next_work='先从完整可用母事件池核查当前读出选择的入口/出口，避免原电荷截断影响；再以F三尺度协方差及差分组合为约束，检验能分别控制尺度相关波动的最小噪声模型。保持平均ADC、PID协方差和分布宽度约束，候选通过后再冻结独立输运验证。'
    limitations=['本轮只改变原有限队列中的电荷条件，原范围之外的母事件不能重新进入；并未完成当前触发/质量选择和探测效率验证。',
        '所有参考均值、响应、PCA与Q半箱目标冻结，实验及模拟池都已参与探索；512次条件区间不含重拟合、系统误差或多重比较校正。',
        '每次读出先分别按当前Q重权，再平均协方差；重复作为同一母事件整体抽样。权重随噪声变化后不套用旧的固定权重母事件内/间分解。',
        'F50/F102/F202是相关的波动统计；正交组合只是特征轴，不能解释为相互独立频带或已辨识的微观噪声。',
        'λ是方差端点的代数插值比例，越界值和不稳定区间原样报告。未证明原生CV与此插值线性，不能据此推断物理CV或排除所有非线性模型。',
        'Am源含α及γ/电子成分。来源和主带不提供实验粒子真值；本轮不更新实验ROC或宣称PID全面一致。']
    inputs={relative(OUT/n):sha(OUT/n) for n in ['protocol.json','algebra_tests.json','point_analysis.json','repeat_analysis.json']}
    save(OUT/'decision.json',{'conclusion_zh':conclusion,'findings_zh':findings,'next_work_zh':next_work,'limitations_zh':limitations,
        'current_Q_lambda50_minus202_ranges':gaps,'moment_line_incompatibility_flags':incompatible,
        'inputs_sha256':inputs,'code_sha256':sha(__file__),'new_independent_mothers':0,'new_readout_replays':0,'default_changed':False,'official_PID_changed':False})
    save(OUT/'figure_record.json',{'figures':FIGURES,'inputs_sha256':inputs,'code_sha256':sha(__file__)})
    print('Reported current-charge sensitivity and joint-scale constraints.',flush=True)


if __name__=='__main__':main()
