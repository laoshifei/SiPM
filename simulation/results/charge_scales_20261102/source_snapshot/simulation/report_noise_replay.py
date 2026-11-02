"""Selection fairness, conditional noise variance and joint zero-CV tradeoffs."""
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from noise_replay_common import *
from analyze_noise_replay import POINT_METRICS,NESTED_METRICS

FIGURES=['noise_selection_audit','noise_repeat_decomposition','noise_zero_cv_joint_checks',
         'noise_cv_component_changes','noise_mean_waveforms','noise_repeat_convergence']
COLORS=['#267a9b','#cf7432'];LABELS=['Current response','Zero signal-variance CV']
SCOPE_LABELS=['All noise','Signal only','Electronics only','Zero CV: all']


def finish(fig,name,note):
    fig.tight_layout(rect=[0,.14,1,.91]);fig.text(.5,.025,note,ha='center',va='bottom',fontsize=9)
    for ext in ['png','svg','pdf']:fig.savefig(OUT/'figures'/f'{name}.{ext}',dpi=180)
    plt.close(fig)


def point(ax,x,r,name,color,label=None,marker='o'):
    y=r['point'][name];ci=r['CI95'][name]['CI95']
    if y is None:return
    ax.plot(x,y,marker=marker,color=color,ls='none',ms=5,label=label)
    if ci is not None:
        ax.plot([x,x],ci,color=color,lw=1)
        for v in ci:ax.plot([x-.025,x+.025],[v,v],color=color,lw=1)


def figures(selection,points,repeated):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.14})
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for col,(source,run) in enumerate(SOURCES):
        group=selection['groups'][f'run{run}_last20']['128']
        for row,metric in enumerate(['F202_SD_ratio','QDC_width95_ratio']):
            for iv,v in enumerate(VARIANTS):
                for k,case in enumerate(CASES):
                    point(axes[row,col],k+(iv-1)*.15,group['comparisons'][v][case],metric,
                          ['#6d7780','#25858b','#cf7432'][iv],['Frozen','RC + inherited','RC + extended'][iv] if k==0 else None)
            axes[row,col].set_xticks(range(4),['All selected','Inherited','Matched band','Fixed MC band'],rotation=15)
            axes[row,col].axhline(1,color='black',ls='--',lw=1)
            axes[row,col].set_ylabel('F202 event SD ratio' if row==0 else 'QDC central95% width ratio')
        axes[0,col].set_title(source)
    h,l=axes[0,0].get_legend_handles_labels();fig.legend(h,l,loc='upper center',ncol=3)
    finish(fig,FIGURES[0],'Last20 / block128. Old 0.4 QDC separator applied as an observable cut, with original Q-half targets.\nAll selected includes measured off-band events; source mixture and quality acceptance remain conditional.')
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for col,(source,run) in enumerate(SOURCES):
        r=repeated[source]
        for k,case in enumerate(REPLAY_CASES):
            x=r['cases'][case]
            point(axes[0,col],k,x,'F202_predicted_SD_ratio','#267a9b')
            for j,kind in enumerate(['within','between']):
                point(axes[1,col],k+(j-.5)*.16,x,f'F202_{kind}_fraction',COLORS[j],['Within readout','Between mothers (corrected)'][j] if k==0 else None)
        axes[0,col].set_title(source);axes[0,col].set_ylabel('Expected one-readout F202 SD / measured')
        axes[0,col].axhline(1,color='black',ls='--',lw=1)
        axes[1,col].set_ylabel('Contribution / expected F202 variance');axes[1,col].axhline(0,color='black',ls='--',lw=1)
        for ax in axes[:,col]:ax.set_xticks(range(4),SCOPE_LABELS,rotation=15)
    h,l=axes[1,0].get_legend_handles_labels();fig.legend(h,l,loc='upper center',ncol=2)
    finish(fig,FIGURES[1],'Last20 / block128 / extended response. Whole-mother bootstrap; finite-repeat correction retained, including negative estimates.\nSingle-stream cases freeze the other noise at replica0; their variances are conditional and must not be added as independent parts.')
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for col,(source,run) in enumerate(SOURCES):
        g=points[f'run{run}_last20_B128']
        for row,names in enumerate([['F50_SD_ratio','F102_SD_ratio','F202_SD_ratio'],
            ['QDC_width50_ratio','QDC_width95_ratio','negative_mean_time_us_width50_ratio','negative_mean_time_us_width95_ratio']]):
            for iv,v in enumerate(['current','zero_cv']):
                for k,name in enumerate(names):point(axes[row,col],k+(iv-.5)*.18,g['variants'][v],name,COLORS[iv],LABELS[iv] if k==0 else None)
            axes[row,col].set_xticks(range(len(names)),['F50','F102','F202'] if row==0 else ['QDC 50%','QDC 95%','Time 50%','Time 95%'])
            axes[row,col].set_ylabel('MC / measured width');axes[row,col].axhline(1,color='black',ls='--',lw=1)
        axes[0,col].set_title(source)
    h,l=axes[0,0].get_legend_handles_labels();fig.legend(h,l,loc='upper center',ncol=2)
    finish(fig,FIGURES[2],'Paired replica0 controls on all original mothers. Only signal-noise event variance CV changes.\nSame original ADC selection, Q bins, references and means; conditional ranges do not include refitting or systematics.')
    fig,axes=plt.subplots(2,3,figsize=(13,8))
    for row,(source,run) in enumerate(SOURCES):
        x=repeated[source]['zero_minus_current_components'];values=np.asarray(x['point'])
        for col,(feature,label) in enumerate([('F50','F50'),('F102','F102'),('F202','F202')]):
            ax=axes[row,col];j=FEATURES.index(feature)
            for k,kind in enumerate(['within','between','total']):
                rec={'point':{kind:values[j,k]},'CI95':{kind:x['CI95'][feature][kind]}}
                point(ax,k,rec,kind,['#267a9b','#cf7432','#303b44'][k])
            ax.set_xticks(range(3),['Within','Between','Total']);ax.axhline(0,color='black',ls='--',lw=1)
            ax.set_title(f'{source}: {label}');ax.set_ylabel('Zero CV - current / measured variance')
    finish(fig,FIGURES[3],'All-noise replicas1..16, paired indices. Finite-repeat corrected within + between changes equal the total exactly.\nConditional signal means include noise-dependent alignment and normalization; between mothers is not pure scintillation variance.')
    fig,axes=plt.subplots(2,2,figsize=(12,8));time=-20+40*np.arange(76)
    for col,(source,run) in enumerate(SOURCES):
        x=repeated[source];data=np.asarray(x['data_mean_profile'])
        axes[0,col].plot(time,data,color='#202b35',label='Measured')
        for j,case in enumerate(['current_all','zero_cv_all']):
            y=np.asarray(x['cases'][case]['mean_profile'])
            axes[0,col].plot(time,y,color=COLORS[j],label=LABELS[j])
            axes[1,col].plot(time,y-data,color=COLORS[j])
        axes[0,col].set_title(source);axes[0,col].set_ylabel('Baseline-subtracted mean ADC')
        axes[1,col].set_ylabel('MC - measured (ADC)');axes[1,col].axhline(0,color='black',ls='--',lw=1)
        for ax in axes[:,col]:ax.set_xlabel('Time from integration anchor (ns)');ax.set_xlim(-40,3000)
    h,l=axes[0,0].get_legend_handles_labels();fig.legend(h,l,loc='upper center',ncol=3)
    finish(fig,FIGURES[4],'Last20 / block128, Q-weighted means over original mothers and16 readouts. No event amplitude rescaling.\nNoiseless signal and avalanche histories are identical; mean differences include finite noise and observed-anchor effects.')
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for col,(source,run) in enumerate(SOURCES):
        x=repeated[source];g=points[f'run{run}_last20_B128'];dvar=x['data_summary']['cov'][5][5]
        for j,(case,variant) in enumerate([('current_all','current'),('zero_cv_all','zero_cv')]):
            z=x['cases'][case];values=np.asarray(z['per_repeat_contrasts'])[:,POINT_METRICS.index('F202_SD_ratio')]
            axes[0,col].plot(range(1,17),values,'o-',ms=4,color=COLORS[j],label=LABELS[j])
            axes[0,col].axhline(g['variants'][variant]['point']['F202_SD_ratio'],color=COLORS[j],ls=':',lw=1)
            for k,label in enumerate(['first8','last8','all16']):
                s=z['summary'] if label=='all16' else z['repeat_sensitivity'][label]
                y=np.sqrt(s['one_readout_expected_covariance'][5][5]/dvar)
                axes[1,col].plot(k+(j-.5)*.15,y,'o',color=COLORS[j])
        axes[0,col].set_title(source);axes[0,col].set_xlabel('Noise replica index');axes[0,col].set_ylabel('F202 SD ratio for each readout draw')
        axes[1,col].set_xticks(range(3),['First8','Last8','All16']);axes[1,col].set_ylabel('Conditional expected F202 SD ratio')
        for ax in axes[:,col]:ax.axhline(1,color='black',ls='--',lw=1)
    h,l=axes[0,0].get_legend_handles_labels();fig.legend(h,l,loc='upper center',ncol=2)
    finish(fig,FIGURES[5],'Dotted lines are inherited replica0; all positive replicas share the same physical mothers.\nReadout-draw variability is not independent transport replication. Half-repeat comparisons are predeclared sensitivity checks.')


def write_table(name,rows0):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows0[0]),lineterminator='\n');w.writeheader();w.writerows(rows0)


def numeric(r,names):
    result=dict(r['point'])
    for n in names:
        c=r['CI95'][n];result[n+'_lo']=None if c['CI95'] is None else c['CI95'][0]
        result[n+'_hi']=None if c['CI95'] is None else c['CI95'][1];result[n+'_valid']=c['valid_draws']
    return result


def main():
    assert not (OUT/'record.json').exists() and not (OUT/'verification.json').exists()
    selection=read(OUT/'selection_analysis.json');points=read(OUT/'point_analysis.json');repeated=read(OUT/'repeat_analysis.json')
    tables=[]
    for group,bb in selection['groups'].items():
        for B,g in bb.items():
            for v in VARIANTS:
                for case in CASES:tables.append({'group':group,'block':B,'variant':v,'case':case,**numeric(g['comparisons'][v][case],METRICS)})
    write_table('selection_table.csv',tables)
    write_table('zero_cv_point_table.csv',[{'context':context,'variant':v,**numeric(g['variants'][v],POINT_METRICS)}
                  for context,g in points.items() for v in ['current','zero_cv']])
    write_table('zero_cv_loss_table.csv',[{'context':context,'metric':n,'change':g['loss_change_zero_minus_current']['point'][n],
        'low':g['loss_change_zero_minus_current']['CI95'][n]['CI95'][0],
        'high':g['loss_change_zero_minus_current']['CI95'][n]['CI95'][1],
        'valid':g['loss_change_zero_minus_current']['CI95'][n]['valid_draws']}
        for context,g in points.items() for n in POINT_METRICS])
    write_table('repeat_table.csv',[{'source':source,'case':case,**numeric(g['cases'][case],NESTED_METRICS)}
                                      for source,g in repeated.items() for case in REPLAY_CASES])
    figures(selection,points,repeated)
    guard_metrics=[n for n in POINT_METRICS if not n.startswith('fixed_')]
    regressions=[]
    for context,g in points.items():
        for n in guard_metrics:
            r=g['loss_change_zero_minus_current'];ci=r['CI95'][n]['CI95']
            if ci is not None and ci[0]>0:regressions.append({'context':context,'metric':n,'loss_change':r['point'][n],'CI95':ci})
    target_benefits=[]
    for source,run in SOURCES:
        g=points[f'run{run}_last20_B128'];ci=g['loss_change_zero_minus_current']['CI95']['F202_SD_ratio']['CI95']
        target_benefits.append(ci is not None and ci[1]<0)
    passed=not regressions and all(target_benefits)
    def fmt(r,n):
        x=r['point'][n];c=r['CI95'][n];ci=c['CI95']
        return '不可估计' if x is None else f'{x:.3f}'+('（区间不可估计）' if ci is None else f' [{ci[0]:.3f}, {ci[1]:.3f}]')
    findings=[]
    retained=all(g['survival'][v]['all']['events']==g['survival'][v]['own_band']['events'] for bb in selection['groups'].values() for g in bb.values() for v in VARIANTS)
    findings.append('主带审计：所有36个固定MC条件/响应都通过既定0.4条件，统一主带后各宽度指标完全不变；这排除了本次主带删除差别作为当前MC偏宽的解释。' if retained else '主带筛选改变了部分MC样本，影响及有效统计量已逐条件记录。')
    for source,run in SOURCES:
        point0=points[f'run{run}_last20_B128'];rep=repeated[source]
        findings.append(f"**{source} 晚段块128：**零号配对对照的F202 SD比从"+fmt(point0['variants']['current'],'F202_SD_ratio')+'变为'+fmt(point0['variants']['zero_cv'],'F202_SD_ratio')+
            '；F50 SD比从'+fmt(point0['variants']['current'],'F50_SD_ratio')+'变为'+fmt(point0['variants']['zero_cv'],'F50_SD_ratio')+'。两种尺度同时报告，不只选择改善的一项。')
        findings.append(f"**{source} 重复噪声：**当前响应F202的同一母事件内方差比例为"+fmt(rep['cases']['current_all'],'F202_within_fraction')+
            '，不同母事件间的修正比例为'+fmt(rep['cases']['current_all'],'F202_between_fraction')+'。这是固定已有样本、有限重复校正后的条件分解。')
    if passed:
        conclusion='零CV对照通过本轮描述性联合保护条件；它仍只是需要另行冻结独立Geant4母事件验证的候选，未升级默认模型。'
        next_work='先冻结该候选及已有误差指标，再用新的完整Geant4母事件检查均波形、F三尺度和PID联合分布；同时评估完整当前ADC选择，不能将既有实验称为盲验证。'
    else:
        conclusion='零CV对照未通过联合保护条件。保留当前默认参数；本轮明确检验了主带选择与随机读出贡献，并记录了降CV的收益和代价。'
        next_work='围绕明确恶化的尺度/指标构建有针对性的噪声波动模型，并核查当前ADC电荷条件对分布的影响。单一全事件方差乘子不足以由本轮结果自动升级；候选通过联合约束后再冻结新母事件验证。'
    inputs={relative(OUT/n):sha(OUT/n) for n in ['selection_protocol.json','selection_analysis.json','replay_protocol.json','analysis_protocol.json','point_analysis.json','repeat_analysis.json','replay_tests.json']}
    limitations=[
        '实验只有来源标签。Am包含α、γ与电子贡献；主带和MC真实来源信息不能用来宣称实验粒子纯度。',
        '母事件及原Q来自原ADC的质量/共同电荷选择，重复读出不重新选择。条件内机制诊断不等于完整触发、重新计算电荷选择或源人群效率的验证。',
        '信号-only或电子学-only固定另一噪声在零号实现，其方差不能作为两个可相加的独立物理成分。',
        '不同母事件间的项包含噪声条件均值、定位与归一化效应；有限重复校正可能为负，不能截零或当作独立闪烁方差。',
        '512次区间按原实验块和完整母事件抽样，所有重复随母事件一起抽取；不把重复当作独立Geant4统计。区间固定参数/参考，不含拟合和系统误差。',
        '保护条件中的多项区间只是保守的描述性警报，未作多重比较修正。所有实验及本MC池均已探索，不是盲验证。']
    save(OUT/'decision.json',{'conclusion_zh':conclusion,'findings_zh':findings,'next_work_zh':next_work,'limitations_zh':limitations,
        'guard_metrics':guard_metrics,'clear_loss_regressions':regressions,'primary_F202_benefits':target_benefits,'candidate_guard_passed':passed,
        'inputs_sha256':inputs,'code_sha256':sha(__file__),'default_changed':False,'official_PID_changed':False,'new_independent_mothers':0})
    save(OUT/'figure_record.json',{'figures':FIGURES,'inputs_sha256':inputs,'code_sha256':sha(__file__)})
    print('Reported selection, repeated-noise decomposition, all joint controls and candidate guard:',passed,flush=True)


if __name__=='__main__':main()
