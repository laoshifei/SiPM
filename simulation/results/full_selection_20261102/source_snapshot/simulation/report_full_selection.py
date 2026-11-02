"""Export full-pool selection evidence and its limited scientific conclusions."""
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from full_selection_common import *
from charge_scale_common import correlation


def merged(items):
    keys=['mothers','old_common','current_common','old_supported','current_supported','enter_support',
          'leave_support','new_beyond_old_common','fold_reference_QC_disagreement',
          'exit_current_selection_failure','exit_current_charge_failure']
    out={k:sum(x[k] for x in items) for k in keys}
    for k in ['supported_transition','enter_truth','leave_truth','current_truth','migration_union_support']:
        out[k]=np.sum([x[k] for x in items],axis=0).tolist()
    out['entrance_reasons']={k:sum(x['entrance_reasons'][k] for x in items) for k in items[0]['entrance_reasons']}
    return out


def main():
    assert not (OUT/'decision.json').exists(),'Refuse report overwrite'
    results=read(OUT/'analysis.json');selection={s:{v:merged(items) for v,items in a['selection'].items()} for s,a in results.items()}
    save(OUT/'selection_summary.json',selection)
    rows=[]
    for source,a in results.items():
        for c,g in a['conditions'].items():
            for v,r in g['variants'].items():
                row={'source':source,'condition':c,'variant':v,**r['point']}
                for n,ci in r['CI95'].items():
                    lo,hi=ci['CI95'] if ci['CI95'] is not None else [None,None]
                    row.update({n+'_lo':lo,n+'_hi':hi,n+'_valid_draws':ci['valid_draws']})
                rows.append(row)
    with (OUT/'comparison_table.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with (OUT/'selection_table.csv').open('w',newline='') as stream:
        rows2=[]
        for s,variants in selection.items():
            for v,a in variants.items():rows2.append({'source':s,'variant':v,**{k:value for k,value in a.items() if isinstance(value,(int,float))},**a['entrance_reasons']})
        writer=csv.DictWriter(stream,fieldnames=list(rows2[0]));writer.writeheader();writer.writerows(rows2)
    plt.rcParams.update({'font.size':10,'savefig.dpi':160})
    figures=[]
    def finish(fig,name):
        fig.tight_layout();path=OUT/'figures'/name;fig.savefig(path);plt.close(fig);figures.append(path)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,(s,variants) in zip(axes,selection.items()):
        x=np.arange(4);retained=[a['supported_transition'][1][1] for a in variants.values()]
        enter=[a['enter_support'] for a in variants.values()];leave=[a['leave_support'] for a in variants.values()]
        ax.bar(x,retained,label='Retained in support',color='#426e91');ax.bar(x,enter,bottom=retained,label='Entered support',color='#49a078')
        ax.bar(x,-np.asarray(leave),label='Left support',color='#cc6677')
        for i in range(4):ax.text(i,retained[i]+enter[i]+15,f'{retained[i]+enter[i]}',ha='center',fontsize=9)
        ax.set_xticks(x,['Current\nr0','Zero CV\nr0','Current\nr1','Zero CV\nr1']);ax.set_title(s);ax.set_ylabel('Original mother count');ax.axhline(0,color='k',lw=.6)
    axes[0].legend(fontsize=8);fig.suptitle('Complete existing pool: current software selection');finish(fig,'full_pool_selection.png')
    styles=[('old_cohort_current_Q','current_r0','Old cohort, current Q','#777777','o'),
            ('old_cohort_current_selection','current_r0','Old cohort, current QC/Q','#dc8b32','s'),
            ('full_pool_current_selection','current_r0','Full pool, r0','#1672a4','o'),
            ('full_pool_current_selection','current_r1','Full pool, r1','#389b78','^')]
    fig,axes=plt.subplots(2,2,figsize=(11,7),sharex=True)
    for row,(source,a) in enumerate(results.items()):
        for col,model in enumerate(['current','zero_cv']):
            ax=axes[row,col]
            for j,(c,v,label,color,marker) in enumerate(styles):
                r=a['conditions'][c]['variants'][v.replace('current',model)];names=[f'F{s}_SD_ratio' for s in [50,102,202]]
                y=np.array([r['point'][n] for n in names]);limits=np.array([r['CI95'][n]['CI95'] for n in names])
                x=np.arange(3)+(j-1.5)*.065
                ax.plot(x,y,marker=marker,color=color,label=label,ms=4,lw=1)
                # Percentile intervals need not bracket the point; draw endpoints.
                ax.vlines(x,limits[:,0],limits[:,1],color=color,alpha=.6,lw=1)
            ax.axhline(1,color='k',ls='--',lw=.8);ax.set_xticks(range(3),['F50','F102','F202']);ax.set_ylabel('MC / measured SD');ax.set_title(source+' / '+model)
    axes[0,0].legend(fontsize=7);fig.suptitle('Fluctuation widths after complete-pool reselection');finish(fig,'full_pool_scales.png')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    names=['QDC_SD_ratio','negative_mean_time_us_SD_ratio','PCA_SD_ratio','QDC_width95_ratio','negative_mean_time_us_width95_ratio','PCA_width95_ratio']
    for ax,(source,a) in zip(axes,results.items()):
        for j,(c,v,label,color,marker) in enumerate(styles):
            r=a['conditions'][c]['variants'][v];y=[r['point'][n] for n in names]
            ax.plot(np.arange(6),y,marker=marker,color=color,label=label,ms=4)
        ax.axhline(1,color='k',ls='--',lw=.8);ax.set_title(source+' / current response');ax.set_ylabel('MC / measured width')
        ax.set_xticks(range(6),['QDC SD','Time SD','PCA SD','QDC 95%','Time 95%','PCA 95%'],rotation=30,ha='right')
    axes[0].legend(fontsize=7);finish(fig,'full_pool_pid.png')
    fig,axes=plt.subplots(2,2,figsize=(11,7),sharex=True);time=-40+19+40*np.arange(76)
    for col,(source,a) in enumerate(results.items()):
        measured=np.array(a['data_summary']['profile']);axes[0,col].plot(time,measured,'k',label='Measured')
        for c,v,label,color,_ in [styles[0],styles[2],('full_pool_current_selection','zero_cv_r0','Full pool, zero CV','#bb5566','o')]:
            r=a['conditions'][c];y=np.asarray(r['summaries'][v]['profile'])
            axes[0,col].plot(time,y,color=color,label=label);axes[1,col].plot(time,y-measured,color=color,label=label)
        axes[0,col].set_title(source);axes[0,col].set_ylabel('Baseline-subtracted ADC');axes[1,col].set_ylabel('MC minus measured ADC');axes[1,col].set_xlabel('Time from integer alignment anchor (ns)');axes[1,col].axhline(0,color='k',lw=.7)
    axes[0,0].legend(fontsize=8);finish(fig,'full_pool_mean_adc.png')
    fig,axes=plt.subplots(2,4,figsize=(13,6))
    for row,(source,a) in enumerate(results.items()):
        d=np.asarray(a['data_summary']['cov'])
        for col,(v,sl,label) in enumerate([('current_r0',slice(0,3),'Current PID'),('zero_cv_r0',slice(0,3),'Zero CV PID'),('current_r0',slice(3,6),'Current F'),('zero_cv_r0',slice(3,6),'Zero CV F')]):
            m=np.asarray(a['conditions']['full_pool_current_selection']['summaries'][v]['cov']);sd=np.sqrt(np.diag(d)[sl]);z=(m[sl,sl]-d[sl,sl])/(sd[:,None]*sd[None,:])
            ax=axes[row,col];im=ax.imshow(z,cmap='coolwarm',vmin=-1,vmax=1)
            for i in range(3):
                for j in range(3):ax.text(j,i,f'{z[i,j]:.2f}',ha='center',va='center',fontsize=9)
            names=['QDC','Time','PCA'] if col<2 else ['F50','F102','F202']
            ax.set_xticks(range(3),names);ax.set_yticks(range(3),names);ax.set_title(source+' / '+label,fontsize=10)
    fig.suptitle('Full-pool covariance difference / measured SD products');finish(fig,'full_pool_covariance.png')
    findings=[]
    for source,a in results.items():
        for replica in [0,1]:
            v=f'current_r{replica}';count=selection[source][v]
            old=a['conditions']['old_cohort_current_Q']['variants'][v];full=a['conditions']['full_pool_current_selection']['variants'][v]
            cv=a['conditions']['full_pool_current_selection']['variants'][f'zero_cv_r{replica}']
            ci=full['CI95']['F202_SD_ratio']['CI95'];change=full['changes']['old_cohort_current_Q']['CI95']['F202_SD_ratio']['CI95']
            findings.append(f"{source}，读出r{replica}：原支持{count['old_supported']}个，当前支持{count['current_supported']}个，移入{count['enter_support']}个、移出{count['leave_support']}个。F202 SD比由原队列/当前Q的{old['point']['F202_SD_ratio']:.3f}变为完整选择的{full['point']['F202_SD_ratio']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]；配对变化范围[{change[0]:.3f}, {change[1]:.3f}]。")
            findings.append(f"{source}，r{replica}完整选择下零CV：F50 SD比{cv['point']['F50_SD_ratio']:.3f}，F202 SD比{cv['point']['F202_SD_ratio']:.3f}；当前响应分别为{full['point']['F50_SD_ratio']:.3f}、{full['point']['F202_SD_ratio']:.3f}。均ADC相对L2为{full['point']['mean_ADC_relative_L2']:.4f}。")
    decision={'conclusion_zh':'完整已有母事件池的当前软件重选没有消除F202过宽：主读出下Co60与Am241的模拟/实测SD比分别为1.235和1.188，第二读出流也保留差异。原队列外事件进入带来的变化较小；零CV仍不能同时满足短、长尺度宽度。已覆盖256000个原Geant4母事件，原队列特征与配对统计逐项闭合，没有重新拟合或采用新参数。',
        'findings_zh':findings,'new_independent_mothers':0,'readout_records':1024000,'default_changed':False,'official_PID_changed':False,
        'next_work_zh':'后续候选模型应在完整母事件池上应用当前质量与电荷选择，再共同约束F50/F102/F202协方差、平均ADC、PID核心与尾宽。先试能区分短长尺度相关波动的最小残差结构；通过联合检查后再冻结参数并做独立Geant4输运验证。',
        'limitations_zh':['这是已有256000个母事件的完整重选，不是新增独立输运，也不是未经探索的验证集。',
            '仅覆盖last20/B128及两个预定读出流。r0保留原噪声，r1是一条正编号噪声敏感性；没有声称两个流已收敛到噪声平均。',
            '512次区间固定已生成读出、全局QC阈值、PCA、响应和Q目标，不含重拟合、读出重抽样或装置系统误差。',
            '统计仍以共同电荷支持和实测主带为条件。模拟选择率是给定源几何下的软件切选，不是实测硬件触发效率；MC祖先标签不是实测事件真值。',
            'Am241含alpha及gamma/电子成分。尚未由这些结果唯一辨识PID带宽的微观成因，也未证明所有PID维度及尾部一致。'],
        'inputs_sha256':{relative(OUT/n):sha(OUT/n) for n in ['protocol.json','analysis_protocol.json','analysis.json','algebra_tests.json']},'code_sha256':sha(__file__)}
    save(OUT/'decision.json',decision);save(OUT/'figure_record.json',{'figures_sha256':{relative(x):sha(x) for x in figures}})
    print('\n'.join(findings),flush=True)


if __name__=='__main__':main()
