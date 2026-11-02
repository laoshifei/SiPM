"""Report the spectral-state candidate, including failures and variance floors."""
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from spectral_split_common import *


def main():
    assert not (OUT/'decision.json').exists()
    fit=read(OUT/'fit.json');analysis=read(OUT/'analysis.json');figures=[]
    def finish(fig,name):
        fig.tight_layout();path=OUT/'figures'/name;fig.savefig(path,dpi=160);plt.close(fig);figures.append(path)
    rows=[];cal=[]
    for source,g in analysis.items():
        for v,r in g['variants'].items():
            row={'source':source,'variant':v,**r['point']}
            for n,ci in r['CI95'].items():
                lo,hi=ci['CI95'] if ci['CI95'] else [None,None];row.update({n+'_lo':lo,n+'_hi':hi,n+'_valid_draws':ci['valid_draws']})
            rows.append(row)
        for ef,f in fit[source]['folds'].items():
            for i,c in enumerate(f['candidates']):cal.append({'source':source,'evaluation_fold':ef,'chosen':i==f['chosen_index'],
                **{k:c[k] for k in ['delay_samples','common_variance_cv','high_variance_cv','a','b','objective']},
                **{f'predicted_F{s}_SD_ratio':c['predicted_SD_ratio'][j] for j,s in enumerate([50,102,202])}})
    for name,table in [('comparison_table.csv',rows),('proposal_table.csv',cal)]:
        with (OUT/name).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    plt.rcParams.update({'font.size':10})
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    for row,(source,group) in enumerate(fit.items()):
        for col,ef in enumerate(['0','1']):
            f=group['folds'][ef];ax=axes[row,col]
            ax.plot(range(3),f['zero_CV_SD_floor'],'k--o',label='Zero-CV floor',ms=4)
            for j,c in enumerate(f['candidates']):ax.plot(range(3),c['predicted_SD_ratio'],marker='o',lw=2.5 if j==f['chosen_index'] else .8,
                label=f"Delay {c['delay_samples']} samples"+(' (chosen)' if j==f['chosen_index'] else ''))
            ax.axhline(1,color='k',ls=':',lw=.8);ax.set_xticks(range(3),['F50','F102','F202']);ax.set_ylabel('Predicted MC / training SD');ax.set_title(f'{source}, evaluation fold {ef}')
    axes[0,0].legend(fontsize=8);fig.suptitle('Bounded full-covariance proposal: all candidates retained');finish(fig,'spectral_proposal.png')
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    for row,(source,group) in enumerate(fit.items()):
        for col,ef in enumerate(['0','1']):
            cfg=read(PROJECT/group['folds'][ef]['config']);model=cfg['signal_dependent_noise'];h=np.array(model['coefficients']);d=model['spectral_split']['delay_samples']
            freq=np.fft.rfftfreq(8192,2.)*1000;power=abs(np.fft.rfft(h,n=8192))**2;high=np.sin(np.pi*freq*(2*d)/1000)**2
            ax=axes[row,col];ax.loglog(freq[1:],power[1:],'k',label='Total (unchanged)');ax.loglog(freq[1:],(power*(1-high))[1:],label='Sum component');ax.loglog(freq[1:],(power*high)[1:],label='Difference component')
            ax.set_xlim(.12,250);ax.set_ylim(max(power)*1e-7,max(power)*2);ax.set_xlabel('Frequency (MHz)');ax.set_ylabel('Unit-variance FIR power');ax.set_title(f'{source}, fold {ef}, delay {d} samples')
    axes[0,0].legend(fontsize=8);fig.suptitle('Complementary spectra: overlapping lobes, not separate physical bands');finish(fig,'spectral_partition.png')
    styles=[('current_r0','Current r0','#777777','o'),('zero_cv_r0','Zero CV r0','#c68d32','s'),
            ('split_r0','Split r0','#1879b5','o'),('split_r1','Split r1','#3b9f75','^')]
    fig,axes=plt.subplots(2,2,figsize=(11,7),sharex=True)
    for col,(source,g) in enumerate(analysis.items()):
        for row,kind in enumerate(['SD','mean']):
            ax=axes[row,col]
            for j,(v,label,color,marker) in enumerate(styles):
                r=g['variants'][v];names=[f'F{s}_{kind}_ratio' for s in [50,102,202]];x=np.arange(3)+(j-1.5)*.06
                y=[r['point'][n] for n in names];ci=np.array([r['CI95'][n]['CI95'] for n in names])
                ax.plot(x,y,marker=marker,color=color,label=label,lw=1.2,ms=4);ax.vlines(x,ci[:,0],ci[:,1],color=color,lw=.8,alpha=.6)
            ax.axhline(1,color='k',ls='--',lw=.8);ax.axhspan(.9,1.1,color='#777777',alpha=.07)
            ax.set_xticks(range(3),['F50','F102','F202']);ax.set_title(source+' / '+kind);ax.set_ylabel('MC / measured '+kind)
    axes[0,0].legend(fontsize=8);fig.suptitle('Native complete-pool results with paired conditional intervals');finish(fig,'spectral_native_scales.png')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    names=[f'{n}_{metric}' for metric in ['SD_ratio','width95_ratio'] for n in ['QDC','negative_mean_time_us','PCA']]
    for ax,(source,g) in zip(axes,analysis.items()):
        for v,label,color,marker in [styles[0],styles[2],styles[3]]:ax.plot(range(6),[g['variants'][v]['point'][n] for n in names],marker=marker,color=color,label=label,ms=4)
        ax.axhline(1,color='k',ls='--',lw=.8);ax.set_title(source);ax.set_ylabel('MC / measured width')
        ax.set_xticks(range(6),['QDC SD','Time SD','PCA SD','QDC 95%','Time 95%','PCA 95%'],rotation=30,ha='right')
    axes[0].legend(fontsize=8);finish(fig,'spectral_native_pid.png')
    fig,axes=plt.subplots(2,2,figsize=(11,7),sharex=True);time=-21+40*np.arange(76)
    for col,(source,g) in enumerate(analysis.items()):
        measured=np.asarray(g['data_summary']['profile']);axes[0,col].plot(time,measured,'k',label='Measured')
        for v,label,color,_ in [styles[0],styles[2]]:
            y=np.asarray(g['summaries'][v]['profile']);axes[0,col].plot(time,y,color=color,label=label);axes[1,col].plot(time,y-measured,color=color,label=label)
        axes[0,col].set_title(source);axes[0,col].set_ylabel('Baseline-subtracted ADC');axes[1,col].set_ylabel('MC minus measured ADC');axes[1,col].set_xlabel('Time from integer alignment anchor (ns)');axes[1,col].axhline(0,color='k',lw=.6)
    axes[0,0].legend(fontsize=8);finish(fig,'spectral_native_mean_adc.png')
    findings=[];passed={}
    for source,g in analysis.items():
        passed[source]=all(x['passed'] for x in g['joint_guards'].values())
        for r in [0,1]:
            old=g['variants'][f'current_r{r}'];new=g['variants'][f'split_r{r}'];numbers=lambda v:'/'.join(f"{v['point'][f'F{s}_SD_ratio']:.3f}" for s in [50,102,202])
            count=sum(x['supported'] for x in g['selection'][f'split_r{r}']);ci=new['CI95']['F202_SD_ratio']['CI95']
            findings.append(f"{source}，r{r}：F50/F102/F202 SD比由{numbers(old)}变为{numbers(new)}；F202条件95%范围[{ci[0]:.3f}, {ci[1]:.3f}]。当前支持{count}个母事件，均ADC相对L2为{new['point']['mean_ADC_relative_L2']:.4f}。")
            findings.append(f"{source}，r{r}联合门槛："+('通过预定数值门槛；仍需新输运验证。' if g['joint_guards'][str(r)]['passed'] else '未通过，失败项：'+', '.join(g['joint_guards'][str(r)]['failed'])+'。'))
    conclusion='已实现并完成保持平均谱的分成分方差候选原生检查。'+('两个源在两条读出均通过预定数值门槛，但尚不能替代独立输运验证。' if all(passed.values()) else '候选尚未在两个源、两条读出中同时通过联合门槛，不能宣布PID与波动全面一致或替换默认配置。')
    decision={'conclusion_zh':conclusion,'findings_zh':findings,'joint_guard_passed':passed,
        'next_work_zh':('冻结候选，安排新种子的独立Geant4输运并重复完整选择与联合检查；原实验段仍是内部检查。' if all(passed.values()) else '保留本轮全部候选和失败项。对仍高于实测的零CV宽度下限，检验平均谱形状、有限时间相关结构及状态混合，而不是继续只增大成分CV；同时保留已得到的短尺度改善和原生均ADC/PID约束。'),
        'limitations_zh':['校准仅使用对折实测，但这些实验段和MC均已参与探索，不是新的盲验证。',
            '矩预测固定原母事件队列、零CV r0锚点和无噪声分母；自身定位、随机分母与重新选择的影响只能由实际原生ADC检查。',
            '差分成分具有频率旁瓣，不能称为独立的物理频带；方差CV描述经验状态，不是实测SiPM增益参数。',
            '本轮保持平均谱。固定分母近似下增加任何非负成分CV都不能将预测方差压到零CV下限以下；原生结果可以因选择和估计量非线性偏离该预测。',
            '仅两条预定读出流，不能视为噪声平均已经收敛。512组区间固定拟合和已实现噪声，不含重拟合、系统误差或新噪声积分。',
            'Am含alpha及gamma/电子成分，MC真值标签不提供实验事件真值；不更新实验ROC或硬件触发效率。'],
        'new_independent_mothers':0,'new_experimental_events':0,'logical_readout_records':1024000,
        'legacy_control_readouts_bit_identical':512000,'new_candidate_readouts':512000,
        'default_changed':False,'official_PID_changed':False,
        'inputs_sha256':{relative(OUT/n):sha(OUT/n) for n in ['protocol.json','fit.json','analysis.json','algebra_tests.json','build_tests.json']},'code_sha256':sha(__file__)}
    save(OUT/'decision.json',decision);save(OUT/'figure_record.json',{'figures_sha256':{relative(x):sha(x) for x in figures}})
    print('\n'.join(findings),flush=True)


if __name__=='__main__':main()
