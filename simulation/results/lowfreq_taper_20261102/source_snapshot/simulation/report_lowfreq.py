"""Report the complete bounded native scan, including joint tradeoffs."""
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lowfreq_common import *


def main():
    assert not (OUT/'decision.json').exists();p=read(OUT/'protocol.json');fit=read(OUT/'fit.json');a=read(OUT/'analysis.json');figures=[]
    reused=(OUT/'full_pool_reuse.json').exists()
    def finish(fig,name):
        fig.tight_layout();path=OUT/'figures'/name;fig.savefig(path,dpi=160);plt.close(fig);figures.append(path)
    rows=[]
    for ef,f in fit['folds'].items():
        for j,c in enumerate(f['candidates']):rows.append({'evaluation_fold':ef,'candidate':c['candidate'],'chosen':j==f['chosen_index'],
            'corner_MHz':c['corner_MHz'],'dc_amplitude':c['dc_amplitude'],'loss':c['loss']['total'],
            **c['pooled_native_prediction'],**{'loss_'+k:v for k,v in c['loss']['terms'].items()}})
    native=[]
    for v,r in a['variants'].items():
        row={'variant':v,**r['point']}
        for n,ci in r['CI95'].items():lo,hi=ci['CI95'] or [None,None];row[n+'_lo']=lo;row[n+'_hi']=hi;row[n+'_valid']=ci['valid_draws']
        native.append(row)
    for name,table in [('grid_table.csv',rows),('comparison_table.csv',native)]:
        with (OUT/name).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(table[0]),lineterminator='\n');writer.writeheader();writer.writerows(table)
    plt.rcParams.update({'font.size':10})
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for ax,(ef,f) in zip(axes,fit['folds'].items()):
        labels=[c['candidate'] for c in f['candidates']];ys=[c['loss']['total'] for c in f['candidates']]
        ax.bar(range(10),ys,color=['#1879b5' if j==f['chosen_index'] else '#bbbbbb' for j in range(10)])
        ax.set_xticks(range(10),labels,rotation=60,ha='right');ax.set_title(f'Am241, evaluation fold {ef}');ax.set_ylabel('Opposite-fold joint calibration loss')
    fig.suptitle('All prespecified native candidates; blue = selected');finish(fig,'lowfreq_grid.png')
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for ax,(ef,f) in zip(axes,fit['folds'].items()):
        for label,path,color in [('Previous split',p['configs'][ef]['identity']['path'],'#777777'),('Selected',f['config'],'#1879b5')]:
            m=read(PROJECT/path)['signal_dependent_noise'];h=np.array(m['coefficients']);freq=np.fft.rfftfreq(16384,2.)*1000
            power=abs(np.fft.rfft(h,n=16384))**2*m['variance_per_signal_adc'];ax.loglog(freq[1:],power[1:],label=label,color=color)
        ax.set_xlabel('Frequency (MHz)');ax.set_ylabel('Signal-residual power per signal ADC');ax.set_title(f"Fold {ef}: {f['candidate']}");ax.set_xlim(.03,250)
    axes[0].legend();fig.suptitle('Actual finite FIR spectra, including truncation');finish(fig,'lowfreq_spectra.png')
    fig,axes=plt.subplots(1,2,figsize=(11,4));styles=[('previous_r0','Previous r0','#777777','o'),('taper_r0','Selected identity r0' if reused else 'Candidate r0','#1879b5','o'),('taper_r1','Selected identity r1' if reused else 'Candidate r1','#3b9f75','^')]
    for ax,kind in zip(axes,['SD','mean']):
        for j,(v,label,color,marker) in enumerate(styles):
            r=a['variants'][v];names=[f'F{s}_{kind}_ratio' for s in [50,102,202]];x=np.arange(3)+(j-1)*.06;y=[r['point'][n] for n in names];ci=np.array([r['CI95'][n]['CI95'] for n in names])
            ax.plot(x,y,color=color,marker=marker,label=label,lw=1.2);ax.vlines(x,ci[:,0],ci[:,1],color=color,alpha=.6,lw=.8)
        ax.axhspan(.9,1.1,color='k',alpha=.04);ax.axhline(1,color='k',ls='--',lw=.8);ax.set_xticks(range(3),['F50','F102','F202']);ax.set_ylabel('MC / measured '+kind)
    axes[0].legend();fig.suptitle('Am241 full-pool waveform fluctuations');finish(fig,'lowfreq_scales.png')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,metric in zip(axes,['SD_ratio','width95_ratio']):
        for v,label,color,marker in styles:ax.plot(range(3),[a['variants'][v]['point'][n+'_'+metric] for n in ['QDC','negative_mean_time_us','PCA']],color=color,marker=marker,label=label)
        ax.axhline(1,color='k',ls='--',lw=.8);ax.set_xticks(range(3),['QDC','Mean time','PCA']);ax.set_title(metric);ax.set_ylabel('MC / measured width')
    axes[0].legend();finish(fig,'lowfreq_pid.png')
    fig,axes=plt.subplots(2,1,figsize=(9,6),sharex=True);t=-21+40*np.arange(76);data=np.asarray(a['data_summary']['profile']);axes[0].plot(t,data,'k',label='Measured')
    for v,label,color,_ in styles[:2]:
        profile=np.asarray(a['summaries'][v]['profile']);axes[0].plot(t,profile,color=color,label=label);axes[1].plot(t,profile-data,color=color,label=label)
    axes[0].set_ylabel('Baseline-subtracted ADC');axes[0].legend();axes[1].set_ylabel('MC minus measured ADC');axes[1].set_xlabel('Time from integer alignment anchor (ns)');axes[1].axhline(0,color='k',lw=.8)
    finish(fig,'lowfreq_mean_adc.png')
    findings=[]
    for ef,f in fit['folds'].items():
        chosen=f['candidates'][f['chosen_index']];findings.append(f"评价折{ef}选择{f['candidate']}，对折校准联合目标由{f['candidates'][0]['loss']['total']:.3f}变为{chosen['loss']['total']:.3f}。")
        strong=next(c for c in f['candidates'] if c['candidate']=='f200_r050')['pooled_native_prediction']
        findings.append(f"评价折{ef}的2MHz、目标DC幅度比0.5对照：校准F202 SD比为{strong['F202_SD_ratio']:.3f}，但QDC/平均时间/PCA SD比降至{strong['QDC_SD_ratio']:.3f}/{strong['negative_mean_time_us_SD_ratio']:.3f}/{strong['PCA_SD_ratio']:.3f}。这些是原队列校准诊断，不是该对照的新完整池验证。")
    for r in [0,1]:
        prev=a['variants'][f'previous_r{r}'];now=a['variants'][f'taper_r{r}'];names=[f'F{s}_SD_ratio' for s in [50,102,202]]
        before='/'.join(f"{prev['point'][n]:.3f}" for n in names);after='/'.join(f"{now['point'][n]:.3f}" for n in names);ci=now['CI95']['F202_SD_ratio']['CI95'];guard=a['joint_guards'][str(r)]
        mode='沿用已核验完整池' if reused else '完整重选'
        findings.append(f"r{r}{mode}：F50/F102/F202 SD比{before}→{after}，F202条件95%范围[{ci[0]:.3f},{ci[1]:.3f}]，平均ADC相对L2={now['point']['mean_ADC_relative_L2']:.4f}。")
        findings.append(f"r{r}联合门槛："+('通过；仍需独立验证。' if guard['passed'] else '未通过；'+', '.join(guard['failed'])+'。'))
    passed=all(g['passed'] for g in a['joint_guards'].values());chosen=[f['candidate'] for f in fit['folds'].values()]
    conclusion=('两折均选择identity：本轮九组低频衰减参数都未改善预定联合目标；保留上一轮配置，Am241长尺度差异仍存在。' if reused else
        '两条读出在完整Am241母事件池上通过预定联合数值检查，仍属已有样本内部研究。' if passed else '低频衰减候选尚未在两条读出中同时通过联合检查，不能宣称Am241与实测全面一致。')
    next_work=('冻结当前候选并安排独立Geant4母事件验证，同时检查Co分折稳定性。' if passed else '依据完整网格的PID与三尺度权衡，检验有限时间相关结构及随机分母耦合；保留本轮全部失败项，不继续扩大同一低频衰减网格来追逐单个指标。')
    save(OUT/'decision.json',{'conclusion_zh':conclusion,'findings_zh':findings,'next_work_zh':next_work,'joint_guard_passed':passed,'selected':chosen,
        'limitations_zh':['全部实验段及原MC母事件均已探索；对折隔离新增校准层，不是盲验证。',
            '参数网格比较两条已实现读出，拟合后仍用相同原母事件池复核；这不是新的噪声或输运验证。',
            '低频传递函数只是经验读出敏感性，有限FIR截断会改变实际频率响应；报告实际系数频谱，不能当成已测量的电路。',
            '条件512组区间固定拟合、参考、质量阈值及已实现噪声，不包含网格选择、拟合误差、系统误差或噪声平均。',
            '校准原队列不能补回外部母事件，最终结果必须以全部192000个母事件的当前选择为准。',
            'Am含alpha及gamma/电子；这里没有实验粒子真值，也未更新ROC或硬件触发效率。'],
        'grid_native_readouts':29720,'grid_identity_readouts':2972,'full_parent_events':192000,'referenced_full_logical_readouts':768000,
        'new_full_pool_readouts':0 if reused else 768000,'full_pool_reused':reused,
        'inherited_legacy_control_readouts_bit_identical':384000,'new_independent_mothers':0,'default_changed':False,'official_PID_changed':False,
        'inputs_sha256':{relative(OUT/n):sha(OUT/n) for n in ['protocol.json','fit.json','analysis.json']},'code_sha256':sha(__file__)})
    save(OUT/'figure_record.json',{'figures_sha256':{relative(f):sha(f) for f in figures}})
    print('\n'.join(findings),flush=True)


if __name__=='__main__':main()
