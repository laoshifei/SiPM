"""Report all temporal terms without converting covariance into causal noise labels."""
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from time_window_common import *


def main():
    assert not (OUT/'decision.json').exists();a=read(OUT/'analysis.json');figures=[];table=[];windows=[]
    for source,g in a.items():
        for v,scales in g['variants'].items():
            for s,r in scales.items():
                row={'source':source,'variant':v,'scale_ns':s,**r['point']}
                for n,ci in r['CI95'].items():lo,hi=ci['CI95'] or [None,None];row[n+'_lo']=lo;row[n+'_hi']=hi;row[n+'_valid']=ci['valid_draws']
                table.append(row)
                for w in range(4):windows.append({'source':source,'variant':v,'scale_ns':s,'window_start_ns':500+500*w,'window_stop_ns':1000+500*w,
                    'local_mean_ratio':r['point'][f'local_mean_ratio_w{w}'],'local_SD_ratio':r['point'][f'local_SD_ratio_w{w}'],
                    'local_SD_lo':r['CI95'][f'local_SD_ratio_w{w}']['CI95'][0],'local_SD_hi':r['CI95'][f'local_SD_ratio_w{w}']['CI95'][1]})
    for name,rows in [('attribution_table.csv',table),('local_window_table.csv',windows)]:
        with (OUT/name).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    def finish(fig,name):
        if not fig.get_constrained_layout():fig.tight_layout()
        path=OUT/'figures'/name;fig.savefig(path,dpi=160);plt.close(fig);figures.append(path)
    colors=['#1879b5','#3b9f75'];times=np.array([.75,1.25,1.75,2.25]);plt.rcParams.update({'font.size':10})
    fig,axes=plt.subplots(2,2,figsize=(11,7),sharex=True)
    for col,(source,g) in enumerate(a.items()):
        for row,kind in enumerate(['mean','SD']):
            ax=axes[row,col]
            for j,v in enumerate(VARIANTS):
                r=g['variants'][v]['202'];names=[f'local_{kind}_ratio_w{w}' for w in range(4)];y=[r['point'][n] for n in names];ci=np.array([r['CI95'][n]['CI95'] for n in names]);x=times+(j-.5)*.025
                ax.plot(x,y,'o-',color=colors[j],label=v,ms=4);ax.vlines(x,ci[:,0],ci[:,1],color=colors[j],alpha=.6,lw=.8)
            ax.axhline(1,color='k',ls='--',lw=.8);ax.set_title(source);ax.set_ylabel('Local F202 MC / measured '+kind);ax.set_xticks(times);ax.set_xlabel('Window center after t10 (us)')
    axes[0,0].legend();fig.suptitle('Four fixed500ns windows; conditional paired intervals');finish(fig,'time_window_local_F202.png')
    fig,axes=plt.subplots(1,2,figsize=(13,4.8));labels=['0.5–1.0','1.0–1.5','1.5–2.0','2.0–2.5','Between\nmeans','Pretrigger','Denominator']
    for ax,(source,g) in zip(axes,a.items()):
        for j,v in enumerate(VARIANTS):
            r=g['variants'][v]['202'];names=['excess_'+t for t in TERMS];x=np.arange(7)+(j-.5)*.3;y=np.array([r['point'][n] for n in names]);ci=np.array([r['CI95'][n]['CI95'] for n in names])
            ax.bar(x,y,width=.28,color=colors[j],alpha=.85,label=v);ax.vlines(x,ci[:,0],ci[:,1],color=colors[j],lw=.9)
        ax.axhline(0,color='k',lw=.8);ax.set_title(source);ax.set_xticks(range(7),labels,rotation=30,ha='right');ax.set_ylabel('Signed excess / measured F202 variance')
    axes[0].legend();fig.suptitle('Cov(term,F): signed attribution, not independent physical noise');finish(fig,'time_window_excess_terms.png')
    differences={}
    for source,g in a.items():
        dc=np.asarray(g['summaries']['data']['cov'])[2];vd=dc.sum()
        for v in VARIANTS:differences[(source,v)]=(np.asarray(g['summaries'][v]['cov'])[2,:4,:4]-dc[:4,:4])/vd
    bound=max(np.max(abs(x)) for x in differences.values());fig,axes=plt.subplots(2,2,figsize=(10,8),layout='constrained')
    for row,(source,_) in enumerate(a.items()):
        for col,v in enumerate(VARIANTS):
            ax=axes[row,col];x=differences[(source,v)];im=ax.imshow(x,cmap='RdBu_r',vmin=-bound,vmax=bound)
            for i in range(4):
                for j in range(4):ax.text(j,i,f'{x[i,j]:.3f}',ha='center',va='center',fontsize=9,color='white' if abs(x[i,j])>bound*.6 else 'black')
            ax.set_xticks(range(4),['1','2','3','4']);ax.set_yticks(range(4),['1','2','3','4']);ax.set_title(f'{source}, {v}');ax.set_xlabel('Time window');ax.set_ylabel('Time window')
    fig.colorbar(im,ax=axes.ravel().tolist(),label='MC minus measured covariance / measured global variance',shrink=.8)
    fig.suptitle('F202 fixed-denominator quarter covariance; off-diagonals count twice');finish(fig,'time_window_cross_covariance.png')
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for ax,(source,g) in zip(axes,a.items()):
        for j,v in enumerate(VARIANTS):
            for kind,style in [('global_SD_ratio','-'),('fixed_denominator_SD_ratio','--')]:
                y=[g['variants'][v][str(s)]['point'][kind] for s in SCALES];ci=np.array([g['variants'][v][str(s)]['CI95'][kind]['CI95'] for s in SCALES]);x=np.arange(3)+j*.035
                ax.plot(x,y,marker='o',ls=style,color=colors[j],label=v+(' / original' if style=='-' else ' / fixed denominator'),ms=4);ax.vlines(x,ci[:,0],ci[:,1],color=colors[j],alpha=.4,lw=.8)
        ax.set_xticks(range(3),['F50','F102','F202']);ax.axhline(1,color='k',ls=':',lw=.8);ax.set_title(source);ax.set_ylabel('MC / measured SD')
    axes[0].legend(fontsize=8);finish(fig,'time_window_denominator.png')
    fig,axes=plt.subplots(1,2,figsize=(11,4.2))
    for ax,(source,g) in zip(axes,a.items()):
        ax.plot(times,g['summaries']['data']['window_mu'],'ko-',label='Measured',ms=4)
        for j,v in enumerate(VARIANTS):ax.plot(times,g['summaries'][v]['window_mu'],'o-',color=colors[j],label=v,ms=4)
        ax.set_title(source);ax.set_xlabel('Window center after t10 (us)');ax.set_ylabel('Weighted local pulse mean (ADC)')
    axes[0].legend();finish(fig,'time_window_pulse_means.png')
    findings=[]
    for source,g in a.items():
        for v in VARIANTS:
            r=g['variants'][v]['202'];point=r['point'];local='/'.join(f"{point[f'local_SD_ratio_w{w}']:.3f}" for w in range(4))
            ordered=sorted(TERMS,key=lambda t:abs(point['excess_'+t]),reverse=True);largest=ordered[0];ci=r['CI95']['excess_'+largest]['CI95']
            findings.append(f"{source} {v}：整体F202 SD比{point['global_SD_ratio']:.3f}，固定分母诊断{point['fixed_denominator_SD_ratio']:.3f}；四段局部F202 SD比依次{local}。")
            findings.append(f"{source} {v}：整体方差差额/实测方差={point['global_variance_excess']:.3f}；绝对值最大的一项为{largest}，带符号贡献{point['excess_'+largest]:.3f}，条件95%范围[{ci[0]:.3f},{ci[1]:.3f}]。")
            findings.append(f"{source} {v}：四段固定分母对角项、段间交叉协方差、其余项的差额依次{point['quarter_diagonal_excess']:.3f}/{point['quarter_cross_covariance_excess']:.3f}/{point['other_covariance_excess']:.3f}，严格相加为整体差额。")
    decision={'conclusion_zh':'完成当前完整池波形的四时段精确方差分解；定位差额的时间分布与相关项，本轮没有更改参数或解决残余差异。',
        'findings_zh':findings,'next_work_zh':'以四时段均值、离散度和段间协方差共同约束下一轮模型。先核查差额集中的时段及其信号强度依赖，再测试时间依赖的信号残差包络；保留已匹配的PID协方差与平均波形，不由一个带符号贡献直接指定物理噪声机制。',
        'limitations_zh':['四段、三个滤波尺度和两个读出共享事件，不是独立重复试验；区间没有多重比较校正。',
            'Cov(term,F)是带符号统计贡献，可为负；包含脉冲曲率、信号/源混合、噪声相关性、归一化和有限参考误差，不能视作独立物理噪声强度。',
            '固定分母及局部F仅用于诊断，没有替换原ADC、原F或正式PID。',
            '512组区间完全继承已有实验块/母事件计数，固定拟合、参考、切选与噪声实现，不含模型选择、重新拟合或系统误差。',
            '既有数据已反复探索，不能称盲验证或证明等效；Am源包括alpha与gamma/电子，实验没有逐事件粒子真值。'],
        'new_independent_mothers':0,'new_native_readouts':0,'default_changed':False,'official_PID_changed':False,
        'inputs_sha256':{relative(OUT/n):sha(OUT/n) for n in ['protocol.json','analysis.json','algebra_tests.json']},'code_sha256':sha(__file__)}
    save(OUT/'decision.json',decision);save(OUT/'figure_record.json',{'figures_sha256':{relative(x):sha(x) for x in figures}});print('\n'.join(findings),flush=True)


if __name__=='__main__':main()
