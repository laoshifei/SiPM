"""Report descriptive conditioning, including unavailable cross-fold resamples."""
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from intensity_condition_common import *


def main():
    assert not (OUT/'decision.json').exists();a=read(OUT/'analysis.json');figures=[]
    for kind in ['global','local']:
        rows=[]
        for source,g in a.items():
            for v,r in g['variants'].items():
                for predictor,scales in r[kind].items():
                    for s,z in scales.items():
                        row={'source':source,'variant':v,'predictor':predictor,'scale_ns':s,**z['point']}
                        for name,ci in z['CI95'].items():
                            lo,hi=ci['CI95'] or [None,None];row[name+'_lo']=lo;row[name+'_hi']=hi;row[name+'_valid']=ci['valid_draws']
                        rows.append(row)
        with (OUT/(kind+'_table.csv')).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    def finish(fig,name):
        fig.tight_layout();path=OUT/'figures'/name;fig.savefig(path,dpi=160);plt.close(fig);figures.append(path)
    colors=['#1879b5','#3b9f75'];labels=['0.5–1.0','1.0–1.5','1.5–2.0','2.0–2.5','Whole']
    plt.rcParams.update({'font.size':10})
    fig,axes=plt.subplots(2,2,figsize=(12,7.6),sharex=True)
    for col,(source,g) in enumerate(a.items()):
        for row,key in enumerate(['residual_SD_ratio','crossfit_residual_SD_ratio']):
            ax=axes[row,col]
            for j,v in enumerate(VARIANTS):
                z=[g['variants'][v]['global'][p]['202'] for p in PREDICTORS];x=np.arange(5)+(j-.5)*.045
                y=[r['point'][key] for r in z];ci=np.array([r['CI95'][key]['CI95'] for r in z])
                ax.plot(x,y,'o-',color=colors[j],label=v,ms=4);ax.vlines(x,ci[:,0],ci[:,1],color=colors[j],alpha=.6,lw=.8)
                ax.axhline(z[0]['point']['original_SD_ratio'],ls='--',lw=.8,color=colors[j],alpha=.6)
            n=g['variants']['split_r0']['global'][PREDICTORS[0]]['202']['CI95'][key]['valid_draws']
            ax.axhline(1,color='k',lw=.7);ax.set_title(f'{source}: '+('within-sample' if row==0 else 'opposite-fold')+f' ({n}/512 valid)')
            ax.set_xticks(range(5),labels);ax.set_xlabel('Intensity predictor: window after t10 (us)');ax.set_ylabel('Global F202 residual SD: MC / measured')
    axes[0,0].legend();fig.suptitle('Linear intensity conditioning; dashed lines = original F202; intervals use available draws')
    finish(fig,'intensity_global_residuals.png')
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    for ax,(source,g) in zip(axes,a.items()):
        rows=[g['variants'][v]['global'][p]['202'] for p in [PREDICTORS[0],PREDICTORS[-1]] for v in VARIANTS]
        for j,(name,color) in enumerate([('linear_excess','#e38e35'),('residual_excess','#1879b5')]):
            x=np.arange(4)+(j-.5)*.30;y=[z['point'][name] for z in rows];ci=np.array([z['CI95'][name]['CI95'] for z in rows])
            ax.bar(x,y,width=.28,color=color,label=name.replace('_',' '));ax.vlines(x,ci[:,0],ci[:,1],color=color,lw=.9)
        ax.plot(range(4),[z['point']['variance_excess'] for z in rows],'kx',label='Total excess',ms=6)
        ax.axhline(0,color='k',lw=.7);ax.set_xticks(range(4),['Early r0','Early r1','Whole r0','Whole r1']);ax.set_title(source)
        ax.set_ylabel('Variance difference / measured F202 variance')
    axes[0].legend(fontsize=9);fig.suptitle('Exact within-sample variance partition; not a physical noise decomposition')
    finish(fig,'intensity_variance_partition.png')
    fig,axes=plt.subplots(2,2,figsize=(11,7.5),sharex=True)
    kinds=[('original_SD_ratio','Original','#555555'),('residual_SD_ratio','Within-sample','#1879b5'),('crossfit_residual_SD_ratio','Opposite-fold','#c0762c')]
    times=np.array([.75,1.25,1.75,2.25])
    for row,(source,g) in enumerate(a.items()):
        for col,v in enumerate(VARIANTS):
            ax=axes[row,col];zs=[g['variants'][v]['local'][p]['202'] for p in PREDICTORS[:4]]
            for j,(key,label,color) in enumerate(kinds):
                x=times+(j-1)*.015;y=[z['point'][key] for z in zs];ci=np.array([z['CI95'][key]['CI95'] for z in zs])
                ax.plot(x,y,'o-',color=color,label=label,ms=4);ax.vlines(x,ci[:,0],ci[:,1],color=color,alpha=.4,lw=.8)
            ax.axhline(1,color='k',lw=.7);ax.set_title(source+', '+v);ax.set_xlabel('Window center after t10 (us)');ax.set_ylabel('SD of window term: MC / measured');ax.set_xticks(times)
    axes[0,0].legend(fontsize=8);fig.suptitle('Each fixed-denominator window term conditioned on its own local intensity, 202 ns scale')
    finish(fig,'intensity_local_residuals.png')
    fig,axes=plt.subplots(1,2,figsize=(11,4.6))
    for ax,(source,g) in zip(axes,a.items()):
        for j,(v,key,label,color) in enumerate([('split_r0','data_slope','Measured','#333333'),('split_r0','MC_slope','split_r0',colors[0]),('split_r1','MC_slope','split_r1',colors[1])]):
            zs=[g['variants'][v]['local'][p]['202'] for p in PREDICTORS[:4]];x=times+(j-1)*.015;y=[z['point'][key] for z in zs];ci=np.array([z['CI95'][key]['CI95'] for z in zs])
            ax.plot(x,y,'o-',color=color,label=label,ms=4);ax.vlines(x,ci[:,0],ci[:,1],color=color,lw=.8,alpha=.6)
        ax.axhline(0,color='k',lw=.7);ax.set_title(source);ax.set_xlabel('Window center after t10 (us)');ax.set_ylabel('Slope: window term / normalized local intensity')
    axes[0].legend();fig.suptitle('Equal-Q average descriptive slopes; predictors use the same noisy ADC')
    finish(fig,'intensity_local_slopes.png')
    fig,axes=plt.subplots(2,2,figsize=(11,7.5),layout='constrained')
    for row,(source,g) in enumerate(a.items()):
        for col,v in enumerate(VARIANTS):
            ax=axes[row,col];rows=g['overlap'][v];bins=sorted({r['Q_bin'] for r in rows});matrix=np.array([[next(r['MC_mass_inside'] for r in rows if r['Q_bin']==b and r['predictor']==p) for p in PREDICTORS] for b in bins])
            im=ax.imshow(matrix,vmin=0,vmax=1,cmap='viridis',aspect='auto')
            for i in range(len(bins)):
                for j in range(5):ax.text(j,i,f'{matrix[i,j]:.2f}',ha='center',va='center',color='white' if matrix[i,j]<.5 else 'black',fontsize=9)
            ax.set_xticks(range(5),labels);ax.set_yticks(range(len(bins)),bins);ax.set_xlabel('Intensity predictor window (us)');ax.set_ylabel('Original Q-bin index');ax.set_title(source+', '+v)
    fig.colorbar(im,ax=axes.ravel().tolist(),label='MC weight in measured 5th–95th intensity interval',shrink=.8)
    fig.suptitle('Coverage diagnostic only; no intensity filtering or reweighting')
    path=OUT/'figures/intensity_range_overlap.png';fig.savefig(path,dpi=160);plt.close(fig);figures.append(path)
    findings=[]
    for source,g in a.items():
        for v in VARIANTS:
            z=g['variants'][v]['global'][PREDICTORS[0]]['202']['point'];l=g['variants'][v]['local'][PREDICTORS[0]]['202']['point']
            findings.append(f"{source} {v}：按500–1000 ns强度做线性诊断，整体F202 SD比原值{z['original_SD_ratio']:.3f}、同样本残差{z['residual_SD_ratio']:.3f}、对折残差{z['crossfit_residual_SD_ratio']:.3f}；线性项/剩余项方差差额为{z['linear_excess']:.3f}/{z['residual_excess']:.3f}，以实测整体方差为单位。")
            findings.append(f"{source} {v}：首窗口固定分母项的SD比原值{l['original_SD_ratio']:.3f}、同样本残差{l['residual_SD_ratio']:.3f}、对折残差{l['crossfit_residual_SD_ratio']:.3f}。该量不是局部F或完整PID宽度。")
        findings.append(f"{source}：原统计和同样本线性分解均保留512组；实测对折检查有效{g['valid_draws']['data']['crossfit']}/512组，未补值或丢弃Q箱。")
    decision={'conclusion_zh':'简单的线性脉冲强度关系未解释Am241的早期及整体F202过宽；扣除此关系后差额仍在。已完成统计定位，没有得到可推广的噪声模型修正。',
        'findings_zh':findings,'next_work_zh':'优先检验早期窗口在相近强度下的残余离散及其与其他时段的协方差，提出有界的时间相关方差起伏模型。原生候选须同时保持局部均值、平均ADC、三尺度F与PID协方差；不要用改变平均强度或事后回归残差替代ADC匹配。',
        'limitations_zh':['新回归在每个有效重采样中重新估计；已有物理拟合、参考、切选和噪声实现仍固定。',
            'Co60有113/512组实测对折重采样因支持不足而不可估计；对折区间只是有效组的分位范围，不能作为完整bootstrap的95%置信区间。Am241也有3组不可估计。',
            '强度与波动来自同一噪声ADC；回归可能包含内生相关、脉冲曲率和源混合，剩余量不是独立物理噪声。',
            '每个数据/MC样本分别拟合；残差变窄不代表存在可用于实验的校正，也不是PID性能改善。',
            '所有窗口、尺度及两条MC读出共享事件；没有新盲样本、实验粒子真值或多重比较校正。'],
        'new_native_readouts':0,'new_independent_mothers':0,'default_changed':False,'official_PID_changed':False,
        'inputs_sha256':{relative(OUT/n):sha(OUT/n) for n in ['protocol.json','analysis.json']},'code_sha256':sha(__file__)}
    save(OUT/'decision.json',decision);save(OUT/'figure_record.json',{'figures_sha256':{relative(x):sha(x) for x in figures}})
    print('\n'.join(findings),flush=True)


if __name__=='__main__':main()
