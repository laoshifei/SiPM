"""Export conditional distributions and native response sensitivity diagnostics."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/sipm-conditional-response-mpl')
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from prepare_conditional_response import OUT, PROJECT, DATA, WIDTH, SOURCES, SEGMENTS, VARIANTS
from prepare_pid import read
from extract_pid import load
from publication_stats import save, sha
from analyze_conditional_response import design_for, average_value, cohort, interval
from analyze_npe_validation import raw_shapes
from analyze_co_tail import finite
from reference import TIME_NS

COLORS=['#606c76','#2166ac','#188a77','#d17b31','#b24963','#7854a1']
MODE_LABELS=['Original Q bins','Q halves matched','Q quartiles matched','Q halves + RMS matched']
CASE_LABELS=['Original residual','Residual off','AP .05 / 60 ns','AP .05 / 300 ns','AP .15 / 300 ns','Recovery 60 ns']


def finish(fig,name,caption,legend=None,ncol=3):
    if legend is not None:
        h,l=legend.get_legend_handles_labels()
        fig.legend(h,l,loc='upper center',ncol=ncol,frameon=False,fontsize=9)
    fig.tight_layout(rect=(0,.05,1,.93 if legend is not None else 1))
    fig.text(.5,.012,caption,ha='center',fontsize=8)
    for ext in ['png','pdf','svg']:
        fig.savefig(OUT/'figures'/f'{name}.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)


def point(ax,x,y,ci,color,marker='o'):
    if y is None or not np.isfinite(y):return
    ax.plot(x,y,marker,color=color,ms=4)
    if ci is not None:
        ax.vlines(x,*ci,color=color,lw=1)
        ax.hlines(ci,x-.025,x+.025,color=color,lw=1)


def main():
    assert not (OUT/'figure_record.json').exists()
    p=read(OUT/'protocol.json');c=read(OUT/'conditional.json');m=read(OUT/'mechanisms.json')
    cb=load(OUT/'conditional_bootstrap.npz');supplement={'data_segment_changes':{},'waveform_checks':{}}
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    modes=p['conditional_modes']
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for col,(source,run) in enumerate(SOURCES):
        for k,mode in enumerate(modes):
            for i,segment in enumerate(SEGMENTS):
                r=c['groups'][f'run{run}_{segment}']['modes'][mode]
                point(axes[0,col],i+(k-1.5)*.13,r['mean_time_difference_ns'],r['uncertainty']['64']['mean_time']['CI95'],COLORS[k])
            r=c['groups'][f'run{run}_last20']['modes'][mode]
            for j in range(3):
                point(axes[1,col],j+(k-1.5)*.13,r['IQR_ratio'][j],r['uncertainty']['64']['IQR_ratio'][j]['CI95'],COLORS[k])
            axes[0,col].plot([],[],'o',color=COLORS[k],label=MODE_LABELS[k])
        axes[0,col].set(title=f'{source}: center after conditioning',ylabel='MC - data mean time (ns)')
        axes[0,col].set_xticks(range(3),['First60','Middle20','Last20']);axes[0,col].axhline(0,color='k',lw=.7,ls='--')
        axes[1,col].set(title=f'{source}: last20 PID width',ylabel='RMS of within-Q IQR: MC / data')
        axes[1,col].set_xticks(range(3),['QDC','Mean time','PCA']);axes[1,col].axhline(1,color='k',lw=.7,ls='--')
        for ax in axes[:,col]:ax.grid(axis='y',alpha=.15)
        for mode in modes:
            first=c['groups'][f'run{run}_first60']['modes'][mode]
            for segment in SEGMENTS[1:]:
                later=c['groups'][f'run{run}_{segment}']['modes'][mode]
                value=-1000*(np.asarray(later['data'],float)[1]-np.asarray(first['data'],float)[1])
                cis={}
                for block in [64,128]:
                    diff=-1000*(cb[f'run{run}_{segment}__block{block}__{mode}'][:,1]-cb[f'run{run}_first60__block{block}__{mode}'][:,1])
                    cis[str(block)]=interval(diff)
                supplement['data_segment_changes'][f'run{run}_{segment}__{mode}']={'data_later_minus_first60_mean_time_ns':float(value),'uncertainty':cis}
    finish(fig,'conditional_pid','512 whole-mother / data-block64 draws; first60 matching targets fixed. Intervals do not include fit or target uncertainty.',axes[0,0],2)

    fig,axes=plt.subplots(2,2,figsize=(12,7))
    for col,(source,run) in enumerate(SOURCES):
        for row,mode in enumerate(['q2','q2r2']):
            ax=axes[row,col]
            sets=[('MC','first60','MC_support'),('Data first60','first60','data_support'),('Data last20','last20','data_support')]
            for k,(label,seg,field) in enumerate(sets):
                s=c['groups'][f'run{run}_{seg}']['modes'][mode][field]
                if s['valid']:
                    ax.plot([r['bin']+1 for r in s['bins']],[r['ESS'] for r in s['bins']],'o-',color=COLORS[k],label=label)
                else:
                    ax.text(.03,.94,f'{label}: unsupported target cell (Q bin {s["missing_bin"]+1})',
                        transform=ax.transAxes,va='top',fontsize=9,color=COLORS[k])
            ax.set(title=f'{source}: {MODE_LABELS[modes.index(mode)]}',xlabel='Original Q bin',ylabel='Effective sample size per bin')
            ax.grid(axis='y',alpha=.15)
    finish(fig,'conditional_support','ESS = (sum w)^2 / sum(w^2); no weight clipping or empty-cell extrapolation. Values refer to selected events, not all mothers.',axes[0,0])

    fig,axes=plt.subplots(2,2,figsize=(13,8))
    schemes=['fixed','reselected','fixed_anchor'];scheme_labels=['Fixed IDs / original Q','Reselected / own Q','Fixed IDs / Q / anchor']
    for col,(source,run) in enumerate(SOURCES):
        for k,scheme in enumerate(schemes):
            for i,variant in enumerate(VARIANTS):
                r=m['sources'][source]['variants'][variant]['schemes'][scheme]
                for row,j in [(0,0),(1,2)]:
                    point(axes[row,col],i+(k-1)*.16,r['change_from_no_residual'][j],r['paired_effect_CI95'][j]['CI95'],COLORS[k])
            axes[0,col].plot([],[],'o',color=COLORS[k],label=scheme_labels[k])
        axes[0,col].set(title=f'{source}: paired mean-time shift',ylabel='Change vs residual off (ns)')
        axes[0,col].axhline(0,color='k',lw=.7,ls='--')
        axes[1,col].set(title=f'{source}: paired mean-time width',ylabel='IQR width / residual-off width')
        axes[1,col].axhline(1,color='k',lw=.7,ls='--')
        for ax in axes[:,col]:
            ax.set_xticks(range(6),CASE_LABELS,rotation=28,ha='right',fontsize=8);ax.grid(axis='y',alpha=.15)
    finish(fig,'native_response_effects','512 paired original-mother draws; Q-half matching. Response cases are sensitivity tests, not fitted detector parameters.',axes[0,0])

    fig,axes=plt.subplots(3,2,figsize=(12,11))
    curves={'time_ns':TIME_NS};region=(TIME_NS>=-40)&(TIME_NS<3000)
    for col,(source,run) in enumerate(SOURCES):
        design=design_for(p,source);d=load(OUT/'arrays'/f'data_run{run}_first60.npz')
        old=load(DATA/f'run{run}_first60.npz');selection=load(WIDTH/'arrays'/f'measured_run{run}_first60.npz')['main_band']
        np.testing.assert_array_equal(d['ids'],old['ids'][selection])
        data_mean=average_value(d,raw_shapes(old['waves'][selection]),design)
        data_F=average_value(d,old['metrics'][selection,:,3],design)
        curves[source+'__data_first60']=data_mean
        axes[0,col].plot(TIME_NS[region],data_mean[region],color='k',lw=2,label='Measured first60')
        row={}
        for k,variant in enumerate(VARIANTS):
            a=load(OUT/'arrays'/f'run{run}_{source}__{variant}.npz');fixed=cohort(a,'fixed')
            mean=average_value(fixed,raw_shapes(a['waves'][a['fixed']]),design)
            F=average_value(fixed,a['metrics'][a['fixed']],design)
            curves[source+'__'+variant]=mean
            row[variant]={'mean_ADC_NRMSE_vs_first60':float(np.sqrt(np.mean((mean[region]-data_mean[region])**2))/data_mean.max()),
                'F_ratio_vs_first60':(F/data_F).tolist(),'scope':'Descriptive point estimates; fixed original MC IDs/Q, first60 data Q2 target. No fit or ranking.'}
            axes[0,col].plot(TIME_NS[region],mean[region],color=COLORS[k],lw=1,label=CASE_LABELS[k])
            axes[1,col].plot(TIME_NS[region],100*(mean[region]-data_mean[region])/data_mean.max(),color=COLORS[k],lw=1)
            axes[2,col].plot([50,102,202],F/data_F,'o-',color=COLORS[k],ms=4)
        axes[0,col].set(title=f'{source}: unscaled mean ADC',ylabel='ADC above baseline',xlabel='Time from t10 (ns)')
        axes[1,col].set(title=f'{source}: mean waveform difference',ylabel='Difference / data peak (%)',xlabel='Time from t10 (ns)')
        axes[1,col].axhline(0,color='k',lw=.7,ls='--')
        axes[2,col].set(title=f'{source}: short-scale fluctuation',ylabel='Mean F: MC / data',xlabel='Centered box scale (ns)')
        axes[2,col].axhline(1,color='k',lw=.7,ls='--')
        supplement['waveform_checks'][source]=row
    finish(fig,'native_response_waveforms','All native ADC, fixed original IDs/Q, first60 Q-half matching. F and waveform panels are descriptive; gain and optics unchanged.',axes[0,0],4)
    np.savez_compressed(OUT/'mean_curves.npz',**curves)
    save(OUT/'supplement.json',finite(supplement))
    figures=['conditional_pid','conditional_support','native_response_effects','native_response_waveforms']
    save(OUT/'figure_record.json',{'code_sha256':sha(Path(__file__)),'inputs_sha256':{str(x.relative_to(PROJECT)):sha(x) for x in [OUT/'protocol.json',OUT/'conditional.json',OUT/'mechanisms.json']},'figures':figures})
    am=c['groups']['run12_last20']['modes']['q2']
    ap=m['sources']['Am241']['variants']['ap15_300ns']['schemes']['fixed']
    ident=m['sources']['Am241']['variants']['identity']['schemes']['fixed']
    conclusion=(f'**本轮未找到可直接替代经验信号残差的后脉冲或恢复时间方案，默认配置保持不变。**\n\n'
        f'Am 后段按箱内电荷两分匹配后，平均时间宽度比为 **{am["IQR_ratio"][1]:.3f}**，'
        f'中心仍为 **+{am["mean_time_difference_ns"]:.2f} ns**。匹配造成的宽度比变化区间包含0；'
        '再同时匹配基线RMS时有空单元，无法可靠估计，不能据此归因于基线噪声。\n\n'
        f'固定同一批母事件和原Q时，原经验残差的平均时间宽度是关闭残差后的 **{ident["change_from_no_residual"][2]:.2f} 倍**；'
        f'15%／300 ns后脉冲压力对照仅为 **{ap["change_from_no_residual"][2]:.3f} 倍**，'
        f'却把中心推迟 **{ap["change_from_no_residual"][0]:.2f} ns**。'
        '这是本次固定光学/读出下的机制敏感性结果，不是对所有后脉冲模型的排除，也没有测出真实器件的AP概率。')
    next_work=('1. **先检查 Am 的采集段状态与信号形状。** 用原始事件顺序、前触发均值/RMS/相关谱建立状态描述；'
        '在每段有实际支持的Q区间，比较均值已去除后的信号协方差、早晚积分份额和平均时间。'
        '不跨缺失的Q×RMS单元外推，也不把信号残差拟合解释成温度测量。\n'
        '2. **在first60内约束所缺的相关波动。** 比较保留光子/SIPM统计后剩余协方差的时间结构，'
        '再判断是需要随采集状态变化的读出响应，还是事件间快慢分量起伏的候选模型。'
        '这些目前都是待检验假设，均需同时约束平均波形、三种PID协方差和F，而不是只调整单个带宽。\n'
        '3. **模型冻结后才增加独立模拟统计。** 目前支持箱仅有Co276、Am180条MC波形，'
        '应先按目标区间精度规划新种子完整Geant4输运量；既有重放不增加独立样本。'
        '后续再做同读出状态的α/γ真值PID检查，不能用放射源标签代替实验真值。')
    save(OUT/'decision.json',{'promoted_to_default':False,'conclusion_zh':conclusion,'next_work_zh':next_work,
        'source_sha256':sha(Path(__file__)),'new_independent_mothers':0,'experimental_ROC_updated':False,
        'limits':'Pointwise conditional intervals, fixed cuts/targets/model. Source main bands are not truth. Prespecified sensitivity settings only; no fitted AP/recovery values. Later data and prior validation mothers are now exploratory.',
        'incomplete_attempts':'Incomplete interrupted ROOT files and empty logs retained under diagnostics; excluded from successful analysis.'})
    print('Exported',len(figures),'figures and conditional segment / waveform checks.',flush=True)


if __name__=='__main__':main()
