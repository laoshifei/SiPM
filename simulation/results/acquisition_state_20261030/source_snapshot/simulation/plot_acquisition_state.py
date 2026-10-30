"""Plot acquisition-state support, estimator controls and signal covariance."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/sipm-acquisition-state-mpl')
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from prepare_acquisition_state import OUT,PROJECT,PARENT,SOURCES,SEGMENTS,CONTROLS,read,load,relative,finite
from analyze_acquisition_state import MODELS,event_weights,interval
from publication_stats import save,sha

COLORS=['#4e6878','#2074b4','#159078','#bd7132']
CONTROL_LABELS=['Original','Recent baseline','Smoothed anchor','Both changes']
MODEL_LABELS=['Q only','Q + pretrigger','Q + event order','Q + both']


def finish(fig,name,caption,legend=None,ncol=4):
    if legend is not None:
        h,l=legend.get_legend_handles_labels();fig.legend(h,l,loc='upper center',ncol=ncol,frameon=False,fontsize=9)
    fig.tight_layout(rect=(0,.05,1,.94 if legend is not None else 1))
    fig.text(.5,.012,caption,ha='center',fontsize=8)
    for ext in ['png','pdf','svg']:fig.savefig(OUT/'figures'/f'{name}.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)


def bar(ax,x,y,ci,color):
    ax.plot(x,y,'o',color=color,ms=4)
    if ci is not None:
        ax.vlines(x,*ci,color=color,lw=1);ax.hlines(ci,x-.025,x+.025,color=color,lw=1)


def main():
    assert not (OUT/'figure_record.json').exists()
    p=read(OUT/'protocol.json');c=read(OUT/'controls.json');s=read(OUT/'states.json')
    diagnostic=load(OUT/'state_diagnostics.npz');boot=load(OUT/'control_bootstrap.npz');supplement={}
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(4,2,figsize=(12,12))
    for col,(source,run) in enumerate(SOURCES):
        rows=[r for r in s['timeline'][source] if r['valid']];x=np.array([r['decile']/10+.05 for r in rows])
        first=load(OUT/'arrays'/f'run{run}_first60.npz');w=event_weights(first,p,source);baseline=w@first['baseline']
        for row,field,ylabel in [(0,'baseline','Mean baseline shift (ADC)'),(1,'baseline_RMS','Mean baseline RMS (ADC)'),(2,'low_frequency_fraction','Pretrigger power below20 MHz')]:
            y=np.array([r[field] for r in rows]);y=y-baseline if row==0 else y
            axes[row,col].plot(x,y,'o-',color=COLORS[col],ms=4)
            axes[row,col].set(title=f'{source}: '+['baseline level','baseline noise','pretrigger spectrum'][row],ylabel=ylabel)
        ref=-1000*np.array(c['groups'][f'run{run}_first60']['data'])[:,1]
        for k in range(4):
            axes[3,col].plot(x,[r['mean_time_ns'][k]-ref[k] for r in rows],'o-',color=COLORS[k],ms=3,label=CONTROL_LABELS[k])
        axes[3,col].set(title=f'{source}: measured mean time',ylabel='Change from own first60 mean (ns)')
        axes[3,col].axhline(0,color='k',lw=.6,ls='--')
        for ax in axes[:,col]:
            ax.axvspan(.6,1,color='#dce5eb',alpha=.3);ax.axvline(.8,color='#aab7bf',lw=.7,ls=':')
            ax.set(xlim=(0,1),xlabel='Fraction of original acquisition order');ax.grid(axis='y',alpha=.15)
    finish(fig,'acquisition_state_traces','Each raw acquisition decile matches fixed first60 Q halves; points are descriptive. Shading: previously explored later data.',axes[3,0])

    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for col,(source,run) in enumerate(SOURCES):
        for k,name in enumerate(CONTROLS):
            for i,seg in enumerate(SEGMENTS):
                r=c['groups'][f'run{run}_{seg}']['comparisons']['identity'][name]
                bar(axes[0,col],i+(k-1.5)*.14,r['mean_time_difference_ns'],r['uncertainty']['64']['contrasts'][0]['CI95'],COLORS[k])
            r=c['groups'][f'run{run}_last20']['comparisons']['identity'][name]
            for j in range(3):bar(axes[1,col],j+(k-1.5)*.14,r['IQR_ratio'][j],r['uncertainty']['64']['contrasts'][j+1]['CI95'],COLORS[k])
            axes[0,col].plot([],[],'o',color=COLORS[k],label=CONTROL_LABELS[k])
        axes[0,col].set(title=f'{source}: estimator sensitivity',ylabel='MC - data mean time (ns)')
        axes[0,col].set_xticks(range(3),['First60','Middle20','Last20']);axes[0,col].axhline(0,color='k',lw=.7,ls='--')
        axes[1,col].set(title=f'{source}: last20 PID widths',ylabel='MC / data RMS within-Q IQR')
        axes[1,col].set_xticks(range(3),['QDC','Mean time','PCA']);axes[1,col].axhline(1,color='k',lw=.7,ls='--')
        for ax in axes[:,col]:ax.grid(axis='y',alpha=.15)
    finish(fig,'estimator_state_controls','512 paired mother / raw-block64 draws. Same original IDs/Q for all estimators; smoothed waveform is used only to find the anchor.',axes[0,0])

    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for col,(source,run) in enumerate(SOURCES):
        row=s['sources'][source]
        for k,model in enumerate(MODELS[1:]):
            axes[0,col].bar(np.arange(3)+(k-1)*.23,row['CV_MSE_ratio_to_Q'][model],width=.22,color=COLORS[k+1],label=MODEL_LABELS[k+1])
        axes[0,col].set(title=f'{source}: six blocked first60 folds',ylabel='Held-out MSE / Q-only MSE')
        axes[0,col].set_xticks(range(3),['QDC','Mean time','PCA']);axes[0,col].axhline(1,color='k',lw=.7,ls='--')
        timeline=[r for r in s['timeline'][source] if r['valid']];x=np.array([r['decile']/10+.05 for r in timeline])
        ref=-1000*c['groups'][f'run{run}_first60']['data'][0][1]
        axes[1,col].plot(x,[r['mean_time_ns'][0]-ref for r in timeline],'ko-',lw=2,ms=4,label='Measured')
        for k,model in enumerate(MODELS):
            axes[1,col].plot(x,[r['predicted_mean_time_ns'][model]-ref for r in timeline],color=COLORS[k],ls='--',label=MODEL_LABELS[k])
        axes[1,col].axvspan(.6,1,color='#dce5eb',alpha=.3);axes[1,col].axhline(0,color='k',lw=.6)
        axes[1,col].set(title=f'{source}: full-first60 fit applied later',xlabel='Fraction of original acquisition order',ylabel='Mean-time change from first60 (ns)',xlim=(0,1))
        axes[1,col].legend(fontsize=8,loc='best',frameon=False)
        axes[1,col].text(.98,.04,f'Last20 state support: {100*row["segments"]["last20"]["weighted_state_support_fraction"]:.1f}%',
            transform=axes[1,col].transAxes,ha='right',fontsize=8,bbox={'facecolor':'white','alpha':.8,'edgecolor':'none'})
    finish(fig,'pretrigger_state_prediction','Top: pooled held-out squared errors; no model selected. Bottom: first60 fits, later exploratory extrapolation; no PID correction applied.',axes[0,0],3)

    fig,axes=plt.subplots(3,2,figsize=(12,11));time=np.arange(76)*40-20
    keysuffixes=['first60','last20','identity','no_signal_residual']
    labels=['Measured first60','Measured last20','Native with residual','Native residual off']
    for col,(source,run) in enumerate(SOURCES):
        for k,suffix in enumerate(keysuffixes):
            key=f'run{run}_{suffix}' if k<2 else f'run{run}__{suffix}'
            cv=diagnostic[key+'__shape_cov'];ac=diagnostic[key+'__hp_ac']
            axes[0,col].plot(time,100*np.sqrt(np.diag(cv)),color=COLORS[k],label=labels[k])
            axes[1,col].plot(p['highpass_lags_ns'],ac,'o-',color=COLORS[k],ms=3)
        axes[0,col].set(title=f'{source}: normalized gate fluctuations',xlabel='Original-anchor time (ns)',ylabel='SD of40 ns charge / Qlong (%)')
        axes[1,col].set(title=f'{source}:50 ns highpass excess covariance',xlabel='Lag (ns)',ylabel='(Signal - pretrigger) covariance / mean (ADC)')
        axes[1,col].set_xscale('symlog',linthresh=2,base=2);axes[1,col].set_xticks(p['highpass_lags_ns'],[str(x) for x in p['highpass_lags_ns']]);axes[1,col].axhline(0,color='k',lw=.6)
        for j,seg in enumerate(SEGMENTS):
            r=s['temporal_covariance'][f'run{run}_{seg}'];axes[2,col].plot(j,r['between_decile_fraction'][1],'ko',ms=5)
            for k,block in enumerate([64,128]):
                q=np.array(r['permutation_reference'][str(block)]['null_p025_p50_p975'])[:,1]
                bar(axes[2,col],j+(k*2-1)*.09,q[1],q[[0,2]],COLORS[k+1])
        axes[2,col].plot([],[],'ko',label='Observed temporal fraction')
        axes[2,col].plot([],[],'o',color=COLORS[1],label='Block64 null median/95%')
        axes[2,col].plot([],[],'o',color=COLORS[2],label='Block128 null median/95%')
        axes[2,col].set(title=f'{source}: time-score covariance between deciles',ylabel='Fraction of within-Q total variance')
        axes[2,col].set_xticks(range(3),['First60','Middle20','Last20']);axes[2,col].legend(fontsize=7,frameon=False,loc='best')
    finish(fig,'signal_state_covariance','Q-half matched; charge normalization and filtering affect covariance. Temporal fractions include sampling noise; permutation is a block-exchangeability reference.',axes[0,0],2)

    for source,run in SOURCES:
        ss={}
        for seg in SEGMENTS:
            a=load(OUT/'arrays'/f'run{run}_{seg}.npz');w=event_weights(a,p,source);bounds=np.array(s['sources'][source]['segments'][seg]['state_support']['bounds_p01_p99'])
            within=(a['state']>=bounds[:,0])&(a['state']<=bounds[:,1])
            ss[seg]={'baseline_ADC':float(w@a['baseline']),'baseline_RMS_ADC':float(w@a['baseline_rms']),
                'low_frequency_fraction':float(w@a['state'][:,2]),'marginal_state_support_fraction':(w@within).tolist(),
                'joint_state_support_fraction':s['sources'][source]['segments'][seg]['weighted_state_support_fraction'],
                'F':(w@a['F']).tolist()}
        delta=-1000*(np.array(c['groups'][f'run{run}_last20']['data'])[:,1]-np.array(c['groups'][f'run{run}_first60']['data'])[:,1])
        ss['measured_last_minus_first60_ns']=delta.tolist();ss['change_uncertainty']={}
        for block in [64,128]:
            diff=-1000*(boot[f'run{run}_last20__block{block}'][:,:,1]-boot[f'run{run}_first60__block{block}'][:,:,1])
            ss['change_uncertainty'][str(block)]=[interval(diff[:,j]) for j in range(4)]
        supplement[source]=ss
    save(OUT/'summary.json',finite(supplement))
    am=supplement['Am241'];last=c['groups']['run12_last20']['comparisons']['identity']
    conclusion=(f'**Am的采集状态发生了明显变化，但本轮没有获得可用于修正PID的可靠前触发状态模型。**\n\n'
        f'按电荷匹配的基线RMS由{am["first60"]["baseline_RMS_ADC"]:.2f}增至{am["last20"]["baseline_RMS_ADC"]:.2f} ADC，'
        f'低于20 MHz的前触发功率占比由{am["first60"]["low_frequency_fraction"]:.3f}增至{am["last20"]["low_frequency_fraction"]:.3f}。'
        f'然而，基线/定位的四种对照中，后段MC−实测平均时间仍为{min(v["mean_time_difference_ns"] for v in last.values()):.2f}–{max(v["mean_time_difference_ns"] for v in last.values()):.2f} ns，'
        '没有把差异解释为一个简单的基线估计或定位偏移。\n\n'
        f'Q+前触发状态模型的首段分块交叉检查，平均时间均方误差为Q-only的{s["sources"]["Am241"]["CV_MSE_ratio_to_Q"]["Q_state"][1]:.3f}倍；'
        f'后段落在既定训练状态支持区域内的加权比例只有{100*am["last20"]["joint_state_support_fraction"]:.1f}%。'
        '因此不能把模型外推值用作温漂或电子学校正。状态变化、波形估计误差与信号期相关波动之间仍不能作唯一物理归因。')
    next_work=('下一步优先检验**分读出状态的信号期响应模型**：保留本轮冻结结果，明确将某个后段作为新的条件校准样本，'
        '按原始事件块划分校准与内部检查，不再把它称为独立验证。先比较信号期低频整体形状起伏与短尺度残差的协方差，'
        '再同时约束均ADC、三种PID协方差和F；读出传递函数变化与快慢分量比例起伏均作为待检验候选，不能直接命名为CsI固有涨落。'
        '\n\n候选冻结后再运行独立Geant4母事件检查模拟统计，并保留原始与条件化两套结果。现有数据无法补足装置参数、温度或实验逐事件粒子真值。')
    save(OUT/'decision.json',{'promoted_to_default':False,'PID_definition_changed':False,'source_sha256':sha(Path(__file__)),
        'conclusion_zh':conclusion,'next_work_zh':next_work,'new_independent_mothers':0,'new_native_replay_entries':0,
        'limits':'Internal exploratory diagnostics. State support is a descriptive training-envelope rule, not a proof of transportability. Covariance normalization/filtering and sparse sampling affect attribution; no identified microscopic noise mechanism.'})
    save(OUT/'figure_record.json',{'code_sha256':sha(Path(__file__)),'inputs_sha256':{relative(path):sha(path) for path in [OUT/'protocol.json',OUT/'controls.json',OUT/'states.json']},
        'figures':['acquisition_state_traces','estimator_state_controls','pretrigger_state_prediction','signal_state_covariance']})
    print('Plotted state, estimator and covariance diagnostics.',flush=True)


if __name__=='__main__':main()
