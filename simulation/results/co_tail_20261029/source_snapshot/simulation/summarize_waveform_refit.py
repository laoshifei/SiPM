"""Export frozen validation numbers and Chinese notebook notes; never refit."""
import csv
import json
from pathlib import Path
import shutil
from diagnose_charge import sha
from reference import PROJECT

BASE = PROJECT / 'simulation/results/waveform_refit_20261029'
OLD = PROJECT / 'simulation/results/signal_fluctuations_20261028'


def read(path):
    return json.loads(path.read_text())


def table(headers, rows):
    return '\n'.join(['|'+'|'.join(headers)+'|', '|'+'|'.join(['---']*len(headers))+'|']+
                     ['|'+'|'.join(map(str,row))+'|' for row in rows])


def percent(x):
    return f'{100*x:+.2f}%'


def interval(values, scale=1, digits=2):
    return '['+', '.join(f'{scale*x:.{digits}f}' for x in values)+']'


def main():
    fresh = read(BASE/'independent/analysis/validation.json')
    conditioned = read(BASE/'conditioned_last20/analysis/validation.json')
    separate = read(BASE/'run18/validation.json')
    feature_ci = read(BASE/'independent/analysis/feature_intervals.json')
    conditioned_ci = read(BASE/'conditioned_last20/analysis/feature_intervals.json')
    old = read(OLD/'independent/validation.json')
    figures = read(BASE/'figures/figure_record.json')
    presets = {}
    for run, source in [(3,'Co60'), (12,'Am241'), (18,'Am241')]:
        for segment in ['first60','last20']:
            if run == 18:
                source_path = BASE/f'run18/Am241_run18_{segment}.json'
            else:
                stage = 'final_parameters' if segment == 'first60' else 'conditioned_last20'
                source_path = BASE/stage/f'{source}_calibrated.json'
            suffix = '' if segment == 'first60' else '_last20_pretrigger'
            destination = PROJECT/f'simulation/configs/csi_tl_run{run}_waveform_20261029{suffix}.json'
            if destination.exists():
                assert sha(destination) == sha(source_path), 'Refuse to replace a changed preset'
            else:
                shutil.copyfile(source_path,destination)
            presets[f'run{run}_{segment}'] = {'path':str(destination.relative_to(PROJECT)),
                'source':str(source_path.relative_to(PROJECT)),'sha256':sha(destination)}

    records = []
    for stage, report in [('previous',old),
         ('optical_proposal',read(BASE/'parallel_shape/analysis/validation.json')),
         ('calibration_replay',read(BASE/'calibrated_mixture/analysis/validation.json')),
         ('fresh_training_state',fresh),('fresh_late_pretrigger',conditioned)]:
        for source, row in report['sources'].items():
            for group, c in row['comparisons'].items():
                for j, width in enumerate([50,102,202]):
                    records.append({'stage':stage,'source':source,'group':group,
                      'n_MC_mothers':row['nparent'],'n_MC_common':row['common_n'],'n_measured':c['n'],
                      'filter_ns':width,'F_MC':row['F'][j],'F_measured':c['measured_F'][j],
                      'relative_difference':c['relative'][j],
                      'relative_ci95_low':c['relative_interval95'][0][j],
                      'relative_ci95_high':c['relative_interval95'][1][j],
                      'shape_nrmse':c['shape_nrmse'],'shape_ci95_low':c['shape_interval95'][0],
                      'shape_ci95_high':c['shape_interval95'][1],'Q_KS':c['common_Q_KS']})
    for variant, row in separate['variants'].items():
        for group,c in row['comparisons'].items():
            for j,width in enumerate([50,102,202]):
                records.append({'stage':'run18_'+variant,'source':'Am241','group':group,
                  'n_MC_mothers':row['nparent'],'n_MC_common':row['common_n'],'n_measured':c['n'],
                  'filter_ns':width,'F_MC':row['F'][j],'F_measured':c['measured_F'][j],
                  'relative_difference':c['relative'][j],
                  'relative_ci95_low':c['relative_interval95'][0][j],
                  'relative_ci95_high':c['relative_interval95'][1][j],
                  'shape_nrmse':c['shape_nrmse'],'shape_ci95_low':c['shape_interval95'][0],
                  'shape_ci95_high':c['shape_interval95'][1],'Q_KS':c['common_Q_KS']})
    with (BASE/'waveform_validation.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)

    feature_rows=[]
    for stage, report, cis in [('fresh_training_state',fresh,feature_ci),
                              ('fresh_late_pretrigger',conditioned,conditioned_ci)]:
        for source,row in report['sources'].items():
            for group,c in row['comparisons'].items():
                for feature,values in c['features'].items():
                    r=cis['sources'][source][group][feature]
                    feature_rows.append({'stage':stage,'source':source,'group':group,'feature':feature,
                        'MC_median':r['MC_median'],'data_median':r['data_median'],
                        'difference':r['median_difference'],'difference_ci95_low':r['difference_interval95'][0],
                        'difference_ci95_high':r['difference_interval95'][1],'KS':values['KS']})
    with (BASE/'feature_validation.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(feature_rows[0]));writer.writeheader();writer.writerows(feature_rows)

    cfg=read(BASE/'final_parameters/Co60_calibrated.json');opt=cfg['optical'];ro=cfg['sipm']['readout']
    notes=['## 3. 冻结参数后的完整 Geant4 结果',
      '本轮完成两批完整输运：候选阶段 Co 4000 + Am 24000；冻结最终参数后另用新种子 '
      'Co 4000（291029503）+ Am 24000（291029512）。合计新增 **56000 个完整母衰变历史**。'
      '下表使用最后一批。其共同 Q 区间内合格 MC 仅为 Co **129**、Am **172** 条，'
      '不能把 28000 个母衰变全当作有效波形。数字化重放复用这些历史，不增加独立 MC 数。',
      'run 3 / 12 的训练波形数为 2450 / 1547；中段为 853 / 460；后段为 826 / 407。'
      'run 18 另用冻结前段 3000 条训练读出残差，冻结中段、后段各 3000 条作检查。'
      'run 18 全部合格前/中/后段分别有 65034 / 21628 / 21521 条；本轮该分支没有宣称使用全部事件。',
      '### 指标定义和比较范围',
      '共同 Q 区间为 **201824.1125–418733.2 ADC·sample**；原冻结实测 ID 不变。'
      'MC 重新执行相同的质量、幅度≥200 ADC、Q 区间与触发前窗口条件。平均形状逐条扣基线、'
      '按 t10 对齐后用 [−40,3000) ns 积分归一化。NRMSE 是 [−100,3500] ns 内平均曲线差的 RMS '
      '除以实测平均曲线最大值，**不是逐点相对误差，也不是幅度误差**。',
      '配置内的 acquisition.amplitude_threshold_adc 是软件触发代理，不是硬件触发仿真。'
      'run18 分支继承 Am 配置的160 ADC代理；本文比较统一再要求幅度≥200 ADC，'
      '不据此还原该次采集的硬件阈值或低于阈值的探测效率。',
      'F50 / F102 / F202：减去对应宽度的中心滑动平均，以 [500,2500) ns 信号残差方差减去 '
      '[−600,−200) ns 前触发残差方差，再除以信号区均值，单位 ADC。它是依赖滤波器的经验量，'
      '不能直接解释为增益或光电子数。表中偏差为 MC/实测−1。',
      '下表使用前段训练状态，未使用后段的基线或脉冲重新拟合。平均形状在 Co 上明显改善，'
      'Am 从上一版约 1.08% 变为约 1.18%，**不能写成每项都改善**。']
    rows=[]
    for source,run in [('Co60',3),('Am241',12)]:
        for title,report in [('上一版',old),('本轮冻结',fresh)]:
            c=report['sources'][source]['comparisons'][f'run{run}_last20']
            rows.append([f'{source} / {title}',f'{100*c["shape_nrmse"]:.3f}%',*[percent(v) for v in c['relative']],f'{c["common_Q_KS"]:.3f}'])
    notes.append(table(['后段检查','平均 NRMSE','F50','F102','F202','Q 谱 KS'],rows))
    rows=[]
    for source,run in [('Co60',3),('Am241',12)]:
        c=fresh['sources'][source]['comparisons'][f'run{run}_last20']
        for j,width in enumerate([50,102,202]):
            rows.append([source,width,percent(c['relative'][j]),interval([x[j] for x in c['relative_interval95']],100)+'%'])
    notes.append(table(['源','滤波宽度 ns','后段偏差','条件 95% 区间'],rows))
    notes.append('区间来自 512 次母衰变重采样与实测原始序号 64 条块重采样；参数、模型选择和前触发校准均固定，'
      '未计入拟合不确定度，未作多重比较修正。Co F202 的区间仍略高于零；小偏差不能自动称为统计等效。'
      '新随机种子检验的是 MC 重复性；既有中/后段在以前阶段已看过，不是新的盲实验数据。')
    notes.append('### 峰值时间、积分比和后段基线')
    rows=[]
    for source,run in [('Co60',3),('Am241',12)]:
        for name,feature in [('峰值−t10 / ns','tpeak_minus_t10_ns'),('短长积分比','short_long_ratio')]:
            r=feature_ci['sources'][source][f'run{run}_last20'][feature]
            digits=4 if feature=='short_long_ratio' else 2
            rows.append([source,name,f'{r["MC_median"]:.{digits}f}',f'{r["data_median"]:.{digits}f}',
                         interval(r['difference_interval95'],digits=digits)])
    notes.append(table(['源','特征','MC 中位数','实测中位数','MC−实测的条件95%区间'],rows))
    notes.append('Co 峰值时间较上一版更接近，但仍略早；Am 的峰值仍偏晚约 27 ns。'
      '因此本轮只达到主要平均形状和起伏均值的近似匹配，尚未通过所有特征分布的一致性检验。')
    rows=[]
    for source,run in [('Co60',3),('Am241',12)]:
        for title,report in [('前段训练状态',fresh),('后段前触发条件化',conditioned)]:
            c=report['sources'][source]['comparisons'][f'run{run}_last20'];f=c['features']['baseline_RMS_ADC']
            rows.append([source,title,f'{f["MC_quantiles"][2]:.3f}',f'{f["data_quantiles"][2]:.3f}',f'{f["KS"]:.3f}',
                f'{100*c["shape_nrmse"]:.3f}%',*[percent(x) for x in c['relative']]])
    notes.append(table(['源','读出状态','MC基线RMS/ADC','实测/ADC','RMS KS','平均NRMSE','F50','F102','F202'],rows))
    notes.append('后段条件化只读取该段最前 128 个样本的统计，保持发光、SiPM、信号残差不变。'
      '它能复现 Am 后段基线的主要变化，但用到了该段数据，不能作为前段参数预测后段基线的证据。'
      '模型允许事件间电平、方差及其相关性变化，未推断变化的电路原因。')
    notes.append('### run 18：同一物理模型，单独训练采集状态')
    transfer=fresh['sources']['Am241']['comparisons']['run18_first60']
    notes.append(f'直接把 run 12 状态用于 run 18 前段时，F50 偏差仍为 **{percent(transfer["relative"][0])}**。'
      '该迁移失败保留。另用 run 18 前段 3000 条重新估计信号残差强度、相关频谱与方差分散，'
      '没有更改几何、闪烁或 SiPM 参数。下面是这个**独立校准分支**，不是宣称 run 12 参数已经迁移成功。')
    rows=[]
    for group in ['run18_middle20','run18_last20']:
        c=separate['variants']['first60']['comparisons'][group]
        rows.append([group,f'{100*c["shape_nrmse"]:.3f}%',*[percent(x) for x in c['relative']],f'{c["common_Q_KS"]:.3f}'])
    notes.append(table(['run18检查 / 前段状态','平均NRMSE','F50','F102','F202','Q谱KS'],rows))
    c=separate['variants']['first60']['comparisons']['run18_last20']
    notes.append('run 18 后段 F50/F102/F202 的条件95%区间分别为 '+
      '、'.join(interval([r[j] for r in c['relative_interval95']],100)+'%' for j in range(3))+
      '。run 18 基线 RMS 中位数 MC/实测为 '
      f'{c["features"]["baseline_RMS_ADC"]["MC_quantiles"][2]:.3f}/{c["features"]["baseline_RMS_ADC"]["data_quantiles"][2]:.3f} ADC。'
      '但 Q 谱差异很大，不能把它当作绝对能谱或所有脉冲幅度均已复现。')
    notes.append('## 4. 当前可复现的参数')
    rows=[['电子/γ沉积时间常数',f'{opt["electron_tau_ns"][0]:.3f}, {opt["electron_tau_ns"][1]:.3f} ns'],
      ['电子快份额',f'{opt["electron_fast_fraction"]:.6f}'],
      ['α沉积时间常数',f'{opt["alpha_tau_ns"][0]:.3f}, {opt["alpha_tau_ns"][1]:.3f} ns'],
      ['α快份额',f'{opt["alpha_fast_fraction"]:.6f}'],
      ['共同被动 RC',f'{ro["lowpass_ns"]:.3f} ns'],
      ['候选快路径电荷份额',f'{cfg["readout_response"]["prompt_charge_fraction"]:.3g}（实际为零）'],
      ['有效光产额',f'{opt["yield_photons_per_MeV"]:.3f} photons/MeV'],
      ['α相对产额',f'{opt["alpha_light_yield_ratio"]:.6f}'],
      ['固定假设','PDE 0.30、增益1.7×10⁶、50 Ω、恢复29 ns、16单元求和、输入500 mV']]
    notes.append(table(['参数','值'],rows))
    rows=[]
    for run in [3,12,18]:
        c=read(PROJECT/presets[f'run{run}_first60']['path'])['signal_dependent_noise']
        rows.append([run,f'{c["variance_per_signal_adc"]:.6f}',f'{c["event_variance_cv"]:.6f}',len(c['coefficients'])])
    notes.append(table(['采集run','信号残差方差系数 κ / ADC','事件方差CV','FIR抽头数（2ns采样）'],rows))
    notes.append('信号残差模型为 nᵢ=√(κ·max(Sᵢ,0)·V)·(h∗z)ᵢ，Sᵢ 是原生无噪声信号的 ADC 值，'
      '∑h²=1，V 是均值1的对数正态方差乘子。1025个抽头由训练频谱与100 ns相关项、白项的正组合生成；'
      '并非1025个自由参数逐点追踪实测波形。完整系数与前触发状态保存在配置 JSON 中。')
    notes.append('电子慢时间 700 ns 位于搜索下界；α慢时间约 8.44 µs 长于本次形状检查窗。'
      '它们与 RC、份额、有限积分窗及光产额相互关联，当前是装置假设下的有效拟合值，'
      '**不能报告为该 CsI(Tl) 晶体本征时间常数或已测 α 淬灭比**。光产额调整仅保持时间参数改变后的期望3µs积分，'
      '没有按事件缩放模拟。快路径份额趋于零，不支持宣称需要额外快路径。'
      '源距仍沿用候选 Co 1 mm（假设0.25 mm Al出口片）/ Am 30 mm，未恢复真实装置历史。'
      '配置中的 evidence 保留历次假设，当前数值应以本表、status 与本次记录为准。')
    notes.append('### 仍未解决、会限制文章结论的部分')
    notes.append('Q 谱 KS 在 Co/run12 后段为约 0.257/0.223，run18 约 0.536；'
      'Am MC 在共同Q高端的事件支持不足，等Q箱比较只能使用满足数据≥20、MC≥10的箱。'
      '不能用等Q箱权重图代替未加权能谱检验。峰值时间和部分分布仍偏离。'
      '读出状态单独校准是一种经验响应补全，不构成唯一物理机理；'
      '没有补测单光电子响应、完整接线、温度、源封装与距离，因此只支持限定Q区间内的条件波形校准。')
    notes.append('## 5. 图与未缩放单条波形')
    notes.append('以下依次显示平均形状与起伏均值、特征分布、单条ADC、未加权Q谱和run18分支。'
      '单条示例固定取实测Q的25%/75%排位（run18取50%），选择最近Q的MC事件，'
      '仅扣基线并按t10对齐，没有按幅度或形状挑选。图例中的 pretrigger-conditioned 表示使用了该段前触发统计。'
      '实验和模拟是独立随机事件，不要求随机细峰逐点重合。')

    result={'scope':__doc__,'code_sha256':sha(Path(__file__)),
      'inputs_sha256':{str(p.relative_to(PROJECT)):sha(p) for p in [
          BASE/'independent/analysis/validation.json',BASE/'conditioned_last20/analysis/validation.json',
          BASE/'run18/validation.json',BASE/'independent/analysis/feature_intervals.json',
          BASE/'conditioned_last20/analysis/feature_intervals.json',BASE/'figures/figure_record.json']},
      'notebook_summary_markdown':'\n\n'.join(notes),'figures':figures['figures'],
      'presets':presets,'waveform_rows':len(records),'feature_rows':len(feature_rows),
      'MC_accounting':{'new_complete_mothers_this_phase':56000,'fresh_final_mothers':28000,
          'replays_are_not_additional_mothers':True},
      'not_all_observables_agree':True}
    (BASE/'final_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('Exported',len(records),'fluctuation rows,',len(feature_rows),'feature rows and',len(presets),'exact presets.')


if __name__=='__main__':
    main()
