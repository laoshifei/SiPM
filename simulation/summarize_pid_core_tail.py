"""Summarize core/tail evidence while preserving the boundary of internal diagnostics."""
import csv
from pid_core_tail_common import *


def main():
    assert not (OUT/'decision.json').exists()
    e=read(OUT/'diagnostics.json');row=e['groups']['run12_last20']['128'];r=row['variants']['spectrum_CV_fit'];x=r['point']
    data=row['data'];mc=r['native'];rows0=[]
    for key,blocks in e['groups'].items():
        for B,g in blocks.items():
            for v,t in g['variants'].items():
                out={'group':key,'raw_block_size':B,'variant':v,**t['point']}
                for n in NAMES:
                    ci=t['CI95'][n]['CI95'];out[n+'_low']=None if ci is None else ci[0];out[n+'_high']=None if ci is None else ci[1]
                rows0.append(out)
    with (OUT/'validation_table.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows0[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows0)
    conclusion=("本轮固定模拟参数和原事件全集，完成PID核心、外侧分位宽度、尾部方差及波形关联检查。"
        f"Am晚段扩展谱（块128）的QDC/PCA核心IQR比为{x['width50_QDC_ratio']:.3f}/{x['width50_PCA_ratio']:.3f}，"
        f"而95%分位宽度比为{x['width95_QDC_ratio']:.3f}/{x['width95_PCA_ratio']:.3f}；"
        f"平均时间IQR比为{x['width50_negative_mean_time_ratio']:.3f}，80%/95%分位宽度比为{x['width80_negative_mean_time_ratio']:.3f}/{x['width95_negative_mean_time_ratio']:.3f}。"
        "点估计提示核心与外侧分布的形状差异，不能用一个整体加宽系数解决；这些宽度比的条件95%区间均包含1，不能称为已确认的分布失配机制。")
    tail_d=np.array(data['tail_probability'][0])*100;tail_m=np.array(mc['tail_probability'][0])*100
    var_d=np.array(data['tail_variance_share'][0])*100;var_m=np.array(mc['tail_variance_share'][0])*100
    findings=[
        f"Am晚段QDC/平均时间/PCA超出训练块±2稳健尺度边界的加权比例，实测为{tail_d[0]:.2f}%/{tail_d[1]:.2f}%/{tail_d[2]:.2f}%，MC为{tail_m[0]:.2f}%/{tail_m[1]:.2f}%/{tail_m[2]:.2f}%。三项差值区间均包含0；MC对应原事件仅为{mc['tail_event_counts'][0]}条，不能由有限尾部样本确定稀有事件模型。",
        f"上述尾部对完整箱内方差的贡献，实测为{var_d[0]:.1f}%/{var_d[1]:.1f}%/{var_d[2]:.1f}%，MC为{var_m[0]:.1f}%/{var_m[1]:.1f}%/{var_m[2]:.1f}%。占比差的条件区间均包含0，不能简单断言实验异常尾部支配了差异。",
        "当前支持Q箱内的276个Co原母事件全为γ家族主导，180个Am原母事件全为α家族主导；能量90%/95%和雪崩电荷90%三个定义一致。按这些标签分解的类间均值方差为0，因此这批已选择MC中的粒子家族混合不能解释其核心/尾部差异；亚主导贡献和类内能损涨落并未被排除，更不能据此给实验赋真值。",
        "新增波形图显示每种PID与40 ns门电荷份额的协方差，并将尾部贡献单独列出；全体事件保持不变。PID、F与归一化波形共享样点，相关性仅用作定位线索，不可单独确认噪声、闪烁或电路原因。",
        "Co晚段扩展谱的95%分位宽度比为"+'/'.join(f"{e['groups']['run3_last20']['128']['variants']['spectrum_CV_fit']['point'][f'width95_{m}_ratio']:.3f}" for m in METHODS)+"。两源和前/中/后段均已保留，不能仅依据Am晚段挑选参数或宣布等效。"
    ]
    next_work=("下一步先检查实际ADC中的极端分位波形，区分基线漂移、定位偏差、长尾形状和可能重叠脉冲的可观察特征；预先固定选取规则，保留全部事件并报告原始ID。"
        "再用训练块约束低维波形形状变化，联合检查50/80/90/95%分位宽度、PID协方差和F分布，重点检验是否能够改变平均时间的外侧宽度而不过度加宽QDC/PCA核心。"
        "这只是待验证方向，不把形状模式预先命名为物理噪声机制。稳定候选冻结后再做新种子Geant4检查；已有实验仍是内部诊断。")
    decision={'scope':__doc__,'code_sha256':sha(__file__),'checkpoint_commit':read(OUT/'protocol.json')['checkpoint_commit'],
        'conclusion_zh':conclusion,'findings_zh':findings,'next_work_zh':next_work,
        'new_independent_mothers':0,'new_native_replays':0,'new_experimental_events':0,'default_changed':False,
        'unique_original_mothers':470,'supported_original_mothers':456,
        'Am_last20_extended_spectrum_point':x,'Am_last20_extended_spectrum_CI95':r['CI95'],
        'scope_zh':'固定参数、固定诊断边界的内部条件检查；全部实验已探索，不是新实验或独立MC验证。',
        'limitations_zh':['交叉划分只隔离本轮诊断边界的估计；已有响应、PCA和Q目标共享先前校准信息。',
            '配对区间固定响应参数、谱族选择和诊断边界，不包含其拟合误差；未进行等效性检验或多重比较校正。',
            '尾部事件很少，零计数可使经验bootstrap区间退化，不能当作尾部率为零或粒子泄漏上限。',
            '±2/3尺度采用训练中位数及IQR/1.349，不假定尾部服从高斯分布；它们仅为诊断边界。',
            '能量及雪崩祖先标签只用于已有MC，主导类相同不等于每条事件没有次要成分；实验Am主带不能当作纯α。'],
        'inputs_sha256':{relative(OUT/n):sha(OUT/n) for n in ['protocol.json','diagnostics.json']},
        'table_sha256':sha(OUT/'validation_table.csv')}
    save(OUT/'decision.json',decision);print(conclusion)


if __name__=='__main__':main()
