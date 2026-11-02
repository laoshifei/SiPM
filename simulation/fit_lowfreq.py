"""Choose a frozen native-grid candidate using only the opposite measured fold."""
from lowfreq_common import *


def main():
    assert not (OUT/'fit.json').exists();p=read(OUT/'protocol.json');data,ref,dp=read_data(p);result={}
    for ef in [0,1]:
        d=statistics(dp,(data['fold']!=ef).astype(int));records=[]
        for label,corner,dc in GRID:
            stats=[];individual=[]
            for r in [0,1]:
                arrays=[load(OUT/'grid_arrays'/f'b{b:02d}_run12_eval{ef}_{label}_r{r}.npz') for b in range(4)]
                a=merge(arrays);assert np.all(a['fold']==ef);s=statistics(full_pack(a,p,SOURCE,ref),keep_Q=True);assert s is not None
                stats.append(s);individual.append(dict(zip(METRICS,contrasts(s,d))))
            m=average(stats);records.append({'candidate':label,'corner_MHz':corner,'dc_amplitude':dc,'loss':objective(m,d),
                'pooled_native_prediction':dict(zip(METRICS,contrasts(m,d))),'per_stream':individual,'summary':m})
        chosen=int(np.argmin([x['loss']['total'] for x in records]));name=records[chosen]['candidate'];cp=p['configs'][str(ef)][name]
        result[str(ef)]={'chosen_index':chosen,'candidate':name,'config':cp['path'],'config_sha256':cp['sha256'],'candidates':records,
            'training_ids':data['ids'][data['fold']!=ef],'evaluation_ids':data['ids'][data['fold']==ef],'target':d}
        print('Am241 fold',ef,'selected',name,'loss',records[chosen]['loss']['total'],'F SD',
            [round(records[chosen]['pooled_native_prediction'][f'F{s}_SD_ratio'],3) for s in [50,102,202]],flush=True)
    save(OUT/'fit.json',serialize({'folds':result,'protocol_sha256':sha(OUT/'protocol.json'),'code_sha256':sha(__file__)}))


if __name__=='__main__':main()
