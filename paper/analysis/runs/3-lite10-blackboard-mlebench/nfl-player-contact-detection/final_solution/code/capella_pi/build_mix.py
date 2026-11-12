import pandas as pd, numpy as np, json
def logit(p,e=1e-6): p=np.clip(np.asarray(p,dtype=np.float64),e,1-e); return np.log(p/(1-p))
W='workspaces/capella_pi'; P='workspaces/canopus_pi/preds'
wv4,wv5,we=0.4,0.3,0.3
# OOF
v4=pd.read_parquet(f'{W}/oof_stack_v4.parquet')[['contact_id','fold','contact','is_ground','stack_oof']].rename(columns={'stack_oof':'v4'})
v5=pd.read_parquet(f'{W}/oof_stack_v5.parquet')[['contact_id','stack_oof']].rename(columns={'stack_oof':'v5'})
en=pd.read_parquet(f'{P}/ens_oof_probs_v4.parquet')[['contact_id','p_pp','p_gr']]
o=v4.merge(v5,on='contact_id').merge(en,on='contact_id')
# align by contact_id, recompute is_ground from labels order not needed
o['ens']=np.where(o.is_ground.values==1,o.p_gr.values,o.p_pp.values)
o['stack_oof']=(wv4*logit(o.v4.values)+wv5*logit(o.v5.values)+we*logit(o.ens.values)).astype(np.float32)
o[['contact_id','fold','contact','is_ground','stack_oof']].to_parquet(f'{W}/oof_stack_mix.parquet',index=False)
# TEST
t4=pd.read_parquet(f'{W}/test_stack_v4.parquet').rename(columns={'stack_pred':'v4'})
t5=pd.read_parquet(f'{W}/test_stack_v5.parquet').rename(columns={'stack_pred':'v5'})
te=pd.read_parquet(f'{P}/ens_test_probs_v4.parquet')[['contact_id','p_pp','p_gr']]
t=t4.merge(t5,on='contact_id').merge(te,on='contact_id')
isg=t.contact_id.str.endswith('_G').values
t['ens']=np.where(isg,t.p_gr.values,t.p_pp.values)
t['stack_pred']=(wv4*logit(t.v4.values)+wv5*logit(t.v5.values)+we*logit(t.ens.values)).astype(np.float32)
t[['contact_id','stack_pred']].to_parquet(f'{W}/test_stack_mix.parquet',index=False)
json.dump({'wv4':wv4,'wv5':wv5,'wens':we},open(f'{W}/mix_weights.json','w'))
print('saved mix',o.shape,t.shape)
