import pandas as pd, numpy as np, json
def logit(p,e=1e-6): p=np.clip(np.asarray(p,dtype=float),e,1-e); return np.log(p/(1-p))
A=0.15
o=pd.read_parquet('workspaces/capella_pi/oof_stack_mix.parquet')
g5=pd.read_parquet('workspaces/canopus_pi/preds/oof_ground_gru_v5.parquet')[['contact_id','pred']].rename(columns={'pred':'g5'})
o=o.merge(g5,on='contact_id',how='left')
ig=o.is_ground.values==1
o['stack_oof']=o.stack_oof.values+A*np.where(ig,logit(o.g5.fillna(1e-5).values),0.0)
o[['contact_id','fold','contact','is_ground','stack_oof']].to_parquet('workspaces/capella_pi/oof_stack_mixv2.parquet',index=False)
t=pd.read_parquet('workspaces/capella_pi/test_stack_mix.parquet')
tg5=pd.read_parquet('workspaces/canopus_pi/preds/test_ground_gru_v5.parquet')[['contact_id','pred']].rename(columns={'pred':'g5'})
t=t.merge(tg5,on='contact_id',how='left')
igt=t.contact_id.str.endswith('_G').values
t['stack_pred']=t.stack_pred.values+A*np.where(igt,logit(t.g5.fillna(1e-5).values),0.0)
t[['contact_id','stack_pred']].to_parquet('workspaces/capella_pi/test_stack_mixv2.parquet',index=False)
json.dump({'alpha_g5':A},open('workspaces/capella_pi/mixv2_weights.json','w'))
print('saved mixv2',o.shape,t.shape)
