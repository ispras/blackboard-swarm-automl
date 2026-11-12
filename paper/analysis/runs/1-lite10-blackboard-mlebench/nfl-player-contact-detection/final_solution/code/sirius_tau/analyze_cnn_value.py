import numpy as np, pandas as pd, sys
from sklearn.metrics import roc_auc_score
sys.path.insert(0,'workspaces/sirius_tau')
from lib_common import best_thr

FEAT='workspaces/fomalhaut_omicron/cache/feat_train.parquet'
idx=pd.read_parquet('workspaces/sirius_tau/crops_index.parquet')
idx['key']=idx.game_play+'_'+idx.step.astype(str)+'_'+idx.nfl_player_id_1.astype(str)+'_'+idx.nfl_player_id_2.astype(str)
f=pd.read_parquet(FEAT,columns=['game_play','step','nfl_player_id_1','nfl_player_id_2'])
f['key']=f.game_play+'_'+f.step.astype(str)+'_'+f.nfl_player_id_1.astype(str)+'_'+f.nfl_player_id_2.astype(str)
oof=np.load('workspaces/sirius_tau/oof_hel.npy')
f['g']=oof
m=f.drop_duplicates('key').set_index('key')['g']
idx['g']=idx.key.map(m)
cnn=np.load('workspaces/sirius_tau/cnn_oof.npy')
idx['cnn']=cnn
val=idx[idx.cnn!=0].copy()
y=val.contact.values
print('val n',len(val),'pos',int(y.sum()))
print('gbdt AUC',roc_auc_score(y,val.g))
print('cnn  AUC',roc_auc_score(y,val.cnn))
print('avg  AUC',roc_auc_score(y,val.g+val.cnn))
print('rank avg AUC',roc_auc_score(y,val.g.rank(pct=True)+val.cnn.rank(pct=True)))
for name,p in [('gbdt',val.g.values),('cnn',val.cnn.values)]:
    t,mcc=best_thr(y,p); print(name,'subset MCC',round(mcc,4),'thr',round(t,4))
comb=val.g.values+val.cnn.values
t,mcc=best_thr(y,comb); print('avg subset MCC',round(mcc,4),'thr',round(t,4))
