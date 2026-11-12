import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score, matthews_corrcoef
import sys; sys.path.insert(0,'workspaces/sirius_tau')
from lib_common import best_thr
FTR='workspaces/procyon_theta/cache/featg_all_train.parquet'
k=pd.read_parquet(FTR,columns=['contact_id','contact','nfl_player_id_2'])
k['y']=k.contact.values
mine=np.load('workspaces/sirius_tau/stack_oof_nocnn3.npy')
p=pd.read_parquet('workspaces/procyon_theta/oof_pred.parquet')
m=k.merge(p,on='contact_id',how='left')
m['mine']=mine
print('rows',len(m),'procyon nan',m.prob.isna().sum())
y=m.y.values
print('AUC mine',roc_auc_score(y,m.mine),'AUC procyon',roc_auc_score(y,m.prob))
for name,col in [('mine','mine'),('procyon','prob')]:
    t,mc=best_thr(y,m[col].values); print(name,'MCC',round(mc,4),'thr',round(t,4),'posrate',round((m[col]>=t).mean(),4))
g=(m.nfl_player_id_2.astype(str)!='G').values
for name,col in [('mine','mine'),('procyon','prob')]:
    print(name,'pair AUC',round(roc_auc_score(y[g],m[col].values[g]),4),'ground AUC',round(roc_auc_score(y[~g],m[col].values[~g]),4))
