import pandas as pd, numpy as np, sys
from sklearn.metrics import roc_auc_score
sys.path.insert(0,'workspaces/sirius_tau'); from lib_common import best_thr
k=pd.read_parquet('workspaces/procyon_theta/cache/featg_all_train.parquet',columns=['contact_id','contact','nfl_player_id_2'])
k['y']=k.contact.values
mine=pd.read_parquet('workspaces/sirius_tau/oof_pred.parquet')
p=pd.read_parquet('workspaces/procyon_theta/oof_pred.parquet')
m=k.merge(mine,on='contact_id',how='left').rename(columns={'prob':'mine'}).merge(p,on='contact_id',how='left')
y=m.y.values; g=(m.nfl_player_id_2.astype(str)!='G').values
print('AUC mine',round(roc_auc_score(y,m['mine']),5),'procyon',round(roc_auc_score(y,m['prob']),5))
for nm,c in [('mine','mine'),('procyon','prob')]:
    t,mc=best_thr(y,m[c].values); print(nm,'MCC',round(mc,4),'pairAUC',round(roc_auc_score(y[g],m[c].values[g]),4),'gndAUC',round(roc_auc_score(y[~g],m[c].values[~g]),4))
