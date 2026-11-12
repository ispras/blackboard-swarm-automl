import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
OUT='workspaces/sirius_tau'
gi=pd.read_parquet(f'{OUT}/ground_all_index.parquet'); y=gi.contact.values
v1=np.load(f'{OUT}/cnn_ground_oof_all.npy'); v2=np.load(f'{OUT}/cnn_ground_oof_all_v2.npy')
avg=(v1+v2)/2
# per-fold rank from saved full-index fold assignment (recompute)
from sklearn.model_selection import GroupKFold
fold=np.zeros(len(gi),int)
for k,(tri,vai) in enumerate(GroupKFold(5).split(gi, y, gi.game_play.values)): fold[vai]=k
for nm,v in [('v1',v1),('v2',v2),('avg',avg)]:
    r=np.zeros_like(v)
    for k in range(5):
        m=fold==k; r[m]=rankdata(v[m])/m.sum()
    print(nm,'rawAUC',round(roc_auc_score(y,v),5),'foldrankAUC',round(roc_auc_score(y,r),5))
np.save(f'{OUT}/cnn_ground_oof_all_avg.npy',avg)
