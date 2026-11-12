import numpy as np, pandas as pd, sys
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import lightgbm as lgb
FEAT='workspaces/fomalhaut_omicron/cache/feat_train.parquet'
gi=pd.read_parquet('workspaces/sirius_tau/ground_index.parquet')
gi['key']=gi.game_play+'_'+gi.step.astype(str)+'_'+gi.nfl_player_id_1.astype(str)
f=pd.read_parquet(FEAT,columns=['game_play','step','nfl_player_id_1','nfl_player_id_2'])
f=f[f.nfl_player_id_2.astype(str)=='G'].copy()
f['key']=f.game_play+'_'+f.step.astype(str)+'_'+f.nfl_player_id_1.astype(str)
f['g']=np.load('workspaces/sirius_tau/oof_hel.npy')[f.index.values]
m=f.drop_duplicates('key').set_index('key')['g']
gi['g']=gi.key.map(m)
gi['cnn']=np.load('workspaces/sirius_tau/cnn_ground_oof.npy')
val=gi[gi.cnn!=0].reset_index(drop=True)
y=val.contact.values; G=val.g.values.astype(float); C=val.cnn.values.astype(float)
def cv_auc(X):
    oof=np.zeros(len(y)); kf=KFold(5,shuffle=True,random_state=0)
    for tri,vai in kf.split(X):
        d=lgb.Dataset(X[tri],y[tri]); mm=lgb.train(dict(objective='binary',verbose=-1,num_leaves=31,learning_rate=0.05,min_data_in_leaf=50,num_threads=4),d,120)
        oof[vai]=mm.predict(X[vai])
    return roc_auc_score(y,oof)
print('n',len(y),'pos',int(y.sum()))
print('AUC g only   ', cv_auc(np.c_[G]),flush=True)
print('AUC g+cnn    ', cv_auc(np.c_[G,C]),flush=True)
print('AUC cnn only ', cv_auc(np.c_[C]),flush=True)
print('AUC g+dist   ', cv_auc(np.c_[G, np.zeros_like(G)]))
