import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import lightgbm as lgb
FEAT='workspaces/fomalhaut_omicron/cache/feat_train.parquet'
ci=pd.read_parquet('workspaces/sirius_tau/crops_index.parquet')
ci['key']=ci.game_play+'_'+ci.step.astype(str)+'_'+ci.nfl_player_id_1.astype(str)+'_'+ci.nfl_player_id_2.astype(str)
f=pd.read_parquet(FEAT,columns=['contact_id','game_play','step','nfl_player_id_1','nfl_player_id_2'])
f['key']=f.game_play+'_'+f.step.astype(str)+'_'+f.nfl_player_id_1.astype(str)+'_'+f.nfl_player_id_2.astype(str)
f['g']=np.load('workspaces/sirius_tau/oof_hel.npy')
m=f.drop_duplicates('key').set_index('key')['g']
ci['g']=ci.key.map(m)
ci['cnn']=np.load('workspaces/sirius_tau/cnn_oof.npy')
val=ci[ci.cnn!=0].reset_index(drop=True)
y=val.contact.values; G=val.g.values.astype(float); C=val.cnn.values.astype(float)
def cv(X):
    oof=np.zeros(len(y)); kf=KFold(5,shuffle=True,random_state=0)
    for tri,vai in kf.split(X):
        d=lgb.Dataset(X[tri],y[tri]); mm=lgb.train(dict(objective='binary',verbose=-1,num_leaves=31,learning_rate=0.05,min_data_in_leaf=50,num_threads=4),d,120)
        oof[vai]=mm.predict(X[vai])
    return roc_auc_score(y,oof)
print('n',len(y),'pos',int(y.sum()))
print('g only',cv(np.c_[G]),flush=True)
print('g+cnn ',cv(np.c_[G,C]),flush=True)
print('cnn    ',cv(np.c_[C]),flush=True)
