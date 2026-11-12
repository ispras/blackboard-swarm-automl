import numpy as np, pandas as pd, sys
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
sys.path.insert(0,'workspaces/sirius_tau')
FEAT='workspaces/fomalhaut_omicron/cache/feat_train.parquet'
idx=pd.read_parquet('workspaces/sirius_tau/crops_index.parquet')
idx['key']=idx.game_play+'_'+idx.step.astype(str)+'_'+idx.nfl_player_id_1.astype(str)+'_'+idx.nfl_player_id_2.astype(str)
f=pd.read_parquet(FEAT,columns=['game_play','step','nfl_player_id_1','nfl_player_id_2'])
f['key']=f.game_play+'_'+f.step.astype(str)+'_'+f.nfl_player_id_1.astype(str)+'_'+f.nfl_player_id_2.astype(str)
f['g']=np.load('workspaces/sirius_tau/oof_hel.npy')
m=f.drop_duplicates('key').set_index('key')['g']
idx['g']=idx.key.map(m)
idx['cnn']=np.load('workspaces/sirius_tau/cnn_oof.npy')
val=idx[idx.cnn!=0].copy()
g=val.g.values.astype(float); c=val.cnn.values.astype(float); y=val.contact.values
# split val in half by play to avoid leakage
plays=val.game_play.values
rng=np.random.RandomState(7); up=np.array(sorted(set(plays))); rng.shuffle(up)
A=set(up[:len(up)//2]); va_mask=np.array([p in A for p in plays])
def auc(yy,pp): return roc_auc_score(yy,pp)
# baseline g on held half
print('g AUC on B', auc(y[~va_mask],g[~va_mask]))
# fit logistic on A
from sklearn.preprocessing import StandardScaler
Xa=np.c_[g[va_mask],c[va_mask]]; Xb=np.c_[g[~va_mask],c[~va_mask]]
for combo,names in [((0,),( 'g',)),((0,1),('g','cnn'))]:
    lr=LogisticRegression(max_iter=1000).fit(Xa[:,list(combo)],y[va_mask])
    pb=lr.predict_proba(Xb[:,list(combo)])[:,1]
    print(names,'logreg AUC on B',auc(y[~va_mask],pb))
# lgbm combo
import lightgbm as lgb
d=lgb.Dataset(Xa,y[va_mask]); m2=lgb.train(dict(objective='binary',verbose=-1,num_leaves=31,learning_rate=0.05),d,50)
pb=m2.predict(Xb); print('lgb g+cnn AUC on B',auc(y[~va_mask],pb))
d1=lgb.Dataset(Xa[:,:1],y[va_mask]); m1=lgb.train(dict(objective='binary',verbose=-1,num_leaves=31,learning_rate=0.05),d1,50)
print('lgb g only AUC on B',auc(y[~va_mask],m1.predict(Xb[:,:1])))
