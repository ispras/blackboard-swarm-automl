import numpy as np, pandas as pd, sys
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
sys.path.insert(0,'workspaces/sirius_tau')
FEAT='workspaces/fomalhaut_omicron/cache/feat_train.parquet'
gi=pd.read_parquet('workspaces/sirius_tau/ground_index.parquet')
gi['key']=gi.game_play+'_'+gi.step.astype(str)+'_'+gi.nfl_player_id_1.astype(str)
f=pd.read_parquet(FEAT,columns=['contact_id','game_play','step','nfl_player_id_1','nfl_player_id_2'])
f=f[f.nfl_player_id_2.astype(str)=='G']
f['key']=f.game_play+'_'+f.step.astype(str)+'_'+f.nfl_player_id_1.astype(str)
f['g']=np.load('workspaces/sirius_tau/oof_hel.npy')[f.index.values] if False else 0
# need aligned oof: oof_hel is aligned to feat row order
oof=np.load('workspaces/sirius_tau/oof_hel.npy')
f['g']=oof[f.index.values]
m=f.drop_duplicates('key').set_index('key')['g']
gi['g']=gi.key.map(m)
gi['cnn']=np.load('workspaces/sirius_tau/cnn_ground_oof.npy')
val=gi[gi.cnn!=0].copy()
y=val.contact.values; g=val.g.values.astype(float); c=val.cnn.values.astype(float)
print('val n',len(val),'pos',int(y.sum()))
print('gbdt AUC',roc_auc_score(y,g),'cnn AUC',roc_auc_score(y,c))
plays=val.game_play.values; up=np.array(sorted(set(plays))); np.random.RandomState(7).shuffle(up)
A=set(up[:len(up)//2]); mk=np.array([p in A for p in plays])
Xa=np.c_[g[mk],c[mk]]; Xb=np.c_[g[~mk],c[~mk]]; ya=y[mk]; yb=y[~mk]
print('gbdt AUC on B',roc_auc_score(yb,g[~mk]))
lr=LogisticRegression(max_iter=1000).fit(Xa,ya); print('logreg g+cnn AUC B',roc_auc_score(yb,lr.predict_proba(Xb)[:,1]))
d=lgb.Dataset(Xa,ya); m2=lgb.train(dict(objective='binary',verbose=-1,num_leaves=31,learning_rate=0.05),d,80)
print('lgb g+cnn AUC B',roc_auc_score(yb,m2.predict(Xb)))
