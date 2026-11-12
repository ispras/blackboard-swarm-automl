import pandas as pd, numpy as np, lightgbm as lgb, os, sys, time
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
import pyarrow.parquet as pq
sys.path.insert(0,'workspaces/sirius_tau')
from lib_common import best_thr
OUT='workspaces/sirius_tau'
FTR='workspaces/procyon_theta/cache/featg_all_train.parquet'
INC=int(os.environ.get('INC','1')); ROUNDS=int(os.environ.get('ROUNDS','1500'))
LR=float(os.environ.get('LR','0.05')); LEAVES=int(os.environ.get('LEAVES','128'))

def get_fold_map():
    full=pd.read_parquet(f'{OUT}/ground_all_index.parquet'); fmap={}
    for k,(tri,vai) in enumerate(GroupKFold(n_splits=5).split(full, full.contact.values, full.game_play.values)):
        for p in np.unique(full.game_play.values[vai]): fmap[p]=k
    return fmap

schema=pq.read_schema(FTR).names
drop=['game_play','datetime','step','nfl_player_id_1','nfl_player_id_2','contact','p1','p2','contact_id']
feats=[c for c in schema if c not in drop]
cols=['contact_id','contact','game_play','step','nfl_player_id_1','nfl_player_id_2']+feats
df=pd.read_parquet(FTR,columns=cols)
df=df[df.nfl_player_id_2.astype(str)=='G'].reset_index(drop=True)
if INC:
    gi=pd.read_parquet(f'{OUT}/ground_all_index.parquet'); oof=np.load(f'{OUT}/cnn_ground_oof_all.npy')
    km={f"{a}_{b}_{c}_G":v for a,b,c,v in zip(gi.game_play,gi.step,gi.nfl_player_id_1,oof)}
    df['cnn_g']=df.contact_id.map(km).astype(np.float32); feats=feats+['cnn_g']
for c in feats:
    if df[c].dtype==object: df[c]=df[c].astype('category')
fmap=get_fold_map(); fold=np.array([fmap[p] for p in df.game_play.values])
y=df.contact.values.astype(int); X=df[feats]
params=dict(objective='binary',learning_rate=LR,num_leaves=LEAVES,min_data_in_leaf=60,
            feature_fraction=0.8,bagging_fraction=0.8,bagging_freq=1,lambda_l2=1.0,
            max_bin=255,metric='auc',verbose=-1,num_threads=24)
oof=np.zeros(len(X))
for k in range(5):
    vai=np.where(fold==k)[0]; tri=np.where(fold!=k)[0]
    tp=tri[y[tri]==1]; tn=tri[y[tri]==0]; tn=np.random.RandomState(1000+k).choice(tn,int(len(tn)*0.5),replace=False)
    tr=np.r_[tp,tn]
    m=lgb.train(params,lgb.Dataset(X.iloc[tr],y[tr]),num_boost_round=ROUNDS,
                valid_sets=[lgb.Dataset(X.iloc[vai],y[vai])],
                callbacks=[lgb.early_stopping(150,verbose=False),lgb.log_evaluation(300)])
    oof[vai]=m.predict(X.iloc[vai],num_iteration=m.best_iteration)
    print(f'fold {k} it {m.best_iteration} valAUC {roc_auc_score(y[vai],oof[vai]):.4f}',flush=True)
from scipy.stats import rankdata
oofr=np.zeros_like(oof)
for k in range(5):
    vai=np.where(fold==k)[0]; oofr[vai]=rankdata(oof[vai])/len(vai)
print('POOLED raw AUC',roc_auc_score(y,oof),flush=True)
print('POOLED fold-rank AUC',roc_auc_score(y,oofr),flush=True)
np.save(f'{OUT}/tg_oof_inc{INC}.npy',oof); np.save(f'{OUT}/tg_fold.npy',fold)
