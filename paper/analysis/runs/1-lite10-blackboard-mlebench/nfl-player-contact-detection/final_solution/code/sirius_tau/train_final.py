import pandas as pd, numpy as np, lightgbm as lgb, os, sys, json, time
from sklearn.model_selection import GroupKFold
from scipy.stats import rankdata
import pyarrow.parquet as pq
sys.path.insert(0,'workspaces/sirius_tau')
from lib_common import best_thr

OUT='workspaces/sirius_tau'
FTR='workspaces/procyon_theta/cache/featg_all_train.parquet'
FTE='workspaces/procyon_theta/cache/featg_all_test.parquet'
INC=int(os.environ.get('INC','1'))
TAG=os.environ.get('TAG','final')
NEG_FRAC=float(os.environ.get('NEG_FRAC','0.35'))
ROUNDS=int(os.environ.get('ROUNDS','1500'))
LR=float(os.environ.get('LR','0.05'))
LEAVES=int(os.environ.get('LEAVES','128'))
MDL=int(os.environ.get('MDL','60'))
SAVE=int(os.environ.get('SAVE','0'))
W=int(os.environ.get('W','7'))
t0=time.time()

CNNOOF=os.environ.get('CNNOOF',f'{OUT}/cnn_ground_oof_all.npy')
CNNTEST=os.environ.get('CNNTEST',f'{OUT}/cnn_ground_test_all.npy')
def cnn_maps():
    gi=pd.read_parquet(f'{OUT}/ground_all_index.parquet'); oof=np.load(CNNOOF)
    km={f"{a}_{b}_{c}_G":v for a,b,c,v in zip(gi.game_play,gi.step,gi.nfl_player_id_1,oof)}
    ti=pd.read_parquet(f'{OUT}/ground_test_index.parquet'); tp=np.load(CNNTEST)
    kt={f"{a}_{b}_{c}_G":v for a,b,c,v in zip(ti.game_play,ti.step,ti.nfl_player_id_1,tp)}
    return km,kt

def fold_map():
    full=pd.read_parquet(f'{OUT}/ground_all_index.parquet'); fm={}
    for k,(tri,vai) in enumerate(GroupKFold(n_splits=5).split(full, full.contact.values, full.game_play.values)):
        for p in np.unique(full.game_play.values[vai]): fm[p]=k
    return fm

def smooth(prob, keys):
    k=keys.copy(); k['p']=prob
    k=k.sort_values(['game_play','nfl_player_id_1','nfl_player_id_2','step'])
    sm=k.groupby(['game_play','nfl_player_id_1','nfl_player_id_2'],sort=False)['p'].transform(
        lambda x:x.rolling(W,center=True,min_periods=1).mean()).values
    idx=k.index.values
    out=np.zeros(len(prob)); out[idx]=sm
    return out

schema=pq.read_schema(FTR).names
drop=['game_play','datetime','step','nfl_player_id_1','nfl_player_id_2','contact','p1','p2','contact_id']
feats=[c for c in schema if c not in drop]
cols=['contact_id','contact','game_play','step','nfl_player_id_1','nfl_player_id_2']+feats
df=pd.read_parquet(FTR,columns=cols)
if INC:
    km,kt=cnn_maps(); df['cnn_g']=df.contact_id.map(km).astype(np.float32); feats=feats+['cnn_g']
for c in feats:
    if df[c].dtype==object: df[c]=df[c].astype('category')
y=df.contact.values.astype(int); groups=df.game_play.values
fm=fold_map(); fold_row=np.array([fm.get(p,-1) for p in groups])
print('loaded',df.shape,'cnn_cov',float(df.cnn_g.notna().mean()) if INC else 0,'t',round(time.time()-t0),flush=True)
X=df[feats]
params=dict(objective='binary',learning_rate=LR,num_leaves=LEAVES,min_data_in_leaf=MDL,
            feature_fraction=0.8,bagging_fraction=0.8,bagging_freq=1,lambda_l2=1.0,
            max_bin=255,metric='auc',verbose=-1,num_threads=24)
oof=np.zeros(len(df)); bi=[]
for k in range(5):
    vai=np.where(fold_row==k)[0]; tri=np.where(fold_row!=k)[0]
    tp=tri[y[tri]==1]; tn=tri[y[tri]==0]
    tn=np.random.RandomState(k).choice(tn,int(len(tn)*NEG_FRAC),replace=False)
    tr=np.r_[tp,tn]
    m=lgb.train(params,lgb.Dataset(X.iloc[tr],y[tr]),num_boost_round=ROUNDS,
                valid_sets=[lgb.Dataset(X.iloc[vai],y[vai])],
                callbacks=[lgb.early_stopping(150,verbose=False),lgb.log_evaluation(500)])
    oof[vai]=m.predict(X.iloc[vai],num_iteration=m.best_iteration); bi.append(m.best_iteration)
    print(f'fold {k} it {m.best_iteration} t {round(time.time()-t0)}',flush=True)
keys=df[['game_play','step','nfl_player_id_1','nfl_player_id_2']]
oof_sm=smooth(oof,keys)
thr,mcc=best_thr(y,oof_sm); q=float((oof_sm>=thr).mean())
thr0,mcc0=best_thr(y,oof)
print(f'OOF raw MCC {mcc0:.5f} | SM({W}) MCC {mcc:.5f} thr {thr:.4f} posrate {q:.4f}',flush=True)
np.save(f'{OUT}/final_oof_{TAG}.npy',oof_sm)
json.dump({'raw':float(mcc0),'sm':float(mcc),'thr':float(thr),'q':q,'inc':INC,'bi':bi},
          open(f'{OUT}/final_meta_{TAG}.json','w'))
if SAVE:
    df[['contact_id']].assign(prob=oof_sm).to_parquet(f'{OUT}/oof_pred_{TAG}.parquet')
    tp=np.where(y==1)[0]; tn=np.where(y==0)[0]
    tn=np.random.RandomState(0).choice(tn,int(len(tn)*NEG_FRAC),replace=False)
    tr=np.r_[tp,tn]
    it=int(np.mean(bi)*1.1)+1
    mfull=lgb.train(params,lgb.Dataset(X.iloc[tr],y[tr]),num_boost_round=it)
    teschema=pq.read_schema(FTE).names
    tcols=['contact_id','game_play','step','nfl_player_id_1','nfl_player_id_2']+[c for c in feats if c in teschema]
    te=pd.read_parquet(FTE,columns=tcols)
    if INC:
        km,kt=cnn_maps(); te['cnn_g']=te.contact_id.map(kt).astype(np.float32)
    for c in feats:
        if c in te and te[c].dtype==object: te[c]=te[c].astype('category')
    pt=mfull.predict(te[feats])
    tkeys=te[['game_play','step','nfl_player_id_1','nfl_player_id_2']]
    pt_sm=smooth(pt,tkeys)
    thr_test=float(np.quantile(pt_sm,1-q))
    te[['contact_id']].assign(prob=pt_sm).to_parquet(f'{OUT}/test_pred_{TAG}.parquet')
    sub=pd.read_csv('data/sample_submission.csv'); pm=dict(zip(te.contact_id.values,pt_sm))
    sub['p']=sub.contact_id.map(pm); sub['contact']=(sub.p>=thr_test).astype(int)
    sub[['contact_id','contact']].to_csv(f'{OUT}/sub_{TAG}.csv',index=False)
    print('SAVED sub posrate',sub.contact.mean(),'thr_test',thr_test,'iters',it,flush=True)
print('done',round(time.time()-t0),flush=True)
