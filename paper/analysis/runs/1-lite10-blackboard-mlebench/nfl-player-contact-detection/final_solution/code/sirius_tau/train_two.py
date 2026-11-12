import pandas as pd, numpy as np, lightgbm as lgb, os, sys, json, time
from sklearn.model_selection import GroupKFold
import pyarrow.parquet as pq
sys.path.insert(0,'workspaces/sirius_tau')
from lib_common import best_thr

OUT='workspaces/sirius_tau'
FTR='workspaces/procyon_theta/cache/featg_all_train.parquet'
FTE='workspaces/procyon_theta/cache/featg_all_test.parquet'
INC=int(os.environ.get('INC','1'))
TAG=os.environ.get('TAG','two_cnn')
PAIR_NEG=float(os.environ.get('PAIR_NEG','0.3'))
GND_NEG=float(os.environ.get('GND_NEG','0.5'))
ROUNDS=int(os.environ.get('ROUNDS','2000'))
SAVE=int(os.environ.get('SAVE','0'))
LR=float(os.environ.get('LR','0.05'))
LEAVES=int(os.environ.get('LEAVES','128'))

def cnn_maps():
    gi=pd.read_parquet(f'{OUT}/ground_all_index.parquet'); oof=np.load(f'{OUT}/cnn_ground_oof_all.npy')
    km={f"{a}_{b}_{c}_G":v for a,b,c,v in zip(gi.game_play,gi.step,gi.nfl_player_id_1,oof)}
    ti=pd.read_parquet(f'{OUT}/ground_test_index.parquet'); tp=np.load(f'{OUT}/cnn_ground_test_all.npy')
    kt={f"{a}_{b}_{c}_G":v for a,b,c,v in zip(ti.game_play,ti.step,ti.nfl_player_id_1,tp)}
    return km,kt

def smooth_vals(keys):
    keys=keys.sort_values(['game_play','nfl_player_id_1','nfl_player_id_2','step'])
    sm=keys.groupby(['game_play','nfl_player_id_1','nfl_player_id_2'],sort=False)['p'].transform(
        lambda x:x.rolling(5,center=True,min_periods=1).mean()).values
    return keys.index.values, sm

def get_fold_map():
    full=pd.read_parquet(f'{OUT}/ground_all_index.parquet')
    fmap={}
    for k,(tri,vai) in enumerate(GroupKFold(n_splits=5).split(full, full.contact.values, full.game_play.values)):
        for p in np.unique(full.game_play.values[vai]): fmap[p]=k
    return fmap

def train_model(X,y,fold_of_row,neg_frac,seed):
    oof=np.zeros(len(X))
    params=dict(objective='binary',learning_rate=LR,num_leaves=LEAVES,min_data_in_leaf=60,
                feature_fraction=0.8,bagging_fraction=0.8,bagging_freq=1,lambda_l2=1.0,
                max_bin=255,metric='auc',verbose=-1,num_threads=24)
    for k in range(5):
        vai=np.where(fold_of_row==k)[0]; tri=np.where(fold_of_row!=k)[0]
        tp=tri[y[tri]==1]; tn=tri[y[tri]==0]
        tn=np.random.RandomState(seed+k).choice(tn,int(max(2000,len(tn)*neg_frac)),replace=False)
        tr=np.r_[tp,tn]
        m=lgb.train(params,lgb.Dataset(X.iloc[tr],y[tr]),num_boost_round=ROUNDS,
                    valid_sets=[lgb.Dataset(X.iloc[vai],y[vai])],
                    callbacks=[lgb.early_stopping(150,verbose=False),lgb.log_evaluation(500)])
        oof[vai]=m.predict(X.iloc[vai],num_iteration=m.best_iteration)
        print(f'  fold {k} it {m.best_iteration}',flush=True)
    return oof

def main():
    t0=time.time()
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
    isg=(df.nfl_player_id_2.astype(str)=='G').values
    print('loaded',df.shape,'t',round(time.time()-t0),flush=True)
    X=df[feats]
    fmap=get_fold_map()
    fold_row=np.array([fmap.get(p,-1) for p in groups])
    print('fold coverage',int((fold_row>=0).mean()),flush=True)
    print('== ground model ==',flush=True)
    gidx=np.where(isg)[0]
    Xg=X.iloc[gidx].reset_index(drop=True); yg=y[gidx]; fg=fold_row[gidx]
    oof_g=train_model(Xg,yg,fg,GND_NEG,1000)
    from sklearn.metrics import roc_auc_score
    print('ground AUC',roc_auc_score(yg,oof_g),flush=True)
    print('== pair model ==',flush=True)
    pidx=np.where(~isg)[0]
    Xp=X.iloc[pidx].reset_index(drop=True); yp=y[pidx]; fp=fold_row[pidx]
    oof_p=train_model(Xp,yp,fp,PAIR_NEG,0)
    print('pair AUC',roc_auc_score(yp,oof_p),flush=True)
    oof=np.zeros(len(df)); oof[gidx]=oof_g; oof[pidx]=oof_p
    np.save(f'{OUT}/two_oof_{TAG}.npy',oof)
    thr,mcc=best_thr(y,oof); print(f'RAW OOF MCC {mcc:.5f} thr {thr:.4f}',flush=True)
    keys=df[['game_play','step','nfl_player_id_1','nfl_player_id_2']].copy(); keys['p']=oof
    exp,sm=smooth_vals(keys); oof_sm=np.zeros(len(df)); oof_sm[exp]=sm
    thr2,mcc2=best_thr(y,oof_sm); print(f'SMOOTH OOF MCC {mcc2:.5f} thr {thr2:.4f}',flush=True)
    json.dump({'raw':float(mcc),'sm':float(mcc2),'thr':float(thr2),'inc':INC},open(f'{OUT}/two_meta_{TAG}.json','w'))
    if SAVE:
        df[['contact_id']].assign(prob=oof).to_parquet(f'{OUT}/oof_pred.parquet')
        params=dict(objective='binary',learning_rate=LR,num_leaves=LEAVES,min_data_in_leaf=60,
                    feature_fraction=0.8,bagging_fraction=0.8,bagging_freq=1,lambda_l2=1.0,
                    max_bin=255,metric='auc',verbose=-1,num_threads=24)
        def full(Xa,ya,neg,seed):
            tp=np.where(ya==1)[0]; tn=np.where(ya==0)[0]
            tn=np.random.RandomState(seed).choice(tn,int(max(2000,len(tn)*neg)),replace=False)
            tr=np.r_[tp,tn]
            return lgb.train(params,lgb.Dataset(Xa.iloc[tr],ya[tr]),num_boost_round=int(np.mean([0,ROUNDS])) if False else 1200)
        gm=full(Xg,yg,GND_NEG,123); pm=full(Xp,yp,PAIR_NEG,321)
        # test
        teschema=pq.read_schema(FTE).names
        tcols=['contact_id','game_play','step','nfl_player_id_1','nfl_player_id_2']+[c for c in feats if c in teschema or c=='cnn_g']
        te=pd.read_parquet(FTE,columns=[c for c in tcols if c in teschema])
        if INC:
            km,kt=cnn_maps(); te['cnn_g']=te.contact_id.map(kt).astype(np.float32)
        for c in feats:
            if c in te and te[c].dtype==object: te[c]=te[c].astype('category')
        teg=(te.nfl_player_id_2.astype(str)=='G').values
        pt=np.zeros(len(te))
        pt[teg]=gm.predict(te.loc[teg,feats])
        pt[~teg]=pm.predict(te.loc[~teg,feats])
        te[['contact_id']].assign(prob=pt).to_parquet(f'{OUT}/test_pred.parquet')
        keys=te[['game_play','step','nfl_player_id_1','nfl_player_id_2']].copy(); keys['p']=pt
        texp,tsm=smooth_vals(keys); pt_sm=np.zeros(len(te)); pt_sm[texp]=tsm
        sub=pd.read_csv('data/sample_submission.csv'); pmap=dict(zip(te.contact_id.values,pt_sm))
        sub['p']=sub.contact_id.map(pmap); sub['contact']=(sub.p>=thr2).astype(int)
        sub[['contact_id','contact']].to_csv(f'{OUT}/sub_two_{TAG}.csv',index=False)
        print('saved artifacts + sub, posrate',sub.contact.mean(),flush=True)
    print('done',round(time.time()-t0),flush=True)

if __name__=='__main__': main()
