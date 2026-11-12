import pandas as pd, numpy as np, lightgbm as lgb, os, sys, json, time
from sklearn.model_selection import GroupKFold
import pyarrow.parquet as pq
sys.path.insert(0,'workspaces/sirius_tau')
from lib_common import best_thr

OUT='workspaces/sirius_tau'
FTR='workspaces/procyon_theta/cache/featg_all_train.parquet'
FTE='workspaces/procyon_theta/cache/featg_all_test.parquet'
INC=int(os.environ.get('INC','1'))
TAG=os.environ.get('TAG','cnn')
NEG_FRAC=float(os.environ.get('NEG_FRAC','0.35'))
ROUNDS=int(os.environ.get('ROUNDS','1500'))
LR=float(os.environ.get('LR','0.05'))
LEAVES=int(os.environ.get('LEAVES','128'))
MDL=int(os.environ.get('MDL','100'))
SAVE=int(os.environ.get('SAVE','0'))

def cnn_maps():
    gi=pd.read_parquet(f'{OUT}/ground_all_index.parquet')
    oof=np.load(f'{OUT}/cnn_ground_oof_all.npy')
    km={f"{a}_{b}_{c}_G":v for a,b,c,v in zip(gi.game_play,gi.step,gi.nfl_player_id_1,oof)}
    ti=pd.read_parquet(f'{OUT}/ground_test_index.parquet')
    tp=np.load(f'{OUT}/cnn_ground_test_all.npy')
    kt={f"{a}_{b}_{c}_G":v for a,b,c,v in zip(ti.game_play,ti.step,ti.nfl_player_id_1,tp)}
    return km,kt

def smooth(keys):
    keys=keys.sort_values(['game_play','nfl_player_id_1','nfl_player_id_2','step']).reset_index(drop=True)
    g=keys.groupby(['game_play','nfl_player_id_1','nfl_player_id_2'],sort=False)
    return g['p'].transform(lambda x:x.rolling(7,center=True,min_periods=1).mean()).values

def main():
    t0=time.time()
    schema=pq.read_schema(FTR).names
    drop=['game_play','datetime','step','nfl_player_id_1','nfl_player_id_2','contact','p1','p2','contact_id']
    feats=[c for c in schema if c not in drop]
    cols=['contact_id','contact','game_play','step','nfl_player_id_1','nfl_player_id_2']+feats
    df=pd.read_parquet(FTR,columns=cols)
    print('loaded',df.shape,'t',round(time.time()-t0),flush=True)
    if INC:
        km,kt=cnn_maps()
        df['cnn_g']=df.contact_id.map(km).astype(np.float32)
        feats=feats+['cnn_g']
        print('cnn_g coverage',float(df.cnn_g.notna().mean()),flush=True)
    y=df.contact.values.astype(int); groups=df.game_play.values
    for c in feats:
        if df[c].dtype==object: df[c]=df[c].astype('category')
    X=df[feats]
    params=dict(objective='binary',learning_rate=LR,num_leaves=LEAVES,min_data_in_leaf=MDL,
                feature_fraction=0.8,bagging_fraction=0.8,bagging_freq=1,lambda_l2=1.0,
                max_bin=255,metric='auc',verbose=-1,num_threads=24)
    gkf=GroupKFold(n_splits=5); oof=np.zeros(len(df))
    for k,(tri,vai) in enumerate(gkf.split(X,y,groups)):
        tri_pos=tri[y[tri]==1]; tri_neg=tri[y[tri]==0]
        tri_neg=np.random.RandomState(k).choice(tri_neg,int(len(tri_neg)*NEG_FRAC),replace=False)
        tr=np.r_[tri_pos,tri_neg]
        m=lgb.train(params,lgb.Dataset(X.iloc[tr],y[tr]),num_boost_round=ROUNDS,
                    valid_sets=[lgb.Dataset(X.iloc[vai],y[vai])],
                    callbacks=[lgb.early_stopping(200,verbose=False),lgb.log_evaluation(400)])
        oof[vai]=m.predict(X.iloc[vai],num_iteration=m.best_iteration)
        print(f'fold {k} it {m.best_iteration} t {time.time()-t0:.0f}',flush=True)
    np.save(f'{OUT}/stack_oof_{TAG}.npy',oof)
    thr,mcc=best_thr(y,oof); print(f'RAW OOF MCC {mcc:.5f} thr {thr:.4f}',flush=True)
    keys=df[['game_play','step','nfl_player_id_1','nfl_player_id_2']].copy(); keys['p']=oof
    sm=smooth(keys); exp=keys.sort_values(['game_play','nfl_player_id_1','nfl_player_id_2','step']).index.values
    # sm is in sorted order; map back
    oof_sm=np.zeros(len(df)); oof_sm[exp]=sm
    thr2,mcc2=best_thr(y,oof_sm); print(f'SMOOTH OOF MCC {mcc2:.5f} thr {thr2:.4f}',flush=True)
    json.dump({'raw_mcc':float(mcc),'sm_mcc':float(mcc2),'thr':float(thr2),'inc':INC,'tag':TAG},
              open(f'{OUT}/stack_meta_{TAG}.json','w'))
    if SAVE:
        df[['contact_id']].assign(prob=oof).to_parquet(f'{OUT}/oof_pred_{TAG}.parquet')
        # full model for test
        tr_pos=np.where(y==1)[0]; tr_neg=np.where(y==0)[0]
        tr_neg=np.random.RandomState(0).choice(tr_neg,int(len(tr_neg)*NEG_FRAC),replace=False)
        tr=np.r_[tr_pos,tr_neg]
        ters=df[feats].dtypes.to_dict()
        tcols=['contact_id','game_play','step','nfl_player_id_1','nfl_player_id_2']+feats
        te=pd.read_parquet(FTE,columns=[c for c in tcols if c in pq.read_schema(FTE).names])
        for c in feats:
            if c in te and te[c].dtype==object: te[c]=te[c].astype('category')
        if INC:
            km,kt=cnn_maps(); te['cnn_g']=te.contact_id.map(kt).astype(np.float32)
        mfull=lgb.train(params,lgb.Dataset(X.iloc[tr],y[tr]),num_boost_round=1200)
        pt=mfull.predict(te[feats])
        te[['contact_id']].assign(prob=pt).to_parquet(f'{OUT}/test_pred_{TAG}.parquet')
        keys=te[['game_play','step','nfl_player_id_1','nfl_player_id_2']].copy(); keys['p']=pt
        tsm=smooth(keys); texp=keys.sort_values(['game_play','nfl_player_id_1','nfl_player_id_2','step']).index.values
        pt_sm=np.zeros(len(te)); pt_sm[texp]=tsm
        sub=pd.read_csv('data/sample_submission.csv')
        pm=dict(zip(te.contact_id.values,pt_sm))
        sub['p']=sub.contact_id.map(pm); sub['contact']=(sub.p>=thr2).astype(int)
        sub[['contact_id','contact']].to_csv(f'{OUT}/sub_stack_{TAG}.csv',index=False)
        print('saved test pred + sub, posrate',sub.contact.mean(),flush=True)
    print('done',round(time.time()-t0),flush=True)

if __name__=='__main__': main()
