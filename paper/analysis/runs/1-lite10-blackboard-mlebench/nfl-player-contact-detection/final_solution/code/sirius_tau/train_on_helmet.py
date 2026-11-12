import pandas as pd, numpy as np, lightgbm as lgb, os, sys, json
from sklearn.model_selection import GroupKFold
sys.path.insert(0,'workspaces/sirius_tau')
from lib_common import best_thr

OUT='workspaces/sirius_tau'
FT='workspaces/fomalhaut_omicron/cache/feat_train.parquet'
FS='workspaces/fomalhaut_omicron/cache/feat_test.parquet'

drop = ['contact_id','game_play','datetime','step','nfl_player_id_1','nfl_player_id_2','contact','p1','p2']

def main():
    df = pd.read_parquet(FT)
    test = pd.read_parquet(FS)
    feats=[c for c in df.columns if c not in drop]
    y=df.contact.values.astype(int); groups=df.game_play.values
    X=df[feats]
    params=dict(objective='binary',learning_rate=0.05,num_leaves=255,min_data_in_leaf=100,
                feature_fraction=0.8,bagging_fraction=0.8,bagging_freq=1,lambda_l2=1.0,
                max_bin=255,verbose=-1,num_threads=32)
    oof=np.zeros(len(df)); bi=[]
    for i,(tri,vai) in enumerate(GroupKFold(n_splits=5).split(X,y,groups)):
        dtr=lgb.Dataset(X.iloc[tri],y[tri]); dva=lgb.Dataset(X.iloc[vai],y[vai])
        m=lgb.train(params,dtr,num_boost_round=4000,valid_sets=[dva],
                    callbacks=[lgb.early_stopping(150,verbose=False),lgb.log_evaluation(300)])
        oof[vai]=m.predict(X.iloc[vai],num_iteration=m.best_iteration)
        bi.append(m.best_iteration)
        _,fv=best_thr(y[vai],oof[vai])
        print(f'fold {i} it {m.best_iteration} mcc {fv:.4f}',flush=True)
    thr,mcc=best_thr(y,oof)
    print(f'HELMET OOF thr {thr:.4f} MCC {mcc:.5f}',flush=True)
    np.save(f'{OUT}/oof_hel.npy',oof)
    it=int(np.mean(bi)*1.1)+1
    mfull=lgb.train(params,lgb.Dataset(X,y),num_boost_round=it)
    pt=mfull.predict(test[feats]); np.save(f'{OUT}/test_hel.npy',pt)
    sub=pd.read_csv('data/sample_submission.csv'); sub['contact']=(pt>=thr).astype(int)
    sub.to_csv(f'{OUT}/sub_hel.csv',index=False)
    json.dump({'oof_mcc':float(mcc),'thr':float(thr),'iters':it},open(f'{OUT}/hel_meta.json','w'))
    print('done pos',sub.contact.mean(),'iters',it)

if __name__=='__main__': main()
