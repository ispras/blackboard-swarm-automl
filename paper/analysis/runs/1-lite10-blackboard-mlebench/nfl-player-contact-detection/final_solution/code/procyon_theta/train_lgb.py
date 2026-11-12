import os, time, json
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import matthews_corrcoef, roc_auc_score

CACHE=os.path.join(os.path.dirname(__file__),'cache')

def load():
    tr=pd.read_parquet(os.path.join(CACHE,'train_track_feat.parquet'))
    te=pd.read_parquet(os.path.join(CACHE,'test_track_feat.parquet'))
    return tr,te

def feat_cols(df):
    drop={'game_play','step','nfl_player_id_1','nfl_player_id_2','contact'}
    cols=[c for c in df.columns if c not in drop and df[c].dtype!=object]
    return cols

def best_thr(y,p):
    idx=np.argsort(-p); ys=y[idx].astype(np.float64); ps=p[idx]
    P=float(ys.sum()); N=float(len(ys)-P)
    tp=np.cumsum(ys); fp=np.cumsum(1-ys)
    fn=P-tp; tn=N-fp
    den=np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    num=(tp*tn-fp*fn)
    mcc=np.divide(num,den,out=np.zeros_like(num),where=den>0)
    k=int(np.nanargmax(mcc))
    return float(mcc[k]), float(ps[k])

def main():
    t0=time.time()
    tr,te=load()
    cols=feat_cols(tr)
    print('n features',len(cols))
    tr['game_key']=tr.game_play.str.split('_').str[0]
    y=tr['contact'].values
    g=tr['game_key'].values
    gkf=GroupKFold(n_splits=5)
    oof=np.zeros(len(tr))
    params=dict(objective='binary',learning_rate=0.03,num_leaves=64,
                min_data_in_leaf=100,feature_fraction=0.8,bagging_fraction=0.8,
                bagging_freq=1,lambda_l2=1.0,metric='auc',verbose=-1,
                num_threads=8)
    te_pred=np.zeros(len(te))
    for i,(tri,vai) in enumerate(gkf.split(tr, y, g)):
        dtr=lgb.Dataset(tr.iloc[tri][cols],label=y[tri])
        dva=lgb.Dataset(tr.iloc[vai][cols],label=y[vai])
        m=lgb.train(params,dtr,num_boost_round=2000,valid_sets=[dva],
                    callbacks=[lgb.early_stopping(100,verbose=False)])
        oof[vai]=m.predict(tr.iloc[vai][cols])
        te_pred+=m.predict(te[cols])/5
        print('fold',i,'best_iter',m.best_iteration,'auc',roc_auc_score(y[vai],oof[vai]),'t',round(time.time()-t0))
    auc=roc_auc_score(y,oof)
    mcc05=matthews_corrcoef(y,(oof>=0.5).astype(int))
    bm,bt=best_thr(y,oof)
    print(f'AUC {auc:.5f} MCC@0.5 {mcc05:.5f} bestMCC {bm:.5f} @thr {bt:.4f}')
    np.save(os.path.join(CACHE,'oof_lgb.npy'),oof)
    np.save(os.path.join(CACHE,'te_lgb.npy'),te_pred)
    json.dump({'auc':float(auc),'mcc05':float(mcc05),'best_mcc':float(bm),'best_thr':float(bt),'cols':cols},
              open(os.path.join(CACHE,'lgb_result.json'),'w'))
    # importance
    imp=pd.Series(m.feature_importance('gain'),index=cols).sort_values(ascending=False)
    print(imp.head(25).to_string())

if __name__=='__main__':
    main()
