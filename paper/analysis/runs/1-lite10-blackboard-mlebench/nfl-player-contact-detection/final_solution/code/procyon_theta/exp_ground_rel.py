import os, time
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import GroupKFold
HERE=os.path.dirname(os.path.abspath(__file__)); CACHE=os.path.join(HERE,'cache')
META={'contact_id','game_play','datetime','nfl_player_id_1','nfl_player_id_2','contact','p1','p2','game_key'}

def best_mcc(y,p):
    order=np.argsort(-p); ys=y[order].astype(np.float64); ps=p[order]
    tp=np.cumsum(ys); fp=np.cumsum(1.0-ys); P=float(y.sum()); N=float(len(y))-P; fn=P-tp; tn=N-fp
    den=np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    mcc=np.where(den>0,(tp*tn-fp*fn)/np.where(den>0,den,1.0),0.0); i=int(np.argmax(mcc)); return float(mcc[i]),float(ps[i])

def main():
    d=pd.read_parquet(os.path.join(CACHE,'featg_all_train.parquet'))
    d=d[d.p2=='G'].reset_index(drop=True)
    base0=[c for c in d.columns if c not in META]
    rp=pd.read_parquet(os.path.join(CACHE,'relpose_train.parquet'))
    d['player']=d.p1.astype(np.int64)
    d=d.merge(rp,on=['game_play','step','player'],how='left')
    rel=[c for c in rp.columns if c not in ('game_play','step','player')]
    base=base0
    y=d.contact.values.astype(np.int8); groups=d.game_play.values
    params=dict(objective='binary',learning_rate=0.04,num_leaves=128,min_data_in_leaf=80,
                feature_fraction=0.8,bagging_fraction=0.8,bagging_freq=1,lambda_l2=1.0,
                n_jobs=16,seed=5,verbose=-1,metric='auc')
    for name,feats in [('base',base),('base+rel',base+rel)]:
        oof=np.zeros(len(d)); gkf=GroupKFold(n_splits=5); t0=time.time()
        for k,(tri,vai) in enumerate(gkf.split(d,y,groups)):
            dtr=lgb.Dataset(d.iloc[tri][feats],y[tri]); dva=lgb.Dataset(d.iloc[vai][feats],y[vai])
            m=lgb.train(params,dtr,num_boost_round=2000,valid_sets=[dva],callbacks=[lgb.early_stopping(80,verbose=False)])
            oof[vai]=m.predict(d.iloc[vai][feats],num_iteration=m.best_iteration)
        mm,tt=best_mcc(y,oof); print('%s ground raw MCC %.5f thr %.4f t=%d'%(name,mm,tt,time.time()-t0),flush=True)
        if name=='base+rel':
            b=lgb.Booster(model_file=None) if False else m
        imp=pd.Series(m.feature_importance('gain'),index=feats).sort_values(ascending=False)
        print(imp.head(12).to_string(),flush=True)

if __name__=='__main__': main()
