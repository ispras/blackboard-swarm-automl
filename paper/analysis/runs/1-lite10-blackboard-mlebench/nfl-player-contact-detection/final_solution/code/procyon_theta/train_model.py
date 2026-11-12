import os, sys, time, json, gc, argparse
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

HERE=os.path.dirname(os.path.abspath(__file__))
CACHE=os.path.join(HERE,'cache')

META={'contact_id','game_play','datetime','nfl_player_id_1','nfl_player_id_2','contact','p1','p2','game_key'}
CAT=['is_ground','pos1_cat','pos2_cat']

def best_mcc(y,p):
    order=np.argsort(-p); ys=y[order].astype(np.float64); ps=p[order]
    tp=np.cumsum(ys); fp=np.cumsum(1.0-ys)
    P=float(y.sum()); N=float(len(y))-P
    fn=P-tp; tn=N-fp
    den=np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    mcc=np.where(den>0,(tp*tn-fp*fn)/np.where(den>0,den,1.0),0.0)
    i=int(np.argmax(mcc)); lo=ps[i]; hi=ps[min(i+1,len(ps)-1)]
    return float(mcc[i]), float((lo+hi)/2)

def smooth_eval(d, oof, groups=('game_play','p1','p2'), thrs=(1,3,5,7,9)):
    tmp=d[list(groups)+['contact']].copy(); tmp['prob']=oof
    tmp=tmp.sort_values(list(groups)).reset_index(drop=True)
    out={}
    for w in thrs:
        pp=tmp.groupby(list(groups),sort=False)['prob'].rolling(w,min_periods=1,center=True).mean()
        pp=pp.reset_index(level=[0,1,2],drop=True).values
        m,t=best_mcc(tmp['contact'].values, pp); out[w]=(m,t)
    wbest=max(out,key=lambda k:out[k][0])
    return out,(wbest,)+out[wbest]

def run(feat, subset, rounds=3000, leaves=160, lr=0.03, seed=42, smooth=5, folds=5):
    t0=time.time()
    d=pd.read_parquet(os.path.join(CACHE,f'featg_all_{feat}.parquet'))
    if subset=='ground': d=d[d.p2=='G'].reset_index(drop=True)
    elif subset=='pair': d=d[d.p2!='G'].reset_index(drop=True)
    feats=[c for c in d.columns if c not in META]
    y=d['contact'].values.astype(np.int8)
    groups=d['game_play'].values
    params=dict(objective='binary',learning_rate=lr,num_leaves=leaves,min_data_in_leaf=100,
                feature_fraction=0.8,bagging_fraction=0.8,bagging_freq=1,lambda_l2=1.0,
                n_jobs=16,seed=seed,verbose=-1,metric='auc')
    oof=np.zeros(len(d)); iters=[]
    gkf=GroupKFold(n_splits=folds)
    for k,(tri,vai) in enumerate(gkf.split(d,y,groups)):
        dtr=lgb.Dataset(d.iloc[tri][feats],y[tri],categorical_feature=[c for c in CAT if c in feats])
        dva=lgb.Dataset(d.iloc[vai][feats],y[vai],categorical_feature=[c for c in CAT if c in feats],reference=dtr)
        m=lgb.train(params,dtr,num_boost_round=rounds,valid_sets=[dva],
                    callbacks=[lgb.early_stopping(100,verbose=False)])
        oof[vai]=m.predict(d.iloc[vai][feats],num_iteration=m.best_iteration)
        iters.append(m.best_iteration)
        m.save_model(os.path.join(CACHE,f'm_{feat}_{subset}_{k}.txt'))
        del dtr,dva,m; gc.collect()
    mcc,thr=best_mcc(y,oof)
    sm,smbest=smooth_eval(d,oof)
    print(f'[{feat}/{subset}] rows={len(d)} OOF MCC {mcc:.5f} thr {thr:.4f} | smooth w{smbest[0]} MCC {smbest[1]:.5f} thr {smbest[2]:.4f} | iters {iters} t={round(time.time()-t0)}')
    for w,v in sm.items(): print(f'   smooth w{w}: {v[0]:.5f}')
    np.save(os.path.join(CACHE,f'oof_{feat}_{subset}.npy'),oof.astype(np.float32))
    np.save(os.path.join(CACHE,f'ids_{feat}_{subset}.npy'),d.contact_id.values)
    # test predictions
    dt=pd.read_parquet(os.path.join(CACHE,f'featg_all_test.parquet'))
    if subset=='ground': dt=dt[dt.p2=='G'].reset_index(drop=True)
    elif subset=='pair': dt=dt[dt.p2!='G'].reset_index(drop=True)
    pt=np.zeros(len(dt))
    for k in range(folds):
        m=lgb.Booster(model_file=os.path.join(CACHE,f'm_{feat}_{subset}_{k}.txt'))
        pt+=m.predict(dt[feats],num_iteration=iters[k])/folds
    np.save(os.path.join(CACHE,f'ptest_{feat}_{subset}.npy'),pt.astype(np.float32))
    np.save(os.path.join(CACHE,f'pids_{feat}_{subset}.npy'),dt.contact_id.values)
    return dict(mcc=mcc,thr=thr,best_w=smbest[0],best_sm=smbest[1],best_sm_thr=smbest[2],iters=iters,feats=feats,subset=subset,feat=feat)

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--feat',default='train'); ap.add_argument('--subset',default='all')
    ap.add_argument('--leaves',type=int,default=160); ap.add_argument('--smooth',type=int,default=5)
    ap.add_argument('--save_models',type=int,default=1)
    a=ap.parse_args()
    res=run(a.feat,a.subset)
    json.dump(res,open(os.path.join(CACHE,f'res_{a.feat}_{a.subset}.json'),'w'))
