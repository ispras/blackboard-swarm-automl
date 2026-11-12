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
    P=float(y.sum()); N=float(len(y))-P; fn=P-tp; tn=N-fp
    den=np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    mcc=np.where(den>0,(tp*tn-fp*fn)/np.where(den>0,den,1.0),0.0)
    i=int(np.argmax(mcc)); return float(mcc[i]), float((ps[i]+ps[min(i+1,len(ps)-1)])/2)

def smooth_eval(tmp, groups, thrs=(1,3,5,7,9)):
    out={}
    for w in thrs:
        r=tmp.groupby(list(groups),sort=False)['prob'].rolling(w,min_periods=1,center=True).mean()
        pp=r.reset_index(level=[0,1,2],drop=True).reindex(tmp.index).values
        out[w]=best_mcc(tmp['contact'].values, pp)
    wbest=max(out,key=lambda k:out[k][0]); return out,(wbest,out[wbest][0],out[wbest][1])

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--feat',default='train'); ap.add_argument('--subset',default='all')
    ap.add_argument('--neg_frac',type=float,default=0.35)
    ap.add_argument('--leaves',type=int,default=128); ap.add_argument('--lr',type=float,default=0.04)
    ap.add_argument('--rounds',type=int,default=1500); ap.add_argument('--folds',type=int,default=5)
    ap.add_argument('--seed',type=int,default=7); ap.add_argument('--tag',default='g1')
    ap.add_argument('--fbase',default='featg_all')
    ap.add_argument('--artifact',type=int,default=0)
    a=ap.parse_args()
    t0=time.time()
    d=pd.read_parquet(os.path.join(CACHE,f'{a.fbase}_{a.feat}.parquet'))
    if a.subset!='all':
        want=1 if a.subset=='ground' else 0
        d=d[d.is_ground==want].reset_index(drop=True)
    feats=[c for c in d.columns if c not in META]
    y=d['contact'].values.astype(np.int8); groups=d['game_play'].values
    rng=np.random.RandomState(a.seed)
    oof=np.zeros(len(d)); iters=[]
    gkf=GroupKFold(n_splits=a.folds)
    for k,(tri,vai) in enumerate(gkf.split(d,y,groups)):
        tr_pos=tri[y[tri]==1]
        tr_neg=tri[y[tri]==0]
        if a.neg_frac<1.0:
            tr_neg=tr_neg[rng.rand(len(tr_neg))<a.neg_frac]
        tri2=np.concatenate([tr_pos,tr_neg])
        params=dict(objective='binary',learning_rate=a.lr,num_leaves=a.leaves,min_data_in_leaf=80,
                    feature_fraction=0.8,bagging_fraction=0.8,bagging_freq=1,lambda_l2=1.0,
                    n_jobs=16,seed=a.seed+k,verbose=-1,metric='auc')
        dtr=lgb.Dataset(d.iloc[tri2][feats],y[tri2],categorical_feature=[c for c in CAT if c in feats])
        dva=lgb.Dataset(d.iloc[vai][feats],y[vai],categorical_feature=[c for c in CAT if c in feats],reference=dtr)
        m=lgb.train(params,dtr,num_boost_round=a.rounds,valid_sets=[dva],
                    callbacks=[lgb.early_stopping(80,verbose=False),lgb.log_evaluation(200)])
        oof[vai]=m.predict(d.iloc[vai][feats],num_iteration=m.best_iteration)
        iters.append(m.best_iteration)
        m.save_model(os.path.join(CACHE,f'm_{a.tag}_{k}.txt'))
        print('fold %d iter %d ntr %d t=%d'%(k,m.best_iteration,len(tri2),time.time()-t0),flush=True)
        del dtr,dva,m; gc.collect()
    mcc,thr=best_mcc(y,oof)
    tmp=d[['game_play','p1','p2','step','contact']].copy(); tmp['prob']=oof
    tmp=tmp.sort_values(['game_play','p1','p2','step']).reset_index(drop=True)
    sm,smbest=smooth_eval(tmp,('game_play','p1','p2'))
    print(f'[{a.tag}] OOF raw MCC {mcc:.5f} thr {thr:.4f} | smooth w{smbest[0]} MCC {smbest[1]:.5f} thr {smbest[2]:.4f} t={round(time.time()-t0)}',flush=True)
    for w,v in sm.items(): print('  smooth w%d %.5f'%(w,v[0]),flush=True)
    np.save(os.path.join(CACHE,f'oof_{a.tag}.npy'),oof.astype(np.float32))
    np.save(os.path.join(CACHE,f'ids_{a.tag}.npy'),d.contact_id.values)
    res=dict(raw_mcc=mcc,raw_thr=thr,smooth_w=smbest[0],smooth_mcc=smbest[1],smooth_thr=smbest[2],iters=iters,neg_frac=a.neg_frac,leaves=a.leaves,lr=a.lr)
    json.dump(res,open(os.path.join(CACHE,f'res_{a.tag}.json'),'w'))
    # test preds
    dt=pd.read_parquet(os.path.join(CACHE,f'{a.fbase}_test.parquet'))
    if a.subset!='all':
        want=1 if a.subset=='ground' else 0
        dt=dt[dt.is_ground==want].reset_index(drop=True)
    pt=np.zeros(len(dt))
    for k in range(a.folds):
        m=lgb.Booster(model_file=os.path.join(CACHE,f'm_{a.tag}_{k}.txt'))
        pt+=m.predict(dt[feats])/a.folds
    np.save(os.path.join(CACHE,f'ptest_{a.tag}.npy'),pt.astype(np.float32))
    np.save(os.path.join(CACHE,f'pids_{a.tag}.npy'),dt.contact_id.values)
    print('saved test',len(dt),flush=True)
    if a.artifact and a.subset=='all':
        od=pd.DataFrame({'contact_id':d.contact_id.values,'prob':oof})
        od.to_parquet(os.path.join(HERE,'oof_pred.parquet'))
        pd.DataFrame({'contact_id':dt.contact_id.values,'prob':pt}).to_parquet(os.path.join(HERE,'test_pred.parquet'))
        print('artifact written',flush=True)

if __name__=='__main__':
    main()
