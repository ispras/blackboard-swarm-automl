import os, sys, time, json, argparse
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold

HERE=os.path.dirname(os.path.abspath(__file__)); CACHE=os.path.join(HERE,'cache')
META={'contact_id','game_play','datetime','nfl_player_id_1','nfl_player_id_2','contact','p1','p2','game_key'}
CAT=['is_ground','pos1_cat','pos2_cat']

def best_mcc(y,p):
    order=np.argsort(-p); ys=y[order].astype(np.float64); ps=p[order]
    tp=np.cumsum(ys); fp=np.cumsum(1.0-ys)
    P=float(y.sum()); N=float(len(y))-P; fn=P-tp; tn=N-fp
    den=np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    mcc=np.where(den>0,(tp*tn-fp*fn)/np.where(den>0,den,1.0),0.0)
    i=int(np.argmax(mcc)); return float(mcc[i]), float((ps[i]+ps[min(i+1,len(ps)-1)])/2)

def smooth_sorted(vals,grp,w):
    r=pd.Series(vals).groupby(grp).rolling(w,min_periods=1,center=True).mean()
    return r.values

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--neg_frac',type=float,default=0.35); ap.add_argument('--folds',type=int,default=5)
    ap.add_argument('--iters',type=int,default=1500); ap.add_argument('--lr',type=float,default=0.05)
    ap.add_argument('--depth',type=int,default=8); ap.add_argument('--tag',default='cb')
    a=ap.parse_args()
    from catboost import CatBoostClassifier, Pool
    t0=time.time()
    d=pd.read_parquet(os.path.join(CACHE,'featg_all_train.parquet'))
    d['is_ground']=(d.p2=='G').astype(np.int8)
    feats=[c for c in d.columns if c not in META]
    y=d.contact.values.astype(np.int8); groups=d.game_play.values
    rng=np.random.RandomState(3); oof=np.zeros(len(d))
    gkf=GroupKFold(n_splits=a.folds)
    cats=[c for c in CAT if c in feats]
    for k,(tri,vai) in enumerate(gkf.split(d,y,groups)):
        trp=tri[y[tri]==1]; trn=tri[y[tri]==0]; trn=trn[rng.rand(len(trn))<a.neg_frac]
        tri2=np.concatenate([trp,trn])
        dtr=Pool(d.iloc[tri2][feats],y[tri2],cat_features=cats)
        dva=Pool(d.iloc[vai][feats],y[vai],cat_features=cats)
        m=CatBoostClassifier(iterations=a.iters,learning_rate=a.lr,depth=a.depth,
            loss_function='Logloss',eval_metric='AUC',task_type='GPU',devices='0',
            random_seed=k,od_type='Iter',od_wait=80,verbose=200,allow_writing_files=False)
        m.fit(dtr,eval_set=dva,use_best_model=True)
        oof[vai]=m.predict_proba(d.iloc[vai][feats])[:,1]
        m.save_model(os.path.join(CACHE,f'm_{a.tag}_{k}.cbm'))
        print('fold %d best %d t=%d'%(k,m.get_best_iteration(),time.time()-t0),flush=True)
        del dtr,dva,m
    raw,thr=best_mcc(y,oof)
    dd=d[['game_play','p1','p2','step','contact']].copy(); dd['prob']=oof
    dd=dd.sort_values(['game_play','p1','p2','step']).reset_index(drop=True)
    grp=dd.groupby(['game_play','p1','p2'],sort=False).ngroup().values
    best=(-1,0,0)
    for w in (3,5,7):
        sm=smooth_sorted(dd.prob.values,grp,w); m,th=best_mcc(dd.contact.values,sm)
        if m>best[0]: best=(m,w,th)
        print('smooth w%d %.5f'%(w,m),flush=True)
    print('[%s] raw %.5f thr %.4f | smooth %.5f w%d thr %.5f t=%d'%(a.tag,raw,thr,best[0],best[1],best[2],time.time()-t0),flush=True)
    np.save(os.path.join(CACHE,f'oof_{a.tag}.npy'),oof.astype(np.float32))
    np.save(os.path.join(CACHE,f'ids_{a.tag}.npy'),d.contact_id.values)
    json.dump(dict(raw=raw,smooth=best[0],w=best[1],thr=best[2]),open(os.path.join(CACHE,f'res_{a.tag}.json'),'w'))
    # test
    dt=pd.read_parquet(os.path.join(CACHE,'featg_all_test.parquet'))
    dt['is_ground']=(dt.p2=='G').astype(np.int8)
    pt=np.zeros(len(dt))
    for k in range(a.folds):
        m=CatBoostClassifier(); m.load_model(os.path.join(CACHE,f'm_{a.tag}_{k}.cbm'))
        pt+=m.predict_proba(dt[feats])[:,1]/a.folds
    np.save(os.path.join(CACHE,f'ptest_{a.tag}.npy'),pt.astype(np.float32))
    np.save(os.path.join(CACHE,f'pids_{a.tag}.npy'),dt.contact_id.values)
    pd.DataFrame({'contact_id':d.contact_id.values,'prob':oof}).to_parquet(os.path.join(HERE,f'oof_pred_{a.tag}.parquet'))
    pd.DataFrame({'contact_id':dt.contact_id.values,'prob':pt}).to_parquet(os.path.join(HERE,f'test_pred_{a.tag}.parquet'))
    print('saved test + artifacts',flush=True)

if __name__=='__main__':
    main()
