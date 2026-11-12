import os, sys, time, json
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import GroupKFold

HERE=os.path.dirname(os.path.abspath(__file__))
CACHE=os.path.join(HERE,'cache')
META={'contact_id','game_play','datetime','nfl_player_id_1','nfl_player_id_2','contact','p1','p2','game_key'}

def best_mcc(y,p):
    order=np.argsort(-p); ys=y[order].astype(np.float64); ps=p[order]
    tp=np.cumsum(ys); fp=np.cumsum(1.0-ys)
    P=float(y.sum()); N=float(len(y))-P; fn=P-tp; tn=N-fp
    den=np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    mcc=np.where(den>0,(tp*tn-fp*fn)/np.where(den>0,den,1.0),0.0)
    i=int(np.argmax(mcc)); return float(mcc[i]), float((ps[i]+ps[min(i+1,len(ps)-1)])/2)

def smooth(tmp, groups, w):
    r=tmp.groupby(list(groups),sort=False)['prob'].rolling(w,min_periods=1,center=True).mean()
    return r.reset_index(level=[0,1,2],drop=True).reindex(tmp.index).values

def main(feat='train', subset='ground', folds=5):
    d=pd.read_parquet(os.path.join(CACHE,f'featg_all_{feat}.parquet'))
    if subset=='ground': d=d[d.p2=='G'].reset_index(drop=True)
    elif subset=='pair': d=d[d.p2!='G'].reset_index(drop=True)
    y=d['contact'].values.astype(np.int8); groups=d['game_play'].values
    m0=lgb.Booster(model_file=os.path.join(CACHE,f'm_{feat}_{subset}_0.txt'))
    feats=m0.feature_name()
    oof=np.zeros(len(d)); iters=[]
    gkf=GroupKFold(n_splits=folds)
    for k,(tri,vai) in enumerate(gkf.split(d,y,groups)):
        m=lgb.Booster(model_file=os.path.join(CACHE,f'm_{feat}_{subset}_{k}.txt'))
        iters.append(m.best_iteration if m.best_iteration else m.num_trees())
        oof[vai]=m.predict(d.iloc[vai][feats])
    mcc,thr=best_mcc(y,oof)
    print('iters',iters,'raw MCC %.5f thr %.4f'%(mcc,thr))
    tmp=d[['game_play','p1','p2','step','contact']].copy(); tmp['prob']=oof
    tmp=tmp.sort_values(['game_play','p1','p2','step']).reset_index(drop=True)
    best=(-1,0,0)
    for w in (1,3,5,7,9):
        pp=smooth(tmp,('game_play','p1','p2'),w); mm,tt=best_mcc(tmp['contact'].values,pp)
        print('smooth w%d MCC %.5f thr %.4f'%(w,mm,tt))
        if mm>best[0]: best=(mm,w,tt)
    print('BEST smooth MCC %.5f w%d thr %.4f'%(best[0],best[1],best[2]))
    np.save(os.path.join(CACHE,f'oof_{feat}_{subset}.npy'),oof.astype(np.float32))
    np.save(os.path.join(CACHE,f'ids_{feat}_{subset}.npy'),d.contact_id.values)
    json.dump(dict(raw_mcc=mcc,raw_thr=thr,smooth_mcc=best[0],smooth_w=best[1],smooth_thr=best[2],iters=iters),
              open(os.path.join(CACHE,f'res_{feat}_{subset}.json'),'w'))
    # test
    dt=pd.read_parquet(os.path.join(CACHE,f'featg_all_test.parquet'))
    if subset=='ground': dt=dt[dt.p2=='G'].reset_index(drop=True)
    elif subset=='pair': dt=dt[dt.p2!='G'].reset_index(drop=True)
    pt=np.zeros(len(dt))
    for k in range(folds):
        m=lgb.Booster(model_file=os.path.join(CACHE,f'm_{feat}_{subset}_{k}.txt'))
        pt+=m.predict(dt[feats])/folds
    np.save(os.path.join(CACHE,f'ptest_{feat}_{subset}.npy'),pt.astype(np.float32))
    np.save(os.path.join(CACHE,f'pids_{feat}_{subset}.npy'),dt.contact_id.values)
    print('saved test',len(dt))

if __name__=='__main__':
    main(*sys.argv[1:]) if len(sys.argv)>1 else main()
