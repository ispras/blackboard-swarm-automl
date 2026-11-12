import argparse, time, json, os, sys, gc
import numpy as np
import pandas as pd
import catboost as cb

ROOT='/download_job_code/payload/sandbox'
OUT=f'{ROOT}/workspaces/aldebaran_omicron'
sys.path.insert(0, f'{ROOT}/workspaces/mimosa_omicron')
from test_features import build_test, add_counts_te
from common import load_te

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--tag',type=str,default='catS')
    ap.add_argument('--depth',type=int,default=10)
    ap.add_argument('--lr',type=float,default=0.06)
    ap.add_argument('--iters',type=int,default=4000)
    ap.add_argument('--l2',type=float,default=3.0)
    ap.add_argument('--subsample',type=float,default=0.8)
    ap.add_argument('--border',type=int,default=128)
    ap.add_argument('--seed',type=int,default=7)
    ap.add_argument('--threads',type=int,default=16)
    ap.add_argument('--corrupt_constant',type=float,default=12.2)
    a=ap.parse_args()
    print(json.dumps(vars(a)),flush=True)
    X=np.load(f'{OUT}/shared_X.npy',mmap_mode='r'); y=np.load(f'{OUT}/shared_y.npy',mmap_mode='r')
    oi=np.load(f'{OUT}/shared_origidx.npy',mmap_mode='r'); names=list(np.load(f'{OUT}/shared_featnames.npy'))
    hold=(np.asarray(oi)%100==0)
    tr=np.where(~hold)[0]; va=np.where(hold)[0]
    print('n',len(tr),len(va),flush=True)
    Xtr=np.asarray(X[tr]); ytr=np.asarray(y[tr]); Xva=np.asarray(X[va]); yva=np.asarray(y[va]); ova=np.asarray(oi[va])
    del X,y,oi; gc.collect()
    print('loaded',Xtr.shape,time.time(),flush=True)
    m=cb.CatBoostRegressor(iterations=a.iters,depth=a.depth,learning_rate=a.lr,loss_function='RMSE',
        eval_metric='RMSE',task_type='GPU',devices='0',bootstrap_type='Bernoulli',subsample=a.subsample,
        l2_leaf_reg=a.l2,border_count=a.border,random_seed=a.seed,od_type='Iter',od_wait=150,verbose=100)
    t0=time.time()
    m.fit(Xtr,ytr,eval_set=(Xva,yva),use_best_model=True)
    pv=m.predict(Xva)
    rmse=float(np.sqrt(np.mean((pv-yva)**2)))
    print(f'CAT_HOLDOUT_VALID_RMSE {rmse:.5f} best {m.get_best_iteration()} time {time.time()-t0:.0f}s',flush=True)
    m.save_model(f'{OUT}/model_{a.tag}.cbm')
    np.save(f'{OUT}/holdout_pred_{a.tag}.npy',pv.astype(np.float32))
    np.save(f'{OUT}/holdout_orig_{a.tag}.npy',ova)
    np.save(f'{OUT}/holdout_true_{a.tag}.npy',yva.astype(np.float32))
    del Xtr; gc.collect()
    te=load_te(); df=build_test(); df=add_counts_te(df,te)
    corrupt=df['corrupt'].to_numpy().astype(bool)
    Xte=df[names].to_numpy(np.float32)
    tp=np.asarray(m.predict(Xte),dtype=np.float32)
    tp[corrupt]=a.corrupt_constant
    tp=np.clip(tp,2.5,500.0)
    np.save(f'{OUT}/test_pred_{a.tag}.npy',tp)
    pd.DataFrame({'key':pd.read_csv(f'{ROOT}/data/test.csv')['key'],'fare_amount':tp}).to_csv(f'{OUT}/submission_{a.tag}.csv',index=False)
    with open(f'{OUT}/result_{a.tag}.json','w') as f:
        json.dump({'tag':a.tag,'rmse':rmse,'best':int(m.get_best_iteration()),'corrupt':int(corrupt.sum()),
                   'args':vars(a),'time_s':time.time()-t0},f,indent=2)
    print('test corrupt',int(corrupt.sum()),'mean',float(tp.mean()),flush=True)

if __name__=='__main__':
    main()
