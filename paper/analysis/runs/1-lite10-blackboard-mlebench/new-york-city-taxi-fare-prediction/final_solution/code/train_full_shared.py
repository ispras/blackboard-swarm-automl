import argparse, time, json, os, sys, gc
import numpy as np
import pandas as pd
import xgboost as xgb

ROOT='/download_job_code/payload/sandbox'
OUT=f'{ROOT}/workspaces/aldebaran_omicron'
sys.path.insert(0, f'{ROOT}/workspaces/mimosa_omicron')
from test_features import build_test, add_counts_te
from common import load_te

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--tag',type=str,default='xgbS')
    ap.add_argument('--depth',type=int,default=11)
    ap.add_argument('--eta',type=float,default=0.03)
    ap.add_argument('--mcw',type=float,default=150)
    ap.add_argument('--subsample',type=float,default=0.85)
    ap.add_argument('--colsample',type=float,default=0.75)
    ap.add_argument('--l2',type=float,default=1.0)
    ap.add_argument('--max_bin',type=int,default=128)
    ap.add_argument('--rounds',type=int,default=6000)
    ap.add_argument('--early',type=int,default=150)
    ap.add_argument('--seed',type=int,default=123)
    ap.add_argument('--threads',type=int,default=16)
    ap.add_argument('--corrupt_constant',type=float,default=12.2)
    a=ap.parse_args()
    print(json.dumps(vars(a)),flush=True)
    X=np.load(f'{OUT}/shared_X.npy',mmap_mode='r'); y=np.load(f'{OUT}/shared_y.npy',mmap_mode='r')
    oi=np.load(f'{OUT}/shared_origidx.npy',mmap_mode='r'); names=list(np.load(f'{OUT}/shared_featnames.npy'))
    N=len(y); hold=(np.asarray(oi)%100==0)
    tr=np.where(~hold)[0]; va=np.where(hold)[0]
    print('n',len(tr),len(va),flush=True)
    Xtr=np.asarray(X[tr]); ytr=np.asarray(y[tr])
    Xva=np.asarray(X[va]); yva=np.asarray(y[va]); ova=np.asarray(oi[va])
    print('loaded',Xtr.shape,time.time(),flush=True)
    dtr=xgb.QuantileDMatrix(Xtr,label=ytr,feature_names=names,max_bin=a.max_bin,nthread=a.threads)
    dva=xgb.QuantileDMatrix(Xva,label=yva,feature_names=names,max_bin=a.max_bin,ref=dtr,nthread=a.threads)
    del Xtr; gc.collect()
    params=dict(device='cuda',tree_method='hist',max_bin=a.max_bin,objective='reg:squarederror',
                eval_metric='rmse',eta=a.eta,max_depth=a.depth,min_child_weight=a.mcw,
                subsample=a.subsample,colsample_bytree=a.colsample,reg_lambda=a.l2,
                seed=a.seed,nthread=a.threads,verbosity=1)
    t0=time.time()
    bst=xgb.train(params,dtr,num_boost_round=a.rounds,evals=[(dva,'val')],early_stopping_rounds=a.early,verbose_eval=100)
    best=bst.best_iteration
    pv=bst.predict(dva,iteration_range=(0,best+1))
    rmse=float(np.sqrt(np.mean((pv-np.asarray(yva))**2)))
    print(f'BEST={best} HOLDOUT_VALID_RMSE={rmse:.5f} time={time.time()-t0:.0f}s',flush=True)
    bst.save_model(f'{OUT}/model_{a.tag}.json')
    np.save(f'{OUT}/holdout_pred_{a.tag}.npy',pv.astype(np.float32))
    np.save(f'{OUT}/holdout_orig_{a.tag}.npy',ova)
    np.save(f'{OUT}/holdout_true_{a.tag}.npy',yva.astype(np.float32))
    te=load_te(); df=build_test(); df=add_counts_te(df,te)
    corrupt=df['corrupt'].to_numpy().astype(bool)
    Xte=df[names].to_numpy(np.float32)
    dte=xgb.DMatrix(Xte,feature_names=names,nthread=a.threads)
    tp=np.asarray(bst.predict(dte,iteration_range=(0,best+1)),dtype=np.float32)
    tp[corrupt]=a.corrupt_constant
    tp=np.clip(tp,2.5,500.0)
    np.save(f'{OUT}/test_pred_{a.tag}.npy',tp)
    pd.DataFrame({'key':pd.read_csv(f'{ROOT}/data/test.csv')['key'],'fare_amount':tp}).to_csv(f'{OUT}/submission_{a.tag}.csv',index=False)
    with open(f'{OUT}/result_{a.tag}.json','w') as f:
        json.dump({'tag':a.tag,'rmse':rmse,'best':int(best),'corrupt':int(corrupt.sum()),'args':vars(a),'time_s':time.time()-t0},f,indent=2)
    print('test corrupt',int(corrupt.sum()),'mean',float(tp.mean()),flush=True)

if __name__=='__main__':
    main()
