import pandas as pd, numpy as np, os, time, json
import lightgbm as lgb
from sklearn.model_selection import KFold
W='/download_job_code/payload/sandbox/workspaces/canopus_pi'
t0=time.time()
D=pd.read_parquet(W+'/corrupt_train2.parquet')
T=pd.read_parquet(W+'/corrupt_test2.parquet')
y=D['fare_amount'].to_numpy(np.float32)
feats=[c for c in D.columns if c not in ('fare_amount','orig_idx')]
print('n',len(D),'feats',len(feats),feats,flush=True)
X=D[feats].to_numpy(np.float32)
# baselines
print('const global rmse',np.sqrt(np.mean((y-y.mean())**2)))
ym=D.groupby('year')['fare_amount'].transform('mean').to_numpy()
print('const by-year rmse',np.sqrt(np.mean((y-ym)**2)))
params=dict(objective='regression',metric='rmse',learning_rate=0.03,num_leaves=127,
   min_data_in_leaf=100,feature_fraction=0.85,bagging_fraction=0.8,bagging_freq=1,
   lambda_l2=5.0,max_bin=255,num_threads=12,seed=42,verbosity=-1)
# 5-fold CV by orig_idx%5
fold=(D['orig_idx'].to_numpy()%5).astype(int)
oof=np.zeros(len(y)); iters=[]
for f in range(5):
    tr=fold!=f; va=fold==f
    dtr=lgb.Dataset(X[tr],label=y[tr]); dva=lgb.Dataset(X[va],label=y[va],reference=dtr)
    m=lgb.train(params,dtr,num_boost_round=3000,valid_sets=[dva],
       callbacks=[lgb.early_stopping(100),lgb.log_evaluation(0)])
    oof[va]=m.predict(X[va],num_iteration=m.best_iteration); iters.append(m.best_iteration)
    print('fold',f,'rmse',np.sqrt(np.mean((oof[va]-y[va])**2)),'iter',m.best_iteration,flush=True)
cv=np.sqrt(np.mean((oof-y)**2)); print('CORRUPT_CV_RMSE',cv,'iters',iters,flush=True)
# segment breakdown
for nm,mask in [('bothzero',D['both_zero'].values==1),('partial',D['both_zero'].values==0)]:
    print(nm,mask.sum(),'rmse',np.sqrt(np.mean((oof[mask]-y[mask])**2)),'const',np.sqrt(np.mean((y[mask]-y[mask].mean())**2)))
# final model full data
dtr=lgb.Dataset(X,label=y)
mf=lgb.train(params,dtr,num_boost_round=int(np.mean(iters)*1.1))
predT=mf.predict(T[feats].to_numpy(np.float32))
np.save(W+'/preds/corrupt_test_pred_canopus.npy',predT.astype(np.float32))
np.save(W+'/preds/corrupt_oof_canopus.npy',oof.astype(np.float32))
D[['orig_idx','fare_amount']].assign(oof=oof).to_parquet(W+'/preds/corrupt_oof_canopus.parquet',index=False)
json.dump({'cv':float(cv),'iters':iters,'n':int(len(D))},open(W+'/corrupt_result.json','w'))
print('saved corrupt test preds',predT.shape,flush=True)
print('test pred stats',np.nanmin(predT),np.nanmax(predT),np.nanmean(predT))
print('DONE',f'{time.time()-t0:.0f}s')
