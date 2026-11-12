import numpy as np, pandas as pd, pyarrow.parquet as pq, lightgbm as lgb, argparse, time, json, os
CACHE="/download_job_code/payload/sandbox/workspaces/arcturus_upsilon/cache"
OUT="/download_job_code/payload/sandbox/workspaces/alkaid_phi"
p=argparse.ArgumentParser()
p.add_argument('--maxrows',type=int,default=0)
p.add_argument('--limit_rows',type=int,default=0)
p.add_argument('--leaves',type=int,default=1023)
p.add_argument('--lr',type=float,default=0.04)
p.add_argument('--threads',type=int,default=24)
p.add_argument('--rounds',type=int,default=5000)
p.add_argument('--seed',type=int,default=7)
p.add_argument('--tag',type=str,default='v2lgb')
p.add_argument('--minleaf',type=int,default=200)
p.add_argument('--ff',type=float,default=0.8)
a=p.parse_args()
t0=time.time()
FNAME=f"{CACHE}/train_v2.parquet"
pf=pq.ParquetFile(FNAME)
cols=pf.schema_arrow.names
feats=[c for c in cols if c not in ('key','fare_amount','holdout')]
print('nfeat',len(feats),flush=True)
total=pf.metadata.num_rows
arr=np.empty((total,len(feats)),dtype=np.float32)
labs=np.empty(total,dtype=np.float32)
cnt=0; hparts=[]; hfare=[]; hkey=[]
for b in pf.iter_batches(batch_size=1_000_000):
    d=b.to_pandas()
    hb=d[d['holdout']]
    if len(hb):
        hparts.append(hb[feats].values); hfare.append(hb['fare_amount'].values); hkey.append(hb['key'].values)
    d=d[~d['holdout'].values]
    n=len(d)
    arr[cnt:cnt+n]=d[feats].values
    labs[cnt:cnt+n]=d['fare_amount'].values
    cnt+=n
    del d,hb
    if a.limit_rows>0 and cnt>=a.limit_rows:
        break
print('filled train rows',cnt,'time',time.time()-t0,flush=True)
n=cnt if a.maxrows<=0 else min(a.maxrows,cnt)
X=arr[:n]; y=labs[:n]
Xh=np.vstack(hparts); yh=np.concatenate(hfare); hk=np.concatenate(hkey)
print('train',n,'holdout',Xh.shape,'time',time.time()-t0,flush=True)
params=dict(objective='regression',metric='rmse',learning_rate=a.lr,num_leaves=a.leaves,
    feature_fraction=a.ff,bagging_fraction=0.8,bagging_freq=1,min_data_in_leaf=a.minleaf,
    max_bin=255,lambda_l2=1.0,num_threads=a.threads,seed=a.seed,verbosity=-1)
dtr=lgb.Dataset(X,y,free_raw_data=True)
dva=lgb.Dataset(Xh,yh,reference=dtr,free_raw_data=True)
m=lgb.train(params,dtr,num_boost_round=a.rounds,valid_sets=[dva],valid_names=['val'],
    callbacks=[lgb.early_stopping(120),lgb.log_evaluation(50)])
pred=m.predict(Xh,num_iteration=m.best_iteration)
rmse=float(np.sqrt(np.mean((pred-yh)**2)))
print("HOLDOUT_RMSE",rmse,"best_iter",m.best_iteration,"time",time.time()-t0,flush=True)
for lo,hi in [(2.5,200),(2.5,500),(0,300)]:
    print("clip",lo,hi,float(np.sqrt(np.mean((np.clip(pred,lo,hi)-yh)**2))),flush=True)
m.save_model(f"{OUT}/{a.tag}_model.txt")
json.dump({'rmse':rmse,'best_iter':m.best_iteration,'feats':feats,'params':params,'rows':int(n)},open(f"{OUT}/{a.tag}_meta.json",'w'))
pd.DataFrame({'key':hk,'fare_amount':yh,'pred':pred}).to_parquet(f"{OUT}/{a.tag}_val.parquet",index=False)
t=pd.read_parquet(f"{CACHE}/test_v2.parquet")
assert all(f in t.columns for f in feats)
pt=m.predict(t[feats].values,num_iteration=m.best_iteration)
sub=pd.DataFrame({'key':t.key.values,'fare_amount':np.clip(pt,2.5,200)})
sub.to_csv(f"{OUT}/{a.tag}_submission.csv",index=False)
print('test pred saved',sub.shape,sub.fare_amount.describe()[['mean','min','max']].to_dict(),'time',time.time()-t0,flush=True)
