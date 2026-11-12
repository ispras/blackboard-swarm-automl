import numpy as np, pandas as pd, lightgbm as lgb, argparse, os, time, json
OUT="/download_job_code/payload/sandbox/workspaces/alkaid_phi"
p=argparse.ArgumentParser()
p.add_argument('--rows', type=int, default=0)
p.add_argument('--leaves', type=int, default=511)
p.add_argument('--lr', type=float, default=0.05)
p.add_argument('--threads', type=int, default=16)
p.add_argument('--rounds', type=int, default=4000)
p.add_argument('--seed', type=int, default=1)
p.add_argument('--tag', type=str, default='lgb')
p.add_argument('--feat', type=str, default='train_feat.parquet')
a=p.parse_args()
t0=time.time()
df=pd.read_parquet(f"{OUT}/{a.feat}")
print("loaded", df.shape, time.time()-t0, flush=True)
if a.rows and a.rows < len(df):
    df=df.sample(a.rows, random_state=42).reset_index(drop=True)
feats=[c for c in df.columns if c not in ('key','fare_amount','holdout')]
print("nfeat",len(feats),feats, flush=True)
tr=df[df.holdout==0]; va=df[df.holdout==1]
print("train",tr.shape,"val",va.shape, flush=True)
Xtr=tr[feats]; ytr=tr.fare_amount.values
Xva=va[feats]; yva=va.fare_amount.values
vakeys=va[['key','fare_amount']].reset_index(drop=True)
del df, tr, va
params=dict(objective='regression',metric='rmse',learning_rate=a.lr,num_leaves=a.leaves,
    feature_fraction=0.85,bagging_fraction=0.8,bagging_freq=1,min_data_in_leaf=100,
    max_bin=255,lambda_l2=1.0,num_threads=a.threads,seed=a.seed,verbosity=-1)
dtr=lgb.Dataset(Xtr,ytr,free_raw_data=True)
dva=lgb.Dataset(Xva,yva,reference=dtr,free_raw_data=True)
evals={}
m=lgb.train(params,dtr,num_boost_round=a.rounds,valid_sets=[dva],valid_names=['val'],
    callbacks=[lgb.early_stopping(150),lgb.log_evaluation(50),lgb.record_evaluation(evals)])
pred=m.predict(Xva, num_iteration=m.best_iteration)
rmse=float(np.sqrt(np.mean((pred-yva)**2)))
print("HOLDOUT_RMSE",rmse,"best_iter",m.best_iteration, flush=True)
# clipped variants
for lo,hi in [(2.5,500),(1.0,300),(0,500)]:
    pc=np.clip(pred,lo,hi); print("clip",lo,hi,float(np.sqrt(np.mean((pc-yva)**2))), flush=True)
m.save_model(f"{OUT}/{a.tag}_model.txt")
json.dump({'rmse':rmse,'best_iter':m.best_iteration,'feats':feats,'params':params},open(f"{OUT}/{a.tag}_meta.json",'w'))
# feature importance
imp=pd.DataFrame({'f':feats,'g':m.feature_importance('gain'),'s':m.feature_importance('split')}).sort_values('g',ascending=False)
imp.to_csv(f"{OUT}/{a.tag}_imp.csv",index=False)
print(imp.head(30).to_string(), flush=True)
np.save(f"{OUT}/{a.tag}_val_pred.npy", pred)
vakeys.assign(pred=pred).to_parquet(f"{OUT}/{a.tag}_val.parquet", index=False)
print("done",time.time()-t0, flush=True)
