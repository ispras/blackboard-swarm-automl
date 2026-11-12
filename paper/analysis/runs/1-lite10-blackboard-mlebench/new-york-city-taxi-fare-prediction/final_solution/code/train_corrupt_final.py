import sys, os, json, time
import numpy as np, pandas as pd, lightgbm as lgb
ROOT='/download_job_code/payload/sandbox'
MIM=f'{ROOT}/workspaces/mimosa_omicron'; ME=f'{ROOT}/workspaces/aldebaran_omicron'
sys.path.insert(0,MIM)
from build_corrupt import add_corrupt_feats
from test_features import build_test
df=pd.read_parquet(f'{MIM}/corrupt_feat.parquet')
y=df['fare_amount'].to_numpy(np.float32)
feats=[c for c in df.columns if c not in ('fare_amount','orig_idx')]
X=df[feats].to_numpy(np.float32)
print('train corrupt',X.shape,flush=True)
params=dict(objective='regression',metric='rmse',learning_rate=0.05,num_leaves=31,min_data_in_leaf=500,
            feature_fraction=0.9,bagging_fraction=0.9,bagging_freq=1,max_bin=255,num_threads=8,verbose=-1,seed=0)
t0=time.time()
m=lgb.train(params,lgb.Dataset(X,y,feature_name=feats),num_boost_round=650)
m.save_model(f'{ME}/model_corrupt_final.txt')
# test features
tdf=build_test(); corrupt=tdf['corrupt'].to_numpy().astype(bool)
print('test corrupt',corrupt.sum(),flush=True)
sub=tdf.loc[corrupt].copy()
if 'pickup_datetime' not in sub.columns:
    sub['pickup_datetime']=pd.read_csv(f'{ROOT}/data/test.csv').loc[corrupt,'pickup_datetime'].to_numpy()
fb=add_corrupt_feats(sub)
Xte=fb[feats].to_numpy(np.float32)
p=m.predict(Xte).astype(np.float32)
print('corrupt pred stats',np.percentile(p,[0,25,50,75,100]),'mean',p.mean(),flush=True)
full=np.full(len(tdf),np.nan,dtype=np.float32); full[corrupt]=p
np.save(f'{ME}/corrupt_test_pred.npy',full)
with open(f'{ME}/corrupt_final_result.json','w') as f:
    json.dump({'n_train':len(y),'n_corrupt_test':int(corrupt.sum()),'iters':650,'time_s':time.time()-t0,
               'cv_rmse_est':12.81,'corrupt_mean_pred':float(p.mean())},f,indent=2)
print('done',time.time()-t0,flush=True)
