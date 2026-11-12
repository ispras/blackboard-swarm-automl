import numpy as np, pandas as pd, lightgbm as lgb, pyarrow.parquet as pq
from sklearn.model_selection import KFold
df=pd.read_parquet('/download_job_code/payload/sandbox/workspaces/mimosa_omicron/corrupt_feat.parquet')
y=df['fare_amount'].to_numpy(np.float32)
feats=[c for c in df.columns if c not in ('fare_amount','orig_idx')]
X=df[feats].to_numpy(np.float32)
print('n',len(y),'feats',len(feats),'mean',y.mean(),'std',y.std())
print('const-mean rmse',np.sqrt(np.mean((y-y.mean())**2)))
print('const 12.19 rmse',np.sqrt(np.mean((y-12.19)**2)))
params=dict(objective='regression',metric='rmse',learning_rate=0.05,num_leaves=31,min_data_in_leaf=500,
            feature_fraction=0.9,bagging_fraction=0.9,bagging_freq=1,max_bin=255,num_threads=8,verbose=-1)
kf=KFold(5,shuffle=True,random_state=0); oof=np.zeros(len(y))
for tr,te in kf.split(X):
    dtr=lgb.Dataset(X[tr],y[tr],feature_name=feats); dva=lgb.Dataset(X[te],y[te],reference=dtr)
    m=lgb.train(params,dtr,num_boost_round=600,valid_sets=[dva],callbacks=[lgb.early_stopping(40)])
    oof[te]=m.predict(X[te],num_iteration=m.best_iteration)
print('CV model rmse',np.sqrt(np.mean((oof-y)**2)))
# blend with constant
for a in [0.5,0.6,0.7,0.8,0.9,1.0]:
    p=a*oof+(1-a)*y.mean()
    print('alpha',a,'rmse',np.sqrt(np.mean((p-y)**2)))
# by zero type
pz=(df['p_zero']==1)&(df['d_zero']==1)
print('both zero frac',pz.mean(),'rmse model',np.sqrt(np.mean((oof[pz]-y[pz])**2)),'const',np.sqrt(np.mean((y[pz]-y[pz].mean())**2)))
print('not both rmse model',np.sqrt(np.mean((oof[~pz]-y[~pz])**2)),'const',np.sqrt(np.mean((y[~pz]-y[~pz].mean())**2)))
np.save('/download_job_code/payload/sandbox/workspaces/aldebaran_omicron/corrupt_oof.npy',oof)
