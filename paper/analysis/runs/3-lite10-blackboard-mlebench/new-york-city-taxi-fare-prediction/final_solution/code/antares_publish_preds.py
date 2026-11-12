import numpy as np, pandas as pd, pyarrow.parquet as pq, sys, os
W='/download_job_code/payload/sandbox/workspaces/antares_omicron'
tag=sys.argv[1]
lo,hi=float(sys.argv[2]) if len(sys.argv)>2 else 2.5, float(sys.argv[3]) if len(sys.argv)>3 else 200.0
os.makedirs(f'{W}/runs',exist_ok=True)
# val
idx=np.load(f'{W}/{tag}_valmask_ok.npy'); pv=np.load(f'{W}/{tag}_valpred.npy')
keys=np.array(pq.read_table(f'{W}/feat/keys.parquet')['key'].to_pylist(),dtype=object)
fare=pq.read_table(f'{W}/feat/train_feats.parquet',columns=['fare'])['fare'].to_numpy()
pv=np.clip(pv,lo,hi)
v=pd.DataFrame({'key':keys[idx],'y':fare[idx],'pred':pv})
v.to_parquet(f'{W}/runs/{tag}_val.parquet',index=False)
rmse=float(np.sqrt(np.mean((pv-fare[idx])**2)))
print(tag,'val rmse (clipped)',rmse)
# test
pt=np.clip(np.load(f'{W}/{tag}_testpred.npy'),lo,hi)
ss=pd.read_csv('/download_job_code/payload/sandbox/data/sample_submission.csv',dtype={'key':str})
t=pd.DataFrame({'key':ss['key'].values,'pred':pt})
t.to_parquet(f'{W}/runs/{tag}_test.parquet',index=False)
sub=pd.DataFrame({'key':t['key'],'fare_amount':t['pred']})
sub.to_csv(f'{W}/runs/{tag}_submission.csv',index=False)
print(tag,'test pred mean',pt.mean(),'min',pt.min(),'max',pt.max(),'n',len(pt))
print('nulls',sub.isna().sum().sum())
