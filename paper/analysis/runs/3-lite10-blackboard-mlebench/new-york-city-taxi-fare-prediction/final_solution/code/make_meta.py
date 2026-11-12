import pandas as pd, numpy as np, pyarrow.parquet as pq
CACHE="/download_job_code/payload/sandbox/workspaces/arcturus_upsilon/cache"
OUT="/download_job_code/payload/sandbox/workspaces/alkaid_phi"
pf=pq.ParquetFile(f"{CACHE}/train_v2.parquet")
co=['key','fare_amount','holdout','has_pickup','has_dropoff','distance_km']
parts=[]
for b in pf.iter_batches(batch_size=2_000_000,columns=co):
    d=b.to_pandas(); d=d[d.holdout]
    if len(d): parts.append(d)
h=pd.concat(parts,ignore_index=True)[['key','fare_amount','has_pickup','has_dropoff','distance_km']]
h.to_parquet(f"{OUT}/val_meta.parquet",index=False)
print('val_meta',h.shape)
t=pd.read_parquet(f"{CACHE}/test_v2.parquet",columns=['key','has_pickup','has_dropoff','distance_km'])
t.to_parquet(f"{OUT}/test_meta.parquet",index=False)
print('test_meta',t.shape,'missing',((t.has_pickup==0)|(t.has_dropoff==0)).sum(),'degen',(((t.has_pickup==1)&(t.has_dropoff==1))&(t.distance_km<0.1)).sum())
