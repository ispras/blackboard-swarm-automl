import numpy as np, pyarrow.parquet as pq
PARQ='/download_job_code/payload/sandbox/workspaces/aldebaran_omicron/features_train.parquet'
LONMIN,LONMAX,LATMIN,LATMAX=-75.0,-72.0,39.0,42.0
pf=pq.ParquetFile(PARQ); ys=[]; off=0
for b in pf.iter_batches(batch_size=750_000, columns=['fare_amount','plat','plon','dlat','dlon','passenger_count']):
    fare=b.column('fare_amount').to_numpy(zero_copy_only=False).astype(np.float32)
    plat=b.column('plat').to_numpy(zero_copy_only=False).astype(np.float32)
    plon=b.column('plon').to_numpy(zero_copy_only=False).astype(np.float32)
    dlat=b.column('dlat').to_numpy(zero_copy_only=False).astype(np.float32)
    dlon=b.column('dlon').to_numpy(zero_copy_only=False).astype(np.float32)
    pax=b.column('passenger_count').to_numpy(zero_copy_only=False).astype(np.float32)
    n=len(fare); pos=np.arange(off,off+n); off+=n
    bad=lambda lon,lat:(lon<LONMIN)|(lon>LONMAX)|(lat<LATMIN)|(lat>LATMAX)
    cm=bad(plon,plat)|bad(dlon,dlat)
    clean=(fare>=2.5)&(fare<=500)&(pax>=0)&(pax<=8)&np.isfinite(fare)
    ys.append(fare[clean&cm&(pos%100==0)])
y=np.concatenate(ys)
print('corrupt holdout n',len(y),'mean',y.mean(),'std',y.std(),'rmse const=mean',np.sqrt(np.mean((y-y.mean())**2)))
print('quantiles',np.quantile(y,[.5,.9,.99,.999,1]))
print('const 12.2 rmse',np.sqrt(np.mean((y-12.2)**2)),'const 11.3',np.sqrt(np.mean((y-11.3)**2)))
np.save('/download_job_code/payload/sandbox/workspaces/aldebaran_omicron/corrupt_holdout_y.npy',y)
