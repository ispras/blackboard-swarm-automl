import numpy as np,pandas as pd,sys,time
sys.path.insert(0,'/download_job_code/payload/sandbox/workspaces/alkaid_phi')
import prep_data
from add_te import derive,cell,route_code,K
OUT="/download_job_code/payload/sandbox/workspaces/alkaid_phi"; DATA="/download_job_code/payload/sandbox/data"
t0=time.time()
d=pd.read_parquet(f"{OUT}/train_te.parquet",columns=['pickup_latitude','pickup_longitude','dropoff_latitude','dropoff_longitude','fare_amount','holdout'])
trd=d[d.holdout==0]
prior=float(trd.fare_amount.mean()); ytr=trd.fare_amount.values.astype(np.float64)
specs={'p01':('pickup_latitude','pickup_longitude',0.01),'d01':('dropoff_latitude','dropoff_longitude',0.01),
       'p05':('pickup_latitude','pickup_longitude',0.05),'d05':('dropoff_latitude','dropoff_longitude',0.05)}
tr_codes_tr={name:cell(trd[la],trd[lo],res) for name,(la,lo,res) in specs.items()}
tr_route_tr=route_code(tr_codes_tr['p01'],tr_codes_tr['d01'])
def fg(codes_tr,codes_te):
    codes_te=pd.Series(codes_te)
    g=pd.DataFrame({'c':codes_tr,'y':ytr}).groupby('c')['y'].agg(['sum','count'])
    mean=(g['sum']+prior*K)/(g['count']+K); cnt=np.log1p(g['count'])
    return codes_te.map(mean).fillna(prior).astype(np.float32).values,codes_te.map(cnt).fillna(0).astype(np.float32).values
raw=pd.read_csv(f"{DATA}/test.csv")
td=pd.DataFrame(prep_data.feat(raw)); td['key']=raw['key'].values
flag=np.zeros(len(td),dtype=np.int8)
for p in ['pickup','dropoff']:
    la=td[f'{p}_latitude'].values.copy(); lo=td[f'{p}_longitude'].values.copy()
    ok=(la>39)&(la<43)&(lo>-76)&(lo<-71)
    sw=(lo>39)&(lo<43)&(la>-76)&(la<-71)&(~ok)
    la2=np.where(sw,lo,la); lo2=np.where(sw,la,lo); flag=flag|sw.astype(np.int8)
    td[f'{p}_latitude']=la2.astype(np.float32); td[f'{p}_longitude']=lo2.astype(np.float32)
td=derive(td); td['swap_flag']=flag
la=td.pickup_latitude.values.astype(np.float64); lo=td.pickup_longitude.values.astype(np.float64)
la2=td.dropoff_latitude.values.astype(np.float64); lo2=td.dropoff_longitude.values.astype(np.float64)
td['haversine']=prep_data.hav(la,lo,la2,lo2).astype(np.float32)
td['manhattan']=(prep_data.hav(la,lo,la,lo2)+prep_data.hav(la,lo2,la2,lo2)).astype(np.float32)
td['log_hav']=np.log1p(td.haversine).astype(np.float32)
td['pickup_latitude']=la.astype(np.float32); td['pickup_longitude']=lo.astype(np.float32)
td['dropoff_latitude']=la2.astype(np.float32); td['dropoff_longitude']=lo2.astype(np.float32)
te_codes={}
for name,(laN,loN,res) in specs.items():
    te_codes[name]=cell(td[laN],td[loN],res)
    a,b=fg(tr_codes_tr[name],te_codes[name]); td[f'te_{name}']=a; td[f'cnt_{name}']=b
te_route=route_code(te_codes['p01'],te_codes['d01'])
a,b=fg(tr_route_tr,te_route); td['te_route']=a; td['cnt_route']=b
td.to_parquet(f"{OUT}/test_te.parquet",index=False)
print("saved test_te",td.shape,"swapped",int(flag.sum()),"prior",prior,time.time()-t0)
