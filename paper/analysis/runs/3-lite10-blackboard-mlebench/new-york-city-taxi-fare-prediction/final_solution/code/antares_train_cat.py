import numpy as np, pyarrow.parquet as pq, time, sys, json, os
from catboost import CatBoostRegressor, Pool
W='/download_job_code/payload/sandbox/workspaces/antares_omicron'
N=int(sys.argv[1]) if len(sys.argv)>1 else 0
ITERS=int(sys.argv[2]) if len(sys.argv)>2 else 3000
TAG=sys.argv[3] if len(sys.argv)>3 else 'cat'
t0=time.time()
FEATS=['lon1','lat1','lon2','lat2','passenger_count','year','month','day','hour','minute','dow','doy','is_weekend','dist_km','dlat','dlon','abs_dlat','abs_dlon','manhattan_km','bearing','p_jfk','d_jfk','min_jfk','p_lga','d_lga','min_lga','p_ewr','d_ewr','min_ewr','p_midtown','d_midtown','min_midtown','p_dt','d_dt','min_dt','swap_flag','has_pickup','has_dropoff']
TE=['te_p01','cnt_p01','te_d01','cnt_d01','te_p05','cnt_p05','te_d05','cnt_d05','te_route05','cnt_route05','same_cell05']
FEATS=FEATS+TE
pf=pq.ParquetFile(f'{W}/feat/train_feats.parquet')
cols=[f for f in FEATS if not (f.startswith('te_') or f.startswith('cnt_') or f=='same_cell05')]+['fare']
hc=np.load(f'{W}/feat/holdout_crc.npy')
if N>0:
    nrows=0; parts=[]; rg=0
    import pyarrow as pa
    while nrows<N:
        t=pf.read_row_group(rg,columns=cols); parts.append(t); nrows+=t.num_rows; rg+=1
    tab=pa.concat_tables(parts).slice(0,N); hc=hc[:N]
else:
    tab=pf.read(columns=cols)
NROWS=tab.num_rows
fare=tab['fare'].to_numpy(); X=np.empty((NROWS,len(FEATS)),dtype=np.float32)
base=len(FEATS)-len(TE)
for i,f in enumerate(FEATS[:base]): X[:,i]=tab[f].to_numpy(zero_copy_only=False)
del tab
tt=pq.read_table(f'{W}/feat/te_feats.parquet')
for j,f in enumerate(TE): X[:,base+j]=tt[f].to_numpy(zero_copy_only=False)[:NROWS]
del tt
print('loaded',NROWS,time.time()-t0,flush=True)
hp=X[:,FEATS.index('has_pickup')]>0.5; hd=X[:,FEATS.index('has_dropoff')]>0.5
p1=(X[:,FEATS.index('lon1')]>=-75.5)&(X[:,FEATS.index('lon1')]<=-72.5)&(X[:,FEATS.index('lat1')]>=39.5)&(X[:,FEATS.index('lat1')]<=42)
p2=(X[:,FEATS.index('lon2')]>=-75.5)&(X[:,FEATS.index('lon2')]<=-72.5)&(X[:,FEATS.index('lat2')]>=39.5)&(X[:,FEATS.index('lat2')]<=42)
ok=(fare>=1)&(fare<=500)&(p1|(~hp))&(p2|(~hd))
ok&=(X[:,FEATS.index('passenger_count')]>=0)&(X[:,FEATS.index('passenger_count')]<=8)
tr=ok&(~hc); va=ok&hc
print('train',int(tr.sum()),'val',int(va.sum()),flush=True)
m=CatBoostRegressor(iterations=ITERS,learning_rate=0.04,depth=10,loss_function='RMSE',
    l2_leaf_reg=3.0,task_type='GPU',border_count=254,random_seed=7,od_type='Iter',od_wait=100,verbose=50)
m.fit(X[tr],fare[tr],eval_set=(X[va],fare[va]),use_best_model=True)
pv=m.predict(X[va]); rmse=float(np.sqrt(np.mean((pv-fare[va])**2)))
print('VAL_RMSE',rmse,'best_iter',m.get_best_iteration(),time.time()-t0,flush=True)
m.save_model(f'{W}/{TAG}.cbm')
np.save(f'{W}/{TAG}_valpred.npy',pv.astype(np.float32)); np.save(f'{W}/{TAG}_valmask_ok.npy',np.where(va)[0])
# test
te=pq.read_table(f'{W}/feat/test_feats.parquet'); Xt=np.empty((te.num_rows,len(FEATS)),dtype=np.float32)
for i,f in enumerate(FEATS[:base]): Xt[:,i]=te[f].to_numpy(zero_copy_only=False)
tt=pq.read_table(f'{W}/feat/te_test.parquet')
for j,f in enumerate(TE): Xt[:,base+j]=tt[f].to_numpy(zero_copy_only=False)
np.save(f'{W}/{TAG}_testpred.npy',m.predict(Xt).astype(np.float32))
json.dump({'rmse':rmse,'best_iter':int(m.get_best_iteration()),'feats':FEATS},open(f'{W}/{TAG}_meta.json','w'))
print('DONE',rmse,time.time()-t0,flush=True)
