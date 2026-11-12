import numpy as np, pandas as pd, pyarrow.parquet as pq, xgboost as xgb, time, sys, json, os
ROOT='/download_job_code/payload/sandbox'; W=f'{ROOT}/workspaces/antares_omicron'
N=int(sys.argv[1]) if len(sys.argv)>1 else 0   # 0 = full
ROUNDS=int(sys.argv[2]) if len(sys.argv)>2 else 3000
TAG=sys.argv[3] if len(sys.argv)>3 else 'xgb'
t0=time.time()
FEATS=['lon1','lat1','lon2','lat2','passenger_count','year','month','day','hour','minute',
 'dow','doy','is_weekend','dist_km','dlat','dlon','abs_dlat','abs_dlon','manhattan_km','bearing',
 'p_jfk','d_jfk','min_jfk','p_lga','d_lga','min_lga','p_ewr','d_ewr','min_ewr',
 'p_midtown','d_midtown','min_midtown','p_dt','d_dt','min_dt','swap_flag','has_pickup','has_dropoff']
EXTRA=['ts_days','hour_sin','hour_cos','dow_sin','dow_cos','log_dist','euclid_deg','manh_ratio','dist_ratio','bearing_sin','bearing_cos','p_air_min','d_air_min','is_airport_trip']
if os.environ.get('USE_EXTRA','1')=='1': FEATS=FEATS+EXTRA
TE=['te_p01','cnt_p01','te_d01','cnt_d01','te_p05','cnt_p05','te_d05','cnt_d05','te_route05','cnt_route05','same_cell05']
USE_TE=os.path.exists(f'{W}/feat/te_feats.parquet')
FEATS=FEATS+TE if USE_TE else FEATS
pf=pq.ParquetFile(f'{W}/feat/train_feats.parquet')
cols=FEATS+['fare']
if N>0:
    # read only first N rows (file order random)
    tab=pf.read_row_group(0, columns=cols)
    need=N
    parts=[tab]
    got=tab.num_rows
    rg=1
    while got<need:
        t=pf.read_row_group(rg, columns=cols); parts.append(t); got+=t.num_rows; rg+=1
    import pyarrow as pa
    tab=pa.concat_tables(parts).slice(0,need)
    hc=np.load(f'{W}/feat/holdout_crc.npy')[:need]
else:
    tab=pf.read(columns=cols); hc=np.load(f'{W}/feat/holdout_crc.npy')
print('loaded',tab.num_rows,time.time()-t0,flush=True)
fare=tab['fare'].to_numpy(zero_copy_only=False)
NROWS=tab.num_rows
base_cols=[f for f in FEATS if f not in TE] if USE_TE else FEATS
nbase=len(base_cols)
X=np.empty((NROWS,len(FEATS)),dtype=np.float32)
for i,f in enumerate(base_cols): X[:,i]=tab[f].to_numpy(zero_copy_only=False)
del tab
if USE_TE:
    tt=pq.read_table(f'{W}/feat/te_feats.parquet')
    for j,f in enumerate(TE): X[:,nbase+j]=tt[f].to_numpy(zero_copy_only=False)[:NROWS]
    del tt
# clean
hp=X[:,FEATS.index('has_pickup')]>0.5; hd=X[:,FEATS.index('has_dropoff')]>0.5
ok=(fare>=1)&(fare<=500)
p1=(X[:,FEATS.index('lon1')]>= -75.5)&(X[:,FEATS.index('lon1')]<=-72.5)&(X[:,FEATS.index('lat1')]>=39.5)&(X[:,FEATS.index('lat1')]<=42.0)
p2=(X[:,FEATS.index('lon2')]>= -75.5)&(X[:,FEATS.index('lon2')]<=-72.5)&(X[:,FEATS.index('lat2')]>=39.5)&(X[:,FEATS.index('lat2')]<=42.0)
ok&=(p1|(~hp))&(p2|(~hd))
pc_i=FEATS.index('passenger_count'); ok&=(X[:,pc_i]>=0)&(X[:,pc_i]<=8)
print('clean kept',int(ok.sum()),'of',len(ok),time.time()-t0,flush=True)
tr=ok&(~hc); va=ok&hc
print('train',int(tr.sum()),'val',int(va.sum()),flush=True)
dtr=xgb.QuantileDMatrix(X[tr],label=fare[tr],max_bin=256)
dva=xgb.QuantileDMatrix(X[va],label=fare[va],ref=dtr,max_bin=256)
del X
GP=os.environ.get('GROW','depth')
params=dict(objective='reg:squarederror',eval_metric='rmse',tree_method='hist',device='cuda',
 eta=float(os.environ.get('ETA','0.03')),max_depth=int(os.environ.get('MAXDEPTH','11')),
 min_child_weight=float(os.environ.get('MCW','20')),subsample=0.8,colsample_bytree=0.8,
 reg_lambda=2.0,reg_alpha=0.0,max_bin=256)
if GP=='lossguide':
    params.update(grow_policy='lossguide',max_depth=0,max_leaves=int(os.environ.get('LEAVES','1024')),min_child_weight=float(os.environ.get('MCW','50')))
evals=[(dva,'val')]
bst=xgb.train(params,dtr,num_boost_round=ROUNDS,evals=evals,
    early_stopping_rounds=100,verbose_eval=25)
print('best',bst.best_iteration,bst.best_score,time.time()-t0,flush=True)
bst.save_model(f'{W}/{TAG}.json')
# val preds
pv=bst.predict(dva,iteration_range=(0,bst.best_iteration+1))
np.save(f'{W}/{TAG}_valpred.npy',pv.astype(np.float32))
np.save(f'{W}/{TAG}_valmask_ok.npy',np.where(va)[0])
rmse=float(np.sqrt(np.mean((pv-fare[va])**2)))
print('VAL_RMSE',rmse,flush=True)
# test preds
te=pq.read_table(f'{W}/feat/test_feats.parquet')
Xt=np.empty((te.num_rows,len(FEATS)),dtype=np.float32)
for i,f in enumerate(base_cols): Xt[:,i]=te[f].to_numpy(zero_copy_only=False)
if USE_TE:
    tt=pq.read_table(f'{W}/feat/te_test.parquet')
    for j,f in enumerate(TE): Xt[:,nbase+j]=tt[f].to_numpy(zero_copy_only=False)
pt=bst.predict(xgb.DMatrix(Xt,feature_names=FEATS,missing=np.nan),iteration_range=(0,bst.best_iteration+1))
np.save(f'{W}/{TAG}_testpred.npy',pt.astype(np.float32))
json.dump({'rmse':rmse,'best_iter':int(bst.best_iteration),'feats':FEATS,'N':int(tab.num_rows) if False else int(len(ok))},open(f'{W}/{TAG}_meta.json','w'),indent=1)
print('DONE',rmse,time.time()-t0,flush=True)
