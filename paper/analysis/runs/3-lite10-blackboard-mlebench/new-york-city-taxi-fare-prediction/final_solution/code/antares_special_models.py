import numpy as np, pyarrow.parquet as pq, xgboost as xgb, time, json
W='/download_job_code/payload/sandbox/workspaces/antares_omicron'
t0=time.time()
FEATS=['lon1','lat1','lon2','lat2','passenger_count','year','month','day','hour','minute','dow','doy','is_weekend',
 'ts_days','hour_sin','hour_cos','dow_sin','dow_cos',
 'p_jfk','d_jfk','min_jfk','p_lga','d_lga','min_lga','p_ewr','d_ewr','min_ewr',
 'p_midtown','d_midtown','p_dt','d_dt','dist_km','manhattan_km','bearing','swap_flag','has_pickup','has_dropoff']
t=pq.read_table(f'{W}/feat/train_feats.parquet',columns=FEATS+['fare'])
hc=np.load(f'{W}/feat/holdout_crc.npy')
X=np.empty((t.num_rows,len(FEATS)),dtype=np.float32)
for i,f in enumerate(FEATS): X[:,i]=t[f].to_numpy(zero_copy_only=False)
fare=t['fare'].to_numpy(); del t
hp=X[:,FEATS.index('has_pickup')]; hd=X[:,FEATS.index('has_dropoff')]; dk=X[:,FEATS.index('dist_km')]
ok=(fare>=1)&(fare<=500)
miss=ok&((hp<1)|(hd<1))
degen=ok&(hp>0.5)&(hd>0.5)&(dk<0.1)
print('miss',int(miss.sum()),'degen',int(degen.sum()),time.time()-t0,flush=True)
# test features
te=pq.read_table(f'{W}/feat/test_feats.parquet',columns=FEATS)
Xt=np.empty((te.num_rows,len(FEATS)),dtype=np.float32)
for i,f in enumerate(FEATS): Xt[:,i]=te[f].to_numpy(zero_copy_only=False)
thp=Xt[:,FEATS.index('has_pickup')]; thd=Xt[:,FEATS.index('has_dropoff')]; tdk=Xt[:,FEATS.index('dist_km')]
tmiss=((thp<1)|(thd<1)); tdegen=(thp>0.5)&(thd>0.5)&(tdk<0.1)
print('test miss',int(tmiss.sum()),'degen',int(tdegen.sum()),flush=True)
def run(mask, tmask, name):
    tr=mask&(~hc); va=mask&hc
    m=xgb.XGBRegressor(n_estimators=3000,learning_rate=0.03,max_depth=8,min_child_weight=20,
        subsample=0.8,colsample_bytree=0.8,reg_lambda=2.0,tree_method='hist',device='cuda',
        early_stopping_rounds=100,eval_metric='rmse',n_jobs=8)
    m.fit(X[tr],fare[tr],eval_set=[(X[va],fare[va])],verbose=100)
    pv=m.predict(X[va]); pv=np.clip(pv,2.5,200)
    rmse=float(np.sqrt(np.mean((pv-fare[va])**2)))
    print(name,'VAL_RMSE',rmse,'n',int(va.sum()),'best',m.best_iteration,time.time()-t0,flush=True)
    np.save(f'{W}/{name}_valpred.npy',pv.astype(np.float32)); np.save(f'{W}/{name}_valmask_ok.npy',np.where(va)[0])
    pt=np.clip(m.predict(Xt[tmask]),2.5,200).astype(np.float32)
    np.save(f'{W}/{name}_testpred.npy',pt); np.save(f'{W}/{name}_testmask.npy',np.where(tmask)[0])
    return rmse
r1=run(miss,tmiss,'spec_miss')
r2=run(degen,tdegen,'spec_degen')
json.dump({'spec_miss_rmse':r1,'spec_degen_rmse':r2},open(f'{W}/spec_meta.json','w'))
print('DONE',time.time()-t0)
