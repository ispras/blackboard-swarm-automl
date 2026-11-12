import glob, time
import numpy as np
import pyarrow.parquet as pq

ROOT='/download_job_code/payload/sandbox'
V2=sorted(glob.glob(f'{ROOT}/workspaces/mimosa_omicron/feat_v2/*.parquet'))
TE=f'{ROOT}/workspaces/mimosa_omicron/te_maps.npz'
OUT=f'{ROOT}/workspaces/aldebaran_omicron'
FEATS=[
 "pickup_latitude","pickup_longitude","dropoff_latitude","dropoff_longitude",
 "dlat","dlon","abs_dlat","abs_dlon","dist","dist_man","bearing",
 "p_jfk","d_jfk","p_lga","d_lga","p_ewr","d_ewr","p_airport_min","d_airport_min",
 "p_mid","d_mid","p_lower","d_lower","year","month","day","weekday","hour","minute","doy","week","is_weekend","hour_sin","hour_cos","passenger_count",
 "ru_p","rv_p","ru_d","rv_d","ru_delta","rv_delta","sum_jfk","sum_lga","sum_ewr","sum_airport_min",
 "p_near_airport","d_near_airport","both_near_airport","p_cell_count","d_cell_count","route_count"]
TEc=["te_pcell_oof","te_dcell_oof","te_route_oof"]
ALL=FEATS+TEc
te=np.load(TE)
NT=te['te_pcell_oof'].shape[0]
print('total',NT,'nfeat',len(ALL),flush=True)
X=np.lib.format.open_memmap(f'{OUT}/shared_X.npy',mode='w+',dtype=np.float32,shape=(NT,len(ALL)))
y=np.lib.format.open_memmap(f'{OUT}/shared_y.npy',mode='w+',dtype=np.float32,shape=(NT,))
oi=np.lib.format.open_memmap(f'{OUT}/shared_origidx.npy',mode='w+',dtype=np.int64,shape=(NT,))
off=0; t0=time.time()
for i,p in enumerate(V2):
    t=pq.read_table(p,columns=['orig_idx','fare_amount']+FEATS)
    n=t.num_rows; sl=slice(off,off+n)
    M=np.column_stack([t[c].to_numpy().astype(np.float32) for c in FEATS])
    for c in TEc:
        M=np.column_stack([M, te[c][sl].astype(np.float32)])
    X[sl]=M; y[sl]=t['fare_amount'].to_numpy().astype(np.float32); oi[sl]=t['orig_idx'].to_numpy().astype(np.int64)
    off+=n; del t,M
    if i%3==0: print('part',i,off,round(time.time()-t0),flush=True)
X.flush();y.flush();oi.flush()
np.save(f'{OUT}/shared_featnames.npy',np.array(ALL))
print('DONE',off,time.time()-t0,flush=True)
