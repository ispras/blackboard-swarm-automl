import argparse, time, json, os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import xgboost as xgb

BASE = '/download_job_code/payload/sandbox'
PARQ = f'{BASE}/workspaces/aldebaran_omicron/features_train.parquet'
EXTRA = f'{BASE}/workspaces/aldebaran_omicron/extra_aligned.npy'
TEST = f'{BASE}/data/test.csv'
OUT = f'{BASE}/workspaces/aldebaran_omicron'
N = 55_413_942
LONMIN, LONMAX, LATMIN, LATMAX = -75.0, -72.0, 39.0, 42.0

RAW = ['plat','plon','dlat','dlon','rlat','rlon','arlat','arlon','dist',
       'dist_center_p','dist_jfk_p','dist_lga_p','dist_ewr_p','dist_center_d','bearing']
TIME = ['passenger_count','year','month','day','hour','minute','weekday','dayofyear','weekofyear']
DER = ['hour_sin','hour_cos','is_weekend','p_rotx','p_roty','d_rotx','d_roty','drx','dry','airp_min_p']
EXT = ['te_pcell','te_dcell','te_route','p_cell_count','d_cell_count','route_count']
FEATS = RAW + TIME + DER + EXT
SPATIAL = set(RAW + ['p_rotx','p_roty','d_rotx','d_roty','drx','dry','airp_min_p'])

C_LAT, C_LON = 40.7580, -73.9855
COSC = np.cos(np.radians(C_LAT))
TH = np.radians(29.0); CTH = np.cos(TH); STH = np.sin(TH)

def bad(lon, lat):
    return (lon < LONMIN) | (lon > LONMAX) | (lat < LATMIN) | (lat > LATMAX)

def derive(d):
    h = d['hour']
    d['hour_sin'] = np.sin(h/24.0*2*np.pi).astype(np.float32)
    d['hour_cos'] = np.cos(h/24.0*2*np.pi).astype(np.float32)
    d['is_weekend'] = (d['weekday'] >= 5).astype(np.float32)
    up = (d['plon']-C_LON)*COSC*111.32; vp = (d['plat']-C_LAT)*111.32
    ud = (d['dlon']-C_LON)*COSC*111.32; vd = (d['dlat']-C_LAT)*111.32
    d['p_rotx'] = (up*CTH + vp*STH).astype(np.float32); d['p_roty'] = (-up*STH + vp*CTH).astype(np.float32)
    d['d_rotx'] = (ud*CTH + vd*STH).astype(np.float32); d['d_roty'] = (-ud*STH + vd*CTH).astype(np.float32)
    d['drx'] = (ud-up).astype(np.float32); d['dry'] = (vd-vp).astype(np.float32)
    d['airp_min_p'] = np.minimum(np.minimum(d['dist_jfk_p'],d['dist_lga_p']),d['dist_ewr_p']).astype(np.float32)
    return d

def load(sample_frac=1.0, seed=0):
    pf = pq.ParquetFile(PARQ)
    extra = np.load(EXTRA, mmap_mode='r')
    cap = 54_350_000
    Xtr = np.empty((cap, len(FEATS)), dtype=np.float32); ytr = np.empty(cap, np.float32)
    Vcap = 700_000
    Xva = np.empty((Vcap, len(FEATS)), dtype=np.float32); yva = np.empty(Vcap, np.float32)
    ova = np.empty(Vcap, np.int64)
    ptr = pv = 0; off = 0
    fbsum = 0.0; fbcount = 0
    thr = sample_frac
    t0 = time.time()
    for batch in pf.iter_batches(batch_size=750_000):
        cols = RAW + TIME + ['fare_amount']
        d = {c: np.array(batch.column(c).to_numpy(zero_copy_only=False), dtype=np.float32, copy=True) for c in cols}
        n = len(d['fare_amount']); pos = np.arange(off, off+n, dtype=np.int64); off += n
        d = derive(d)
        cm = bad(d['plon'], d['plat']) | bad(d['dlon'], d['dlat'])
        valid = ~cm
        fare = d['fare_amount']
        pax_ok = (d['passenger_count'] >= 0) & (d['passenger_count'] <= 8)
        clean = (fare >= 2.5) & (fare <= 500.0) & pax_ok & np.isfinite(fare)
        ex = np.asarray(extra[off-n:off], dtype=np.float32)
        for j, c in enumerate(EXT): d[c] = ex[:, j]
        M = np.column_stack([d[c] for c in FEATS])
        hold = (pos % 100 == 0)
        hsh = ((pos * 2654435761) & 0xFFFFFFFF) / 4294967296.0
        vs = clean & valid & hold
        k = int(vs.sum())
        if pv + k <= Vcap:
            Xva[pv:pv+k] = M[vs]; yva[pv:pv+k] = fare[vs]; ova[pv:pv+k] = pos[vs]; pv += k
        ts = clean & valid & (~hold) & (hsh < thr)
        k2 = int(ts.sum())
        if ptr + k2 > cap:
            raise MemoryError(f'cap exceeded {ptr+k2}')
        Xtr[ptr:ptr+k2] = M[ts]; ytr[ptr:ptr+k2] = fare[ts]; ptr += k2
        cs = clean & cm & (~hold)
        fbsum += float(fare[cs].sum()); fbcount += int(cs.sum())
    print(f'loaded train {ptr} val {pv} fallback {fbcount} in {time.time()-t0:.0f}s', flush=True)
    return Xtr[:ptr], ytr[:ptr], Xva[:pv], yva[:pv], ova[:pv], fbsum/max(fbcount,1)

def havarr(a,b,c,e):
    a=np.radians(a); c=np.radians(c); dl=c-a; dn=np.radians(e-b)
    x=np.sin(dl/2)**2+np.cos(a)*np.cos(c)*np.sin(dn/2)**2
    return (2*6371.0*np.arcsin(np.sqrt(np.clip(x,0,1)))).astype(np.float32)

def cellkey(lat, lon, res=0.01):
    clat = np.floor((lat-40.0)/res).astype(np.int64)
    clon = np.floor((lon+75.0)/res).astype(np.int64)
    return clat*10000+clon

def build_test():
    te = np.load(f'{BASE}/workspaces/mimosa_omicron/te_maps.npz')
    t = pd.read_csv(TEST)
    d = {}
    plat=t['pickup_latitude'].values.astype(np.float32); plon=t['pickup_longitude'].values.astype(np.float32)
    dlat=t['dropoff_latitude'].values.astype(np.float32); dlon=t['dropoff_longitude'].values.astype(np.float32)
    d['plat']=plat; d['plon']=plon; d['dlat']=dlat; d['dlon']=dlon
    d['rlat']=(dlat-plat).astype(np.float32); d['rlon']=(dlon-plon).astype(np.float32)
    d['arlat']=np.abs(dlat-plat).astype(np.float32); d['arlon']=np.abs(dlon-plon).astype(np.float32)
    d['dist']=havarr(plat,plon,dlat,dlon)
    d['dist_center_p']=havarr(plat,plon,C_LAT,C_LON)
    d['dist_jfk_p']=havarr(plat,plon,40.6413,-73.7781)
    d['dist_lga_p']=havarr(plat,plon,40.7769,-73.8740)
    d['dist_ewr_p']=havarr(plat,plon,40.6895,-74.1745)
    d['dist_center_d']=havarr(dlat,dlon,C_LAT,C_LON)
    yy=np.sin(np.radians(dlon-plon))*np.cos(np.radians(dlat))
    xx=np.cos(np.radians(plat))*np.sin(np.radians(dlat))-np.sin(np.radians(plat))*np.cos(np.radians(dlat))*np.cos(np.radians(dlon-plon))
    d['bearing']=np.degrees(np.arctan2(yy,xx)).astype(np.float32)
    dt=pd.to_datetime(t['pickup_datetime'],utc=True)
    d['year']=dt.dt.year.astype(np.float32); d['month']=dt.dt.month.astype(np.float32)
    d['day']=dt.dt.day.astype(np.float32); d['hour']=dt.dt.hour.astype(np.float32)
    d['minute']=dt.dt.minute.astype(np.float32); d['weekday']=dt.dt.weekday.astype(np.float32)
    d['dayofyear']=dt.dt.dayofyear.astype(np.float32); d['weekofyear']=dt.dt.isocalendar().week.values.astype(np.float32)
    d['passenger_count']=t['passenger_count'].astype(np.float32)
    d=derive(d)
    cm = bad(plon,plat)|bad(dlon,dlat)
    pc = cellkey(plat,plon); dc = cellkey(dlat,dlon)
    rt = (pc+600000)*4000000+(dc+600000)
    def look(u, enc, cnt, key):
        key = key.astype(np.int64)
        pos = np.searchsorted(u, key)
        ok = (pos < len(u)) & (u[np.clip(pos,0,len(u)-1)] == key)
        e = np.where(ok, enc[np.clip(pos,0,len(u)-1)], np.nan).astype(np.float32)
        c = np.where(ok, cnt[np.clip(pos,0,len(u)-1)], np.nan).astype(np.float32)
        return e, c
    tp, cp = look(te['u_pcell'], te['enc_pcell_full'], te['cnt_pcell'], pc)
    td, cd = look(te['u_dcell'], te['enc_dcell_full'], te['cnt_dcell'], dc)
    tr, cr = look(te['u_route'], te['enc_route_full'], te['cnt_route'], rt)
    d['te_pcell']=tp; d['te_dcell']=td; d['te_route']=tr
    d['p_cell_count']=cp; d['d_cell_count']=cd; d['route_count']=cr
    for c in SPATIAL:
        d[c] = np.where(cm, np.nan, d[c]).astype(np.float32)
    X = np.column_stack([np.asarray(d[c], dtype=np.float32) for c in FEATS])
    return t['key'].values, X, cm

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--eta',type=float,default=0.03)
    ap.add_argument('--depth',type=int,default=10)
    ap.add_argument('--min_child_weight',type=float,default=300)
    ap.add_argument('--subsample',type=float,default=0.8)
    ap.add_argument('--colsample',type=float,default=0.8)
    ap.add_argument('--l2',type=float,default=1.0)
    ap.add_argument('--max_bin',type=int,default=128)
    ap.add_argument('--rounds',type=int,default=6000)
    ap.add_argument('--early',type=int,default=150)
    ap.add_argument('--seed',type=int,default=7)
    ap.add_argument('--threads',type=int,default=16)
    ap.add_argument('--sample_frac',type=float,default=1.0)
    ap.add_argument('--tag',type=str,default='xgbA')
    a=ap.parse_args()
    print(json.dumps(vars(a)), flush=True)
    Xtr,ytr,Xva,yva,ova,fbmean = load(a.sample_frac,a.seed)
    dtr = xgb.QuantileDMatrix(Xtr, label=ytr, max_bin=a.max_bin, feature_names=FEATS, nthread=a.threads)
    dva = xgb.QuantileDMatrix(Xva, label=yva, max_bin=a.max_bin, feature_names=FEATS, ref=dtr, nthread=a.threads)
    del Xtr; import gc; gc.collect()
    params={'device':'cuda','tree_method':'hist','max_bin':a.max_bin,
            'objective':'reg:squarederror','eval_metric':'rmse',
            'eta':a.eta,'max_depth':a.depth,'min_child_weight':a.min_child_weight,
            'subsample':a.subsample,'colsample_bytree':a.colsample,
            'reg_lambda':a.l2,'seed':a.seed,'nthread':a.threads,'verbosity':1}
    t0=time.time()
    bst=xgb.train(params,dtr,num_boost_round=a.rounds,evals=[(dva,'val')],
                  early_stopping_rounds=a.early,verbose_eval=100)
    best=bst.best_iteration
    pv=bst.predict(dva,iteration_range=(0,best+1))
    rmse=float(np.sqrt(np.mean((pv-np.asarray(yva))**2)))
    print(f'BEST_ITER={best} HOLDOUT_VALID_RMSE={rmse:.5f} time={time.time()-t0:.0f}s',flush=True)
    bst.save_model(f'{OUT}/model_{a.tag}.json')
    np.save(f'{OUT}/holdout_pred_{a.tag}.npy',pv.astype(np.float32))
    np.save(f'{OUT}/holdout_orig_{a.tag}.npy',ova)
    np.save(f'{OUT}/holdout_true_{a.tag}.npy',yva.astype(np.float32))
    keys,Xte,cm=build_test()
    tp=bst.predict(xgb.DMatrix(Xte,feature_names=FEATS,nthread=a.threads),iteration_range=(0,best+1))
    tp=np.asarray(tp,dtype=np.float32)
    tp[cm]=fbmean
    np.save(f'{OUT}/test_pred_{a.tag}.npy',tp)
    pd.DataFrame({'key':keys,'fare_amount':np.clip(tp,0,None)}).to_csv(f'{OUT}/submission_{a.tag}.csv',index=False)
    with open(f'{OUT}/result_{a.tag}.json','w') as f:
        json.dump({'tag':a.tag,'rmse':rmse,'best_iter':int(best),'fbmean':fbmean,
                   'args':vars(a),'time_s':time.time()-t0},f,indent=2)
    print('test corrupt',int(cm.sum()),'mean',float(tp.mean()),flush=True)

if __name__=='__main__':
    main()
