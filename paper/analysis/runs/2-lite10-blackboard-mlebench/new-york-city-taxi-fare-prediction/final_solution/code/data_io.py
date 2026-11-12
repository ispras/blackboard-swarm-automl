import os, sys, time, json, argparse
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(BASE, '..', '..'))
sys.path.insert(0, BASE)
from fe_lib import FEATURES

TRAIN_PQ = os.path.join(BASE, 'features_train.parquet')
TEST_PQ = os.path.join(BASE, 'features_test.parquet')
HOLD_MASK = os.path.join(BASE, 'holdout_mask.npy')
RAND = os.path.join(BASE, 'sample_rand.npy')
HOLD_KEYS = os.path.join(BASE, 'holdout_keys.csv')

FARE_MIN, FARE_MAX = 0.5, 500.0
PAX_MIN, PAX_MAX = 0, 8
LON_MIN, LON_MAX = -75.0, -72.0
LAT_MIN, LAT_MAX = 39.0, 42.0

EXTRA = ['log_hav', 'ratio_hav_manh', 'is_same', 'bearing_sin', 'bearing_cos',
         'p_air_min', 'd_air_min', 'p_is_air', 'd_is_air']
FEATURES_ALL = FEATURES + EXTRA


_JFK = (40.6413, -73.7781); _LGA = (40.7769, -73.8740); _EWR = (40.6895, -74.1745)

def _hav(a, b, c, d):
    R = 6371.0
    p1 = np.radians(a); p2 = np.radians(c)
    dphi = np.radians(c - a); dl = np.radians(d - b)
    x = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return (2 * R * np.arcsin(np.sqrt(np.clip(x, 0, 1)))).astype(np.float32)

def add_extras(base):
    hav = base[:, FEATURES.index('haversine')].astype(np.float32)
    man = base[:, FEATURES.index('manhattan')].astype(np.float32)
    brg = base[:, FEATURES.index('bearing')].astype(np.float32)
    plat = base[:, FEATURES.index('pickup_latitude')]
    plon = base[:, FEATURES.index('pickup_longitude')]
    dlat = base[:, FEATURES.index('dropoff_latitude')]
    dlon = base[:, FEATURES.index('dropoff_longitude')]
    pmin = np.minimum(np.minimum(_hav(plat, plon, *_JFK), _hav(plat, plon, *_LGA)), _hav(plat, plon, *_EWR))
    dmin = np.minimum(np.minimum(_hav(dlat, dlon, *_JFK), _hav(dlat, dlon, *_LGA)), _hav(dlat, dlon, *_EWR))
    return np.column_stack([
        np.log1p(hav).astype(np.float32),
        (hav / (man * 111.0 + 0.01)).astype(np.float32),
        (hav < 0.01).astype(np.float32),
        np.sin(np.radians(brg)).astype(np.float32),
        np.cos(np.radians(brg)).astype(np.float32),
        pmin, dmin,
        (pmin < 1.0).astype(np.float32), (dmin < 1.0).astype(np.float32),
    ]).astype(np.float32)


def valid_mask(fare, plat, plon, dlat, dlon, pax):
    m = (fare >= FARE_MIN) & (fare <= FARE_MAX)
    m &= (pax >= PAX_MIN) & (pax <= PAX_MAX)
    m &= (plon >= LON_MIN) & (plon <= LON_MAX) & (dlon >= LON_MIN) & (dlon <= LON_MAX)
    m &= (plat >= LAT_MIN) & (plat <= LAT_MAX) & (dlat >= LAT_MIN) & (dlat <= LAT_MAX)
    m &= np.isfinite(fare) & np.isfinite(plat) & np.isfinite(plon) & np.isfinite(dlat) & np.isfinite(dlon)
    return m


def load(frac=0.25, log=print, holdout_all=True):
    hold = np.load(HOLD_MASK)
    rnd = np.load(RAND)
    cols = ['fare_amount'] + FEATURES
    pf = pq.ParquetFile(TRAIN_PQ)
    n = pf.metadata.num_rows
    F = len(FEATURES_ALL)
    cap_tr = int(frac * n) + 1000
    cap_h = int(0.011 * n) + 1000 if holdout_all else int(0.0102 * n) + 1000
    X = np.empty((cap_tr, F), dtype=np.float32)
    y = np.empty(cap_tr, dtype=np.float32)
    Xh = np.empty((cap_h, F), dtype=np.float32)
    yh = np.empty(cap_h, dtype=np.float32)
    hv = np.empty(cap_h, dtype=bool)
    ntr = nh = 0
    off = 0
    t0 = time.time()
    for batch in pf.iter_batches(batch_size=500_000, columns=cols):
        df = batch.to_pandas()
        b = len(df)
        sl = slice(off, off + b)
        fare = df['fare_amount'].to_numpy(np.float32)
        plat = df['pickup_latitude'].to_numpy(np.float32)
        plon = df['pickup_longitude'].to_numpy(np.float32)
        dlat = df['dropoff_latitude'].to_numpy(np.float32)
        dlon = df['dropoff_longitude'].to_numpy(np.float32)
        pax = df['passenger_count'].to_numpy(np.float32)
        v = valid_mask(fare, plat, plon, dlat, dlon, pax)
        h = hold[sl]
        r = rnd[sl]
        tr = v & (~h) & (r < frac)
        if tr.any():
            sub = df.loc[tr]
            base = sub[FEATURES].to_numpy(np.float32)
            k = base.shape[0]
            X[ntr:ntr+k, :len(FEATURES)] = base
            X[ntr:ntr+k, len(FEATURES):] = add_extras(base)
            y[ntr:ntr+k] = fare[tr]
            ntr += k
        if h.any():
            sub = df.loc[h]
            base = sub[FEATURES].to_numpy(np.float32)
            k = base.shape[0]
            Xh[nh:nh+k, :len(FEATURES)] = base
            Xh[nh:nh+k, len(FEATURES):] = add_extras(base)
            yh[nh:nh+k] = fare[h]
            vv = np.zeros(k, dtype=bool)
            vv[:] = v[h]
            hv[nh:nh+k] = vv
            nh += k
        off += b
    X = X[:ntr]; y = y[:ntr]
    Xh = Xh[:nh]; yh = yh[:nh]; hv = hv[:nh]
    log(f'loaded train={X.shape} hold={Xh.shape} valid_hold={int(hv.sum())} elapsed={time.time()-t0:.0f}s')
    return X, y, Xh, yh, hv


def load_test():
    df = pd.read_parquet(TEST_PQ)
    keys = df['key'].to_numpy()
    base = df[FEATURES].to_numpy(np.float32)
    X = np.concatenate([base, add_extras(base)], axis=1)
    return keys, X


def rmse(y, p):
    return float(np.sqrt(np.mean((y.astype(np.float64) - p.astype(np.float64))**2)))
