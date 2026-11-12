import numpy as np
import pandas as pd
import os, sys, time

DATA = "/download_job_code/payload/sandbox/data"
OUT = "/download_job_code/payload/sandbox/workspaces/alkaid_phi"
os.makedirs(OUT, exist_ok=True)

R = 6371.0
def hav(lat1, lon1, lat2, lon2):
    lat1 = np.radians(lat1); lat2 = np.radians(lat2)
    dlat = lat2 - lat1
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    return 2*R*np.arcsin(np.sqrt(np.clip(a, 0, 1)))

def feat(df):
    plat = df['pickup_latitude'].values.astype(np.float64)
    plon = df['pickup_longitude'].values.astype(np.float64)
    dlat = df['dropoff_latitude'].values.astype(np.float64)
    dlon = df['dropoff_longitude'].values.astype(np.float64)
    out = {}
    out['pickup_latitude'] = plat; out['pickup_longitude'] = plon
    out['dropoff_latitude'] = dlat; out['dropoff_longitude'] = dlon
    out['passenger_count'] = df['passenger_count'].values.astype(np.float32)
    out['dlat'] = (dlat-plat).astype(np.float32); out['dlon'] = (dlon-plon).astype(np.float32)
    out['abs_dlat'] = np.abs(dlat-plat).astype(np.float32); out['abs_dlon'] = np.abs(dlon-plon).astype(np.float32)
    dh = hav(plat, plon, dlat, dlon)
    out['haversine'] = dh.astype(np.float32)
    # manhattan via intermediate point
    dh2 = hav(plat, plon, plat, dlon) + hav(plat, dlon, dlat, dlon)
    out['manhattan'] = dh2.astype(np.float32)
    out['euclid_deg'] = np.sqrt((dlat-plat)**2 + (dlon-plon)**2).astype(np.float32)
    # bearing
    y = np.sin(np.radians(dlon-plon))*np.cos(np.radians(dlat))
    x = np.cos(np.radians(plat))*np.sin(np.radians(dlat)) - np.sin(np.radians(plat))*np.cos(np.radians(dlat))*np.cos(np.radians(dlon-plon))
    out['bearing'] = np.degrees(np.arctan2(y, x)).astype(np.float32)
    out['bearing_sin'] = np.sin(np.radians(out['bearing'])).astype(np.float32)
    out['bearing_cos'] = np.cos(np.radians(out['bearing'])).astype(np.float32)
    # airports and center
    pts = {'jfk': (40.6413, -73.7781), 'lga': (40.7769, -73.8740),
           'ewr': (40.6895, -74.1745), 'ctr': (40.7580, -73.9855)}
    for name, (la, lo) in pts.items():
        dp = hav(plat, plon, la, lo); dd = hav(dlat, dlon, la, lo)
        out['p_'+name] = dp.astype(np.float32); out['d_'+name] = dd.astype(np.float32)
        out['pd_'+name] = (dp-dd).astype(np.float32)
    # datetime
    dt = pd.to_datetime(df['pickup_datetime'], format='%Y-%m-%d %H:%M:%S UTC', utc=True)
    out['year'] = dt.dt.year.values.astype(np.int16)
    out['month'] = dt.dt.month.values.astype(np.int8)
    out['day'] = dt.dt.day.values.astype(np.int8)
    out['weekday'] = dt.dt.weekday.values.astype(np.int8)
    out['hour'] = dt.dt.hour.values.astype(np.int8)
    out['minute'] = dt.dt.minute.values.astype(np.int8)
    out['weekofyear'] = dt.dt.isocalendar().week.values.astype(np.int8)
    # timestamp seconds
    out['ts'] = (dt.view('int64')//10**9).values.astype(np.int64)
    return out

def main(p_keep):
    parts = []
    n_in = n_out = 0
    t0 = time.time()
    rng = np.random.default_rng(42)
    for i, chunk in enumerate(pd.read_csv(f"{DATA}/labels.csv", chunksize=2_000_000)):
        n_in += len(chunk)
        m = (chunk.fare_amount >= 0.5) & (chunk.fare_amount <= 500)
        m &= chunk.pickup_longitude.between(-75.5, -72.0) & chunk.pickup_latitude.between(39.0, 42.0)
        m &= chunk.dropoff_longitude.between(-75.5, -72.0) & chunk.dropoff_latitude.between(39.0, 42.0)
        m &= chunk.passenger_count.between(0, 9)
        c = chunk[m]
        if len(c)==0: continue
        keep = rng.random(len(c)) < p_keep
        c = c[keep]
        if len(c)==0: continue
        f = feat(c)
        f['fare_amount'] = c['fare_amount'].values.astype(np.float32)
        f['key'] = c['key'].values
        d = pd.DataFrame(f)
        import zlib
        d['holdout'] = np.array([ (zlib.crc32(k.encode()) % 1000) < 4 for k in d['key'].values ])
        parts.append(d)
        n_out += len(c)
        del chunk, m, c, f
        if i % 5 == 0:
            print(f"chunk {i} in={n_in} kept={n_out} t={time.time()-t0:.0f}s", flush=True)
    df = pd.concat(parts, ignore_index=True)
    print("total", df.shape, flush=True)
    df.to_parquet(f"{OUT}/train_feat.parquet", index=False)
    print("saved", flush=True)

if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv)>1 else 0.30)
