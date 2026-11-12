import numpy as np, pandas as pd

R = 6371.0088

def hav(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    return 2.0*R*np.arcsin(np.sqrt(np.clip(a, 0, 1)))

def crc_holdout(keys):
    import zlib
    return np.fromiter((zlib.crc32(k.encode()) % 1000 < 4 for k in keys), dtype=bool, count=len(keys))

def fix_swaps(plon, plat, dlon, dlat):
    def fix(lon, lat):
        bad = ~((lon > -76) & (lon < -71) & (lat > 38) & (lat < 43))
        good_sw = (lat > -76) & (lat < -71) & (lon > 38) & (lon < 43)
        m = bad & good_sw
        lon2, lat2 = lon.copy(), lat.copy()
        lon2[m], lat2[m] = lat[m], lon[m]
        return lon2, lat2
    plon, plat = fix(plon, plat)
    dlon, dlat = fix(dlon, dlat)
    return plon, plat, dlon, dlat

def plaus(lon, lat):
    return (lon > -80) & (lon < -70) & (lat > 35) & (lat < 45)

def mark_missing(lon, lat):
    m = (lon == 0) & (lat == 0)
    m |= ~(np.isfinite(lon) & np.isfinite(lat) & plaus(lon, lat))
    return m

def build(df, is_test=False):
    out = pd.DataFrame(index=df.index)
    plon = df['pickup_longitude'].to_numpy(np.float64)
    plat = df['pickup_latitude'].to_numpy(np.float64)
    dlon = df['dropoff_longitude'].to_numpy(np.float64)
    dlat = df['dropoff_latitude'].to_numpy(np.float64)
    plon, plat, dlon, dlat = fix_swaps(plon, plat, dlon, dlat)
    pm = mark_missing(plon, plat)
    dm = mark_missing(dlon, dlat)
    plon[pm] = np.nan; plat[pm] = np.nan
    dlon[dm] = np.nan; dlat[dm] = np.nan

    out['pickup_longitude'] = plon.astype('float32')
    out['pickup_latitude'] = plat.astype('float32')
    out['dropoff_longitude'] = dlon.astype('float32')
    out['dropoff_latitude'] = dlat.astype('float32')
    out['passenger_count'] = df['passenger_count'].to_numpy(np.float32)
    out['has_pickup'] = (~pm).astype('int8')
    out['has_dropoff'] = (~dm).astype('int8')

    dt = pd.to_datetime(df['pickup_datetime'].str.slice(0, 19), format='%Y-%m-%d %H:%M:%S', errors='coerce')
    out['year'] = dt.dt.year.to_numpy(np.float32)
    out['month'] = dt.dt.month.to_numpy(np.float32)
    out['day'] = dt.dt.day.to_numpy(np.float32)
    out['hour'] = dt.dt.hour.to_numpy(np.float32)
    out['minute'] = dt.dt.minute.to_numpy(np.float32)
    out['weekday'] = dt.dt.weekday.to_numpy(np.float32)
    out['dayofyear'] = dt.dt.dayofyear.to_numpy(np.float32)
    out['weekofyear'] = dt.dt.isocalendar().week.to_numpy(np.float32)
    hf = out['hour'].to_numpy() + out['minute'].to_numpy()/60.0
    out['hour_float'] = hf.astype('float32')
    out['is_weekend'] = (out['weekday'] >= 5).astype('int8')

    dlatd = dlat - plat
    dlond = dlon - plon
    out['delta_lat'] = dlatd.astype('float32')
    out['delta_lon'] = dlond.astype('float32')
    dist = hav(plon, plat, dlon, dlat)
    out['distance_km'] = dist.astype('float32')
    mlat = np.radians((plat + dlat) / 2.0)
    dx = dlond * 111.32 * np.cos(mlat)
    dy = dlatd * 110.57
    out['manhattan_km'] = (np.abs(dx) + np.abs(dy)).astype('float32')
    out['l1_deg'] = (np.abs(dlatd) + np.abs(dlond)).astype('float32')
    out['log_distance'] = np.log1p(dist).astype('float32')
    bearing = np.degrees(np.arctan2(np.sin(np.radians(dlond))*np.cos(np.radians(dlat)),
                                    np.cos(np.radians(plat))*np.sin(np.radians(dlat)) -
                                    np.sin(np.radians(plat))*np.cos(np.radians(dlat))*np.cos(np.radians(dlond))))
    out['bearing'] = bearing.astype('float32')
    out['bearing_sin'] = np.sin(np.radians(bearing)).astype('float32')
    out['bearing_cos'] = np.cos(np.radians(bearing)).astype('float32')

    points = {
        'jfk': (40.6413, -73.7781), 'lga': (40.7769, -73.8740), 'ewr': (40.6895, -74.1745),
        'midtown': (40.7580, -73.9855), 'downtown': (40.7128, -74.0060), 'columbia': (40.8075, -73.9626),
    }
    for name, (lat, lon) in points.items():
        out[f'pdist_{name}'] = hav(plon, plat, lon, lat).astype('float32')
        out[f'ddist_{name}'] = hav(dlon, dlat, lon, lat).astype('float32')

    # flags based on pickup/dropoff presence; NaN distances compare False
    out['pickup_jfk'] = (out['pdist_jfk'] < 2.0).astype('int8')
    out['dropoff_jfk'] = (out['ddist_jfk'] < 2.0).astype('int8')
    out['pickup_lga'] = (out['pdist_lga'] < 2.0).astype('int8')
    out['dropoff_lga'] = (out['ddist_lga'] < 2.0).astype('int8')
    out['pickup_ewr'] = (out['pdist_ewr'] < 2.0).astype('int8')
    out['dropoff_ewr'] = (out['ddist_ewr'] < 2.0).astype('int8')
    out['airport_trip'] = ((out['pickup_jfk']+out['dropoff_jfk']+out['pickup_lga']+
                            out['dropoff_lga']+out['pickup_ewr']+out['dropoff_ewr']) > 0).astype('int8')

    out['key'] = df['key'].to_numpy()
    if is_test:
        return out
    out['fare_amount'] = df['fare_amount'].to_numpy(np.float32)
    out['holdout'] = crc_holdout(df['key'].to_numpy())
    return out

def clean_mask(df):
    fare = df['fare_amount'].to_numpy(np.float64)
    plon = df['pickup_longitude'].to_numpy(np.float64); plat = df['pickup_latitude'].to_numpy(np.float64)
    dlon = df['dropoff_longitude'].to_numpy(np.float64); dlat = df['dropoff_latitude'].to_numpy(np.float64)
    pax = df['passenger_count'].to_numpy(np.float64)
    plon, plat, dlon, dlat = fix_swaps(plon, plat, dlon, dlat)
    pm = mark_missing(plon, plat); dm = mark_missing(dlon, dlat)
    p_ok = pm | (np.isfinite(plon) & np.isfinite(plat) & plaus(plon, plat))
    d_ok = dm | (np.isfinite(dlon) & np.isfinite(dlat) & plaus(dlon, dlat))
    ok = (fare >= 1.0) & (fare <= 500.0) & (pax >= 0) & (pax <= 8) & p_ok & d_ok
    return ok

if __name__ == '__main__':
    import sys, time, pyarrow as pa, pyarrow.parquet as pq, os
    mode = sys.argv[1] if len(sys.argv) > 1 else 'train'
    t0 = time.time()
    outdir = 'workspaces/arcturus_upsilon/cache'
    os.makedirs(outdir, exist_ok=True)
    path = f'{outdir}/{mode}_v2.parquet'
    writer = None
    n_in = n_out = n_miss = 0
    chunk_iter = pd.read_csv('data/labels.csv', chunksize=2_000_000) if mode == 'train' else [pd.read_csv('data/test.csv')]
    for i, df in enumerate(chunk_iter):
        n_in += len(df)
        if mode == 'train':
            df = df.copy()
            plon = df['pickup_longitude'].to_numpy(np.float64); plat = df['pickup_latitude'].to_numpy(np.float64)
            dlon = df['dropoff_longitude'].to_numpy(np.float64); dlat = df['dropoff_latitude'].to_numpy(np.float64)
            plon, plat, dlon, dlat = fix_swaps(plon, plat, dlon, dlat)
            df['pickup_longitude'] = plon; df['pickup_latitude'] = plat
            df['dropoff_longitude'] = dlon; df['dropoff_latitude'] = dlat
            m = clean_mask(df)
            df = df.loc[m].reset_index(drop=True)
        feat = build(df, is_test=(mode != 'train'))
        n_miss += int((feat['has_pickup'] == 0).sum() + (feat['has_dropoff'] == 0).sum())
        table = pa.Table.from_pandas(feat, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(path, table.schema, compression='zstd')
        writer.write_table(table)
        n_out += len(feat)
        print(f'chunk {i}: in={len(df)} total_in={n_in} total_out={n_out} miss_pts={n_miss} t={time.time()-t0:.0f}s', flush=True)
        if mode != 'train':
            break
    writer.close()
    print(f'DONE {mode}: in={n_in} out={n_out} miss_pts={n_miss} time={time.time()-t0:.0f}s -> {path}')
