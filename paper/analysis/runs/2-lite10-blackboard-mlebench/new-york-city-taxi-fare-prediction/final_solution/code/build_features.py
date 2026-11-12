import os, sys, time
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pv
import pyarrow.compute as pc
import pyarrow.parquet as pq

SRC = "data/labels.csv"
OUT = "workspaces/deneb_phi/features_train.parquet"
BATCH_ROWS = 2_000_000

# airports
JFK = (40.6413, -73.7781)
LGA = (40.7769, -73.8740)
EWR = (40.6895, -74.1745)

def hav(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1 = np.radians(lat1); p2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dphi/2.0)**2 + np.cos(p1)*np.cos(p2)*np.sin(dl/2.0)**2
    return (2*R*np.arcsin(np.sqrt(np.clip(a, 0, 1)))).astype(np.float32)

def bearing(lat1, lon1, lat2, lon2):
    p1 = np.radians(lat1); p2 = np.radians(lat2)
    dl = np.radians(lon2 - lon1)
    y = np.sin(dl)*np.cos(p2)
    x = np.cos(p1)*np.sin(p2) - np.sin(p1)*np.cos(p2)*np.cos(dl)
    return np.degrees(np.arctan2(y, x)).astype(np.float32)

ANG = np.radians(30.0)
COS, SIN = np.cos(ANG), np.sin(ANG)

def process(batch):
    df = batch.to_pandas()
    keys = batch.column('key').cast(pa.int64()).to_numpy(zero_copy_only=False)
    is_hold = (np.mod(keys, 100) == 0).astype(np.int8)

    plat = df['pickup_latitude'].to_numpy(np.float64)
    plon = df['pickup_longitude'].to_numpy(np.float64)
    dlat = df['dropoff_latitude'].to_numpy(np.float64)
    dlon = df['dropoff_longitude'].to_numpy(np.float64)
    fare = df['fare_amount'].to_numpy(np.float32)
    passc = df['passenger_count'].to_numpy(np.float32)

    # time
    s = df['pickup_datetime'].astype(str).str.slice(0, 19)
    dt = pd.to_datetime(s, format='%Y-%m-%d %H:%M:%S', errors='coerce')
    year = dt.dt.year.to_numpy(np.float32)
    month = dt.dt.month.to_numpy(np.float32)
    day = dt.dt.day.to_numpy(np.float32)
    hour = dt.dt.hour.to_numpy(np.float32)
    minute = dt.dt.minute.to_numpy(np.float32)
    dow = dt.dt.dayofweek.to_numpy(np.float32)
    doy = dt.dt.dayofyear.to_numpy(np.float32)
    is_wknd = (dow >= 5).astype(np.float32)

    hd = hav(plat, plon, dlat, dlon)
    dlata = (dlat - plat).astype(np.float32)
    dlona = (dlon - plon).astype(np.float32)
    manh = (np.abs(dlata) + np.abs(dlona)).astype(np.float32)
    brg = bearing(plat, plon, dlat, dlon)

    jfk = np.minimum(hav(plat, plon, *JFK), hav(dlat, dlon, *JFK))
    lga = np.minimum(hav(plat, plon, *LGA), hav(dlat, dlon, *LGA))
    ewr = np.minimum(hav(plat, plon, *EWR), hav(dlat, dlon, *EWR))
    air_min = np.minimum(np.minimum(jfk, lga), ewr)
    is_air = (air_min < 1.0).astype(np.float32)

    pux = (plon*COS + plat*SIN).astype(np.float32)
    puy = (plat*COS - plon*SIN).astype(np.float32)
    dox = (dlon*COS + dlat*SIN).astype(np.float32)
    doy2 = (dlat*COS - dlon*SIN).astype(np.float32)

    out = pd.DataFrame({
        'is_holdout': is_hold,
        'fare_amount': fare,
        'pickup_longitude': plon.astype(np.float32),
        'pickup_latitude': plat.astype(np.float32),
        'dropoff_longitude': dlon.astype(np.float32),
        'dropoff_latitude': dlat.astype(np.float32),
        'passenger_count': passc,
        'haversine': hd,
        'manhattan': manh,
        'bearing': brg,
        'dlat': dlata,
        'dlon': dlona,
        'year': year,
        'month': month,
        'day': day,
        'hour': hour,
        'minute': minute,
        'weekday': dow,
        'dayofyear': doy,
        'is_weekend': is_wknd,
        'jfk_dist': jfk,
        'lga_dist': lga,
        'ewr_dist': ewr,
        'airport_min': air_min,
        'is_airport': is_air,
        'rot_pu_x': pux,
        'rot_pu_y': puy,
        'rot_do_x': dox,
        'rot_do_y': doy2,
    })
    return pa.Table.from_pandas(out, preserve_index=False)

def main():
    t0 = time.time()
    reader = pv.open_csv(SRC)
    writer = None
    total = 0
    for i, batch in enumerate(reader):
        tbl = process(batch)
        if writer is None:
            writer = pq.ParquetWriter(OUT, tbl.schema, compression='snappy')
        writer.write_table(tbl)
        total += tbl.num_rows
        print(f"batch {i} rows={tbl.num_rows} total={total} t={time.time()-t0:.0f}s", flush=True)
    if writer:
        writer.close()
    print(f"DONE total={total} elapsed={time.time()-t0:.0f}s")

if __name__ == '__main__':
    main()
