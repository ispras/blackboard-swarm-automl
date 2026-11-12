import sys, time, os
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

SRC = '/download_job_code/payload/sandbox/data/labels.csv'
OUT = '/download_job_code/payload/sandbox/workspaces/aldebaran_omicron/features_train.parquet'
CHUNK = 1_500_000

usecols = ['fare_amount','pickup_datetime','pickup_longitude','pickup_latitude',
           'dropoff_longitude','dropoff_latitude','passenger_count']
dtypes = {'fare_amount':'float32','pickup_longitude':'float32','pickup_latitude':'float32',
          'dropoff_longitude':'float32','dropoff_latitude':'float32','passenger_count':'float32'}

R = 6371.0
CENTER = (40.7580, -73.9855)
JFK = (40.6413, -73.7781)
LGA = (40.7769, -73.8740)
EWR = (40.6895, -74.1745)
NYC = (40.7128, -74.0060)

def hav(lat1, lon1, lat2, lon2):
    lat1 = np.radians(lat1); lat2 = np.radians(lat2)
    dlat = lat2 - lat1
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    return (2*R*np.arcsin(np.sqrt(np.clip(a,0,1)))).astype(np.float32)

writer = None
t0 = time.time()
n = 0
for i, df in enumerate(pd.read_csv(SRC, usecols=usecols, dtype=dtypes, chunksize=CHUNK)):
    plat = df['pickup_latitude'].values; plon = df['pickup_longitude'].values
    dlat = df['dropoff_latitude'].values; dlon = df['dropoff_longitude'].values
    out = pd.DataFrame()
    out['fare_amount'] = df['fare_amount'].values
    out['passenger_count'] = df['passenger_count'].values
    out['plat'] = plat; out['plon'] = plon; out['dlat'] = dlat; out['dlon'] = dlon
    out['rlat'] = (dlat - plat).astype(np.float32)
    out['rlon'] = (dlon - plon).astype(np.float32)
    out['arlat'] = np.abs(dlat - plat).astype(np.float32)
    out['arlon'] = np.abs(dlon - plon).astype(np.float32)
    out['dist'] = hav(plat, plon, dlat, dlon)
    out['dist_center_p'] = hav(plat, plon, CENTER[0], CENTER[1])
    out['dist_jfk_p'] = hav(plat, plon, JFK[0], JFK[1])
    out['dist_lga_p'] = hav(plat, plon, LGA[0], LGA[1])
    out['dist_ewr_p'] = hav(plat, plon, EWR[0], EWR[1])
    out['dist_center_d'] = hav(dlat, dlon, CENTER[0], CENTER[1])
    # bearing
    y = np.sin(np.radians(dlon - plon)) * np.cos(np.radians(dlat))
    x = np.cos(np.radians(plat))*np.sin(np.radians(dlat)) - np.sin(np.radians(plat))*np.cos(np.radians(dlat))*np.cos(np.radians(dlon-plon))
    out['bearing'] = np.degrees(np.arctan2(y, x)).astype(np.float32)
    # datetime
    dt = pd.to_datetime(df['pickup_datetime'], utc=True, format='%Y-%m-%d %H:%M:%S UTC', errors='coerce')
    out['year'] = dt.dt.year.astype(np.float32)
    out['month'] = dt.dt.month.astype(np.float32)
    out['day'] = dt.dt.day.astype(np.float32)
    out['hour'] = dt.dt.hour.astype(np.float32)
    out['minute'] = dt.dt.minute.astype(np.float32)
    out['weekday'] = dt.dt.weekday.astype(np.float32)
    out['dayofyear'] = dt.dt.dayofyear.astype(np.float32)
    out['weekofyear'] = dt.dt.isocalendar().week.values.astype(np.float32)
    table = pa.Table.from_pandas(out, preserve_index=False)
    if writer is None:
        writer = pq.ParquetWriter(OUT, table.schema, compression='zstd')
    writer.write_table(table)
    n += len(out)
    if i % 5 == 0:
        print(f'{time.time()-t0:.1f}s chunk {i} rows {n}', flush=True)
if writer is not None:
    writer.close()
print('DONE rows', n, 'time', time.time()-t0)
