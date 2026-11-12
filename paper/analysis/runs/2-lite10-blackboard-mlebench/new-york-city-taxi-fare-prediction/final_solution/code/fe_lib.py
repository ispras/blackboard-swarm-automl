import numpy as np
import pandas as pd

JFK = (40.6413, -73.7781)
LGA = (40.7769, -73.8740)
EWR = (40.6895, -74.1745)
ANG = np.radians(30.0)
COS, SIN = np.cos(ANG), np.sin(ANG)

def _hav(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1 = np.radians(lat1); p2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dphi/2.0)**2 + np.cos(p1)*np.cos(p2)*np.sin(dl/2.0)**2
    return (2*R*np.arcsin(np.sqrt(np.clip(a, 0, 1)))).astype(np.float32)

def _bearing(lat1, lon1, lat2, lon2):
    p1 = np.radians(lat1); p2 = np.radians(lat2)
    dl = np.radians(lon2 - lon1)
    y = np.sin(dl)*np.cos(p2)
    x = np.cos(p1)*np.sin(p2) - np.sin(p1)*np.cos(p2)*np.cos(dl)
    return np.degrees(np.arctan2(y, x)).astype(np.float32)

def engineer(df):
    """df must have pickup/dropoff lon/lat, passenger_count, pickup_datetime 'YYYY-MM-DD HH:MM:SS UTC'."""
    plat = df['pickup_latitude'].to_numpy(np.float64)
    plon = df['pickup_longitude'].to_numpy(np.float64)
    dlat = df['dropoff_latitude'].to_numpy(np.float64)
    dlon = df['dropoff_longitude'].to_numpy(np.float64)
    passc = df['passenger_count'].to_numpy(np.float32)

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

    dlata = (dlat - plat).astype(np.float32)
    dlona = (dlon - plon).astype(np.float32)
    jfk = np.minimum(_hav(plat, plon, *JFK), _hav(dlat, dlon, *JFK))
    lga = np.minimum(_hav(plat, plon, *LGA), _hav(dlat, dlon, *LGA))
    ewr = np.minimum(_hav(plat, plon, *EWR), _hav(dlat, dlon, *EWR))
    air_min = np.minimum(np.minimum(jfk, lga), ewr)

    out = pd.DataFrame({
        'pickup_longitude': plon.astype(np.float32),
        'pickup_latitude': plat.astype(np.float32),
        'dropoff_longitude': dlon.astype(np.float32),
        'dropoff_latitude': dlat.astype(np.float32),
        'passenger_count': passc,
        'haversine': _hav(plat, plon, dlat, dlon),
        'manhattan': (np.abs(dlata) + np.abs(dlona)).astype(np.float32),
        'bearing': _bearing(plat, plon, dlat, dlon),
        'dlat': dlata,
        'dlon': dlona,
        'year': year, 'month': month, 'day': day, 'hour': hour,
        'minute': minute, 'weekday': dow, 'dayofyear': doy, 'is_weekend': is_wknd,
        'jfk_dist': jfk, 'lga_dist': lga, 'ewr_dist': ewr,
        'airport_min': air_min, 'is_airport': (air_min < 1.0).astype(np.float32),
        'rot_pu_x': (plon*COS + plat*SIN).astype(np.float32),
        'rot_pu_y': (plat*COS - plon*SIN).astype(np.float32),
        'rot_do_x': (dlon*COS + dlat*SIN).astype(np.float32),
        'rot_do_y': (dlat*COS - dlon*SIN).astype(np.float32),
    })
    return out

FEATURES = ['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude',
    'passenger_count','haversine','manhattan','bearing','dlat','dlon',
    'year','month','day','hour','minute','weekday','dayofyear','is_weekend',
    'jfk_dist','lga_dist','ewr_dist','airport_min','is_airport',
    'rot_pu_x','rot_pu_y','rot_do_x','rot_do_y']
