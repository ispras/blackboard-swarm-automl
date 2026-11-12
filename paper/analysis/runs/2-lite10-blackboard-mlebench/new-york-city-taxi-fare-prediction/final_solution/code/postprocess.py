import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
LON_MIN, LON_MAX = -75.0, -72.0
LAT_MIN, LAT_MAX = 39.0, 42.0
CLIP_LO = 2.5

def invalid_coord_mask(df):
    """True for rows whose coords carry no usable info (zero or outside NYC bbox)."""
    zero = ((df['pickup_latitude'] == 0) | (df['pickup_longitude'] == 0) |
            (df['dropoff_latitude'] == 0) | (df['dropoff_longitude'] == 0))
    oob = ~(df['pickup_longitude'].between(LON_MIN, LON_MAX) &
            df['dropoff_longitude'].between(LON_MIN, LON_MAX) &
            df['pickup_latitude'].between(LAT_MIN, LAT_MAX) &
            df['dropoff_latitude'].between(LAT_MIN, LAT_MAX))
    return (zero | oob).to_numpy()

def build_fallback_table(holdout_feats_path=None):
    """Group means of fare by (year,hour,passenger_count) computed on non-invalid rows."""
    if holdout_feats_path is None:
        holdout_feats_path = os.path.join(BASE, 'holdout_feats.parquet')
    h = pd.read_parquet(holdout_feats_path)
    m = ~invalid_coord_mask(h)
    h = h[m]
    g = h.groupby(['year', 'hour', 'passenger_count'])['fare_amount'].agg(['mean', 'count'])
    gmean = float(h['fare_amount'].mean())
    return g, gmean

def apply_postprocess(pred, feats, table, gmean, clip_lo=CLIP_LO):
    """pred: array aligned with feats (test features). Returns corrected array."""
    pred = np.asarray(pred, dtype=np.float64).copy()
    mask = invalid_coord_mask(feats)
    if mask.any():
        idx = pd.MultiIndex.from_arrays([feats['year'].to_numpy()[mask],
                                         feats['hour'].to_numpy()[mask],
                                         feats['passenger_count'].to_numpy()[mask]],
                                        names=['year', 'hour', 'passenger_count'])
        vals = table['mean'].reindex(idx).to_numpy()
        vals = np.where(np.isnan(vals), gmean, vals)
        pred[mask] = vals
    pred = np.clip(pred, clip_lo, None)
    return pred, mask
