import os, json
import numpy as np
import pandas as pd

ROOT = "/download_job_code/payload/sandbox"
DATA = os.path.join(ROOT, "data")
FEATDIR = os.path.join(ROOT, "workspaces/mimosa_omicron", "feat")

FEATS = [
    "pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude",
    "dlat", "dlon", "abs_dlat", "abs_dlon",
    "dist", "dist_man", "bearing",
    "p_jfk", "d_jfk", "p_lga", "d_lga", "p_ewr", "d_ewr",
    "p_airport_min", "d_airport_min",
    "p_mid", "d_mid", "p_lower", "d_lower",
    "year", "month", "day", "weekday", "hour", "minute", "doy", "week",
    "is_weekend", "hour_sin", "hour_cos", "passenger_count",
]
SPATIAL = [
    "pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude",
    "dlat", "dlon", "abs_dlat", "abs_dlon",
    "dist", "dist_man", "bearing",
    "p_jfk", "d_jfk", "p_lga", "d_lga", "p_ewr", "d_ewr",
    "p_airport_min", "d_airport_min",
    "p_mid", "d_mid", "p_lower", "d_lower",
]
NONSPATIAL = ["year", "month", "day", "weekday", "hour", "minute", "doy", "week",
              "is_weekend", "hour_sin", "hour_cos", "passenger_count"]


def parts():
    return sorted(os.path.join(FEATDIR, f) for f in os.listdir(FEATDIR) if f.endswith(".parquet"))


def load_features(parts_list=None, columns=None):
    """Return concatenated DataFrame with orig_idx, fare_amount + requested columns."""
    if parts_list is None:
        parts_list = parts()
    if columns is None:
        columns = ["orig_idx", "fare_amount"] + FEATS
    dfs = []
    for p in parts_list:
        dfs.append(pd.read_parquet(p, columns=columns))
    return pd.concat(dfs, ignore_index=True)

FEATS_V2 = FEATS + [
    "ru_p", "rv_p", "ru_d", "rv_d", "ru_delta", "rv_delta",
    "sum_jfk", "sum_lga", "sum_ewr", "sum_airport_min",
    "p_near_airport", "d_near_airport", "both_near_airport",
    "p_cell_count", "d_cell_count", "route_count",
]
TE_COLS = ["te_pcell_oof", "te_dcell_oof", "te_route_oof"]
V2DIR = os.path.join(ROOT, "workspaces/mimosa_omicron", "feat_v2")


def parts_v2():
    return sorted(os.path.join(V2DIR, f) for f in os.listdir(V2DIR) if f.endswith(".parquet"))


def load_te():
    d = np.load(os.path.join(ROOT, "workspaces/mimosa_omicron", "te_maps.npz"))
    return d
