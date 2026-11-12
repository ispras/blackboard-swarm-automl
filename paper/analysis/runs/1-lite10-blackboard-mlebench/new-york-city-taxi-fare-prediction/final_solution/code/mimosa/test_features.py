import os
import numpy as np
import pandas as pd
import prep
from add_features import augment
from common import FEATS_V2
from recover_coords import fix, valid

ROOT = "/download_job_code/payload/sandbox"


def build_test():
    t = pd.read_csv(os.path.join(ROOT, "data", "test.csv"))
    t["fare_amount"] = 0.0
    plon0 = t["pickup_longitude"].to_numpy(np.float64); plat0 = t["pickup_latitude"].to_numpy(np.float64)
    dlon0 = t["dropoff_longitude"].to_numpy(np.float64); dlat0 = t["dropoff_latitude"].to_numpy(np.float64)
    orig_bad = ~(valid(plon0, plat0) & valid(dlon0, dlat0))
    fplo, fpla = fix(plon0, plat0)
    fdlo, fdla = fix(dlon0, dlat0)
    recovered = orig_bad & valid(fplo, fpla) & valid(fdlo, fdla)
    t["pickup_longitude"] = np.where(recovered, fplo, plon0)
    t["pickup_latitude"] = np.where(recovered, fpla, plat0)
    t["dropoff_longitude"] = np.where(recovered, fdlo, dlon0)
    t["dropoff_latitude"] = np.where(recovered, fdla, dlat0)
    df = prep.build(t, 0)
    df["pickup_datetime"] = t["pickup_datetime"].to_numpy()
    del t
    df = augment(df)
    plat = df["pickup_latitude"].to_numpy(np.float32)
    plon = df["pickup_longitude"].to_numpy(np.float32)
    dlat = df["dropoff_latitude"].to_numpy(np.float32)
    dlon = df["dropoff_longitude"].to_numpy(np.float32)
    bad = ~(np.isfinite(plat) & np.isfinite(plon) & np.isfinite(dlat) & np.isfinite(dlon))
    bad |= ~((plon > -75.5) & (plon < -72.5) & (plat > 39.5) & (plat < 42.0) &
             (dlon > -75.5) & (dlon < -72.5) & (dlat > 39.5) & (dlat < 42.0))
    bad |= ~((df["dist"].to_numpy() >= 0) & (df["dist"].to_numpy() <= 300))
    df["corrupt"] = (orig_bad & ~recovered).astype(np.float32)
    return df


def add_counts_te(df, te):
    pc = df["p_cell"].to_numpy(np.int64)
    dc = df["d_cell"].to_numpy(np.int64)
    rt = df["route"].to_numpy(np.int64)
    for nm, arr in [("pcell", pc), ("dcell", dc), ("route", rt)]:
        u = te[f"u_{nm}"]
        enc = te[f"enc_{nm}_full"]
        cnt = te[f"cnt_{nm}"]
        pos = np.searchsorted(u, arr)
        pos = np.clip(pos, 0, len(u) - 1)
        hit = u[pos] == arr
        e = np.where(hit, enc[pos], te["gmean"]).astype(np.float32)
        c = np.where(hit, cnt[pos], 0.0).astype(np.float32)
        if nm == "pcell":
            df["te_pcell_oof"] = e; df["p_cell_count"] = c
        elif nm == "dcell":
            df["te_dcell_oof"] = e; df["d_cell_count"] = c
        else:
            df["te_route_oof"] = e; df["route_count"] = c
    return df


def add_te3(df, te3):
    res = float(te3["res"])
    lat_p = df["pickup_latitude"].to_numpy(np.float64); lon_p = df["pickup_longitude"].to_numpy(np.float64)
    lat_d = df["dropoff_latitude"].to_numpy(np.float64); lon_d = df["dropoff_longitude"].to_numpy(np.float64)
    def cell(lat, lon):
        clat = np.floor((lat - 40.0) / res).astype(np.int64)
        clon = np.floor((lon + 75.0) / res).astype(np.int64)
        return clat * 10000 + clon
    pc = cell(lat_p, lon_p); dc = cell(lat_d, lon_d)
    rf = (pc + 1200000) * 8000000 + (dc + 1200000)
    gm = te3["gmean"]
    for nm, arr in [("pcf", pc), ("dcf", dc), ("routef", rf)]:
        u = te3["u_" + nm]; enc = te3["enc_" + nm + "_full"]
        pos = np.clip(np.searchsorted(u, arr), 0, len(u) - 1)
        hit = u[pos] == arr
        df["te_" + nm + "_oof"] = np.where(hit, enc[pos], gm).astype(np.float32)
    return df


def add_rot29(df):
    import math
    c = math.cos(math.radians(29)); s = math.sin(math.radians(29))
    plat = df["pickup_latitude"].to_numpy(np.float32); plon = df["pickup_longitude"].to_numpy(np.float32)
    dlat = df["dropoff_latitude"].to_numpy(np.float32); dlon = df["dropoff_longitude"].to_numpy(np.float32)
    df["ru29_p"] = plat * c + plon * s; df["rv29_p"] = -plat * s + plon * c
    df["ru29_d"] = dlat * c + dlon * s; df["rv29_d"] = -dlat * s + dlon * c
    df["ru29_delta"] = df["ru29_d"] - df["ru29_p"]; df["rv29_delta"] = df["rv29_d"] - df["rv29_p"]
    return df
