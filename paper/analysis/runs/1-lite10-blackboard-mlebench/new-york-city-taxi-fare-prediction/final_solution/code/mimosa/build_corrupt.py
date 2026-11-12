import os
import numpy as np
import pandas as pd
import prep

ROOT = "/download_job_code/payload/sandbox"
OUT = os.path.join(ROOT, "workspaces/mimosa_omicron")
USECOLS = ["key", "fare_amount", "pickup_datetime", "pickup_longitude", "pickup_latitude",
           "dropoff_longitude", "dropoff_latitude", "passenger_count"]


def add_corrupt_feats(df):
    plon = df["pickup_longitude"].to_numpy(np.float32); plat = df["pickup_latitude"].to_numpy(np.float32)
    dlon = df["dropoff_longitude"].to_numpy(np.float32); dlat = df["dropoff_latitude"].to_numpy(np.float32)
    pv = (np.isfinite(plon) & np.isfinite(plat) & (plon > -75.5) & (plon < -72.5) & (plat > 39.5) & (plat < 42.0))
    dv = (np.isfinite(dlon) & np.isfinite(dlat) & (dlon > -75.5) & (dlon < -72.5) & (dlat > 39.5) & (dlat < 42.0))
    dt = pd.to_datetime(df["pickup_datetime"].str.slice(0, 19), format="%Y-%m-%d %H:%M:%S", errors="coerce")
    out = pd.DataFrame(index=df.index)
    out["year"] = dt.dt.year.to_numpy(np.float32); out["month"] = dt.dt.month.to_numpy(np.float32)
    out["day"] = dt.dt.day.to_numpy(np.float32); out["weekday"] = dt.dt.weekday.to_numpy(np.float32)
    out["hour"] = dt.dt.hour.to_numpy(np.float32); out["minute"] = dt.dt.minute.to_numpy(np.float32)
    out["doy"] = dt.dt.dayofyear.to_numpy(np.float32); out["week"] = dt.dt.isocalendar().week.to_numpy(np.float32)
    out["is_weekend"] = (out["weekday"] >= 5).astype(np.float32)
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24).astype(np.float32)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24).astype(np.float32)
    out["passenger_count"] = df["passenger_count"].to_numpy(np.float32)
    out["p_valid"] = pv.astype(np.float32); out["d_valid"] = dv.astype(np.float32)
    out["p_zero"] = ((plon == 0) & (plat == 0)).astype(np.float32)
    out["d_zero"] = ((dlon == 0) & (dlat == 0)).astype(np.float32)
    for nm, ok, lo, la in [("p", pv, plon, plat), ("d", dv, dlon, dlat)]:
        for lm, (alon, alat) in [("jfk", prep.JFK), ("lga", prep.LGA), ("ewr", prep.EWR),
                                 ("mid", prep.MID), ("lower", prep.LOWER)]:
            v = prep.hav(lo, la, alon, alat)
            out[f"{nm}_{lm}"] = np.where(ok, v, np.nan).astype(np.float32)
        out[f"{nm}_lat"] = np.where(ok, la, np.nan).astype(np.float32)
        out[f"{nm}_lon"] = np.where(ok, lo, np.nan).astype(np.float32)
    return out


def main():
    parts_bad, parts_good = [], []
    start = 0
    for ch in pd.read_csv(os.path.join(ROOT, "data", "labels.csv"), usecols=USECOLS, chunksize=4_000_000,
                          dtype={"pickup_longitude": np.float32, "pickup_latitude": np.float32,
                                 "dropoff_longitude": np.float32, "dropoff_latitude": np.float32,
                                 "fare_amount": np.float32, "passenger_count": np.float32}):
        n = len(ch)
        idx = np.arange(start, start + n)
        start += n
        plon = ch["pickup_longitude"].to_numpy(); plat = ch["pickup_latitude"].to_numpy()
        dlon = ch["dropoff_longitude"].to_numpy(); dlat = ch["dropoff_latitude"].to_numpy()
        y = ch["fare_amount"].to_numpy()
        pv = np.isfinite(plon) & np.isfinite(plat) & (plon > -75.5) & (plon < -72.5) & (plat > 39.5) & (plat < 42.0)
        dv = np.isfinite(dlon) & np.isfinite(dlat) & (dlon > -75.5) & (dlon < -72.5) & (dlat > 39.5) & (dlat < 42.0)
        ok = (y >= 2.5) & (y <= 500) & np.isfinite(y)
        bad = (~pv | ~dv) & ok
        good = pv & dv & ok
        fb = add_corrupt_feats(ch[bad])
        fb["orig_idx"] = idx[bad].astype(np.int32); fb["fare_amount"] = y[bad].astype(np.float32)
        parts_bad.append(fb)
        print("chunk", len(ch), "bad", bad.sum(), "good", good.sum(), flush=True)
    bdf = pd.concat(parts_bad, ignore_index=True)
    bdf.to_parquet(os.path.join(OUT, "corrupt_feat.parquet"), index=False)
    print("saved corrupt_feat", bdf.shape, flush=True)


if __name__ == "__main__":
    main()
