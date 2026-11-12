import os, sys, time
import numpy as np
import pandas as pd

ROOT = "/download_job_code/payload/sandbox"
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "workspaces/mimosa_omicron", "feat")
os.makedirs(OUT, exist_ok=True)

USECOLS = ["key", "fare_amount", "pickup_datetime", "pickup_longitude",
           "pickup_latitude", "dropoff_longitude", "dropoff_latitude", "passenger_count"]

# airport / landmark coords
JFK = (-73.7781, 40.6413)
LGA = (-73.8740, 40.7769)
EWR = (-74.1745, 40.6895)
MID = (-73.9857, 40.7580)
LOWER = (-74.0120, 40.7060)

R = 6371.0088

def hav(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return (2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))).astype(np.float32)

def bearing(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return np.arctan2(x, y).astype(np.float32)

def build(chunk, start_idx):
    n = len(chunk)
    plat = chunk["pickup_latitude"].to_numpy(np.float32)
    plon = chunk["pickup_longitude"].to_numpy(np.float32)
    dlat = chunk["dropoff_latitude"].to_numpy(np.float32)
    dlon = chunk["dropoff_longitude"].to_numpy(np.float32)
    y = chunk["fare_amount"].to_numpy(np.float32)
    pc = chunk["passenger_count"].to_numpy(np.float32)

    dt = pd.to_datetime(chunk["pickup_datetime"].str.slice(0, 19), format="%Y-%m-%d %H:%M:%S", errors="coerce")

    out = {
        "orig_idx": np.arange(start_idx, start_idx + n, dtype=np.int32),
        "fare_amount": y,
        "pickup_latitude": plat, "pickup_longitude": plon,
        "dropoff_latitude": dlat, "dropoff_longitude": dlon,
        "passenger_count": pc,
        "year": dt.dt.year.to_numpy(np.float32),
        "month": dt.dt.month.to_numpy(np.float32),
        "day": dt.dt.day.to_numpy(np.float32),
        "weekday": dt.dt.weekday.to_numpy(np.float32),
        "hour": dt.dt.hour.to_numpy(np.float32),
        "minute": dt.dt.minute.to_numpy(np.float32),
        "doy": dt.dt.dayofyear.to_numpy(np.float32),
        "week": dt.dt.isocalendar().week.to_numpy(np.float32),
    }
    out["dlat"] = (dlat - plat).astype(np.float32)
    out["dlon"] = (dlon - plon).astype(np.float32)
    out["abs_dlat"] = np.abs(out["dlat"]).astype(np.float32)
    out["abs_dlon"] = np.abs(out["dlon"]).astype(np.float32)
    out["dist"] = hav(plon, plat, dlon, dlat)
    out["dist_man"] = (hav(plon, plat, plon, dlat) + hav(plon, plat, dlon, plat)).astype(np.float32)
    out["bearing"] = bearing(plon, plat, dlon, dlat)
    for name, (alon, alat) in [("jfk", JFK), ("lga", LGA), ("ewr", EWR)]:
        out["p_" + name] = hav(plon, plat, alon, alat)
        out["d_" + name] = hav(dlon, dlat, alon, alat)
    out["p_mid"] = hav(plon, plat, MID[0], MID[1])
    out["d_mid"] = hav(dlon, dlat, MID[0], MID[1])
    out["p_lower"] = hav(plon, plat, LOWER[0], LOWER[1])
    out["d_lower"] = hav(dlon, dlat, LOWER[0], LOWER[1])
    out["p_airport_min"] = np.minimum(np.minimum(out["p_jfk"], out["p_lga"]), out["p_ewr"])
    out["d_airport_min"] = np.minimum(np.minimum(out["d_jfk"], out["d_lga"]), out["d_ewr"])
    out["is_weekend"] = (out["weekday"] >= 5).astype(np.float32)
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24).astype(np.float32)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24).astype(np.float32)
    return pd.DataFrame(out)

def main():
    t0 = time.time()
    chunks = pd.read_csv(os.path.join(DATA, "labels.csv"), usecols=USECOLS, chunksize=4_000_000,
                         dtype={"pickup_longitude": np.float32, "pickup_latitude": np.float32,
                                "dropoff_longitude": np.float32, "dropoff_latitude": np.float32,
                                "fare_amount": np.float32, "passenger_count": np.float32})
    start = 0
    total_in = total_out = 0
    for i, ch in enumerate(chunks):
        df = build(ch, start)
        start += len(ch)
        total_in += len(ch)
        # cleaning
        m = (
            df["fare_amount"].between(2.5, 500) &
            df["pickup_longitude"].between(-75.5, -72.5) &
            df["pickup_latitude"].between(39.5, 42.0) &
            df["dropoff_longitude"].between(-75.5, -72.5) &
            df["dropoff_latitude"].between(39.5, 42.0) &
            df["dist"].between(0.0, 300.0) &
            df["passenger_count"].between(0, 8)
        )
        df = df[m].reset_index(drop=True)
        total_out += len(df)
        df.to_parquet(os.path.join(OUT, f"part_{i:03d}.parquet"), index=False)
        print(f"chunk {i}: in={len(ch)} out={len(df)} total={total_out} elapsed={time.time()-t0:.0f}s", flush=True)
    print(f"DONE total_in={total_in} total_out={total_out} time={time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
