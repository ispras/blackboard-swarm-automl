import numpy as np, pandas as pd, os, json, time

ROOT = "/download_job_code/payload/sandbox"
OUT = f"{ROOT}/workspaces/dubhe_tau/features"
os.makedirs(OUT, exist_ok=True)

NMAX = 55_413_942
CHUNK = 2_000_000

FEATURES = [
    "pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude",
    "passenger_count",
    "year", "month", "day", "hour", "minute", "weekday", "dayofyear",
    "haversine_km", "manhattan_km", "bearing",
    "dlat", "dlon", "abs_dlat", "abs_dlon",
    "dist_pickup_center", "dist_dropoff_center",
    "dist_pickup_JFK", "dist_dropoff_JFK",
    "dist_pickup_LGA", "dist_dropoff_LGA",
    "dist_pickup_EWR", "dist_dropoff_EWR",
    "airport_pickup", "airport_dropoff",
]
F = len(FEATURES)

CENTER = (40.7580, -73.9855)
JFK = (40.6413, -73.7781)
LGA = (40.7769, -73.8740)
EWR = (40.6895, -74.1745)
R = 6371.0088


def haversine(lat1, lon1, lat2, lon2):
    lat1 = np.radians(lat1); lat2 = np.radians(lat2)
    dlat = lat2 - lat1
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return R * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def make_features(df, is_test=False):
    plon = df["pickup_longitude"].to_numpy(np.float32)
    plat = df["pickup_latitude"].to_numpy(np.float32)
    dlon = df["dropoff_longitude"].to_numpy(np.float32)
    dlat_ = df["dropoff_latitude"].to_numpy(np.float32)
    pc = df["passenger_count"].to_numpy(np.float32)
    pc = np.nan_to_num(pc, nan=1.0)

    dt = pd.to_datetime(df["pickup_datetime"], format="%Y-%m-%d %H:%M:%S UTC", errors="coerce")
    year = dt.dt.year.to_numpy(np.float32)
    month = dt.dt.month.to_numpy(np.float32)
    day = dt.dt.day.to_numpy(np.float32)
    hour = dt.dt.hour.to_numpy(np.float32)
    minute = dt.dt.minute.to_numpy(np.float32)
    weekday = dt.dt.weekday.to_numpy(np.float32)
    dayofyear = dt.dt.dayofyear.to_numpy(np.float32)

    hav = haversine(plat, plon, dlat_, dlon).astype(np.float32)
    dlat = (dlat_ - plat).astype(np.float32)
    dlonv = (dlon - plon).astype(np.float32)
    # equirectangular manhattan
    mlat = np.radians((plat + dlat_) / 2.0)
    dy = dlat * 110.574
    dx = dlonv * 111.320 * np.cos(mlat)
    manh = (np.abs(dx) + np.abs(dy)).astype(np.float32)
    # bearing
    la1 = np.radians(plat); la2 = np.radians(dlat_); dlo = np.radians(dlonv)
    y_ = np.sin(dlo) * np.cos(la2)
    x_ = np.cos(la1) * np.sin(la2) - np.sin(la1) * np.cos(la2) * np.cos(dlo)
    bearing = np.degrees(np.arctan2(y_, x_)).astype(np.float32)

    def dist_to(pt):
        return haversine(plat, plon, pt[0], pt[1]).astype(np.float32)
    def dist_to2(pt):
        return haversine(dlat_, dlon, pt[0], pt[1]).astype(np.float32)

    dpc = haversine(plat, plon, CENTER[0], CENTER[1]).astype(np.float32)
    ddc = haversine(dlat_, dlon, CENTER[0], CENTER[1]).astype(np.float32)
    pj = dist_to(JFK); dj = dist_to2(JFK)
    pl = dist_to(LGA); dl = dist_to2(LGA)
    pe = dist_to(EWR); de = dist_to2(EWR)
    apick = (np.minimum(np.minimum(pj, pl), pe) < 2.0).astype(np.float32)
    adrop = (np.minimum(np.minimum(dj, dl), de) < 2.0).astype(np.float32)

    X = np.column_stack([
        plon, plat, dlon, dlat_, pc,
        year, month, day, hour, minute, weekday, dayofyear,
        hav, manh, bearing,
        dlat, dlonv, np.abs(dlat), np.abs(dlonv),
        dpc, ddc, pj, dj, pl, dl, pe, de,
        apick, adrop,
    ]).astype(np.float32)
    return X, dt


def main():
    t0 = time.time()
    Xmm = np.lib.format.open_memmap(f"{OUT}/X_full.npy", mode="w+", dtype=np.float32, shape=(NMAX, F))
    ymm = np.lib.format.open_memmap(f"{OUT}/y_full.npy", mode="w+", dtype=np.float32, shape=(NMAX,))
    hmm = np.lib.format.open_memmap(f"{OUT}/holdout_full.npy", mode="w+", dtype=np.uint8, shape=(NMAX,))

    writer = 0
    total = 0
    kept = 0
    hold_keys = []
    hold_fares = []
    reader = pd.read_csv(
        f"{ROOT}/data/labels.csv",
        chunksize=CHUNK,
        dtype={"fare_amount": np.float32, "pickup_longitude": np.float32,
               "pickup_latitude": np.float32, "dropoff_longitude": np.float32,
               "dropoff_latitude": np.float32, "passenger_count": np.float32},
    )
    for ci, ch in enumerate(reader):
        total += len(ch)
        X, dt = make_features(ch)
        fare = ch["fare_amount"].to_numpy(np.float32)
        pc = ch["passenger_count"].to_numpy(np.float32)
        pcok = (np.nan_to_num(pc, nan=-1.0) >= 0.0) & (np.nan_to_num(pc, nan=99.0) <= 8.0)
        ok = (
            np.isfinite(fare) & (fare >= 0.5) & (fare <= 500.0)
            & np.isfinite(X[:, 0]) & (X[:, 0] >= -75.5) & (X[:, 0] <= -72.5)
            & (X[:, 1] >= 39.5) & (X[:, 1] <= 42.0)
            & (X[:, 2] >= -75.5) & (X[:, 2] <= -72.5)
            & (X[:, 3] >= 39.5) & (X[:, 3] <= 42.0)
            & pcok
            & dt.notna().to_numpy()
        )
        Xo = X[ok]
        fo = fare[ok]
        keys = ch["key"].to_numpy()[ok]
        ho = (pd.util.hash_array(keys) % 100 == 0).astype(np.uint8)
        hold_keys.append(keys[ho == 1])
        hold_fares.append(fo[ho == 1])
        n = len(Xo)
        Xmm[writer:writer + n] = Xo
        ymm[writer:writer + n] = fo
        hmm[writer:writer + n] = ho
        writer += n
        kept += n
        if ci % 2 == 0:
            print(f"chunk {ci} total={total} kept={kept} rate={kept/max(total,1):.3f} "
                  f"holdout={int(ho.sum())} {time.time()-t0:.0f}s", flush=True)
    del Xmm, ymm, hmm
    meta = {"n_valid": int(kept), "n_total": int(total), "n_features": F,
            "features": FEATURES, "n_holdout": int(np.asarray(np.load(f"{OUT}/holdout_full.npy", mmap_mode='r')[:kept]).sum())}
    json.dump(meta, open(f"{OUT}/meta.json", "w"), indent=2)
    hk = np.concatenate(hold_keys)
    hf = np.concatenate(hold_fares)
    pd.DataFrame({"key": hk, "fare_amount": hf}).to_csv(f"{OUT}/holdout_reference.csv", index=False)
    print("holdout reference rows", len(hk), flush=True)

    # test features
    te = pd.read_csv(f"{ROOT}/data/test.csv",
                     dtype={"pickup_longitude": np.float32, "pickup_latitude": np.float32,
                            "dropoff_longitude": np.float32, "dropoff_latitude": np.float32,
                            "passenger_count": np.float32})
    Xte, _ = make_features(te, is_test=True)
    np.save(f"{OUT}/test_X.npy", Xte)
    te[["key"]].to_csv(f"{OUT}/test_keys.csv", index=False)
    print("DONE", meta, f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
