import os, time
import numpy as np
import pandas as pd
from common import parts_v2

OUT = "/download_job_code/payload/sandbox/workspaces/mimosa_omicron"
K = 20.0
NFOLD = 5
RES = 0.005


def oof_encode(codes, y, fold, gmean, k):
    n = len(y)
    u, c = np.unique(codes, return_inverse=True)
    ne = len(u)
    sum_all = np.bincount(c, weights=y, minlength=ne)
    cnt_all = np.bincount(c, minlength=ne)
    full = ((sum_all + k * gmean) / (cnt_all + k)).astype(np.float32)
    oof = np.empty(n, dtype=np.float32)
    for f in range(NFOLD):
        m = fold == f
        sum_f = sum_all - np.bincount(c[m], weights=y[m], minlength=ne)
        cnt_f = cnt_all - np.bincount(c[m], minlength=ne)
        pf = float(y[~m].mean())
        ef = ((sum_f + k * pf) / (cnt_f + k)).astype(np.float32)
        oof[m] = ef[c[m]]
    return u, full, oof, cnt_all.astype(np.float32), ne


def cell(lat, lon):
    clat = np.floor((lat - 40.0) / RES).astype(np.int64)
    clon = np.floor((lon + 75.0) / RES).astype(np.int64)
    return clat * 10000 + clon


def main():
    t0 = time.time()
    pl = parts_v2()
    Y, OI, PC, DC = [], [], [], []
    for p in pl:
        df = pd.read_parquet(p, columns=["orig_idx", "fare_amount", "pickup_latitude",
                                         "pickup_longitude", "dropoff_latitude", "dropoff_longitude"])
        pc = cell(df["pickup_latitude"].to_numpy(), df["pickup_longitude"].to_numpy())
        dc = cell(df["dropoff_latitude"].to_numpy(), df["dropoff_longitude"].to_numpy())
        Y.append(df["fare_amount"].to_numpy(np.float32))
        OI.append(df["orig_idx"].to_numpy(np.int64))
        PC.append(pc); DC.append(dc)
    y = np.concatenate(Y); oi = np.concatenate(OI); pc = np.concatenate(PC); dc = np.concatenate(DC)
    del Y, OI, PC, DC
    fold = (oi % NFOLD).astype(np.int8)
    gmean = float(y.mean())
    res = {"gmean": np.float64(gmean), "k": np.float64(K), "res": np.float64(RES)}
    for nm, arr in [("pcf", pc), ("dcf", dc)]:
        u, full, oof, cnt, ne = oof_encode(arr, y, fold, gmean, K)
        res["te_" + nm + "_oof"] = oof; res["u_" + nm] = u
        res["enc_" + nm + "_full"] = full; res["cnt_" + nm] = cnt
        print(nm, ne, time.time() - t0, flush=True)
    rf = (pc + 1200000) * 8000000 + (dc + 1200000)
    del pc, dc
    u, full, oof, cnt, ne = oof_encode(rf, y, fold, gmean, K)
    res["te_routef_oof"] = oof; res["u_routef"] = u
    res["enc_routef_full"] = full; res["cnt_routef"] = cnt
    print("routef", ne, time.time() - t0, flush=True)
    np.savez_compressed(os.path.join(OUT, "te3_maps.npz"), **res)
    print("saved", time.time() - t0, flush=True)


if __name__ == "__main__":
    main()
