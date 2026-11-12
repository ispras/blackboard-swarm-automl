import os, time
import numpy as np
import pandas as pd
from common import FEATS, parts

OUT = "/download_job_code/payload/sandbox/workspaces/mimosa_omicron/feat_v2"
os.makedirs(OUT, exist_ok=True)
C = np.cos(np.deg2rad(45.0)); S = np.sin(np.deg2rad(45.0))


def cell(lat, lon, res=0.01):
    clat = np.floor((lat - 40.0) / res).astype(np.int32)
    clon = np.floor((lon + 75.0) / res).astype(np.int32)
    return clat.astype(np.int64) * 10000 + clon


def augment(df):
    plat = df["pickup_latitude"].to_numpy(np.float32)
    plon = df["pickup_longitude"].to_numpy(np.float32)
    dlat = df["dropoff_latitude"].to_numpy(np.float32)
    dlon = df["dropoff_longitude"].to_numpy(np.float32)
    df["ru_p"] = (plat * C + plon * S).astype(np.float32)
    df["rv_p"] = (-plat * S + plon * C).astype(np.float32)
    df["ru_d"] = (dlat * C + dlon * S).astype(np.float32)
    df["rv_d"] = (-dlat * S + dlon * C).astype(np.float32)
    df["ru_delta"] = (df["ru_d"] - df["ru_p"]).astype(np.float32)
    df["rv_delta"] = (df["rv_d"] - df["rv_p"]).astype(np.float32)
    df["sum_jfk"] = (df["p_jfk"] + df["d_jfk"]).astype(np.float32)
    df["sum_lga"] = (df["p_lga"] + df["d_lga"]).astype(np.float32)
    df["sum_ewr"] = (df["p_ewr"] + df["d_ewr"]).astype(np.float32)
    df["sum_airport_min"] = (df["p_airport_min"] + df["d_airport_min"]).astype(np.float32)
    df["p_near_airport"] = (df["p_airport_min"] < 2.0).astype(np.float32)
    df["d_near_airport"] = (df["d_airport_min"] < 2.0).astype(np.float32)
    df["both_near_airport"] = (df["p_near_airport"] * df["d_near_airport"]).astype(np.float32)
    df["p_cell"] = cell(plat, plon)
    df["d_cell"] = cell(dlat, dlon)
    df["route"] = (df["p_cell"] + 600000) * 4000000 + (df["d_cell"] + 600000)
    return df


def main():
    t0 = time.time()
    pl = parts()
    pcs, dcs, rts = [], [], []
    for p in pl:
        df = pd.read_parquet(p, columns=["pickup_latitude", "pickup_longitude",
                                         "dropoff_latitude", "dropoff_longitude"])
        plat = df["pickup_latitude"].to_numpy(np.float32); plon = df["pickup_longitude"].to_numpy(np.float32)
        dlat = df["dropoff_latitude"].to_numpy(np.float32); dlon = df["dropoff_longitude"].to_numpy(np.float32)
        pc = cell(plat, plon); dc = cell(dlat, dlon)
        pcs.append(pc); dcs.append(dc); rts.append((pc + 600000) * 4000000 + (dc + 600000))
        del df
    pc = np.concatenate(pcs); dc = np.concatenate(dcs); rt = np.concatenate(rts)
    del pcs, dcs, rts
    print("cells built", len(pc), time.time() - t0, flush=True)
    upc, cpc = np.unique(pc, return_counts=True)
    udc, cdc = np.unique(dc, return_counts=True)
    urt, crt = np.unique(rt, return_counts=True)
    print(f"unique p={len(upc)} d={len(udc)} route={len(urt)}", flush=True)
    pc_count = cpc[np.searchsorted(upc, pc)].astype(np.float32)
    dc_count = cdc[np.searchsorted(udc, dc)].astype(np.float32)
    rt_count = crt[np.searchsorted(urt, rt)].astype(np.float32)
    del pc, dc, rt
    start = 0
    for i, p in enumerate(pl):
        df = pd.read_parquet(p)
        n = len(df)
        df = augment(df)
        df["p_cell_count"] = pc_count[start:start + n]
        df["d_cell_count"] = dc_count[start:start + n]
        df["route_count"] = rt_count[start:start + n]
        start += n
        df["p_cell"] = df["p_cell"].astype(np.int64)
        df["d_cell"] = df["d_cell"].astype(np.int64)
        df["route"] = df["route"].astype(np.int64)
        df.to_parquet(os.path.join(OUT, f"part_{i:03d}.parquet"), index=False)
        print(f"saved {i} n={n} elapsed={time.time()-t0:.0f}", flush=True)
    print("DONE", time.time() - t0, flush=True)


if __name__ == "__main__":
    main()
