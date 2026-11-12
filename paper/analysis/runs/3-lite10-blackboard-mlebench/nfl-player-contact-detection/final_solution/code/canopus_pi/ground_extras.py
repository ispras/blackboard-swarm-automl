import os, time, gc
import numpy as np
import pandas as pd

ROOT = "/download_job_code/payload/sandbox"
FEAT = f"{ROOT}/workspaces/canopus_pi/features/all_features.parquet"
OUT = f"{ROOT}/workspaces/canopus_pi/features/ground_extra.parquet"
t0 = time.time()
def log(*a):
    print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)

log("meta")
m = pd.read_parquet(FEAT, columns=["contact_id", "split", "step", "is_ground", "nfl_player_id_1"])
p = m.contact_id.str.rsplit("_", n=3, expand=True)
m["gp"] = p[0].values
m["step"] = p[1].astype(np.int32).values
m["p1"] = p[2].astype(np.int64).values
m["p2n"] = pd.to_numeric(p[3], errors="coerce").fillna(-1).astype(np.int64).values
m = m.reset_index(drop=True)

log("tracking")
tr = []
for f in ["train_player_tracking.csv", "test_player_tracking.csv"]:
    d = pd.read_csv(f"{ROOT}/data/{f}",
                    usecols=["game_play", "step", "nfl_player_id", "speed", "sa",
                             "acceleration", "distance", "direction"])
    tr.append(d)
tr = pd.concat(tr, ignore_index=True)
tr["gp"] = tr["game_play"].astype(str)
tr["step"] = tr["step"].astype(np.int32)
tr["player"] = tr["nfl_player_id"].astype(np.int64)
tr = tr.sort_values(["gp", "player", "step"]).reset_index(drop=True)
log("tracking", tr.shape)

g = tr.groupby(["gp", "player"], sort=False)
# low-speed run ending at t and starting at t (future run)
for th in [0.5, 1.5, 2.5]:
    low = (tr["speed"] < th).astype(np.int8)
    block = (~low.astype(bool)).groupby([tr["gp"], tr["player"]], sort=False).cumsum()
    run = low.groupby([tr["gp"], tr["player"], block], sort=False).cumsum()
    tr[f"run_low{th}"] = run.values.astype(np.float32)
    rev = tr.iloc[::-1]
    low_r = (rev["speed"] < th).astype(np.int8)
    block_r = (~low_r.astype(bool)).groupby([rev["gp"], rev["player"]], sort=False).cumsum()
    run_r = low_r.groupby([rev["gp"], rev["player"], block_r], sort=False).cumsum()
    tr[f"fut_low{th}"] = run_r.reindex(tr.index).values.astype(np.float32)

# centered windows
for c in ["speed", "sa", "acceleration", "distance"]:
    arrs = [tr[c].values]
    for k in range(1, 6):
        arrs.append(g[c].shift(k).values)
        arrs.append(g[c].shift(-k).values)
    M = np.vstack(arrs)
    tr[f"{c}_c5_min"] = np.nanmin(M, axis=0)
    tr[f"{c}_c5_max"] = np.nanmax(M, axis=0)
    tr[f"{c}_c5_mean"] = np.nanmean(M, axis=0)

# cumulative distance over next 5 steps (motion after)
tr["dist_fut5_sum"] = sum(g["distance"].shift(-k).fillna(0).values for k in range(0, 6))
tr["dist_past5_sum"] = sum(g["distance"].shift(k).fillna(0).values for k in range(0, 6))

# cross-sectional ranks per (gp, step)
for c in ["speed", "sa", "acceleration", "distance"]:
    tr[f"rank_{c}"] = tr.groupby(["gp", "step"], sort=False)[c].rank(pct=True).astype(np.float32)
tr["rank_speed_d1"] = tr.groupby(["gp", "step"], sort=False)["speed"].rank(pct=True).values

keep = ["gp", "step", "player", "run_low0.5", "fut_low0.5", "run_low1.5", "fut_low1.5",
        "run_low2.5", "fut_low2.5",
        "speed_c5_min", "speed_c5_max", "speed_c5_mean",
        "sa_c5_min", "sa_c5_max", "sa_c5_mean",
        "acceleration_c5_min", "acceleration_c5_max", "acceleration_c5_mean",
        "distance_c5_min", "distance_c5_max", "distance_c5_mean",
        "dist_fut5_sum", "dist_past5_sum",
        "rank_speed", "rank_sa", "rank_acceleration", "rank_distance"]
pl = tr[keep].copy()
del tr, g
gc.collect()

# helmet ranks
log("helmet ranks")
hp = pd.read_parquet(f"{ROOT}/workspaces/canopus_pi/features/helmet_player_v2.parquet",
                     columns=["gp", "player", "step", "sid_h_top", "end_h_top",
                              "sid_h_h", "end_h_h", "sid_h_bottom", "end_h_bottom"])
for view in ["sid", "end"]:
    hp[f"{view}_top_rank2"] = hp.groupby(["gp", "step"], sort=False)[f"{view}_h_top"].rank(pct=True)
    hp[f"{view}_bottom_rank2"] = hp.groupby(["gp", "step"], sort=False)[f"{view}_h_bottom"].rank(pct=True)
    hp[f"{view}_h_rank2"] = hp.groupby(["gp", "step"], sort=False)[f"{view}_h_h"].rank(pct=True)
pl1 = pl.rename(columns={"player": "p1"})
mm = m.merge(pl1, on=["gp", "step", "p1"], how="left")
hp1 = hp[["gp", "player", "step", "sid_top_rank2", "end_top_rank2",
          "sid_bottom_rank2", "end_bottom_rank2", "sid_h_rank2", "end_h_rank2"]].rename(columns={"player": "p1"})
mm = mm.merge(hp1, on=["gp", "step", "p1"], how="left")
# mm has duplicate player cols; select
newcols = [c for c in pl.columns if c not in ("gp", "step", "player")]
hcols = ["sid_top_rank2", "end_top_rank2", "sid_bottom_rank2", "end_bottom_rank2",
         "sid_h_rank2", "end_h_rank2"]
out = pd.DataFrame({"contact_id": mm["contact_id"].values})
for c in newcols + hcols:
    out[c] = mm[c].values.astype(np.float32)
out.to_parquet(OUT, index=False)
log("saved", out.shape)
print(out.columns.tolist())
