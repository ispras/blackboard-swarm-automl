import os, time, gc
import numpy as np
import pandas as pd

ROOT = "/download_job_code/payload/sandbox"
FEAT = f"{ROOT}/workspaces/canopus_pi/features/all_features.parquet"
OUT = f"{ROOT}/workspaces/canopus_pi/features/new_feats_v2.parquet"
t0 = time.time()
def log(*a):
    print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)

log("meta")
meta = pd.read_parquet(FEAT, columns=["contact_id", "split", "step", "is_ground",
                                      "nfl_player_id_1", "dist", "closing_speed", "rel_speed"])
p = meta.contact_id.str.rsplit("_", n=3, expand=True)
meta["gp"] = p[0].values
meta["step"] = p[1].astype(np.int32).values
meta["p1"] = p[2].astype(np.int64).values
meta["p2n"] = pd.to_numeric(p[3], errors="coerce").fillna(-1).astype(np.int64).values
meta = meta.reset_index(drop=True)
n = len(meta)
log("meta", meta.shape)

new = pd.DataFrame({"contact_id": meta.contact_id.values})
new["log_dist"] = np.log1p(meta["dist"].values).astype(np.float32)
new["inv_dist"] = (1.0 / (meta["dist"].values + 1.0)).astype(np.float32)
new["time_to_collision"] = (meta["dist"].values / (np.abs(meta["closing_speed"].values) + 0.5)).astype(np.float32)
new["close_per_speed"] = (meta["closing_speed"].values / (meta["rel_speed"].values + 0.5)).astype(np.float32)
new["dist_x_closing"] = (meta["dist"].values * meta["closing_speed"].values).astype(np.float32)

# per-player nearest-neighbour features from tracking
log("tracking")
tr = []
for f in ["train_player_tracking.csv", "test_player_tracking.csv"]:
    d = pd.read_csv(f"{ROOT}/data/{f}",
                    usecols=["game_play", "step", "nfl_player_id", "x_position", "y_position", "team"])
    tr.append(d)
tr = pd.concat(tr, ignore_index=True)
tr["gp"] = tr["game_play"].astype(str)
tr["step"] = tr["step"].astype(np.int32)
tr["player"] = tr["nfl_player_id"].astype(np.int64)
tr["_tidx"] = np.arange(len(tr))
tr = tr.sort_values(["gp", "step", "player"]).reset_index(drop=True)

cols = ["nb1", "nb2", "nb3", "opp1", "same1", "n2", "n5", "nb_mean", "nn_mutual"]
P = np.full((len(tr), len(cols)), np.nan, dtype=np.float32)

# candidate row lookup by (gp,step)
log("cand lookup")
cand_idx = {}
for (gp, st), g in meta.groupby(["gp", "step"], sort=False):
    cand_idx[(gp, st)] = g.index.values

pair_third = np.full(n, np.nan, dtype=np.float32)
pair_c2_1 = np.full(n, np.nan, dtype=np.float32)
pair_c2_2 = np.full(n, np.nan, dtype=np.float32)
pair_is_nn1 = np.full(n, np.nan, dtype=np.float32)
pair_is_nn2 = np.full(n, np.nan, dtype=np.float32)

log("group loop")
tgroups = tr.groupby(["gp", "step"], sort=False)
for gi, ((gp, st), d) in enumerate(tgroups):
    idx = d["_tidx"].values
    x = d["x_position"].values.astype(np.float64)
    y = d["y_position"].values.astype(np.float64)
    pids = d["player"].values
    teams = d["team"].values
    m = len(x)
    D = np.sqrt((x[:, None] - x[None, :]) ** 2 + (y[:, None] - y[None, :]) ** 2)
    np.fill_diagonal(D, np.inf)
    order = np.argsort(D, axis=1)
    d1 = D[np.arange(m), order[:, 0]]
    d2 = D[np.arange(m), order[:, 1]]
    d3 = D[np.arange(m), order[:, 2]] if m > 2 else np.full(m, np.nan)
    arg1 = order[:, 0]
    mutual = (arg1[arg1] == np.arange(m)).astype(np.float32)
    # opponents
    same = teams[:, None] == teams[None, :]
    Dopp = np.where(same, np.inf, D)
    Dsame = np.where(same & ~np.eye(m, dtype=bool), D, np.inf)
    opp1 = np.min(Dopp, axis=1)
    same1 = np.min(Dsame, axis=1)
    n2 = (D < 2.0).sum(1)
    n5 = (D < 5.0).sum(1)
    nb_mean = np.nanmean(np.where(np.isfinite(D), D, np.nan), axis=1)
    P[idx, 0] = d1; P[idx, 1] = d2; P[idx, 2] = d3; P[idx, 3] = opp1
    P[idx, 4] = same1; P[idx, 5] = n2; P[idx, 6] = n5
    P[idx, 7] = nb_mean; P[idx, 8] = mutual

    ci = cand_idx.get((gp, st))
    if ci is None or len(ci) == 0:
        continue
    p1a = meta["p1"].values[ci]; p2a = meta["p2n"].values[ci]
    pos = {int(pp): k for k, pp in enumerate(pids)}
    i1 = np.array([pos.get(int(v), -1) for v in p1a])
    i2 = np.array([pos.get(int(v), -1) for v in p2a])
    pp_mask = i2 >= 0
    ii1 = i1[pp_mask]; ii2 = i2[pp_mask]
    is_nn1 = arg1[ii1] == ii2
    is_nn2 = arg1[ii2] == ii1
    t1 = np.where(is_nn1, d2[ii1], d1[ii1])
    t2 = np.where(is_nn2, d2[ii2], d1[ii2])
    third = np.minimum(t1, t2)
    in2 = (D[ii1, ii2] < 2.0).astype(np.float32)
    c21 = n2[ii1] - in2
    c22 = n2[ii2] - in2
    sub = ci[pp_mask]
    pair_third[sub] = third
    pair_c2_1[sub] = c21
    pair_c2_2[sub] = c22
    pair_is_nn1[sub] = is_nn1.astype(np.float32)
    pair_is_nn2[sub] = is_nn2.astype(np.float32)
    if gi % 10000 == 0:
        log("group", gi, gp, st, len(ci))
log("group loop done")
del tr, tgroups, cand_idx
gc.collect()

new["pair_third_min"] = pair_third
new["pair_c2_1"] = pair_c2_1
new["pair_c2_2"] = pair_c2_2
new["pair_is_nn1"] = pair_is_nn1
new["pair_is_nn2"] = pair_is_nn2

# build player nn table in original tracking order (we sorted, so reconstruct)
tr2 = []
for f in ["train_player_tracking.csv", "test_player_tracking.csv"]:
    d = pd.read_csv(f"{ROOT}/data/{f}", usecols=["game_play", "step", "nfl_player_id"])
    tr2.append(d)
tr2 = pd.concat(tr2, ignore_index=True)
tr2["gp"] = tr2["game_play"].astype(str)
tr2["step"] = tr2["step"].astype(np.int32)
tr2["player"] = tr2["nfl_player_id"].astype(np.int64)
tr2 = tr2.sort_values(["gp", "step", "player"]).reset_index(drop=True)
pf = pd.DataFrame({"split": 0, "gp": tr2["gp"].values, "step": tr2["step"].values,
                   "player": tr2["player"].values})
for j, c in enumerate(cols):
    pf[c] = P[:, j]
del tr2, P
gc.collect()
# per-player temporal deltas of nearest-neighbour
pf = pf.sort_values(["gp", "player", "step"]).reset_index(drop=True)
gpf = pf.groupby(["gp", "player"], sort=False)
pf["nb1_lag1"] = gpf["nb1"].shift(1).values
pf["nb1_lag3"] = gpf["nb1"].shift(3).values
pf["nb1_lead1"] = gpf["nb1"].shift(-1).values
pf["nb1_lead3"] = gpf["nb1"].shift(-3).values
pf["nb1_d1"] = pf["nb1"] - pf["nb1_lag1"]
pf["nb1_fd1"] = pf["nb1_lead1"] - pf["nb1"]
pf["n2_lag1"] = gpf["n2"].shift(1).values
pf["n2_lead1"] = gpf["n2"].shift(-1).values
pf.to_parquet(f"{ROOT}/workspaces/canopus_pi/features/player_nn.parquet", index=False)
log("player_nn saved", pf.shape)

# merge player nn into candidates for p1 and p2
pf_cols = [c for c in pf.columns if c not in ("split", "gp", "player", "step")]
for suffix, idcol in [("1", "p1"), ("2", "p2n")]:
    t = pf.rename(columns={"player": idcol})
    mm = meta[["gp", "step", idcol]].merge(t, on=["gp", "step", idcol], how="left")
    for c in pf_cols:
        new[f"{c}_{suffix}"] = mm[c].values
    log("merged nn", suffix)

# ---------------- helmet temporal per-player ----------------
log("helmet temporal")
hp = pd.read_parquet(f"{ROOT}/workspaces/canopus_pi/features/helmet_player.parquet")
hp["gp"] = hp["gp"].astype(str)
hp["player"] = hp["player"].astype(np.int64)
hp["step"] = hp["step"].astype(np.int32)
hp = hp.sort_values(["gp", "player", "step"]).reset_index(drop=True)
gh = hp.groupby(["gp", "player"], sort=False)
hc = []
for view in ["sid", "end"]:
    for c in ["h_top", "h_cy", "h_h"]:
        src = f"{view}_{c}"
        if src not in hp.columns:
            continue
        for k in [1, 3]:
            hp[f"{src}_lag{k}"] = gh[src].shift(k).values
            hp[f"{src}_lead{k}"] = gh[src].shift(-k).values
        hp[f"{src}_d1"] = hp[src] - hp[f"{src}_lag1"]
        hp[f"{src}_fd1"] = hp[f"{src}_lead1"] - hp[src]
        hp[f"{src}_fd3"] = hp[f"{src}_lead3"] - hp[src]
        hp[f"{src}_pmed"] = hp[src] - gh[src].transform("median").values
        hc += [f"{src}_d1", f"{src}_fd1", f"{src}_fd3", f"{src}_pmed", f"{src}_lag3", f"{src}_lead3"]
hp["helmet_n_views"] = hp[["sid_h_top", "end_h_top"]].notna().sum(1).values
hc.append("helmet_n_views")
hp.to_parquet(f"{ROOT}/workspaces/canopus_pi/features/helmet_player_v2.parquet", index=False)
log("helmet v2", len(hc), "cols")

for suffix, idcol in [("1", "p1"), ("2", "p2n")]:
    t = hp[["gp", "player", "step"] + hc].rename(columns={"player": idcol})
    mm = meta[["gp", "step", idcol]].merge(t, on=["gp", "step", idcol], how="left")
    for c in hc:
        new[f"{c}_{suffix}"] = mm[c].values
    log("merged helmet", suffix)

# fill pair-only cols for ground with NaN automatically (already)
new = new.drop(columns=["contact_id"])
for c in new.columns:
    new[c] = new[c].astype(np.float32)
assert len(new) == n
new.to_parquet(OUT, index=False)
log("saved", new.shape, new.memory_usage(deep=True).sum() / 1e9)
print(new.columns.tolist())
