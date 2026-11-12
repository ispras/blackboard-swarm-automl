import os, sys, time, gc
import numpy as np
import pandas as pd

DATA = "/download_job_code/payload/sandbox/data"
OUT = "/download_job_code/payload/sandbox/workspaces/canopus_pi/features"
os.makedirs(OUT, exist_ok=True)
FPS = 59.94
FRAMES_PER_STEP = FPS / 10.0
SNAP_FRAME = 300.0

t0 = time.time()
def log(*a):
    print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)

def ang_diff(a, b):
    d = np.abs(a - b) % 360.0
    return np.minimum(d, 360.0 - d)

# ---------------- candidate rows ----------------
log("load candidates")
lab = pd.read_csv(f"{DATA}/train_labels.csv",
                  usecols=["contact_id", "game_play", "step", "nfl_player_id_1",
                           "nfl_player_id_2", "contact"])
lab["split"] = 0
ss = pd.read_csv(f"{DATA}/sample_submission.csv")
parts = ss.contact_id.str.split("_", expand=True)
ss["game_play"] = parts[0] + "_" + parts[1]
ss["step"] = parts[2].astype(np.int32)
ss["nfl_player_id_1"] = parts[3].astype(np.int64)
ss["nfl_player_id_2"] = parts[4]
ss["contact"] = np.nan
ss["split"] = 1
ss = ss[["contact_id", "game_play", "step", "nfl_player_id_1",
         "nfl_player_id_2", "contact", "split"]]
lab["nfl_player_id_2"] = lab["nfl_player_id_2"].astype(str)
ss["nfl_player_id_2"] = ss["nfl_player_id_2"].astype(str)
cand = pd.concat([lab, ss], ignore_index=True)
cand["is_ground"] = (cand["nfl_player_id_2"] == "G").astype(np.int8)
cand["game_play"] = cand["game_play"].astype("category")
cand["step"] = cand["step"].astype(np.int32)
cand["nfl_player_id_1"] = cand["nfl_player_id_1"].astype(np.int64)
# numeric p2 (-1 for ground) for joins
cand["p2_num"] = pd.to_numeric(cand["nfl_player_id_2"], errors="coerce").fillna(-1).astype(np.int64)
log("cand", cand.shape, "pos", cand.contact.mean())
gp_codes = cand["game_play"].cat.categories
cand["gp_code"] = cand["game_play"].cat.codes.astype(np.int32)

# ---------------- tracking features ----------------
log("load tracking")
tr = []
for sp, f in [(0, "train_player_tracking.csv"), (1, "test_player_tracking.csv")]:
    d = pd.read_csv(f"{DATA}/{f}")
    d["split"] = sp
    tr.append(d)
tr = pd.concat(tr, ignore_index=True)
tr["game_play"] = tr["game_play"].astype("category")
# align categories with candidate (test plays not in train cats -> use codes based on concat)
tr["gp"] = tr["game_play"].astype(str)
cand["gp"] = cand["game_play"].astype(str)
tr["player"] = tr["nfl_player_id"].astype(np.int64)
tr["step"] = tr["step"].astype(np.int32)
tr = tr.sort_values(["split", "gp", "player", "step"]).reset_index(drop=True)
log("tracking", tr.shape)

g = tr.groupby(["split", "gp", "player"], sort=False)
base = ["x_position", "y_position", "speed", "distance", "direction",
        "orientation", "acceleration", "sa"]
feat = pd.DataFrame({
    "split": tr["split"].values, "gp": tr["gp"].values,
    "player": tr["player"].values, "step": tr["step"].values,
})
# raw + shifted features
for c in base:
    feat[c] = tr[c].astype(np.float32).values

for k in [1, 2, 3, 5]:
    for c in ["speed", "acceleration", "sa", "direction", "orientation"]:
        feat[f"{c}_lag{k}"] = g[c].shift(k).astype(np.float32).values
for k in [1, 2, 3]:
    for c in ["speed", "acceleration", "sa"]:
        feat[f"{c}_lead{k}"] = g[c].shift(-k).astype(np.float32).values

feat["speed_d1"] = feat["speed"] - feat["speed_lag1"]
feat["speed_d3"] = feat["speed"] - feat["speed_lag3"]
feat["speed_d5"] = feat["speed"] - feat["speed_lag5"]
feat["speed_lead_d1"] = feat["speed_lead1"] - feat["speed"]
feat["accel_d1"] = feat["acceleration"] - feat["acceleration_lag1"]
feat["sa_d1"] = feat["sa"] - feat["sa_lag1"]
feat["sa_d3"] = feat["sa"] - feat["sa_lag3"]
feat["jerk"] = feat["acceleration"] - feat["acceleration_lag1"]
feat["dir_change1"] = ang_diff(feat["direction"].values, feat["direction_lag1"].values).astype(np.float32)
feat["dir_change3"] = ang_diff(feat["direction"].values, feat["direction_lag3"].values).astype(np.float32)
feat["ori_change1"] = ang_diff(feat["orientation"].values, feat["orientation_lag1"].values).astype(np.float32)
feat["ori_change3"] = ang_diff(feat["orientation"].values, feat["orientation_lag3"].values).astype(np.float32)

# rolling means/max using lags (past windows)
for w in [3, 5, 10]:
    cols = [f"{c}_lag{k}" if k > 0 else c for k in range(w)]
    # ensure base present
    for c in ["speed", "acceleration", "sa"]:
        arrs = []
        for k in range(w):
            col = c if k == 0 else f"{c}_lag{k}"
            if col not in feat.columns:
                feat[col] = g[c].shift(k).astype(np.float32).values
            arrs.append(feat[col].values)
        M = np.vstack(arrs)
        feat[f"{c}_roll{w}_mean"] = np.nanmean(M, axis=0).astype(np.float32)
        feat[f"{c}_roll{w}_max"] = np.nanmax(M, axis=0).astype(np.float32)
        feat[f"{c}_roll{w}_min"] = np.nanmin(M, axis=0).astype(np.float32)

# future speed window
for w in [3, 5]:
    arrs = []
    for k in range(0, w):
        col = "speed" if k == 0 else f"speed_lead{k}"
        if col not in feat.columns:
            feat[col] = g["speed"].shift(-k).astype(np.float32).values
        arrs.append(feat[col].values)
    M = np.vstack(arrs)
    feat[f"speed_fut{w}_mean"] = np.nanmean(M, axis=0).astype(np.float32)
    feat[f"speed_fut{w}_max"] = np.nanmax(M, axis=0).astype(np.float32)
    feat[f"speed_fut{w}_min"] = np.nanmin(M, axis=0).astype(np.float32)

feat["team_home"] = (tr["team"] == "home").astype(np.int8).values
pos_series = tr["position"].astype("category")
feat["position_code"] = pos_series.cat.codes.astype(np.int16).values
feat["jersey"] = tr["jersey_number"].astype(np.float32).values

# cumulative motion / path features
feat["x_lag1"] = g["x_position"].shift(1).astype(np.float32).values
feat["y_lag1"] = g["y_position"].shift(1).astype(np.float32).values
feat["vx"] = (feat["x_position"] - feat["x_lag1"]) / 0.1
feat["vy"] = (feat["y_position"] - feat["y_lag1"]) / 0.1

log("tracking feats", feat.shape)
feat.to_parquet(f"{OUT}/track_feats.parquet", index=False)
del tr, g
gc.collect()

# ---------------- merge tracking into candidates ----------------
log("merge track into candidates")
tf_cols = [c for c in feat.columns if c not in ("split", "gp", "player", "step")]
tf = feat[["split", "gp", "player", "step"] + tf_cols].copy()

cand = cand.reset_index(drop=True)
cand["_row"] = np.arange(len(cand), dtype=np.int64)

# player 1 join
tf1 = tf.rename(columns={"player": "nfl_player_id_1"})
a = cand[["_row", "split", "gp", "step", "nfl_player_id_1"]].merge(
    tf1, on=["split", "gp", "step", "nfl_player_id_1"], how="left")
a = a.rename(columns={c: c + "_1" for c in tf_cols})
a = a[["_row"] + [c + "_1" for c in tf_cols]]
a = a.sort_values("_row").reset_index(drop=True)

# player 2 join (only player-player rows)
p2 = cand.loc[cand.is_ground == 0, ["_row", "split", "gp", "step", "p2_num"]]
tf2 = tf.rename(columns={"player": "p2_num"})
b = p2.merge(tf2, on=["split", "gp", "step", "p2_num"], how="left")
b = b.rename(columns={c: c + "_2" for c in tf_cols})
b = b[["_row"] + [c + "_2" for c in tf_cols]]
b = b.sort_values("_row").reset_index(drop=True)

cand = cand.merge(a, on="_row", how="left")
cand = cand.merge(b, on="_row", how="left")
del a, b, p2
gc.collect()
log("after track merge", cand.shape)

# ---------------- pairwise tracking geometry ----------------
log("pair geometry")
dx = cand["x_position_1"] - cand["x_position_2"]
dy = cand["y_position_1"] - cand["y_position_2"]
dist = np.sqrt(dx * dx + dy * dy)
cand["dist"] = dist.astype(np.float32)
cand["dx_field"] = dx.astype(np.float32)
cand["dy_field"] = dy.astype(np.float32)
cand["same_team"] = (cand["team_home_1"] == cand["team_home_2"]).astype(np.float32)
cand["speed_sum"] = (cand["speed_1"] + cand["speed_2"]).astype(np.float32)
cand["speed_diff"] = (cand["speed_1"] - cand["speed_2"]).astype(np.float32)
cand["accel_sum"] = (cand["acceleration_1"] + cand["acceleration_2"]).astype(np.float32)
cand["accel_diff"] = (cand["acceleration_1"] - cand["acceleration_2"]).astype(np.float32)
cand["sa_sum"] = (cand["sa_1"] + cand["sa_2"]).astype(np.float32)
cand["sa_min"] = np.minimum(cand["sa_1"], cand["sa_2"]).astype(np.float32)
cand["dir_diff"] = ang_diff(cand["direction_1"].values, cand["direction_2"].values).astype(np.float32)
cand["ori_diff"] = ang_diff(cand["orientation_1"].values, cand["orientation_2"].values).astype(np.float32)

# direction of line between players
line_ang = np.degrees(np.arctan2(dy, dx))
cand["line_ang"] = line_ang.astype(np.float32)
cand["dir1_to_line"] = ang_diff(cand["direction_1"].values, line_ang).astype(np.float32)
cand["dir2_to_line"] = ang_diff(cand["direction_2"].values, line_ang + 180.0).astype(np.float32)
cand["ori1_to_line"] = ang_diff(cand["orientation_1"].values, line_ang).astype(np.float32)

# projection of relative velocity onto connecting line (closing speed)
vx1 = cand["vx_1"].values; vy1 = cand["vy_1"].values
vx2 = cand["vx_2"].values; vy2 = cand["vy_2"].values
ux = np.where(dist > 1e-6, dx / dist, 0.0)
uy = np.where(dist > 1e-6, dy / dist, 0.0)
rel_vx = vx1 - vx2; rel_vy = vy1 - vy2
cand["closing_speed"] = -(rel_vx * ux + rel_vy * uy)
cand["rel_speed"] = np.sqrt(rel_vx ** 2 + rel_vy ** 2).astype(np.float32)

# ---------------- temporal pair distance features ----------------
log("pair temporal features")
ppmask = cand["is_ground"] == 0
pairkey = ["split", "gp", "nfl_player_id_1", "p2_num"]
pp = cand.loc[ppmask, ["_row"] + pairkey + ["step", "dist", "closing_speed"]].copy()
pp = pp.sort_values(pairkey + ["step"]).reset_index(drop=True)
gpp = pp.groupby(pairkey, sort=False)
for k in [1, 2, 3, 5]:
    pp[f"dist_lag{k}"] = gpp["dist"].shift(k).values
    pp[f"dist_lead{k}"] = gpp["dist"].shift(-k).values
for k in [1, 3]:
    pp[f"dist_d{k}"] = pp["dist"] - pp[f"dist_lag{k}"]
    pp[f"dist_fd{k}"] = pp[f"dist_lead{k}"] - pp["dist"]

def roll_stat(df, gcol, col, w, kind):
    arrs = []
    for k in range(0, w):
        if k == 0:
            arrs.append(df[col].values)
        else:
            arrs.append(df.groupby(gcol, sort=False)[col].shift(k).values)
    M = np.vstack(arrs)
    if kind == "min":
        return np.nanmin(M, axis=0)
    if kind == "mean":
        return np.nanmean(M, axis=0)
    if kind == "max":
        return np.nanmax(M, axis=0)

for w in [3, 5]:
    pp[f"dist_roll{w}_min"] = roll_stat(pp, pairkey, "dist", w, "min")
    pp[f"dist_roll{w}_mean"] = roll_stat(pp, pairkey, "dist", w, "mean")

# future window min distance (lead)
def fut_stat(df, gcol, col, w, kind):
    arrs = []
    for k in range(0, w):
        if k == 0:
            arrs.append(df[col].values)
        else:
            arrs.append(df.groupby(gcol, sort=False)[col].shift(-k).values)
    M = np.vstack(arrs)
    if kind == "min":
        return np.nanmin(M, axis=0)
    return np.nanmean(M, axis=0)

for w in [3, 5]:
    pp[f"dist_fut{w}_min"] = fut_stat(pp, pairkey, "dist", w, "min")
    pp[f"dist_fut{w}_mean"] = fut_stat(pp, pairkey, "dist", w, "mean")

# rolling speed/accel of both players already in tf; add pair-sum dynamics
pp["speed_sum_lag1"] = 0.0  # placeholder
tmpcols = ["dist_lag1", "dist_lag2", "dist_lag3", "dist_lag5", "dist_lead1",
           "dist_lead2", "dist_lead3", "dist_lead5", "dist_d1", "dist_d3",
           "dist_fd1", "dist_fd3", "dist_roll3_min", "dist_roll3_mean",
           "dist_roll5_min", "dist_roll5_mean", "dist_fut3_min", "dist_fut3_mean",
           "dist_fut5_min", "dist_fut5_mean"]
pp = pp[["_row"] + tmpcols]
cand = cand.merge(pp, on="_row", how="left")
del pp, gpp
gc.collect()
log("after pair temporal", cand.shape)

# ---------------- helmet features ----------------
log("load helmets")
hel = []
for sp, f in [(0, "train_baseline_helmets.csv"), (1, "test_baseline_helmets.csv")]:
    d = pd.read_csv(f"{DATA}/{f}",
                    usecols=["game_play", "view", "frame", "nfl_player_id",
                             "left", "width", "top", "height"])
    d["split"] = sp
    hel.append(d)
hel = pd.concat(hel, ignore_index=True)
hel["gp"] = hel["game_play"].astype(str)
hel["player"] = hel["nfl_player_id"].astype(np.int64)
hel["step_est"] = np.round((hel["frame"].values - SNAP_FRAME) / FRAMES_PER_STEP).astype(np.int32)
log("helmets raw", hel.shape)

agg = hel.groupby(["split", "gp", "view", "player", "step_est"], sort=False).agg(
    h_left=("left", "mean"), h_top=("top", "mean"),
    h_w=("width", "mean"), h_h=("height", "mean"),
    h_cnt=("frame", "size")).reset_index()
agg["h_cx"] = agg["h_left"] + agg["h_w"] / 2.0
agg["h_cy"] = agg["h_top"] + agg["h_h"] / 2.0
agg["h_bottom"] = agg["h_top"] + agg["h_h"]
agg["h_area"] = agg["h_w"] * agg["h_h"]
agg.rename(columns={"step_est": "step"}, inplace=True)
# relative position within view/step
agg["top_rank"] = agg.groupby(["split", "gp", "view", "step"])["h_top"].rank(pct=True).astype(np.float32)
agg["cy_rank"] = agg.groupby(["split", "gp", "view", "step"])["h_cy"].rank(pct=True).astype(np.float32)
agg["area_rank"] = agg.groupby(["split", "gp", "view", "step"])["h_area"].rank(pct=True).astype(np.float32)
agg["top_minus_med"] = (agg["h_top"] - agg.groupby(["split", "gp", "view", "step"])["h_top"].transform("median")).astype(np.float32)
agg["h_h_med"] = agg.groupby(["split", "gp", "view", "step"])["h_h"].transform("median").astype(np.float32)

# pivot view into columns for player-level use (sideline + endzone)
def view_pivot(df, prefix):
    out = None
    for view, tag in [("Sideline", "sid"), ("Endzone", "end")]:
        d = df[df["view"] == view].copy()
        cols = ["h_left", "h_top", "h_w", "h_h", "h_cnt", "h_cx", "h_cy",
                "h_bottom", "h_area", "top_rank", "cy_rank", "area_rank",
                "top_minus_med", "h_h_med"]
        d = d[["split", "gp", "player", "step"] + cols]
        d = d.rename(columns={c: f"{tag}_{c}" for c in cols})
        if out is None:
            out = d
        else:
            out = out.merge(d, on=["split", "gp", "player", "step"], how="outer")
    return out

player_hel = view_pivot(agg, "")
player_hel.to_parquet(f"{OUT}/helmet_player.parquet", index=False)

# player level merge for p1
hl_cols = [c for c in player_hel.columns if c not in ("split", "gp", "player", "step")]
ph1 = player_hel.rename(columns={"player": "nfl_player_id_1"})
h1 = cand[["_row", "split", "gp", "step", "nfl_player_id_1"]].merge(
    ph1, on=["split", "gp", "step", "nfl_player_id_1"], how="left")
h1 = h1.rename(columns={c: c + "_1" for c in hl_cols})
cols1 = ["_row"] + [c + "_1" for c in hl_cols]
h1 = h1[cols1].sort_values("_row").reset_index(drop=True)
cand = cand.merge(h1, on="_row", how="left")
del h1; gc.collect()

# player level merge for p2
ph2 = player_hel.rename(columns={"player": "p2_num"})
h2 = cand.loc[cand.is_ground == 0, ["_row", "split", "gp", "step", "p2_num"]].merge(
    ph2, on=["split", "gp", "step", "p2_num"], how="left")
h2 = h2.rename(columns={c: c + "_2" for c in hl_cols})
cols2 = ["_row"] + [c + "_2" for c in hl_cols]
h2 = h2[cols2].sort_values("_row").reset_index(drop=True)
cand = cand.merge(h2, on="_row", how="left")
del h2; gc.collect()
log("after helmet player merge", cand.shape)

# pair helmet geometry per view
for tag in ["sid", "end"]:
    cx1 = cand[f"{tag}_h_cx_1"].values; cy1 = cand[f"{tag}_h_cy_1"].values
    cx2 = cand[f"{tag}_h_cx_2"].values; cy2 = cand[f"{tag}_h_cy_2"].values
    w1 = cand[f"{tag}_h_w_1"].values; h1_ = cand[f"{tag}_h_h_1"].values
    w2 = cand[f"{tag}_h_w_2"].values; h2_ = cand[f"{tag}_h_h_2"].values
    ddx = cx1 - cx2; ddy = cy1 - cy2
    dpx = np.sqrt(ddx ** 2 + ddy ** 2)
    cand[f"{tag}_pair_dpx"] = dpx.astype(np.float32)
    cand[f"{tag}_pair_ddx"] = ddx.astype(np.float32)
    cand[f"{tag}_pair_ddy"] = ddy.astype(np.float32)
    mw = (w1 + w2) / 2.0
    mh = (h1_ + h2_) / 2.0
    cand[f"{tag}_pair_dpx_norm"] = (dpx / np.where(mw > 0, mw, np.nan)).astype(np.float32)
    cand[f"{tag}_pair_dpy_norm"] = (ddy / np.where(mh > 0, mh, np.nan)).astype(np.float32)
    # box overlap
    l1 = cand[f"{tag}_h_left_1"].values; t1 = cand[f"{tag}_h_top_1"].values
    l2 = cand[f"{tag}_h_left_2"].values; t2 = cand[f"{tag}_h_top_2"].values
    iw = np.minimum(l1 + w1, l2 + w2) - np.maximum(l1, l2)
    ih = np.minimum(t1 + h1_, t2 + h2_) - np.maximum(t1, t2)
    iw = np.maximum(iw, 0); ih = np.maximum(ih, 0)
    inter = iw * ih
    union = w1 * h1_ + w2 * h2_ - inter
    cand[f"{tag}_pair_iou"] = (inter / np.where(union > 0, union, np.nan)).astype(np.float32)
    cand[f"{tag}_both"] = (~np.isnan(cx1) & ~np.isnan(cx2)).astype(np.int8)

# combine views
cand["pair_dpx_min"] = np.nanmin(
    np.vstack([cand["sid_pair_dpx"].values, cand["end_pair_dpx"].values]), axis=0).astype(np.float32)
cand["pair_dpx_mean"] = np.nanmean(
    np.vstack([cand["sid_pair_dpx"].values, cand["end_pair_dpx"].values]), axis=0).astype(np.float32)
cand["pair_iou_max"] = np.nanmax(
    np.vstack([cand["sid_pair_iou"].values, cand["end_pair_iou"].values]), axis=0).astype(np.float32)
cand["views_both"] = (cand["sid_both"].fillna(0) + cand["end_both"].fillna(0)).astype(np.int8)

# temporal helmet distance for pp (min over +-2 steps)
log("helmet temporal")
pp2 = cand.loc[cand.is_ground == 0, ["_row"] + pairkey + ["step", "sid_pair_dpx", "end_pair_dpx", "pair_dpx_min"]].copy()
pp2 = pp2.sort_values(pairkey + ["step"]).reset_index(drop=True)
gpp2 = pp2.groupby(pairkey, sort=False)
for w in [2, 4]:
    cols = []
    for k in range(1, w + 1):
        pp2[f"pdpx_lag{k}"] = gpp2["pair_dpx_min"].shift(k).values
        pp2[f"pdpx_lead{k}"] = gpp2["pair_dpx_min"].shift(-k).values
        cols += [f"pdpx_lag{k}", f"pdpx_lead{k}"]
    M = np.vstack([pp2["pair_dpx_min"].values] + [pp2[c].values for c in cols])
    pp2[f"pdpx_win{w*2+1}_min"] = np.nanmin(M, axis=0)
    pp2[f"pdpx_win{w*2+1}_mean"] = np.nanmean(M, axis=0)
out_cols = ["_row", "pdpx_lag1", "pdpx_lead1", "pdpx_win5_min", "pdpx_win5_mean",
            "pdpx_win9_min", "pdpx_win9_mean"]
pp2 = pp2[out_cols]
cand = cand.merge(pp2, on="_row", how="left")
del pp2, gpp2; gc.collect()

# ---------------- finalize ----------------
log("finalize")
cand["step"] = cand["step"].astype(np.int16)
cand["gp_code"] = cand["game_play"].cat.codes.astype(np.int32) if str(cand["game_play"].dtype) == "category" else cand["gp_code"]
drop_cols = ["gp", "game_play"]
# game_play is categorical; keep gp_code only
cand = cand.drop(columns=[c for c in drop_cols if c in cand.columns])
cand = cand.drop(columns=["p2_num", "_row", "nfl_player_id_2"], errors="ignore")
# downcast floats
for c in cand.columns:
    if cand[c].dtype == np.float64:
        cand[c] = cand[c].astype(np.float32)
cand.to_parquet(f"{OUT}/all_features.parquet", index=False)
log("saved", cand.shape, cand.memory_usage(deep=True).sum() / 1e9, "GB")
print(cand.columns.tolist())
