import pandas as pd, numpy as np, gc, time

ROOT = "/download_job_code/payload/sandbox/data"
OUT = "/download_job_code/payload/sandbox/workspaces/achernar_sigma"
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)

POSITIONS = None

def load_track(path):
    d = pd.read_csv(path, usecols=["game_play","step","nfl_player_id","team","position","jersey_number",
                                   "x_position","y_position","speed","distance","direction","orientation",
                                   "acceleration","sa"])
    d = d.sort_values(["game_play","nfl_player_id","step"]).reset_index(drop=True)
    return d

def build_player_features(d):
    log("player features", d.shape)
    g = d.groupby(["game_play","nfl_player_id"], sort=False)
    # velocity from central position differences
    xp = g.x_position.shift(1); xn = g.x_position.shift(-1)
    yp = g.y_position.shift(1); yn = g.y_position.shift(-1)
    dtp = g.step.diff(); dtn = g.step.shift(-1) - d.step
    vx = (xn - xp) / (((dtn + dtp).replace(0, np.nan)) * 0.1)
    vy = (yn - yp) / (((dtn + dtp).replace(0, np.nan)) * 0.1)
    # fallback one-sided
    vx = vx.fillna((xn - d.x_position)/(dtn*0.1)).fillna((d.x_position - xp)/(dtp*0.1))
    vy = vy.fillna((yn - d.y_position)/(dtn*0.1)).fillna((d.y_position - yp)/(dtp*0.1))
    # direction/speed based fallback
    ang = np.deg2rad(d.direction.values)
    vx = vx.fillna(pd.Series(d.speed.values*np.sin(ang), index=d.index))
    vy = vy.fillna(pd.Series(d.speed.values*np.cos(ang), index=d.index))
    d["vx"] = vx.astype(np.float32); d["vy"] = vy.astype(np.float32)
    d["speed_lag1"] = g.speed.shift(1).astype(np.float32)
    d["speed_lead1"] = g.speed.shift(-1).astype(np.float32)
    d["speed_d1"] = (g.speed.shift(-1) - g.speed.shift(1)).astype(np.float32)
    d["acc_lag1"] = g.acceleration.shift(1).astype(np.float32)
    d["acc_lead1"] = g.acceleration.shift(-1).astype(np.float32)
    d["sa_lag1"] = g.sa.shift(1).astype(np.float32)
    d["sa_lead1"] = g.sa.shift(-1).astype(np.float32)
    d["cum_dist"] = g.distance.cumsum().astype(np.float32)
    d["x_lag1"] = g.x_position.shift(1).astype(np.float32)
    d["y_lag1"] = g.y_position.shift(1).astype(np.float32)
    # orientation vs direction angular diff
    dd = np.abs(d.orientation.values - d.direction.values) % 360
    d["orient_dir_diff"] = np.minimum(dd, 360-dd).astype(np.float32)
    d["speed_rmean5"] = g.speed.transform(lambda x: x.rolling(5, center=True, min_periods=1).mean()).astype(np.float32)
    d["speed_rmin5"] = g.speed.transform(lambda x: x.rolling(5, center=True, min_periods=1).min()).astype(np.float32)
    d["sa_rmean5"] = g.sa.transform(lambda x: x.rolling(5, center=True, min_periods=1).mean()).astype(np.float32)
    d["acc_rmean5"] = g.acceleration.transform(lambda x: x.rolling(5, center=True, min_periods=1).mean()).astype(np.float32)
    return d

def neighbor_features(d):
    log("neighbor features")
    out_parts = []
    for gp, sub in d.groupby("game_play", sort=False):
        steps = np.sort(sub.step.unique())
        sidx = {s:i for i,s in enumerate(steps)}
        pids = np.sort(sub.nfl_player_id.unique())
        pidx = {p:i for i,p in enumerate(pids)}
        nS, nP = len(steps), len(pids)
        X = np.full((nS,nP), np.nan); Y = np.full((nS,nP), np.nan); Tm = np.zeros((nS,nP), dtype=np.int8)
        si = sub.step.map(sidx).values; pi = sub.nfl_player_id.map(pidx).values
        X[si,pi] = sub.x_position.values; Y[si,pi] = sub.y_position.values
        Tm[si,pi] = (sub.team.values == "home").astype(np.int8)
        dx = X[:,:,None] - X[:,None,:]
        dy = Y[:,:,None] - Y[:,None,:]
        D = np.sqrt(dx*dx + dy*dy)
        # set self distance to nan
        idx = np.arange(nP)
        D[:, idx, idx] = np.nan
        nn_any = np.nanmin(D, axis=2)
        same = Tm[:,:,None] == Tm[:,None,:]
        Dopp = np.where(same, np.nan, D)
        Dteam = np.where(~same, np.nan, D)
        nn_opp = np.nanmin(Dopp, axis=2)
        nn_team = np.nanmin(Dteam, axis=2)
        n_within2 = np.nansum(D < 2.0, axis=2)
        n_opp_within2 = np.nansum(Dopp < 2.0, axis=2)
        n_within4 = np.nansum(D < 4.0, axis=2)
        # speed of nearest opponent
        Sp = np.full((nS,nP), np.nan)
        Sp[si,pi] = sub.speed.values
        # nearest opponent id
        Dopp2 = np.where(np.isnan(Dopp), np.inf, Dopp)
        nearest = np.argmin(Dopp2, axis=2)
        speed_nn_opp = Sp[np.arange(nS)[:, None], nearest]
        speed_nn_opp[~np.isfinite(np.nanmin(Dopp2, axis=2))] = np.nan
        nn_opp = np.where(np.isfinite(nn_opp), nn_opp, np.nan)
        for k,p in enumerate(pids):
            valid = ~np.isnan(X[:,k])
            out_parts.append(pd.DataFrame({
                "game_play": gp, "nfl_player_id": p, "step": steps[valid],
                "nn_any": nn_any[valid,k].astype(np.float32),
                "nn_opp": nn_opp[valid,k].astype(np.float32),
                "nn_team": nn_team[valid,k].astype(np.float32),
                "n_within2": n_within2[valid,k].astype(np.float32),
                "n_opp_within2": n_opp_within2[valid,k].astype(np.float32),
                "n_within4": n_within4[valid,k].astype(np.float32),
                "speed_nn_opp": speed_nn_opp[valid,k].astype(np.float32),
            }))
    nb = pd.concat(out_parts, ignore_index=True)
    log("neighbor done", nb.shape)
    return nb

def load_labels(path, is_test):
    if is_test:
        ss = pd.read_csv(path)
        parts = ss.contact_id.str.split("_", expand=True)
        d = pd.DataFrame({"contact_id": ss.contact_id,
                          "game_play": parts[0] + "_" + parts[1],
                          "step": parts[2].astype(np.int32),
                          "nfl_player_id_1": parts[3].astype(np.int64),
                          "nfl_player_id_2": parts[4]})
        d["target"] = np.nan
        return d
    return pd.read_csv(path, usecols=["contact_id","game_play","step","nfl_player_id_1","nfl_player_id_2","contact"])

def make_pair_features(lab, pl):
    d = lab.copy()
    d["nfl_player_id_1"] = d.nfl_player_id_1.astype(np.int64)
    p2i = pd.to_numeric(d.nfl_player_id_2, errors="coerce").fillna(-1).astype(np.int64)
    d["p2i"] = p2i
    is_g = (d.nfl_player_id_2 == "G").values
    # merge player1
    cols = ["game_play","step","nfl_player_id","x_position","y_position","vx","vy","speed","acceleration","sa",
            "direction","orientation","team","position_code","jersey_number","cum_dist","speed_lag1","speed_lead1",
            "speed_d1","acc_lag1","acc_lead1","sa_lag1","sa_lead1","orient_dir_diff","speed_rmean5","speed_rmin5",
            "sa_rmean5","acc_rmean5","nn_any","nn_opp","nn_team","n_within2","n_opp_within2","n_within4","speed_nn_opp"]
    p = pl
    d = d.merge(p.add_suffix("_1"), left_on=["game_play","step","nfl_player_id_1"],
                right_on=["game_play_1","step_1","nfl_player_id_1"], how="left")
    d = d.merge(p.add_suffix("_2"), left_on=["game_play","step","p2i"],
                right_on=["game_play_2","step_2","nfl_player_id_2"], how="left")
    d.drop(columns=[c for c in ["game_play_1","step_1","game_play_2","step_2","nfl_player_id_2"] if c in d.columns],
           inplace=True)
    # position/team categoricals
    d["same_team"] = (d.team_1 == d.team_2).astype(np.float32)
    d["same_pos"] = (d.position_code_1 == d.position_code_2).astype(np.float32)
    d["dx"] = (d.x_position_2 - d.x_position_1).astype(np.float32)
    d["dy"] = (d.y_position_2 - d.y_position_1).astype(np.float32)
    d["dist"] = np.sqrt(d.dx**2 + d.dy**2).astype(np.float32)
    d["adx"] = d.dx.abs().astype(np.float32); d["ady"] = d.dy.abs().astype(np.float32)
    dvx = d.vx_1 - d.vx_2; dvy = d.vy_1 - d.vy_2
    d["rel_speed"] = np.sqrt(dvx**2 + dvy**2).astype(np.float32)
    d["closing"] = (-(d.dx*dvx + d.dy*dvy) / (d.dist + 1e-3)).astype(np.float32)
    d["speed_sum"] = (d.speed_1 + d.speed_2).astype(np.float32)
    d["speed_max"] = np.maximum(d.speed_1, d.speed_2).astype(np.float32)
    d["speed_min"] = np.minimum(d.speed_1, d.speed_2).astype(np.float32)
    d["speed_absdiff"] = (d.speed_1 - d.speed_2).abs().astype(np.float32)
    d["acc_sum"] = (d.acceleration_1 + d.acceleration_2).astype(np.float32)
    d["sa_sum"] = (d.sa_1 + d.sa_2).astype(np.float32)
    d["sa_absdiff"] = (d.sa_1 - d.sa_2).abs().astype(np.float32)
    # cos of each player's velocity with vector toward the other
    eps = 1e-3
    d["toward1"] = ((d.vx_1*d.dx + d.vy_1*d.dy) / (d.speed_1*d.dist + eps)).astype(np.float32)
    d["toward2"] = ((d.vx_2*(-d.dx) + d.vy_2*(-d.dy)) / (d.speed_2*d.dist + eps)).astype(np.float32)
    d["is_nn1"] = (d.dist <= d.nn_any_1 + 0.05).astype(np.float32)
    d["is_nn2"] = (d.dist <= d.nn_any_2 + 0.05).astype(np.float32)
    d["is_nn_opp1"] = (d.dist <= d.nn_opp_1 + 0.05).astype(np.float32)
    d["is_nn_opp2"] = (d.dist <= d.nn_opp_2 + 0.05).astype(np.float32)
    return d, is_g

def add_temporal_pair(d, is_g):
    log("pair temporal")
    gcols = ["game_play","nfl_player_id_1","p2i"]
    d["_gid"] = d.groupby(gcols, sort=False).ngroup()
    d = d.sort_values(gcols + ["step"]).reset_index(drop=True)
    g = d.groupby("_gid", sort=False)
    for c in ["dist"]:
        d[c+"_lag1"] = g[c].shift(1).astype(np.float32)
        d[c+"_lead1"] = g[c].shift(-1).astype(np.float32)
        d[c+"_lag2"] = g[c].shift(2).astype(np.float32)
        d[c+"_lead2"] = g[c].shift(-2).astype(np.float32)
        d[c+"_lag3"] = g[c].shift(3).astype(np.float32)
        d[c+"_lead3"] = g[c].shift(-3).astype(np.float32)
        d[c+"_d1"] = (g[c].shift(-1) - g[c].shift(1)).astype(np.float32)
    d["dist_rmin3"] = g["dist"].transform(lambda x: x.rolling(3, center=True, min_periods=1).min()).astype(np.float32)
    d["dist_rmin7"] = g["dist"].transform(lambda x: x.rolling(7, center=True, min_periods=1).min()).astype(np.float32)
    d["dist_rmean5"] = g["dist"].transform(lambda x: x.rolling(5, center=True, min_periods=1).mean()).astype(np.float32)
    d["closing_rmean5"] = g["closing"].transform(lambda x: x.rolling(5, center=True, min_periods=1).mean()).astype(np.float32)
    d["close_flag"] = (d["dist"] < 1.5).astype(np.float32)
    d["past_close_5"] = g["close_flag"].transform(lambda x: x.rolling(5, min_periods=1).mean()).astype(np.float32)
    d.drop(columns=["_gid","close_flag"], inplace=True)
    return d

def main():
    tr = load_track(f"{ROOT}/train_player_tracking.csv")
    te = load_track(f"{ROOT}/test_player_tracking.csv")
    log("tracks", tr.shape, te.shape)
    # global position mapping
    allpos = pd.concat([tr.position, te.position]).astype("category")
    cats = allpos.cat.categories
    posmap = {p:i for i,p in enumerate(cats)}
    tr["position_code"] = tr.position.map(posmap).astype(np.int16)
    te["position_code"] = te.position.map(posmap).astype(np.int16)
    tr["team"] = (tr.team == "home").astype(np.int8)
    te["team"] = (te.team == "home").astype(np.int8)
    del allpos, cats
    tr = build_player_features(tr)
    te = build_player_features(te)
    trnb = neighbor_features(tr)
    tenb = neighbor_features(te)
    tr = tr.merge(trnb, on=["game_play","nfl_player_id","step"], how="left")
    te = te.merge(tenb, on=["game_play","nfl_player_id","step"], how="left")
    del trnb, tenb; gc.collect()
    # temporal context around nearest-neighbour / opponent-count signals
    newcols = []
    for df in (tr, te):
        df.sort_values(["game_play","nfl_player_id","step"], inplace=True)
        g = df.groupby(["game_play","nfl_player_id"], sort=False)
        for c in ["nn_opp","nn_any","n_opp_within2","speed_nn_opp"]:
            df[c+"_lag1"] = g[c].shift(1).astype(np.float32); newcols.append(c+"_lag1")
            df[c+"_lead1"] = g[c].shift(-1).astype(np.float32); newcols.append(c+"_lead1")
            df[c+"_d1"] = (g[c].shift(-1) - g[c].shift(1)).astype(np.float32); newcols.append(c+"_d1")
        for c in ["nn_opp","n_opp_within2"]:
            for stat, fn in [("rmin5","min"), ("rmax5","max"), ("rmean5","mean")]:
                df[c+"_"+stat] = g[c].transform(lambda x, fn=fn: x.rolling(5, center=True, min_periods=1).agg(fn)).astype(np.float32)
                newcols.append(c+"_"+stat)
        for c in ["speed","sa"]:
            df[c+"_lag3"] = g[c].shift(3).astype(np.float32); newcols.append(c+"_lag3")
            df[c+"_lead3"] = g[c].shift(-3).astype(np.float32); newcols.append(c+"_lead3")
    newcols = sorted(set(newcols))
    log("newcols", len(newcols), newcols[:5])
    trcols = ["game_play","step","nfl_player_id","x_position","y_position","vx","vy","speed","acceleration","sa",
              "direction","orientation","team","position_code","jersey_number","cum_dist","speed_lag1","speed_lead1",
              "speed_d1","acc_lag1","acc_lead1","sa_lag1","sa_lead1","orient_dir_diff","speed_rmean5","speed_rmin5",
              "sa_rmean5","acc_rmean5","nn_any","nn_opp","nn_team","n_within2","n_opp_within2","n_within4","speed_nn_opp"] + newcols
    trp = tr[trcols].astype({c:np.float32 for c in trcols if c not in ["game_play","step","nfl_player_id"]})
    tep = te[trcols].astype({c:np.float32 for c in trcols if c not in ["game_play","step","nfl_player_id"]})
    del tr, te; gc.collect()

    for split, lpath, pl, is_test in [
        ("train", f"{ROOT}/train_labels.csv", trp, False),
        ("test", f"{ROOT}/sample_submission.csv", tep, True),
    ]:
        lab = load_labels(lpath, is_test)
        d, is_g = make_pair_features(lab, pl)
        d = add_temporal_pair(d, is_g)
        # teammate/position categorical: keep ints
        d["step"] = d["step"].astype(np.int32)
        d.drop(columns=["nfl_player_id_1","p2i","nfl_player_id_2"], inplace=True, errors="ignore")
        keep = ["contact_id","target","step"] + [c for c in d.columns if c not in ["contact_id","target","step"]]
        d = d[[c for c in keep if c in d.columns]]
        out = f"{OUT}/{split}_tracking_features.parquet"
        d.to_parquet(out, index=False)
        log(split, "saved", out, d.shape, "ground", int(is_g.sum()))

if __name__ == "__main__":
    main()
