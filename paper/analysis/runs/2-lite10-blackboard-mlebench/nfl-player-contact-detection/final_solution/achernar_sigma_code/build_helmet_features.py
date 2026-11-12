import pandas as pd, numpy as np, gc, os, time

ROOT = "/download_job_code/payload/sandbox/data"
OUT = "/download_job_code/payload/sandbox/workspaces/achernar_sigma"
A = 5.994
B = float(os.environ.get("HELMET_B", "290"))
SUFFIX = os.environ.get("FEAT_SUFFIX", "")

t0 = time.time()

def log(*a):
    print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)

def load_labels(path, is_test):
    if is_test:
        ss = pd.read_csv(path)
        parts = ss.contact_id.str.split("_", expand=True)
        d = pd.DataFrame({
            "contact_id": ss.contact_id,
            "game_play": parts[0] + "_" + parts[1],
            "step": parts[2].astype(np.int16),
            "nfl_player_id_1": parts[3],
            "nfl_player_id_2": parts[4],
        })
        return d
    d = pd.read_csv(path, usecols=["contact_id","game_play","step","nfl_player_id_1","nfl_player_id_2","contact"])
    return d

def build_view_lookup(hel, view):
    h = hel[hel.view == view].copy()
    h["cx"] = h.left + h.width/2
    h["cy"] = h.top + h.height/2
    h["area"] = h.width*h.height
    h["size"] = np.sqrt(h.area)
    # frame-level context
    g = h.groupby(["game_play","frame"])
    h["cy_med"] = g.cy.transform("median")
    h["cy_rank"] = g.cy.rank(pct=True)
    h["size_med"] = g["size"].transform("median")
    h["cy_max"] = g.cy.transform("max")
    h["bottom"] = h.cy + h.height/2
    h["bottom_max"] = h.groupby(["game_play","frame"]).bottom.transform("max")

    # ---- image-space neighbour context per helmet (pile detection) ----
    cols = ["nn_hel","nn_hel_ratio","n_hel_near","max_iou","n_hel_ovl"]
    for c in cols: h[c] = np.nan
    for gp, sub in h.groupby("game_play", sort=False):
        frames = np.sort(sub.frame.unique()); fmap = {v:i for i,v in enumerate(frames)}
        pids = np.sort(sub.nfl_player_id.unique()); pmap = {v:i for i,v in enumerate(pids)}
        nF, nP = len(frames), len(pids)
        if nP < 2: continue
        X = np.full((nF,nP), np.nan); Y = np.full((nF,nP), np.nan); S = np.full((nF,nP), np.nan)
        L = np.full((nF,nP), np.nan); T = np.full((nF,nP), np.nan)
        Wd = np.full((nF,nP), np.nan); Ht = np.full((nF,nP), np.nan)
        fi = sub.frame.map(fmap).values; pi = sub.nfl_player_id.map(pmap).values
        X[fi,pi] = sub.cx.values; Y[fi,pi] = sub.cy.values; S[fi,pi] = sub["size"].values
        L[fi,pi] = (sub.cx - sub.width/2).values; T[fi,pi] = (sub.cy - sub.height/2).values
        Wd[fi,pi] = sub.width.values; Ht[fi,pi] = sub.height.values
        dx = X[:,:,None] - X[:,None,:]; dy = Y[:,:,None] - Y[:,None,:]
        D = np.sqrt(dx*dx + dy*dy)
        idx = np.arange(nP); D[:, idx, idx] = np.nan
        nn = np.nanmin(D, axis=2)
        Dinf = np.where(np.isnan(D), np.inf, D)
        arg = np.argmin(Dinf, axis=2)
        Snn = S[np.arange(nF)[:, None], arg]
        ratio = nn / np.maximum((S + Snn)/2.0, 1e-3)
        near = np.nansum(D < (2.5*S[:,:,None]), axis=2)
        R = L + Wd; B = T + Ht; A = Wd*Ht
        iw = np.maximum(0, np.minimum(R[:,:,None], R[:,None,:]) - np.maximum(L[:,:,None], L[:,None,:]))
        ih = np.maximum(0, np.minimum(B[:,:,None], B[:,None,:]) - np.maximum(T[:,:,None], T[:,None,:]))
        inter = iw*ih
        union = A[:,:,None] + A[:,None,:] - inter
        iou = inter/np.maximum(union, 1e-6)
        iou[:, idx, idx] = np.nan
        maxiou = np.nanmax(iou, axis=2)
        novl = np.nansum(iou > 0.05, axis=2)
        frames_i = np.repeat(frames, nP); pids_i = np.tile(pids, nF)
        key = pd.MultiIndex.from_arrays([np.repeat(gp, nF*nP), frames_i, pids_i])
        vals = pd.DataFrame({"nn_hel":nn.ravel(), "nn_hel_ratio":ratio.ravel(), "n_hel_near":near.ravel(),
                             "max_iou":maxiou.ravel(), "n_hel_ovl":novl.ravel()}, index=key)
        # assign back by index
        sub_idx = pd.MultiIndex.from_arrays([sub.game_play, sub.frame, sub.nfl_player_id])
        vals = vals.reindex(sub_idx)
        for c in cols:
            h.loc[sub.index, c] = vals[c].values
    h = h.set_index(["game_play","frame","nfl_player_id"])
    return h[["cx","cy","width","height","area","size","cy_med","cy_rank","size_med","cy_max","bottom_max",
              "nn_hel","nn_hel_ratio","n_hel_near","max_iou","n_hel_ovl"]]

def reindex_view(h, d, pid):
    mi = pd.MultiIndex.from_arrays([d.game_play.values, d.frame.values, d[pid].values])
    return h.reindex(mi).reset_index(drop=True)

def main():
    only_test = os.environ.get("ONLY_TEST", "0") == "1"
    splits = [("test", f"{ROOT}/sample_submission.csv", True)]
    if not only_test:
        splits = [("train", f"{ROOT}/train_labels.csv", False)] + splits
    for split, lpath, is_test in splits:
        # IMPORTANT: per-split helmet source (train vs test)
        hpath = f"{ROOT}/{split}_baseline_helmets.csv"
        hel = pd.read_csv(hpath,
                          usecols=["game_play","view","frame","nfl_player_id","left","width","top","height"])
        hel = hel[hel.view.isin(["Endzone","Sideline"])].copy()
        log(split, "helmets loaded", hel.shape)
        lookups = {v: build_view_lookup(hel, v) for v in ["Endzone","Sideline"]}
        del hel; gc.collect()
        d = load_labels(lpath, is_test)
        d["frame"] = np.round(A*d.step.values.astype(np.float64) + B).astype(np.int32)
        d["nfl_player_id_1"] = d.nfl_player_id_1.astype(np.int64)
        is_g = (d.nfl_player_id_2 == "G").values
        p2 = d.nfl_player_id_2.copy()
        p2i = pd.to_numeric(p2, errors="coerce").fillna(-1).astype(np.int64).values
        d["p2i"] = p2i
        log(split, "rows", len(d), "ground", int(is_g.sum()))

        feats = pd.DataFrame({"contact_id": d.contact_id.values})
        for v in ["Endzone","Sideline"]:
            h = lookups[v]
            m1 = reindex_view(h, d, "nfl_player_id_1")
            m2 = reindex_view(h, d, "p2i")
            c1x, c1y = m1.cx.values, m1.cy.values
            c2x, c2y = m2.cx.values, m2.cy.values
            w1, h1, s1, a1 = m1.width.values, m1.height.values, m1["size"].values, m1.area.values
            w2, h2, s2, a2 = m2.width.values, m2.height.values, m2["size"].values, m2.area.values
            dx = c1x - c2x; dy = c1y - c2y
            dist = np.sqrt(dx*dx + dy*dy)
            meansize = (s1 + s2)/2
            present = (~np.isnan(dist)).astype(np.float32)

            # IoU
            def iou():
                l1 = c1x - w1/2; r1 = c1x + w1/2; t1 = c1y - h1/2; btm1 = c1y + h1/2
                l2 = c2x - w2/2; r2 = c2x + w2/2; t2 = c2y - h2/2; btm2 = c2y + h2/2
                iw = np.maximum(0, np.minimum(r1,r2) - np.maximum(l1,l2))
                ih = np.maximum(0, np.minimum(btm1,btm2) - np.maximum(t1,t2))
                inter = iw*ih
                union = a1 + a2 - inter
                return inter/union
            feats[f"{v}_dist"] = dist.astype(np.float32)
            feats[f"{v}_ndist"] = (dist/meansize).astype(np.float32)
            feats[f"{v}_dx"] = dx.astype(np.float32)
            feats[f"{v}_dy"] = dy.astype(np.float32)
            feats[f"{v}_adx_n"] = (np.abs(dx)/meansize).astype(np.float32)
            feats[f"{v}_ady_n"] = (np.abs(dy)/meansize).astype(np.float32)
            feats[f"{v}_iou"] = iou().astype(np.float32)
            feats[f"{v}_sratio"] = np.log((s1+1)/(s2+1)).astype(np.float32)
            feats[f"{v}_smsum"] = (s1+s2).astype(np.float32)
            feats[f"{v}_pair_present"] = present
            # ground-oriented features on player1 only
            feats[f"{v}_cy"] = c1y.astype(np.float32)
            feats[f"{v}_top"] = (c1y - h1/2).astype(np.float32)
            feats[f"{v}_bottom"] = (c1y + h1/2).astype(np.float32)
            feats[f"{v}_height"] = h1.astype(np.float32)
            feats[f"{v}_width"] = w1.astype(np.float32)
            feats[f"{v}_area"] = a1.astype(np.float32)
            feats[f"{v}_cy_rank"] = m1.cy_rank.values.astype(np.float32)
            feats[f"{v}_cy_rel"] = (c1y - m1.cy_med.values).astype(np.float32)
            feats[f"{v}_size_rel"] = (s1/np.maximum(m1.size_med.values,1e-3)).astype(np.float32)
            feats[f"{v}_cy_to_max"] = (c1y - m1.cy_max.values).astype(np.float32)
            feats[f"{v}_bottom_to_max"] = ((c1y + h1/2) - m1.bottom_max.values).astype(np.float32)
            feats[f"{v}_p1_present"] = (~np.isnan(c1x)).astype(np.float32)
            for c in ["nn_hel","nn_hel_ratio","n_hel_near","max_iou","n_hel_ovl"]:
                feats[f"{v}_{c}_p1"] = m1[c].values.astype(np.float32)
                feats[f"{v}_{c}_p2"] = m2[c].values.astype(np.float32)
            del m1, m2; gc.collect()
            log(split, v, "done")

        # cross-view pair aggregates
        e = feats["Endzone_ndist"].values; s = feats["Sideline_ndist"].values
        feats["x_ndist_min"] = np.nanmin(np.vstack([e,s]), axis=0).astype(np.float32)
        feats["x_ndist_max"] = np.nanmax(np.vstack([e,s]), axis=0).astype(np.float32)
        feats["x_ndist_mean"] = np.nanmean(np.vstack([e,s]), axis=0).astype(np.float32)
        feats["x_nviews"] = ((~np.isnan(e)).astype(np.float32) + (~np.isnan(s)).astype(np.float32))
        feats["x_dist_mean"] = ((feats.Endzone_dist.values + feats.Sideline_dist.values)/2).astype(np.float32)
        feats["x_cy_rank_mean"] = np.nanmean(np.vstack([feats.Endzone_cy_rank.values, feats.Sideline_cy_rank.values]), axis=0).astype(np.float32)

        # temporal (pair) features
        feats["game_play"] = d.game_play.values
        feats["step"] = d.step.values
        feats["p1"] = d.nfl_player_id_1.values
        feats["p2"] = p2i
        feats = feats.sort_values(["game_play","p1","p2","step"]).reset_index(drop=True)
        feats["_gid"] = feats.groupby(["game_play","p1","p2"], sort=False).ngroup()
        gb = feats.groupby("_gid", sort=False)
        for col in ["x_ndist_min","Endzone_ndist","Sideline_ndist"]:
            feats[col+"_lag1"] = gb[col].shift(1).astype(np.float32)
            feats[col+"_lead1"] = gb[col].shift(-1).astype(np.float32)
            feats[col+"_d1"] = (gb[col].shift(-1) - gb[col].shift(1)).astype(np.float32)
        feats["x_ndist_rmin5"] = gb["x_ndist_min"].transform(
            lambda x: x.rolling(5, center=True, min_periods=1).min()).astype(np.float32)
        feats["x_ndist_rmean5"] = gb["x_ndist_min"].transform(
            lambda x: x.rolling(5, center=True, min_periods=1).mean()).astype(np.float32)
        gr = gb["x_cy_rank_mean"]
        feats["x_cy_rank_d1"] = (gr.shift(-1) - gr.shift(1)).astype(np.float32)
        feats["x_cy_rank_rmax5"] = gb["x_cy_rank_mean"].transform(
            lambda x: x.rolling(5, center=True, min_periods=1).max()).astype(np.float32)
        feats["x_cy_rank_rmean5"] = gb["x_cy_rank_mean"].transform(
            lambda x: x.rolling(5, center=True, min_periods=1).mean()).astype(np.float32)
        # per-view vertical / size dynamics (fall detection for ground contact)
        for col in ["Endzone_cy","Sideline_cy","Endzone_height","Sideline_height",
                    "Endzone_bottom_to_max","Sideline_bottom_to_max"]:
            if col not in feats.columns: continue
            feats[col+"_lag1"] = gb[col].shift(1).astype(np.float32)
            feats[col+"_lead1"] = gb[col].shift(-1).astype(np.float32)
            feats[col+"_d1"] = (gb[col].shift(-1) - gb[col].shift(1)).astype(np.float32)
        feats.drop(columns=["game_play","step","p1","p2","_gid"], inplace=True)

        out = f"{OUT}/{split}_helmet_features{SUFFIX}.parquet"
        feats.to_parquet(out, index=False)
        log(split, "saved", out, feats.shape)

if __name__ == "__main__":
    main()
