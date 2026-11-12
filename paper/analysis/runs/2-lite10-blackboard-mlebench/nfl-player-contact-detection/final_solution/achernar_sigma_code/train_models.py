import pandas as pd, numpy as np, gc, time, argparse, os
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

ROOT = "/download_job_code/payload/sandbox"
W = f"{ROOT}/workspaces/achernar_sigma"
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)

DROP = {"contact_id","game_play","contact","target","nfl_player_id_2_x","nfl_player_id_2_y","is_ground","fold"}
GROUND_HELMET_SUFFIX = ("_cy","_top","_bottom","_height","_width","_area","_cy_rank","_cy_rel",
                        "_size_rel","_cy_to_max","_bottom_to_max","_p1_present",
                        "_nn_hel_p1","_nn_hel_ratio_p1","_n_hel_near_p1","_max_iou_p1","_n_hel_ovl_p1")

def load_split(split, kind):
    tr = pd.read_parquet(f"{W}/{split}_tracking_features.parquet")
    is_g = tr.contact_id.str.endswith("_G").values
    keep = is_g if kind == "ground" else ~is_g
    tr = tr.loc[keep].reset_index(drop=True)
    if split == "train":
        tr["target"] = tr["contact"].astype(np.int8)
        tr = tr.drop(columns=["contact"])
    else:
        tr["target"] = np.nan
    tr = tr.drop(columns=[c for c in ["nfl_player_id_2_x","nfl_player_id_2_y"] if c in tr.columns])
    he = pd.read_parquet(f"{W}/{split}_helmet_features{os.environ.get('HELMET_SUFFIX','')}.parquet")
    # align helmet rows to filtered set via contact_id (memory-light: merge then subset)
    he = he.set_index("contact_id")
    he = he.reindex(tr.contact_id.values).reset_index()
    d = pd.concat([tr.reset_index(drop=True), he.drop(columns=["contact_id"])], axis=1)
    del tr, he; gc.collect()
    return d

def features_for(d, kind):
    num = [c for c in d.columns if c not in DROP and d[c].dtype != object]
    if kind == "ground":
        f = [c for c in num if c.endswith("_1")]
        f += [c for c in num if c.startswith(("Endzone_","Sideline_")) and c.endswith(GROUND_HELMET_SUFFIX)]
        f += [c for c in num if c.startswith("x_cy_rank")]
        f = sorted(set(f)) + ["step"]
    else:
        f = [c for c in num if c not in ("step",)]
        f += ["step"]
    return [c for c in f if c in d.columns]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=["pair","ground"])
    ap.add_argument("--folds", default="all")
    ap.add_argument("--lr", type=float, default=0.04)
    ap.add_argument("--leaves", type=int, default=127)
    ap.add_argument("--rounds", type=int, default=4000)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--negfrac", type=float, default=1.0)
    ap.add_argument("--extra_trees", action="store_true")
    args = ap.parse_args()
    folds = pd.read_csv(f"{ROOT}/workspaces/castor_alpha/folds.csv")
    fold_ids = sorted(folds.fold.unique()) if args.folds == "all" else [int(x) for x in args.folds.split(",")]

    log("loading train", args.kind)
    d = load_split("train", args.kind)
    log("loading test", args.kind)
    te = load_split("test", args.kind)
    d = d.merge(folds, on="game_play", how="left")
    assert d.fold.notna().all(), "fold join failed"
    d["fold"] = d.fold.astype(np.int8)
    log("loaded", d.shape, te.shape, "pos", int(d.target.sum()))

    feats = features_for(d, args.kind)
    cat = [c for c in ["position_code_1","position_code_2"] if c in feats]
    for c in cat:
        d[c] = d[c].fillna(-1).astype("int32"); te[c] = te[c].fillna(-1).astype("int32")
    log("n features", len(feats), "cat", cat)

    y = d.target.values.astype(np.int8)
    Xdf = d[feats]
    X = Xdf.values.astype(np.float32)
    del Xdf; gc.collect()
    Xte = te[feats].values.astype(np.float32)
    te_ids = te.contact_id.values
    ids = d.contact_id.values; gps = d.game_play.values; foldv = d.fold.values
    del d, te; gc.collect()
    log("matrices", X.shape, Xte.shape)

    oof = np.full(len(X), np.nan, dtype=np.float32)
    testp = np.zeros(len(Xte), dtype=np.float64)
    params = dict(objective="binary", metric="auc", learning_rate=args.lr,
                  num_leaves=args.leaves, min_data_in_leaf=(200 if args.kind=="pair" else 60),
                  feature_fraction=(0.65 if args.kind=="pair" else 0.8), bagging_fraction=0.85, bagging_freq=1,
                  lambda_l2=(2.0 if args.kind=="pair" else 1.0), max_bin=127, num_threads=args.threads,
                  verbose=-1, seed=42, extra_trees=bool(args.extra_trees))
    for f in fold_ids:
        va = foldv == f
        trn = ~va
        rng = np.random.RandomState(100+f)
        if args.negfrac < 1.0:
            pos_idx = np.where(trn & (y==1))[0]
            neg_idx = np.where(trn & (y==0))[0]
            neg_keep = rng.choice(neg_idx, int(len(neg_idx)*args.negfrac), replace=False)
            tr_idx = np.concatenate([pos_idx, neg_keep]); rng.shuffle(tr_idx)
        else:
            tr_idx = np.where(trn)[0]
        log(args.kind, "fold", f, "train", len(tr_idx), "val", int(va.sum()))
        dtrain = lgb.Dataset(X[tr_idx], label=y[tr_idx], feature_name=feats, categorical_feature=cat, free_raw_data=True)
        dval = lgb.Dataset(X[va], label=y[va], reference=dtrain, free_raw_data=True)
        bst = lgb.train(params, dtrain, num_boost_round=args.rounds, valid_sets=[dval],
                        callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(200)])
        oof[va] = bst.predict(X[va], num_iteration=bst.best_iteration).astype(np.float32)
        testp += bst.predict(Xte, num_iteration=bst.best_iteration) / len(fold_ids)
        log(args.kind, "fold", f, "iter", bst.best_iteration, "auc", round(roc_auc_score(y[va], oof[va]),5))
        del dtrain, dval, bst; gc.collect()
    fin = np.isfinite(oof)
    log(args.kind, "OOF AUC", roc_auc_score(y[fin], oof[fin]), "n", int(fin.sum()))
    pd.DataFrame({"contact_id": ids, "y": y, "pred": oof, "fold": foldv, "game_play": gps}
                 ).to_parquet(f"{W}/oof_{args.kind}_{args.tag}.parquet", index=False)
    pd.DataFrame({"contact_id": te_ids, "pred": testp.astype(np.float32)}
                 ).to_parquet(f"{W}/testpred_{args.kind}_{args.tag}.parquet", index=False)
    log("DONE")

if __name__ == "__main__":
    main()
