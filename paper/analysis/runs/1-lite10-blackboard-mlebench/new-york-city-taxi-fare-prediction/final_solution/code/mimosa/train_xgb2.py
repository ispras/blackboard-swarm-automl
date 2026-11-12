import os, json, time, argparse
import numpy as np
import pandas as pd
import xgboost as xgb
from common import FEATS_V2, TE_COLS, parts_v2, load_te

OUT = "/download_job_code/payload/sandbox/workspaces/mimosa_omicron"
ALLF = FEATS_V2 + TE_COLS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frac", type=float, default=1.0)
    ap.add_argument("--rounds", type=int, default=3000)
    ap.add_argument("--eta", type=float, default=0.04)
    ap.add_argument("--depth", type=int, default=10)
    ap.add_argument("--min_child_weight", type=float, default=300)
    ap.add_argument("--subsample", type=float, default=0.8)
    ap.add_argument("--colsample", type=float, default=0.8)
    ap.add_argument("--l2", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=0.0)
    ap.add_argument("--max_bin", type=int, default=128)
    ap.add_argument("--early", type=int, default=60)
    ap.add_argument("--tag", type=str, default="xgb2")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--use_te2", action="store_true")
    ap.add_argument("--use_te3", action="store_true")
    ap.add_argument("--log_target", action="store_true")
    ap.add_argument("--filter_noise", action="store_true")
    ap.add_argument("--rot29", action="store_true")
    ap.add_argument("--grow", type=str, default="depthwise")
    ap.add_argument("--max_leaves", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    te = load_te()
    pl = parts_v2()
    dfs = []
    off = 0
    for p in pl:
        df = pd.read_parquet(p, columns=["orig_idx", "fare_amount"] + FEATS_V2)
        df["pos"] = np.arange(off, off + len(df))
        off += len(df)
        if args.frac < 1.0:
            df = df.sample(frac=args.frac, random_state=args.seed)
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    del dfs
    keep = df["pos"].to_numpy(np.int64)
    for c in TE_COLS:
        df[c] = te[c][keep]
    allf = list(ALLF)
    if args.use_te2:
        te2 = np.load(os.path.join(OUT, "te2_maps.npz"))
        df["te_ry_oof"] = te2["te_ry_oof"][keep]
        allf = allf + ["te_ry_oof"]
    if args.use_te3:
        te3 = np.load(os.path.join(OUT, "te3_maps.npz"))
        for c in ["te_pcf_oof", "te_dcf_oof", "te_routef_oof"]:
            df[c] = te3[c][keep]
            allf.append(c)
    if args.rot29:
        import math
        c29 = math.cos(math.radians(29)); s29 = math.sin(math.radians(29))
        plat = df["pickup_latitude"].to_numpy(np.float32); plon = df["pickup_longitude"].to_numpy(np.float32)
        dlat = df["dropoff_latitude"].to_numpy(np.float32); dlon = df["dropoff_longitude"].to_numpy(np.float32)
        df["ru29_p"] = plat * c29 + plon * s29; df["rv29_p"] = -plat * s29 + plon * c29
        df["ru29_d"] = dlat * c29 + dlon * s29; df["rv29_d"] = -dlat * s29 + dlon * c29
        df["ru29_delta"] = df["ru29_d"] - df["ru29_p"]; df["rv29_delta"] = df["rv29_d"] - df["rv29_p"]
        allf += ["ru29_p", "rv29_p", "ru29_d", "rv29_d", "ru29_delta", "rv29_delta"]
    print(f"loaded {len(df)} {df.shape} {time.time()-t0:.0f}s", flush=True)

    hold = (df["orig_idx"].to_numpy() % 100 == 0)
    hold_orig = df.loc[hold, "orig_idx"].to_numpy(np.int64)
    y = df["fare_amount"].to_numpy(np.float32)
    if args.filter_noise:
        dist = df["dist"].to_numpy(np.float32)
        keep = ~(((y > 80) & (dist < 3)) | ((y > 150) & (dist < 8)) | ((y < 5) & (dist > 20)))
        print("filter_noise: removing", int((~keep & ~hold).sum()), "of", int((~hold).sum()), "train rows", flush=True)
    else:
        keep = np.ones(len(df), dtype=bool)
    X = df[allf].to_numpy(np.float32)
    del df
    trm = (~hold) & keep
    Xtr, ytr = X[trm], y[trm]
    Xva, yva = X[hold], y[hold]
    del X, y
    print(f"train {Xtr.shape} val {Xva.shape}", flush=True)

    med = {c: float(np.median(Xtr[:, i])) for i, c in enumerate(allf)}
    with open(os.path.join(OUT, f"medians_{args.tag}.json"), "w") as f:
        json.dump(med, f)

    if args.log_target:
        ytr = np.log1p(ytr); yva = np.log1p(yva)
    dtr = xgb.DMatrix(Xtr, label=ytr, feature_names=allf, nthread=args.threads)
    dva = xgb.DMatrix(Xva, label=yva, feature_names=allf, nthread=args.threads)
    del Xtr, ytr, Xva, yva
    params = {
        "device": "cuda", "tree_method": "hist", "max_bin": args.max_bin,
        "objective": "reg:squarederror", "eval_metric": "rmse",
        "eta": args.eta, "max_depth": args.depth, "grow_policy": args.grow,
        "min_child_weight": args.min_child_weight,
        "subsample": args.subsample, "colsample_bytree": args.colsample,
        "reg_lambda": args.l2, "reg_alpha": args.alpha,
        "seed": args.seed, "nthread": args.threads, "verbosity": 1,
    }
    if args.max_leaves > 0:
        params["max_leaves"] = args.max_leaves
    bst = xgb.train(params, dtr, num_boost_round=args.rounds, evals=[(dva, "val")],
                    early_stopping_rounds=args.early, verbose_eval=50)
    best = bst.best_iteration
    pv = bst.predict(dva, iteration_range=(0, best + 1))
    if args.log_target:
        pv = np.expm1(pv)
        ytrue = np.expm1(np.asarray(dva.get_label()))
    else:
        ytrue = np.asarray(dva.get_label())
    rmse = float(np.sqrt(np.mean((pv - ytrue) ** 2)))
    print(f"BEST_ITER={best} HOLDOUT_RMSE={rmse:.5f}", flush=True)
    bst.save_model(os.path.join(OUT, f"model_{args.tag}.json"))
    np.save(os.path.join(OUT, f"holdout_pred_{args.tag}.npy"), pv.astype(np.float32))
    np.save(os.path.join(OUT, f"holdout_orig_{args.tag}.npy"), hold_orig)
    np.save(os.path.join(OUT, f"holdout_true_{args.tag}.npy"), ytrue.astype(np.float32))
    with open(os.path.join(OUT, f"result_{args.tag}.json"), "w") as f:
        json.dump({"tag": args.tag, "rmse": rmse, "best_iter": int(best),
                   "n_train": int((~hold).sum()), "n_val": int(hold.sum()),
                   "params": params, "args": vars(args), "time_s": time.time() - t0}, f, indent=2)
    print("done", time.time() - t0, flush=True)


if __name__ == "__main__":
    main()
