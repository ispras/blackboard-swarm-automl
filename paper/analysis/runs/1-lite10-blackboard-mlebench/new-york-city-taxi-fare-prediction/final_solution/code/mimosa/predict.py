import os, sys, json, time, argparse
import numpy as np
import pandas as pd
import xgboost as xgb
from common import FEATS_V2, TE_COLS, load_te, NONSPATIAL
from test_features import build_test, add_counts_te, add_te3, add_rot29
from build_corrupt import add_corrupt_feats

OUT = "/download_job_code/payload/sandbox/workspaces/mimosa_omicron"
ROOT = "/download_job_code/payload/sandbox"
ALLF = FEATS_V2 + TE_COLS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model_xgb2_full.json")
    ap.add_argument("--fallback", default="model_corrupt_fb.json")
    ap.add_argument("--out", default="submission_xgb2_full.csv")
    ap.add_argument("--clip", type=float, nargs=2, default=(2.5, 500.0))
    ap.add_argument("--tag", default="")
    ap.add_argument("--log_target", action="store_true")
    ap.add_argument("--te3", action="store_true")
    ap.add_argument("--rot29", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    te = load_te()
    df = build_test()
    df = add_counts_te(df, te)
    allf = list(ALLF)
    if args.te3:
        te3 = np.load(os.path.join(OUT, "te3_maps.npz"))
        df = add_te3(df, te3)
        allf = allf + ["te_pcf_oof", "te_dcf_oof", "te_routef_oof"]
    if args.rot29:
        df = add_rot29(df)
        allf += ["ru29_p", "rv29_p", "ru29_d", "rv29_d", "ru29_delta", "rv29_delta"]
    corrupt = df["corrupt"].to_numpy().astype(bool)
    print("test rows", len(df), "corrupt", corrupt.sum(), flush=True)

    bst = xgb.Booster()
    bst.load_model(os.path.join(OUT, args.model))
    bst.set_param({"device": "cpu"})
    X = df[allf].to_numpy(np.float32)
    d = xgb.DMatrix(X, feature_names=allf)
    pred = bst.predict(d)
    if args.log_target:
        pred = np.expm1(pred)

    fb = xgb.Booster()
    fb.load_model(os.path.join(OUT, args.fallback))
    fb.set_param({"device": "cpu"})
    fb_feats = json.load(open(os.path.join(OUT, "result_corrupt_fb.json")))["feats"]
    cdf = add_corrupt_feats(df)
    dfb = xgb.DMatrix(cdf[fb_feats].to_numpy(np.float32), feature_names=fb_feats)
    pf = fb.predict(dfb)
    pred = np.where(corrupt, pf, pred)
    pred = np.clip(pred, args.clip[0], args.clip[1]).astype(np.float64)

    sub = pd.DataFrame({"key": pd.read_csv(os.path.join(ROOT, "data", "test.csv"))["key"],
                        "fare_amount": pred})
    print(sub["fare_amount"].describe().to_dict(), flush=True)
    sub.to_csv(os.path.join(OUT, args.out), index=False)
    if args.tag:
        np.save(os.path.join(OUT, f"test_pred_{args.tag}.npy"), pred.astype(np.float32))
    print("saved", args.out, "corrupt_fallback_mean", float(pf[corrupt].mean()) if corrupt.any() else None,
          "time", time.time() - t0, flush=True)


if __name__ == "__main__":
    main()
