import os, sys, time, argparse, gc, json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import xgboost as xgb
from sklearn.metrics import roc_auc_score, matthews_corrcoef

ROOT = "/download_job_code/payload/sandbox"
FEAT = f"{ROOT}/workspaces/canopus_pi/features/all_features.parquet"
NEWF = f"{ROOT}/workspaces/canopus_pi/features/new_feats_v2.parquet"
OUTD = f"{ROOT}/workspaces/canopus_pi/preds"
sys.path.insert(0, f"{ROOT}/workspaces/canopus_pi")
from features_list import pp_features_v2, ground_features_v2
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)


def sweep(y, p, lo=0.4, hi=0.9998):
    o = np.argsort(-p, kind="stable"); ys = y[o].astype(np.int64); ps = p[o]
    tp = np.cumsum(ys); fp = np.cumsum(1 - ys); P = int(ys.sum()); N = len(y) - P
    fn = P - tp; tn = N - fp
    den = np.sqrt((tp + fp).astype(np.float64) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = np.where(den > 0, (tp * tn - fp * fn) / np.maximum(den, 1e-12), 0.0)
    qlo, qhi = np.quantile(p, [lo, hi]); mcc = np.where((ps >= qlo) & (ps <= qhi), mcc, -1)
    i = int(np.argmax(mcc)); return float(mcc[i]), float(ps[i])


def read_X(spec, nrows):
    total = sum(len(c) for _, c in spec)
    X = np.empty((nrows, total), dtype=np.float32); j = 0
    for path, cols in spec:
        pf = pq.ParquetFile(path)
        for c in cols:
            X[:, j] = pf.read(columns=[c]).column(c).to_numpy(zero_copy_only=False).astype(np.float32); j += 1
    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["pp", "ground"], required=True)
    ap.add_argument("--neg_frac", type=float, default=0.12)
    ap.add_argument("--rounds", type=int, default=1500)
    ap.add_argument("--es", type=int, default=80)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="xgb1")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    meta = pd.read_parquet(FEAT, columns=["contact_id", "split", "contact", "is_ground", "step"])
    meta["gp"] = meta.contact_id.str.rsplit("_", n=3).str[0]
    folds = pd.read_csv(f"{ROOT}/workspaces/alkaid_psi/cv_folds.csv")
    fmap = dict(zip(folds.game_play, folds.fold))
    meta["fold"] = meta["gp"].map(fmap).fillna(-1).astype(np.int16)
    n = len(meta)
    feats = pp_features_v2() if args.task == "pp" else ground_features_v2()
    b = set(pq.ParquetFile(FEAT).schema.names); nn = set(pq.ParquetFile(NEWF).schema.names)
    fb = [c for c in feats if c in b]; fn = [c for c in feats if c in nn and c not in fb]
    feats = fb + fn
    log(args.task, "feats", len(feats))
    X = read_X([(FEAT, fb), (NEWF, fn)], n)
    log("X", X.shape, X.nbytes / 1e9)

    task_mask = (meta.is_ground == 0).values if args.task == "pp" else (meta.is_ground == 1).values
    G = np.where(task_mask)[0]
    y_all = meta.contact.fillna(0).values.astype(np.int8)
    folds_all = meta.fold.values
    params = dict(objective="binary:logistic", eval_metric="auc", tree_method="hist",
                  max_depth=args.depth, eta=args.lr, subsample=0.8, colsample_bytree=0.7,
                  min_child_weight=5, reg_lambda=1.0, nthread=args.threads, seed=args.seed,
                  max_bin=128)
    oof_full = np.full(n, np.nan, dtype=np.float32)
    testG = G[folds_all[G] < 0]
    test_pred = np.zeros(len(testG), dtype=np.float32)
    dopred = G[folds_all[G] >= 0]

    for k in range(5):
        trG = G[(folds_all[G] != k) & (folds_all[G] >= 0)]
        vaG = G[folds_all[G] == k]
        if args.task == "pp":
            rng = np.random.RandomState(100 + k)
            pos = trG[y_all[trG] == 1]; neg = trG[y_all[trG] == 0]
            tr_s = np.concatenate([pos, neg[rng.rand(len(neg)) < args.neg_frac]]); rng.shuffle(tr_s)
        else:
            tr_s = trG
        rng2 = np.random.RandomState(200 + k)
        vpos = vaG[y_all[vaG] == 1]; vneg = vaG[y_all[vaG] == 0]
        va_es = np.concatenate([vpos, vneg[rng2.rand(len(vneg)) < 0.2]])
        dtr = xgb.DMatrix(X[tr_s], label=y_all[tr_s])
        dva = xgb.DMatrix(X[va_es], label=y_all[va_es])
        bst = xgb.train(params, dtr, num_boost_round=args.rounds, evals=[(dva, "es")],
                        early_stopping_rounds=args.es, verbose_eval=200)
        oof_full[vaG] = bst.predict(xgb.DMatrix(X[vaG]), iteration_range=(0, bst.best_iteration + 1)).astype(np.float32)
        test_pred += bst.predict(xgb.DMatrix(X[testG]), iteration_range=(0, bst.best_iteration + 1)).astype(np.float32) / 5
        log(f"fold {k} best {bst.best_iteration}")
        del dtr, dva, bst; gc.collect()

    y = y_all[dopred]; oof = oof_full[dopred]
    mcc, th = sweep(y, oof); auc = roc_auc_score(y, oof)
    log(f"{args.task} XGB OOF MCC {mcc:.5f} th {th:.4f} AUC {auc:.5f}")
    m = meta.iloc[dopred][["contact_id", "gp", "step", "fold"]].copy(); m["pred"] = oof; m["contact"] = y
    m.to_parquet(f"{OUTD}/oof_{args.task}_{args.tag}.parquet", index=False)
    pd.DataFrame({"contact_id": meta.iloc[testG].contact_id.values, "pred": test_pred}).to_parquet(
        f"{OUTD}/test_{args.task}_{args.tag}.parquet", index=False)
    json.dump({"mcc": float(mcc), "th": float(th), "auc": float(auc)}, open(f"{OUTD}/metrics_{args.task}_{args.tag}.json", "w"), indent=2)
    log("saved")


if __name__ == "__main__":
    main()
