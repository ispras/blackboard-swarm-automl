import os, sys, time, argparse, gc, json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import lightgbm as lgb
from sklearn.metrics import matthews_corrcoef, roc_auc_score

ROOT = "/download_job_code/payload/sandbox"
FEAT = f"{ROOT}/workspaces/canopus_pi/features/all_features.parquet"
NEWF = f"{ROOT}/workspaces/canopus_pi/features/new_feats_v2.parquet"
OUTD = f"{ROOT}/workspaces/canopus_pi/preds"
sys.path.insert(0, f"{ROOT}/workspaces/canopus_pi")
from features_list import pp_features_v2, ground_features_v2
os.makedirs(OUTD, exist_ok=True)

t0 = time.time()
def log(*a):
    print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)


def sweep(y, p, lo=0.5, hi=0.9995):
    order = np.argsort(-p, kind="stable")
    ys = y[order].astype(np.int64); ps = p[order]
    tp = np.cumsum(ys); fp = np.cumsum(1 - ys)
    P = int(ys.sum()); N = len(y) - P
    fn = P - tp; tn = N - fp
    den = np.sqrt((tp + fp).astype(np.float64) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = np.where(den > 0, (tp * tn - fp * fn) / np.maximum(den, 1e-12), 0.0)
    qlo, qhi = np.quantile(p, [lo, hi])
    mcc = np.where((ps >= qlo) & (ps <= qhi), mcc, -1)
    i = int(np.argmax(mcc))
    return float(mcc[i]), float(ps[i])


def load_meta():
    m = pd.read_parquet(FEAT, columns=["contact_id", "split", "contact", "is_ground", "step"])
    m["gp"] = m["contact_id"].str.rsplit("_", n=3).str[0]
    folds = pd.read_csv(f"{ROOT}/workspaces/alkaid_psi/cv_folds.csv")
    fmap = dict(zip(folds.game_play, folds.fold))
    m["fold"] = m["gp"].map(fmap).fillna(-1).astype(np.int16)
    return m


def read_X(spec, nrows):
    total = sum(len(c) for _, c in spec)
    X = np.empty((nrows, total), dtype=np.float32)
    j = 0
    for path, cols in spec:
        pf = pq.ParquetFile(path)
        for c in cols:
            tbl = pf.read(columns=[c])
            arr = tbl.column(c).to_numpy(zero_copy_only=False).astype(np.float32)
            X[:, j] = arr
            j += 1
            del tbl, arr
    gc.collect()
    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["pp", "ground"], required=True)
    ap.add_argument("--neg_frac", type=float, default=0.12)
    ap.add_argument("--lr", type=float, default=0.035)
    ap.add_argument("--leaves", type=int, default=63)
    ap.add_argument("--rounds", type=int, default=2500)
    ap.add_argument("--es", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="v2")
    ap.add_argument("--v2", type=int, default=1)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--extra", default="")
    ap.add_argument("--stack_oof", default="")
    ap.add_argument("--stack_test", default="")
    args = ap.parse_args()

    meta = load_meta()
    n = len(meta)
    if args.v2:
        feats = pp_features_v2() if args.task == "pp" else ground_features_v2()
    else:
        from features_list import pp_features, ground_features
        feats = pp_features() if args.task == "pp" else ground_features()

    base_avail = set(pq.ParquetFile(FEAT).schema.names)
    new_avail = set(pq.ParquetFile(NEWF).schema.names)
    fb = [c for c in feats if c in base_avail]
    fn = [c for c in feats if c in new_avail and c not in fb]
    feats = fb + fn
    log(args.task, "features", len(feats), "base", len(fb), "new", len(fn))

    X = read_X([(FEAT, fb), (NEWF, fn)], n)
    log("X", X.shape, X.nbytes / 1e9, "GB")

    if args.extra:
        ex = pd.read_parquet(args.extra)
        ex = ex.set_index("contact_id").reindex(meta.contact_id.values)
        ecols = [c for c in ex.columns]
        X = np.column_stack([X, ex[ecols].to_numpy(np.float32)]).astype(np.float32)
        feats = feats + ecols
        log("extra added", ecols)
    if args.stack_oof:
        o = pd.read_parquet(args.stack_oof)
        ocol = [c for c in o.columns if c != "contact_id"][-1] if "pred" not in o.columns else "pred"
        o = o.set_index("contact_id")[ocol].reindex(meta.contact_id.values).values.astype(np.float32)
        if args.stack_test:
            t = pd.read_parquet(args.stack_test)
            tcol = "pred" if "pred" in t.columns else [c for c in t.columns if c != "contact_id"][-1]
            t = t.set_index("contact_id")[tcol].reindex(meta.contact_id.values).values.astype(np.float32)
        else:
            t = np.full(n, np.nan, np.float32)
        st = np.where((meta.split.values == 0), o, t).astype(np.float32)
        X = np.column_stack([X, st]).astype(np.float32)
        feats = feats + ["stack_pred"]
        log("stack added cov", np.isfinite(st).mean())

    if args.task == "ground":
        gi = feats.index("sid_h_top_1") if "sid_h_top_1" in feats else None
        pres1 = (~np.isnan(X[:, gi])).astype(np.float32) if gi is not None else np.zeros(n, np.float32)
        gj = feats.index("end_h_top_1") if "end_h_top_1" in feats else None
        pres2 = (~np.isnan(X[:, gj])).astype(np.float32) if gj is not None else np.zeros(n, np.float32)
        X = np.column_stack([X, pres1, pres2]).astype(np.float32)
        feats = feats + ["sid_present", "end_present"]
        log("ground extra", X.shape)

    task_mask = (meta.is_ground == 0).values if args.task == "pp" else (meta.is_ground == 1).values
    G = np.where(task_mask)[0]
    y_all = meta.contact.fillna(0).values.astype(np.int8)
    folds_all = meta.fold.values
    log("task rows", len(G), "pos", y_all[G].mean())

    params = dict(objective="binary", metric="auc", learning_rate=args.lr,
                  num_leaves=args.leaves, min_child_samples=30,
                  feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1,
                  lambda_l2=1.0, max_bin=127, num_threads=args.threads, verbosity=-1,
                  seed=args.seed)

    oof_full = np.full(n, np.nan, dtype=np.float32)
    test_G = G[(folds_all[G] < 0)]
    test_pred = np.zeros(len(test_G), dtype=np.float32)
    models = []
    nfold = 5
    for k in range(nfold):
        trG = G[(folds_all[G] != k) & (folds_all[G] >= 0)]
        vaG = G[folds_all[G] == k]
        if args.task == "pp":
            rng = np.random.RandomState(100 + k)
            pos = trG[y_all[trG] == 1]
            neg = trG[y_all[trG] == 0]
            tr_s = np.concatenate([pos, neg[rng.rand(len(neg)) < args.neg_frac]])
            rng.shuffle(tr_s)
        else:
            tr_s = trG
        rng2 = np.random.RandomState(200 + k)
        vpos = vaG[y_all[vaG] == 1]
        vneg = vaG[y_all[vaG] == 0]
        va_es = np.concatenate([vpos, vneg[rng2.rand(len(vneg)) < 0.20]])
        dtr = lgb.Dataset(X[tr_s], label=y_all[tr_s], free_raw_data=True)
        dva = lgb.Dataset(X[va_es], label=y_all[va_es], reference=dtr, free_raw_data=True)
        model = lgb.train(params, dtr, num_boost_round=args.rounds,
                          valid_sets=[dva], valid_names=["es"],
                          callbacks=[lgb.early_stopping(args.es, verbose=False),
                                     lgb.log_evaluation(300)])
        models.append(model)
        oof_full[vaG] = model.predict(X[vaG], num_iteration=model.best_iteration).astype(np.float32)
        test_pred += model.predict(X[test_G], num_iteration=model.best_iteration).astype(np.float32) / nfold
        log(f"fold {k} best_iter {model.best_iteration}")
        del dtr, dva; gc.collect()

    trainG = G[folds_all[G] >= 0]
    y = y_all[trainG]; oof = oof_full[trainG]
    mcc, th = sweep(y, oof)
    auc = roc_auc_score(y, oof)
    log(f"{args.task} OOF MCC {mcc:.5f} @th {th:.6f} AUC {auc:.5f} MCC@0.5 {matthews_corrcoef(y, (oof>=0.5).astype(np.int8)):.5f}")

    m = meta.iloc[trainG][["contact_id", "gp", "step", "fold"]].copy()
    m["pred"] = oof
    m["contact"] = y
    m.to_parquet(f"{OUTD}/oof_{args.task}_{args.tag}.parquet", index=False)
    te_ids = meta.iloc[test_G]["contact_id"].reset_index(drop=True)
    pd.DataFrame({"contact_id": te_ids, "pred": test_pred}).to_parquet(
        f"{OUTD}/test_{args.task}_{args.tag}.parquet", index=False)
    json.dump({"mcc": float(mcc), "th": float(th), "auc": float(auc), "n": int(len(G)),
               "pos": float(y.mean()), "best_iters": [int(mo.best_iteration) for mo in models],
               "nfeat": len(feats)},
              open(f"{OUTD}/metrics_{args.task}_{args.tag}.json", "w"), indent=2)
    log("saved")


if __name__ == "__main__":
    main()
