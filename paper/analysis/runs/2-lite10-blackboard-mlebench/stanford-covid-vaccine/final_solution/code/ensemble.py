import argparse
import glob
import json
import os
import numpy as np
from scipy.optimize import nnls

import common as C

SCORED_IDX = [0, 1, 3]


def norm_oof(a):
    a = np.asarray(a, dtype=np.float64)
    if a.shape[1] > 68:
        a = a[:, :68, :]
    return a


def norm_test(a):
    return np.asarray(a, dtype=np.float64)


def fit_weights(P, y, intercept=False):
    """P: (M, N) preds, y: (N,) truth. NNLS weights, optional intercept via aug col.
    Ridge-augmented NNLS for numerical stability with collinear models."""
    if intercept:
        A = np.vstack([P, np.ones((1, P.shape[1]))])
    else:
        A = P
    M = A.shape[0]
    lam = 1e-1
    Aa = np.vstack([A.T, np.sqrt(lam) * np.eye(M)])
    ya = np.concatenate([y, np.zeros(M)])
    w, _ = nnls(Aa, ya, maxiter=50000)
    return w


def blend_cv(Plist, Y, sn, folds, intercept=False):
    """Return per-column weights fit on all sn rows, plus CV estimate of blend."""
    M = len(Plist)
    W = np.zeros((M + (1 if intercept else 0), 5))
    cv_pred = np.zeros_like(Y)
    for j in SCORED_IDX:
        Pj = np.stack([p[:, :, j] for p in Plist], 0)  # (M, N, 68)
        for k, (tidx, vidx) in enumerate(folds):
            tmask = sn[tidx]
            if tmask.sum() < 10:
                continue
            Pt = Pj[:, tidx[tmask], :].reshape(M, -1)
            yt = Y[tidx[tmask], :, j].reshape(-1)
            w = fit_weights(Pt, yt, intercept)
            Pv = Pj[:, vidx, :].reshape(M, max(1, (vidx.size) * 68))
            if intercept:
                cv_pred[vidx, :, j] = (w[:-1] @ Pv + w[-1]).reshape(vidx.size, 68)
            else:
                cv_pred[vidx, :, j] = (w @ Pv).reshape(vidx.size, 68)
        # final weights on all sn rows
        Pm = Pj[:, sn, :].reshape(M, -1)
        ym = Y[sn, :, j].reshape(-1)
        W[:, j] = fit_weights(Pm, ym, intercept)
    return W, cv_pred


def apply_weights(Plist, W, intercept=False, ref=None):
    """Return (N,L,5) blended test preds using per-column weights."""
    N, L = Plist[0].shape[0], Plist[0].shape[1]
    out = np.zeros((N, L, 5))
    for j in range(5):
        stack = np.stack([p[:, :, j] for p in Plist], 0)  # (M,N,L)
        M = stack.shape[0]
        Pj = stack.reshape(M, -1)
        if j in SCORED_IDX:
            if intercept:
                pred = W[:-1, j] @ Pj + W[-1, j]
            else:
                pred = W[:, j] @ Pj
            out[:, :, j] = pred.reshape(N, L)
        else:
            out[:, :, j] = (ref[:, :, j] if ref is not None else stack.mean(0))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None, help="name=oofpath=testpath")
    ap.add_argument("--registry", default="", help="json list of {name,oof,test,enabled}")
    ap.add_argument("--intercept", action="store_true")
    ap.add_argument("--clip", type=float, default=None)
    ap.add_argument("--affine", action="store_true")
    ap.add_argument("--out", default="ensemble_submission.csv")
    ap.add_argument("--save", default="")
    ap.add_argument("--save-cv", default="")
    ap.add_argument("--unscored-ref", default="", help="model name to supply unscored columns")
    ap.add_argument("--knn-gate", action="store_true")
    ap.add_argument("--knn-thr", type=float, default=0.85)
    ap.add_argument("--knn-wmax", type=float, default=0.25)
    args = ap.parse_args()

    tr = C.load_train()
    Y = C.make_targets(tr)
    sn = np.array([r["SN_filter"] for r in tr], dtype=bool)
    folds = C.kfold_indices(len(tr), 5, 42)

    if args.registry:
        reg = json.load(open(args.registry))
        args.models = [f"{m['name']}={m['oof']}={m['test']}" for m in reg if m.get("enabled", True)]
    if not args.models:
        raise SystemExit("no models")
    names, oofs, tests = [], [], []
    for spec in args.models:
        name, op, tp = spec.split("=")
        o = np.load(op)
        t = np.load(tp)
        o = norm_oof(o)
        if o.shape[0] != len(tr):
            print("SKIP", name, o.shape)
            continue
        names.append(name)
        oofs.append(o)
        tests.append(norm_test(t))
        m, r = C.mcrmse(Y[sn], o[sn])
        print(f"{name:28s} sn1={m:.5f} cols={['%.4f' % x for x in r]}")
        if args.clip is not None:
            o2 = np.clip(o, args.clip, None)
            m2, _ = C.mcrmse(Y[sn], o2[sn])
            print(f"{'  +clip':28s} sn1={m2:.5f}")

    # simple mean
    mean_oof = np.mean(oofs, 0)
    mean_test = np.mean(tests, 0)
    m, r = C.mcrmse(Y[sn], mean_oof[sn])
    print(f"\nSIMPLE MEAN sn1={m:.5f} cols={['%.4f' % x for x in r]}")

    W, cv_pred = blend_cv(oofs, Y, sn, folds, args.intercept)
    m, r = C.mcrmse(Y[sn], cv_pred[sn])
    print(f"NNLS blend CV sn1={m:.5f} cols={['%.4f' % x for x in r]}")
    for j in SCORED_IDX:
        print(f"  col{j} weights:", {names[i]: round(float(W[i, j]), 4) for i in range(len(names))},
              ("b=%.4f" % W[-1, j]) if args.intercept else "")

    ref = None
    if args.unscored_ref:
        if args.unscored_ref in names:
            ref = tests[names.index(args.unscored_ref)]
            print("unscored cols from", args.unscored_ref)
        else:
            print("WARN unscored-ref not found:", args.unscored_ref)
    test = apply_weights(tests, W, args.intercept, ref=ref)
    if args.clip is not None:
        test = np.clip(test, args.clip, None)
    if args.affine:
        for j in SCORED_IDX:
            x = cv_pred[sn, :, j].ravel()
            y = Y[sn, :, j].ravel()
            A = np.vstack([x, np.ones_like(x)]).T
            a, b = np.linalg.lstsq(A, y, rcond=None)[0]
            test[..., j] = a * test[..., j] + b
            print(f"  affine col{j}: a={a:.4f} b={b:.4f}")
    if args.knn_gate:
        knn_oof = np.load("knn_best_oof.npy").astype(np.float64)
        knn_test = np.load("knn_best_test.npy").astype(np.float64)
        S_tr = np.load("knn_best_S_tr.npy")
        S_te = np.load("knn_best_S_te.npy")
        id_tr = S_tr.max(1)
        id_te = S_te.max(1)
        wtr = np.clip((id_tr - args.knn_thr) / (1 - args.knn_thr), 0, 1)[:, None, None] * args.knn_wmax
        wte = np.clip((id_te - args.knn_thr) / (1 - args.knn_thr), 0, 1)[:, None, None] * args.knn_wmax
        cv_g = (1 - wtr) * cv_pred + wtr * knn_oof
        mg, rg = C.mcrmse(Y[sn], cv_g[sn])
        print(f"KNN-GATED CV sn1={mg:.5f} cols={['%.4f' % x for x in rg]}")
        if args.clip is not None:
            cv_g = np.clip(cv_g, args.clip, None)
            mg2, rg2 = C.mcrmse(Y[sn], cv_g[sn])
            print(f"KNN-GATED+CLIP CV sn1={mg2:.5f} cols={['%.4f' % x for x in rg2]}")
        test[:, :68, :] = (1 - wte) * test[:, :68, :] + wte * knn_test[:, :68, :]
        if args.clip is not None:
            test = np.clip(test, args.clip, None)
        print(f"  gated test rows: thr={args.knn_thr} wmax={args.knn_wmax}; "
              f"id>=thr test={int((id_te >= args.knn_thr).sum())}")
    if args.save:
        np.save(args.save, test.astype(np.float32))
    if args.save_cv:
        np.save(args.save_cv, cv_pred.astype(np.float32))
    import make_submission
    make_submission.write_submission(test, args.out)


if __name__ == "__main__":
    main()
