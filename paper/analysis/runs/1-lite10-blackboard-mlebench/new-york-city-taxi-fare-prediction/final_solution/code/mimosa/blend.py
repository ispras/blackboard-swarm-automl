import os, glob, sys, json
import numpy as np
from scipy.optimize import nnls

OUT = "/download_job_code/payload/sandbox/workspaces/mimosa_omicron"


def load_models(min_len=400000):
    models = {}
    for f in sorted(glob.glob(os.path.join(OUT, "holdout_pred_*.npy"))):
        tag = os.path.basename(f)[len("holdout_pred_"):-4]
        of = os.path.join(OUT, f"holdout_orig_{tag}.npy")
        if not os.path.exists(of):
            continue
        p = np.load(f); o = np.load(of)
        if len(p) < min_len or len(o) != len(p):
            continue
        d = {}
        for oi, pv in zip(o, p):
            d[int(oi)] = float(pv)
        models[tag] = d
    return models


def main():
    models = load_models()
    tags = list(models.keys())
    common = None
    for t in tags:
        s = set(models[t].keys())
        common = s if common is None else (common & s)
    common = np.array(sorted(common), dtype=np.int64)
    print("models", tags, "common holdout rows", len(common))
    yf = os.path.join(OUT, "holdout_true_xgb2_full.npy")
    of = np.load(os.path.join(OUT, "holdout_orig_xgb2_full.npy"))
    ymap = dict(zip([int(x) for x in of], np.load(yf)))
    y = np.array([ymap[int(i)] for i in common], dtype=np.float64)
    P = np.zeros((len(common), len(tags)))
    for j, t in enumerate(tags):
        P[:, j] = [models[t][int(i)] for i in common]
    for j, t in enumerate(tags):
        print(f"{t:22s} rmse {np.sqrt(np.mean((P[:,j]-y)**2)):.5f}")
    C = np.corrcoef(P.T)
    print("resid corr")
    R = P - y[:, None]
    print(np.round(np.corrcoef(R.T), 4))
    # full-fit nnls with intercept
    A = np.hstack([P, np.ones((len(y), 1))])
    w, _ = nnls(A, y)
    pf = A @ w
    print("nnls weights", dict(zip(tags + ["bias"], np.round(w, 4))), "in-sample", np.sqrt(np.mean((pf - y)**2)))
    # 5-fold CV on holdout by hash
    fold = (common // 100) % 5
    cv = np.zeros(len(y))
    for f in range(5):
        m = fold == f
        Atr = np.hstack([P[~m], np.ones((int((~m).sum()), 1))])
        wf, _ = nnls(Atr, y[~m])
        cv[m] = np.hstack([P[m], np.ones((int(m.sum()), 1))]) @ wf
    print("CV nnls blend", np.sqrt(np.mean((cv - y)**2)))
    # simple average of top models
    print("simple avg all", np.sqrt(np.mean((P.mean(1) - y)**2)))
    res = {"tags": tags, "weights": dict(zip(tags + ["bias"], [float(x) for x in w])),
           "cv_rmse": float(np.sqrt(np.mean((cv - y)**2))),
           "insample": float(np.sqrt(np.mean((pf - y)**2)))}
    json.dump(res, open(os.path.join(OUT, "blend_result.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
