import json, os
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.optimize import lsq_linear


def nnls_fit(X, y, alpha=1e-3):
    """Robust non-negative least squares with column scaling + ridge (never hits maxiter)."""
    s = X.std(0)
    s[s == 0] = 1.0
    Xn = X / s
    m = X.shape[1]
    A = np.vstack([Xn, np.sqrt(alpha) * np.eye(m)])
    b = np.concatenate([y, np.zeros(m)])
    res = lsq_linear(A, b, bounds=(0.0, np.inf), max_iter=500, tol=1e-8)
    return res.x / s
from sklearn.model_selection import KFold

T = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
SCORED_IDX = [0, 1, 3]


def load_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def true_array():
    d = load_jsonl("data/train.json")
    y = np.zeros((len(d), 107, 5), dtype=np.float32)
    for i, x in enumerate(d):
        for j, t in enumerate(T):
            y[i, :68, SCORED_IDX[j]] = x[t]
    sn = np.array([x.get("SN_filter", 1) == 1 for x in d])
    return y, sn


def pad(o):
    if o.ndim == 3 and o.shape[1] == 68:
        z = np.zeros((o.shape[0], 107, o.shape[2]), dtype=np.float32)
        z[:, :68] = o
        return z
    return o


def score(pred, Y, sel):
    a = pred[sel][:, :68][:, :, SCORED_IDX]
    b = Y[sel][:, :68][:, :, SCORED_IDX]
    return float(np.sqrt(((a - b) ** 2).mean((0, 1))).mean())


def discover():
    """Auto-discover every agent's full 5-fold OOF/test pair and validate them."""
    import glob
    reg = {}
    pairs = []
    # nested-aggregate convention: runs/<tag>/oof.npy + test.npy
    for o in glob.glob("workspaces/*/runs/*/oof.npy") + glob.glob("workspaces/*/preds/*/oof.npy"):
        pairs.append((o, os.path.join(os.path.dirname(o), "test.npy"),
                      os.path.basename(os.path.dirname(o))))
    # flat convention: <tag>_oof.npy + <tag>_test.npy
    for o in glob.glob("workspaces/*/runs/*_oof.npy") + glob.glob("workspaces/*/preds/*_oof.npy"):
        base = o[:-len("_oof.npy")]
        pairs.append((o, base + "_test.npy", os.path.basename(base)))
    seen = set()
    for o, t, tag in pairs:
        if tag in seen or not os.path.exists(t):
            continue
        low = tag.lower()
        if any(s in low for s in ("mask", "blend", "final", "fold", "oof")):
            continue
        oo = pad(np.nan_to_num(np.load(o)))
        tt = pad(np.nan_to_num(np.load(t)))
        if oo.shape != (2160, 107, 5) or tt.shape != (240, 107, 5):
            continue
        cov = int(np.any(oo[:, :68, :] != 0, axis=(1, 2)).sum())
        if cov < 1000:
            continue
        seen.add(tag)
        reg[tag] = (oo, tt)
    return reg


def greedy_blend(preds, Y, sel, n_iter=30, per_target=False):
    """preds: list of (N,107,5) arrays (already sel-sliced? no, full). returns weights (M,3) if per_target else (M,)"""
    M = len(preds)
    if per_target:
        W = np.zeros((M, 3))
        for j, c in enumerate(SCORED_IDX):
            cur = np.zeros_like(preds[0][..., c], dtype=np.float64)
            counts = np.zeros(M)
            best = 1e9
            for step in range(n_iter):
                bi, bs = None, best
                for i in range(M):
                    cand = (cur * step + preds[i][..., c]) / (step + 1)
                    s = np.sqrt(((cand[sel][:, :68] - Y[sel][:, :68, c]) ** 2).mean())
                    if s < bs - 1e-6:
                        bs, bi = s, i
                if bi is None:
                    break
                cur = (cur * step + preds[bi][..., c]) / (step + 1)
                counts[bi] += 1
                best = bs
            W[:, j] = counts / counts.sum()
        return W
    else:
        cur = np.zeros_like(preds[0], dtype=np.float64)
        counts = np.zeros(M)
        best = 1e9
        for step in range(n_iter):
            bi, bs = None, best
            for i in range(M):
                cand = (cur * step + preds[i]) / (step + 1)
                s = score(cand, Y, sel)
                if s < bs - 1e-6:
                    bs, bi = s, i
            if bi is None:
                break
            cur = (cur * step + preds[bi]) / (step + 1)
            counts[bi] += 1
            best = bs
        return (counts / counts.sum())[:, None] * np.ones((1, 3))


def apply_w(preds, W):
    out = np.zeros((preds[0].shape[0], 107, 5), dtype=np.float64)
    for i in range(len(preds)):
        row = np.zeros(5)
        for j, c in enumerate(SCORED_IDX):
            row[c] = W[i, j]
        out += preds[i] * row[None, None, :]
    return out


def fit_predict(method, train_preds, trainY, train_sel, target_preds):
    """fit weights on train subset, apply to target."""
    if method == "mean":
        W = (np.ones(len(train_preds)) / len(train_preds))[:, None] * np.ones((1, 3))
        return apply_w(target_preds, W)
    if method == "greedy":
        W = greedy_blend(train_preds, trainY, train_sel, per_target=False)
        return apply_w(target_preds, W)
    if method == "greedy_pt":
        W = greedy_blend(train_preds, trainY, train_sel, per_target=True)
        return apply_w(target_preds, W)
    if method == "nnls":
        W = np.zeros((len(train_preds), 3))
        # build per target design
        for j, c in enumerate(SCORED_IDX):
            Xf = np.stack([p[train_sel][:, :68, c].ravel() for p in train_preds], 1)
            yf = trainY[train_sel][:, :68, c].ravel()
            W[:, j] = nnls_fit(Xf, yf)
        return apply_w(target_preds, W)
    if method == "nnls_bag":
        rng = np.random.default_rng(0)
        W = np.zeros((len(train_preds), 3))
        B = 20
        for b in range(B):
            idx = rng.choice(train_sel, size=len(train_sel), replace=True)
            for j, c in enumerate(SCORED_IDX):
                Xf = np.stack([p[idx][:, :68, c].ravel() for p in train_preds], 1)
                yf = trainY[idx][:, :68, c].ravel()
                W[:, j] += nnls_fit(Xf, yf)
        W /= B
        return apply_w(target_preds, W)
    if method == "nnls_calib":
        W = np.zeros((len(train_preds), 3))
        ab = np.zeros((3, 2))
        for j, c in enumerate(SCORED_IDX):
            Xf = np.stack([p[train_sel][:, :68, c].ravel() for p in train_preds], 1)
            yf = trainY[train_sel][:, :68, c].ravel()
            w = nnls_fit(Xf, yf)
            W[:, j] = w
            btr = (Xf @ w).reshape(-1)
            cal = LinearRegression().fit(btr.reshape(-1, 1), yf)
            ab[j] = [cal.coef_[0], cal.intercept_]
        out = apply_w(target_preds, W)
        for j, c in enumerate(SCORED_IDX):
            out[..., c] = ab[j, 0] * out[..., c] + ab[j, 1]
        return out
    raise ValueError(method)


METHODS = ["mean", "greedy", "nnls", "nnls_calib"]


def cv_estimate(reg, Y, sn):
    names = list(reg)
    preds = [reg[k][0] for k in names]
    sel_all = np.where(sn)[0]
    kf = KFold(5, shuffle=True, random_state=777)
    res = {m: [] for m in METHODS}
    for tr_i, te_i in kf.split(sel_all):
        tr = sel_all[tr_i]
        te = sel_all[te_i]
        for m in METHODS:
            try:
                pr = fit_predict(m, preds, Y, tr, preds)
                res[m].append(score(pr, Y, te))
            except Exception as e:
                print(f"  [warn] method {m} failed: {e}", flush=True)
                res[m].append(np.nan)
    return {m: float(np.nanmean(v)) for m, v in res.items()}


if __name__ == "__main__":
    Y, sn = true_array()
    reg = discover()
    print("candidates:")
    for k, (o, t) in reg.items():
        print(f"  {k:12s} SN1={score(o, Y, sn):.4f}")
    if len(reg) < 1:
        raise SystemExit("no candidates")
    print("\nCV-estimated blend generalization (fit on 4/5 clean, eval 1/5):")
    est = cv_estimate(reg, Y, sn)
    for m, v in sorted(est.items(), key=lambda x: x[1]):
        print(f"  {m:10s} {v:.4f}")
    finite = {m: v for m, v in est.items() if np.isfinite(v)}
    best_method = min(finite, key=finite.get)
    print("best method:", best_method)

    names = list(reg)
    preds = [reg[k][0] for k in names]
    tpreds = [reg[k][1] for k in names]
    sel_all = np.where(sn)[0]
    try:
        blend_oof = fit_predict(best_method, preds, Y, sel_all, preds)
        blend_test = fit_predict(best_method, preds, Y, sel_all, tpreds)
    except Exception as e:
        print(f"[warn] full-fit {best_method} failed ({e}); falling back to greedy", flush=True)
        best_method = "greedy"
        blend_oof = fit_predict("greedy", preds, Y, sel_all, preds)
        blend_test = fit_predict("greedy", preds, Y, sel_all, tpreds)
    blend_oof[..., 2] = 0
    blend_oof[..., 4] = 0
    print(f"\nfull-fit {best_method} OOF SN1 = {score(blend_oof, Y, sn):.4f}")
    np.save("workspaces/mirfak_zeta/blend_test.npy", blend_test)
    np.save("workspaces/mirfak_zeta/blend_oof.npy", blend_oof)
    json.dump({"method": best_method, "names": names,
               "oof_sn1": score(blend_oof, Y, sn), "cv_est": est},
              open("workspaces/mirfak_zeta/blend_meta.json", "w"), indent=1)
