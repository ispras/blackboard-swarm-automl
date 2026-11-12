import json, os
import numpy as np
import lightgbm as lgb

T = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
SI = [0, 1, 3]
SEQ = {c: i for i, c in enumerate("ACGU")}
ST = {c: i for i, c in enumerate(".()")}
LP = {c: i for i, c in enumerate("SMIBHEX")}


def pair_dist(struct):
    stack, pd = [], np.zeros(len(struct), np.float32)
    for i, c in enumerate(struct):
        if c == "(":
            stack.append(i)
        elif c == ")":
            if stack:
                j = stack.pop()
                d = min(i - j, 30)
                pd[i] = pd[j] = d / 30.0
    return pd


def feats(x, w=5):
    seq, st, lp = x["sequence"], x["structure"], x["predicted_loop_type"]
    L = len(seq)
    pd = pair_dist(st)
    rows = []
    for p in range(L):
        f = []
        for o in range(-w, w + 1):
            q = p + o
            for vocab, s in ((SEQ, seq), (ST, st), (LP, lp)):
                v = np.zeros(len(vocab), np.float32)
                if 0 <= q < L:
                    v[vocab.get(s[q], 0)] = 1
                f.append(v)
        f.append([pd[p]])
        f.append([1.0 if st[p] in "()" else 0.0])
        # local stem fraction +-8
        lo, hi = max(0, p - 8), min(L, p + 9)
        f.append([np.mean([1.0 if st[k] in "()" else 0.0 for k in range(lo, hi)])])
        f.append([p / 107.0])
        f.append([x["sequence"].count("G") / 107.0, x["sequence"].count("C") / 107.0])
        rows.append(np.concatenate(f))
    return np.array(rows, np.float32)


def build(d):
    X = np.stack([feats(x) for x in d])  # (N,107,F)
    return X


if __name__ == "__main__":
    d = [json.loads(l) for l in open("data/train.json") if l.strip()]
    test = [json.loads(l) for l in open("data/test.json") if l.strip()]
    folds = np.load("blackboard/mirfak_zeta_cv_split.npy")
    sn = np.array([x.get("SN_filter", 1) == 1 for x in d])
    y = np.zeros((len(d), 68, 3), np.float32)
    for i, x in enumerate(d):
        for j, t in enumerate(T):
            y[i, :68, j] = x[t]

    print("building features...", flush=True)
    Xtr = build(d)
    Xte = build(test)
    F = Xtr.shape[-1]
    print("feat dim", F, flush=True)

    oof = np.zeros((len(d), 107, 5), np.float32)
    yfull = np.zeros((len(d), 107, 3), np.float32)
    yfull[:, :68, :] = y
    tpreds = []
    for f in range(5):
        tr = (folds != f)
        # train only on clean rows (aligned with protocol)
        trc = tr & sn
        ptest = np.zeros((len(test), 107, 5), np.float32)
        for j in range(3):
            Xf = Xtr.reshape(-1, F)
            yf = yfull[:, :, j].reshape(-1)
            grp = np.repeat(np.arange(len(d)), 107)
            posmask = np.tile(np.arange(107) < 68, len(d))
            m = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=63,
                                  subsample=0.8, colsample_bytree=0.7, min_child_samples=30,
                                  reg_lambda=1.0, verbose=-1, n_jobs=4)
            trm = np.isin(grp, np.where(trc)[0]) & posmask
            m.fit(Xf[trm], yf[trm])
            Xv = Xtr[folds == f].reshape(-1, F)
            oof[folds == f, :107, SI[j]] = m.predict(Xv).reshape(-1, 107)
            Xt = Xte.reshape(-1, F)
            ptest[:, :, SI[j]] = m.predict(Xt).reshape(len(test), 107)
        tpreds.append(ptest)
        print(f"fold{f} done", flush=True)
    test_pred = np.mean(tpreds, 0)
    oof[..., 2] = 0
    oof[..., 4] = 0

    def sc(sel):
        a = oof[sel][:, :68][:, :, SI]
        b = y[sel]
        return float(np.sqrt(((a - b) ** 2).mean((0, 1))).mean())
    print("GBM-base OOF SN1 =", round(sc(sn), 4), "all =", round(sc(np.ones(len(d), bool)), 4))
    os.makedirs("workspaces/mirfak_zeta/runs/gbm_base", exist_ok=True)
    np.save("workspaces/mirfak_zeta/runs/gbm_base/oof.npy", oof)
    np.save("workspaces/mirfak_zeta/runs/gbm_base/test.npy", test_pred)
    print("saved")
