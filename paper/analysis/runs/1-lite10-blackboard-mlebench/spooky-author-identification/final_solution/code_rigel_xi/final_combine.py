import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from common import load_data, logloss, WS, FINAL, CLASSES, folds

EPS = 1e-15


def norm(p):
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1)
    return p / p.sum(1, keepdims=True)


def main():
    tr, te, y = load_data()
    cands = {}
    mine_o = os.path.join(WS, "blend_oof.npy")
    mine_t = os.path.join(WS, "blend_test.npy")
    if os.path.exists(mine_o):
        cands["rigel_blend"] = (np.load(mine_o), np.load(mine_t))
    ext = {
        "deneb_s2_lgbm": "/download_job_code/payload/sandbox/workspaces/deneb_upsilon/stack2_lgbm",
        "deneb_s2_mlp": "/download_job_code/payload/sandbox/workspaces/deneb_upsilon/stack2_mlp",
    }
    for name, pre in ext.items():
        o, t = pre + "_oof.npy", pre + "_test.npy"
        if os.path.exists(o) and os.path.exists(t):
            cands[name] = (np.load(o), np.load(t))

    if not cands:
        print("no candidates"); return
    print("candidates:")
    for k, (o, t) in cands.items():
        if o.shape[0] == len(y):
            print(f"  {k}: OOF={logloss(y, o):.5f}")

    names = list(cands.keys())
    if len(names) == 1:
        oof_b, test_b, best = cands[names[0]]
        print("single candidate")
    else:
        M = np.stack([norm(cands[k][0]) for k in names])
        Mt = np.stack([norm(cands[k][1]) for k in names])

        def loss(w):
            w = np.abs(w); w = w / w.sum()
            return logloss(y, np.tensordot(w, M, axes=1))

        best = (1e9, None)
        for _ in range(4):
            w0 = np.random.dirichlet(np.ones(len(names)))
            r = minimize(loss, w0, method="Nelder-Mead",
                         options={"maxiter": 3000, "xatol": 1e-4, "fatol": 1e-8})
            if r.fun < best[0]:
                best = (r.fun, np.abs(r.x) / np.abs(r.x).sum())
        w = best[1]
        oof_b = np.tensordot(w, M, axes=1)
        test_b = np.tensordot(w, Mt, axes=1)
        print(f"combined OOF={best[0]:.5f} w=" +
              str({n: round(float(x), 3) for n, x in zip(names, w)}))

        # honest nested estimate of the combine weights
        nested = np.zeros((len(y), 3))
        for itr, iva in folds(y):
            def loss_tr(w):
                w = np.abs(w); w = w / w.sum()
                return logloss(y[itr], np.tensordot(w, M[:, itr], axes=1))
            b = (1e9, None)
            for _ in range(3):
                w0 = np.random.dirichlet(np.ones(len(names)))
                r = minimize(loss_tr, w0, method="Nelder-Mead",
                             options={"maxiter": 2000, "fatol": 1e-8})
                if r.fun < b[0]:
                    b = (r.fun, np.abs(r.x) / np.abs(r.x).sum())
            nested[iva] = np.tensordot(b[1], M[:, iva], axes=1)
        print(f"nested-combine OOF={logloss(y, nested):.5f}")
    np.save(os.path.join(WS, "final_oof.npy"), oof_b)
    np.save(os.path.join(WS, "final_test.npy"), test_b)
    sub = pd.DataFrame(norm(test_b), columns=CLASSES)
    sub.insert(0, "id", te["id"].values)
    sub.to_csv(os.path.join(WS, "submission_final.csv"), index=False)
    print("saved submission_final.csv")


if __name__ == "__main__":
    main()
