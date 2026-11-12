import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import log_loss

CLASSES = ["EAP", "HPL", "MWS"]
BASE = "/download_job_code/payload/sandbox"
train = pd.read_csv(f"{BASE}/data/train.csv")
test = pd.read_csv(f"{BASE}/data/test.csv")
folds = pd.read_csv(f"{BASE}/workspaces/deneb_upsilon/shared_folds.csv")[["id", "fold"]]
train = train.merge(folds, on="id", how="left")
y = train["author"].map({c: i for i, c in enumerate(CLASSES)}).values


def load(p):
    a = np.load(p); a = np.clip(a, 1e-15, 1)
    return a / a.sum(1, keepdims=True)


cands = {
    "rigel_final": (f"{BASE}/workspaces/rigel_xi/final_oof.npy", f"{BASE}/workspaces/rigel_xi/final_test.npy"),
    "rigel_lr": (f"{BASE}/workspaces/rigel_xi/blend_oof.npy", f"{BASE}/workspaces/rigel_xi/blend_test.npy"),
    "s2_lgbm": (f"{BASE}/workspaces/deneb_upsilon/stack2_lgbm_oof.npy", f"{BASE}/workspaces/deneb_upsilon/stack2_lgbm_test.npy"),
    "s2_mlp": (f"{BASE}/workspaces/deneb_upsilon/stack2_mlp_oof.npy", f"{BASE}/workspaces/deneb_upsilon/stack2_mlp_test.npy"),
}
names = list(cands)
O = np.stack([load(cands[n][0]) for n in names])
T = np.stack([load(cands[n][1]) for n in names])
for i, n in enumerate(names):
    print(f"{n}: {log_loss(y, O[i]):.5f}")


def f(w):
    w = np.abs(w); w = w / w.sum()
    p = np.tensordot(w, O, axes=1)
    return log_loss(y, p)


res = minimize(f, np.ones(len(names)) / len(names), method="Nelder-Mead",
               options={"maxiter": 3000, "xatol": 1e-4, "fatol": 1e-7})
w = np.abs(res.x); w /= w.sum()
print("weights:", {n: round(float(wi), 3) for n, wi in zip(names, w)})
print("COMBINED OOF (in-sample weights)", res.fun)

# honest: fit weights on 4 canonical folds, evaluate on held-out fold
fold = train["fold"].values


def _fit(idx, n):
    def f2(wt):
        ww = np.abs(wt); ww = ww / ww.sum()
        return log_loss(y[idx], np.tensordot(ww, O[:, idx], axes=1))
    r = minimize(f2, np.ones(n) / n, method="Nelder-Mead", options={"maxiter": 3000})
    ww = np.abs(r.x); return ww / ww.sum()


cv = []
for ff in range(5):
    trm = fold != ff
    ww = _fit(trm, len(names))
    cv.append(log_loss(y[~trm], np.tensordot(ww, O[:, ~trm], axes=1)))
print("HONEST nested-weight OOF", float(np.mean(cv)), [round(c, 4) for c in cv])
# also sub-combos
for combo in [("rigel_lr", "s2_lgbm"), ("rigel_lr", "s2_mlp"),
              ("s2_lgbm", "s2_mlp"), tuple(names)]:
    idx = [names.index(c) for c in combo]
    for ww in [0.3, 0.4, 0.5]:
        p = ww * O[idx[0]] + (1 - ww) * O[idx[1]] if len(idx) == 2 else None
        if len(idx) == 2:
            print(combo, ww, f"{log_loss(y, p):.5f}")

pb = np.tensordot(w, T, axes=1)
pd.DataFrame({"id": test["id"], "EAP": pb[:, 0], "HPL": pb[:, 1], "MWS": pb[:, 2]}).to_csv(
    f"{BASE}/workspaces/deneb_upsilon/submission_comb2.csv", index=False)
np.save(f"{BASE}/workspaces/deneb_upsilon/comb2_test.npy", pb)
print("saved comb2")
