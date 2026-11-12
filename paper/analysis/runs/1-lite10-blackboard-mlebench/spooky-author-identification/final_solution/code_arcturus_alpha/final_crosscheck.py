import glob, os, json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

WS = "workspaces/arcturus_alpha"
y = np.load(f"{WS}/y.npy")
folds = np.load(f"{WS}/folds.npy")
NF = 5
EXCLUDE = ("blend", "stack", "ensemble", "meta", "final", "comb", "submission", "seedavg_bertcased")

oofs, tests, names = [], [], []
for ws in ["workspaces/arcturus_alpha", "workspaces/rigel_xi", "workspaces/deneb_upsilon"]:
    for f in sorted(glob.glob(f"{ws}/*oof*.npy")):
        base = os.path.basename(f)
        if any(x in base for x in EXCLUDE):
            continue
        t = f.replace("oof", "test")
        if not os.path.exists(t):
            continue
        o = np.load(f); te = np.load(t)
        if o.shape != (len(y), 3) or te.shape != (1958, 3):
            continue
        if np.isnan(o).any() or np.isnan(te).any():
            continue
        oofs.append(np.clip(o, 1e-7, 1)); tests.append(np.clip(te, 1e-7, 1))
        names.append(os.path.basename(ws) + ":" + base)

print(f"pooled {len(oofs)} base models")
O = np.hstack(oofs); T = np.hstack(tests)
Olog = np.log(O); Tlog = np.log(T)
X = np.hstack([O, Olog]); XT = np.hstack([T, Tlog])

meta = np.zeros((len(y), 3))
for f in range(NF):
    tr = folds != f; va = folds == f
    clf = LogisticRegression(C=1.0, max_iter=3000, n_jobs=8)
    clf.fit(X[tr], y[tr])
    meta[va] = clf.predict_proba(X[va])
ll = log_loss(y, meta, labels=[0, 1, 2])
print("nested LR-stack OOF (independent cross-check):", round(ll, 5))

clf = LogisticRegression(C=1.0, max_iter=3000, n_jobs=8)
clf.fit(X, y)
te = clf.predict_proba(XT)
sub = pd.DataFrame({"id": pd.read_csv("data/test.csv")["id"], "EAP": te[:, 0], "HPL": te[:, 1], "MWS": te[:, 2]})
sub.to_csv(f"{WS}/submission_crosscheck.csv", index=False)
print("saved", f"{WS}/submission_crosscheck.csv", "n=", len(sub))
