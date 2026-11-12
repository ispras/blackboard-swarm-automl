import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss

CLASSES = ["EAP", "HPL", "MWS"]
BASE = "/download_job_code/payload/sandbox"
train = pd.read_csv(f"{BASE}/data/train.csv")
test = pd.read_csv(f"{BASE}/data/test.csv")
folds = pd.read_csv(f"{BASE}/workspaces/deneb_upsilon/shared_folds.csv")[["id", "fold"]]
train = train.merge(folds, on="id", how="left")
y = train["author"].map({c: i for i, c in enumerate(CLASSES)}).values

reg = {}
for d in ["deneb_upsilon", "rigel_xi", "arcturus_alpha"]:
    p = f"{BASE}/workspaces/{d}"
    if not os.path.isdir(p):
        continue
    for f in os.listdir(p):
        if any(s in f for s in ["blend", "stack", "geo", "final", "ensemble", "meta", "lstm", "skeleton", "punct"]):
            continue
        if f.endswith("_oof.npy"):
            nm = f[:-8]; tp = f"{p}/{nm}_test.npy"
            if os.path.exists(tp):
                reg[f"{d}:{nm}"] = (f"{p}/{f}", tp)
        elif f.startswith("oof_") and f.endswith(".npy"):
            nm = f[4:-4]; tp = f"{p}/test_{nm}.npy"
            if os.path.exists(tp):
                reg[f"{d}:{nm}"] = (f"{p}/{f}", tp)

names, O, T = [], [], []
for k, (op, tp) in reg.items():
    o = np.load(op); t = np.load(tp)
    if o.shape != (len(train), 3):
        continue
    if (np.abs(o).sum(1) < 1e-9).mean() > 0.02:
        continue
    o = np.clip(o, 1e-15, 1); o = o / o.sum(1, keepdims=True)
    t = np.clip(t, 1e-15, 1); t = t / t.sum(1, keepdims=True)
    if log_loss(y, o, labels=[0, 1, 2]) > 0.60:
        print("skip weak", k); continue
    names.append(k); O.append(o); T.append(t)
O = np.stack(O); T = np.stack(T)
print(f"{len(names)} base models")



def handcraft(df):
    tx = df["text"].fillna("")
    fe = pd.DataFrame(index=df.index)
    fe["nchar"] = tx.str.len(); fe["nword"] = tx.str.split().str.len()
    fe["avg_word"] = fe["nchar"] / fe["nword"].clip(lower=1)
    fe["nuniq"] = tx.apply(lambda s: len(set(s.lower().split())))
    fe["uniq_ratio"] = fe["nuniq"] / fe["nword"].clip(lower=1)
    fe["upper"] = tx.apply(lambda s: sum(c.isupper() for c in s))
    fe["upper_ratio"] = fe["upper"] / fe["nchar"].clip(lower=1)
    fe["digits"] = tx.apply(lambda s: sum(c.isdigit() for c in s))
    for nm, pat in [("comma", ","), ("semi", ";"), ("colon", ":"), ("excl", "!"),
                    ("quest", r"\?"), ("quote", '"'), ("apos", "'"), ("dash", "-"),
                    ("period", r"\.")]:
        fe[nm] = tx.str.count(pat)
    for nm, pat in [("the", r"\bthe\b"), ("and", r"\band\b"), ("of", r"\bof\b"),
                    ("i", r"\bI\b"), ("thou", r"\bthou\b")]:
        fe[nm] = tx.str.count(pat)
    return fe.values.astype(np.float32)


Htr = handcraft(train); Hte = handcraft(test)
logO = np.log(O).transpose(1, 2, 0).reshape(len(train), -1)
logT = np.log(T).transpose(1, 2, 0).reshape(len(test), -1)
X = np.hstack([logO, Htr]); Xte = np.hstack([logT, Hte])


def lgbm():
    return lgb.LGBMClassifier(objective="multiclass", num_class=3, n_estimators=700,
                              learning_rate=0.02, num_leaves=15, min_child_samples=60,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.6,
                              reg_lambda=1.0, n_jobs=8, verbose=-1)


def mlp():
    return MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-3, batch_size=256,
                         learning_rate_init=1e-3, max_iter=120, early_stopping=True,
                         n_iter_no_change=10, random_state=0)


seeds = [0, 1, 2]
oof_gb = np.zeros((len(train), 3)); oof_ml = np.zeros((len(train), 3))
for f in range(5):
    tr = train["fold"].values != f; va = train["fold"].values == f
    pg = np.zeros((va.sum(), 3)); pm = np.zeros((va.sum(), 3))
    sc = StandardScaler().fit(X[tr])
    for s in seeds:
        g = lgbm(); g.set_params(random_state=s); g.fit(X[tr], y[tr])
        pg += g.predict_proba(X[va])[:, [list(g.classes_).index(c) for c in range(3)]] / len(seeds)
        m = mlp(); m.set_params(random_state=s)
        m.fit(sc.transform(X[tr]), y[tr])
        pm += m.predict_proba(sc.transform(X[va]))[:, [list(m.classes_).index(c) for c in range(3)]] / len(seeds)
    oof_gb[va] = pg; oof_ml[va] = pm
    print(f"fold {f} gb {log_loss(y[va], pg):.5f} mlp {log_loss(y[va], pm):.5f}", flush=True)

print("OOF LGBM seedavg", log_loss(y, oof_gb))
print("OOF MLP  seedavg", log_loss(y, oof_ml))
for w in [0, 0.25, 0.35, 0.5]:
    print(f"mix w_lgbm={w}:", log_loss(y, w * oof_gb + (1 - w) * oof_ml))

# fit full and predict test
sc = StandardScaler().fit(X)
tg = np.zeros((len(test), 3)); tm = np.zeros((len(test), 3))
for s in seeds:
    g = lgbm(); g.set_params(random_state=s); g.fit(X, y)
    tg += g.predict_proba(Xte)[:, [list(g.classes_).index(c) for c in range(3)]] / len(seeds)
    m = mlp(); m.set_params(random_state=s); m.fit(sc.transform(X), y)
    tm += m.predict_proba(sc.transform(Xte))[:, [list(m.classes_).index(c) for c in range(3)]] / len(seeds)
np.save(f"{BASE}/workspaces/deneb_upsilon/stack2_lgbm_oof.npy", oof_gb)
np.save(f"{BASE}/workspaces/deneb_upsilon/stack2_lgbm_test.npy", tg)
np.save(f"{BASE}/workspaces/deneb_upsilon/stack2_mlp_oof.npy", oof_ml)
np.save(f"{BASE}/workspaces/deneb_upsilon/stack2_mlp_test.npy", tm)
for w, nm in [(0.65, "mix65")]:
    o = w * oof_gb + (1 - w) * oof_ml
    t = w * tg + (1 - w) * tm
    print(f"FINAL {nm} OOF", log_loss(y, o))
    pd.DataFrame({"id": test["id"], "EAP": t[:, 0], "HPL": t[:, 1], "MWS": t[:, 2]}).to_csv(
        f"{BASE}/workspaces/deneb_upsilon/submission_stack2_{nm}.csv", index=False)
