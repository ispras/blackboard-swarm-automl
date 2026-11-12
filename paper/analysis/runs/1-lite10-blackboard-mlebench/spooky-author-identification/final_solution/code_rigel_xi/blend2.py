import os, glob, json
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from common import load_data, folds, logloss, WS, FINAL, CLASSES

WSROOT = "/download_job_code/payload/sandbox/workspaces"
EPS = 1e-15


BAD_TOKENS = ("blend", "stack", "ensemble", "meta", "submission", "final", "comb", "geoblend")


def _bad(stem):
    s = stem.lower()
    return any(t in s for t in BAD_TOKENS)


def discover():
    pairs = {}
    for ws in sorted(os.listdir(WSROOT)):
        d = os.path.join(WSROOT, ws)
        if not os.path.isdir(d):
            continue
        for p in glob.glob(os.path.join(d, "oof_*.npy")):
            stem = os.path.basename(p)[len("oof_"):-4]
            if _bad(stem):
                continue
            tp = os.path.join(d, f"test_{stem}.npy")
            if os.path.exists(tp):
                pairs[f"{ws}:{stem}"] = (p, tp)
        for p in glob.glob(os.path.join(d, "*_oof.npy")):
            stem = os.path.basename(p)[:-len("_oof.npy")]
            if _bad(stem):
                continue
            tp = os.path.join(d, f"{stem}_test.npy")
            if os.path.exists(tp):
                pairs[f"{ws}:{stem}"] = (p, tp)
    return pairs


def norm(p):
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1)
    return p / p.sum(1, keepdims=True)


def softmax_temp(logits, t):
    z = logits / t
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def best_temp(oof, y):
    logits = np.log(norm(oof))
    best_t, best_ll = 1.0, logloss(y, oof)
    for t in np.linspace(0.4, 3.0, 53):
        ll = logloss(y, softmax_temp(logits, t))
        if ll < best_ll:
            best_ll, best_t = ll, t
    return best_t, best_ll


def calibrate(oof, test, y):
    T, _ = best_temp(oof, y)
    o = softmax_temp(np.log(norm(oof)), T)
    t = softmax_temp(np.log(norm(test)), T)
    return o, t, T


def opt_weights(oofs, y, geo=False):
    n = len(oofs)
    if geo:
        M = np.stack([np.log(norm(o)) for o in oofs])
    else:
        M = np.stack([norm(o) for o in oofs])

    def loss(w):
        w = np.abs(w); w = w / w.sum()
        z = np.tensordot(w, M, axes=1)
        if geo:
            z = z - z.max(1, keepdims=True)
            e = np.exp(z); p = e / e.sum(1, keepdims=True)
        else:
            p = z
        return logloss(y, p)

    best = (1e9, None)
    for _ in range(4):
        w0 = np.random.dirichlet(np.ones(n))
        r = minimize(loss, w0, method="Nelder-Mead",
                     options={"maxiter": 4000, "xatol": 1e-4, "fatol": 1e-8})
        if r.fun < best[0]:
            best = (r.fun, r.x)
    w = np.abs(best[1]); w = w / w.sum()
    return w, best[0]


def combine(oofs, tests, w, geo):
    if geo:
        o = np.tensordot(w, np.stack([np.log(norm(x)) for x in oofs]), axes=1)
        t = np.tensordot(w, np.stack([np.log(norm(x)) for x in tests]), axes=1)
        o = np.exp(o - o.max(1, keepdims=True)); o /= o.sum(1, keepdims=True)
        t = np.exp(t - t.max(1, keepdims=True)); t /= t.sum(1, keepdims=True)
        return o, t
    o = np.tensordot(w, np.stack([norm(x) for x in oofs]), axes=1)
    t = np.tensordot(w, np.stack([norm(x) for x in tests]), axes=1)
    return o, t


def greedy(oofs, y, geo=False, iters=60):
    n = len(oofs)
    P = [np.log(norm(o)) if geo else norm(o) for o in oofs]
    counts = np.zeros(n)
    s = None
    for t in range(iters):
        best = (1e9, -1)
        for i in range(n):
            cand = P[i] if s is None else s + P[i]
            p = cand / (t + 1)
            if geo:
                z = p - p.max(1, keepdims=True)
                e = np.exp(z); p = e / e.sum(1, keepdims=True)
            ll = logloss(y, p)
            if ll < best[0]:
                best = (ll, i)
        i = best[1]
        s = P[i] if s is None else s + P[i]
        counts[i] += 1
    w = counts / counts.sum()
    return w, best[0]


def hand_feats(texts):
    t = pd.Series(texts).astype(str)
    L = t.str.len().clip(lower=1)
    nw = t.str.split().apply(len).clip(lower=1)
    f = {}
    f["len"] = (t.str.len() / 100.0)
    f["nwords"] = (nw / 20.0)
    f["avglen"] = (t.str.len() / nw / 10.0)
    f["unique"] = (t.str.split().apply(lambda x: len(set(x))) / nw)
    f["upper"] = t.str.count(r"[A-Z]") / L
    f["digit"] = t.str.count(r"\d") / L
    f["punct"] = t.str.count(r"[^\w\s]") / L
    f["comma"] = t.str.count(",") / 5.0
    f["quote"] = t.str.count('"') / 5.0
    f["aper"] = t.str.count("'") / 5.0
    f["semi"] = t.str.count(";") / 5.0
    f["colon"] = t.str.count(":") / 5.0
    f["excl"] = t.str.count("!") / 5.0
    f["ques"] = t.str.count(r"\?") / 5.0
    f["dash"] = t.str.count("-") / 5.0
    tl = t.str.lower()
    for w in ["the", "and", "of", "to", "a", "i", "my"]:
        f["w_" + w] = (tl.str.count(r"\b" + w + r"\b") / nw)
    X = np.stack([f[k].values for k in f], axis=1)
    return X.astype(np.float64)


def nested_combine(so_list, y, seed=42):
    M = np.stack([norm(o) for o in so_list])
    oof = np.zeros((len(y), 3))
    for itr, iva in folds(y, seed=seed):
        def loss(w):
            w = np.abs(w); w = w / w.sum()
            return logloss(y[itr], np.tensordot(w, M[:, itr], axes=1))
        best = (1e9, None)
        for _ in range(2):
            w0 = np.random.dirichlet(np.ones(len(so_list)))
            r = minimize(loss, w0, method="Nelder-Mead",
                         options={"maxiter": 800, "fatol": 1e-8})
            if r.fun < best[0]:
                best = (r.fun, np.abs(r.x))
        w = best[1] / best[1].sum()
        oof[iva] = np.tensordot(w, M[:, iva], axes=1)
    return logloss(y, oof)


def stack_model(X, Xt, y, make_clf, seed=42):
    oof_meta = np.zeros((len(y), 3))
    for itr, iva in folds(y, seed=seed):
        clf = make_clf()
        clf.fit(X[itr], y[itr])
        oof_meta[iva] = clf.predict_proba(X[iva])
    clf = make_clf()
    clf.fit(X, y)
    test_meta = clf.predict_proba(Xt)
    return oof_meta, test_meta, logloss(y, oof_meta)


def lr_factory(C=1.0):
    def f():
        return LogisticRegression(C=C, max_iter=4000, multi_class="multinomial")
    return f


def mlp_factory(seed=42):
    from sklearn.neural_network import MLPClassifier
    def f():
        return MLPClassifier(hidden_layer_sizes=(64,), alpha=1e-3, max_iter=60,
                             early_stopping=True, n_iter_no_change=8,
                             validation_fraction=0.12, random_state=seed)

    return f


def lgbm_factory(seed=42, n_estimators=500, lr=0.03):
    import lightgbm as lgb
    def f():
        return lgb.LGBMClassifier(n_estimators=n_estimators, learning_rate=lr, num_leaves=15,
                                  subsample=0.8, colsample_bytree=0.8, min_child_samples=60,
                                  reg_lambda=1.0, random_state=seed, n_jobs=4, verbose=-1)
    return f


class SeedAvg:
    def __init__(self, factories):
        self.factories = factories
        self.models = []

    def fit(self, X, y):
        self.models = [f().fit(X, y) for f in self.factories]
        return self

    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.models], axis=0)


def seedavg_factory(base_factory, seeds=(1, 2, 3)):
    return lambda: SeedAvg([base_factory(seed=s) for s in seeds])


def main():
    tr, te, y = load_data()
    pairs = discover()
    names, oofs_c, tests_c, temps = [], [], [], {}
    print("=== base models ===")
    for name, (op, tp) in pairs.items():
        o, t = np.load(op), np.load(tp)
        if o.shape != (len(y), 3) or t.shape != (len(te), 3):
            print("  skip bad shape", name, o.shape, t.shape); continue
        if (o.sum(1) <= 0).any() or (t.sum(1) <= 0).any():
            print("  skip empty rows", name); continue
        raw_ll = logloss(y, o)
        if raw_ll > 0.5:
            print(f"  skip weak ({raw_ll:.3f})", name); continue
        if raw_ll < 0.28:
            print(f"  !! SKIP suspicious meta-leak ({raw_ll:.3f})", name); continue
        oc, tc, T = calibrate(o, t, y)
        names.append(name); oofs_c.append(oc); tests_c.append(tc); temps[name] = T
        print(f"  {name:45s} raw={logloss(y,o):.5f} cal={logloss(y,oc):.5f} T={T:.2f}")

    results = {}
    for geo in [False, True]:
        w, ll = opt_weights(oofs_c, y, geo=geo)
        tag = "geo" if geo else "arith"
        results[tag] = (ll, w)
        print(f"[{tag}] weighted OOF={ll:.5f} w=" +
              str({n: round(float(x), 3) for n, x in zip(names, w) if x > 0.01}))

    for geo in [False, True]:
        gw, gll = greedy(oofs_c, y, geo=geo)
        gw = gw.copy()
        tag = "greedy_geo" if geo else "greedy_arith"
        results[tag] = (gll, gw)
        print(f"[{tag}] OOF={gll:.5f} w=" +
              str({n: round(float(x), 3) for n, x in zip(names, gw) if x > 0.01}))

    hf_tr = hand_feats(tr["text"].values)
    hf_te = hand_feats(te["text"].values)

    def feats(arrs):
        p = np.concatenate([norm(a) for a in arrs], axis=1)
        lp = np.log(p)
        return np.concatenate([p, lp], axis=1)

    X = np.concatenate([feats(oofs_c), hf_tr], axis=1)
    Xt = np.concatenate([feats(tests_c), hf_te], axis=1)

    stackers = {"stack_lr": lr_factory(1.0), "stack_lr3": lr_factory(3.0),
                "stack_mlp": mlp_factory(), "stack_lgbm": lgbm_factory(),
                "stack_lgbm_sa": seedavg_factory(lambda seed: lgbm_factory(seed)),
                "stack_mlp_sa": seedavg_factory(lambda seed: mlp_factory(seed))}
    stack_out = {}
    for sname, fac in stackers.items():
        try:
            so, st, sll = stack_model(X, Xt, y, fac)
            results[sname] = (sll, None)
            stack_out[sname] = (so, st)
            print(f"[{sname}] OOF={sll:.5f}")
        except Exception as e:
            print(f"[{sname}] FAILED {e}")

    # pairwise combine stackers (1 weight fitted on OOF)
    comb_pairs = {}
    snames = list(stack_out.keys())
    for i in range(len(snames)):
        for j in range(i + 1, len(snames)):
            a, b = snames[i], snames[j]
            oa, ta = stack_out[a]
            ob, tb = stack_out[b]
            best = (1e9, 0.5)
            for w in np.linspace(0, 1, 21):
                ll = logloss(y, w * norm(oa) + (1 - w) * norm(ob))
                if ll < best[0]:
                    best = (ll, w)
            key = f"comb_{a}_{b}"
            comb_pairs[key] = (a, b)
            results[key] = (best[0], None)
            stack_out[key] = (best[1] * norm(oa) + (1 - best[1]) * norm(ob),
                              best[1] * norm(ta) + (1 - best[1]) * norm(tb))
            print(f"[{key}] OOF={best[0]:.5f} w={best[1]:.2f}")

    # pick best
    best_name = min(results, key=lambda k: results[k][0])
    print("BEST:", best_name, results[best_name][0])
    if best_name in comb_pairs:
        a, b = comb_pairs[best_name]
        try:
            nec = nested_combine([stack_out[a][0], stack_out[b][0]], y)
            print(f"NESTED-COMBINE check for {best_name}: honest OOF={nec:.5f}")
        except Exception as e:
            print("nested check failed", e)

    if best_name in stack_out:
        oof_b, test_b = stack_out[best_name]
    else:
        geo = best_name.endswith("geo")
        w = results[best_name][1]
        oof_b, test_b = combine(oofs_c, tests_c, w, geo)

    sub = pd.DataFrame(norm(test_b), columns=CLASSES)
    sub.insert(0, "id", te["id"].values)
    sub.to_csv(os.path.join(WS, "submission_blend.csv"), index=False)
    np.save(os.path.join(WS, "blend_oof.npy"), oof_b)
    np.save(os.path.join(WS, "blend_test.npy"), test_b)
    report = {"models": names, "temps": {k: float(v) for k, v in temps.items()},
              "results": {k: float(v[0]) for k, v in results.items()},
              "best": best_name, "best_ll": float(results[best_name][0])}
    with open(os.path.join(WS, "blend_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("saved submission_blend.csv and blend_report.json")


if __name__ == "__main__":
    main()
