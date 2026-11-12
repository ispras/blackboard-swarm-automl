import os, argparse, json
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.special import softmax

DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))
P = [f'p{i}' for i in range(5)]


def norm(a):
    return a / np.clip(a.sum(-1, keepdims=True), 1e-9, None)


def logloss(p, y):
    return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-9, 1)).mean())


def acc(p, y):
    return float((p.argmax(1) == y).mean())


def load(spec):
    name, vpath, tpath = spec.split(':', 2)
    v = pd.read_csv(vpath)
    t = pd.read_csv(tpath)
    for d in (v, t):
        s = d[P].sum(1).values.reshape(-1, 1)
        d[P] = d[P].values / np.clip(s, 1e-9, None)
    return name, v, t


def fit_weights(V, y, n_restart=6, seed=0):
    """V: (M,N,5) probs, y: (N,) labels. Return nonneg weights sum1 minimizing logloss."""
    M = V.shape[0]

    def obj(z):
        w = softmax(z)
        return logloss(np.tensordot(w, V, axes=(0, 0)), y)

    best = (None, 1e9)
    rng = np.random.RandomState(seed)
    for i in range(n_restart):
        z0 = rng.randn(M) * 0.5 if i else np.zeros(M)
        r = minimize(obj, z0, method='Nelder-Mead',
                     options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-8})
        if r.fun < best[1]:
            best = (softmax(r.x), r.fun)
    return best[0], best[1]


def build(specs):
    names, vs, ts = [], [], []
    for s in specs:
        n, v, t = load(s)
        names.append(n); vs.append(v); ts.append(t)
    return names, vs, ts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--full', action='append', default=[])
    ap.add_argument('--single', action='append', default=[])
    ap.add_argument('--out', default='blend2_out')
    ap.add_argument('--force-equal', action='store_true')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    report = {}

    # ---------- stage A: full-OOF members ----------
    fnames, fvs, fts = build(a.full)
    common = None
    for v in fvs:
        s = set(v.image_id)
        common = s if common is None else (common & s)
    common = pd.Index(sorted(common))
    yA = fvs[0].set_index('image_id').loc[common, 'label'].values.astype(int)
    VA = np.stack([norm(v.set_index('image_id').loc[common, P].values) for v in fvs], 0)
    for i, n in enumerate(fnames):
        print(f'[full] {n:24s} n={len(VA[i])} acc {acc(VA[i], yA):.4f} ll {logloss(VA[i], yA):.4f}')
    wA, llA = fit_weights(VA, yA, seed=1)
    pA = np.tensordot(wA, VA, axes=(0, 0))
    print(f'[full] weights {dict(zip(fnames, np.round(wA,3)))} acc {acc(pA,yA):.4f} ll {llA:.4f}')
    report['stageA'] = {'members': fnames, 'weights': wA.tolist(), 'n': int(len(yA)),
                        'acc': acc(pA, yA), 'logloss': llA}

    # E1 probs on the union of all full-member val rows
    allids = None
    for v in fvs:
        s = set(v.image_id)
        allids = s if allids is None else (allids | s)
    allids = pd.Index(sorted(allids))
    E1v = {i: np.zeros((len(allids), 5)) for i in []}
    num = np.zeros((len(allids), 5)); den = np.zeros(len(allids))
    for n, v, w in zip(fnames, fvs, wA):
        d = v.set_index('image_id').reindex(allids)
        mask = ~d[P].isna().any(axis=1).values
        pv = np.nan_to_num(d[P].values)
        num[mask] += w * norm(pv[mask]); den[mask] += w
    E1v_all = norm(num / np.clip(den, 1e-9, None)[:, None])
    lab = fvs[0].set_index('image_id').reindex(allids)['label'].values
    m = ~pd.isna(lab)
    E1v_all = E1v_all[m]; allids = allids[m]; yA_all = lab[m].astype(int)
    # test E1
    tcommon = None
    for t in fts:
        s = set(t.image_id)
        tcommon = s if tcommon is None else (tcommon & s)
    tcommon = pd.Index(sorted(tcommon))
    E1t = np.zeros((len(tcommon), 5))
    for n, t, w in zip(fnames, fts, wA):
        E1t += w * norm(t.set_index('image_id').loc[tcommon, P].values)
    E1t = norm(E1t)
    print(f'[stageA] E1 test rows {len(E1t)}')

    # ---------- stage B: single-split members on canonical val ----------
    snames, svs, sts = build(a.single)
    # canonical rows = intersection of E1 rows and all single val rows
    idx = allids
    for v in svs:
        idx = idx.intersection(pd.Index(v.image_id))
    idx = idx.sort_values()
    y = yA_all[np.isin(allids, idx)]
    members_B = [E1v_all[np.isin(allids, idx)]]
    bnames = ['E1_fullOOF']
    for n, v in zip(snames, svs):
        members_B.append(norm(v.set_index('image_id').loc[idx, P].values))
        bnames.append(n)
    VB = np.stack(members_B, 0)
    for i, n in enumerate(bnames):
        print(f'[single] {n:22s} n={len(VB[i])} acc {acc(VB[i], y):.4f} ll {logloss(VB[i], y):.4f}')
    wB, llB = fit_weights(VB, y, seed=2)
    pB = np.tensordot(wB, VB, axes=(0, 0))
    print(f'[single] weights {dict(zip(bnames, np.round(wB,3)))} acc {acc(pB,y):.4f} ll {llB:.4f}')
    report['stageB'] = {'members': bnames, 'weights': wB.tolist(), 'n': int(len(y)),
                        'acc': acc(pB, y), 'logloss': llB}

    # per-class bias (multiplicative) tuned on stage B val
    def cobj(b):
        return logloss(norm(pB * np.exp(b)[None, :]), y)
    r = minimize(cobj, np.zeros(5), method='Nelder-Mead', options={'maxiter': 3000})
    pb = norm(pB * np.exp(r.x)[None, :])
    print(f'[bias] {np.round(r.x,3)} acc {acc(pb,y):.4f} ll {logloss(pb,y):.4f}')
    bias = r.x if acc(pb, y) > acc(pB, y) + 0.001 else np.zeros(5)
    report['bias'] = bias.tolist()
    report['final_val_acc'] = acc(norm(pB * np.exp(bias)[None, :]), y)

    # ---------- final test ----------
    final_t = wB[0] * E1t
    for n, t, w in zip(snames, sts, wB[1:]):
        final_t += w * norm(t.set_index('image_id').loc[tcommon, P].values)
    final_t = norm(final_t * np.exp(bias)[None, :])
    sub = pd.DataFrame({'image_id': tcommon.values, 'label': final_t.argmax(1)})
    sub.to_csv(os.path.join(a.out, 'submission.csv'), index=False)
    report['test_dist'] = {int(k): int(v) for k, v in sub.label.value_counts().items()}
    with open(os.path.join(a.out, 'report.json'), 'w') as f:
        json.dump(report, f, indent=1, default=float)
    print('final val acc', report['final_val_acc'])
    print(sub.label.value_counts().to_dict())
    print('wrote', os.path.join(a.out, 'submission.csv'))


if __name__ == '__main__':
    main()
