import os, argparse, json
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.special import softmax
from sklearn.model_selection import StratifiedKFold

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


def fit_weights(V, y, n_restart=6, seed=0, alpha=0.0, maxiter=4000):
    M = V.shape[0]
    eq = np.ones(M) / M

    def obj(z):
        w = softmax(z)
        p = np.tensordot(w, V, axes=(0, 0))
        return logloss(p, y) + alpha * float(((w - eq) ** 2).sum())

    best = (None, 1e9)
    rng = np.random.RandomState(seed)
    for i in range(n_restart):
        z0 = rng.randn(M) * 0.6 if i else np.zeros(M)
        r = minimize(obj, z0, method='Nelder-Mead',
                     options={'maxiter': maxiter, 'xatol': 1e-6, 'fatol': 1e-9})
        if r.fun < best[1]:
            best = (softmax(r.x), r.fun)
    return best[0], best[1]


def greedy_accuracy(V, y, rounds=25):
    M = V.shape[0]
    counts = np.zeros(M)
    cur = np.zeros((len(y), 5))
    for t in range(rounds):
        best_i, best_key = None, None
        for i in range(M):
            p = (cur * t + V[i]) / (t + 1)
            a = acc(p, y)
            key = (a, -logloss(p, y))
            if best_key is None or key > best_key:
                best_key, best_i = key, i
        counts[best_i] += 1
        cur = (cur * t + V[best_i]) / (t + 1)
    return counts / counts.sum()


def temper(V, temps):
    logits = np.log(np.clip(V, 1e-9, None)) / temps[:, None, None]
    return norm(np.exp(logits - logits.max(-1, keepdims=True)))


def fit_temps(V, y):
    M = V.shape[0]
    temps = np.ones(M)

    def obj(t, i):
        ti = np.clip(t[0], 0.2, 5.0)
        p = temper(V[i:i + 1], np.array([ti]))[0]
        return logloss(p, y)

    for i in range(M):
        r = minimize(obj, [1.0], args=(i,), method='Nelder-Mead',
                     options={'maxiter': 400})
        temps[i] = float(np.clip(r.x[0], 0.2, 5.0))
    return temps


def class_bias(p, y, maxiter=3000):
    def obj(b):
        return logloss(norm(p * np.exp(b)[None, :]), y)
    r = minimize(obj, np.zeros(5), method='Nelder-Mead', options={'maxiter': maxiter})
    return r.x


def cv_eval(V, y, kinds=('equal', 'logloss', 'greedy'), nfold=5, seed=7):
    """Return dict kind -> (cv_acc, cv_ll) using out-of-fold weight fitting."""
    skf = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=seed)
    res = {k: [[], []] for k in kinds}
    for extra in ('logloss+bias', 'logloss+temp', 'logloss+temp+bias'):
        res[extra] = [[], []]
    for tr, va in skf.split(np.zeros(len(y)), y):
        Vtr, Vva = V[:, tr], V[:, va]
        ytr, yva = y[tr], y[va]
        strat = {}
        strat['equal'] = np.ones(V.shape[0]) / V.shape[0]
        strat['logloss'], _ = fit_weights(Vtr, ytr, n_restart=4, seed=seed)
        strat['greedy'] = greedy_accuracy(Vtr, ytr, rounds=20)
        for k in ('equal', 'logloss', 'greedy'):
            p = norm(np.tensordot(strat[k], Vva, axes=(0, 0)))
            res[k][0].append(acc(p, yva)); res[k][1].append(logloss(p, yva))
        p = norm(np.tensordot(strat['logloss'], Vva, axes=(0, 0)))
        ptr = norm(np.tensordot(strat['logloss'], Vtr, axes=(0, 0)))
        b = class_bias(ptr, ytr)
        p = norm(p * np.exp(b)[None, :])
        res['logloss+bias'][0].append(acc(p, yva)); res['logloss+bias'][1].append(logloss(p, yva))
        # temperature scaling fitted on train part, then weights
        temps = fit_temps(Vtr, ytr)
        Vtr_t, Vva_t = temper(Vtr, temps), temper(Vva, temps)
        w_t, _ = fit_weights(Vtr_t, ytr, n_restart=4, seed=seed)
        pt = norm(np.tensordot(w_t, Vva_t, axes=(0, 0)))
        res['logloss+temp'][0].append(acc(pt, yva)); res['logloss+temp'][1].append(logloss(pt, yva))
        ptr_t = norm(np.tensordot(w_t, Vtr_t, axes=(0, 0)))
        b_t = class_bias(ptr_t, ytr)
        ptb = norm(pt * np.exp(b_t)[None, :])
        res['logloss+temp+bias'][0].append(acc(ptb, yva))
        res['logloss+temp+bias'][1].append(logloss(ptb, yva))
    out = {}
    for k, (aa, ll) in res.items():
        out[k] = (float(np.mean(aa)), float(np.mean(ll)))
    return out, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--full', action='append', default=[])
    ap.add_argument('--single', action='append', default=[])
    ap.add_argument('--out', default='final_blend_out')
    ap.add_argument('--skip-cv', action='store_true')
    ap.add_argument('--force-strategy', default='')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    report = {}

    # ---------- Stage A ----------
    fnames, fvs, fts = [], [], []
    for s in a.full:
        n, v, t = load(s); fnames.append(n); fvs.append(v); fts.append(t)
    common = None
    for v in fvs:
        common = set(v.image_id) if common is None else (common & set(v.image_id))
    common = pd.Index(sorted(common))
    yA = fvs[0].set_index('image_id').loc[common, 'label'].values.astype(int)
    VA = np.stack([norm(v.set_index('image_id').loc[common, P].values) for v in fvs], 0)
    for i, n in enumerate(fnames):
        print(f'[A] {n:24s} n={len(VA[i])} acc {acc(VA[i], yA):.4f} ll {logloss(VA[i], yA):.4f}')
    wA, llA = fit_weights(VA, yA, seed=11)
    pA = np.tensordot(wA, VA, axes=(0, 0))
    print(f'[A] weights {dict(zip(fnames, np.round(wA,3)))} acc {acc(pA,yA):.4f} ll {llA:.4f}')
    report['stageA'] = {'members': fnames, 'weights': wA.tolist(), 'n': int(len(yA)),
                        'acc': acc(pA, yA), 'logloss': llA}

    allids = None
    for v in fvs:
        allids = set(v.image_id) if allids is None else (allids | set(v.image_id))
    allids = pd.Index(sorted(allids))
    num = np.zeros((len(allids), 5)); den = np.zeros(len(allids))
    for n, v, w in zip(fnames, fvs, wA):
        d = v.set_index('image_id').reindex(allids)
        mask = ~d[P].isna().any(axis=1).values
        pv = np.nan_to_num(d[P].values)
        num[mask] += w * norm(pv[mask]); den[mask] += w
    E1v_all = norm(num / np.clip(den, 1e-9, None)[:, None])
    lab = fvs[0].set_index('image_id').reindex(allids)['label'].values
    m = ~pd.isna(lab)
    E1v_all = E1v_all[m]; allids = allids[m]; yAll = lab[m].astype(int)
    pd.DataFrame(E1v_all, columns=P).assign(image_id=allids.values, label=yAll).to_csv(
        os.path.join(a.out, 'E1_oof.csv'), index=False)
    print(f'[A] E1 oof rows {len(E1v_all)} acc {acc(E1v_all,yAll):.4f}')

    tcommon = None
    for t in fts:
        tcommon = set(t.image_id) if tcommon is None else (tcommon & set(t.image_id))
    tcommon = pd.Index(sorted(tcommon))
    E1t = np.zeros((len(tcommon), 5))
    for n, t, w in zip(fnames, fts, wA):
        E1t += w * norm(t.set_index('image_id').loc[tcommon, P].values)
    E1t = norm(E1t)
    pd.DataFrame(E1t, columns=P).assign(image_id=tcommon.values).to_csv(
        os.path.join(a.out, 'E1_test.csv'), index=False)

    # ---------- Stage B ----------
    snames, svs, sts = [], [], []
    for s in a.single:
        n, v, t = load(s); snames.append(n); svs.append(v); sts.append(t)
    idx = allids
    for v in svs:
        idx = idx.intersection(pd.Index(v.image_id))
    idx = idx.sort_values()
    sel = np.isin(allids, idx)
    y = yAll[sel]
    VB = [E1v_all[sel]]
    bnames = ['E1_fullOOF']
    for n, v in zip(snames, svs):
        VB.append(norm(v.set_index('image_id').loc[idx, P].values)); bnames.append(n)
    VB = np.stack(VB, 0)
    for i, n in enumerate(bnames):
        print(f'[B] {n:22s} n={len(VB[i])} acc {acc(VB[i], y):.4f} ll {logloss(VB[i], y):.4f}')

    if a.force_strategy:
        best_kind = a.force_strategy
    elif not a.skip_cv:
        cv, _ = cv_eval(VB, y)
        report['cv'] = {k: list(v) for k, v in cv.items()}
        for k, v in cv.items():
            print(f'[B-CV] {k:16s} acc {v[0]:.4f} ll {v[1]:.4f}')
        kinds = sorted(cv, key=lambda k: (cv[k][0], -cv[k][1]), reverse=True)
        best_kind = kinds[0]
    else:
        best_kind = 'logloss'
    print(f'[B] chosen strategy: {best_kind}')

    temps = np.ones(VB.shape[0])
    V_use = VB
    if 'temp' in best_kind:
        temps = fit_temps(VB, y)
        V_use = temper(VB, temps)
    if best_kind == 'equal':
        wB = np.ones(V_use.shape[0]) / V_use.shape[0]
    elif best_kind == 'greedy':
        wB = greedy_accuracy(V_use, y, rounds=25)
    else:
        wB, _ = fit_weights(V_use, y, seed=13)
    pB = norm(np.tensordot(wB, V_use, axes=(0, 0)))
    bias = class_bias(pB, y) if 'bias' in best_kind else np.zeros(5)
    pB = norm(pB * np.exp(bias)[None, :])
    print(f'[B] weights {dict(zip(bnames, np.round(wB,3)))} temps {np.round(temps,3)} '
          f'bias {np.round(bias,3)} acc {acc(pB,y):.4f} ll {logloss(pB,y):.4f}')
    report['stageB'] = {'members': bnames, 'weights': wB.tolist(), 'bias': bias.tolist(),
                        'temps': temps.tolist(), 'n': int(len(y)), 'strategy': best_kind,
                        'acc': acc(pB, y), 'logloss': logloss(pB, y)}

    # ---------- final test ----------
    Tlist = [E1t]
    for n, t in zip(snames, sts):
        Tlist.append(norm(t.set_index('image_id').loc[tcommon, P].values))
    T_use = temper(np.stack(Tlist, 0), temps) if 'temp' in best_kind else np.stack(Tlist, 0)
    final_t = np.tensordot(wB, T_use, axes=(0, 0))
    final_t = norm(final_t * np.exp(bias)[None, :])
    sub = pd.DataFrame({'image_id': tcommon.values, 'label': final_t.argmax(1)})
    sub.to_csv(os.path.join(a.out, 'submission.csv'), index=False)
    report['test_dist'] = {int(k): int(v) for k, v in sub.label.value_counts().items()}
    with open(os.path.join(a.out, 'report.json'), 'w') as f:
        json.dump(report, f, indent=1, default=float)
    print('FINAL val acc', report['stageB']['acc'], 'n', len(y))
    print(sub.label.value_counts().to_dict())
    print('wrote', os.path.join(a.out, 'submission.csv'))


if __name__ == '__main__':
    main()
