import argparse, os, json
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from scipy.optimize import minimize

ROOT = '/download_job_code/payload/sandbox'


def norm(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-9, None)
    return p / p.sum(1, keepdims=True)


def acc_of(w, V, y):
    return (np.tensordot(w, V, 1).argmax(1) == y).mean()


def fit_logloss(V, y, n):
    def obj(z):
        w = np.exp(z); w /= w.sum()
        p = np.tensordot(w, V, 1)
        return -np.log(np.clip(p[np.arange(len(y)), y], 1e-9, None)).mean()
    z0 = np.zeros(n)
    r = minimize(obj, z0, method='Nelder-Mead', options={'maxiter': 3000, 'xatol': 1e-4, 'fatol': 1e-7})
    w = np.exp(r.x); return w / w.sum()


def fit_acc(V, y, n, rng, iters=8000):
    best_w = np.ones(n) / n; best = acc_of(best_w, V, y)
    for _ in range(iters):
        w = rng.dirichlet(np.ones(n) * rng.choice([0.3, 0.5, 1.0, 2.0]))
        a = acc_of(w, V, y)
        if a > best:
            best, best_w = a, w
    for _ in range(30):
        improved = False
        for i in range(n):
            for d in (-0.15, -0.08, -0.04, 0.04, 0.08, 0.15):
                w = best_w.copy(); w[i] = max(0, w[i] + d)
                if w.sum() == 0:
                    continue
                w /= w.sum()
                a = acc_of(w, V, y)
                if a > best + 1e-9:
                    best, best_w = a, w; improved = True
        if not improved:
            break
    return best_w


def fit_topk(V, y, n, k=3):
    accs = np.array([acc_of(np.eye(n)[i], V, y) for i in range(n)])
    idx = np.argsort(-accs)[:k]
    w = np.zeros(n); w[idx] = 1.0 / k
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', required=True)
    ap.add_argument('--cv', default=os.path.join(ROOT, 'workspaces/aldebaran_epsilon/cv_split.csv'))
    ap.add_argument('--sample', default=os.path.join(ROOT, 'data/sample_submission.csv'))
    ap.add_argument('--out', default=os.path.join(ROOT, 'workspaces/bellatrix_delta/ensemble_final'))
    ap.add_argument('--method', default='auto')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(0)

    cv = pd.read_csv(args.cv); va = cv[cv.fold == 0].reset_index(drop=True)
    y = va.label.values; va_ids = va.image_id.values
    names, V, T = [], [], []
    for spec in args.models:
        parts = spec.split(':')
        name, vp, tp = parts[0], parts[1], parts[2]
        vi = parts[3] if len(parts) > 3 else None
        v = norm(np.load(vp)); t = norm(np.load(tp))
        if vi is not None and os.path.exists(vi):
            ids = pd.read_csv(vi).image_id.values
            if len(ids) == len(v) and not np.array_equal(ids, va_ids):
                order = pd.Index(ids)
                v = v[[order.get_loc(i) for i in va_ids]]
        assert len(v) == len(va), (name, len(v))
        names.append(name); V.append(v); T.append(t)
        print(f'{name:10s} val_acc={(v.argmax(1)==y).mean():.5f}')
    V = np.stack(V); T = np.stack(T); n = len(names)

    methods = {
        'equal': lambda Vt, yt: np.ones(n) / n,
        'logloss': lambda Vt, yt: fit_logloss(Vt, yt, n),
        'acc': lambda Vt, yt: fit_acc(Vt, yt, n, rng),
        'top3': lambda Vt, yt: fit_topk(Vt, yt, n, 3),
    }
    # cross-validated comparison inside fold0
    skf = StratifiedKFold(5, shuffle=True, random_state=7)
    scores = {m: [] for m in methods}
    for tri, tei in skf.split(V[0], y):
        for m, fn in methods.items():
            w = fn(V[:, tri], y[tri])
            scores[m].append(acc_of(w, V[:, tei], y[tei]))
    cvres = {m: float(np.mean(v)) for m, v in scores.items()}
    print('BLEND METHOD CV (inside fold0):', {k: round(v, 5) for k, v in cvres.items()})

    method = args.method
    if method == 'auto':
        method = max(cvres, key=cvres.get)
    print('chosen method:', method)
    w = {'equal': lambda: np.ones(n) / n, 'logloss': lambda: fit_logloss(V, y, n),
         'acc': lambda: fit_acc(V, y, n, rng), 'top3': lambda: fit_topk(V, y, n, 3)}[method]()
    full_acc = acc_of(w, V, y)
    print('weights:', dict(zip(names, np.round(w, 4))))
    print(f'full-fold0 accuracy of chosen method = {full_acc:.5f}')

    ens_val = np.tensordot(w, V, 1); ens_test = np.tensordot(w, T, 1)
    sample = pd.read_csv(args.sample)
    pd.DataFrame({'image_id': sample.image_id, 'label': ens_test.argmax(1)}).to_csv(
        os.path.join(args.out, 'submission.csv'), index=False)
    meta = {'names': names, 'weights': dict(zip(names, w.tolist())), 'method': method,
            'cv_scores': cvres, 'fold0_acc': float(full_acc),
            'singles': {names[i]: float((V[i].argmax(1) == y).mean()) for i in range(n)}}
    json.dump(meta, open(os.path.join(args.out, 'blend_meta.json'), 'w'), indent=2)
    np.save(os.path.join(args.out, 'ens_val_probs.npy'), ens_val)
    np.save(os.path.join(args.out, 'ens_test_probs.npy'), ens_test)
    print('wrote', os.path.join(args.out, 'submission.csv'))


if __name__ == '__main__':
    main()
