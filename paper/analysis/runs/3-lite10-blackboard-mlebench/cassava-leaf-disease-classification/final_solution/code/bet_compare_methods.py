import itertools
import numpy as np, pandas as pd, os
from sklearn.model_selection import StratifiedKFold
from scipy.optimize import minimize
ROOT = '/download_job_code/payload/sandbox'


def norm(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-9, None); return p / p.sum(1, keepdims=True)


cv = pd.read_csv(os.path.join(ROOT, 'workspaces/aldebaran_epsilon/cv_split.csv'))
va = cv[cv.fold == 0].reset_index(drop=True); y = va.label.values; va_ids = va.image_id.values
specs = {
 'b5': ('workspaces/bellatrix_delta/run_b5_512/tta4/val_probs_tta4.npy', None),
 'b5_640': ('workspaces/bellatrix_delta/run_b5_640/tta4/val_probs_tta4.npy', None),
 'cnxt': ('workspaces/bellatrix_delta/run_cnxt_384/tta4/val_probs_tta4.npy', None),
 'eva02': ('workspaces/vega_eta/runs/eva02b448_f0_s42/val_probs_tta.npy', None),
 'caformer': ('workspaces/vega_eta/runs/caformer384_f0_s42/val_probs_tta.npy', None),
 'swin': ('workspaces/aldebaran_epsilon/swin/val_probs.npy', 'workspaces/aldebaran_epsilon/swin/val_ids.csv'),
 'vit': ('workspaces/aldebaran_epsilon/vit/tta4/val_probs_tta4.npy', 'workspaces/aldebaran_epsilon/vit/val_ids.csv'),
 'cnv2': ('workspaces/vega_eta/runs/cnv2b384_f0_s42/val_probs_tta.npy', None),
}
names = list(specs); V = []
for k in names:
    p, vi = specs[k]; v = norm(np.load(os.path.join(ROOT, p)))
    if vi and os.path.exists(os.path.join(ROOT, vi)):
        ids = pd.read_csv(os.path.join(ROOT, vi)).image_id.values
        if len(ids) == len(v) and not np.array_equal(ids, va_ids):
            order = pd.Index(ids); v = v[[order.get_loc(i) for i in va_ids]]
    V.append(v)
V = np.stack(V); n = len(names)
print('singles', {names[i]: round(float((V[i].argmax(1) == y).mean()), 5) for i in range(n)})


def fit_logloss(A, yy, l2=0.0):
    m = A.shape[0]
    def obj(z):
        w = np.exp(z); w /= w.sum()
        p = np.tensordot(w, A, 1)
        return -np.log(np.clip(p[np.arange(len(yy)), yy], 1e-9, None)).mean() + l2 * np.sum((w - 1.0/m) ** 2)
    r = minimize(obj, np.zeros(m), method='Nelder-Mead', options={'maxiter': 3000, 'fatol': 1e-8})
    w = np.exp(r.x); return w / w.sum()


def fit_acc(A, yy, seed=0):
    rng = np.random.default_rng(seed); m = A.shape[0]
    best_w = np.ones(m) / m; best = ((np.tensordot(best_w, A, 1)).argmax(1) == yy).mean()
    for _ in range(6000):
        w = rng.dirichlet(np.ones(m) * rng.choice([0.3, 0.5, 1.0, 2.0]))
        a = ((np.tensordot(w, A, 1)).argmax(1) == yy).mean()
        if a > best: best, best_w = a, w
    return best_w


methods = {
 'equal': lambda A, yy: np.ones(A.shape[0]) / A.shape[0],
 'logloss': lambda A, yy: fit_logloss(A, yy, 0.0),
 'logloss_l2': lambda A, yy: fit_logloss(A, yy, 0.05),
 'acc': lambda A, yy: fit_acc(A, yy),
 'top3': lambda A, yy: (lambda accs: (lambda w: w)(np.isin(np.arange(A.shape[0]), np.argsort(-accs)[:3]) / 3.0))(
     np.array([((A[i].argmax(1)) == yy).mean() for i in range(A.shape[0])])),
}
splits = list(StratifiedKFold(5, shuffle=True, random_state=7).split(V[0], y))
print('--- nested CV inside fold0 ---')
for mname, fn in methods.items():
    accs = []
    for tri, tei in splits:
        w = fn(V[:, tri], y[tri])
        # renormalize top3 (isdin gives bool*1/3)
        w = w / w.sum()
        accs.append(((np.tensordot(w, V[:, tei], 1)).argmax(1) == y[tei]).mean())
    print(f'{mname:12s} cv={np.mean(accs):.5f} +/- {np.std(accs)/np.sqrt(len(accs)):.5f}')
