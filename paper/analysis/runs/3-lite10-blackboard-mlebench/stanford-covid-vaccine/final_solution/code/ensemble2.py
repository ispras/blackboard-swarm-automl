"""Greedy (Caruana) ensemble selection with nested-CV honest evaluation + submission builder."""
import os, glob, json, argparse
import numpy as np
from sklearn.model_selection import KFold
from common import load_train, load_test, mcrmse, SCORE_IDX, ALL_TARGETS, WS, write_submission

AGG = os.path.join(WS, 'agg')
tr = load_train()
te = load_test()
n = len(tr['seq'])
snf = np.array([r['SN_filter'] for r in tr['rows']], dtype=bool)
mask = tr['mask'][:, :68]
yv = tr['y'][:, :68, :][:, :, SCORE_IDX]
mask_sn = mask[snf]

names = sorted([os.path.basename(f)[4:-4] for f in glob.glob(os.path.join(AGG, 'oof_*.npy')) if not os.path.basename(f).startswith('oof_ens')])
ENS = os.path.join(WS, 'ens'); os.makedirs(ENS, exist_ok=True)
O = np.stack([np.load(os.path.join(AGG, f'oof_{m}.npy')) for m in names]).astype(np.float64)
T = np.stack([np.load(os.path.join(AGG, f'test_{m}.npy')) for m in names]).astype(np.float64)
M = len(names)
print('models:', names)


def score(p, rows, use_sn=True):
    y = yv[rows]; m = mask[rows]
    if use_sn:
        r = snf[rows]
        return mcrmse(y[r], p[r], m[r], cols=[0, 1, 2])[0]
    return mcrmse(y, p, m, cols=[0, 1, 2])[0]


def greedy(rows, iters=40, use_sn=True):
    cnt = np.zeros(M)
    cur = np.zeros((len(rows), 68, 3))
    best_hist = []
    for it in range(iters):
        best_s, best_j = 1e9, -1
        for j in range(M):
            cand = (cur * cnt.sum() + np.take(O, rows, axis=1)[j]) / (cnt.sum() + 1)
            cand = np.maximum(cand, -0.5)
            s = score(cand, rows, use_sn)
            if s < best_s:
                best_s, best_j = s, j
        cnt[best_j] += 1
        cur = (cur * (cnt.sum() - 1) + np.take(O, rows, axis=1)[best_j]) / cnt.sum()
        best_hist.append(best_s)
        if it > 5 and best_hist[-1] > best_hist[-2] - 1e-5:
            break
    return cnt / cnt.sum(), best_hist[-1]


# nested honest evaluation
kf = KFold(5, shuffle=True, random_state=42)
pred = np.zeros_like(yv)
for tr_i, va_i in kf.split(np.arange(n)):
    w, _ = greedy(tr_i)
    pred[va_i] = np.einsum('m,mptc->ptc', w, np.take(O, va_i, axis=1))
pred = np.maximum(pred, -0.5)
s_all = mcrmse(yv, pred, mask, cols=[0, 1, 2])[0]
s_sn = mcrmse(yv[snf], pred[snf], mask_sn, cols=[0, 1, 2])[0]
print(f"NESTED greedy: all={s_all:.4f} sn={s_sn:.4f}")

# also compare equal-weight avg and best single
avg = np.maximum(O.mean(0), -0.5)
print(f"avg: sn={mcrmse(yv[snf], avg[snf], mask_sn, cols=[0,1,2])[0]:.4f}")
singles = {m: mcrmse(yv[snf], np.maximum(O[i], -0.5)[snf], mask_sn, cols=[0, 1, 2])[0]
           for i, m in enumerate(names)}
print('singles:', {k: round(v, 4) for k, v in sorted(singles.items(), key=lambda x: x[1])})

# final weights on full OOF
W, sfinal = greedy(np.arange(n))
print('final weights:', {names[i]: round(W[i], 3) for i in range(M) if W[i] > 1e-4}, 'oof', round(sfinal, 4))

test_pred = np.maximum(np.einsum('m,mptc->ptc', W, T), -0.5)
oof_pred = np.maximum(np.einsum('m,mptc->ptc', W, O), -0.5)
print('final oof sn:', mcrmse(yv[snf], oof_pred[snf], mask_sn, cols=[0, 1, 2])[0])
np.save(os.path.join(ENS, 'test_ens_greedy.npy'), test_pred.astype(np.float32))
np.save(os.path.join(ENS, 'oof_ens_greedy.npy'), oof_pred.astype(np.float32))
with open(os.path.join(ENS, 'greedy_weights.json'), 'w') as f:
    json.dump(dict(models=names, W=W.tolist(), nested_all=s_all, nested_sn=s_sn), f, indent=2)

# build full (240,107,5) submission: scored 3 cols at 0:68 from ensemble, rest from train means
full = np.zeros((len(te['seq']), 107, 5), dtype=np.float64)
# position-wise train mean for every column/position
posmean = np.zeros((107, 5))
for c in range(5):
    for p in range(107):
        posmean[p, c] = tr['y'][:, p, c][tr['mask'][:, p] > 0].mean() if (tr['mask'][:, p] > 0).any() else 0.0
for p in range(107):
    full[:, p, :] = posmean[p]
for k, c in enumerate(SCORE_IDX):
    full[:, :68, c] = test_pred[:, :, k]
write_submission(full, te['ids'], os.path.join(WS, 'submission_greedy.csv'))
print('wrote submission_greedy.csv')
