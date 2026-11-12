"""
Ensemble combiner for Cassava. Fits blend weights on the shared fold-0 holdout
(workspaces/shared/folds5_seed42.csv) and writes final_solution/submission.csv.

Usage:
  python ensemble.py --models name1:val1.npy:test1.npy name2:val2.npy:test2.npy ...
Optional: --val_idx name1:idx1.npy ... if val rows are not already in fold-0 order.

val_*.npy : (n_fold0, 5) probabilities, rows aligned to folds5 fold==0 order
test_*.npy: (2676, 5) probabilities, rows aligned to sample_submission order
"""
import argparse, os, json
import numpy as np
import pandas as pd
from itertools import product


def acc(p, y):
    return (p.argmax(1) == y).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', required=True,
                    help='name:val_npy:test_npy')
    ap.add_argument('--split', default='workspaces/shared/folds5_seed42.csv')
    ap.add_argument('--fold', type=int, default=0)
    ap.add_argument('--mode', default='arith', choices=['arith', 'geom', 'logit'])
    ap.add_argument('--grid', type=int, default=11, help='weight grid resolution')
    ap.add_argument('--out', default='final_solution/submission.csv')
    a = ap.parse_args()

    df = pd.read_csv(a.split)
    y = df[df.fold == a.fold].label.values
    test = pd.read_csv('data/sample_submission.csv')

    V, T, names = [], [], []
    for spec in a.models:
        n, vp, tp = spec.split(':')
        v = np.load(vp); t = np.load(tp)
        assert v.shape == (len(y), 5), (n, v.shape, len(y))
        assert t.shape == (len(test), 5), (n, t.shape)
        V.append(v); T.append(t); names.append(n)

    for n, v in zip(names, V):
        print(f'{n:28s} val_acc {acc(v, y):.4f}')

    if a.mode == 'geom':
        Vt = [np.log(np.clip(v, 1e-9, 1)) for v in V]
        Tt = [np.log(np.clip(t, 1e-9, 1)) for t in T]
    elif a.mode == 'logit':
        Vt = [np.log(np.clip(v, 1e-9, 1)) - np.log(np.clip(1 - v, 1e-9, 1)) for v in V]
        Tt = [np.log(np.clip(t, 1e-9, 1)) - np.log(np.clip(1 - t, 1e-9, 1)) for t in T]
    else:
        Vt, Tt = V, T

    k = len(V)
    best = (-1, None)
    g = np.linspace(0, 1, a.grid)
    # coarse grid over simplex (only for small k)
    if k <= 3:
        combos = [c for c in product(g, repeat=k) if abs(sum(c) - 1) < 1e-6]
    else:
        combos = [tuple(np.ones(k) / k)]
    for w in combos:
        p = sum(wi * vi for wi, vi in zip(w, Vt))
        sc = acc(p, y)
        if sc > best[0]:
            best = (sc, w)
    print('best weights', np.round(best[1], 3), 'val_acc', round(best[0], 4))

    w = np.array(best[1])
    pt = sum(wi * ti for wi, ti in zip(w, Tt))
    pred = pt.argmax(1)
    out = pd.DataFrame({'image_id': test.image_id, 'label': pred})
    out.to_csv(a.out, index=False)
    print('wrote', a.out, out.shape, out.label.value_counts().to_dict())
    np.save('workspaces/capella_kappa/ensemble_weights.npy', w)
    with open('workspaces/capella_kappa/ensemble_info.json', 'w') as f:
        json.dump({'names': names, 'weights': list(map(float, w)),
                   'val_acc': float(best[0])}, f, indent=2)


if __name__ == '__main__':
    main()
