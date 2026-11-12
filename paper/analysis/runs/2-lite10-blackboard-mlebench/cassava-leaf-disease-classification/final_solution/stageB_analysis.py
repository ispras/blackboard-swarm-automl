import os, sys, itertools, json, argparse
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
sys.path.insert(0, os.path.dirname(__file__))
from final_blend import (norm, acc, logloss, fit_weights, greedy_accuracy,
                         class_bias, fit_temps, temper, P)

MEM = os.path.join(os.path.dirname(__file__), 'members')


def load_oof(name):
    d = pd.read_csv(os.path.join(MEM, name))
    return d.set_index('image_id')


def load_e1(path):
    d = pd.read_csv(path)
    return d.set_index('image_id')


def cv_strategies(V, y, seed):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    out = {k: [] for k in ['equal', 'logloss', 'logloss+bias', 'greedy', 'temp+bias']}
    for tr, va in skf.split(np.zeros(len(y)), y):
        Vtr, Vva = V[:, tr], V[:, va]
        ytr, yva = y[tr], y[va]
        M = V.shape[0]
        for k in out:
            if k == 'equal':
                w = np.ones(M) / M; T = np.ones(M); b = np.zeros(5)
            elif k == 'greedy':
                w = greedy_accuracy(Vtr, ytr, rounds=20); T = np.ones(M); b = np.zeros(5)
            elif k == 'temp+bias':
                T = fit_temps(Vtr, ytr)
                Vtr2, Vva2 = temper(Vtr, T), temper(Vva, T)
                w, _ = fit_weights(Vtr2, ytr, n_restart=4, seed=seed)
                ptr = norm(np.tensordot(w, Vtr2, axes=(0, 0)))
                b = class_bias(ptr, ytr)
                p = norm(np.tensordot(w, Vva2, axes=(0, 0)) * np.exp(b)[None, :])
                out[k].append(acc(p, yva)); continue
            else:
                w, _ = fit_weights(Vtr, ytr, n_restart=4, seed=seed)
                T = np.ones(M); b = np.zeros(5)
            Vtr2, Vva2 = Vtr, Vva
            p = norm(np.tensordot(w, Vva2, axes=(0, 0)))
            if k == 'logloss+bias':
                ptr = norm(np.tensordot(w, Vtr2, axes=(0, 0)))
                b = class_bias(ptr, ytr)
                p = norm(p * np.exp(b)[None, :])
            out[k].append(acc(p, yva))
    return {k: (float(np.mean(v)), float(np.std(v))) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--e1', default='blend_v2/E1_oof.csv')
    a = ap.parse_args()
    E1 = load_e1(a.e1)
    singles = {}
    for n in [x for x in ['bellatrix_swin_oof.csv', 'bellatrix_swin_s123_oof.csv', 'bellatrix_eva02_oof.csv', 'bellatrix_b0_oof.csv', 'altair_effml_oof.csv', 'altair_cnb_oof.csv'] if os.path.exists(os.path.join(MEM, x))]:
        singles[n.replace('_oof.csv', '')] = load_oof(n)
    idx = E1.index
    for d in singles.values():
        idx = idx.intersection(d.index)
    idx = idx.sort_values()
    y = E1.loc[idx, 'label'].values.astype(int)
    base = {'E1': E1.loc[idx, P].values}
    for n, d in singles.items():
        base[n] = d.loc[idx, P].values
    subsets = [
        ['E1', 'bellatrix_swin', 'bellatrix_swin_s123', 'altair_effml'],
        ['E1', 'bellatrix_swin', 'bellatrix_swin_s123', 'altair_effml', 'altair_cnb'],
        ['E1', 'bellatrix_swin', 'bellatrix_swin_s123', 'altair_effml', 'altair_cnb', 'bellatrix_b0'],
        ['E1', 'bellatrix_swin', 'bellatrix_swin_s123', 'altair_cnb'],
        ['E1', 'bellatrix_swin', 'bellatrix_swin_s123', 'altair_effml', 'altair_cnb', 'bellatrix_eva02'],
    ]
    for sub in subsets:
        V = np.stack([norm(base[n]) for n in sub], 0)
        agg = {}
        for seed in [1, 2, 3]:
            r = cv_strategies(V, y, seed * 100 + 7)
            for k, v in r.items():
                agg.setdefault(k, []).append(v[0])
        print('\n==', sub, 'n=', len(y))
        for k, v in sorted(agg.items(), key=lambda kv: -np.mean(kv[1])):
            print(f'  {k:14s} {np.mean(v):.4f} +- {np.std(v):.4f}')


if __name__ == '__main__':
    main()
