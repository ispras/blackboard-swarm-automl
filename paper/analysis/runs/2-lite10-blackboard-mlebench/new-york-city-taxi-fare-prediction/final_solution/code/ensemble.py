import os, sys, json, argparse
import numpy as np
import pandas as pd
from scipy.optimize import nnls
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import postprocess as PP

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(BASE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'workspaces', 'dubhe_tau'))

CAND = {
    'lgb_big': (f'{BASE}/lgb_big_holdout_pred.csv', f'{BASE}/lgb_big_submission.csv'),
    'lgb_te': (f'{BASE}/lgb_te_holdout_pred.csv', f'{BASE}/lgb_te_submission.csv'),
    'cat_gpu': (f'{BASE}/cat_gpu_holdout_pred.csv', f'{BASE}/cat_gpu_submission.csv'),
    'xgb_aug': (f'{ROOT}/workspaces/dubhe_tau/models/xgb_aug_d13_holdout_pred.csv',
                f'{ROOT}/workspaces/dubhe_tau/models/xgb_aug_d13_test.csv'),
    'xgb_full': (f'{ROOT}/workspaces/dubhe_tau/models/xgb_full_d13_holdout_pred.csv',
                 f'{ROOT}/workspaces/dubhe_tau/models/xgb_full_d13_test.csv'),
    'xgb_b512': (f'{ROOT}/workspaces/dubhe_tau/models/xgb_aug_b512_holdout_pred.csv',
                 f'{ROOT}/workspaces/dubhe_tau/models/xgb_aug_b512_test.csv'),
    'nn_emb': (f'{ROOT}/workspaces/dubhe_tau/models/nn_emb_holdout_pred.csv',
               f'{ROOT}/workspaces/dubhe_tau/models/nn_emb_test.csv'),
    'xgb_te': (f'{ROOT}/workspaces/dubhe_tau/models/xgb_te_holdout_pred.csv',
                f'{ROOT}/workspaces/dubhe_tau/models/xgb_te_test.csv'),
    'nn_te': (f'{ROOT}/workspaces/dubhe_tau/models/nn_te_holdout_pred.csv',
              f'{ROOT}/workspaces/dubhe_tau/models/nn_te_test.csv'),

    'lgb_lin': (f'{BASE}/lgb_lin_clip_holdout_pred.csv', f'{BASE}/lgb_lin_clip_submission.csv'),
}

def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(p, float))**2)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', type=str, default='')
    ap.add_argument('--out', type=str, default='')
    ap.add_argument('--weight', type=str, default='nnls')
    args = ap.parse_args()
    if args.models:
        names = args.models.split(',')
    else:
        names = [n for n in CAND if os.path.exists(CAND[n][0]) and os.path.exists(CAND[n][1])]
    print('models:', names)

    keys = pd.read_csv(f'{BASE}/holdout_keys.csv')['key'].values
    hf = pd.read_parquet(f'{BASE}/holdout_feats.parquet')
    y = hf['fare_amount'].to_numpy(np.float64)
    clean = ~PP.invalid_coord_mask(hf)

    H = pd.DataFrame({'key': keys, 'y': y, 'clean': clean})
    T = None
    for n in names:
        hp, tp = CAND[n]
        h = pd.read_csv(hp)
        pcol = 'pred' if 'pred' in h.columns else 'fare_amount'
        H = H.merge(h[['key', pcol]].rename(columns={pcol: n}), on='key', how='left')
        t = pd.read_csv(tp)
        tcol = 'pred' if 'pred' in t.columns else 'fare_amount'
        t = t[['key', tcol]].rename(columns={tcol: n})
        T = t if T is None else T.merge(t, on='key', how='left')

    avail = H[names].notna().all(axis=1)
    fitm = avail & clean
    print('fit rows (all models, clean):', int(fitm.sum()))
    A = H.loc[fitm, names].to_numpy(np.float64)
    b = H.loc[fitm, 'y'].to_numpy(np.float64)
    for i, n in enumerate(names):
        print(f'  {n:10s} clean RMSE={rmse(b, A[:, i]):.5f}')
    if args.weight == 'nnls':
        w, _ = nnls(A, b)
    else:
        w = np.ones(len(names)) / len(names)
    print('weights', dict(zip(names, np.round(w, 4))), 'sum', w.sum())

    blend_fit = A @ w
    print(f'BLEND clean RMSE (fit rows) = {rmse(b, blend_fit):.5f}')
    # evaluate on all available clean rows
    am = avail & clean
    print(f'BLEND clean RMSE (all avail) = {rmse(H.loc[am,"y"], H.loc[am,names].to_numpy()@w):.5f}')
    # simple average comparison
    print(f'MEAN  clean RMSE (all avail) = {rmse(H.loc[am,"y"], H.loc[am,names].to_numpy().mean(1)):.5f}')

    # full postprocessed holdout metric
    raw_full = H[names].to_numpy(np.float64) @ w
    valid_pred = ~np.isnan(raw_full)
    fill = np.nanmean(raw_full[valid_pred])
    raw_full = np.where(np.isnan(raw_full), fill, raw_full)
    try:
        import postprocess_final as DPP
        pp = DPP.apply_postprocess(raw_full, hf, verbose=False)
        print(f'BLEND full-holdout after dubhe postprocess = {rmse(y, pp):.5f}')
    except Exception as e:
        print('dubhe postprocess failed:', e)

    if args.out:
        tw = T[names].to_numpy(np.float64) @ w
        pd.DataFrame({'key': T['key'], 'fare_amount': tw}).to_csv(args.out + '_raw.csv', index=False)
        import postprocess_final as DPP
        test_df = pd.read_csv(f'{ROOT}/data/test.csv')
        pp = DPP.apply_postprocess(tw, test_df)
        pd.DataFrame({'key': test_df['key'], 'fare_amount': pp}).to_csv(args.out, index=False)
        print('wrote', args.out)
    json.dump({'models': names, 'weights': list(map(float, w))},
              open(f'{BASE}/ensemble_weights.json', 'w'), indent=2)

if __name__ == '__main__':
    main()
