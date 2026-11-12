import os, sys, json, argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import postprocess as PP

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(BASE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'workspaces', 'dubhe_tau'))

CAND = {
    'lgb_big': (f'{BASE}/lgb_big_holdout_pred.csv', f'{BASE}/lgb_big_submission.csv'),
    'cat_gpu': (f'{BASE}/cat_gpu_holdout_pred.csv', f'{BASE}/cat_gpu_submission.csv'),
    'lgb_te2': (f'{BASE}/lgb_te2_holdout_pred.csv', f'{BASE}/lgb_te2_submission.csv'),
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
}
META_COLS = ['haversine', 'log_hav', 'hour', 'weekday', 'year', 'airport_min', 'is_airport', 'passenger_count']
RAW_META = ['haversine', 'hour', 'weekday', 'year', 'airport_min', 'is_airport', 'passenger_count']

def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2)))

def load_all(names):
    keys = pd.read_csv(f'{BASE}/holdout_keys.csv')['key'].values
    hf = pd.read_parquet(f'{BASE}/holdout_feats.parquet')
    y = hf['fare_amount'].to_numpy(np.float64)
    clean = ~PP.invalid_coord_mask(hf)
    H = pd.DataFrame({'key': keys, 'y': y, 'clean': clean})
    for c in RAW_META:
        H[c] = hf[c].to_numpy()
    H['log_hav'] = np.log1p(H['haversine'])
    T = None
    for n in names:
        hp, tp = CAND[n]
        h = pd.read_csv(hp); pcol = 'pred' if 'pred' in h.columns else 'fare_amount'
        H = H.merge(h[['key', pcol]].rename(columns={pcol: n}), on='key', how='left')
        t = pd.read_csv(tp); tcol = 'pred' if 'pred' in t.columns else 'fare_amount'
        t = t[['key', tcol]].rename(columns={tcol: n})
        T = t if T is None else T.merge(t, on='key', how='left')
    return H, T, names

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', type=str, default='lgb_big,xgb_aug,xgb_full,xgb_b512,cat_gpu,nn_emb')
    ap.add_argument('--with_meta', action='store_true')
    ap.add_argument('--out', type=str, default='')
    ap.add_argument('--alpha', type=float, default=1.0)
    args = ap.parse_args()
    names = args.models.split(',')
    H, T, names = load_all(names)
    avail = H[names].notna().all(axis=1)
    fit = avail & H['clean'].to_numpy()
    print('fit rows', int(fit.sum()))

    base_cols = names + (META_COLS if args.with_meta else [])
    X = H.loc[fit, base_cols].to_numpy(np.float64)
    y = H.loc[fit, 'y'].to_numpy(np.float64)
    kf = KFold(n_splits=5, shuffle=True, random_state=7)
    oof = np.zeros(len(y))
    for tr, va in kf.split(X):
        m = Ridge(alpha=args.alpha, positive=False).fit(X[tr], y[tr])
        oof[va] = m.predict(X[va])
    print(f'STACK clean CV RMSE = {rmse(y, oof):.5f}  (with_meta={args.with_meta})')
    # compare plain NNLS in same 5-fold
    from scipy.optimize import nnls
    A = H.loc[fit, names].to_numpy(np.float64)
    oof2 = np.zeros(len(y))
    for tr, va in kf.split(A):
        w, _ = nnls(A[tr], y[tr]); oof2[va] = A[va] @ w
    print(f'NNLS  clean CV RMSE = {rmse(y, oof2):.5f}')

    if args.out:
        m = Ridge(alpha=args.alpha).fit(X, y)
        base_cols_t = names + (META_COLS if args.with_meta else [])
        Xt = T[base_cols_t].to_numpy(np.float64)
        p = m.predict(Xt)
        pd.DataFrame({'key': T['key'], 'fare_amount': p}).to_csv(args.out + '_raw.csv', index=False)
        import postprocess_final as DPP
        test_df = pd.read_csv(f'{ROOT}/data/test.csv')
        pp = DPP.apply_postprocess(p, test_df)
        pd.DataFrame({'key': test_df['key'], 'fare_amount': pp}).to_csv(args.out, index=False)
        print('wrote', args.out, 'pred quantiles', np.quantile(pp, [0, .5, 1]))

if __name__ == '__main__':
    main()
