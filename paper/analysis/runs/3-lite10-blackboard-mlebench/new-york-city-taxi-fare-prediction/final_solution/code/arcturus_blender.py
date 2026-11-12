import os, sys, json
import numpy as np, pandas as pd
from scipy.optimize import nnls

BASE = '/download_job_code/payload/sandbox'
HOLDOUT = f'{BASE}/workspaces/arcturus_upsilon/runs/holdout_flags.parquet'
TESTFLAGS = f'{BASE}/workspaces/arcturus_upsilon/cache/test_v2.parquet'

CAND = {
    'lgb20m':  ('workspaces/arcturus_upsilon/runs/lgb20m_1023_val.parquet',
                'workspaces/arcturus_upsilon/runs/lgb20m_1023_test.parquet'),
    'xgb_v2':  ('workspaces/antares_omicron/runs/xgb_v2_val.parquet',
                'workspaces/antares_omicron/runs/xgb_v2_test.parquet'),
    'xgb_v3':  ('workspaces/antares_omicron/runs/xgb_v3_val.parquet',
                'workspaces/antares_omicron/runs/xgb_v3_test.parquet'),
    'cat':     ('workspaces/antares_omicron/runs/cat_val.parquet',
                'workspaces/antares_omicron/runs/cat_test.parquet'),
    'alkaid_v2lgb': ('workspaces/alkaid_phi/v2lgb_val.parquet',
                     'workspaces/alkaid_phi/v2lgb_submission.csv'),
    'alkaid_lgb': ('workspaces/alkaid_phi/v2lgb_val.parquet',
                   'workspaces/alkaid_phi/v2lgb_submission.csv'),
    'alkaid_te16': ('workspaces/alkaid_phi/te16m_val.parquet',
                    'workspaces/alkaid_phi/te16m_test.parquet'),
    'alkaid_full': ('workspaces/alkaid_phi/full16m_val.parquet',
                    'workspaces/alkaid_phi/full16m_test.parquet'),
    'lgb30m':  ('workspaces/arcturus_upsilon/runs/lgb30m_val.parquet',
                'workspaces/arcturus_upsilon/runs/lgb30m_test.parquet'),
}

def norm_key(s):
    s = pd.Series(s).astype(str)
    out = np.empty(len(s), dtype=object)
    for i, k in enumerate(s.to_numpy()):
        if '.' in k:
            b, f = k.rsplit('.', 1)
            out[i] = b + '.' + f.ljust(9, '0')[:9]
        else:
            out[i] = k
    return out

def norm_df(df):
    df = df.copy()
    df['key'] = norm_key(df['key'])
    return df

def load_pred(path, col_candidates=('pred', 'fare_amount', 'y_pred', 'prediction')):
    if path.endswith('.csv'):
        df = pd.read_csv(path)
    else:
        df = pd.read_parquet(path)
    if 'key' not in df.columns:
        raise ValueError(f'{path} has no key: {list(df.columns)}')
    pc = next((c for c in col_candidates if c in df.columns), None)
    if pc is None:
        raise ValueError(f'{path} no pred col: {list(df.columns)}')
    df = norm_df(df[['key', pc]]).rename(columns={pc: 'pred'})
    df = df.drop_duplicates('key')
    return df

def fit(X, y):
    A = np.column_stack([X, np.ones(len(y))])
    w, _ = nnls(A, y)
    return w[:X.shape[1]], float(w[-1])

def apply(X, w, b):
    return X @ w + b

def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))

def main(out=None, use=None, min_cov=0.90):
    hol = norm_df(pd.read_parquet(HOLDOUT)).rename(columns={'fare_amount': 'y'})
    hol = hol.drop_duplicates('key')
    models = {}
    for name, (vp, tp) in CAND.items():
        if use and name not in use.split(','):
            continue
        vp_abs = os.path.join(BASE, vp)
        if not os.path.exists(vp_abs):
            continue
        try:
            v = load_pred(vp_abs)
        except Exception as e:
            print('skip', name, e); continue
        models[name] = {'val': v, 'test': tp}
        print('loaded', name, len(v))
    if not models:
        print('no models'); return
    df = hol.merge(pd.DataFrame({'key': hol.key}), on='key')
    for name, m in models.items():
        df = df.merge(m['val'].rename(columns={'pred': name}), on='key', how='left')
    df = df.set_index('key')
    miss = ((df.has_pickup == 0) | (df.has_dropoff == 0)).to_numpy()
    dist = df['distance_km'].to_numpy()
    degen = (~miss) & (dist < 0.1)
    clean = (~miss) & (dist >= 0.1)
    masks = {'clean': clean, 'degen': degen, 'miss': miss}

    print('\ncoverage / per-model RMSE:')
    chosens = {}
    for sub, mask in masks.items():
        n = int(mask.sum())
        if n == 0:
            chosens[sub] = []; continue
        cov = {name: float(df[name].to_numpy()[mask].astype(float).__len__() and np.isfinite(df[name].to_numpy()[mask]).mean()) for name in models}
        chosen = [name for name in models if cov[name] >= min_cov]
        print(f'[{sub}] n={n} coverage:', {k: round(v, 3) for k, v in cov.items()}, '-> use', chosen)
        for name in models:
            col = df[name].to_numpy()[mask]
            ok = np.isfinite(col)
            if ok.sum() > 0:
                print(f'   {name:12s} rmse={rmse(col[ok], df.y.to_numpy()[mask][ok]):.4f}')
        chosens[sub] = chosen

    # fit per subset on rows where all chosen are finite
    params = {}
    for sub, mask in masks.items():
        chosen = chosens[sub]
        if not chosen:
            continue
        finite = mask & np.isfinite(df[chosen].to_numpy()).all(1)
        X = df.loc[finite, chosen].to_numpy(float); y = df.loc[finite, 'y'].to_numpy(float)
        w, b = fit(X, y)
        p = apply(X, w, b)
        print(f'[{sub}] fit n={len(y)} blend_rmse={rmse(p, y):.4f}')
        for n_, w_ in zip(chosen, w): print(f'   w {n_:12s} {w_:.4f}')
        print(f'   intercept {b:.4f}')
        params[sub] = {'models': chosen, 'w': w.tolist(), 'b': b, 'n_fit': len(y), 'rmse': rmse(p, y)}
    # overall val blend on rows where subset param exists
    allpred = np.full(len(df), np.nan)
    for sub, mask in masks.items():
        if sub not in params: continue
        chosen = params[sub]['models']
        finite = mask & np.isfinite(df[chosen].to_numpy()).all(1)
        allpred[finite] = apply(df.loc[finite, chosen].to_numpy(float), np.array(params[sub]['w']), params[sub]['b'])
    ok = np.isfinite(allpred)
    print(f'\nVAL BLEND (covered {ok.sum()}/{len(df)}) rmse={rmse(allpred[ok], df.y.to_numpy()[ok]):.4f}')

    # TEST
    tf = norm_df(pd.read_parquet(TESTFLAGS)[['key', 'has_pickup', 'has_dropoff', 'distance_km']]).drop_duplicates('key')
    tb = tf.set_index('key')
    tmiss = ((tb.has_pickup == 0) | (tb.has_dropoff == 0)).to_numpy()
    tdist = tb['distance_km'].to_numpy()
    t_degen = (~tmiss) & (tdist < 0.1)
    t_clean = (~tmiss) & (tdist >= 0.1)
    test_cols = {}
    for name, m in models.items():
        tp = os.path.join(BASE, m['test'])
        if not os.path.exists(tp):
            print('no test pred:', name); continue
        try:
            t = load_pred(tp).set_index('key')
        except Exception as e:
            print('skip test', name, e); continue
        test_cols[name] = tb.join(t, how='left')['pred'].to_numpy()
    print('test models with preds:', list(test_cols))
    pt = np.full(len(tb), np.nan)
    for sub, mask in {'clean': t_clean, 'degen': t_degen, 'miss': tmiss}.items():
        if sub not in params: continue
        chosen = [n for n in params[sub]['models'] if n in test_cols]
        if not chosen:
            continue
        X = np.column_stack([test_cols[n] for n in chosen])
        # if some are nan, fall back to row mean of available
        fin = np.isfinite(X).all(1)
        w = np.array([params[sub]['w'][params[sub]['models'].index(n)] for n in chosen])
        b = params[sub]['b']
        pred = np.full(len(tb), np.nan)
        pred[fin] = X[fin] @ w + b
        # fallback rows: simple average of chosen available
        if (~fin).any():
            pred[~fin] = np.nanmean(X[~fin], axis=1)
        pt[mask] = pred[mask]
    # any remaining nan -> global median
    if np.isnan(pt).any():
        print('WARN nan test preds', int(np.isnan(pt).sum()))
        pt[np.isnan(pt)] = np.nanmedian(pt)
    pt = np.clip(pt, 2.5, 200.0)
    out = out or f'{BASE}/workspaces/arcturus_upsilon/runs/blend_submission.csv'
    pd.DataFrame({'key': tf.key.to_numpy(), 'fare_amount': pt}).to_csv(out, index=False)
    print('wrote', out, 'stats min/med/max/mean', float(pt.min()), float(np.median(pt)), float(pt.max()), float(pt.mean()))
    json.dump({'params': params, 'val_covered': int(ok.sum()), 'val_rmse': rmse(allpred[ok], df.y.to_numpy()[ok])},
              open(f'{BASE}/workspaces/arcturus_upsilon/runs/blend_meta.json', 'w'), indent=1)

if __name__ == '__main__':
    out = sys.argv[1] if len(sys.argv) > 1 else None
    use = sys.argv[2] if len(sys.argv) > 2 else None
    main(out, use)
