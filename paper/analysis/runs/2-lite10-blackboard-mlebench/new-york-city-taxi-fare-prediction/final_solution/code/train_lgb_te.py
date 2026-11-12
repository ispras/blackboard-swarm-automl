import os, sys, time, argparse, json
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data_io as D
import lightgbm as lgb

BASE = os.path.dirname(os.path.abspath(__file__))
FI = {c: i for i, c in enumerate(D.FEATURES)}


def cell_keys(lat, lon, dec=2):
    m = 10 ** dec
    rlat = np.rint(lat * m).astype(np.int64)
    rlon = np.rint(lon * m).astype(np.int64)
    return rlat * (10 ** (dec + 4)) + rlon


def route_keys(pu, do):
    u = (pu.astype(np.uint64) * np.uint64(1000003)) ^ (do.astype(np.uint64) + np.uint64(12345))
    return u


def te_oof(keys, y, prior, alpha=50.0, n_folds=5, seed=7, log=print):
    n = len(keys)
    fold = np.random.default_rng(seed).integers(0, n_folds, n)
    df = pd.DataFrame({'k': keys, 'y': y.astype(np.float64), 'f': fold})
    tot = df.groupby('k')['y'].agg(['sum', 'count'])
    per = df.groupby(['k', 'f'])['y'].agg(['sum', 'count'])
    per = per.reset_index()
    d = df.merge(tot, left_on='k', right_index=True, how='left')
    d = d.merge(per, on=['k', 'f'], how='left', suffixes=('', '_fold'))
    os_sum = d['sum'].to_numpy() - d['sum_fold'].to_numpy()
    os_cnt = d['count'].to_numpy() - d['count_fold'].to_numpy()
    te = (os_sum + prior * alpha) / (os_cnt + alpha)
    return te.astype(np.float32), tot


def te_full(keys, tot, prior, alpha=50.0):
    s = tot['sum'].reindex(keys).to_numpy()
    c = tot['count'].reindex(keys).to_numpy()
    s = np.where(np.isnan(s), 0.0, s)
    c = np.where(np.isnan(c), 0.0, c)
    return ((s + prior * alpha) / (c + alpha)).astype(np.float32)


def add_te(X, coords, y, prior, alpha, train=True, totbags=None, log=print):
    plat = X[:, FI['pickup_latitude']]
    plon = X[:, FI['pickup_longitude']]
    dlat = X[:, FI['dropoff_latitude']]
    dlon = X[:, FI['dropoff_longitude']]
    pu = cell_keys(plat, plon)
    do = cell_keys(dlat, dlon)
    rt = route_keys(pu, do)
    if train:
        pu_te, tot_pu = te_oof(pu, y, prior, alpha)
        do_te, tot_do = te_oof(do, y, prior, alpha)
        rt_te, tot_rt = te_oof(rt, y, prior, alpha)
        totbags = (tot_pu, tot_do, tot_rt)
        feats = [
            pu_te, do_te, rt_te,
            np.log1p(tot_pu['count'].reindex(pu).to_numpy()).astype(np.float32),
            np.log1p(tot_do['count'].reindex(do).to_numpy()).astype(np.float32),
            np.log1p(tot_rt['count'].reindex(rt).to_numpy()).astype(np.float32),
        ]
    else:
        tot_pu, tot_do, tot_rt = totbags
        pu_te = te_full(pu, tot_pu, prior, alpha)
        do_te = te_full(do, tot_do, prior, alpha)
        rt_te = te_full(rt, tot_rt, prior, alpha)
        feats = [
            pu_te, do_te, rt_te,
            np.log1p(tot_pu['count'].reindex(pu).fillna(0).to_numpy()).astype(np.float32),
            np.log1p(tot_do['count'].reindex(do).fillna(0).to_numpy()).astype(np.float32),
            np.log1p(tot_rt['count'].reindex(rt).fillna(0).to_numpy()).astype(np.float32),
        ]
    TE_NAMES = ['pu_te', 'do_te', 'rt_te', 'pu_cnt', 'do_cnt', 'rt_cnt']
    Xa = np.concatenate([X, np.column_stack(feats).astype(np.float32)], axis=1)
    return Xa, TE_NAMES, totbags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frac', type=float, default=0.25)
    ap.add_argument('--leaves', type=int, default=384)
    ap.add_argument('--lr', type=float, default=0.05)
    ap.add_argument('--rounds', type=int, default=2000)
    ap.add_argument('--threads', type=int, default=16)
    ap.add_argument('--alpha', type=float, default=50.0)
    ap.add_argument('--tag', type=str, default='lgb_te')
    args = ap.parse_args()

    t0 = time.time()
    X, y, Xh, yh, hv = D.load(frac=args.frac)
    prior = float(y.mean())
    print('prior', prior, flush=True)
    Xa, te_names, bags = add_te(X, None, y, prior, args.alpha, train=True)
    del X
    Xha, _, _ = add_te(Xh, None, yh, prior, args.alpha, train=False, totbags=bags)
    names = D.FEATURES_ALL + te_names
    print('augmented', Xa.shape, flush=True)

    trim = hv & (yh >= 0.5) & (yh <= 100)
    params = dict(objective='regression', metric='rmse', learning_rate=args.lr,
                  num_leaves=args.leaves, min_data_in_leaf=300, feature_fraction=0.85,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, max_bin=255,
                  num_threads=args.threads, verbose=-1, seed=42)
    dtrain = lgb.Dataset(Xa, label=y, feature_name=names, free_raw_data=True)
    dvalid = lgb.Dataset(Xha[trim], label=yh[trim], reference=dtrain, free_raw_data=True)
    model = lgb.train(params, dtrain, num_boost_round=args.rounds,
                      valid_sets=[dvalid], valid_names=['hold_trim'],
                      callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)])
    model.save_model(os.path.join(BASE, f'{args.tag}.txt'))
    ph = model.predict(Xha, num_iteration=model.best_iteration)
    r_v = D.rmse(yh[hv], ph[hv]); r_all = D.rmse(yh, ph); r_trim = D.rmse(yh[trim], ph[trim])
    print(f'TE HOLDOUT valid={r_v:.5f} all={r_all:.5f} trim={r_trim:.5f} best_iter={model.best_iteration}', flush=True)
    keys = pd.read_csv(D.HOLD_KEYS)['key'].values
    pd.DataFrame({'key': keys, 'fare_amount': yh, 'pred': ph}).to_csv(
        os.path.join(BASE, f'{args.tag}_holdout_pred.csv'), index=False)

    tkeys, Xt = D.load_test()
    Xta, _, _ = add_te(Xt, None, np.zeros(len(Xt)), prior, args.alpha, train=False, totbags=bags)
    pt = model.predict(Xta, num_iteration=model.best_iteration)
    pd.DataFrame({'key': tkeys, 'fare_amount': pt}).to_csv(
        os.path.join(BASE, f'{args.tag}_submission.csv'), index=False)
    json.dump({'tag': args.tag, 'frac': args.frac, 'best_iter': int(model.best_iteration),
               'holdout_rmse_valid': r_v, 'holdout_rmse_all': r_all, 'holdout_rmse_trim': r_trim,
               'leaves': args.leaves, 'lr': args.lr, 'alpha': args.alpha, 'elapsed': time.time()-t0},
              open(os.path.join(BASE, f'{args.tag}_meta.json'), 'w'), indent=2)
    print('DONE', time.time()-t0)


if __name__ == '__main__':
    main()
