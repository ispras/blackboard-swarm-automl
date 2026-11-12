import argparse, time, os, json
import numpy as np, pandas as pd
import pyarrow.parquet as pq
import lightgbm as lgb

CACHE = 'workspaces/arcturus_upsilon/cache'
META = {'key', 'fare_amount', 'holdout'}

def load_sample(n_train, n_val, feats=None, cache='train_v2'):
    pf = pq.ParquetFile(f'{CACHE}/{cache}.parquet')
    all_cols = pf.schema.names
    if feats is None:
        feats = [c for c in all_cols if c not in META]
    cols = feats + ['fare_amount', 'holdout', 'key']
    k = len(feats)
    X = np.empty((n_train, k), np.float32)
    y = np.empty(n_train, np.float32)
    Vs, Ks = [], []
    nt = nv = 0
    t0 = time.time()
    for bi, batch in enumerate(pf.iter_batches(batch_size=2_000_000, columns=cols)):
        d = batch.to_pandas()
        ho = d['holdout'].to_numpy()
        Xb = d[feats].to_numpy(np.float32)
        yb = d['fare_amount'].to_numpy(np.float32)
        keys = d['key'].to_numpy()
        del d
        if nv < n_val:
            vm = np.flatnonzero(ho)[:n_val - nv]
            if len(vm):
                Vs.append(np.column_stack([yb[vm], Xb[vm]])); Ks.append(keys[vm]); nv += len(vm)
        if nt < n_train:
            tm = np.flatnonzero(~ho)[:n_train - nt]
            if len(tm):
                X[nt:nt + len(tm)] = Xb[tm]; y[nt:nt + len(tm)] = yb[tm]; nt += len(tm)
        del Xb, yb, keys
        print(f'batch {bi}: train={nt} val={nv} t={time.time()-t0:.0f}s', flush=True)
        if nt >= n_train and nv >= n_val:
            break
    V = np.concatenate(Vs) if Vs else np.empty((0, k + 1), np.float32)
    K = np.concatenate(Ks) if Ks else np.empty(0, object)
    return X[:nt], y[:nt], V[:, 0], V[:, 1:], feats, K

def cell2d(cols, feats, lon_name, lat_name):
    li = feats.index(lon_name); ai = feats.index(lat_name)
    lon = cols[:, li]; lat = cols[:, ai]
    ok = np.isfinite(lon) & np.isfinite(lat)
    lonr = np.where(ok, np.round(lon * 100), 0).astype(np.int64)
    latr = np.where(ok, np.round(lat * 100), 0).astype(np.int64)
    cell = (lonr + 7600) * 10000 + (latr - 3000)
    cell[~ok] = -1
    return cell

def te_features(Xtr, ytr, Xva, Xte, feats, n_folds=5, seed=7, smooth_p=20.0, smooth_r=10.0):
    t0 = time.time()
    pc_tr = cell2d(Xtr, feats, 'pickup_longitude', 'pickup_latitude')
    dc_tr = cell2d(Xtr, feats, 'dropoff_longitude', 'dropoff_latitude')
    rt_tr = pc_tr * 1_000_000 + dc_tr
    pc_va = cell2d(Xva, feats, 'pickup_longitude', 'pickup_latitude')
    dc_va = cell2d(Xva, feats, 'dropoff_longitude', 'dropoff_latitude')
    rt_va = pc_va * 1_000_000 + dc_va
    pc_te = cell2d(Xte, feats, 'pickup_longitude', 'pickup_latitude')
    dc_te = cell2d(Xte, feats, 'dropoff_longitude', 'dropoff_latitude')
    rt_te = pc_te * 1_000_000 + dc_te
    prior = float(np.mean(ytr))

    def build_oof(keys_tr, y, keys_va, keys_te, smooth):
        n = len(keys_tr)
        fold = np.random.default_rng(seed).integers(0, n_folds, n)
        out_tr = np.empty(n, np.float32); cnt_tr = np.empty(n, np.float32)
        out_va = np.empty(len(keys_va), np.float32); cnt_va = np.empty(len(keys_va), np.float32)
        out_te = np.empty(len(keys_te), np.float32); cnt_te = np.empty(len(keys_te), np.float32)
        for f in range(n_folds):
            m = fold != f
            s = pd.Series(y[m]).groupby(keys_tr[m]).agg(['sum', 'count'])
            gmean = float(y[m].mean())
            enc = (s['sum'] + gmean * smooth) / (s['count'] + smooth)
            idx = ~m
            v = pd.Index(keys_tr[idx]).map(enc)
            out_tr[idx] = v.fillna(gmean).to_numpy(np.float32)
            cnt_tr[idx] = pd.Index(keys_tr[idx]).map(s['count']).fillna(0).to_numpy(np.float32)
        s = pd.Series(y).groupby(keys_tr).agg(['sum', 'count'])
        gmean = prior
        enc = (s['sum'] + gmean * smooth) / (s['count'] + smooth)
        for keys, o, c in [(keys_va, out_va, cnt_va), (keys_te, out_te, cnt_te)]:
            o[:] = pd.Index(keys).map(enc).fillna(gmean).to_numpy(np.float32)
            c[:] = pd.Index(keys).map(s['count']).fillna(0).to_numpy(np.float32)
        return out_tr, np.log1p(cnt_tr).astype(np.float32), out_va, np.log1p(cnt_va).astype(np.float32), out_te, np.log1p(cnt_te).astype(np.float32)

    new_tr, new_va, new_te, names = [], [], [], []
    for keys_tr, keys_va, keys_te, tag, sm in [
        (pc_tr, pc_va, pc_te, 'te_p', smooth_p),
        (dc_tr, dc_va, dc_te, 'te_d', smooth_p),
        (rt_tr, rt_va, rt_te, 'te_r', smooth_r)]:
        a, ca, b, cb, c, cc = build_oof(keys_tr, ytr, keys_va, keys_te, sm)
        new_tr += [a, ca]; new_va += [b, cb]; new_te += [c, cc]
        names += [f'{tag}_mean', f'{tag}_logcnt']
        print(f'{tag} done t={time.time()-t0:.0f}s', flush=True)
    Xtr = np.column_stack([Xtr] + new_tr).astype(np.float32)
    Xva = np.column_stack([Xva] + new_va).astype(np.float32)
    Xte = np.column_stack([Xte] + new_te).astype(np.float32)
    return Xtr, Xva, Xte, feats + names


def add_extra(X, V, Xte, feats):
    def build(A):
        idx = {f: i for i, f in enumerate(feats)}
        pj, pl, pe = A[:, idx['pdist_jfk']], A[:, idx['pdist_lga']], A[:, idx['pdist_ewr']]
        dj, dl, de = A[:, idx['ddist_jfk']], A[:, idx['ddist_lga']], A[:, idx['ddist_ewr']]
        min_p = np.fmin(np.fmin(pj, pl), pe)
        min_d = np.fmin(np.fmin(dj, dl), de)
        jfk = np.fmax(A[:, idx['pickup_jfk']], A[:, idx['dropoff_jfk']])
        lga = np.fmax(A[:, idx['pickup_lga']], A[:, idx['dropoff_lga']])
        ewr = np.fmax(A[:, idx['pickup_ewr']], A[:, idx['dropoff_ewr']])
        dist = A[:, idx['distance_km']]; man = A[:, idx['manhattan_km']]
        eucl = np.sqrt(A[:, idx['delta_lat']]**2 + A[:, idx['delta_lon']]**2)
        ym = A[:, idx['year']] + (A[:, idx['month']] - 1) / 12.0
        dr = dist / (man + 0.01)
        near = np.fmin(min_p, min_d)
        cols = [min_p, min_d, jfk, lga, ewr, eucl, ym, dr, np.log1p(man), near,
                dist * jfk, dist * lga]
        return np.column_stack([A] + cols).astype(np.float32)
    names = ['min_air_p','min_air_d','jfk_trip','lga_trip','ewr_trip','euclid_deg','year_month',
             'dist_ratio','log_manhattan','near_air','dist_x_jfk','dist_x_lga']
    return build(X), build(V), build(Xte), feats + names

def add_trig(X, V, Xte, feats):
    hi = feats.index('hour'); mi = feats.index('minute')
    def tf(A):
        hf = A[:, hi] + A[:, mi] / 60.0
        return np.column_stack([A, np.sin(2*np.pi*hf/24), np.cos(2*np.pi*hf/24)]).astype(np.float32)
    return tf(X), tf(V), tf(Xte), feats + ['hour_sin', 'hour_cos']

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_train', type=int, default=6_000_000)
    ap.add_argument('--n_val', type=int, default=120_000)
    ap.add_argument('--leaves', type=int, default=255)
    ap.add_argument('--lr', type=float, default=0.08)
    ap.add_argument('--rounds', type=int, default=4000)
    ap.add_argument('--min_data', type=int, default=100)
    ap.add_argument('--ff', type=float, default=0.8)
    ap.add_argument('--bf', type=float, default=0.8)
    ap.add_argument('--threads', type=int, default=32)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--tag', type=str, default='te')
    ap.add_argument('--te', action='store_true')
    ap.add_argument('--trig', action='store_true')
    ap.add_argument('--extra', action='store_true')
    ap.add_argument('--cache', type=str, default='train_v2')
    ap.add_argument('--predict_test', action='store_true')
    ap.add_argument('--clip_hi', type=float, default=200.0)
    args = ap.parse_args()
    outdir = 'workspaces/arcturus_upsilon/runs'; os.makedirs(outdir, exist_ok=True)
    X, y, yv, V, feats, vkeys = load_sample(args.n_train, args.n_val, cache=args.cache)
    tkeys = None
    if args.predict_test:
        tdf = pd.read_parquet(f'{CACHE}/test_v2.parquet')
        tkeys = tdf['key'].to_numpy()
        Xte = tdf[feats].to_numpy(np.float32)
    else:
        Xte = np.zeros((1, len(feats)), np.float32)
    if args.te:
        X, V, Xte, feats = te_features(X, y, V, Xte, feats)
    if args.extra:
        X, V, Xte, feats = add_extra(X, V, Xte, feats)
    if args.trig:
        X, V, Xte, feats = add_trig(X, V, Xte, feats)
    print('loaded', X.shape, 'val', V.shape, 'feats', len(feats), flush=True)
    dtrain = lgb.Dataset(X, label=y, feature_name=feats)
    dval = lgb.Dataset(V, label=yv, reference=dtrain, feature_name=feats)
    params = dict(objective='regression', metric='rmse', num_leaves=args.leaves,
                  learning_rate=args.lr, feature_fraction=args.ff, bagging_fraction=args.bf,
                  bagging_freq=1, min_data_in_leaf=args.min_data, max_bin=255,
                  num_threads=args.threads, seed=args.seed, verbosity=-1, lambda_l2=1.0)
    t0 = time.time(); evals = {}
    model = lgb.train(params, dtrain, num_boost_round=args.rounds, valid_sets=[dval],
                      valid_names=['val'], callbacks=[lgb.early_stopping(100),
                      lgb.log_evaluation(100), lgb.record_evaluation(evals)])
    pred = model.predict(V, num_iteration=model.best_iteration)
    rmse = float(np.sqrt(np.mean((pred - yv) ** 2)))
    print(f'BEST_ITER={model.best_iteration} VAL_RMSE={rmse:.5f} time={time.time()-t0:.0f}s', flush=True)
    model.save_model(f'{outdir}/{args.tag}.txt')
    imp = sorted(zip(feats, model.feature_importance('gain')), key=lambda z: -z[1])
    json.dump({'tag': args.tag, 'rmse': rmse, 'best_iter': model.best_iteration,
               'n_train': int(X.shape[0]), 'params': params, 'importance': imp[:60]},
              open(f'{outdir}/{args.tag}_meta.json', 'w'), indent=1)
    np.save(f'{outdir}/{args.tag}_valpred.npy', pred); np.save(f'{outdir}/{args.tag}_yval.npy', yv)
    pd.DataFrame({'key': vkeys, 'pred': pred, 'y': yv}).to_parquet(f'{outdir}/{args.tag}_val.parquet', index=False)
    if args.predict_test:
        pte = model.predict(Xte, num_iteration=model.best_iteration)
        pte = np.clip(pte, 2.5, args.clip_hi)
        sub = pd.DataFrame({'key': tkeys, 'fare_amount': pte})
        sub.to_parquet(f'{outdir}/{args.tag}_test.parquet', index=False)
        sub.to_csv(f'{outdir}/{args.tag}_submission.csv', index=False)
        print('TEST pred stats', float(pte.min()), float(np.median(pte)), float(pte.max()), flush=True)
    print('TOP:', imp[:12])

if __name__ == '__main__':
    main()
