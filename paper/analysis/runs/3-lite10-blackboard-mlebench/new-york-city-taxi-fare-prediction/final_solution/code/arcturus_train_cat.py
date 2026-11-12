import argparse, time, os, json
import numpy as np, pandas as pd
import pyarrow.parquet as pq
from catboost import CatBoostRegressor, Pool

CACHE = 'workspaces/arcturus_upsilon/cache'
META = {'key', 'fare_amount', 'holdout'}

def load_sample(n_train, n_val, feats=None, cache='train_v2'):
    pf = pq.ParquetFile(f'{CACHE}/{cache}.parquet')
    all_cols = pf.schema.names
    if feats is None:
        feats = [c for c in all_cols if c not in META]
    cols = feats + ['fare_amount', 'holdout', 'key']
    k = len(feats)
    X = np.empty((n_train, k), np.float32); y = np.empty(n_train, np.float32)
    Vs, Ks = [], []; nt = nv = 0; t0 = time.time()
    for bi, batch in enumerate(pf.iter_batches(batch_size=2_000_000, columns=cols)):
        d = batch.to_pandas(); ho = d['holdout'].to_numpy()
        Xb = d[feats].to_numpy(np.float32); yb = d['fare_amount'].to_numpy(np.float32); keys = d['key'].to_numpy(); del d
        if nv < n_val:
            vm = np.flatnonzero(ho)[:n_val - nv]
            if len(vm): Vs.append(np.column_stack([yb[vm], Xb[vm]])); Ks.append(keys[vm]); nv += len(vm)
        if nt < n_train:
            tm = np.flatnonzero(~ho)[:n_train - nt]
            if len(tm): X[nt:nt+len(tm)] = Xb[tm]; y[nt:nt+len(tm)] = yb[tm]; nt += len(tm)
        del Xb, yb, keys
        print(f'batch {bi}: train={nt} val={nv} t={time.time()-t0:.0f}s', flush=True)
        if nt >= n_train and nv >= n_val: break
    V = np.concatenate(Vs); K = np.concatenate(Ks)
    return X[:nt], y[:nt], V[:, 0], V[:, 1:], feats, K

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_train', type=int, default=15_000_000)
    ap.add_argument('--n_val', type=int, default=200_000)
    ap.add_argument('--depth', type=int, default=10)
    ap.add_argument('--lr', type=float, default=0.05)
    ap.add_argument('--iters', type=int, default=3000)
    ap.add_argument('--l2', type=float, default=3.0)
    ap.add_argument('--threads', type=int, default=16)
    ap.add_argument('--seed', type=int, default=99)
    ap.add_argument('--tag', type=str, default='cat15m')
    ap.add_argument('--task_type', type=str, default='GPU')
    args = ap.parse_args()
    outdir = 'workspaces/arcturus_upsilon/runs'; os.makedirs(outdir, exist_ok=True)
    X, y, yv, V, feats, vkeys = load_sample(args.n_train, args.n_val)
    print('loaded', X.shape, V.shape, flush=True)
    tr = Pool(X, y, feature_names=feats)
    va = Pool(V, yv, feature_names=feats)
    m = CatBoostRegressor(loss_function='RMSE', iterations=args.iters, learning_rate=args.lr,
                          depth=args.depth, l2_leaf_reg=args.l2, random_seed=args.seed,
                          task_type=args.task_type, devices='0', thread_count=args.threads,
                          od_type='Iter', od_wait=150, verbose=100, allow_writing_files=False)
    t0 = time.time()
    m.fit(tr, eval_set=va, use_best_model=True)
    pred = m.predict(V)
    rmse = float(np.sqrt(np.mean((pred - yv) ** 2)))
    print(f'BEST={m.get_best_iteration()} VAL_RMSE={rmse:.5f} time={time.time()-t0:.0f}s', flush=True)
    m.save_model(f'{outdir}/{args.tag}.cbm')
    pd.DataFrame({'key': vkeys, 'pred': pred, 'y': yv}).to_parquet(f'{outdir}/{args.tag}_val.parquet', index=False)
    tdf = pd.read_parquet(f'{CACHE}/test_v2.parquet')
    pt = np.clip(m.predict(Pool(tdf[feats].to_numpy(np.float32), feature_names=feats)), 2.5, 200.0)
    tdf[['key']].assign(pred=pt).to_parquet(f'{outdir}/{args.tag}_test.parquet', index=False)
    tdf[['key']].assign(fare_amount=pt).to_csv(f'{outdir}/{args.tag}_submission.csv', index=False)
    json.dump({'tag': args.tag, 'rmse': rmse, 'best_iter': m.get_best_iteration(),
               'n_train': int(len(X)), 'feats': feats}, open(f'{outdir}/{args.tag}_meta.json', 'w'), indent=1)
    print('test stats', float(pt.min()), float(np.median(pt)), float(pt.max()), float(pt.mean()))
    print('done')

if __name__ == '__main__':
    main()
