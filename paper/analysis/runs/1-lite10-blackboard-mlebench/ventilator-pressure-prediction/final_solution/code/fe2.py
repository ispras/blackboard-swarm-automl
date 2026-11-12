import os, time
import numpy as np
import pandas as pd

DATA = 'data'
OUT = 'workspaces/sirius_tau/cache2'
os.makedirs(OUT, exist_ok=True)
SEQ = 80


def build(df, is_train):
    t0 = time.time()
    df = df.sort_values(['breath_id', 'time_step', 'id'], kind='mergesort').reset_index(drop=True)
    cnt = df.groupby('breath_id').size()
    assert (cnt == SEQ).all(), cnt.value_counts()
    order = df['breath_id'].values
    breath_ids = order[::SEQ]
    n = len(df) // SEQ
    u_in = df['u_in'].values.reshape(n, SEQ).astype(np.float32)
    u_out = df['u_out'].values.reshape(n, SEQ).astype(np.float32)
    ts = df['time_step'].values.reshape(n, SEQ).astype(np.float32)
    R = df['R'].values.reshape(n, SEQ)[:, :1].astype(np.float32)
    C = df['C'].values.reshape(n, SEQ)[:, :1].astype(np.float32)

    feats = {}
    feats['u_in'] = u_in
    feats['u_out'] = u_out
    feats['time_step'] = ts
    feats['step'] = np.broadcast_to(np.arange(SEQ, dtype=np.float32), (n, SEQ)).copy()
    dt = np.empty_like(ts)
    dt[:, 0] = ts[:, 0]
    dt[:, 1:] = ts[:, 1:] - ts[:, :-1]
    feats['dt'] = dt
    feats['R'] = np.broadcast_to(R, (n, SEQ)).copy()
    feats['C'] = np.broadcast_to(C, (n, SEQ)).copy()
    feats['RC'] = (R * C).astype(np.float32).repeat(SEQ, axis=1)
    feats['R_plus_C'] = (R + C).astype(np.float32).repeat(SEQ, axis=1)
    feats['R_div_C'] = (R / C).astype(np.float32).repeat(SEQ, axis=1)

    u_dt = u_in * dt
    cum = np.cumsum(u_dt, axis=1)
    csum = np.cumsum(u_in, axis=1)
    feats['u_in_cumsum'] = cum
    feats['u_in_cummean'] = csum / (np.arange(SEQ, dtype=np.float32) + 1)
    feats['u_in_cummax'] = np.maximum.accumulate(u_in, axis=1)
    feats['u_in_cummin'] = np.minimum.accumulate(u_in, axis=1)
    feats['u_in_cumsum_dt'] = csum * dt
    feats['vol'] = cum
    feats['vol_over_C'] = cum / np.repeat(C, SEQ, axis=1)
    feats['R_times_u_in'] = u_in * np.repeat(R, SEQ, axis=1)
    feats['phys_pressure'] = u_in * np.repeat(R, SEQ, axis=1) + cum / np.repeat(C, SEQ, axis=1)
    feats['u_in_over_C'] = u_in / np.repeat(C, SEQ, axis=1)
    feats['u_in_over_RC'] = u_in / np.repeat(R * C, SEQ, axis=1)
    feats['u_in_sq'] = u_in * u_in
    feats['time_u_in'] = ts * u_in

    # per-breath total volume broadcast
    total = cum[:, -1:].copy()
    feats['tot_vol'] = np.broadcast_to(total, (n, SEQ)).copy()
    feats['cum_frac'] = cum / (total + 1e-6)
    feats['u_in_frac'] = u_in / (u_in.sum(1, keepdims=True) + 1e-6)

    # rolling stats
    for w in (3, 5):
        pad = np.pad(u_in, ((0, 0), (w // 2, w // 2)), mode='edge')
        win = np.lib.stride_tricks.sliding_window_view(pad, w, axis=1)
        feats[f'u_in_roll_mean{w}'] = win.mean(-1).astype(np.float32)
        feats[f'u_in_roll_std{w}'] = win.std(-1).astype(np.float32)

    for k in (1, 2, 3, 4, 6):
        lag = np.empty_like(u_in)
        lag[:, :k] = u_in[:, :1]
        lag[:, k:] = u_in[:, :-k]
        feats[f'u_in_lag{k}'] = lag
    for k in (1, 2, 3):
        lead = np.empty_like(u_in)
        lead[:, -k:] = u_in[:, -1:]
        lead[:, :-k] = u_in[:, k:]
        feats[f'u_in_lead{k}'] = lead
    d1 = np.empty_like(u_in); d1[:, 0] = 0; d1[:, 1:] = u_in[:, 1:] - u_in[:, :-1]
    feats['u_in_diff1'] = d1
    d2 = np.empty_like(u_in); d2[:, 0] = 0; d2[:, 1:] = d1[:, 1:] - d1[:, :-1]
    feats['u_in_diff2'] = d2

    for v in (5, 20, 50):
        feats[f'R_{v}'] = np.broadcast_to((R == v).astype(np.float32), (n, SEQ)).copy()
    for v in (10, 20, 50):
        feats[f'C_{v}'] = np.broadcast_to((C == v).astype(np.float32), (n, SEQ)).copy()

    names = list(feats.keys())
    X = np.stack([feats[k] for k in names], axis=-1).astype(np.float32)
    y = df['pressure'].values.reshape(n, SEQ).astype(np.float32) if is_train else None
    print(f'built {n} breaths, X={X.shape} feats={len(names)} in {time.time()-t0:.1f}s', flush=True)
    return X, y, breath_ids, names


if __name__ == '__main__':
    tr = pd.read_csv(f'{DATA}/train.csv')
    te = pd.read_csv(f'{DATA}/test.csv')
    Xtr, ytr, btr, names = build(tr, True)
    Xte, _, bte, names2 = build(te, False)
    assert names == names2
    np.save(f'{OUT}/X_train.npy', Xtr)
    np.save(f'{OUT}/y_train.npy', ytr)
    np.save(f'{OUT}/breath_ids_train.npy', btr)
    np.save(f'{OUT}/X_test.npy', Xte)
    np.save(f'{OUT}/breath_ids_test.npy', bte)
    with open(f'{OUT}/feature_names.txt', 'w') as f:
        f.write('\n'.join(names))
    # same canonical split as cache (sorted unique breath ids permutation seed42)
    rng = np.random.RandomState(42)
    idx = np.arange(len(btr)); rng.shuffle(idx)
    nval = int(round(0.10 * len(btr)))
    val_idx = np.sort(idx[:nval]); trn_idx = np.sort(idx[nval:])
    np.save(f'{OUT}/val_idx.npy', val_idx)
    np.save(f'{OUT}/trn_idx.npy', trn_idx)
    print('saved', OUT, Xtr.shape, Xte.shape, flush=True)
