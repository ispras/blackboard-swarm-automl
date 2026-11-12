import numpy as np, pandas as pd, json, os, glob
import lightgbm as lgb

d = np.load('cache3/train.npz'); dt = np.load('cache3/test.npz')
X = d['X']; y = d['y']; uo = d['u_out']; folds = np.load('cache3/folds.npy'); bid = d['breath_id']
f0 = np.where(folds == 0)[0]
Xte = dt['X']; ute = dt['u_out']
y0 = y[f0]; uo0 = uo[f0]; X0 = X[f0]
y0f = y0.reshape(-1); m0f = (uo0.reshape(-1) == 0)

# sample-order reorder for test (cache test is breath-id sorted; shared id also breath-id sorted)
ids_sorted = np.load('../shared/test_id_sorted.npy').reshape(-1)
order = np.argsort(ids_sorted, kind='stable')
Xte_sub = Xte.reshape(-1, Xte.shape[2])[order]
ute_sub = ute.reshape(-1)[order]
assert len(Xte_sub) == 603600

def load_v(path, idx_path=None):
    p = np.load(path)
    if idx_path is not None:
        vi = np.load(idx_path)
        full = np.full((len(y), 80), np.nan, np.float32); full[vi] = p[:len(vi)]; p = full
    return p[f0]

def load_t(path, ids_path=None, per_row_ids=False):
    p = np.load(path).reshape(-1).astype(np.float32)
    if ids_path is not None:
        ids = np.load(ids_path).reshape(-1)
        if len(ids) == len(p):
            p = p[np.argsort(ids, kind='stable')]
        else:
            p = p[np.argsort(np.repeat(ids, 80), kind='stable')]
    return p

def collect():
    M = []
    def add(name, vp, tp, vi=None, ti=None):
        try:
            v = load_v(vp, vi); t = load_t(tp, ti)
            M.append({'name': name, 'val': v, 'test': t})
            print('loaded', name, v.shape, t.shape)
        except Exception as e:
            print('skip', name, e)
    add('canopus_lstm384', '../canopus_iota/oof_lstm384.npy', '../canopus_iota/test_pred_lstm384.npy', ti='../shared/test_id_sorted.npy')
    add('castor_gru_v1', '../castor_eta/runs/gru_v1_val.npy', '../castor_eta/runs/gru_v1_test.npy', vi='../castor_eta/runs/gru_v1_val_idx.npy', ti='../castor_eta/cache/test_id.npy')
    add('castor_bilstm_big', '../castor_eta/runs/bilstm_big_val.npy', '../castor_eta/runs/bilstm_big_test.npy', vi='../castor_eta/runs/bilstm_big_val_idx.npy', ti='../castor_eta/cache/test_id.npy')
    add('castor_bilstm_c2', '../castor_eta/runs/bilstm_c2_val.npy', '../castor_eta/runs/bilstm_c2_test.npy', vi='../castor_eta/runs/bilstm_c2_val_idx.npy', ti='../castor_eta/cache/test_id.npy')
    add('bellatrix_lstm_phys', 'preds/oof_lstm_phys_f0_s4.npy', 'preds/test_lstm_phys_f0_s4.npy', vi='preds/valid_idx_lstm_phys_f0_s4.npy')
    # 5-fold canopus variants if present
    for n in ['lstm384_5f']:
        add('canopus_'+n, f'../canopus_iota/oof_{n}.npy', f'../canopus_iota/test_pred_{n}.npy', ti='../shared/test_id_sorted.npy')
    return M

def meta_features(V, T, Xrows, uorows):
    # V (K,13581,80) or (K,603600); T same flattened
    Pv = V.reshape(V.shape[0], -1).T  # (nrows, K) -- careful shapes differ
    Pt = T.reshape(T.shape[0], -1).T
    return Pv, Pt

def main():
    M = collect()
    K = len(M)
    V = np.stack([m['val'] for m in M])        # (K,13581,80)
    T = np.stack([m['test'] for m in M])       # (K,603600)
    names = [m['name'] for m in M]
    # report singles
    for i, n in enumerate(names):
        vf = V[i].reshape(-1)
        print(f'{n:26s} insp {np.abs(vf[m0f]-y0f[m0f]).mean():.5f} all {np.abs(vf-y0f).mean():.5f}')
    P0 = V.reshape(K, -1).T                    # (N0*80, K)
    Ptest = T.T                                # (Ntest, K)
    step = np.tile(np.arange(80), len(f0))
    grp = np.repeat(np.arange(len(f0)), 80)
    base_cols = [X0.reshape(-1, X0.shape[2])[:, i] for i in (0, 1, 3, 6, 19, 20, 21)]
    F0 = np.column_stack([P0, P0.mean(1), P0.std(1), *base_cols, uo0.reshape(-1), step])
    base_cols_t = [Xte_sub[:, i] for i in (0, 1, 3, 6, 19, 20, 21)]
    Ft = np.column_stack([Ptest, Ptest.mean(1), Ptest.std(1), *base_cols_t, ute_sub, np.tile(np.arange(80), 603600//80)])
    feat_names = names + ['pmean','pstd','u_in','time_step','cumsum','area','step_f','R','C','u_out','step']
    # CV within fold0, grouped by breath
    rng = np.random.RandomState(7); perm = rng.permutation(len(f0)); foldg = np.empty(len(f0), int); foldg[perm] = np.arange(len(f0)) % 5
    pg = np.repeat(foldg, 80)
    oof = np.zeros(len(y0f), np.float32)
    params = dict(objective='regression_l1', n_estimators=700, learning_rate=0.04, num_leaves=63,
                  min_child_samples=200, subsample=0.8, colsample_bytree=0.8, n_jobs=4, verbose=-1)
    for k in range(5):
        tr = pg != k; va = pg == k
        mdl = lgb.LGBMRegressor(**params)
        mdl.fit(F0[tr], y0f[tr], feature_name=feat_names)
        oof[va] = mdl.predict(F0[va])
    print('STACK CV insp', np.abs(oof[m0f]-y0f[m0f]).mean(), 'all', np.abs(oof-y0f).mean())
    mdl = lgb.LGBMRegressor(**params)
    mdl.fit(F0, y0f, feature_name=feat_names)
    pt = mdl.predict(Ft)
    np.save('preds/stack_test.npy', pt)
    np.save('preds/stack_oof_cv.npy', oof)
    print('saved stack_test.npy', pt.shape, pt.min(), pt.max())

if __name__ == '__main__':
    main()
