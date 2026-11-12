import numpy as np, pandas as pd, json, os
from scipy.optimize import minimize

d = np.load('cache3/train.npz'); dt = np.load('cache3/test.npz')
X = d['X']; y = d['y']; uo = d['u_out']; folds = np.load('cache3/folds.npy'); bid = d['breath_id']
f0 = np.where(folds == 0)[0]
y0 = y[f0].astype(np.float32); uo0 = uo[f0].astype(np.float64)
meta = pd.read_csv('breath_meta.csv').set_index('breath_id')
R0 = meta.loc[bid[f0], 'R'].to_numpy(); C0 = meta.loc[bid[f0], 'C'].to_numpy()
y0f = y0.reshape(-1); m0f = (uo0.reshape(-1) == 0)
n0 = len(f0)
step = np.tile(np.arange(80), n0)
grp = (R0.astype(int) * 100 + C0.astype(int))
grp_row = np.repeat(grp, 80)
breath_row = np.repeat(np.arange(n0), 80)

ids_sorted = np.load('../shared/test_id_sorted.npy').reshape(-1)
order = np.argsort(ids_sorted, kind='stable')
nte = len(order) // 80
Xte = dt['X']; ute = dt['u_out']
Xte_sub = Xte.reshape(-1, Xte.shape[2])[order]
ute_sub = ute.reshape(-1)[order]
_tm = pd.read_csv('../../data/test.csv', usecols=['breath_id', 'R', 'C']).groupby('breath_id').first()
bte_sorted = dt['breath_id']
Rte = _tm.loc[bte_sorted, 'R'].to_numpy(); Cte = _tm.loc[bte_sorted, 'C'].to_numpy()
# reorder test breath-level arrays to sample order
bix = np.argsort(ids_sorted.reshape(-1, 80)[:, 0] if ids_sorted.size == 603600 else np.arange(nte), kind='stable')
# simpler: rows order built from argsort; build breath order from block starts
cnt = np.arange(len(order)) // 80
# order contains row indices in breath-sorted layout; map each output block to source block
src_block = order[::80] // 80
Rte_row = np.repeat(Rte[src_block], 80); Cte_row = np.repeat(Cte[src_block], 80)
grp_te = np.repeat((Rte[src_block].astype(int) * 100 + Cte[src_block].astype(int)), 80)
step_te = np.tile(np.arange(80), nte)

def load_v(path, idx_path=None):
    p = np.load(path)
    if idx_path is not None:
        vi = np.load(idx_path)
        full = np.full((len(y), 80), np.nan, np.float32); full[vi] = p[:len(vi)]; p = full
    return p[f0]

def load_t(path, ids_path=None):
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
            M.append({'name': name, 'val': v.reshape(-1), 'test': t})
            print('loaded', name, flush=True)
        except Exception as e:
            print('skip', name, e, flush=True)
    TIDS='../shared/test_id_sorted.npy'; CIDS='../castor_eta/cache/test_id.npy'
    add('castor_bilstm_c2', '../castor_eta/runs/bilstm_c2_val.npy', '../castor_eta/runs/bilstm_c2_test.npy', vi='../castor_eta/runs/bilstm_c2_val_idx.npy', ti=CIDS)
    add('castor_bilstm_c2_250', '../castor_eta/runs/bilstm_c2_250_val.npy', '../castor_eta/runs/bilstm_c2_250_test.npy', vi='../castor_eta/runs/bilstm_c2_250_val_idx.npy', ti=CIDS)
    add('castor_bilstm_e3_220', '../castor_eta/runs/bilstm_e3_220_val.npy', '../castor_eta/runs/bilstm_e3_220_test.npy', vi='../castor_eta/runs/bilstm_e3_220_val_idx.npy', ti=CIDS)
    add('canopus_e3_s456', '../canopus_iota/runs/e3_s456_val.npy', '../canopus_iota/runs/e3_s456_test.npy', vi='../canopus_iota/runs/e3_s456_val_idx.npy', ti=TIDS)
    add('canopus_e3_s789', '../canopus_iota/runs/e3_s789_val.npy', '../canopus_iota/runs/e3_s789_test.npy', vi='../canopus_iota/runs/e3_s789_val_idx.npy', ti=TIDS)
    add('canopus_c2s123', '../canopus_iota/runs/c2s123_val.npy', '../canopus_iota/runs/c2s123_test.npy', vi='../canopus_iota/runs/c2s123_val_idx.npy', ti=TIDS)
    add('castor_bilstm_big', '../castor_eta/runs/bilstm_big_val.npy', '../castor_eta/runs/bilstm_big_test.npy', vi='../castor_eta/runs/bilstm_big_val_idx.npy', ti=CIDS)
    add('castor_cnn_gru_c2', '../castor_eta/runs/cnn_gru_c2_val.npy', '../castor_eta/runs/cnn_gru_c2_test.npy', vi='../castor_eta/runs/cnn_gru_c2_val_idx.npy', ti=CIDS)
    add('castor_gru_v1', '../castor_eta/runs/gru_v1_val.npy', '../castor_eta/runs/gru_v1_test.npy', vi='../castor_eta/runs/gru_v1_val_idx.npy', ti=CIDS)
    add('canopus_lstm384', '../canopus_iota/oof_lstm384.npy', '../canopus_iota/test_pred_lstm384.npy', ti=TIDS)
    add('canopus_5f2', '../canopus_iota/oof_lstm384_5f2.npy', '../canopus_iota/test_pred_lstm384_5f2.npy', ti=TIDS)
    add('canopus_v2f', '../canopus_iota/oof_lstm384_v2f.npy', '../canopus_iota/test_pred_lstm384_v2f.npy', ti=TIDS)
    add('canopus_512c2b', '../canopus_iota/oof_lstm512_c2b.npy', '../canopus_iota/test_pred_lstm512_c2b.npy', ti=TIDS)
    add('bellatrix_phys', 'preds/oof_lstm_phys_f0_s4.npy', 'preds/test_lstm_phys_f0_s4.npy', vi='preds/valid_idx_lstm_phys_f0_s4.npy')
    add('bellatrix_big4', 'preds/oof_lstm_big4_f0_s7.npy', 'preds/test_lstm_big4_f0_s7.npy', vi='preds/valid_idx_lstm_big4_f0_s7.npy')
    return M

TWINS = {
  'castor_bilstm_c2': [('../castor_eta/runs/bilstm_c2_all_test.npy', '../castor_eta/cache/test_id.npy'),
                       ('../castor_eta/runs/bilstm_c2_all2_test.npy', '../castor_eta/cache/test_id.npy')],
  'castor_bilstm_c2_250': [('../castor_eta/runs/bilstm_c2_all250_test.npy', '../castor_eta/cache/test_id.npy')],
  'castor_bilstm_e3_220': [('../castor_eta/runs/bilstm_e3_all_test.npy', '../castor_eta/cache/test_id.npy')],
  'canopus_e3_s456': [('../canopus_iota/runs/e3_s456_all_test.npy', '../shared/test_id_sorted.npy')],
  'castor_bilstm_big': [('../castor_eta/runs/bilstm_v1_all_test.npy', '../castor_eta/cache/test_id.npy')],
  'canopus_5f2': [('../canopus_iota/test_pred_lstm384_5f2_all.npy', '../shared/test_id_sorted.npy')],
}

def fit_w(P, yy, mm, n_steps=120):
    Pi = P[:, mm]; yi = yy[mm]
    K = Pi.shape[0]
    S = np.zeros(Pi.shape[1], np.float64); n = 0; counts = np.zeros(K)
    for t in range(n_steps):
        bm = 1e18; bk = -1; bc = None
        for k in range(K):
            cand = (S + Pi[k]) / (n + 1)
            m = float(np.abs(cand - yi).mean())
            if m < bm: bm = m; bk = k; bc = cand
        S = S + Pi[bk]; n += 1; counts[bk] += 1
    return counts / counts.sum()

def seg_eval(P, T, seg0, segt, mode):
    rng = np.random.RandomState(3)
    half = np.empty(n0, np.int8); half[rng.permutation(n0)] = np.arange(n0) % 2
    half_row = np.repeat(half, 80)
    pred_cf = np.zeros(P.shape[1], np.float32)
    for h in range(2):
        tr = half_row != h; va = half_row == h
        wg = fit_w(P[:, tr], y0f[tr], m0f[tr])
        for ss in np.unique(seg0):
            st = seg0 == ss
            if (st & tr).sum() < 5000:
                pred_cf[va & st] = wg @ P[:, va & st]; continue
            w = fit_w(P[:, st & tr], y0f[st & tr], m0f[st & tr])
            pred_cf[va & st] = w @ P[:, va & st]
    insp_cf = float(np.abs(pred_cf[m0f] - y0f[m0f]).mean()); all_cf = float(np.abs(pred_cf - y0f).mean())
    Wall = fit_w(P, y0f, m0f)
    W = {}
    for ss in np.unique(seg0):
        st = seg0 == ss
        W[int(ss)] = Wall if st.sum() < 5000 else fit_w(P[:, st], y0f[st], m0f[st])
    pred_full = np.zeros(P.shape[1], np.float32)
    for ss in np.unique(seg0):
        st = seg0 == ss; pred_full[st] = W[int(ss)] @ P[:, st]
    insp_full = float(np.abs(pred_full[m0f] - y0f[m0f]).mean()); all_full = float(np.abs(pred_full - y0f).mean())
    print(f'{mode:18s} CV insp {insp_cf:.5f} all {all_cf:.5f} | full {insp_full:.5f}/{all_full:.5f}', flush=True)
    pt = np.zeros(T.shape[1], np.float32)
    for ss in np.unique(seg0):
        w = W[int(ss)]; st = segt == ss
        if st.any(): pt[st] = w @ T[:, st]
    return {'mode': mode, 'cv_insp': insp_cf, 'cv_all': all_cf, 'full_insp': insp_full,
            'full_all': all_full, 'W': {str(k): v.tolist() for k, v in W.items()}, 'test': pt}

def main():
    M = collect(); names = [m['name'] for m in M]; K = len(M)
    print('models', names)
    P = np.stack([m['val'] for m in M]); T = np.stack([m['test'] for m in M])
    Tu = T.copy()
    for i, n in enumerate(names):
        if n == 'canopus_5f2':
            ap = '../canopus_iota/test_pred_lstm384_5f2_all.npy'
            if os.path.exists(ap):
                Tu[i] = (5.0 * T[i] + load_t(ap, '../shared/test_id_sorted.npy')) / 6.0
                print('MIX 5/6 5f2avg + 1/6 all-data for', n, flush=True)
        elif n in TWINS:
            arrs = []
            for tp_, ti_ in TWINS[n]:
                if os.path.exists(tp_):
                    arrs.append(load_t(tp_, ti_))
                    print('SUBSTITUTED test-side all-data for', n, tp_, flush=True)
            if arrs:
                m = np.mean(arrs, axis=0)
                if n in ('castor_bilstm_c2_250', 'castor_bilstm_e3_220', 'canopus_e3_s456'):
                    Tu[i] = 0.5 * T[i] + 0.5 * m
                else:
                    Tu[i] = m
    for i, n in enumerate(names):
        print(f'{n:24s} insp {np.abs(P[i][m0f]-y0f[m0f]).mean():.5f} all {np.abs(P[i]-y0f).mean():.5f}')
    res = {}
    res['global'] = seg_eval(P, Tu, np.zeros_like(step), np.zeros_like(step_te), 'global')
    res['group'] = seg_eval(P, Tu, grp_row, grp_te, 'group')
    bins = np.digitize(step, [8, 16, 24, 30])
    bins_te = np.digitize(step_te, [8, 16, 24, 30])
    res['stepbin'] = seg_eval(P, Tu, bins, bins_te, 'stepbin')
    # res['grp_stepbin'] = seg_eval(P, T, grp_row * 10 + bins, grp_te * 10 + bins_te, 'grp_stepbin')
    # choose by CV insp
    b_insp = min(res.values(), key=lambda r: r['cv_insp'])
    # hedge: prefer global if insp is within 0.0005 and it is better on all-rows
    if res['global']['cv_insp'] <= b_insp['cv_insp'] + 0.0005 and res['global']['cv_all'] <= b_insp['cv_all']:
        best = res['global']
    else:
        best = b_insp
    print('CHOOSEN', best['mode'], best['cv_insp'], best['cv_all'])
    np.save('preds/blend2_test.npy', best['test'])
    json.dump({k: {kk: vv for kk, vv in v.items() if kk != 'test'} for k, v in res.items()},
              open('preds/blend2_report.json', 'w'), indent=1)
    ss = pd.read_csv('../../data/sample_submission.csv')
    ss['pressure'] = best['test']
    ss.to_csv('submission_blend2.csv', index=False)
    print('wrote submission_blend2.csv')

if __name__ == '__main__':
    main()
