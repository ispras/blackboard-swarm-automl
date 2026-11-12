import argparse, json
import numpy as np
import pandas as pd
from scipy.optimize import minimize

CACHE = 'workspaces/sirius_tau/cache'
SEQ = 80


def load_val_truth():
    val_idx = np.load(f'{CACHE}/val_idx.npy')
    ids = np.load(f'{CACHE}/id_train.npy')
    val_ids = ids[val_idx].reshape(-1)
    step = np.tile(np.arange(SEQ), len(val_idx))
    tr = pd.read_csv('data/train.csv', usecols=['id', 'pressure', 'u_out', 'R', 'C']).set_index('id')
    t = tr.reindex(val_ids)
    assert t['pressure'].notna().all()
    return dict(val_ids=val_ids, y=t['pressure'].values.astype(np.float64),
                mask=(t['u_out'].values == 0), step=step,
                R=t['R'].values.astype(np.int64), C=t['C'].values.astype(np.int64))


def read_pred(path, ids):
    df = pd.read_csv(path)
    s = df.drop_duplicates('id').set_index('id')['pressure']
    out = s.reindex(ids).values.astype(np.float64)
    if np.isnan(out).any():
        raise ValueError(f'{path}: missing {np.isnan(out).sum()} ids')
    return out


def mae(y, p, m=None):
    if m is not None:
        y = y[m]; p = p[m]
    return float(np.abs(y - p).mean())


def fit_w(P, y, m, sub=250000, seed=0, n_starts=6):
    idx = np.where(m)[0]
    if len(idx) > sub:
        idx = np.random.RandomState(seed).choice(idx, sub, replace=False)
    Ps, ys = P[idx], y[idx]
    k = P.shape[1]

    def obj(w):
        return np.abs(ys - Ps @ w).mean()
    cons = ({'type': 'eq', 'fun': lambda w: w.sum() - 1.0},)
    bnds = [(0.0, 1.0)] * k
    # multi-start to avoid SLSQP local optima (dubhe observed bad single-start solutions)
    starts = [np.ones(k) / k]
    rng = np.random.RandomState(seed + 7)
    for _ in range(max(0, n_starts - 1)):
        starts.append(rng.dirichlet(np.ones(k)))
    # also start concentrated on the best single model
    single = np.array([obj(np.eye(k)[j]) for j in range(k)])
    e = np.zeros(k); e[int(np.argmin(single))] = 1.0
    starts.append(e)
    best_w, best_f = None, 1e18
    for s in starts:
        r = minimize(obj, s, method='SLSQP', bounds=bnds, constraints=cons,
                     options={'maxiter': 500, 'ftol': 1e-11})
        w = np.clip(r.x, 0, None)
        if w.sum() > 0:
            w = w / w.sum()
            f = obj(w)
            if f < best_f:
                best_f, best_w = f, w
    return best_w if best_w is not None else np.ones(k) / k


def bias_table(pred, y, m, R, C, step, shrink_n=200.0):
    r = y - pred
    df = pd.DataFrame({'r': r, 'R': R, 'C': C, 'step': step, 'm': m})
    g = df[df.m].groupby(['R', 'C', 'step'])['r']
    med = g.median(); cnt = g.size()
    return med * (cnt / (cnt + shrink_n))


def apply_bias(pred, R, C, step, tbl):
    idx = pd.MultiIndex.from_arrays([R, C, step])
    corr = tbl.reindex(idx).values
    return pred + np.nan_to_num(corr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    models = json.load(open(args.manifest))
    T = load_val_truth()
    y, mask, R, C, step = T['y'], T['mask'], T['R'], T['C'], T['step']

    V, exp_ok = [], []
    for m in models:
        p = read_pred(m['val'], T['val_ids'])
        V.append(p)
        exp_ok.append(bool(m.get('exp_ok', True)))
        print(f"  {m['name']:26s} insp={mae(y,p,mask):.5f} all={mae(y,p):.5f} exp_ok={exp_ok[-1]}")
    P = np.stack(V, 1)
    exp_ok = np.array(exp_ok)

    # inspiratory weights over all models
    w_insp = fit_w(P, y, mask)
    insp_pred = P @ w_insp
    print('insp weights:', {m['name']: round(float(wi), 4) for m, wi in zip(models, w_insp)})
    print(f'insp weighted val_insp={mae(y,insp_pred,mask):.5f}')

    # expiratory weights over exp-capable models only
    emask = ~mask
    if exp_ok.sum() >= 1:
        Pe = P[:, exp_ok]
        w_exp = fit_w(Pe, y, emask, seed=1)
        exp_pred = Pe @ w_exp
        print('exp models:', [m['name'] for m, e in zip(models, exp_ok) if e])
        print('exp weights:', [round(float(x), 4) for x in w_exp])
        print(f'exp weighted val_all_exp={mae(y,exp_pred,emask):.5f}')
    else:
        w_exp = None

    # bias correction with half-split honest estimate
    nvb = len(T['val_ids']) // SEQ
    rng = np.random.RandomState(123)
    half = np.zeros(nvb, bool); ii = rng.permutation(nvb); half[ii[:nvb // 2]] = True
    block = np.repeat(half, SEQ)
    tbl_half = bias_table(insp_pred, y, mask & block, R, C, step)
    ev = ~block
    base_ev = mae(y[ev], insp_pred[ev], mask[ev])
    corr_ev = apply_bias(insp_pred, R, C, step, tbl_half)
    print(f'bias-corr honest: base={base_ev:.5f} corr={mae(y[ev],corr_ev[ev],mask[ev]):.5f}')
    tbl_full = bias_table(insp_pred, y, mask, R, C, step)
    insp_corr = apply_bias(insp_pred, R, C, step, tbl_full)
    use_bias = mae(y[ev], corr_ev[ev], mask[ev]) < base_ev - 1e-6
    print('use_bias =', use_bias)

    # test
    sample = pd.read_csv('data/sample_submission.csv')
    tids = sample['id'].values
    Tst = np.stack([read_pred(m['test'], tids) for m in models], 1)
    te = pd.read_csv('data/test.csv', usecols=['id', 'breath_id', 'time_step', 'u_out', 'R', 'C']).set_index('id').reindex(tids)
    te_step = pd.DataFrame({'b': te['breath_id'].values, 't': te['time_step'].values})
    tstep = te_step.groupby('b')['t'].rank(method='first').astype(int).values - 1
    tR = te['R'].values.astype(np.int64); tC = te['C'].values.astype(np.int64)
    tuout = te['u_out'].values

    insp_t = Tst @ w_insp
    if use_bias:
        insp_t = apply_bias(insp_t, tR, tC, tstep, tbl_full)
    if w_exp is not None:
        exp_t = Tst[:, exp_ok] @ w_exp
    else:
        exp_t = insp_t
    final = np.where(tuout == 0, insp_t, exp_t)
    final = np.clip(final, -2.0, 65.0)
    sub = pd.DataFrame({'id': tids, 'pressure': final})
    sub.to_csv(args.out, index=False)
    rep = {'models': models, 'w_insp': w_insp.tolist(), 'w_exp': (w_exp.tolist() if w_exp is not None else None),
           'exp_ok': exp_ok.tolist(), 'val_insp_weighted': mae(y, insp_pred, mask), 'use_bias': bool(use_bias),
           'val_insp_corrected': mae(y, insp_corr, mask), 'bias_honest_base': base_ev,
           'bias_honest_corr': mae(y[ev], corr_ev[ev], mask[ev])}
    json.dump(rep, open(args.out.replace('.csv', '_report.json'), 'w'), indent=2)
    print('wrote', args.out, sub.shape)


if __name__ == '__main__':
    main()
