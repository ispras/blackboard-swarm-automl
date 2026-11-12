"""Meta-stacking of base oof_pred/test_pred parquet artifacts.
Usage: python stack.py --smooth 5 --out sub_stack.csv
"""
import pandas as pd, numpy as np, glob, os, argparse, lightgbm as lgb
from sklearn.model_selection import GroupKFold

B = 'workspaces/fomalhaut_omicron/'


def best_mcc(y, p):
    order = np.argsort(-p)
    ys = y[order].astype(np.float64); ps = p[order]
    tp = np.cumsum(ys); fp = np.cumsum(1.0 - ys)
    P = float(y.sum()); N = float(len(y)) - P
    fn = P - tp; tn = N - fp
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = np.where(den > 0, (tp * tn - fp * fn) / np.where(den > 0, den, 1.0), 0.0)
    i = int(np.argmax(mcc))
    return float(mcc[i]), float((ps[i] + ps[min(i + 1, len(ps) - 1)]) / 2)


def addkeys(df):
    df = df.copy()
    df['game_play'] = df.contact_id.str.rsplit('_', n=3).str[0]
    df['step'] = df.contact_id.str.rsplit('_', n=3).str[1].astype(int)
    df['p1'] = df.contact_id.str.rsplit('_', n=3).str[2]
    df['p2'] = df.contact_id.str.rsplit('_', n=3).str[3]
    return df


def smooth_by_pair(df, col, w):
    d = df.sort_values(['game_play', 'p1', 'p2', 'step'])
    v = d.groupby(['game_play', 'p1', 'p2'], sort=False)[col].transform(
        lambda s: s.rolling(w, min_periods=1, center=True).mean()).values
    return d, v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smooth', type=int, default=5)
    ap.add_argument('--leaves', type=int, default=64)
    ap.add_argument('--rounds', type=int, default=1500)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', default='sub_stack.csv')
    a = ap.parse_args()

    lab = addkeys(pd.read_csv('data/train_labels.csv', usecols=['contact_id', 'contact']))
    lab['game_key'] = lab.game_play.str.split('_').str[0]
    base = pd.read_parquet(B + 'cache/feat3_train.parquet',
                           columns=['contact_id', 'p2', 'dist', 'nn_dist1', 'nn_dist2',
                                    'speed1', 'speed2', 'speed_sum', 'closing', 'hdist_e', 'hdist_s'])
    base['is_ground'] = (base.p2 == 'G').astype(np.int8)
    base = base.drop(columns=['p2'])
    lab = lab.merge(base, on='contact_id', how='left')
    ss = addkeys(pd.read_csv('data/sample_submission.csv', usecols=['contact_id']))
    tbase = pd.read_parquet(B + 'cache/feat3_test.parquet',
                            columns=['contact_id', 'p2', 'dist', 'nn_dist1', 'nn_dist2',
                                     'speed1', 'speed2', 'speed_sum', 'closing', 'hdist_e', 'hdist_s'])
    tbase['is_ground'] = (tbase.p2 == 'G').astype(np.int8)
    tbase = tbase.drop(columns=['p2'])
    ss = ss.merge(tbase, on='contact_id', how='left')

    cand = sorted(set(glob.glob(B + 'oof_pred*.parquet')) | set(glob.glob('workspaces/*/oof_pred*.parquet')))
    models = []
    for f in cand:
        tf = f.replace('oof_pred', 'test_pred')
        if os.path.exists(tf):
            models.append((os.path.relpath(f, 'workspaces'), f, tf))
    print('base models:', [m[0] for m in models])

    for n, f, tf in models:
        o = pd.read_parquet(f).drop_duplicates('contact_id').rename(columns={'prob': 'p_' + n})
        t = pd.read_parquet(tf).drop_duplicates('contact_id').rename(columns={'prob': 'p_' + n})
        lab = lab.merge(o, on='contact_id', how='left')
        ss = ss.merge(t, on='contact_id', how='left')

    pcols = ['p_' + n for n, _, _ in models]
    lab = lab.sort_values(['game_play', 'p1', 'p2', 'step']).reset_index(drop=True)
    for c in pcols:
        d = lab.sort_values(['game_play', 'p1', 'p2', 'step'])
        lab[c + '_sm'] = d.groupby(['game_play', 'p1', 'p2'], sort=False)[c].transform(
            lambda s: s.rolling(a.smooth, min_periods=1, center=True).mean()).values
    ss = ss.sort_values(['game_play', 'p1', 'p2', 'step']).reset_index(drop=True)
    for c in pcols:
        d = ss.sort_values(['game_play', 'p1', 'p2', 'step'])
        ss[c + '_sm'] = d.groupby(['game_play', 'p1', 'p2'], sort=False)[c].transform(
            lambda s: s.rolling(a.smooth, min_periods=1, center=True).mean()).values

    mfeats = pcols + [c + '_sm' for c in pcols] + ['is_ground', 'dist', 'nn_dist1', 'nn_dist2',
                                                   'speed1', 'speed2', 'speed_sum', 'closing', 'hdist_e', 'hdist_s']
    y = lab.contact.values
    X = lab[mfeats]
    print('meta matrix', X.shape)
    params = dict(objective='binary', learning_rate=0.03, num_leaves=a.leaves, min_data_in_leaf=200,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  n_jobs=32, seed=a.seed, verbose=-1, metric='auc')
    oof = np.zeros(len(lab)); iters = []
    gkf = GroupKFold(n_splits=5)
    for k, (tri, vai) in enumerate(gkf.split(X, y, lab.game_key.values)):
        dtr = lgb.Dataset(X.iloc[tri], y[tri]); dva = lgb.Dataset(X.iloc[vai], y[vai], reference=dtr)
        m = lgb.train(params, dtr, num_boost_round=a.rounds, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(80), lgb.log_evaluation(200)])
        oof[vai] = m.predict(X.iloc[vai], num_iteration=m.best_iteration)
        iters.append(m.best_iteration)
        print('meta fold', k, 'best_iter', m.best_iteration, 'MCC %.5f' % best_mcc(y[vai], oof[vai])[0])
    mcc, thr = best_mcc(y, oof)
    print('META STACK OOF MCC %.5f thr %.5f' % (mcc, thr))
    # final model on all data
    dall = lgb.Dataset(X, y)
    mfull = lgb.train(params, dall, num_boost_round=int(np.mean(iters)))
    pt = mfull.predict(ss[mfeats], num_iteration=int(np.mean(iters)))
    sub = pd.DataFrame({'contact_id': ss.contact_id.values, 'contact': (pt >= thr).astype(int)})
    sub = pd.read_csv('data/sample_submission.csv', usecols=['contact_id']).merge(sub, on='contact_id', how='left')
    sub['contact'] = sub.contact.fillna(0).astype(int)
    sub.to_csv(B + a.out, index=False)
    print('wrote', a.out, 'pos rate', sub.contact.mean())


if __name__ == '__main__':
    main()
