"""Ensemble standard oof_pred/test_pred parquet artifacts by contact_id.
Usage: python ensemble.py [--files a.parquet,b.parquet] [--out sub_ens.csv]
"""
import pandas as pd, numpy as np, glob, os, argparse, itertools
from sklearn.metrics import matthews_corrcoef

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


def rank01(x):
    r = pd.Series(x).rank(method='average').values
    return (r - 1) / (len(r) - 1)


def smooth(df, col, w=5):
    d = df.sort_values(['game_play', 'p1', 'p2', 'step'])
    v = d.groupby(['game_play', 'p1', 'p2'], sort=False)[col].transform(
        lambda s: s.rolling(w, min_periods=1, center=True).mean()).values
    return d, v


def keyify(ids):
    s = pd.Series(ids)
    pr = s.str.rsplit('_', n=3, expand=True)
    return pr[0] + '|' + pr[2] + '|' + pr[3] + '|' + pr[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--oof', default=None, help='comma list; default: auto-discover')
    ap.add_argument('--test', default=None)
    ap.add_argument('--names', default=None)
    ap.add_argument('--w', type=int, default=5)
    ap.add_argument('--out', default='sub_ens.csv')
    a = ap.parse_args()

    lab = pd.read_csv('data/train_labels.csv', usecols=['contact_id', 'contact', 'game_play', 'step', 'nfl_player_id_1', 'nfl_player_id_2'])
    lab['p1'] = lab.nfl_player_id_1.astype(str)
    lab['p2'] = lab.nfl_player_id_2.astype(str)
    ss = pd.read_csv('data/sample_submission.csv', usecols=['contact_id'])
    ss['game_play'] = ss.contact_id.str.rsplit('_', n=3).str[0]
    ss['step'] = ss.contact_id.str.rsplit('_', n=3).str[1].astype(int)
    ss['p1'] = ss.contact_id.str.rsplit('_', n=3).str[2]
    ss['p2'] = ss.contact_id.str.rsplit('_', n=3).str[3]

    if a.oof:
        oof_files = a.oof.split(','); test_files = a.test.split(','); names = a.names.split(',')
    else:
        cand = sorted(set(glob.glob(B + 'oof_pred*.parquet')) | set(glob.glob('workspaces/*/oof_pred*.parquet')))
        oof_files, test_files, names = [], [], []
        for f in cand:
            tf = f.replace('oof_pred', 'test_pred')
            if os.path.exists(tf):
                oof_files.append(f); test_files.append(tf)
                names.append(os.path.relpath(f, 'workspaces'))
    print('models:', names)

    O = {}
    T = {}
    for f, tf, n in zip(oof_files, test_files, names):
        o = pd.read_parquet(f).drop_duplicates('contact_id')
        t = pd.read_parquet(tf).drop_duplicates('contact_id')
        O[n] = lab[['contact_id', 'contact', 'game_play', 'step', 'p1', 'p2']].merge(o, on='contact_id', how='left')
        T[n] = ss[['contact_id']].merge(t, on='contact_id', how='left')
        oo = O[n]
        assert oo.prob.notna().all(), n
        assert T[n].prob.notna().all(), n
        oo = oo.sort_values(['game_play', 'p1', 'p2', 'step'])
        v = oo.groupby(['game_play', 'p1', 'p2'], sort=False)['prob'].transform(
            lambda s: s.rolling(a.w, min_periods=1, center=True).mean()).values
        m, thr = best_mcc(oo.contact.values, v)
        print('%-40s rawOOF %.5f smoothOOF %.5f thr %.4f' % (n, best_mcc(oo.contact.values, oo.prob.values)[0], m, thr))

    names = [n for n in names if n in O]
    # rank transforms
    OR = {n: rank01(O[n].sort_values(['game_play', 'p1', 'p2', 'step']).prob.values) for n in names}
    y = O[names[0]].sort_values(['game_play', 'p1', 'p2', 'step']).contact.values
    TR = {n: rank01(T[n].prob.values) for n in names}
    ysort_df = O[names[0]].sort_values(['game_play', 'p1', 'p2', 'step'])

    dfb = ysort_df[['game_play', 'p1', 'p2', 'step', 'contact']].copy()

    def eval_w(weights, want_thr=False):
        b = np.zeros(len(dfb)); tot = 0.0
        for n, wv in weights.items():
            if wv:
                b += wv * OR[n]; tot += wv
        b /= max(tot, 1e-9)
        dfb['prob'] = b
        d2 = dfb.sort_values(['game_play', 'p1', 'p2', 'step'])
        v = d2.groupby(['game_play', 'p1', 'p2'], sort=False)['prob'].transform(
            lambda s: s.rolling(a.w, min_periods=1, center=True).mean()).values
        m, t = best_mcc(d2.contact.values, v)
        return (m, t) if want_thr else m

    weights = {n: 1 for n in names}
    best = eval_w(weights)
    print('EQUAL BLEND (%d models) smoothOOF %.5f' % (len(names), best))
    if len(names) > 1:
        improved = True
        while improved:
            improved = False
            for n in names:
                for wv in [0, 1, 2, 3]:
                    if wv == weights[n]:
                        continue
                    trial = dict(weights); trial[n] = wv
                    if sum(trial.values()) == 0:
                        continue
                    m = eval_w(trial)
                    if m > best + 5e-6:
                        best = m; weights = trial; improved = True
        print('WEIGHTED-ac smoothOOF %.5f' % best)
    use = [n for n in names if weights[n] > 0]
    use_thr = eval_w(weights, want_thr=True)[1]
    use_mcc = best
    print('SELECTED', {k: v for k, v in weights.items() if v > 0}, 'OOF %.5f thr %.4f' % (best, use_thr))
    # calibration-free threshold: transfer OOF-optimal positive rate to test quantile
    ob = np.zeros(len(dfb)); tot = 0.0
    for n in use:
        ob += weights[n] * OR[n]; tot += weights[n]
    dfb['prob'] = ob / max(tot, 1e-9)
    dob = dfb.sort_values(['game_play', 'p1', 'p2', 'step'])
    ov = dob.groupby(['game_play', 'p1', 'p2'], sort=False)['prob'].transform(
        lambda s: s.rolling(a.w, min_periods=1, center=True).mean()).values
    pos_rate = float((ov >= use_thr).mean())
    # test blend + smoothing
    tdf = ss[['contact_id', 'game_play', 'p1', 'p2', 'step']].copy()
    tb = np.zeros(len(tdf)); tot = 0.0
    for n in use:
        tb += weights[n] * TR[n]; tot += weights[n]
    tdf['prob'] = tb / max(tot, 1e-9)
    tdf = tdf.sort_values(['game_play', 'p1', 'p2', 'step'])
    tv = tdf.groupby(['game_play', 'p1', 'p2'], sort=False)['prob'].transform(
        lambda s: s.rolling(a.w, min_periods=1, center=True).mean()).values
    test_thr = float(np.quantile(tv, 1.0 - pos_rate))
    print('OOF optimal pos_rate %.5f -> test_thr %.5f' % (pos_rate, test_thr))
    sub = pd.DataFrame({'contact_id': tdf.contact_id.values, 'contact': (tv >= test_thr).astype(int)})
    sub = ss[['contact_id']].merge(sub, on='contact_id', how='left')
    sub['contact'] = sub.contact.fillna(0).astype(int)
    sub.to_csv(B + a.out, index=False)
    print('SELECTED', use, 'OOF %.5f thr %.4f' % (use_mcc, use_thr))
    print('wrote', a.out, 'pos rate', sub.contact.mean())


if __name__ == '__main__':
    main()
