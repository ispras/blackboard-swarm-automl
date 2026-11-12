import pandas as pd, numpy as np, lightgbm as lgb, argparse, json, gc, re
from sklearn.model_selection import GroupKFold

META = {'contact_id', 'game_play', 'datetime', 'nfl_player_id_1', 'nfl_player_id_2',
        'contact', 'p1', 'p2', 'game_key'}
CAT = ['is_ground', 'pos1_cat', 'pos2_cat']


def best_mcc(y, p):
    order = np.argsort(-p)
    ys = y[order].astype(np.float64); ps = p[order]
    tp = np.cumsum(ys); fp = np.cumsum(1.0 - ys)
    P = float(y.sum()); N = float(len(y)) - P
    fn = P - tp; tn = N - fp
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = np.where(denom > 0, (tp * tn - fp * fn) / np.where(denom > 0, denom, 1.0), 0.0)
    i = int(np.argmax(mcc))
    lo = ps[i]; hi = ps[min(i + 1, len(ps) - 1)]
    return float(mcc[i]), float((lo + hi) / 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--feat', default='feat2')
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--rounds', type=int, default=3000)
    ap.add_argument('--leaves', type=int, default=160)
    ap.add_argument('--lr', type=float, default=0.03)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--group', default='game')
    ap.add_argument('--smooth', type=int, default=5)
    ap.add_argument('--tag', default='')
    ap.add_argument('--sub', default='sub_cv')
    a = ap.parse_args()
    d = pd.read_parquet('workspaces/fomalhaut_omicron/cache/{}_{}.parquet'.format(a.feat, 'train'))
    d['is_ground'] = (d.p2 == 'G').astype(np.int8)
    d['game_key'] = d.game_play.str.split('_').str[0]
    feats = [c for c in d.columns if c not in META]
    y = d['contact'].values
    groups = d['game_key'].values if a.group == 'game' else d['game_play'].values
    params = dict(objective='binary', learning_rate=a.lr, num_leaves=a.leaves,
                  min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8,
                  bagging_freq=1, lambda_l2=1.0, n_jobs=48, seed=a.seed, verbose=-1, metric='auc')
    oof = np.zeros(len(d))
    iters = []
    gkf = GroupKFold(n_splits=a.folds)
    for k, (tri, vai) in enumerate(gkf.split(d, y, groups)):
        dtr = lgb.Dataset(d.iloc[tri][feats], y[tri], categorical_feature=[c for c in CAT if c in feats])
        dva = lgb.Dataset(d.iloc[vai][feats], y[vai], categorical_feature=[c for c in CAT if c in feats], reference=dtr)
        m = lgb.train(params, dtr, num_boost_round=a.rounds, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(100), lgb.log_evaluation(300)])
        oof[vai] = m.predict(d.iloc[vai][feats], num_iteration=m.best_iteration)
        iters.append(m.best_iteration)
        mm, tt = best_mcc(y[vai], oof[vai])
        print('fold', k, 'iter', m.best_iteration, 'fold MCC %.5f' % mm, 'thr %.4f' % tt)
        m.save_model('workspaces/fomalhaut_omicron/cache/cv_%s_%d.txt' % (a.feat + a.tag, k))
        del dtr, dva, m; gc.collect()
    np.save('workspaces/fomalhaut_omicron/cache/oof_%s.npy' % (a.feat + a.tag), oof)
    mcc, thr = best_mcc(y, oof)
    print('OOF MCC %.5f thr %.5f mean_iter %d' % (mcc, thr, int(np.mean(iters))))
    res = {'oof_mcc': mcc, 'thr': thr, 'iters': iters, 'feats': feats}
    json.dump(res, open('workspaces/fomalhaut_omicron/cache/cv_%s.json' % (a.feat + a.tag), 'w'), indent=1)

    # ---- postprocessing: rolling-mean smoothing on OOF ----
    oofdf = d[['game_play', 'p1', 'p2', 'step', 'contact']].copy()
    oofdf['prob'] = oof
    oofdf = oofdf.sort_values(['game_play', 'p1', 'p2', 'step'])
    g = oofdf.groupby(['game_play', 'p1', 'p2'], sort=False)['prob']
    bestw = (-1, 0, 0)
    for w in [1, 3, 5, 7, 9]:
        pp = g.transform(lambda s: s.rolling(w, min_periods=1, center=True).mean()).values
        m2, t2 = best_mcc(oofdf['contact'].values, pp)
        print('smoothmean w=%d MCC %.5f thr %.5f' % (w, m2, t2))
        if m2 > bestw[0]:
            bestw = (m2, t2, w)
    print('BEST smooth w=%d MCC %.5f thr %.5f' % (bestw[2], bestw[0], bestw[1]))
    res['smooth_w'] = bestw[2]; res['smooth_thr'] = bestw[1]; res['smooth_mcc'] = bestw[0]
    json.dump(res, open('workspaces/fomalhaut_omicron/cache/cv_%s.json' % (a.feat + a.tag), 'w'), indent=1)

    if a.sub:
        dt = pd.read_parquet('workspaces/fomalhaut_omicron/cache/{}_{}.parquet'.format(a.feat, 'test'))
        dt['is_ground'] = (dt.p2 == 'G').astype(np.int8)
        dt = dt.sort_values(['game_play', 'p1', 'p2', 'step'])
        pt = np.zeros(len(dt))
        for k in range(a.folds):
            m = lgb.Booster(model_file='workspaces/fomalhaut_omicron/cache/cv_%s_%d.txt' % (a.feat + a.tag, k))
            pt += m.predict(dt[feats], num_iteration=iters[k]) / a.folds
        np.save('workspaces/fomalhaut_omicron/cache/ptest_%s.npy' % (a.feat + a.tag), pt)
        np.save('workspaces/fomalhaut_omicron/cache/ptest_ids_%s.npy' % (a.feat + a.tag), dt.contact_id.values)
        w = bestw[2]
        dt = dt.assign(_p=pt)
        psm = dt.groupby(['game_play', 'p1', 'p2'], sort=False)['_p'].transform(
            lambda s: s.rolling(w, min_periods=1, center=True).mean()).values
        sub = pd.DataFrame({'contact_id': dt.contact_id, 'contact': (psm >= bestw[1]).astype(int)})
        sub = sub.set_index('contact_id').loc[pd.read_csv('data/sample_submission.csv').contact_id].reset_index()
        sub.to_csv('workspaces/fomalhaut_omicron/%s.csv' % a.sub, index=False)
        print('sub saved', a.sub, 'pos rate', sub.contact.mean())


if __name__ == '__main__':
    main()
