"""Consensus blender: weight search -> temporal smoothing -> joint thresholds -> submission.
Models are merged on contact_id; all OOF must use castor_alpha/folds.csv.
"""
import os, json, sys, itertools
import numpy as np
import pandas as pd
sys.path.insert(0, '.')
from postprocess_search import smooth, KEYS


def mcc_curve(y, p):
    idx = np.argsort(-p)
    ys = y[idx].astype(np.float64)
    tp = np.cumsum(ys); fp = np.cumsum(1 - ys)
    pos = ys.sum(); neg = len(ys) - pos
    fn = pos - tp; tn = neg - fp
    den = np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    with np.errstate(divide='ignore', invalid='ignore'):
        m = np.where(den > 0, (tp*tn - fp*fn)/den, -1)
    k = int(np.argmax(m))
    return float(m[k]), float(np.sort(p)[::-1][k])


def counts_sorted(ps, ys, thr):
    k = int(np.searchsorted(-ps, -thr, side='right'))
    tp = int(ys[:k].sum()); fp = k - tp
    pos = int(ys.sum()); neg = len(ys) - pos
    return tp, fp, pos - tp, neg - fp


def joint_mcc(pp, yp, pg, yg, nq=140):
    gp_t = np.unique(np.quantile(pp, np.linspace(0.85, 0.9995, nq)))
    gg_t = np.unique(np.quantile(pg, np.linspace(0.85, 0.9995, nq)))
    o = np.argsort(-pp); pps = pp[o]; yps = yp[o].astype(np.int64)
    o = np.argsort(-pg); pgs = pg[o]; ygs = yg[o].astype(np.int64)
    cp = [counts_sorted(pps, yps, t) for t in gp_t]
    cg = [counts_sorted(pgs, ygs, t) for t in gg_t]
    best = (-1, 0, 0)
    for i, a in enumerate(cp):
        for j, b in enumerate(cg):
            tp = float(a[0]+b[0]); fp = float(a[1]+b[1]); fn = float(a[2]+b[2]); tn = float(a[3]+b[3])
            den = np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
            if den == 0: continue
            m = (tp*tn - fp*fn)/den
            if m > best[0]: best = (m, float(gp_t[i]), float(gg_t[j]))
    return best


def load_model(oof_path, test_path, oof_col='prob', test_col='prob'):
    o = pd.read_parquet(oof_path)[['contact_id', oof_col]].rename(columns={oof_col: 'p'})
    t = pd.read_parquet(test_path)[['contact_id', test_col]].rename(columns={test_col: 'p'})
    return o, t


def fit_weights(truth, models, y, step=0.1, max_models=4, rng=None):
    """coarse simplex grid / random dirichlet; returns weights maximizing subtype MCC."""
    names = list(models.keys())
    k = len(names)
    P = np.stack([models[n] for n in names], axis=1)
    if k <= 4:
        vals = np.round(np.arange(0, 1.0001, step), 3)
        cands = [()]
        for _ in range(k - 1):
            cands = [c + (v,) for c in cands for v in vals if sum(c) + v <= 1.0001]
        cands = [c + (round(1 - sum(c), 3),) for c in cands]
    else:
        rng = np.random.default_rng(0)
        cands = [tuple(np.round(rng.dirichlet(np.ones(k)), 3)) for _ in range(500)]
        cands += [tuple(1.0 if i == j else 0.0 for i in range(k)) for j in range(k)]
    best = (-1, None, None)
    for w in cands:
        p = P @ np.asarray(w)
        m, thr = mcc_curve(y, p)
        if m > best[0]:
            best = (m, w, thr)
    # local refine around best
    bm, bw, bt = best
    for _ in range(2):
        for i in range(k):
            for j in range(k):
                if i == j: continue
                w2 = list(bw); d = min(step / 2, w2[j])
                if d <= 0: continue
                w2[i] += d; w2[j] -= d
                p = P @ np.asarray(w2)
                m, thr = mcc_curve(y, p)
                if m > bm:
                    bm, bw, bt = m, tuple(np.round(w2, 4)), thr
    return bm, bw, bt


def main(pair_models, ground_models, tag='blend'):
    tr_p = pd.read_parquet('oof_pair.parquet')[['contact_id','contact']]
    yp = tr_p.contact.values.astype(np.int8)
    pm = {}; pt = {}
    for name, (op, tp, oc, tc) in pair_models.items():
        o, t = load_model(op, tp, oc, tc)
        d = tr_p[['contact_id']].merge(o, on='contact_id', how='left')
        assert d.p.notna().all(), f'{name} oof missing rows'
        pm[name] = d.p.values.astype(np.float64); pt[name] = t
        print('pair model', name, 'oof', len(o), 'test', len(t))
    bp = fit_weights(None, pm, yp, step=0.1)
    print('PAIR weights', bp[1], 'mcc', round(bp[0],5))

    tr_g = pd.read_parquet('oof_ground.parquet')[['contact_id','contact']]
    yg = tr_g.contact.values.astype(np.int8)
    gm = {}; gt = {}
    for name, (op, tp, oc, tc) in ground_models.items():
        o, t = load_model(op, tp, oc, tc)
        d = tr_g[['contact_id']].merge(o, on='contact_id', how='left')
        if d.p.isna().any():
            print('WARN ground model', name, 'missing', int(d.p.isna().sum()), '-> fill median')
            d['p'] = d.p.fillna(d.p.median())
        gm[name] = d.p.values.astype(np.float64); gt[name] = t
        print('ground model', name, 'oof', len(o), 'test', len(t))
    bg = fit_weights(None, gm, yg, step=0.1)
    print('GROUND weights', bg[1], 'mcc', round(bg[0],5))

    P = np.stack([pm[n] for n in pm], axis=1); wp = np.asarray(bp[1])
    G = np.stack([gm[n] for n in gm], axis=1); wg = np.asarray(bg[1])
    p_pair = P @ wp; p_ground = G @ wg

    # smoothing + joint thresholds
    op = pd.read_parquet('oof_pair.parquet')[['contact_id','game_play','nfl_player_id_1','nfl_player_id_2','step','contact']]
    og = pd.read_parquet('oof_ground.parquet')[['contact_id','game_play','nfl_player_id_1','nfl_player_id_2','step','contact']]
    pp = op.copy(); pp['_p'] = p_pair
    gg = og.copy(); gg['_p'] = p_ground
    best = (-1, None)
    for wpi in [1,3,5,7]:
        if wpi == 1:
            dsp = pp.sort_values(KEYS, kind='mergesort').reset_index(drop=True); sp = dsp['_p'].values.astype(np.float64)
        else:
            dsp, sp = smooth(pp, p_pair, wpi); sp = sp.astype(np.float64)
        yps = dsp['contact'].values.astype(np.int8)
        for wgi in [1,5,9,15]:
            if wgi == 1:
                dsg = gg.sort_values(KEYS, kind='mergesort').reset_index(drop=True); sg = dsg['_p'].values.astype(np.float64)
            else:
                dsg, sg = smooth(gg, p_ground, wgi); sg = sg.astype(np.float64)
            ygs = dsg['contact'].values.astype(np.int8)
            r = joint_mcc(sp, yps, sg, ygs)
            print('  smooth', wpi, wgi, round(r[0],5), flush=True)
            if r[0] > best[0]:
                best = (r[0], (wpi, wgi, r[1], r[2]))
    m, (wpi, wgi, t1, t2) = best
    print('BEST JOINT', m, 'wp', wpi, 'wg', wgi, 'thr', t1, t2)

    # test predictions
    def test_blend(ptd, weights, w):
        base = None
        for name, wt in zip(weights, w):
            t = ptd[name][['contact_id','p']].rename(columns={'p': name})
            base = t if base is None else base.merge(t, on='contact_id', how='outer')
        cols = list(weights)
        base['_p'] = sum(base[c].fillna(0) * ww for c, ww in zip(cols, w))
        return base

    tp = test_blend(pt, list(pm), wp)
    tg = test_blend(gt, list(gm), wg)
    pkeys = pd.read_parquet('testpred_pair.parquet')[['contact_id','game_play','nfl_player_id_1','nfl_player_id_2','step']]
    gkeys = pd.read_parquet('testpred_ground.parquet')[['contact_id','game_play','nfl_player_id_1','nfl_player_id_2','step']]
    tp = pkeys.merge(tp[['contact_id','_p']], on='contact_id', how='left')
    tg = gkeys.merge(tg[['contact_id','_p']], on='contact_id', how='left')
    sub = pd.read_csv('../../data/sample_submission.csv')
    # pair
    dsp, sp_test = smooth(tp, tp['_p'].values, wpi)
    pair_out = pd.DataFrame({'contact_id': dsp['contact_id'].values, 'prob': sp_test})
    dsg, sg_test = smooth(tg, tg['_p'].values, wgi)
    ground_out = pd.DataFrame({'contact_id': dsg['contact_id'].values, 'prob': sg_test})
    pr = pd.concat([pair_out, ground_out])
    sub = sub.merge(pr, on='contact_id', how='left')
    sub['contact'] = (sub.prob.fillna(0) >= np.where(sub.contact_id.str.endswith('_G'), t2, t1)).astype(int)
    outp = f'submission_{tag}.csv'
    sub[['contact_id','contact']].to_csv(outp, index=False)

    # rate-matched variant: target OOF optimal positive rate (robust to fold-averaging shrinkage)
    dsp2, sp2 = smooth(pp, p_pair, wpi)
    dsg2, sg2 = smooth(gg, p_ground, wgi)
    r_pair = float((sp2 >= t1).mean()); r_ground = float((sg2 >= t2).mean())
    tgt_pair = 0.0115; tgt_ground = 0.045
    t1r = float(np.quantile(pair_out['prob'].values, 1 - tgt_pair))
    t2r = float(np.quantile(ground_out['prob'].values, 1 - tgt_ground))
    sub2 = sub.copy()
    sub2['contact'] = (sub2.prob.fillna(0) >= np.where(sub2.contact_id.str.endswith('_G'), t2r, t1r)).astype(int)
    outr = f'submission_{tag}_rate.csv'
    sub2[['contact_id','contact']].to_csv(outr, index=False)

    cfg = {'pair_models': list(pm), 'pair_weights': list(map(float, wp)),
           'ground_models': list(gm), 'ground_weights': list(map(float, wg)),
           'wp': wpi, 'wg': wgi, 'thr_pair': t1, 'thr_ground': t2, 'oof_joint_mcc': m,
           'oof_rate_pair': r_pair, 'oof_rate_ground': r_ground,
           'rate_thr_pair': t1r, 'rate_thr_ground': t2r,
           'test_pos_rate': float(sub.contact.mean()), 'test_pos_rate_ratematch': float(sub2.contact.mean())}
    json.dump(cfg, open(f'blend_{tag}.json','w'), indent=2)
    print('saved', outp, 'pos rate', sub.contact.mean(), 'ratematch', sub2.contact.mean(), 'nan', sub.prob.isna().sum())
    print('OOF rates', r_pair, r_ground, 'rate thr', t1r, t2r)
    print(cfg)


if __name__ == '__main__':
    pair_models = {}
    ground_models = {}
    # always-available models
    pair_models['track'] = ('oof_pair.parquet','testpred_pair.parquet','prob','prob')
    ground_models['track'] = ('oof_ground.parquet','testpred_ground.parquet','prob','prob')
    ground_models['cast_tab'] = ('../castor_alpha/ground_oof_tab.parquet','../castor_alpha/ground_test_tab.parquet','oof','pred')
    ground_models['ach'] = ('../achernar_sigma/oof_ground_lgbv1.parquet','../achernar_sigma/testpred_ground_lgbv1.parquet','pred','pred')
    # optional newer models
    if os.path.exists('oof_pair_h.parquet') and os.path.exists('testpred_pair_h.parquet'):
        pair_models['mine_h'] = ('oof_pair_h.parquet','testpred_pair_h.parquet','prob','prob')
    if os.path.exists('../achernar_sigma/oof_pair_lgbv2.parquet'):
        pair_models['ach_pair'] = ('../achernar_sigma/oof_pair_lgbv2.parquet',
                                   '../achernar_sigma/testpred_pair_lgbv2.parquet','pred','pred')
    elif os.path.exists('../achernar_sigma/oof_pair_lgbv1.parquet'):
        pair_models['ach_pair'] = ('../achernar_sigma/oof_pair_lgbv1.parquet',
                                   '../achernar_sigma/testpred_pair_lgbv1.parquet','pred','pred')
    if os.path.exists('../castor_alpha/ground_oof_video.parquet') and os.path.exists('../castor_alpha/ground_test_video.parquet'):
        ground_models['cast_vid'] = ('../castor_alpha/ground_oof_video.parquet',
                                     '../castor_alpha/ground_test_video.parquet','pred','pred')
    if os.path.exists('ground_oof_stack_r.parquet') and os.path.exists('ground_test_stack_r.parquet'):
        ground_models['stack_r'] = ('ground_oof_stack_r.parquet','ground_test_stack_r.parquet','prob','prob')
    if os.path.exists('../achernar_sigma/oof_ground_gfus.parquet'):
        ground_models['ach_gfus'] = ('../achernar_sigma/oof_ground_gfus.parquet',
                                     '../achernar_sigma/testpred_ground_gfus.parquet','pred','pred')
    if os.path.exists('../achernar_sigma/oof_ground_lgbv2.parquet'):
        ground_models['ach_gv2'] = ('../achernar_sigma/oof_ground_lgbv2.parquet',
                                    '../achernar_sigma/testpred_ground_lgbv2.parquet','pred','pred')
    if os.path.exists('../achernar_sigma/oof_pair_lgbv1.parquet'):
        pair_models['ach1'] = ('../achernar_sigma/oof_pair_lgbv1.parquet',
                               '../achernar_sigma/testpred_pair_lgbv1.parquet','pred','pred')
    # pair_stack intentionally excluded from final (meta-stacking leakage risk, tiny gain)
    if os.path.exists('../castor_alpha/oof_pair_video.parquet'):
        pair_models['cast_vid'] = ('../castor_alpha/oof_pair_video.parquet',
                                   '../castor_alpha/pair_test_video.parquet','pred','pred')
    if 'stack_r' in ground_models:
        keep = ('stack_r','ach_gfus','ach_gv2','cast_vid')
        ground_models = {k:v for k,v in ground_models.items() if k in keep}
    main(pair_models, ground_models, tag='consensus')
