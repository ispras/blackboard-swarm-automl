"""Per-type ensemble with per-type smoothing-window search + calibration-free threshold.
Discovers all workspaces/*/oof_pred*.parquet|test_pred*.parquet.
"""
import os, glob, json
import numpy as np, pandas as pd
from scipy.stats import rankdata
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', '..', 'data')
WS = os.path.join(HERE, '..')


def best_mcc(y, p):
    o = np.argsort(-p); ys = y[o].astype(np.float64); ps = p[o]
    tp = np.cumsum(ys); fp = np.cumsum(1 - ys); P = float(y.sum()); N = len(y) - P
    fn = P - tp; tn = N - fp
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    m = np.where(den > 0, (tp * tn - fp * fn) / np.where(den > 0, den, 1), 0)
    i = int(np.argmax(m))
    return float(m[i]), float((ps[i] + ps[min(i + 1, len(ps) - 1)]) / 2)


def keys_of(cid):
    p = cid.str.split('_')
    return (p.str[0] + '_' + p.str[1]), p.str[2].astype(int), p.str[3], p.str[4]


def discover():
    out = {}
    for p in glob.glob(os.path.join(WS, '*', 'oof_pred*.parquet')):
        name = os.path.basename(os.path.dirname(p)) + '_' + os.path.basename(p).replace(
            'oof_pred', '').replace('.parquet', '').strip('_')
        tp = p.replace('oof_pred', 'test_pred')
        if os.path.exists(tp):
            out[name] = (p, tp)
    return out


def roll(vals, grp, w):
    return pd.Series(vals).groupby(grp).rolling(w, min_periods=1, center=True).mean().values


def greedy(names, R, y, grp, mask, w=5, maxit=10):
    total = np.zeros(len(y)); cnt = 0; sel = []; best_overall = 0
    for it in range(maxit):
        best = (-1, None)
        for n in names:
            sc = (total + R[n]) / (cnt + 1)
            sm = roll(sc, grp, w); m, _ = best_mcc(y[mask], sm[mask])
            if m > best[0]:
                best = (m, n)
        if best[0] <= best_overall + 1e-7:
            break
        best_overall = best[0]; total = total + R[best[1]]; cnt += 1; sel.append(best[1])
    return Counter(sel), best_overall


def combine_smooth(bg, bp, grp, isg, wg, wp):
    out = np.empty(len(bg))
    gi = np.where(isg)[0]; pi = np.where(~isg)[0]
    out[gi] = roll(bg[gi], grp[gi], wg)
    out[pi] = roll(bp[pi], grp[pi], wp)
    return out


def main():
    arts = discover()
    print('artifacts', list(arts.keys()), flush=True)
    lab = pd.read_csv(os.path.join(DATA, 'train_labels.csv'), usecols=['contact_id', 'contact'])
    ref = pd.read_parquet(os.path.join(HERE, 'oof_pred_feat5.parquet'),
                          columns=['contact_id']).merge(lab, on='contact_id', how='left')
    ref['game_play'], ref['step'], ref['p1'], ref['p2'] = keys_of(ref.contact_id)
    ref = ref.sort_values(['game_play', 'p1', 'p2', 'step']).reset_index(drop=True)
    grp = ref.groupby(['game_play', 'p1', 'p2'], sort=False).ngroup().values
    y = ref.contact.values.astype(np.float64); isg = (ref.p2 == 'G').values
    R = {}; good = {}
    for name, (op, tp) in arts.items():
        try:
            o = pd.read_parquet(op).drop_duplicates('contact_id').set_index('contact_id')
            v = o['prob'].reindex(ref.contact_id.values).values
            v = np.nan_to_num(v, nan=np.nanmedian(v)); R[name] = rankdata(v) / len(v)
            good[name] = (op, tp)
        except Exception as e:
            print('SKIP', name, repr(e)[:60], flush=True)
    arts = good; names = list(R.keys())
    wg, mg = greedy(names, R, y, grp, isg)
    wp, mp = greedy(names, R, y, grp, ~isg)
    print('ground greedy', dict(wg), 'ground MCC %.5f' % mg, flush=True)
    print('pair greedy', dict(wp), 'pair MCC %.5f' % mp, flush=True)
    bg = sum(R[n] * c for n, c in wg.items()) / sum(wg.values())
    bp = sum(R[n] * c for n, c in wp.items()) / sum(wp.values())
    best = (-1, None)
    for a in (3, 5, 7, 9):
        for b in (3, 5, 7, 9):
            sc = combine_smooth(bg, bp, grp, isg, a, b)
            m, th = best_mcc(y, sc)
            if m > best[0]:
                best = (m, a, b, th)
            print('wg%d wp%d overall %.5f thr %.5f' % (a, b, m, th), flush=True)
    m, a, b, th = best
    print('BEST pertype2 %.5f wg%d wp%d thr %.5f' % (m, a, b, th), flush=True)
    # positive rate at OOF optimum
    sc = combine_smooth(bg, bp, grp, isg, a, b)
    pos_rate = float((sc >= th).mean())
    # test
    ss = pd.read_csv(os.path.join(DATA, 'sample_submission.csv'))

    def tscore(w):
        s = np.zeros(len(ss))
        for n, c in w.items():
            try:
                t = pd.read_parquet(arts[n][1]).drop_duplicates('contact_id').set_index('contact_id')
                tv = t['prob'].reindex(ss.contact_id.values).values
                s += rankdata(np.nan_to_num(tv, nan=0.0)) / len(tv) * c
            except Exception as e:
                print('SKIP test', n, repr(e)[:60], flush=True)
        return s / sum(w.values())
    td = pd.DataFrame({'contact_id': ss.contact_id})
    td['game_play'], td['step'], td['p1'], td['p2'] = keys_of(td.contact_id)
    isgt = (td.p2 == 'G').values
    tg = tscore(wg); tp_ = tscore(wp)
    sc_t = np.where(isgt, tg, tp_)
    td = td.assign(prob=sc_t).sort_values(['game_play', 'p1', 'p2', 'step']).reset_index(drop=True)
    g2 = td.groupby(['game_play', 'p1', 'p2'], sort=False).ngroup().values
    isgt2 = (td.p2 == 'G').values
    td['sm'] = combine_smooth(td.prob.values, td.prob.values, g2, isgt2, a, b)
    test_thr = float(np.quantile(td.sm.values, 1.0 - pos_rate))
    td['pred'] = (td.sm.values >= test_thr).astype(int)
    out = td.set_index('contact_id')['pred'].reindex(ss.contact_id).reset_index()
    out.columns = ['contact_id', 'contact']
    assert out.contact.notna().all() and len(out) == len(ss)
    out.to_csv(os.path.join(HERE, 'sub_pertype2.csv'), index=False)
    print('OOF pos_rate %.5f test_thr %.5f test posrate %.5f' % (pos_rate, test_thr, out.contact.mean()), flush=True)
    json.dump(dict(wg=dict(wg), wp=dict(wp), wg_win=a, wp_win=b, thr=th, mcc=m, pos_rate=pos_rate),
              open(os.path.join(HERE, 'ens_pertype2.json'), 'w'))


if __name__ == '__main__':
    main()
