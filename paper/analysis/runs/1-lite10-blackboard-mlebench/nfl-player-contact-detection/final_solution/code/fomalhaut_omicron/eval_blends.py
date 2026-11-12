import pandas as pd, numpy as np, glob, os

B = 'workspaces/fomalhaut_omicron/'


def best_mcc(y, p):
    o = np.argsort(-p); ys = y[o].astype(np.float64); ps = p[o]
    tp = np.cumsum(ys); fp = np.cumsum(1 - ys); P = float(y.sum()); N = len(y) - P
    fn = P - tp; tn = N - fp
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    m = np.where(den > 0, (tp * tn - fp * fn) / np.where(den > 0, den, 1), 0)
    i = int(np.argmax(m))
    return float(m[i]), float((ps[i] + ps[min(i + 1, len(ps) - 1)]) / 2)


def rank01(x):
    r = pd.Series(x).rank(method='average').values
    return (r - 1) / (len(r) - 1)


lab = pd.read_csv('data/train_labels.csv', usecols=['contact_id', 'contact', 'game_play', 'step',
                                                    'nfl_player_id_1', 'nfl_player_id_2'])
lab['p1'] = lab.nfl_player_id_1.astype(str); lab['p2'] = lab.nfl_player_id_2.astype(str)
cand = sorted(set(glob.glob(B + 'oof_pred*.parquet')) | set(glob.glob('workspaces/*/oof_pred*.parquet')))
OR = {}
for f in cand:
    tf = f.replace('oof_pred', 'test_pred')
    if not os.path.exists(tf):
        continue
    n = os.path.relpath(f, 'workspaces')
    o = pd.read_parquet(f).drop_duplicates('contact_id').rename(columns={'prob': 'prob'})
    d = lab[['contact_id', 'contact', 'game_play', 'step', 'p1', 'p2']].merge(o, on='contact_id', how='left')
    d = d.sort_values(['game_play', 'p1', 'p2', 'step'])
    OR[n] = rank01(d.prob.values)
    y = d.contact.values
    keys = d[['game_play', 'p1', 'p2', 'step', 'contact']]
print('loaded', list(OR))


def ev(w, sw=5):
    b = np.zeros(len(keys)); tot = 0
    for n, v in w.items():
        if v:
            b += v * OR[n]; tot += v
    b /= tot
    df = keys.copy(); df['prob'] = b
    df = df.sort_values(['game_play', 'p1', 'p2', 'step'])
    v = df.groupby(['game_play', 'p1', 'p2'], sort=False)['prob'].transform(
        lambda s: s.rolling(sw, min_periods=1, center=True).mean()).values
    m, t = best_mcc(df.contact.values, v)
    return m, t, float((v >= t).mean())


allnames = list(OR)
def K(frag):
    for n in allnames:
        if frag in n: return n
    raise KeyError(frag)
L2=K('procyon_theta/oof_pred_l2.'); GALLR=K('gallr'); GRU=K('procyon_theta/oof_pred_gru.'); GRUC=K('gruc')
F4=K('feat4'); F3=K('fomalhaut_omicron/oof_pred_feat3.'); F3D=K('feat3_d2'); SIR=K('sirius_tau/'); SEQ=K('procyon_theta/oof_pred_seq.'); SEQC=K('seqc'); CB=K('_cb.'); BASE=K('procyon_theta/oof_pred.parquet')
cands = {
 'ref_greedy': {L2:2, F4:1, GRUC:1, GALLR:1, GRU:1},
 'ref_plus_sirius': {L2:2, F4:1, GRUC:1, GALLR:1, GRU:1, SIR:1},
 'ref_plus_f3d': {L2:2, F4:1, GRUC:1, GALLR:1, GRU:1, F3D:1},
 'ref_plus_sirius_f3d': {L2:2, F4:1, GRUC:1, GALLR:1, GRU:1, SIR:1, F3D:1},
 'ref_plus_f3_sirius': {L2:2, F4:1, GRUC:1, GALLR:1, GRU:1, F3:1, SIR:1},
 'strong_equal': {L2:1, GALLR:1, F4:1, F3D:1, GRUC:1, GRU:1, SEQC:1, SIR:1},
 'L2x2_gallr2': {L2:2, GALLR:2, F4:1, GRUC:1, GRU:1},
 'L2x2_gallr2_sir': {L2:2, GALLR:2, F4:1, GRUC:1, GRU:1, SIR:1},
 'equal_all': {n:1 for n in allnames},
}
for name, w in cands.items():
    w = {k: v for k, v in w.items() if k in OR}
    for sw in [3, 5]:
        m, t, pr = ev(w, sw)
        print('%-22s sw=%d OOF %.5f thr %.4f posrate %.5f' % (name, sw, m, t, pr))
