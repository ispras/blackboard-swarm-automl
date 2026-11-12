"""Fine 2D per-type-threshold search on the locked blend, then emit a submission at those rates.
Only writes sub_fine_wg7wp3.csv (never final_solution)."""
import os, glob, json
import numpy as np, pandas as pd
from scipy.stats import rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', '..', 'data')
WS = os.path.join(HERE, '..')
cfg = json.load(open(os.path.join(HERE, 'ens_pertype2.json')))


def keys_of(cid):
    p = cid.str.split('_')
    return (p.str[0] + '_' + p.str[1]), p.str[2].astype(int), p.str[3], p.str[4]


def roll(vals, grp, w):
    return pd.Series(vals).groupby(grp).rolling(w, min_periods=1, center=True).mean().values


# resolve artifacts
arts = {}
for p in glob.glob(os.path.join(WS, '*', 'oof_pred*.parquet')):
    name = os.path.basename(os.path.dirname(p)) + '_' + os.path.basename(p).replace(
        'oof_pred', '').replace('.parquet', '').strip('_')
    arts[name] = (p, p.replace('oof_pred', 'test_pred'))

lab = pd.read_csv(os.path.join(DATA, 'train_labels.csv'), usecols=['contact_id', 'contact'])
ref = pd.read_parquet(os.path.join(HERE, 'oof_pred_feat5.parquet'),
                      columns=['contact_id']).merge(lab, on='contact_id', how='left')
ref['game_play'], ref['step'], ref['p1'], ref['p2'] = keys_of(ref.contact_id)
ref = ref.sort_values(['game_play', 'p1', 'p2', 'step']).reset_index(drop=True)
grp = ref.groupby(['game_play', 'p1', 'p2'], sort=False).ngroup().values
y = ref.contact.values.astype(np.float64); isg = (ref.p2 == 'G').values

R = {}
for n in set(cfg['wg']) | set(cfg['wp']):
    o = pd.read_parquet(arts[n][0]).drop_duplicates('contact_id').set_index('contact_id')
    v = o['prob'].reindex(ref.contact_id.values).values
    v = np.nan_to_num(v, nan=np.nanmedian(v)); R[n] = rankdata(v) / len(v)
bg = sum(R[n] * c for n, c in cfg['wg'].items()) / sum(cfg['wg'].values())
bp = sum(R[n] * c for n, c in cfg['wp'].items()) / sum(cfg['wp'].values())
sc = np.empty(len(bg))
gi = np.where(isg)[0]; pi = np.where(~isg)[0]
sc[gi] = roll(bg[gi], grp[gi], 7)
sc[pi] = roll(bp[pi], grp[pi], 3)


def curve(mask):
    idx = np.where(mask)[0]; pp = sc[idx]; yy = y[idx]
    o = np.argsort(-pp); yso = yy[o]
    tp = np.cumsum(yso); fp = np.cumsum(1 - yso)
    return tp, fp, pp[o]


tpG, fpG, _ = curve(isg); tpP, fpP, _ = curve(~isg)
PG = tpG[-1]; NG = len(tpG) - PG; PP = tpP[-1]; NP = len(tpP) - PP
P = PG + PP; N = NG + NP

# fine sampled grid (rates from 0.2% to 15%)
NGRID = 1500
qg = np.linspace(0.002, 0.15, NGRID)   # ground positive rate
qp = np.linspace(0.002, 0.15, NGRID)   # pair positive rate
ig = np.clip(np.round(qg * len(tpG)).astype(int) - 1, 0, len(tpG) - 1)
ip = np.clip(np.round(qp * len(tpP)).astype(int) - 1, 0, len(tpP) - 1)
TP = tpG[ig][:, None] + tpP[ip][None, :]
FP = fpG[ig][:, None] + fpP[ip][None, :]
FN = P - TP; TN = N - FP
den = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
M = np.where(den > 0, (TP * TN - FP * FN) / np.where(den > 0, den, 1), 0.0)
a, b = np.unravel_index(np.argmax(M), M.shape)
best_mcc = float(M[a, b]); rate_g = float(qg[a]); rate_p = float(qp[b])
print('FINE best OOF %.5f | ground rate %.5f pair rate %.5f' % (best_mcc, rate_g, rate_p))
json.dump(dict(mcc=best_mcc, rate_g=rate_g, rate_p=rate_p),
          open(os.path.join(HERE, 'fine_thr_wg7wp3.json'), 'w'))

# ---- build test submission at these rates ----
ss = pd.read_csv(os.path.join(DATA, 'sample_submission.csv'))
def tscore(w):
    s = np.zeros(len(ss))
    for n, c in w.items():
        t = pd.read_parquet(arts[n][1]).drop_duplicates('contact_id').set_index('contact_id')
        tv = t['prob'].reindex(ss.contact_id.values).values
        s += rankdata(np.nan_to_num(tv, nan=0.0)) / len(tv) * c
    return s / sum(w.values())
tg = tscore(cfg['wg']); tp_ = tscore(cfg['wp'])
td = pd.DataFrame({'contact_id': ss.contact_id})
td['game_play'], td['step'], td['p1'], td['p2'] = keys_of(td.contact_id)
isgt = (td.p2 == 'G').values
td = td.assign(prob=np.where(isgt, tg, tp_)).sort_values(
    ['game_play', 'p1', 'p2', 'step']).reset_index(drop=True)
g2 = td.groupby(['game_play', 'p1', 'p2'], sort=False).ngroup().values
isgt2 = (td.p2 == 'G').values
sm = np.empty(len(td))
gi2 = np.where(isgt2)[0]; pi2 = np.where(~isgt2)[0]
sm[gi2] = roll(td.prob.values[gi2], g2[gi2], 7)
sm[pi2] = roll(td.prob.values[pi2], g2[pi2], 3)
thr_g = float(np.quantile(sm[gi2], 1 - rate_g))
thr_p = float(np.quantile(sm[pi2], 1 - rate_p))
pred = np.where(isgt2, sm >= thr_g, sm >= thr_p).astype(int)
out = pd.DataFrame({'contact_id': td.contact_id.values, 'contact': pred})
out = ss[['contact_id']].merge(out, on='contact_id', how='left')
assert out.contact.notna().all() and len(out) == len(ss)
out.to_csv(os.path.join(HERE, 'sub_fine_wg7wp3.csv'), index=False)
out2 = out.copy(); out2['p2'] = out2.contact_id.str.rsplit('_', n=3).str[3]
print('test overall posrate %.5f | ground %.5f | pair %.5f' % (
    out2.contact.mean(), out2[out2.p2 == 'G'].contact.mean(),
    out2[out2.p2 != 'G'].contact.mean()))
