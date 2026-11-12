"""Independently verify per-type-threshold gain on the final blend."""
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


lab = pd.read_csv(os.path.join(DATA, 'train_labels.csv'), usecols=['contact_id', 'contact'])
ref = pd.read_parquet(os.path.join(HERE, 'oof_pred_feat5.parquet'),
                      columns=['contact_id']).merge(lab, on='contact_id', how='left')
ref['game_play'], ref['step'], ref['p1'], ref['p2'] = keys_of(ref.contact_id)
ref = ref.sort_values(['game_play', 'p1', 'p2', 'step']).reset_index(drop=True)
grp = ref.groupby(['game_play', 'p1', 'p2'], sort=False).ngroup().values
y = ref.contact.values.astype(np.float64); isg = (ref.p2 == 'G').values

# resolve weight names -> files
arts = {}
for p in glob.glob(os.path.join(WS, '*', 'oof_pred*.parquet')):
    name = os.path.basename(os.path.dirname(p)) + '_' + os.path.basename(p).replace(
        'oof_pred', '').replace('.parquet', '').strip('_')
    arts[name] = p
need = set(cfg['wg']) | set(cfg['wp'])
R = {}
for n in need:
    o = pd.read_parquet(arts[n]).drop_duplicates('contact_id').set_index('contact_id')
    v = o['prob'].reindex(ref.contact_id.values).values
    v = np.nan_to_num(v, nan=np.nanmedian(v)); R[n] = rankdata(v) / len(v)
bg = sum(R[n] * c for n, c in cfg['wg'].items()) / sum(cfg['wg'].values())
bp = sum(R[n] * c for n, c in cfg['wp'].items()) / sum(cfg['wp'].values())
sc = np.empty(len(bg))
gi = np.where(isg)[0]; pi = np.where(~isg)[0]
sc[gi] = roll(bg[gi], grp[gi], cfg['wg_win'])
sc[pi] = roll(bp[pi], grp[pi], cfg['wp_win'])


def curve(mask):
    idx = np.where(mask)[0]; pp = sc[idx]; yy = y[idx]
    o = np.argsort(-pp); yso = yy[o]
    tp = np.cumsum(yso); fp = np.cumsum(1 - yso)
    return tp, fp, yy.sum(), len(yy) - yy.sum(), pp[o]


tpG, fpG, PG, NG, thrG = curve(isg)
tpP, fpP, PP, NP, thrP = curve(~isg)
P = PG + PP; N = NG + NP


def mcc(tp, fp):
    fn = P - tp; tn = N - fp
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return 0.0 if den == 0 else (tp * tn - fp * fn) / den


# single global
o = np.argsort(-sc); ys = y[o]
tp = np.cumsum(ys); fp = np.cumsum(1 - ys)
i = int(np.argmax([mcc(tp[k], fp[k]) for k in range(len(tp))]))
print('single global MCC %.5f' % mcc(tp[i], fp[i]))

best = (-1, 0, 0, 0, 0)
step = max(1, len(thrG) // 300)
for ig in range(0, len(thrG), step):
    tpg = tpG[ig]; fpg = fpG[ig]
    for ip in range(0, len(thrP), max(1, len(thrP) // 300)):
        m = mcc(tpg + tpP[ip], fpg + fpP[ip])
        if m > best[0]:
            best = (m, ig, ip, ig / len(thrG), ip / len(thrP))
m, ig, ip, qg, qp = best
print('per-type MCC %.5f | ground q=%.4f pair q=%.4f (rates %.4f / %.4f)' % (
    m, qg, qp, (ig + 1) / len(thrG), (ip + 1) / len(thrP)))

# check sirius-reported rates: ground positives 4.51%, pair positives 1.25%
for gq, pq in [(0.0451, 0.0125), (0.0467, 0.0133)]:
    ig2 = int(round(gq * len(thrG))) - 1; ip2 = int(round(pq * len(thrP))) - 1
    ig2 = min(max(ig2, 0), len(thrG) - 1); ip2 = min(max(ip2, 0), len(thrP) - 1)
    print('rates g=%.4f p=%.4f -> MCC %.5f' % (gq, pq, mcc(tpG[ig2] + tpP[ip2], fpG[ig2] + fpP[ip2])))
