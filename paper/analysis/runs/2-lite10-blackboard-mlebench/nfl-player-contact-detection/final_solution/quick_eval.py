import numpy as np, pandas as pd, json
from sklearn.metrics import roc_auc_score


def mcc_curve_curve(y, p):
    idx = np.argsort(-p)
    ys = y[idx].astype(np.float64)
    tp = np.cumsum(ys)
    fp = np.cumsum(1 - ys)
    pos = ys.sum(); neg = len(ys) - pos
    fn = pos - tp; tn = neg - fp
    den = np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    with np.errstate(divide='ignore', invalid='ignore'):
        m = np.where(den > 0, (tp*tn - fp*fn)/den, 0)
    k = int(np.argmax(m))
    return float(m[k]), float(np.sort(p)[::-1][k])


def counts_at(sorted_desc, y_sorted, thr):
    k = int(np.searchsorted(-sorted_desc, -thr, side='right'))
    tp = int(y_sorted[:k].sum())
    fp = k - tp
    pos = int(y_sorted.sum()); neg = len(y_sorted) - pos
    return tp, fp, pos - tp, neg - fp


op = pd.read_parquet('oof_pair.parquet')
og = pd.read_parquet('oof_ground.parquet')
yp = op.contact.values.astype(np.int8); pp = op.prob.values
yg = og.contact.values.astype(np.int8); pg = og.prob.values
res = {'auc_pair': float(roc_auc_score(yp, pp)), 'auc_ground': float(roc_auc_score(yg, pg))}
mp, tp = mcc_curve_curve(yp, pp)
mg, tg = mcc_curve_curve(yg, pg)
res.update(mcc_pair=mp, thr_pair=tp, mcc_ground=mg, thr_ground=tg)
print(res)

# joint threshold via counts on sorted arrays
order_p = np.argsort(-pp); sp = pp[order_p]; ysp = yp[order_p]
order_g = np.argsort(-pg); sg = pg[order_g]; ysg = yg[order_g]
gp = np.unique(np.quantile(pp, np.linspace(0.85, 0.9995, 200)))
gg = np.unique(np.quantile(pg, np.linspace(0.85, 0.9995, 200)))
cp = [counts_at(sp, ysp, t) for t in gp]
cg = [counts_at(sg, ysg, t) for t in gg]
best = (-1, 0, 0)
for i, a in enumerate(cp):
    for j, b in enumerate(cg):
        tp_ = float(a[0]+b[0]); fp_ = float(a[1]+b[1]); fn_ = float(a[2]+b[2]); tn_ = float(a[3]+b[3])
        den = np.sqrt((tp_+fp_)*(tp_+fn_)*(tn_+fp_)*(tn_+fn_))
        if den == 0: continue
        m = (tp_*tn_ - fp_*fn_)/den
        if m > best[0]:
            best = (m, float(gp[i]), float(gg[j]))
res.update(joint_mcc=best[0], joint_thr_pair=best[1], joint_thr_ground=best[2])
print('JOINT', best)
json.dump(res, open('cv_result.json', 'w'), indent=2)

# fallback submission with joint thresholds
tp_df = pd.read_parquet('testpred_pair.parquet')
tg_df = pd.read_parquet('testpred_ground.parquet')
sub = pd.read_csv('../../data/sample_submission.csv')
pr = pd.concat([tp_df[['contact_id','prob']], tg_df[['contact_id','prob']]])
sub = sub.merge(pr, on='contact_id', how='left')
thr = np.where(sub.contact_id.str.endswith('_G'), best[2], best[1])
sub['contact'] = (sub.prob.fillna(0) >= thr).astype(int)
sub[['contact_id','contact']].to_csv('submission_lgb_v1.csv', index=False)
print('submission pos rate', sub.contact.mean(), 'thr', best[1], best[2])
