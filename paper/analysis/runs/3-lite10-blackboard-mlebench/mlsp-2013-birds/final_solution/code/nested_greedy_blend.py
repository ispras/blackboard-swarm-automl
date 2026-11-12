import os, sys, glob, json, numpy as np, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/download_job_code/payload/sandbox/workspaces/altair_gamma')
import model_meta as M
from sklearn.metrics import roc_auc_score
R = '/download_job_code/payload/sandbox/workspaces'
tr = M.tr; folds = M.folds; Y = M.Y[tr]
EPS = 1e-6
def E(p): p = np.clip(p, EPS, 1 - EPS); return np.log(p / (1 - p))
def ga(Ym, S): return roc_auc_score(Ym.ravel(), np.asarray(S).ravel())

# derived blends to exclude
DERIVED = {'V1','V1_z','T1','T1_z','Q2','Q2b','Q2b_z','Q2_z','N1','N4','N7','N9','N2','N5',
           'avg_core5','cal_core5','blender_greedy','stack_lr','blend_histmil','BLEND','STACK',
           'BLEND_S1_bn','BLEND_S2_bn','BLEND_S3_bn','BLEND_B5','BLEND_A4','BLEND_F4meta',
           'M1','M2','M3','M4','M5','M6','M7','M8','M9','V3','V5','V8'}
MODELS = {}
for f in glob.glob(f'{R}/*/preds/oof_*.npy'):
    n = os.path.basename(f)[4:-4]
    if n in DERIVED:
        continue
    tf = f.replace('oof_', 'test_')
    if not os.path.exists(tf):
        continue
    o = np.load(f); t = np.load(tf)
    if o.shape != (258, 19) or t.shape != (64, 19):
        continue
    if ga(Y, o) < 0.90:
        continue
    MODELS[n] = (o, t)
names = list(MODELS)
LO = {n: E(MODELS[n][0]) for n in names}
LT = {n: E(MODELS[n][1]) for n in names}
ind = {n: ga(Y, LO[n]) for n in names}
print('pool', len(names))
for n in sorted(names, key=lambda x: -ind[x]):
    print(f'  {n:34s} {ind[n]:.4f}')

def subset_auc(sel, rows):
    P = np.mean([LO[n][rows] for n in sel], axis=0)
    return ga(Y[rows], P)

def greedy(rows, max_k=8, init=None):
    sel = list(init or [])
    best = subset_auc(sel, rows) if sel else -1
    while len(sel) < max_k:
        cand = None
        for n in names:
            if n in sel:
                continue
            v = subset_auc(sel + [n], rows)
            if cand is None or v > cand[1]:
                cand = (n, v)
        if cand and cand[1] > best + 1e-5:
            sel.append(cand[0]); best = cand[1]
        else:
            break
    return sel, best

# nested greedy
allrows = np.arange(len(Y))
Ps = np.zeros_like(Y, dtype=float); picks = []; fold_aucs = []
for f in range(5):
    va = np.where(folds == f)[0]; rr = np.where(folds != f)[0]
    sel, _ = greedy(rr)
    picks.append(sel)
    P = np.mean([LO[n][va] for n in sel], axis=0)
    Ps[va] = P; fold_aucs.append(ga(Y[va], P))
print('\nNESTED greedy pooled=%.4f meanfold=%.4f' % (ga(Y, Ps), np.mean(fold_aucs)))
for f, s in enumerate(picks):
    print(' fold', f, s)

# final greedy on all
sel, best = greedy(allrows)
print('\nFULL greedy', round(best, 4), sel)
P = np.mean([LT[n] for n in sel], axis=0)
np.save(f'{R}/pollux_beta/preds/oof_GREEDY.npy', np.mean([LO[n] for n in sel], axis=0))
np.save(f'{R}/pollux_beta/preds/test_GREEDY.npy', P)
json.dump({'sel': sel, 'oof': best}, open(f'{R}/pollux_beta/preds/greedy_meta.json', 'w'), indent=1)

# hand sets
def rep(name, sel):
    X = np.mean([LO[n] for n in sel], axis=0)
    per = [ga(Y[folds == f], X[folds == f]) for f in range(5)]
    print(f'{name:40s} pooled={ga(Y, X):.4f} mean={np.mean(per):.4f} min={min(per):.3f}')
hand = {
 '4fam': ['lr_multiast_bndn', 'mlp_rich_mixup', 'lgb_bn_stats', 'rawbirdnet_denoised'],
 '4fam+mlpbn': ['lr_multiast_bndn', 'mlp_rich_mixup', 'mlp_bn_mixup', 'lgb_bn_stats', 'rawbirdnet_denoised'],
 '4fam+bnemb': ['lr_multiast_bndn', 'lr_bnemb_richneigh', 'mlp_rich_mixup', 'lgb_bn_stats', 'rawbirdnet_denoised'],
 'V1like': ['lr_multiast_bnemb', 'lr_bnemb_only', 'mlp_rich_mixup', 'et_emb64log', 'lgb_emb64log', 'rawbirdnet_avg'],
 'v6': ['lr_multiast_bn_stats', 'mlp_rich_mixup', 'mlp_bn_mixup', 'lgb_bn', 'et_emb64log', 'rawbirdnet'],
 'mix6': ['lr_multiast_bndn', 'mlp_rich_mixup', 'mlp_bn_mixup', 'lgb_bn_stats', 'lgb_bn', 'rawbirdnet_denoised'],
}
print()
for k, v in hand.items():
    rep(k, v)
