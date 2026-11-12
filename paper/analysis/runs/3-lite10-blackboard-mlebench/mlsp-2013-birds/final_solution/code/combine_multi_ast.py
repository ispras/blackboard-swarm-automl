import numpy as np, os, json, warnings
warnings.filterwarnings('ignore')
src = open('/download_job_code/payload/sandbox/workspaces/pollux_epsilon/combine_neigh.py').read().split('res = {}')[0]
exec(src)
e12 = np.load(f'{ROOT}/ast1212.npz'); e14 = np.load(f'{ROOT}/ast1414.npz')
def blk(dd):
    return np.hstack([dd['mean'], dd['max'], np.log1p(dd['probs'])])
BASE = np.hstack([blk({'mean': a['mean'], 'max': a['max'], 'probs': L}), blk(e12), blk(e14),
                  hist, mil, meta_feats(np.arange(len(ids)))]).astype(np.float32)
print('BASE', BASE.shape, flush=True)

best = (0, None)
for C in [0.03, 0.1, 0.3, 1.0]:
    P = oof_lr(C, True, 'balanced'); auc = roc_auc_score(Ytr.ravel(), P.ravel())
    print('multiast LR C', C, round(auc, 4), flush=True)
    if auc > best[0]: best = (auc, C)
C = best[1]
np.save(f'{ROOT}/preds/oof_lr_multiastneigh.npy', oof_lr(C, True, 'balanced'))
np.save(f'{ROOT}/preds/test_lr_multiastneigh.npy', test_lr(C, True, 'balanced'))
print('saved multiastneigh', best, flush=True)
