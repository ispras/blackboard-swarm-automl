import sys, numpy as np, time, json, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/download_job_code/payload/sandbox/workspaces/altair_gamma')
import model_meta as M
import lightgbm as lgb
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

R = '/download_job_code/payload/sandbox/workspaces'
tr = M.tr; te = M.te; folds = M.folds; Y = M.Y[tr]
ae = np.load(f'{R}/pollux_epsilon/ast_emb.npz'); o = np.argsort(ae['ids'])
EMB = np.hstack([ae['mean'][o], ae['max'][o], ae['std'][o]])
LOG = np.load(f'{R}/pollux_epsilon/ast_logits.npz')['probs'][o]
cl = np.load(f'{R}/pollux_epsilon/clap_emb.npz'); CLAP = cl['emb'][np.argsort(cl['ids'])]
bn = np.load(f'{R}/pollux_epsilon/birdnet.npz')
FULL = bn['full_max']; BNL = bn['logits19']; BNP = 1 / (1 + np.exp(-BNL)); srt = np.sort(BNP, axis=1)
BNS = np.concatenate([BNP.mean(1), BNP.max(1), BNP.min(1), BNP.std(1), srt[:, -1], srt[:, -2]], axis=1)
BASE = M.base_feats(np.arange(322))
EMBp = PCA(96, random_state=0).fit_transform(StandardScaler().fit_transform(EMB))
CLAPp = PCA(64, random_state=1).fit_transform(StandardScaler().fit_transform(CLAP))
logidx = np.argsort(-LOG.std(0))[:150]
bnidx = np.argsort(-FULL.std(0))[:200]
X = np.hstack([BASE, EMBp, CLAPp, LOG[:, logidx], FULL[:, bnidx], BNS]).astype(np.float32)
print('X', X.shape)

def run(name, model_fn):
    P = np.zeros((len(tr), 19)); T = np.zeros((len(te), 19))
    for k in range(5):
        va = np.where(folds == k)[0]; trn = np.where(folds != k)[0]
        tri = tr[trn]; vai = tr[va]
        Xtr = np.hstack([X[tri], M.neighbor_loo(tri)])
        Xva = np.hstack([X[vai], M.neighbor_between(vai, tri)])
        Xte = np.hstack([X[te], M.neighbor_between(te, tri)])
        for c in range(19):
            y = Y[trn, c]
            if y.sum() == 0:
                continue
            m = model_fn(c); m.fit(Xtr, y)
            P[va, c] = m.predict_proba(Xva)[:, 1]
            T[:, c] += m.predict_proba(Xte)[:, 1] / 5
    auc = roc_auc_score(Y.ravel(), P.ravel())
    np.save(f'{R}/pollux_beta/preds/oof_{name}.npy', P)
    np.save(f'{R}/pollux_beta/preds/test_{name}.npy', T)
    print(name, 'OOF', round(auc, 4), flush=True)
    return auc

if __name__ == '__main__':
    t = time.time()
    res = {}
    res['lgb_bn'] = run('lgb_bn', lambda c: lgb.LGBMClassifier(
        n_estimators=400, learning_rate=0.03, num_leaves=7, min_child_samples=6,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.35, reg_lambda=1.0,
        n_jobs=8, verbose=-1, random_state=c))
    res['et_bn2'] = run('et_bn2', lambda c: ExtraTreesClassifier(
        n_estimators=400, max_features='sqrt', min_samples_leaf=1, n_jobs=8, random_state=c))
    json.dump(res, open(f'{R}/pollux_beta/preds/lgbbn_meta.json', 'w'))
    print('total', time.time() - t)
