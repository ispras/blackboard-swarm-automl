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
bn = np.load(f'{R}/pollux_epsilon/birdnet.npz'); BNL = bn['logits19']
bn15 = np.load(f'{R}/pollux_epsilon/birdnet15.npz')['logits19']
bnd = np.load(f'{R}/pollux_epsilon/birdnet_denoised.npz.npy')
def bnstats(L):
    P = 1/(1+np.exp(-L)); srt = np.sort(P, axis=1)
    return np.concatenate([P.mean(1), P.max(1), P.min(1), P.std(1), srt[:, -1], srt[:, -2]], axis=1)
BNS = np.concatenate([bnstats(BNL), bnstats(bn15), bnstats(bnd)], axis=1).astype(np.float32)
E15 = np.load(f'{R}/pollux_epsilon/birdnet_emb15.npy')
E15d = np.load(f'{R}/pollux_epsilon/birdnet_emb15_denoised.npy')
def emstats(E): return np.concatenate([E.mean(1), E.max(1), E.std(1)], axis=1)
BASE = M.base_feats(np.arange(322))
EMBp = PCA(96, random_state=0).fit_transform(StandardScaler().fit_transform(EMB))
CLAPp = PCA(64, random_state=1).fit_transform(StandardScaler().fit_transform(CLAP))
logidx = np.argsort(-LOG.std(0))[:150]
E15p = PCA(128, random_state=3).fit_transform(StandardScaler().fit_transform(emstats(E15)))
E15dp = PCA(128, random_state=4).fit_transform(StandardScaler().fit_transform(emstats(E15d)))
X = np.hstack([BASE, EMBp, CLAPp, LOG[:, logidx], BNS, E15p, E15dp]).astype(np.float32)
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
    res['lgb_bndn'] = run('lgb_bndn', lambda c: lgb.LGBMClassifier(
        n_estimators=400, learning_rate=0.03, num_leaves=7, min_child_samples=6,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.35, reg_lambda=1.0,
        n_jobs=8, verbose=-1, random_state=c))
    res['et_bndn'] = run('et_bndn', lambda c: ExtraTreesClassifier(
        n_estimators=400, max_features='sqrt', min_samples_leaf=1, n_jobs=8, random_state=c))
    json.dump(res, open(f'{R}/pollux_beta/preds/lgbbndn_meta.json', 'w'))
    print('total', time.time() - t)
