import os, sys, time, json, numpy as np, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/download_job_code/payload/sandbox/workspaces/altair_gamma')
import model_meta as M
import torch, torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
torch.set_num_threads(4)

R = '/download_job_code/payload/sandbox/workspaces'
tr = M.tr; te = M.te; folds = M.folds; Yfull = M.Y; Y = Yfull[tr]
torch.manual_seed(0); np.random.seed(0)

def npz(name):
    d = np.load(f'{R}/pollux_beta/{name}'); return d['X'][np.argsort(d['ids'])]
A1 = npz('features_rawaudio.npz'); A2 = npz('features_audio2.npz')
BASE = M.base_feats(np.arange(322))
ae = np.load(f'{R}/pollux_epsilon/ast_emb.npz'); o = np.argsort(ae['ids'])
LOG = np.load(f'{R}/pollux_epsilon/ast_logits.npz')['probs'][o]
d = np.load(f'{R}/pollux_epsilon/feats.npz')
ASTFULL = np.hstack([ae['mean'][o], ae['max'][o], ae['std'][o], np.log1p(LOG), d['hist'], d['mil']]).astype(np.float32)

# unsupervised PCA on acoustic features (no labels used)
P1 = PCA(n_components=64, random_state=0).fit_transform(StandardScaler().fit_transform(A1))
P2 = PCA(n_components=96, random_state=0).fit_transform(StandardScaler().fit_transform(A2))
X = np.hstack([ASTFULL, BASE, P1, P2]).astype(np.float32)
print('X', X.shape)

def full_with_neigh(use_neigh=True):
    if not use_neigh:
        return X, X
    Xtr_n = np.zeros((len(tr), M.NB), dtype=np.float32)
    Xte_n = np.zeros((len(te), M.NB), dtype=np.float32)
    Xtr_n = M.neighbor_loo(tr)
    Xte_n = M.neighbor_between(te, tr)
    return np.hstack([X[tr], Xtr_n]), np.hstack([X[te], Xte_n])

class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.5),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.4),
            nn.Linear(128, 19))
    def forward(self, x):
        return self.net(x)

def train_one(Xtr, ytr, Xva, Xte, seed, epochs=180):
    torch.manual_seed(seed)
    m = MLP(Xtr.shape[1])
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=3e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    lossf = nn.BCEWithLogitsLoss()
    Xt = torch.tensor(Xtr); yt = torch.tensor(ytr)
    n = len(Xt)
    for ep in range(epochs):
        m.train()
        perm = torch.randperm(n)
        for i in range(0, n, 32):
            idx = perm[i:i + 32]
            if len(idx) < 4:
                continue
            opt.zero_grad()
            loss = lossf(m(Xt[idx]), yt[idx])
            loss.backward(); opt.step()
        sched.step()
    m.eval()
    with torch.no_grad():
        pv = torch.sigmoid(m(torch.tensor(Xva))).numpy()
        pt = torch.sigmoid(m(torch.tensor(Xte))).numpy()
    return pv, pt

t0 = time.time()
P = np.zeros((len(tr), 19)); T = np.zeros((len(te), 19))
Xtr_full, Xte_full = full_with_neigh(True)
for k in range(5):
    va = np.where(folds == k)[0]; trn = np.where(folds != k)[0]
    tri = tr[trn]; vai = tr[va]
    # fold-safe neighbour features
    Xtr = np.hstack([X[tri], M.neighbor_loo(tri)])
    Xva = np.hstack([X[vai], M.neighbor_between(vai, tri)])
    sc = StandardScaler().fit(Xtr); Xtr = sc.transform(Xtr); Xva = sc.transform(Xva)
    for seed in range(3):
        pv, _ = train_one(Xtr, Y[trn], Xva, Xva[:1], seed)
        P[va] += pv / 3
    print('fold', k, 'done', round(time.time() - t0, 1), flush=True)

# test: refit on full train, but use fold models too for stability
for k in range(5):
    va = np.where(folds == k)[0]; trn = np.where(folds != k)[0]
    tri = tr[trn]
    Xtr = np.hstack([X[tri], M.neighbor_loo(tri)])
    Xte = np.hstack([X[te], M.neighbor_between(te, tri)])
    sc = StandardScaler().fit(Xtr); Xtr = sc.transform(Xtr); Xte = sc.transform(Xte)
    for seed in range(3):
        _, pt = train_one(Xtr, Y[trn], Xte[:1], Xte, seed)
        T += pt / 15

auc = roc_auc_score(Y.ravel(), P.ravel())
print('MLP OOF global', round(auc, 4))
np.save(f'{R}/pollux_beta/preds/oof_mlp_multilabel.npy', P)
np.save(f'{R}/pollux_beta/preds/test_mlp_multilabel.npy', T)
json.dump({'oof': auc}, open(f'{R}/pollux_beta/preds/mlp_meta.json', 'w'))
print('total', round(time.time() - t0, 1))
