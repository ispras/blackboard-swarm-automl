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

def npz(name):
    d = np.load(f'{R}/pollux_beta/{name}'); return d['X'][np.argsort(d['ids'])]
A1 = npz('features_rawaudio.npz'); A2 = npz('features_audio2.npz')
BASE = M.base_feats(np.arange(322))
ae = np.load(f'{R}/pollux_epsilon/ast_emb.npz'); o = np.argsort(ae['ids'])
LOG = np.load(f'{R}/pollux_epsilon/ast_logits.npz')['probs'][o]
a12 = np.load(f'{R}/pollux_epsilon/ast1212.npz'); a14 = np.load(f'{R}/pollux_epsilon/ast1414.npz')
cl = np.load(f'{R}/pollux_epsilon/clap_emb.npz'); oc = np.argsort(cl['ids'])
CLAP = cl['emb'][oc]
d = np.load(f'{R}/pollux_epsilon/feats.npz')

# --- variant A: rich embeddings + acoustic PCA ---
AM = np.hstack([ae['mean'][o], ae['max'][o], a12['mean'], a14['mean'], CLAP,
                ae['std'][o], np.log1p(LOG), d['hist'], d['mil']])
P1 = PCA(64, random_state=0).fit_transform(StandardScaler().fit_transform(A1))
P2 = PCA(96, random_state=0).fit_transform(StandardScaler().fit_transform(A2))
XA = np.hstack([AM, BASE, P1, P2]).astype(np.float32)
# --- variant B: no AST, more acoustic (decorrelate) ---
XB = np.hstack([BASE, P1, P2, PCA(80, random_state=1).fit_transform(StandardScaler().fit_transform(A2)),
                PCA(40, random_state=2).fit_transform(StandardScaler().fit_transform(A1))]).astype(np.float32)
print('XA', XA.shape, 'XB', XB.shape)

class MLP(nn.Module):
    def __init__(self, d, h1=512, h2=256, p=0.5):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h1), nn.BatchNorm1d(h1), nn.GELU(), nn.Dropout(p),
                                 nn.Linear(h1, h2), nn.BatchNorm1d(h2), nn.GELU(), nn.Dropout(p * 0.8),
                                 nn.Linear(h2, 19))
    def forward(self, x): return self.net(x)

def train_one(Xtr, ytr, Xte, seed, epochs=220, mixup=0.4):
    torch.manual_seed(seed); np.random.seed(seed)
    m = MLP(Xtr.shape[1]); opt = torch.optim.AdamW(m.parameters(), lr=1.2e-3, weight_decay=3e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    lossf = nn.BCEWithLogitsLoss()
    Xt = torch.tensor(Xtr, dtype=torch.float32); yt = torch.tensor(ytr, dtype=torch.float32)
    n = len(Xt)
    for ep in range(epochs):
        m.train(); perm = torch.randperm(n)
        for i in range(0, n, 32):
            idx = perm[i:i + 32]
            xb, yb = Xt[idx], yt[idx]
            if mixup > 0 and len(idx) > 3:
                lam = np.random.beta(mixup, mixup)
                ridx = torch.randperm(len(idx))
                xb = lam * xb + (1 - lam) * xb[ridx]
                yb = lam * yb + (1 - lam) * yb[ridx]
            opt.zero_grad(); loss = lossf(m(xb), yb); loss.backward(); opt.step()
        sched.step()
    m.eval()
    with torch.no_grad():
        return torch.sigmoid(m(torch.tensor(Xte, dtype=torch.float32))).numpy()

def run(name, X, seeds=range(5), epochs=220):
    P = np.zeros((len(tr), 19)); T = np.zeros((len(te), 19))
    for k in range(5):
        va = np.where(folds == k)[0]; trn = np.where(folds != k)[0]
        tri = tr[trn]; vai = tr[va]
        Xtr = np.hstack([X[tri], M.neighbor_loo(tri)])
        Xva = np.hstack([X[vai], M.neighbor_between(vai, tri)])
        sc = StandardScaler().fit(Xtr); Xtr = sc.transform(Xtr); Xva = sc.transform(Xva)
        for s in seeds:
            P[va] += train_one(Xtr, Y[trn], Xva, 100 * s + k) / len(seeds)
        # test using this fold's model (train subset) -> average over folds/seeds
        Xtr2 = np.hstack([X[tri], M.neighbor_loo(tri)])
        Xte = np.hstack([X[te], M.neighbor_between(te, tri)])
        sc2 = StandardScaler().fit(Xtr2); Xtr2 = sc2.transform(Xtr2); Xte = sc2.transform(Xte)
        for s in seeds:
            T += train_one(Xtr2, Y[trn], Xte, 100 * s + k) / (5 * len(seeds))
        print(name, 'fold', k, 'done', flush=True)
    auc = roc_auc_score(Y.ravel(), P.ravel())
    np.save(f'{R}/pollux_beta/preds/oof_{name}.npy', P)
    np.save(f'{R}/pollux_beta/preds/test_{name}.npy', T)
    print(name, 'OOF', round(auc, 4), flush=True)
    return auc

if __name__ == '__main__':
    t = time.time()
    a = run('mlp_rich_mixup', XA, seeds=range(5))
    b = run('mlp_acoustic_mixup', XB, seeds=range(3))
    json.dump({'mlp_rich_mixup': a, 'mlp_acoustic_mixup': b},
              open(f'{R}/pollux_beta/preds/mlp2_meta.json', 'w'))
    print('total', round(time.time() - t, 1))
