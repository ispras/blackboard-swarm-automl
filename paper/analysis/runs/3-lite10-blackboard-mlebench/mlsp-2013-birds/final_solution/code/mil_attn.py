"""Attention-MIL over BirdNET per-window embeddings (raw + denoised) + global features."""
import os, sys, time, json, numpy as np, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/download_job_code/payload/sandbox/workspaces/altair_gamma')
import model_meta as M
import torch, torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
torch.set_num_threads(4)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'

R = '/download_job_code/payload/sandbox/workspaces'
tr = M.tr; te = M.te; folds = M.folds; Y = M.Y[tr]
E15 = np.load(f'{R}/pollux_epsilon/birdnet_emb15.npy')            # 322,15,1024
E15d = np.load(f'{R}/pollux_epsilon/birdnet_emb15_denoised.npy')  # 322,15,1024
SEQ = np.concatenate([E15, E15d], axis=2).astype(np.float32)      # 322,15,2048
bn = np.load(f'{R}/pollux_epsilon/birdnet.npz'); BNL = bn['logits19']
bn15 = np.load(f'{R}/pollux_epsilon/birdnet15.npz')['logits19']
bnd = np.load(f'{R}/pollux_epsilon/birdnet_denoised.npz.npy')
def bnstats(L):
    P = 1 / (1 + np.exp(-L)); srt = np.sort(P, axis=1)
    return np.concatenate([P.mean(1), P.max(1), P.min(1), P.std(1), srt[:, -1], srt[:, -2]], axis=1)
BNS = np.concatenate([bnstats(BNL), bnstats(bn15), bnstats(bnd)], axis=1).astype(np.float32)
BASE = M.base_feats(np.arange(322))
# standardize sequence dims (unsupervised, all recs)
mu = SEQ.reshape(-1, SEQ.shape[2]).mean(0); sd = SEQ.reshape(-1, SEQ.shape[2]).std(0) + 1e-6
SEQ = (SEQ - mu) / sd
G = np.hstack([BNS, BASE]).astype(np.float32)
print('SEQ', SEQ.shape, 'G', G.shape, flush=True)

class AttnMIL(nn.Module):
    def __init__(self, dseq=2048, dg=0, h=256):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(dseq, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(0.3))
        self.pos = nn.Parameter(torch.zeros(1, 15, h))
        enc = nn.TransformerEncoderLayer(h, 4, 512, dropout=0.2, batch_first=True, activation='gelu')
        self.enc = nn.TransformerEncoder(enc, 1)
        self.att = nn.Sequential(nn.Linear(h, 128), nn.Tanh(), nn.Linear(128, 1))
        self.head = nn.Sequential(nn.LayerNorm(2 * h + dg), nn.Dropout(0.4),
                                  nn.Linear(2 * h + dg, 256), nn.GELU(), nn.Dropout(0.3), nn.Linear(256, 19))
        self.dg = dg
    def forward(self, x, g):
        h = self.enc(self.proj(x) + self.pos)
        a = torch.softmax(self.att(h).squeeze(-1), dim=1)
        pooled = (a.unsqueeze(-1) * h).sum(1)
        mx = h.max(1).values
        return self.head(torch.cat([pooled, mx, g], 1))

def train_one(Xtr, Gtr, ytr, Xva, Gva, seed, epochs=120):
    torch.manual_seed(seed); np.random.seed(seed)
    m = AttnMIL(dg=Gtr.shape[1]).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    lossf = nn.BCEWithLogitsLoss()
    Xt = torch.tensor(Xtr).to(dev); Gt = torch.tensor(Gtr).to(dev); yt = torch.tensor(ytr, dtype=torch.float32).to(dev)
    n = len(Xt)
    for ep in range(epochs):
        m.train(); perm = torch.randperm(n, device=dev)
        for i in range(0, n, 16):
            idx = perm[i:i + 16]
            if len(idx) < 4: continue
            xb, gb, yb = Xt[idx], Gt[idx], yt[idx]
            lam = float(np.random.beta(0.4, 0.4)); ridx = torch.randperm(len(idx), device=dev)
            xb = lam * xb + (1 - lam) * xb[ridx]
            gb = lam * gb + (1 - lam) * gb[ridx]
            yb = lam * yb + (1 - lam) * yb[ridx]
            opt.zero_grad(); loss = lossf(m(xb, gb), yb); loss.backward(); opt.step()
        sched.step()
    m.eval()
    with torch.no_grad():
        return torch.sigmoid(m(torch.tensor(Xva).to(dev), torch.tensor(Gva).to(dev))).cpu().numpy()

if __name__ == '__main__':
    t0 = time.time(); P = np.zeros((258, 19)); T = np.zeros((64, 19)); SEEDS = [0, 1]
    for k in range(5):
        va = np.where(folds == k)[0]; trn = np.where(folds != k)[0]
        tri = tr[trn]; vai = tr[va]
        Xtr = np.concatenate([SEQ[tri], M.neighbor_loo(tri)[:, None, :].repeat(1, 15, 1)], axis=2) if False else SEQ[tri]
        # add neighbours to global instead
        Gtr = np.hstack([G[tri], M.neighbor_loo(tri)])
        Gva = np.hstack([G[vai], M.neighbor_between(vai, tri)])
        Gte = np.hstack([G[te], M.neighbor_between(te, tri)])
        # scale global
        sc = StandardScaler().fit(Gtr); Gtr = sc.transform(Gtr); Gva = sc.transform(Gva); Gte = sc.transform(Gte)
        for s in SEEDS:
            P[va] += train_one(Xtr, Gtr, Y[trn], SEQ[vai], Gva, 10 * s + k) / len(SEEDS)
        for s in SEEDS:
            T += train_one(Xtr, Gtr, Y[trn], SEQ[te], Gte, 10 * s + k) / (5 * len(SEEDS))
        print('fold', k, 'done', round(time.time() - t0, 1), flush=True)
    auc = roc_auc_score(Y.ravel(), P.ravel())
    np.save(f'{R}/pollux_beta/preds/oof_mil_attn.npy', P); np.save(f'{R}/pollux_beta/preds/test_mil_attn.npy', T)
    json.dump({'oof': auc}, open(f'{R}/pollux_beta/preds/mil_attn_meta.json', 'w'))
    print('MIL-attn OOF', round(auc, 4), 'time', round(time.time() - t0, 1))
