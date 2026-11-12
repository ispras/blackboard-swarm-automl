"""MLP on Perch embeddings + metadata, 5-fold OOF + test."""
import sys, numpy as np, pandas as pd, warnings, time
warnings.filterwarnings("ignore")
sys.path.insert(0, "/download_job_code/payload/sandbox/workspaces/arcturus_kappa")
import common as C
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import torch, torch.nn as nn

WS = "/download_job_code/payload/sandbox/workspaces/arcturus_kappa"
ROOT = "/download_job_code/payload/sandbox/workspaces"
tr, Y, te, Yt = C.load_labels()
cv = pd.read_csv(f"{ROOT}/castor_rho/shared/cv_folds.csv"); fm = dict(zip(cv.rec_id, cv.fold))
ftr = tr.rec_id.map(fm).values.astype(int)

d = np.load(f"{ROOT}/castor_rho/artifacts/perch2_emb.npz", allow_pickle=True)
E = pd.DataFrame(d["emb"], index=d["rec_id"])
ids = np.concatenate([tr.rec_id.values, te.rec_id.values])
E = E.reindex(ids).values.astype(np.float32)
# PCA fit on all 322 (unsupervised, transductive, no labels)
E = StandardScaler().fit_transform(E)
pca = PCA(n_components=256, random_state=0).fit(E)
Xe = pca.transform(E).astype(np.float32)
print("PCA evr", pca.explained_variance_ratio_.sum(), flush=True)

def meta_feats(M):
    d = pd.DataFrame({"site": M.site, "month": M.month.astype(str), "hour": M.hour.astype(str),
                      "year": M.year, "doy": pd.to_datetime(M.date, format="%Y%m%d").dt.dayofyear,
                      "minute": M.minute})
    d["site_month"] = d.site + "_" + d.month
    return d
Dtr, Dte = meta_feats(C.parse_meta(tr)), meta_feats(C.parse_meta(te))
cats, num = ["site", "month", "hour", "site_month"], ["year", "doy", "minute"]
allD = pd.concat([Dtr, Dte]); oh = pd.get_dummies(allD[cats], drop_first=False).astype(float)
Xm = np.hstack([oh.values, allD[num].values.astype(float)]).astype(np.float32)
Xm = StandardScaler().fit_transform(Xm).astype(np.float32)
X = np.hstack([Xe, Xm])
n = len(tr)
Xtr_all, Xte = X[:n], X[n:]

class MLP(nn.Module):
    def __init__(self, din):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(din, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.4),
                                 nn.Linear(512, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.3),
                                 nn.Linear(128, 19))
    def forward(self, x): return self.net(x)

def train_fold(ti, vi, seed, epochs=70):
    torch.manual_seed(seed); np.random.seed(seed)
    Xt = torch.tensor(Xtr_all[ti]); Yt_ = torch.tensor(Y[ti].astype(np.float32))
    Xv = torch.tensor(Xtr_all[vi]).cuda()
    Xs = torch.tensor(Xte).cuda()
    model = MLP(X.shape[1]).cuda()
    pos = Y[ti].sum(0); neg = len(ti) - pos
    pw = torch.tensor(np.clip(np.sqrt(neg / np.maximum(pos, 1)), 1, 8), dtype=torch.float32).cuda()
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best = -1; bestp = None; bests = None; bs = 32
    for ep in range(epochs):
        model.train(); perm = torch.randperm(len(ti))
        for b in range(0, len(ti), bs):
            idx = perm[b:b + bs]
            if len(idx) < 4: continue
            x = Xt[idx].cuda(); y = Yt_[idx].cuda()
            if np.random.rand() < 0.4:
                lam = np.random.beta(0.4, 0.4); lam = max(lam, 1 - lam)
                ridx = torch.randperm(x.size(0))
                x = lam * x + (1 - lam) * x[ridx]; y = lam * y + (1 - lam) * y[ridx]
            opt.zero_grad(); loss = crit(model(x), y); loss.backward(); opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            pv = torch.sigmoid(model(Xv)).cpu().numpy()
        auc = C.global_auc(Y[vi].ravel(), pv.ravel())
        if auc > best:
            best = auc; bestp = pv
            with torch.no_grad(): bests = torch.sigmoid(model(Xs)).cpu().numpy()
    return bestp, bests, best

O = np.zeros_like(Y, float); T = np.zeros((len(te), 19), float)
for k in range(5):
    ti = np.where(ftr != k)[0]; vi = np.where(ftr == k)[0]
    pv, pt, best = train_fold(ti, vi, seed=k)
    O[vi] = pv; T += pt
    print(f"fold {k} best micro={best:.4f}", flush=True)
T /= 5
print("MLP-perch OOF micro=%.4f macro=%.4f" % (C.global_auc(Y, O), C.per_class_auc(Y, O)[0]), flush=True)
import os; os.makedirs(f"{WS}/oof", exist_ok=True)
for part, arr, rids in [("oof", O, tr.rec_id.values), ("test", T, te.rec_id.values)]:
    dd = pd.DataFrame(arr, columns=[f"pred_{i}" for i in range(19)]); dd.insert(0, "rec_id", rids)
    dd.to_csv(f"{WS}/oof/mlp_perch_{part}.csv", index=False)
print("saved", flush=True)
