"""MLP on Perch emb + BirdNETv3 emb + Perch/BN3 target probs + metadata. 5-fold OOF/test."""
import sys, os, numpy as np, pandas as pd, warnings
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
ids = np.concatenate([tr.rec_id.values, te.rec_id.values]); n = len(tr)

def emb(path, npca):
    d = np.load(path, allow_pickle=True)
    E = pd.DataFrame(d["emb"], index=d["rec_id"]).reindex(ids).values.astype(np.float32)
    E = StandardScaler().fit_transform(E)
    return PCA(n_components=npca, random_state=0).fit_transform(E).astype(np.float32)

def probs(oof, test):
    A = pd.read_csv(oof).set_index("rec_id").reindex(tr.rec_id.values)
    B = pd.read_csv(test).set_index("rec_id").reindex(te.rec_id.values)
    P = np.vstack([A[[f"pred_{i}" for i in range(19)]].values, B[[f"pred_{i}" for i in range(19)]].values])
    return np.log(np.clip(P, 1e-4, 1 - 1e-4) / (1 - np.clip(P, 1e-4, 1 - 1e-4))).astype(np.float32)

Xp = emb(f"{ROOT}/castor_rho/artifacts/perch2_emb.npz", 256)
Xb = emb(f"{ROOT}/castor_rho/artifacts/birdnet3_emb.npz", 128)
Pp = probs(f"{ROOT}/castor_rho/oof/perch_pc_max_oof.csv", f"{ROOT}/castor_rho/oof/perch_pc_max_test.csv")
Pb = probs(f"{ROOT}/castor_rho/oof/bn3_b3_max_oof.csv", f"{ROOT}/castor_rho/oof/bn3_b3_max_test.csv")

def meta_feats(M):
    d = pd.DataFrame({"site": M.site, "month": M.month.astype(str), "hour": M.hour.astype(str),
                      "year": M.year, "doy": pd.to_datetime(M.date, format="%Y%m%d").dt.dayofyear,
                      "minute": M.minute})
    d["site_month"] = d.site + "_" + d.month
    return d
Dtr, Dte = meta_feats(C.parse_meta(tr)), meta_feats(C.parse_meta(te))
cats, num = ["site", "month", "hour", "site_month"], ["year", "doy", "minute"]
allD = pd.concat([Dtr, Dte]); oh = pd.get_dummies(allD[cats], drop_first=False).astype(float)
Xm = StandardScaler().fit_transform(np.hstack([oh.values, allD[num].values.astype(float)])).astype(np.float32)
X = np.hstack([Xp, Xb, Pp, Pb, Xm]).astype(np.float32)
Xtr_all, Xte = X[:n], X[n:]
print("X", X.shape, flush=True)

class MLP(nn.Module):
    def __init__(self, din):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(din, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.45),
                                 nn.Linear(512, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.35),
                                 nn.Linear(128, 19))
    def forward(self, x): return self.net(x)

def train_fold(ti, vi, seed, epochs=80):
    torch.manual_seed(seed); np.random.seed(seed)
    Xt = torch.tensor(Xtr_all[ti]); Yt_ = torch.tensor(Y[ti].astype(np.float32))
    Xv = torch.tensor(Xtr_all[vi]).cuda(); Xs = torch.tensor(Xte).cuda()
    model = MLP(X.shape[1]).cuda()
    pos = Y[ti].sum(0); neg = len(ti) - pos
    pw = torch.tensor(np.clip(np.sqrt(neg / np.maximum(pos, 1)), 1, 8), dtype=torch.float32).cuda()
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=2e-3)
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
                r = torch.randperm(x.size(0)); x = lam * x + (1 - lam) * x[r]; y = lam * y + (1 - lam) * y[r]
            opt.zero_grad(); loss = crit(model(x), y); loss.backward(); opt.step()
        sched.step(); model.eval()
        with torch.no_grad(): pv = torch.sigmoid(model(Xv)).cpu().numpy()
        auc = C.global_auc(Y[vi].ravel(), pv.ravel())
        if auc > best:
            best = auc; bestp = pv
            with torch.no_grad(): bests = torch.sigmoid(model(Xs)).cpu().numpy()
    return bestp, bests, best

seed0 = int(sys.argv[1]) if len(sys.argv) > 1 else 0
tag = sys.argv[2] if len(sys.argv) > 2 else ""
O = np.zeros_like(Y, float); T = np.zeros((len(te), 19), float)
for k in range(5):
    ti = np.where(ftr != k)[0]; vi = np.where(ftr == k)[0]
    pv, pt, best = train_fold(ti, vi, seed=seed0 * 100 + k)
    O[vi] = pv; T += pt
    print(f"fold {k} best={best:.4f}", flush=True)
T /= 5
print("MLP-all OOF micro=%.4f macro=%.4f" % (C.global_auc(Y, O), C.per_class_auc(Y, O)[0]), flush=True)
os.makedirs(f"{WS}/oof", exist_ok=True)
for part, arr, rids in [("oof", O, tr.rec_id.values), ("test", T, te.rec_id.values)]:
    dd = pd.DataFrame(arr, columns=[f"pred_{i}" for i in range(19)]); dd.insert(0, "rec_id", rids)
    dd.to_csv(f"{WS}/oof/mlp_all{tag}_{part}.csv", index=False)
print("saved", flush=True)
