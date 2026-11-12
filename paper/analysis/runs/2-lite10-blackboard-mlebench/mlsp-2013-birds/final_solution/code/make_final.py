"""Final submission builder for MLSP 2013 birds (arcturus_kappa).

Blend: per-class z-scored component predictions + class-prior offset.
Components (all published by the swarm on the shared 5-fold CV):
  perch  : castor_rho Perch v2 zero-shot, 5s@32k windows, max over windows
  bn     : castor_rho BirdNET h05 (3s@0.5s hop) max
  pp     : castor_rho Perch embedding probe (LR)
  mlpp   : arcturus_kappa MLP on Perch embeddings + metadata
  pw     : arcturus_kappa time-weighted session label prior (tau=300s, seq-scale=30)
"""
import sys, os, json, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/download_job_code/payload/sandbox/workspaces/arcturus_kappa")
import common as C
from scipy.stats import rankdata

ROOT = "/download_job_code/payload/sandbox/workspaces"
WS = f"{ROOT}/arcturus_kappa"
OUT = "/download_job_code/payload/sandbox/final_solution"
SHARED = f"{ROOT}/castor_rho/shared/cv_folds.csv"

tr, Y, te, Yt = C.load_labels()
cv = pd.read_csv(SHARED); fm = dict(zip(cv.rec_id, cv.fold))
ftr = tr.rec_id.map(fm).values

def load_long(path, ids):
    d = pd.read_csv(path).set_index("rec_id").reindex(ids)
    return d[[f"pred_{i}" for i in range(19)]].values.astype(float)

def z(P):
    Q = np.zeros_like(P, float)
    for c in range(P.shape[1]):
        sd = P[:, c].std()
        Q[:, c] = (P[:, c] - P[:, c].mean()) / (sd + 1e-9) if sd > 1e-12 else 0.0
    return Q

# component registry: name -> (oof path, test path, weight)
REG = {
    "perch": (f"{ROOT}/castor_rho/oof/perch_pcr_max_oof.csv", f"{ROOT}/castor_rho/oof/perch_pcr_max_test.csv", 1.0),
    "bn":    (f"{ROOT}/castor_rho/oof/birdnet_h05max_oof.csv", f"{ROOT}/castor_rho/oof/birdnet_h05max_test.csv", 0.2),
    "pp":    (f"{ROOT}/castor_rho/oof/perch_probe_oof.csv",    f"{ROOT}/castor_rho/oof/perch_probe_test.csv", 0.1),
    "bn3max": (f"{ROOT}/castor_rho/oof/bn3_b3_max_oof.csv",     f"{ROOT}/castor_rho/oof/bn3_b3_max_test.csv", 0.5),
    "bn3mean": (f"{ROOT}/castor_rho/oof/bn3_b3_mean_oof.csv",   f"{ROOT}/castor_rho/oof/bn3_b3_mean_test.csv", 0.3),
    "mlpp":  (f"{WS}/oof/mlp_perch_ens5_oof.csv",              f"{WS}/oof/mlp_perch_ens5_test.csv", 0.1),
    "mlpa":  (f"{WS}/oof/mlp_all_avg3_oof.csv",                f"{WS}/oof/mlp_all_avg3_test.csv", 0.3),
    "dn":    (f"{WS}/oof/perch_dn_oof.csv",                    f"{WS}/oof/perch_dn_test.csv", 0.15),
    "meta":  (f"{ROOT}/alderamin_phi/oof/meta_LR_oof.csv",     f"{ROOT}/alderamin_phi/oof/meta_LR_test.csv", 0.1),
}
LAM = 1.2
ALPHA = 0.8  # transductive prevalence estimate weight (rest = train prior)

O_tr, O_te = {}, {}
for k, (op, tp, w) in REG.items():
    assert os.path.exists(op) and os.path.exists(tp), (k, op, tp)
    O_tr[k] = load_long(op, tr.rec_id.values)
    O_te[k] = load_long(tp, te.rec_id.values)

# time-weighted session prior: honest OOF (siblings from other folds) and test (all train)
h = np.load(f"{WS}/priors_weighted.npz")
O_tr["pw"], O_te["pw"] = h["Pw"], h["Tw"]

Ztr = {k: z(v) for k, v in O_tr.items()}
Zte = {k: z(v) for k, v in O_te.items()}

prior = Y.mean(0)
lp = np.log(prior / (1 - prior))

def blend(Zd):
    B = np.zeros_like(next(iter(Zd.values())), float)
    for k, (_, _, w) in REG.items():
        B += w * Zd[k]
    B += 0.2 * Zd["pw"]
    return B + LAM * lp[None, :]

# transductive test prevalence estimate from two calibrated-ish MLP components
mlpp_tr = load_long(f"{WS}/oof/mlp_perch_ens5_oof.csv", tr.rec_id.values)
mlpbn_tr = load_long(f"{ROOT}/alderamin_phi/oof/mlp_embmeta_oof.csv", tr.rec_id.values)
mlpp_te = load_long(f"{WS}/oof/mlp_perch_ens5_test.csv", te.rec_id.values)
mlpbn_te = load_long(f"{ROOT}/alderamin_phi/oof/mlp_embmeta_test.csv", te.rec_id.values)

def offset(est_prevalence):
    e = np.clip(est_prevalence, 1e-3, 1 - 1e-3)
    return (1 - ALPHA) * lp + ALPHA * np.log(e / (1 - e))

# OOF components' offset uses fold-restricted estimated prevalence for the honest check;
# for the full-OOF report we use train prior only (ALPHA is for the transductive test setting).
So = blend(Ztr)
est_te = 0.5 * (mlpp_te.mean(0) + mlpbn_te.mean(0))
St = blend(Zte) - LAM * lp[None, :] + LAM * offset(est_te)[None, :]

print("FINAL BLEND")
print("OOF micro (train-prior offset)=%.4f macro=%.4f" % (C.global_auc(Y, So), C.per_class_auc(Y, So)[0]))
print("test estimated prevalence:", np.round(est_te, 3))
for k in ["perch", "bn", "bn3max", "bn3mean", "pp", "mlpp", "mlpa", "dn", "meta", "pw"]:
    print(f"  {k:6s} micro={C.global_auc(Y, O_tr[k]):.4f} macro={C.per_class_auc(Y, O_tr[k])[0]:.4f}")

# honest small-set check: per shared fold, z inside fold, prior from other folds
from sklearn.metrics import roc_auc_score
fold_scores = []
for k in range(5):
    vi = np.where(ftr == k)[0]; ti = np.where(ftr != k)[0]
    Zf = {kk: z(O_tr[kk][vi]) for kk in list(REG) + ["pw"]}
    Bf = sum(w * Zf[kk] for kk, (_, _, w) in REG.items()) + 0.2 * Zf["pw"]
    p = np.clip(Y[ti].mean(0), 1e-3, 1 - 1e-3); l = np.log(p / (1 - p))
    est = 0.5 * (mlpp_tr[vi].mean(0) + mlpbn_tr[vi].mean(0))
    est = np.clip(est, 1e-3, 1 - 1e-3); le = np.log(est / (1 - est))
    off = (1 - ALPHA) * l + ALPHA * le
    S = Bf + LAM * off[None, :]
    fold_scores.append(roc_auc_score(Y[vi].ravel(), S.ravel()))
print("honest per-fold micro:", np.round(fold_scores, 4), "mean=%.4f" % np.mean(fold_scores))

# submission: sigmoid is a global monotone map -> preserves AUC
def make_submission(S, path):
    rows = []
    for i, rid in enumerate(te.rec_id.values):
        for c in range(19):
            rows.append((int(rid) * 100 + c, float(1 / (1 + np.exp(-S[i, c])))))
    d = pd.DataFrame(rows, columns=["Id", "Probability"]).sort_values("Id")
    d.to_csv(path, index=False)
    return d

os.makedirs(OUT, exist_ok=True)
d = make_submission(St, f"{WS}/final_submission.csv")
sample = pd.read_csv("/download_job_code/payload/sandbox/data/sample_submission.csv")
assert list(d.columns) == list(sample.columns)
assert len(d) == len(sample) and set(d.Id) == set(sample.Id)
assert d.Probability.between(0, 1).all()
print("submission OK", d.shape, "mean prob by class:", np.round(d.Probability.values.reshape(-1, 19).mean(0), 3))

# also save OOF/test score matrices for the swarm
np.savez(f"{WS}/final_blend.npz", So=So, St=St, Y=Y, fold_scores=np.array(fold_scores))
print("saved final_submission.csv and final_blend.npz")
