import numpy as np, pandas as pd, os, json, argparse
from scipy.optimize import minimize

BASE = "/download_job_code/payload/sandbox"
ME = os.path.join(BASE, "workspaces/bellatrix_omicron")
CAN = os.path.join(BASE, "workspaces/canopus_omega")
ALD = os.path.join(BASE, "workspaces/aldebaran_eta")
T = 80

p = argparse.ArgumentParser()
p.add_argument('--out', type=str, default=os.path.join(ME, 'submission_candidate.csv'))
p.add_argument('--final_dir', type=str, default='')
p.add_argument('--fit_fold', type=int, default=0)
p.add_argument('--val_fold', type=int, default=3)
args = p.parse_args()

tr = pd.read_csv(os.path.join(BASE, "data/train.csv"), usecols=["breath_id", "u_out", "pressure", "R", "C"])
y = tr.pressure.values.astype(np.float64)
insp = (tr.u_out.values == 0)
n = len(y)
bid_row = np.repeat(np.load(os.path.join(ME, "breath_train.npy")), T)
foldrow = (bid_row % 5).astype(np.int64)

def get_test(path):
    return np.load(path).reshape(-1).astype(np.float64)

def partial_oof(oofp, idxp):
    o = np.load(oofp).reshape(-1).astype(np.float64)
    idx = np.load(idxp).astype(np.int64)
    full = np.full(n, np.nan)
    rows = (idx[:, None] * T + np.arange(T)[None, :]).reshape(-1)
    full[rows] = o
    return full

# ---------- member: 50-feat family (canopus merged OOF; augment test with all-data seeds)
b50_oof = np.load(os.path.join(CAN, "runs/combined50_oof.npy")).reshape(-1).astype(np.float64)
base50_test = get_test(os.path.join(CAN, "runs/combined50_test.npy"))
all_data_tests = []
for name in ["test_physall_f0.npy", "test_physall2_f0.npy", "test_physall3_f0.npy"]:
    fp = os.path.join(ME, name)
    if os.path.exists(fp):
        all_data_tests.append(get_test(fp))
if all_data_tests:
    b50_test = 0.75*base50_test + 0.25*np.mean(all_data_tests, axis=0)
else:
    b50_test = base50_test
print(f"b50 test = 0.75*combined50 + 0.25*mean({len(all_data_tests)} all-data models)")

# ---------- member: 34-feat
c34_oof = np.load(os.path.join(CAN, "runs/combined34_oof.npy")).reshape(-1).astype(np.float64)
c34_test = get_test(os.path.join(CAN, "runs/combined34_test.npy"))

# ---------- member: aldebaran t3 (prefer merged 5-fold files; add all-data if present)
ald_oof_p = os.path.join(ALD, "preds/oof_t3_5fold.npy")
ald_test_p = os.path.join(ALD, "preds/test_t3_5fold.npy")
if os.path.exists(ald_oof_p) and os.path.exists(ald_test_p):
    ald_oof = get_test(ald_oof_p); ald_base_test = get_test(ald_test_p); ald_folds = [0,1,2,3,4]
    print("ald: using merged 5-fold oof/test")
else:
    ald_specs = {0: ("fold0_oof_f0_t3_120.npy", "fold0_idx_f0_t3_120.npy", "fold0_test_f0_t3_120.npy"),
                 1: ("fold1_oof_t3_150.npy", "fold1_idx_t3_150.npy", "fold1_test_t3_150.npy"),
                 2: ("fold2_oof_t3_150.npy", "fold2_idx_t3_150.npy", "fold2_test_t3_150.npy"),
                 3: ("fold3_oof_t3_150.npy", "fold3_idx_t3_150.npy", "fold3_test_t3_150.npy"),
                 4: ("fold4_oof_t3_150.npy", "fold4_idx_t3_150.npy", "fold4_test_t3_150.npy")}
    ald_oof = np.full(n, np.nan); ald_tests = []; ald_folds = []
    for f in range(5):
        oo, ii, tt = [os.path.join(ALD, "preds", x) for x in ald_specs[f]]
        if not (os.path.exists(oo) and os.path.exists(ii) and os.path.exists(tt)):
            print("ald missing fold", f); continue
        full = partial_oof(oo, ii); m = ~np.isnan(full); ald_oof[m] = full[m]
        ald_tests.append(get_test(tt)); ald_folds.append(f)
    ald_base_test = np.mean(ald_tests, axis=0)
ald_all_p = os.path.join(ALD, "preds/test_t3_all.npy")
if os.path.exists(ald_all_p):
    ald_test = 0.75*ald_base_test + 0.25*get_test(ald_all_p)
    print("ald: test = 0.75*5fold + 0.25*all-data")
else:
    ald_test = ald_base_test
print("ald folds used:", ald_folds)

members = {"b50": (b50_oof, b50_test), "c34": (c34_oof, c34_test), "ald": (ald_oof, ald_test)}
names = list(members)
for k in names:
    m = insp & ~np.isnan(members[k][0])
    print(f"  {k}: coverage {m.sum()} OOF MAE {np.abs(members[k][0][m]-y[m]).mean():.5f}")

def simplex(A, b):
    k = A.shape[1]
    f = lambda w: np.abs(A @ w - b).mean()
    r = minimize(f, np.ones(k)/k, method='SLSQP', bounds=[(0, 1)]*k,
                 constraints=[{'type': 'eq', 'fun': lambda w: w.sum()-1}],
                 options={'maxiter': 500, 'ftol': 1e-12})
    return r.x, float(r.fun)

def masks_for(fold):
    mm = insp & (foldrow == fold)
    for k in names:
        mm = mm & ~np.isnan(members[k][0])
    return mm

# folds on which ALL members have OOF
cov_folds = [f for f in range(5) if masks_for(f).sum() > 10000]
print("covered folds for weight fitting:", cov_folds)
# cross-validated weight check: fit on all but f, eval on f
cv = {}
for f in cov_folds:
    mf = np.zeros(n, bool)
    for g in cov_folds:
        if g != f: mf |= (foldrow == g)
    mf &= insp
    for k in names: mf &= ~np.isnan(members[k][0])
    mv = masks_for(f)
    Af = np.stack([members[k][0][mf] for k in names], 1); bf = y[mf]
    Av = np.stack([members[k][0][mv] for k in names], 1); bv = y[mv]
    wf, _ = simplex(Af, bf)
    cv[f] = float(np.abs(Av@wf-bv).mean())
    print(f"  CV hold f{f}: {cv[f]:.5f} w={np.round(wf,3)}")
print("CV mean", float(np.mean(list(cv.values()))))

# final weights fit on ALL covered folds
mf = np.zeros(n, bool)
for g in cov_folds: mf |= (foldrow == g)
mf &= insp
for k in names: mf &= ~np.isnan(members[k][0])
Af = np.stack([members[k][0][mf] for k in names], 1); bf = y[mf]
w_eq = np.ones(len(names)) / len(names)
w, fv = simplex(Af, bf)
print(f"FINAL weights fit on {len(cov_folds)} folds: EQ {np.abs(Af@w_eq-bf).mean():.5f} OPT {fv:.5f}")
print("weights", {k: round(float(x), 4) for k, x in zip(names, w)})
mall = mf.copy()
Aall = np.stack([members[k][0][mall] for k in names], 1)
oof_mae = float(np.abs(Aall @ w - y[mall]).mean())
print(f"FULL OOF blend MAE {oof_mae:.5f} on {mall.sum()} insp rows")

test_pred = sum(w[i] * members[k][1] for i, k in enumerate(names))
te = pd.read_csv(os.path.join(BASE, "data/test.csv"), usecols=["id"])
assert len(test_pred) == len(te) == 603600
sub = pd.DataFrame({"id": te.id.values, "pressure": test_pred})
sub.to_csv(args.out, index=False)
print("wrote", args.out, "pred range", float(test_pred.min()), float(test_pred.max()))

report = {"members": names, "weights": [float(x) for x in w],
          "fit_folds": cov_folds,
          "cv_hold_mean": float(np.mean(list(cv.values()))), "cv": cv, "full_oof_mae": oof_mae,
          "ald_folds": ald_folds, "n_all_data_in_b50": len(all_data_tests)}
json.dump(report, open(os.path.join(ME, "blend_report_final.json"), "w"), indent=2)
np.save(os.path.join(ME, "blend_test_pred.npy"), test_pred.astype(np.float32))
print(json.dumps(report, indent=2))

if args.final_dir:
    os.makedirs(args.final_dir, exist_ok=True)
    sub.to_csv(os.path.join(args.final_dir, "submission.csv"), index=False)
    print("WROTE FINAL", os.path.join(args.final_dir, "submission.csv"))
