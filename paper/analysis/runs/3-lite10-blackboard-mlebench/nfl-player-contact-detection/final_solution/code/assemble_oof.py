import numpy as np, pandas as pd, os, sys, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
OUT = "workspaces/alkaid_psi"
TAG = sys.argv[1] if len(sys.argv) > 1 else "v1"

plan = pd.read_parquet(f"{OUT}/train_cand.parquet")
pred = np.full(len(plan), np.nan, np.float64)
for vf in glob.glob(f"{OUT}/validx_f*_{TAG}.npy"):
    f = int(vf.split("_f")[1].split("_")[0])
    idx = np.load(vf)
    p = np.load(f"{OUT}/valpred_f{f}_{TAG}.npy")
    pred[idx] = p
print("candidate coverage", np.isfinite(pred).mean(), "n", len(pred))
cand = plan[["contact_id", "game_play", "step", "p1", "p2", "dist", "contact", "fold"]].copy()
cand["pred"] = pred
cand.to_parquet(f"{OUT}/oof_cnn_cand_{TAG}.parquet", index=False)

# full pp row table with floor for non-candidates
pp = pd.read_parquet(f"{OUT}/train_pp_rows.parquet")
m = pp.merge(cand[["contact_id", "pred"]], on="contact_id", how="left")
m["is_cand"] = m.pred.notna()
m["pred"] = m.pred.fillna(0.0)
m.to_parquet(f"{OUT}/oof_cnn_pp_{TAG}.parquet", index=False)
print("pp rows", len(m), "cand", m.is_cand.sum(), "fold vals", sorted(m.fold.dropna().unique()))
# quick metrics
from sklearn.metrics import roc_auc_score, average_precision_score
c = cand.dropna()


def best_mcc(y, p):
    o = np.argsort(-p); ys = y[o]; ps = p[o]; P = ys.sum(); N = len(ys) - P
    tp = np.cumsum(ys); fp = np.cumsum(1 - ys)
    mcc = (tp * N - fp * P) / np.sqrt((tp + fp) * P * N * (N + P - fp - tp) + 1e-12)
    k = int(np.argmax(mcc)); return float(mcc[k]), float(ps[k])


print("cand AUC", roc_auc_score(c.contact, c.pred), "AP", average_precision_score(c.contact, c.pred))
print("cand bestMCC", best_mcc(c.contact.values, c.pred.values))
print("pp bestMCC", best_mcc(m.contact.values, m.pred.values))
