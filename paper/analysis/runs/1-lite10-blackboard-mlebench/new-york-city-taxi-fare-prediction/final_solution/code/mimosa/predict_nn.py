import os, json
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from common import FEATS_V2, TE_COLS, load_te
from test_features import build_test, add_counts_te
from train_nn import Net

OUT = "/download_job_code/payload/sandbox/workspaces/mimosa_omicron"
ROOT = "/download_job_code/payload/sandbox"
CONT = FEATS_V2 + TE_COLS
dev = "cuda" if torch.cuda.is_available() else "cpu"

te = load_te()
df = build_test()
df = add_counts_te(df, te)
ck = torch.load(os.path.join(OUT, "nn_nn_full.pt"), map_location="cpu")
mu = ck["mu"]; sd = ck["sd"]; ym = ck["ym"]; ys = ck["ys"]; emb_dims = ck["emb_dims"]
model = Net(len(CONT), emb_dims)
model.load_state_dict(ck["model"])
model.to(dev).eval()

X = (df[CONT].to_numpy(np.float32) - mu) / sd
up = te["u_pcell"]; ud = te["u_dcell"]
pc = np.clip(np.searchsorted(up, df["p_cell"].to_numpy(np.int64)), 0, len(up) - 1)
dc = np.clip(np.searchsorted(ud, df["d_cell"].to_numpy(np.int64)), 0, len(ud) - 1)
C = np.stack([df["year"].to_numpy(np.float32).astype(np.int32) - 2009,
              df["month"].to_numpy(np.float32).astype(np.int32) - 1,
              df["weekday"].to_numpy(np.float32).astype(np.int32),
              df["hour"].to_numpy(np.float32).astype(np.int32),
              np.clip(df["passenger_count"].to_numpy(np.float32).astype(np.int32), 0, 8),
              pc, dc], axis=1).astype(np.int64)
with torch.no_grad():
    preds = []
    for i in range(0, len(X), 8192):
        xb = torch.from_numpy(X[i:i + 8192]).to(dev)
        cb = [torch.from_numpy(C[i:i + 8192, j]).to(dev) for j in range(C.shape[1])]
        preds.append(model(xb, cb).float().cpu().numpy())
pred = np.concatenate(preds) * ys + ym
corrupt = df["corrupt"].to_numpy().astype(bool)
fb = xgb.Booster(); fb.load_model(os.path.join(OUT, "model_corrupt_fb.json"))
fb.set_param({"device": "cpu"})
from build_corrupt import add_corrupt_feats
fb_feats = json.load(open(os.path.join(OUT, "result_corrupt_fb.json")))["feats"]
cdf = add_corrupt_feats(df)
pf = fb.predict(xgb.DMatrix(cdf[fb_feats].to_numpy(np.float32), feature_names=fb_feats))
pred = np.where(corrupt, pf, pred)
pred = np.clip(pred, 2.5, 500).astype(np.float32)
np.save(os.path.join(OUT, "test_pred_nn_full.npy"), pred)
sub = pd.DataFrame({"key": pd.read_csv(os.path.join(ROOT, "data", "test.csv"))["key"], "fare_amount": pred})
sub.to_csv(os.path.join(OUT, "submission_nn_full.csv"), index=False)
print("saved nn test preds mean", pred.mean(), "corrupt", corrupt.sum())
