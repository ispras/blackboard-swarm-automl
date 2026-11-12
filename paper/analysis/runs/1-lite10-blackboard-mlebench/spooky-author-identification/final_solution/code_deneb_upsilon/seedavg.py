import os
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

CLASSES = ["EAP", "HPL", "MWS"]
BASE = "/download_job_code/payload/sandbox"
train = pd.read_csv(f"{BASE}/data/train.csv")
test = pd.read_csv(f"{BASE}/data/test.csv")
folds = pd.read_csv(f"{BASE}/workspaces/deneb_upsilon/shared_folds.csv")[["id", "fold"]]
train = train.merge(folds, on="id", how="left")
y = train["author"].map({c: i for i, c in enumerate(CLASSES)}).values

ap = argparse.ArgumentParser()
ap.add_argument("--tags", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--use", default="avg")  # avg | best
args = ap.parse_args()

suffix = "avg_oof.npy" if args.use == "avg" else "_oof.npy"


def lp(p):
    a = np.load(p); a = np.clip(a, 1e-15, 1); return a / a.sum(1, keepdims=True)


O, T = [], []
for t in args.tags.split(","):
    op = f"{BASE}/workspaces/deneb_upsilon/{t}{suffix}"
    tp = f"{BASE}/workspaces/deneb_upsilon/{t}{'avg' if args.use=='avg' else ''}_test.npy"
    if not (os.path.exists(op) and os.path.exists(tp)):
        print("missing", t); continue
    o = lp(op); tt = lp(tp)
    print(t, "OOF", log_loss(y, o))
    O.append(o); T.append(tt)
O = np.stack(O); T = np.stack(T)
oa = O.mean(0); ta = T.mean(0)
og = np.exp(np.log(O).mean(0)); og /= og.sum(1, keepdims=True)
tg = np.exp(np.log(T).mean(0)); tg /= tg.sum(1, keepdims=True)
la, lg = log_loss(y, oa), log_loss(y, og)
print(f"SEEDAVG ARITH {la:.5f}  GEO {lg:.5f}  n={len(O)}")
if la <= lg:
    oo, tt = oa, ta
else:
    oo, tt = og, tg
np.save(f"{BASE}/workspaces/deneb_upsilon/{args.out}_oof.npy", oo)
np.save(f"{BASE}/workspaces/deneb_upsilon/{args.out}_test.npy", tt)
pd.DataFrame({"id": test["id"], "EAP": tt[:, 0], "HPL": tt[:, 1], "MWS": tt[:, 2]}).to_csv(
    f"{BASE}/workspaces/deneb_upsilon/submission_{args.out}.csv", index=False)
print("saved", args.out, log_loss(y, oo))
