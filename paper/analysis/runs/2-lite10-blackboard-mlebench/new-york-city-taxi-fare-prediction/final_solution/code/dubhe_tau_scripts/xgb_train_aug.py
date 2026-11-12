import argparse, json, os, time
import numpy as np, pandas as pd, xgboost as xgb

ROOT = "/download_job_code/payload/sandbox"
FEAT = f"{ROOT}/workspaces/dubhe_tau/features"
OUT = f"{ROOT}/workspaces/dubhe_tau/models"
os.makedirs(OUT, exist_ok=True)
C30 = np.cos(np.pi / 6); S30 = np.sin(np.pi / 6)


def augment(X):
    plon = X[:, 0]; plat = X[:, 1]; dlon = X[:, 2]; dlat = X[:, 3]
    hav = X[:, 12]; manh = X[:, 13]; bearing = X[:, 14]
    hour = X[:, 8]; weekday = X[:, 10]
    dpc = X[:, 19]; ddc = X[:, 20]
    pj = X[:, 21]; pl = X[:, 23]; pe = X[:, 25]
    dj = X[:, 22]; dl = X[:, 24]; de = X[:, 26]
    b = np.radians(bearing)
    extra = np.column_stack([
        plon * C30 - plat * S30, plon * S30 + plat * C30,
        dlon * C30 - dlat * S30, dlon * S30 + dlat * C30,
        np.sin(hour * np.pi / 12), np.cos(hour * np.pi / 12),
        np.sin(weekday * 2 * np.pi / 7), np.cos(weekday * 2 * np.pi / 7),
        (weekday >= 5).astype(np.float32),
        np.log1p(np.maximum(hav, 0)), np.log1p(np.maximum(manh, 0)),
        (hav < 0.1).astype(np.float32),
        np.sin(b), np.cos(b),
        (np.minimum(pj, dj) < 2).astype(np.float32),
        (np.minimum(pl, dl) < 2).astype(np.float32),
        (np.minimum(pe, de) < 2).astype(np.float32),
        ddc - dpc,
    ]).astype(np.float32)
    return np.concatenate([X, extra], axis=1)


p = argparse.ArgumentParser()
p.add_argument("--nrows", type=int, default=0)
p.add_argument("--batch", type=int, default=4_000_000)
p.add_argument("--depth", type=int, default=13)
p.add_argument("--lr", type=float, default=0.05)
p.add_argument("--rounds", type=int, default=4000)
p.add_argument("--es", type=int, default=100)
p.add_argument("--seed", type=int, default=1)
p.add_argument("--min_child_weight", type=float, default=30)
p.add_argument("--subsample", type=float, default=0.85)
p.add_argument("--colsample", type=float, default=0.85)
p.add_argument("--max_bin", type=int, default=256)
p.add_argument("--grow_policy", type=str, default="depthwise")
p.add_argument("--max_leaves", type=int, default=0)
p.add_argument("--tag", type=str, default="xgb_aug")
p.add_argument("--device", type=str, default="cuda")
args = p.parse_args()

meta = json.load(open(f"{FEAT}/meta.json"))
n_valid = meta["n_valid"]
X = np.load(f"{FEAT}/X_full.npy", mmap_mode="r")
y = np.load(f"{FEAT}/y_full.npy", mmap_mode="r")
ho = np.load(f"{FEAT}/holdout_full.npy", mmap_mode="r")
hold_idx = np.where(np.asarray(ho[:n_valid]) == 1)[0]
train_idx = np.where(np.asarray(ho[:n_valid]) == 0)[0]
rng = np.random.default_rng(args.seed)
if args.nrows and args.nrows < len(train_idx):
    train_idx = rng.choice(train_idx, size=args.nrows, replace=False)
train_idx = train_idx.astype(np.int64)
print("train", len(train_idx), "holdout", len(hold_idx), flush=True)

t0 = time.time()
B = args.batch
order = rng.permutation(len(train_idx))
nb = (len(order) + B - 1) // B


class BatchIter(xgb.DataIter):
    def __init__(self):
        super().__init__(cache_prefix=None)
        self.i = 0

    def next(self, input_data):
        if self.i >= nb:
            return False
        s = self.i * B
        ii = train_idx[order[s:s + B]]
        Xb = np.asarray(X[ii], dtype=np.float32)
        input_data(data=augment(Xb), label=np.asarray(y[ii], dtype=np.float32))
        self.i += 1
        return True

    def reset(self):
        self.i = 0


dtrain = xgb.QuantileDMatrix(BatchIter(), max_bin=args.max_bin)
print("dtrain built", time.time() - t0, flush=True)
Xva = augment(np.asarray(X[hold_idx], dtype=np.float32))
yva = np.asarray(y[hold_idx], dtype=np.float32)
dval = xgb.QuantileDMatrix(Xva, label=yva, ref=dtrain, max_bin=args.max_bin)
print("dval built", Xva.shape, time.time() - t0, flush=True)

params = dict(objective="reg:squarederror", eval_metric="rmse", tree_method="hist",
              device=args.device, max_depth=args.depth, learning_rate=args.lr,
              subsample=args.subsample, colsample_bytree=args.colsample,
              min_child_weight=args.min_child_weight, reg_lambda=1.0,
              max_bin=args.max_bin, seed=args.seed, grow_policy=args.grow_policy)
if args.grow_policy == "lossguide" and args.max_leaves > 0:
    params["max_leaves"] = args.max_leaves
bst = xgb.train(params, dtrain, num_boost_round=args.rounds,
                evals=[(dval, "val")], early_stopping_rounds=args.es, verbose_eval=50)
best = bst.best_iteration
pred_va = bst.predict(dval, iteration_range=(0, best + 1))
rmse = float(np.sqrt(np.mean((pred_va - yva) ** 2)))
print(f"BEST_ITER {best} HOLDOUT_RMSE {rmse:.6f}", flush=True)
bst.save_model(f"{OUT}/{args.tag}.json")
np.save(f"{OUT}/{args.tag}_valpred.npy", pred_va.astype(np.float32))
np.save(f"{OUT}/{args.tag}_validx.npy", hold_idx.astype(np.int64))
Xte = augment(np.load(f"{FEAT}/test_X.npy"))
pred_te = bst.predict(xgb.DMatrix(Xte), iteration_range=(0, best + 1))
pd.DataFrame({"key": pd.read_csv(f"{FEAT}/test_keys.csv")["key"], "fare_amount": pred_te}).to_csv(
    f"{OUT}/{args.tag}_test.csv", index=False)
json.dump({"tag": args.tag, "holdout_rmse": rmse, "best_iter": int(best), "nrows": int(len(train_idx)),
           "params": {k: str(v) for k, v in params.items()}, "args": vars(args)},
          open(f"{OUT}/{args.tag}_result.json", "w"), indent=2)
print("SAVED", flush=True)
