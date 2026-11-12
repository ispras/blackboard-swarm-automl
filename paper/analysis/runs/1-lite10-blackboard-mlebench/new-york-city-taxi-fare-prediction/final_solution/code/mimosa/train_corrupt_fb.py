import os, json
import numpy as np
import pandas as pd
import xgboost as xgb

OUT = "/download_job_code/payload/sandbox/workspaces/mimosa_omicron"
df = pd.read_parquet(os.path.join(OUT, "corrupt_feat.parquet"))
FEATS = [c for c in df.columns if c not in ("orig_idx", "fare_amount")]
df = df.sort_values("orig_idx").reset_index(drop=True)
hold = (df["orig_idx"].to_numpy() % 100 == 0)
y = df["fare_amount"].to_numpy(np.float32)
X = df[FEATS].to_numpy(np.float32)
Xtr, ytr = X[~hold], y[~hold]
Xva, yva = X[hold], y[hold]
print("n", len(df), "train", Xtr.shape, "val", Xva.shape, "val mean", yva.mean(), flush=True)
const = float(np.mean(ytr))
print("const-mean val rmse", np.sqrt(np.mean((const - yva) ** 2)), flush=True)
params = {"device": "cuda", "tree_method": "hist", "max_bin": 128,
          "objective": "reg:squarederror", "eval_metric": "rmse",
          "eta": 0.03, "max_depth": 8, "min_child_weight": 200,
          "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0,
          "seed": 42, "nthread": 12, "verbosity": 1}
dtr = xgb.DMatrix(Xtr, label=ytr, feature_names=FEATS)
dva = xgb.DMatrix(Xva, label=yva, feature_names=FEATS)
bst = xgb.train(params, dtr, num_boost_round=2500, evals=[(dva, "val")],
                early_stopping_rounds=80, verbose_eval=100)
best = bst.best_iteration
pv = bst.predict(dva, iteration_range=(0, best + 1))
rmse = float(np.sqrt(np.mean((pv - yva) ** 2)))
print(f"CORRUPT_FB_RMSE={rmse:.5f} best={best}", flush=True)
bst.save_model(os.path.join(OUT, "model_corrupt_fb.json"))
np.save(os.path.join(OUT, "holdout_pred_corrupt_fb.npy"), pv.astype(np.float32))
np.save(os.path.join(OUT, "holdout_orig_corrupt_fb.npy"), df.loc[hold, "orig_idx"].to_numpy(np.int64))
np.save(os.path.join(OUT, "holdout_true_corrupt_fb.npy"), yva.astype(np.float32))
json.dump({"rmse": rmse, "best": int(best), "feats": FEATS, "const": const},
          open(os.path.join(OUT, "result_corrupt_fb.json"), "w"), indent=2)
