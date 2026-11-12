"""Build shared breath-level tensor cache + folds for the ventilator task.

Outputs (workspaces/shared):
  folds.csv                breath_id,fold  (5-fold GroupKFold, seed 42, fold 0 holdout)
  ventilator_v1.npz        X_train (N,80,F) f32, y_train, u_out_train, breath_id_train,
                           X_test, breath_id_test, feature_names
"""
import numpy as np
import pandas as pd
import os

OUT = "/download_job_code/payload/sandbox/workspaces/shared"
os.makedirs(OUT, exist_ok=True)
ROOT = "/download_job_code/payload/sandbox"
T = 80


def build(df, is_train):
    if is_train:
        df = df.sort_values(["breath_id", "id"]).reset_index(drop=True)
    else:
        df = df.sort_values(["breath_id", "id"]).reset_index(drop=True)
    df["t_idx"] = df.groupby("breath_id", sort=False).cumcount()
    n = df["breath_id"].nunique()
    assert len(df) == n * T
    bid = df["breath_id"].values[::T]

    def piv(col):
        return df[col].values.reshape(n, T).astype(np.float32)

    u_in = piv("u_in")
    u_out = piv("u_out")
    ts = piv("time_step")
    R = piv("R")
    C = piv("C")

    dt = np.diff(ts, axis=1, prepend=0.0)
    dt[:, 0] = ts[:, 0]

    def lag(a, k):
        out = np.empty_like(a)
        out[:, k:] = a[:, :-k]
        out[:, :k] = a[:, :1]
        return out

    def lead(a, k):
        out = np.empty_like(a)
        out[:, :-k] = a[:, k:]
        out[:, -k:] = a[:, -1:]
        return out

    cum = np.cumsum(u_in, axis=1)
    cummean = cum / np.arange(1, T + 1, dtype=np.float32)[None, :]
    cumvol = np.cumsum(u_in * dt, axis=1)
    cumvol_n = cumvol / (cumvol[:, -1:] + 1e-6)

    Rmin, Rrange = 5.0, 45.0
    Cmin, Crange = 10.0, 40.0
    Rn = (R - Rmin) / Rrange
    Cn = (C - Cmin) / Crange

    feats = {
        "u_in": u_in,
        "u_in_cumsum": cum,
        "u_in_cummean": cummean,
        "u_in_lag1": lag(u_in, 1),
        "u_in_lag2": lag(u_in, 2),
        "u_in_lag3": lag(u_in, 3),
        "u_in_lag4": lag(u_in, 4),
        "u_in_lead1": lead(u_in, 1),
        "u_in_lead2": lead(u_in, 2),
        "u_in_lead3": lead(u_in, 3),
        "u_in_lead4": lead(u_in, 4),
        "u_in_diff1": u_in - lag(u_in, 1),
        "u_in_diff2": u_in - lag(u_in, 2),
        "time_step": ts,
        "dt": dt,
        "u_out": u_out,
        "R": Rn,
        "C": Cn,
        "R_is5": (R == 5).astype(np.float32),
        "R_is20": (R == 20).astype(np.float32),
        "R_is50": (R == 50).astype(np.float32),
        "C_is10": (C == 10).astype(np.float32),
        "C_is20": (C == 20).astype(np.float32),
        "C_is50": (C == 50).astype(np.float32),
        "RC": Rn * Cn,
        "u_in_x_R": u_in * Rn,
        "u_in_div_C": u_in / (C + 1e-6),
        "cumvol": cumvol,
        "cumvol_n": cumvol_n,
    }
    names = list(feats.keys())
    X = np.stack([feats[k] for k in names], axis=2).astype(np.float32)
    return X, bid, names, u_out, (piv("pressure") if is_train else None)


def main():
    tr = pd.read_csv(f"{ROOT}/data/train.csv")
    te = pd.read_csv(f"{ROOT}/data/test.csv")
    Xtr, bid_tr, names, uout_tr, y = build(tr, True)
    Xte, bid_te, _, _, _ = build(te, False)
    print("Xtr", Xtr.shape, "Xte", Xte.shape, "F", len(names))
    print(names)

    # folds over sorted unique breath ids
    ids = np.sort(np.unique(bid_tr))
    rng = np.random.RandomState(42)
    perm = rng.permutation(len(ids))
    fold = np.empty(len(ids), dtype=np.int16)
    for k in range(5):
        fold[perm[k::5]] = k
    folds = pd.DataFrame({"breath_id": ids, "fold": fold})
    folds.to_csv(f"{OUT}/folds.csv", index=False)
    print(folds.fold.value_counts().sort_index().to_dict())

    np.savez_compressed(
        f"{OUT}/ventilator_v1.npz",
        X_train=Xtr, y_train=y.astype(np.float32), u_out_train=uout_tr.astype(np.float32),
        breath_id_train=bid_tr, X_test=Xte, breath_id_test=bid_te,
        feature_names=np.array(names),
    )
    print("saved", os.path.getsize(f"{OUT}/ventilator_v1.npz") / 1e9, "GB")


if __name__ == "__main__":
    main()
