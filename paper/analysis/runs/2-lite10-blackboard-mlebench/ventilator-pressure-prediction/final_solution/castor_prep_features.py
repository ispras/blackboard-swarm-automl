import pandas as pd, numpy as np, json, os, time

DATA = "/download_job_code/payload/sandbox/data"
OUT = "/download_job_code/payload/sandbox/workspaces/castor_eta/cache2"
os.makedirs(OUT, exist_ok=True)
T0 = time.time()

def log(*a):
    print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)

DT = {"id": np.int64, "breath_id": np.int32, "R": np.int16, "C": np.int16,
      "time_step": np.float32, "u_in": np.float32, "u_out": np.int8, "pressure": np.float32}

tr = pd.read_csv(f"{DATA}/train.csv", dtype=DT)
te = pd.read_csv(f"{DATA}/test.csv", dtype={k: v for k, v in DT.items() if k != "pressure"})
log("loaded", tr.shape, te.shape)

# verify contiguity and 80-length ordering by id
assert (tr.groupby("breath_id").size() == 80).all()
assert (te.groupby("breath_id").size() == 80).all()
tr = tr.sort_values(["breath_id", "id"]).reset_index(drop=True)
te = te.sort_values(["breath_id", "id"]).reset_index(drop=True)
tr_bid = tr.breath_id.drop_duplicates().values
te_bid = te.breath_id.drop_duplicates().values
assert len(tr) == 80 * len(tr_bid) and len(te) == 80 * len(te_bid)

def to2d(col):
    return tr[col].values.reshape(-1, 80).astype(np.float32)

u_in = to2d("u_in"); u_out = to2d("u_out"); t = to2d("time_step")
R = to2d("R"); C = to2d("C"); pressure = to2d("pressure")
tu = te["u_in"].values.reshape(-1, 80).astype(np.float32)
tu_out = te["u_out"].values.reshape(-1, 80).astype(np.float32)
tt = te["time_step"].values.reshape(-1, 80).astype(np.float32)
tR = te["R"].values.reshape(-1, 80).astype(np.float32)
tC = te["C"].values.reshape(-1, 80).astype(np.float32)
log("reshaped")

def build(u_in, u_out, t, R, C):
    n = u_in.shape[0]
    dt = np.zeros_like(t)
    dt[:, 1:] = np.diff(t, axis=1)
    dt = np.clip(dt, 0, None)
    insp = (u_out == 0).astype(np.float32)
    vol = np.cumsum(u_in * dt, axis=1)            # integral u_in dt
    csum = np.cumsum(u_in, axis=1)
    outcum = np.cumsum(u_out, axis=1)
    # lags / leads
    def shift(a, k):
        if k == 0: return a.copy()
        r = np.empty_like(a)
        if k > 0:
            r[:, :k] = a[:, :1]; r[:, k:] = a[:, :-k]
        else:
            r[:, k:] = a[:, -1:]; r[:, :k] = a[:, -k:]
        return r
    lag1, lag2, lag3 = shift(u_in, 1), shift(u_in, 2), shift(u_in, 3)
    lag4 = shift(u_in, 4)
    lead1, lead2, lead3 = shift(u_in, -1), shift(u_in, -2), shift(u_in, -3)
    lead4 = shift(u_in, -4)
    diff1 = u_in - lag1
    # trailing rolling mean 5
    cs = np.cumsum(u_in, axis=1)
    rm5 = np.empty_like(u_in)
    rm5[:, :5] = cs[:, :5] / np.arange(1, 6)
    rm5[:, 5:] = (cs[:, 5:] - cs[:, :-5]) / 5.0
    exp_mean = cs / np.arange(1, 81)
    runmax = np.maximum.accumulate(u_in, axis=1)
    step = np.tile(np.arange(80, dtype=np.float32) / 79.0, (n, 1))
    Rc = R; Cc = C
    invR = 1.0 / Rc; invC = 1.0 / Cc
    # breath-level stats
    total_vol = vol[:, -1:].repeat(80, axis=1)
    u_mean = u_in.mean(axis=1, keepdims=True).repeat(80, axis=1)
    u_std = u_in.std(axis=1, keepdims=True).repeat(80, axis=1)
    u_max = u_in.max(axis=1, keepdims=True).repeat(80, axis=1)
    dur_insp = insp.sum(axis=1, keepdims=True).repeat(80, axis=1)
    # --- v2 extra features ---
    diff2 = u_in - 2 * lag1 + lag2
    sq = u_in * u_in
    cssq = np.cumsum(sq, axis=1)
    idx = np.arange(80, dtype=np.float32)
    rs5 = np.empty_like(u_in)
    m1 = cssq[:, :5] / np.arange(1, 6) - (cs[:, :5] / np.arange(1, 6)) ** 2
    rs5[:, :5] = np.sqrt(np.maximum(m1, 0))
    m2 = (cssq[:, 5:] - cssq[:, :-5]) / 5.0 - ((cs[:, 5:] - cs[:, :-5]) / 5.0) ** 2
    rs5[:, 5:] = np.sqrt(np.maximum(m2, 0))
    tte = np.clip(dur_insp - idx[None, :], 0, None)
    insp_frac = idx[None, :] / np.maximum(dur_insp, 1.0)
    u_norm = u_in / (u_mean + 1e-3)
    invRC = 1.0 / (Rc * Cc)
    u_invRC = u_in * invRC
    vol_invRC = vol * invRC
    p_proxy = u_in * Rc + vol * invC
    feats = [
        t, step, u_in, u_out, insp,
        dt, vol, csum, outcum,
        lag1, lag2, lag3, lag4, lead1, lead2, lead3, lead4, diff1,
        rm5, exp_mean, runmax,
        Rc, Cc, invR, invC, Rc * Cc, Rc * invC,
        u_in * Rc, u_in * invC,
        vol * Rc, vol * invC,
        csum * Rc, csum * invC,
        total_vol, u_mean, u_std, u_max, dur_insp,
        diff2, rs5, tte, insp_frac, u_norm, u_invRC, vol_invRC, p_proxy,
    ]
    X = np.stack(feats, axis=-1).astype(np.float32)
    return X

names = ["time_step", "step", "u_in", "u_out", "insp", "dt", "vol", "csum", "outcum",
         "lag1", "lag2", "lag3", "lag4", "lead1", "lead2", "lead3", "lead4", "diff1",
         "rm5", "exp_mean", "runmax", "R", "C", "invR", "invC", "RC", "R_invC",
         "u_R", "u_invC", "vol_R", "vol_invC", "csum_R", "csum_invC",
         "total_vol", "u_mean", "u_std", "u_max", "dur_insp",
         "diff2", "roll_std5", "time_to_exp", "insp_frac", "u_norm", "u_invRC",
         "vol_invRC", "p_proxy"]
F = len(names)

Xtr = build(u_in, u_out, t, R, C)
Xte = build(tu, tu_out, tt, tR, tC)
log("features", Xtr.shape, Xte.shape)

# normalize all features with train stats (skip u_out/insp? harmless)
mu = Xtr.reshape(-1, F).mean(0)
sd = Xtr.reshape(-1, F).std(0)
sd[sd < 1e-6] = 1.0
np.save(f"{OUT}/norm_mean.npy", mu); np.save(f"{OUT}/norm_std.npy", sd)
Xtr = (Xtr - mu) / sd
Xte = (Xte - mu) / sd
log("normalized")

# folds: 5-fold stratified by (R,C) at breath level
rc = (R[:, 0].astype(np.int64) * 100 + C[:, 0].astype(np.int64))
rng = np.random.RandomState(42)
fold = np.zeros(len(rc), dtype=np.int8)
for key in np.unique(rc):
    idx = np.where(rc == key)[0]
    rng.shuffle(idx)
    fold[idx] = np.arange(len(idx)) % 5
log("fold sizes", np.bincount(fold))

np.save(f"{OUT}/X_train.npy", Xtr)
np.save(f"{OUT}/y_train.npy", pressure.astype(np.float32))
np.save(f"{OUT}/m_train.npy", (u_out == 0).astype(np.float32))
np.save(f"{OUT}/fold.npy", fold)
np.save(f"{OUT}/train_breath_ids.npy", tr_bid)
np.save(f"{OUT}/test_breath_ids.npy", te_bid)
np.save(f"{OUT}/X_test.npy", Xte)
np.save(f"{OUT}/test_id.npy", te["id"].values)
json.dump(names, open(f"{OUT}/feature_names.json", "w"))
log("saved to", OUT)
