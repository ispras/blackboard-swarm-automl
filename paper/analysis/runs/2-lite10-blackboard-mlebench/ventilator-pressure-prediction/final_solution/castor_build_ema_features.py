import numpy as np, json, os, shutil

S = "/download_job_code/payload/sandbox/workspaces/castor_eta"
SRC = f"{S}/cache2"; DST = f"{S}/cache3e"
os.makedirs(DST, exist_ok=True)
names = json.load(open(f"{SRC}/feature_names.json"))
idx = {n: i for i, n in enumerate(names)}
mu = np.load(f"{SRC}/norm_mean.npy"); sd = np.load(f"{SRC}/norm_std.npy")
for f in ["y_train.npy", "m_train.npy", "fold_shared.npy", "train_breath_ids.npy",
          "test_breath_ids.npy", "test_id.npy"]:
    shutil.copy(f"{SRC}/{f}", f"{DST}/{f}")


def build(X):
    raw = lambda n: X[:, :, idx[n]] * sd[idx[n]] + mu[idx[n]]
    u = raw("u_in"); dt = raw("dt"); R = raw("R"); C = raw("C")
    tau0 = np.maximum(R * C / 1000.0, 1e-3)  # seconds
    outs = []
    for mult in (0.5, 1.0, 2.0):
        tau = tau0 * mult
        dec = np.exp(-np.clip(dt, 0, None) / tau)
        e = np.empty_like(u)
        e[:, 0] = u[:, 0]
        for t in range(1, u.shape[1]):
            e[:, t] = dec[:, t] * e[:, t - 1] + (1 - dec[:, t]) * u[:, t]
        outs.append(e)
    e1, e2, e3 = outs
    extra = np.stack([e1, e2, e3, e1 * R, e1 / np.maximum(C, 1e-3)], axis=-1).astype(np.float32)
    # standardize appended cols only (original 46 are already normalized)
    return np.concatenate([X, extra], axis=-1)


Xtr = build(np.asarray(np.load(f"{SRC}/X_train.npy", mmap_mode="r")))
F = Xtr.shape[-1]; nold = len(names)
nmu = np.zeros(F); nsd = np.ones(F)
nmu[:nold] = mu; nsd[:nold] = sd
for j in range(nold, F):
    a = Xtr[:, :, j]
    m = float(a.mean()); s = float(a.std()) or 1.0
    nmu[j] = m; nsd[j] = s
    Xtr[:, :, j] = (a - m) / s
np.save(f"{DST}/X_train.npy", Xtr); del Xtr
print("train done", F)

Xte = build(np.asarray(np.load(f"{SRC}/X_test.npy", mmap_mode="r")))
for j in range(nold, F):
    Xte[:, :, j] = (Xte[:, :, j] - nmu[j]) / nsd[j]
np.save(f"{DST}/X_test.npy", Xte)
np.save(f"{DST}/norm_mean.npy", nmu); np.save(f"{DST}/norm_std.npy", nsd)
json.dump(names + ["ema_rc_half", "ema_rc", "ema_rc_double", "ema_rc_x_R", "ema_rc_div_C"],
          open(f"{DST}/feature_names.json", "w"))
print("done", F)
