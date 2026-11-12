import numpy as np, os, glob, json

S = "/download_job_code/payload/sandbox"
C = f"{S}/workspaces/castor_eta/cache"
y = np.load(f"{C}/y_train.npy"); m = np.load(f"{C}/m_train.npy")
fold = np.load(f"{C}/fold_shared.npy"); f0 = np.where(fold == 0)[0]
yy, mm = y[f0], m[f0]; n = len(f0)


def load(path):
    a = np.load(path)
    if a.shape[0] == 67905: return a[f0]
    if a.shape[0] == n: return a
    raise ValueError((path, a.shape))


cands = {}
r = f"{S}/workspaces/castor_eta/runs"
for nm in ["bilstm_big", "bilstm_c2", "cnn_gru_c2", "bilstm_c2_250", "bilstm_e3_220", "gru_v1"]:
    fp = f"{r}/{nm}_val.npy"
    if os.path.exists(fp): cands[f"castor_{nm}"] = load(fp)
cp = f"{S}/workspaces/canopus_iota"
for nm in ["lstm384_5f2", "lstm384_v2f", "lstm512_c2b", "lstm384"]:
    fp = f"{cp}/oof_{nm}.npy"
    if os.path.exists(fp): cands[f"canopus_{nm}"] = load(fp)
for nm in ["c2s123"]:
    fp = f"{cp}/runs/{nm}_val.npy"
    if os.path.exists(fp): cands[f"canopus_{nm}"] = load(fp)
for fp in sorted(glob.glob(f"{S}/workspaces/bellatrix_iota/preds/oof_*.npy")):
    cands[f"bellatrix_{os.path.basename(fp)[4:-4]}"] = load(fp)

names = list(cands); P = np.stack([cands[k] for k in names])
print("models", len(names))


def smae(pred, idx):
    d = np.abs(pred[idx] - yy[idx]); return float((d * mm[idx]).sum() / mm[idx].sum())


def greedy(idx, iters=400):
    w = np.ones(len(names)) / len(names)
    cur = smae((w[:, None, None] * P).sum(0), idx)
    for _ in range(iters):
        imp = False
        for i in range(len(names)):
            for d in (0.05, -0.05, 0.02, -0.02, 0.01, -0.01, 0.005, -0.005):
                w2 = w.copy(); w2[i] = max(0.0, w2[i] + d)
                if w2.sum() == 0: continue
                w2 /= w2.sum()
                s = smae((w2[:, None, None] * P).sum(0), idx)
                if s < cur - 1e-6:
                    cur, w, imp = s, w2, True
        if not imp: break
    return w, cur


rng = np.random.RandomState(7); perm = rng.permutation(n); h1, h2 = perm[:n // 2], perm[n // 2:]
for a, b in [(h1, h2), (h2, h1)]:
    w, _ = greedy(a)
    print("crossfit fit-in-sample", round(smae((w[:, None, None] * P).sum(0), a), 5),
          "held-half", round(smae((w[:, None, None] * P).sum(0), b), 5))
w, full = greedy(np.arange(n))
print("FULL greedy", round(full, 5))
print(json.dumps({k: round(float(v), 3) for k, v in zip(names, w) if v > 0.001}, indent=0))
json.dump({"weights": {k: float(v) for k, v in zip(names, w)}, "full_insp": full},
          open(f"{S}/workspaces/castor_eta/runs/final_weights.json", "w"), indent=1)
