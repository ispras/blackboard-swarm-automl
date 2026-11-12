"""MLSP 2013 Bird Classification - final ensemble submission.

Run:  pyenv shell sis-black && python solution.py
Writes: submission.csv next to this file.

Pipeline
--------
Each component is a per-class score matrix over the 64 test recordings.  We
z-score every component per class (across the 64 test recs), take a weighted
sum, and add a class-prior offset ``lam * logit(train prevalence_c)``.  The
offset is what makes the flattened/global ROC-AUC (the official metric) much
better: global AUC compares scores across species, so the cross-species scale
must reflect prevalence.  A single global sigmoid is applied at the end (a
monotone map, so AUC is unchanged) to put predictions in [0, 1].

Components
----------
  perch : Perch v2 zero-shot probabilities (5 s @ 32 kHz windows, max over windows)
  bn    : BirdNET h05 zero-shot probabilities (3 s windows, 0.5 s hop, max)
  pp    : linear probe on Perch 1536-d embeddings
  mlpp  : MLP (512-128-19) on Perch embeddings (PCA-256) + recording metadata
  pw    : time-weighted same-session label prior from *training* labels only

No test labels are used anywhere.  Validation was done on a 5-fold split of the
258 training recordings (components/*_oof.csv; see README).
"""
import os, re, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "essential_data")
COMP = os.path.join(HERE, "components")
N_CLASS = 19

WEIGHTS = {"perch": 1.0, "bn": 0.2, "bn3dn": 1.0,
           "pp": 0.1, "mlpp": 0.1, "mlpa": 0.3, "dn": 0.15, "meta": 0.1, "pw": 0.2}
LAM = 1.10
ALPHA = 0.8
CLIP = 1.5   # per-class z-score clip before weighted sum


def load_meta():
    fold = pd.read_csv(os.path.join(DATA, "CVfolds_2.txt"))
    rec2fn = pd.read_csv(os.path.join(DATA, "rec_id2filename.txt"))
    df = fold.merge(rec2fn, on="rec_id").sort_values("rec_id").reset_index(drop=True)
    site, date, time, seq = [], [], [], []
    for fn in df.filename:
        m = re.match(r"(PC\d+)_(\d{8})_(\d{6})_(\d+)", fn)
        s, d, t, q = m.groups()
        site.append(s); date.append(d); time.append(t); seq.append(int(q))
    df["site"] = site; df["date"] = date; df["time"] = time; df["seq"] = seq
    df["secs"] = df.time.str[:2].astype(int) * 3600 + df.time.str[2:4].astype(int) * 60
    return df


def load_labels(df_train):
    rows = {}
    with open(os.path.join(DATA, "rec_labels_test_hidden.txt")) as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split(",")
            if not p or not p[0]:
                continue
            rows[int(p[0])] = [int(x) for x in p[1:] if x.strip() not in ("", "?")]
    Y = np.zeros((len(df_train), N_CLASS), np.float32)
    for i, rid in enumerate(df_train.rec_id.values):
        for c in rows.get(int(rid), []):
            if 0 <= c < N_CLASS:
                Y[i, c] = 1
    return Y


def zscore(P):
    Q = np.zeros_like(P, dtype=float)
    for c in range(P.shape[1]):
        sd = P[:, c].std()
        Q[:, c] = (P[:, c] - P[:, c].mean()) / (sd + 1e-9) if sd > 1e-12 else 0.0
    return np.clip(Q, -CLIP, CLIP)


def session_prior(train, Y, targets, exclude=None):
    """Time-weighted label prior from same site+date training recordings."""
    out = np.zeros((len(targets), N_CLASS), np.float32)
    for i, (_, r) in enumerate(targets.iterrows()):
        m = (train.site.values == r.site) & (train.date.values == r.date)
        if exclude is not None:
            m = m & exclude
        if not m.any():
            continue
        w = (np.exp(-np.abs(train.secs.values[m] - r.secs) / 300.0) *
             np.exp(-np.abs(train.seq.values[m] - r.seq) / 30.0))
        out[i] = (w[:, None] * Y[m]).sum(0) / (w.sum() + 1e-9)
    return out


FILE = {"perch": "perch_pcr_max", "bn": "birdnet_h05max", "bn3dn": "bn3_dn",
        "bn3max": "bn3_b3_max", "bn3mean": "bn3_b3_mean",
        "pp": "perch_probe", "mlpp": "mlp_perch_ens5", "mlpa": "mlp_all_avg3",
        "dn": "perch_dn", "meta": "meta_LR", "mlpbn": "mlp_embmeta"}


def load_component(name, ids):
    path = os.path.join(COMP, f"{FILE[name]}_test.csv")
    d = pd.read_csv(path).set_index("rec_id").reindex(ids)
    cols = [f"pred_{i}" for i in range(N_CLASS)]
    assert not d[cols].isna().any().any(), f"missing rec ids in {path}"
    return d[cols].values.astype(float)


def main():
    meta = load_meta()
    train = meta[meta.fold == 0].reset_index(drop=True)
    test = meta[meta.fold == 1].reset_index(drop=True)
    Y = load_labels(train)

    comps = {}
    for name in ["perch", "bn", "bn3dn", "pp", "mlpp", "mlpa", "dn", "meta"]:
        comps[name] = load_component(name, test.rec_id.values)
    comps["pw"] = session_prior(train, Y, test)

    S = np.zeros((len(test), N_CLASS), float)
    for name, w in WEIGHTS.items():
        S += w * zscore(comps[name])

    prior = np.clip(Y.mean(0), 1e-3, 1 - 1e-3)
    lp = np.log(prior / (1 - prior))
    # transductive prevalence estimate: mean predicted probability over the 64 test recs.
    # (uses only test *inputs* + train-trained models, no test labels)
    mlpbn = load_component("mlpbn", test.rec_id.values)
    est = 0.5 * (comps["mlpp"].mean(0) + mlpbn.mean(0))
    est = np.clip(est, 1e-3, 1 - 1e-3)
    le = np.log(est / (1 - est))
    off = (1 - ALPHA) * lp + ALPHA * le
    S += LAM * off[None, :]

    probs = 1.0 / (1.0 + np.exp(-S))
    rows = []
    for i, rid in enumerate(test.rec_id.values):
        for c in range(N_CLASS):
            rows.append((int(rid) * 100 + c, float(probs[i, c])))
    sub = pd.DataFrame(rows, columns=["Id", "Probability"]).sort_values("Id")
    sub.to_csv(os.path.join(HERE, "submission.csv"), index=False)
    print("wrote submission.csv", sub.shape)
    print("mean prob/class:", np.round(probs.mean(0), 3))

    # optional OOF report if OOF component files are present
    try:
        from sklearn.metrics import roc_auc_score
        cv = pd.read_csv(os.path.join(COMP, "cv_folds.csv"))
        fmap = dict(zip(cv.rec_id, cv.fold))
        Z, O = {}, {}
        for name in ["perch", "bn", "bn3dn", "pp", "mlpp", "mlpa", "dn", "meta"]:
            d = pd.read_csv(os.path.join(COMP, f"{FILE[name]}_oof.csv")).set_index("rec_id")
            O[name] = d.reindex(train.rec_id.values)[[f"pred_{i}" for i in range(N_CLASS)]].values
            Z[name] = zscore(O[name])
        # honest fold-restricted session prior (exclude own fold)
        ftr = train.rec_id.map(fmap).values
        Pw = np.zeros_like(Y, float)
        for k in range(5):
            vi = np.where(ftr == k)[0]
            Pw[vi] = session_prior(train, Y, train.iloc[vi], exclude=(ftr != k))
        Z["pw"] = zscore(Pw)
        So = sum(WEIGHTS[k] * Z[k] for k in WEIGHTS) + LAM * np.log(prior / (1 - prior))[None, :]
        print("OOF micro AUC (train-prior offset, baseline) = %.4f" % roc_auc_score(Y.ravel(), So.ravel()))
    except Exception as e:
        print("OOF report skipped:", e)


if __name__ == "__main__":
    main()
