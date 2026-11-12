import os, json, re
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = "/download_job_code/payload/sandbox"
ESS = f"{ROOT}/data/essential_data"
SUP = f"{ROOT}/data/supplemental_data"
WS = f"{ROOT}/workspaces/arcturus_kappa"
N_CLASS = 19

def load_meta():
    fold = pd.read_csv(f"{ESS}/CVfolds_2.txt")
    rec2fn = pd.read_csv(f"{ESS}/rec_id2filename.txt")
    df = fold.merge(rec2fn, on="rec_id")
    return df.sort_values("rec_id").reset_index(drop=True)

def load_labels():
    """Return (rec_ids, Y) for records with known labels (fold 0)."""
    rows = []
    with open(f"{ESS}/rec_labels_test_hidden.txt") as f:
        next(f)
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split(",")
            rid = int(parts[0])
            labs = [int(x) for x in parts[1:] if x.strip() not in ("", "?")]
            rows.append((rid, labs))
    lab = pd.DataFrame({"rec_id": [r for r, _ in rows], "labels": [l for _, l in rows]})
    df = load_meta().merge(lab, on="rec_id")
    train = df[df.fold == 0].reset_index(drop=True)
    test = df[df.fold == 1].reset_index(drop=True)
    Y = np.zeros((len(train), N_CLASS), dtype=np.int8)
    for i, ls in enumerate(train.labels):
        for l in ls:
            if 0 <= l < N_CLASS:
                Y[i, l] = 1
    Yt = np.zeros((len(test), N_CLASS), dtype=np.int8)
    for i, ls in enumerate(test.labels):
        for l in ls:
            if 0 <= l < N_CLASS:
                Yt[i, l] = 1
    return train, Y, test, Yt

def load_hist():
    h = pd.read_csv(f"{SUP}/histogram_of_segments.txt", header=None, skiprows=1)
    h.columns = ["rec_id"] + [f"h{i}" for i in range(h.shape[1] - 1)]
    return h

def load_segments():
    # parse rec_id, segment_id, then 38 features
    rows = []
    with open(f"{SUP}/segment_features.txt") as f:
        for line in f:
            p = line.strip().split(",")
            if len(p) < 40:
                continue
            rows.append([int(p[0]), int(p[1])] + [float(x) for x in p[2:40]])
    cols = ["rec_id", "seg_id"] + [f"s{i}" for i in range(38)]
    return pd.DataFrame(rows, columns=cols)

def load_seg_rect():
    rows = []
    with open(f"{SUP}/segment_rectangles.txt") as f:
        for line in f:
            p = line.strip().split(",")
            if len(p) < 6:
                continue
            rows.append([int(p[0]), int(p[1])] + [float(x) for x in p[2:6]])
    return pd.DataFrame(rows, columns=["rec_id", "seg_id", "xmin", "xmax", "ymin", "ymax"])

def agg_segments(seg):
    """Aggregate 38-d segment features per recording."""
    feats = [c for c in seg.columns if c.startswith("s") and c != "seg_id"]
    out = {}
    for rid, g in seg.groupby("rec_id"):
        v = g[feats].values
        a = [v.mean(0), v.std(0), v.min(0), v.max(0), np.median(v, 0)]
        out[rid] = np.concatenate(a + [np.array([len(v)])])
    return out

def iterative_stratified_folds(Y, n_splits=5, seed=42):
    """Simple iterative stratification for multilabel; returns list of (tr,va)."""
    rng = np.random.RandomState(seed)
    n, L = Y.shape
    folds = [[] for _ in range(n_splits)]
    # desired count per fold per label
    label_counts = Y.sum(0)
    target = label_counts / n_splits
    fold_counts = np.zeros((n_splits, L))
    order = np.argsort(-Y.sum(1))  # rarest/most labels first
    # shuffle among equal counts
    rng.shuffle(order)
    order = sorted(order, key=lambda i: -Y[i].sum())
    for i in order:
        labs = np.where(Y[i] == 1)[0]
        best, best_score = 0, 1e9
        for f in range(n_splits):
            if len(folds[f]) >= np.ceil(n / n_splits):
                continue
            if len(labs) == 0:
                score = len(folds[f]) / (n / n_splits)
            else:
                need = np.maximum(target[labs] - fold_counts[f, labs], 0).sum()
                score = need + 0.01 * len(folds[f]) / (n / n_splits)
            if score < best_score:
                best_score, best = score, f
        folds[best].append(i)
        fold_counts[best] += Y[i]
    return [(np.array([j for f in range(n_splits) if f != k for j in folds[f]]),
             np.array(folds[k])) for k in range(n_splits)]

def global_auc(Y, P):
    return roc_auc_score(Y.ravel(), P.ravel())

def per_class_auc(Y, P):
    aucs = []
    for c in range(Y.shape[1]):
        if Y[:, c].sum() > 0 and Y[:, c].sum() < len(Y):
            aucs.append(roc_auc_score(Y[:, c], P[:, c]))
    return float(np.mean(aucs)), aucs

def parse_meta(df):
    """Metadata from filename: site, date, time."""
    out = []
    for fn in df.filename:
        m = re.match(r"(PC\d+)_(\d{8})_(\d{6})_(\d+)", fn)
        site, date, tm, seq = m.groups()
        out.append(dict(site=site, year=int(date[:4]), month=int(date[4:6]),
                        day=int(date[6:8]), hour=int(tm[:2]), minute=int(tm[2:4]),
                        seq=int(seq), date=date, time=tm))
    m = pd.DataFrame(out)
    return pd.concat([df.reset_index(drop=True), m], axis=1)
