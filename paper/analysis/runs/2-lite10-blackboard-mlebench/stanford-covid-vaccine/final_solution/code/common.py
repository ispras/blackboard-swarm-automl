import json
import numpy as np
import os

DATA_DIR = os.environ.get("DATA_DIR", "data")

SEQ_VOCAB = list("ACGU")
STRUCT_VOCAB = list(".()")
LOOP_VOCAB = list("SMI BHEX".replace(" ", ""))  # S M I B H E X

SCORED = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
ALL_TARGETS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def load_train():
    return load_jsonl(os.path.join(DATA_DIR, "train.json"))


def load_test():
    return load_jsonl(os.path.join(DATA_DIR, "test.json"))


def one_hot(seq, vocab):
    m = {c: i for i, c in enumerate(vocab)}
    arr = np.zeros((len(seq), len(vocab)), dtype=np.float32)
    for i, c in enumerate(seq):
        if c in m:
            arr[i, m[c]] = 1.0
    return arr


def featurize(rec):
    """(L,14) float32 features for one record."""
    return np.concatenate(
        [
            one_hot(rec["sequence"], SEQ_VOCAB),
            one_hot(rec["structure"], STRUCT_VOCAB),
            one_hot(rec["predicted_loop_type"], LOOP_VOCAB),
        ],
        axis=1,
    )


def make_arrays(records, max_len=107):
    X = np.zeros((len(records), max_len, 14), dtype=np.float32)
    for i, r in enumerate(records):
        f = featurize(r)
        X[i, : f.shape[0]] = f
    return X


def make_targets(records, scored_len=68):
    Y = np.zeros((len(records), scored_len, 5), dtype=np.float32)
    for i, r in enumerate(records):
        for j, c in enumerate(ALL_TARGETS):
            Y[i, :scored_len, j] = np.asarray(r[c], dtype=np.float32)
    return Y


def mcrmse(y_true, y_pred, cols=(0, 1, 3)):
    """y_true/y_pred (N, L, 5) on scored positions. Returns MCRMSE over cols."""
    rmses = []
    for j in cols:
        d = y_true[..., j] - y_pred[..., j]
        rmses.append(np.sqrt(np.mean(d * d)))
    return float(np.mean(rmses)), [float(x) for x in rmses]


def kfold_indices(n=2160, n_splits=5, seed=42):
    from sklearn.model_selection import KFold
    return list(KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(np.arange(n)))
