import argparse
import json
import os
import numpy as np
import pandas as pd

import common as C


def write_submission(pred, out_csv):
    """pred: (240, 107, 5) in test order, columns = ALL_TARGETS order."""
    te = C.load_test()
    assert pred.shape[0] == len(te)
    ids, cols = [], {c: [] for c in C.ALL_TARGETS}
    for i, r in enumerate(te):
        L = r["seq_length"]
        for p in range(L):
            ids.append(f"{r['id']}_{p}")
            for j, c in enumerate(C.ALL_TARGETS):
                cols[c].append(float(pred[i, p, j]))
    df = pd.DataFrame({"id_seqpos": ids, **cols})
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df.to_csv(out_csv, index=False, float_format="%.10f")
    print("wrote", out_csv, df.shape)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="(240,107,5) npy")
    ap.add_argument("--out", default="submission.csv")
    args = ap.parse_args()
    write_submission(np.load(args.pred), args.out)
