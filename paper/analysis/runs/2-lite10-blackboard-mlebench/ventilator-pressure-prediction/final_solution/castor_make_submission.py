"""Build submission from breath-major (7545,80) test prediction arrays.
All arrays are ordered by sorted (breath_id, id), same as data/test.csv sorted.
Usage: python make_submission.py out.csv name=path [name=path ...]
Weights default equal if not given; pass path=weight e.g. a.npy=0.2
"""
import sys, numpy as np, pandas as pd

ROOT = "/download_job_code/payload/sandbox"


def main():
    out = sys.argv[1]
    specs = sys.argv[2:]
    preds, ws = [], []
    for s in specs:
        p, _, w = s.partition("=")
        preds.append(np.load(p).astype(np.float64))
        ws.append(float(w) if w else 1.0)
    ws = np.array(ws); ws = ws / ws.sum()
    P = sum(w * p for w, p in zip(ws, preds))
    flat = np.clip(P.reshape(-1), -2.0, 65.0)
    te = pd.read_csv(f"{ROOT}/data/test.csv", usecols=["id", "breath_id"])
    order = te.sort_values(["breath_id", "id"]).reset_index(drop=True)
    sub = pd.DataFrame({"id": order["id"].values, "pressure": flat})
    sample = pd.read_csv(f"{ROOT}/data/sample_submission.csv", usecols=["id"])
    sub = sample.merge(sub, on="id", how="left")
    assert sub.pressure.notna().all() and len(sub) == 603600
    sub["pressure"] = sub["pressure"].round(6)
    sub.to_csv(out, index=False)
    print("wrote", out, sub.shape, "mean", round(sub.pressure.mean(), 4),
          "min", round(sub.pressure.min(), 3), "max", round(sub.pressure.max(), 3))


if __name__ == "__main__":
    main()
