import os, json
import numpy as np
import pandas as pd

ROOT = "/download_job_code/payload/sandbox"
OUT = f"{ROOT}/workspaces/mimosa_omicron"
ME = f"{ROOT}/workspaces/aldebaran_omicron"
NTEST = 9914


def find_test(tag):
    for f in [f"{OUT}/test_pred_{tag}.npy", f"{ME}/test_pred_mimosa_{tag}.npy",
              f"{OUT}/submission_{tag}.csv"]:
        if os.path.exists(f):
            return np.load(f) if f.endswith(".npy") else pd.read_csv(f)["fare_amount"].to_numpy()
    return None


def main():
    res = json.load(open(f"{OUT}/blend_result.json"))
    tags = res["tags"]; w = res["weights"]
    pv = np.zeros(NTEST, dtype=np.float64); wsum = 0.0
    for t in tags:
        wt = float(w.get(t, 0.0))
        if wt <= 1e-6:
            continue
        p = find_test(t)
        if p is None:
            print("missing test pred", t); continue
        pv += wt * p; wsum += wt
        print(f"use {t} w={wt:.4f}")
    pv += float(w.get("bias", 0.0))
    if wsum > 0:
        pv /= wsum
    # corrupt override
    cpath = f"{ME}/corrupt_test_pred.npy"
    if os.path.exists(cpath):
        cp = np.load(cpath)
        cm = np.isfinite(cp)
        pv[cm] = cp[cm]
        print("corrupt override", int(cm.sum()))
    pv = np.clip(pv, 2.5, 500.0)
    sub = pd.DataFrame({"key": pd.read_csv(f"{ROOT}/data/test.csv")["key"], "fare_amount": pv})
    sub.to_csv(f"{OUT}/submission_mimosa_blend.csv", index=False)
    np.save(f"{OUT}/test_pred_mimosa_blend.npy", pv.astype(np.float32))
    print("saved", sub["fare_amount"].describe().to_dict())


if __name__ == "__main__":
    main()
