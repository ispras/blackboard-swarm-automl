import os, sys, argparse, json
import numpy as np, pandas as pd
WS = "/download_job_code/payload/sandbox/workspaces/capella_zeta"
CLASSES = ["large_bowel", "small_bowel", "stomach"]


def load_and_average(specs, out_path=None):
    # specs: list of dict(prob=..., ids=..., w=...)
    ref = None
    arrays = []
    for s in specs:
        p = np.load(s["prob"]).astype(np.float32)
        ids = pd.read_csv(s["ids"])
        m = dict(zip(ids["id"], np.arange(len(ids))))
        if ref is None:
            ref = ids[["id"]].copy()
            ref["order"] = np.arange(len(ref))
        else:
            ref_set = set(ref["id"]); new = set(m)
            if ref_set != new:
                print(f"WARN id mismatch: {len(ref_set - new)} missing, {len(new - ref_set)} extra")
        order = np.array([m[i] for i in ref["id"]])
        arrays.append((p[order], s.get("w", 1.0)))
        print("loaded", s["prob"], p.shape, "w", s.get("w", 1.0))
    tot = sum(w for _, w in arrays)
    avg = np.zeros_like(arrays[0][0])
    for a, w in arrays:
        avg += a * (w / tot)
    if out_path:
        np.save(out_path, avg.astype(np.float16))
        ref[["id"]].to_csv(out_path.replace("_probs.npy", "_ids.csv"), index=False)
        print("saved", out_path)
    return ref, avg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", required=True, help="json list of {prob,ids,w}")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    specs = json.loads(a.specs)
    load_and_average(specs, a.out)
