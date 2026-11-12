"""Build a test submission from prob dumps by id, with per-class 2D+3D postproc. C-order RLE."""
import os, sys, argparse
import numpy as np
from scipy import ndimage
import cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dataset as D
from common import DATA, CLASSES, rle_encode
from ens_eval import read_ids, pp2d, cc3

S = 320


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", nargs="+", required=True)
    ap.add_argument("--weights", nargs="*", type=float, default=None)
    ap.add_argument("--th", default="0.55,0.55,0.7")
    ap.add_argument("--minvol", default="1200,400,1200")
    ap.add_argument("--minarea", type=int, default=50)
    ap.add_argument("--ref_meta", required=True, help="test meta npz with ids,scan_id,k,orig_hw")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ths = [float(x) for x in a.th.split(",")]
    mvs = [int(x) for x in a.minvol.split(",")]
    wts = a.weights if a.weights else [1.0] * len(a.specs)

    ref = np.load(a.ref_meta, allow_pickle=True)
    id2scan = {str(i): (str(s), int(k)) for i, s, k in zip(ref["ids"], ref["scan_id"], ref["k"])}
    id2hw = {str(i): (int(h), int(w)) for i, (h, w) in zip(ref["ids"], ref["orig_hw"])}
    want = set(id2scan.keys())
    scan_ids = sorted({v[0] for v in id2scan.values()})
    # scan lengths
    meta = {m["scan_id"]: m for m in D.load_meta("test")}

    acc = {}; wsum = {}
    for sp, w in zip(a.specs, wts):
        pp, ids, order = read_ids(sp)
        pm = np.load(pp, mmap_mode="r")
        for r, id_ in enumerate(order):
            if id_ not in want:
                continue
            if id_ not in acc:
                acc[id_] = np.asarray(pm[r]).astype(np.float32) * w; wsum[id_] = w
            else:
                acc[id_] += np.asarray(pm[r]).astype(np.float32) * w; wsum[id_] += w
        del pm
    print(f"accumulated {len(acc)} ids from {len(a.specs)} specs", flush=True)

    # build per scan,class volume
    scanvol = {sid: np.zeros((3, meta[sid]["n"], S, S), np.uint8) for sid in scan_ids}
    for id_, p in acc.items():
        p = p / wsum[id_]
        sid, k = id2scan[id_]
        for c in range(3):
            m = (p[c] > ths[c]).astype(np.uint8)
            if m.sum():
                m = pp2d(m, a.minarea, 1)
            scanvol[sid][c, k] = m
    for sid in scan_ids:
        for c in range(3):
            scanvol[sid][c] = cc3(scanvol[sid][c], mvs[c])
    print("built volumes", flush=True)

    id2rle = {}
    for id_, (sid, k) in id2scan.items():
        H, W = id2hw[id_]
        rles = []
        for c in range(3):
            m = cv2.resize(scanvol[sid][c, k], (W, H), interpolation=cv2.INTER_NEAREST)
            rles.append(rle_encode(m))
        id2rle[id_] = rles
    import pandas as pd
    df = pd.read_csv(os.path.join(DATA, "test.csv"))
    preds = []
    miss = 0
    for id_, cls in zip(df["id"], df["class"]):
        if id_ in id2rle:
            preds.append(id2rle[id_][CLASSES.index(cls)])
        else:
            preds.append(""); miss += 1
    out = pd.DataFrame({"id": df["id"], "class": df["class"], "predicted": preds})
    out.to_csv(a.out, index=False)
    print(f"wrote {a.out} rows={len(out)} missing={miss} empty={(out['predicted']=='').mean():.3f}", flush=True)


if __name__ == "__main__":
    main()
