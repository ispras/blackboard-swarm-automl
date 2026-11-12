"""Combine one or more prob dumps into a submission.csv (C-order RLE) with optional postproc."""
import os, sys, argparse
import numpy as np
import pandas as pd
import cv2
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CLASSES, DATA, rle_encode


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--prefixes", nargs="+", required=True, help="prob dump prefixes (without _probs.npy)")
    p.add_argument("--weights", nargs="*", type=float, default=None)
    p.add_argument("--thresh", type=float, default=0.5)
    p.add_argument("--thresh_classes", default="", help="e.g. 0.7,0.7,0.6 for [lb,sb,st]")
    p.add_argument("--min_size", type=int, default=0)
    p.add_argument("--fill_holes", type=int, default=1)
    p.add_argument("--keep_largest", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--ids_from", default=os.path.join(DATA, "test.csv"))
    return p.parse_args()


def postproc(mask, min_size, fill_holes, keep_largest):
    if mask.sum() == 0:
        return mask
    if fill_holes:
        mask = ndimage.binary_fill_holes(mask).astype(np.uint8)
    if keep_largest:
        lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
        if n > 1:
            sizes = ndimage.sum(mask, lab, range(1, n + 1))
            mask = (lab == (np.argmax(sizes) + 1)).astype(np.uint8)
    if min_size > 0:
        lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
        if n > 0:
            sizes = ndimage.sum(mask, lab, range(1, n + 1))
            keep = np.zeros(n + 1, np.uint8)
            keep[1:] = (sizes >= min_size).astype(np.uint8)
            mask = keep[lab]
    return mask.astype(np.uint8)


def main():
    a = parse()
    # load each prob dump -> dict id -> (3,320,320) float16 ; accumulate weighted
    acc = {}
    wsum = {}
    id2hw = {}
    wts = a.weights if a.weights else [1.0] * len(a.prefixes)
    for pref, w in zip(a.prefixes, wts):
        probs = np.load(pref + "_probs.npy")
        meta = np.load(pref + "_meta.npz", allow_pickle=True)
        ids = meta["ids"]; hw = meta["orig_hw"]
        for i, id_ in enumerate(ids):
            if id_ not in acc:
                acc[id_] = probs[i].astype(np.float32) * w
                wsum[id_] = w
            else:
                acc[id_] += probs[i].astype(np.float32) * w
                wsum[id_] += w
            id2hw[id_] = (int(hw[i][0]), int(hw[i][1]))
    for k in acc:
        acc[k] /= wsum[k]
    print(f"loaded {len(acc)} unique ids from {len(a.prefixes)} dump(s)")

    df = pd.read_csv(a.ids_from)
    rows = []
    missing = 0
    for id_, cls, _ in zip(df["id"], df["class"], df.get("predicted", df["id"])):
        p = acc.get(id_)
        if p is None:
            rows.append(""); missing += 1
            continue
        H, W = id2hw[id_]
        ci = CLASSES.index(cls)
        pr = cv2.resize(p[ci], (W, H), interpolation=cv2.INTER_LINEAR)
        thr = a.thresh
        if a.thresh_classes:
            thr = [float(x) for x in a.thresh_classes.split(",")][ci]
        m = (pr > thr).astype(np.uint8)
        m = postproc(m, a.min_size, a.fill_holes, a.keep_largest)
        rows.append(rle_encode(m))
    out = pd.DataFrame({"id": df["id"], "class": df["class"], "predicted": rows})
    out.to_csv(a.out, index=False)
    print(f"wrote {a.out} rows={len(out)} missing={missing} empty={(out['predicted']=='').mean():.3f}")


if __name__ == "__main__":
    main()
