import os, sys, json, argparse
import numpy as np, pandas as pd, cv2
from scipy import ndimage
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import rle_decode, rle_encode, build_hw_dict, get_mask_map, CLASSES
from metric import combo_scores

ROOT = "/download_job_code/payload/sandbox"
WS = f"{ROOT}/workspaces/aldebaran_omicron"
CACHE = f"{WS}/cache"


def clean(mask, min_size=0, keep_largest=True, fill_holes=False):
    if not mask.any():
        return mask
    if min_size > 0 or keep_largest:
        lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
        if n > 1:
            sizes = ndimage.sum(mask, lab, range(1, n + 1))
            keep = sizes >= max(min_size, 1)
            if keep_largest and keep.any():
                keep[int(np.argmax(sizes))] = True
            allowed = np.zeros(n + 1, bool)
            allowed[1:] = keep
            mask = allowed[lab].astype(np.uint8)
    if fill_holes:
        mask = ndimage.binary_fill_holes(mask).astype(np.uint8)
    return mask


def apply_probs(prob, thrs, min_sizes, keep_largest=True, fill_holes=False):
    """prob: (3,S,S) uint8 or float. returns list of 3 binary masks at S."""
    out = []
    for c in range(3):
        is_u8 = (prob.dtype == np.uint8)
        p = prob[c].astype(np.float32)
        if is_u8 or p.max() > 1.01:
            p = p / 255.0
        m = (p > thrs[c]).astype(np.uint8)
        m = clean(m, min_sizes[c], keep_largest, fill_holes)
        out.append(m)
    return out


def eval_val(prob_path, idx_csv, thrs, min_sizes, keep_largest=True, size=320):
    meta = pd.read_csv(idx_csv)
    probs = np.load(prob_path, mmap_mode="r")
    mm = get_mask_map()
    hw = build_hw_dict("train")
    gt, pred = {}, {}
    meta = meta.reset_index(drop=True)
    for cd, sub in meta.groupby("case_day"):
        order = sub.sort_values("slice_int").index.values
        for ci, c in enumerate(CLASSES):
            gv = []
            pv = []
            for i in order:
                r = meta.loc[i]
                h, w = hw[r.id]
                gv.append(rle_decode(mm.at[r.id, c] if r.id in mm.index else None, h, w, size))
                pv.append(clean((probs[i, ci] > thrs[ci]).astype(np.uint8), min_sizes[ci], keep_largest))
            gt[(cd, c)] = np.stack(gv, 0).transpose(1, 2, 0)  # H,W,Z
            pred[(cd, c)] = np.stack(pv, 0).transpose(1, 2, 0)
    res = combo_scores(pred, gt, {})
    return res


def make_submission(prob_path, idx_csv, split, out_csv, thrs, min_sizes,
                    keep_largest=True, fill_holes=False, size=320):
    meta = pd.read_csv(idx_csv).reset_index(drop=True)
    probs = np.load(prob_path, mmap_mode="r")
    hw = build_hw_dict(split)
    rows = []
    for i, r in enumerate(meta.itertuples(index=False)):
        h, w = hw[r.id]
        masks = apply_probs(probs[i], thrs, min_sizes, keep_largest, fill_holes)
        for ci, c in enumerate(CLASSES):
            m = masks[ci]
            if m.sum() == 0:
                rle = ""
            else:
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                rle = rle_encode(m)
            rows.append((r.id, c, rle))
    df = pd.DataFrame(rows, columns=["id", "class", "predicted"])
    df.to_csv(out_csv, index=False)
    print("wrote", out_csv, len(df), flush=True)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prob", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--thr", default="0.5,0.5,0.5")
    ap.add_argument("--min_size", default="0,0,0")
    ap.add_argument("--no_largest", action="store_true")
    ap.add_argument("--fill_holes", action="store_true")
    args = ap.parse_args()
    thrs = [float(x) for x in args.thr.split(",")]
    ms = [int(x) for x in args.min_size.split(",")]
    if args.eval:
        res = eval_val(args.prob, args.index, thrs, ms, not args.no_largest)
        print(json.dumps(res, indent=1))
    else:
        make_submission(args.prob, args.index, args.split, args.out, thrs, ms,
                        not args.no_largest, args.fill_holes)
