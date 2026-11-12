import os, sys, argparse, itertools, json
import numpy as np, pandas as pd
import cv2
from scipy.ndimage import label, binary_fill_holes
from scipy.spatial import cKDTree

WS = "/download_job_code/payload/sandbox/workspaces/capella_zeta"
CLASSES = ["large_bowel", "small_bowel", "stomach"]
S = 320
ROOT = "/download_job_code/payload/sandbox"


def rle_encode(mask):
    pix = mask.reshape(-1)
    if pix.max() == 0:
        return ""
    p = np.concatenate([[0], pix, [0]])
    runs = np.where(p[1:] != p[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(map(str, runs))


def pp_mask(prob2d, thr, min_area=0, keep_largest=False, fill=True):
    m = prob2d > thr
    if not m.any():
        return m.astype(np.uint8)
    if fill:
        m = binary_fill_holes(m)
    if keep_largest or min_area > 0:
        lab, n = label(m)
        if n > 1:
            sizes = np.bincount(lab.ravel()); sizes[0] = 0
            if keep_largest:
                m = lab == sizes.argmax()
            elif min_area > 0:
                keep = np.where(sizes >= min_area)[0]; keep = keep[keep != 0]
                m = np.isin(lab, keep)
    return m.astype(np.uint8)


def pp_slice(prob, params, H=320, W=320, out_hw=None):
    masks = np.zeros((3, H, W), np.uint8)
    for c in range(3):
        p = params[c]
        masks[c] = pp_mask(prob[c], p["thr"], p.get("min_area", 0), p.get("keep_largest", False), p.get("fill", True))
    if out_hw is not None and (out_hw[0] != H or out_hw[1] != W):
        out = np.zeros((3, out_hw[0], out_hw[1]), np.uint8)
        for c in range(3):
            out[c] = cv2.resize(masks[c], (out_hw[1], out_hw[0]), interpolation=cv2.INTER_NEAREST)
        return out
    return masks


def dice_stats(pr, gt):
    # pr, gt [N,H,W] bool
    inter = (pr & gt).sum((1, 2))
    den = pr.sum((1, 2)) + gt.sum((1, 2))
    d = np.where(den > 0, 2 * inter / np.maximum(den, 1), 1.0)
    return d, (gt.sum((1, 2)) > 0), (pr.sum((1, 2)) > 0)


def hd_norm(pred, gt, diag):
    pe, ge = pred.any(), gt.any()
    if not pe and not ge:
        return 0.0
    if not pe or not ge:
        return 1.0
    p = np.argwhere(pred); g = np.argwhere(gt)
    d1 = cKDTree(g).query(p)[0].max()
    d2 = cKDTree(p).query(g)[0].max()
    return float(min(1.0, max(d1, d2) / diag))


def tune(prob_file, ids_file, gt_file, n_hd=600):
    probs = np.load(prob_file)
    ids = pd.read_csv(ids_file)
    gtm = np.load(gt_file, mmap_mode="r")
    idx = pd.read_csv(f"{WS}/cache/train_index.csv")
    id2row = dict(zip(idx["id"], np.arange(len(idx))))
    rows = np.array([id2row[i] for i in ids["id"]])
    N = len(ids)
    print("N", N, probs.shape)
    diag = np.sqrt(2 * S * S)
    best = {}
    for c in range(3):
        gt = gtm[rows, c].astype(bool)
        pc = probs[:, c]
        res = []
        for thr in np.arange(0.20, 0.91, 0.05):
            pr = pc > thr
            d, gpos, ppos = dice_stats(pr, gt)
            res.append((round(d.mean(), 4), round(d[gpos].mean() if gpos.any() else 0, 4), round(thr, 2)))
        res.sort(reverse=True)
        print(CLASSES[c], "thr sweep (macro,pos,thr):", res[:5])
        thr = res[0][2]
        # min_area sweep at best thr
        base = pc > thr
        best_cfg = None; best_score = -1
        for min_area in [0, 10, 25, 50, 100, 200, 400]:
            for keep in [False, True]:
                if keep and min_area > 0:
                    continue
                ms = np.zeros((N, S, S), np.uint8)
                for i in range(N):
                    ms[i] = pp_mask(pc[i], thr, min_area, keep, True)
                d, gpos, ppos = dice_stats(ms.astype(bool), gt)
                sc = d.mean()
                if sc > best_score:
                    best_score = sc; best_cfg = {"thr": float(thr), "min_area": int(min_area),
                                                 "keep_largest": bool(keep), "fill": True}
            print(f"  {CLASSES[c]} min_area {min_area} keep{keep}: macro {d.mean():.4f}")
        best[CLASSES[c]] = best_cfg
        # HD proxy on subset
        rng = np.random.RandomState(0)
        sub = rng.choice(N, min(n_hd, N), replace=False)
        ms = np.stack([pp_mask(pc[i], best_cfg["thr"], best_cfg["min_area"], best_cfg["keep_largest"], True) for i in sub])
        hds = np.array([hd_norm(ms[j].astype(bool), gt[sub[j]], diag) for j in range(len(sub))])
        dsub, _, _ = dice_stats(ms.astype(bool), gt[sub])
        combo = 0.4 * dsub + 0.6 * (1 - hds)
        print(f"  {CLASSES[c]} chosen {best_cfg} | dice {dsub.mean():.4f} hd {hds.mean():.4f} combo {combo.mean():.4f}")
    print(json.dumps(best, indent=2))
    with open(f"{WS}/oof/tuned_params.json", "w") as f:
        json.dump(best, f, indent=2)
    return best


def write_submission(prob_file, ids_file, params, out_csv, sample_csv=f"{ROOT}/data/sample_submission.csv"):
    probs = np.load(prob_file)
    ids = pd.read_csv(ids_file)
    print("prob", probs.shape, "ids", len(ids))
    assert len(ids) == len(probs)
    lut = {}
    for i in range(len(ids)):
        r = ids.iloc[i]
        ps = [params[c] for c in range(3)]
        masks = pp_slice(probs[i].astype(np.float32), ps, S, S, (int(r["H"]), int(r["W"])))
        lut[r["id"]] = masks
    ss = pd.read_csv(sample_csv)
    preds = []
    missing = set()
    for _, r in ss.iterrows():
        if r["id"] not in lut:
            missing.add(r["id"]); preds.append(""); continue
        c = CLASSES.index(r["class"])
        preds.append(rle_encode(lut[r["id"]][c]))
    ss["predicted"] = preds
    ss.to_csv(out_csv, index=False)
    print("wrote", out_csv, "rows", len(ss), "empty", (ss.predicted == "").sum(),
          "missing ids", len(missing))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prob_file"); ap.add_argument("--ids_file")
    ap.add_argument("--gt_file", default=f"{WS}/cache/train_mask{S}.npy")
    ap.add_argument("--mode", default="tune")  # tune | submit
    ap.add_argument("--out_csv", default=f"{ROOT}/final_solution/submission.csv")
    ap.add_argument("--params", default=f"{WS}/oof/tuned_params.json")
    args = ap.parse_args()
    if args.mode == "tune":
        tune(args.prob_file, args.ids_file, args.gt_file)
    else:
        params = json.load(open(args.params))
        ps = [params[c] for c in CLASSES]
        write_submission(args.prob_file, args.ids_file, ps, args.out_csv)
