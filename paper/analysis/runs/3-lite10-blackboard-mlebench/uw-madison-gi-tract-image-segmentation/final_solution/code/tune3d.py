"""3D-aware postprocessing tuner + submission writer (per-slice Dice + per-scan 3D HD)."""
import os, sys, argparse, json
import numpy as np, pandas as pd
import cv2
from scipy import ndimage
from scipy.spatial import cKDTree

WS = "/download_job_code/payload/sandbox/workspaces/capella_zeta"
ROOT = "/download_job_code/payload/sandbox"
S = 320
CLASSES = ["large_bowel", "small_bowel", "stomach"]


def dice_bin(pr, gt):
    pr = pr.astype(bool); gt = gt.astype(bool)
    s = pr.sum() + gt.sum()
    if s == 0:
        return 1.0
    return 2.0 * (pr & gt).sum() / s


def surf(vol):
    er = ndimage.binary_erosion(vol, iterations=1)
    p = np.argwhere(vol & ~er)
    if len(p) == 0:
        p = np.argwhere(vol)
    return p.astype(np.float32)


def hd_surf(pred, gt_surf_norm, dims):
    pa = pred.any()
    if not pa:
        return 0.0 if gt_surf_norm is None else 1.0
    if gt_surf_norm is None:
        return 1.0
    a = surf(pred)
    a = a / dims
    d = max(cKDTree(gt_surf_norm).query(a)[0].max(), cKDTree(a).query(gt_surf_norm)[0].max())
    return float(min(d / np.sqrt(3.0), 1.0))


def fill2d(m):
    return ndimage.binary_fill_holes(m).astype(np.uint8)


def remove_small2d(m, ms):
    if ms <= 0 or not m.any():
        return m
    lab, n = ndimage.label(m, structure=np.ones((3, 3)))
    if n <= 1:
        return m
    sizes = ndimage.sum(m, lab, range(1, n + 1))
    keep = np.zeros(n + 1, np.uint8); keep[1:] = (sizes >= ms).astype(np.uint8)
    return keep[lab]


def binarize_volume(pvol, thr, fill, ms2d):
    Z, H, W = pvol.shape
    out = np.zeros((Z, H, W), np.uint8)
    for z in range(Z):
        m = (pvol[z] > thr).astype(np.uint8)
        if m.any():
            if fill:
                m = fill2d(m)
            if ms2d > 0:
                m = remove_small2d(m, ms2d)
        out[z] = m
    return out


def apply3d(out, cfg):
    if cfg.get("scan_empty", 0) > 0 and out.sum() < cfg["scan_empty"]:
        return np.zeros_like(out)
    if cfg.get("min_vol3d", 0) > 0 or cfg.get("keep_largest3d", False):
        lab, n = ndimage.label(out, structure=np.ones((3, 3, 3)))
        if n > 0:
            sizes = ndimage.sum(out, lab, range(1, n + 1))
            if cfg.get("keep_largest3d", False):
                out = (lab == (int(np.argmax(sizes)) + 1)).astype(np.uint8)
            elif cfg["min_vol3d"] > 0:
                keep = np.zeros(n + 1, np.uint8); keep[1:] = (sizes >= cfg["min_vol3d"]).astype(np.uint8)
                out = keep[lab]
    return out


def scan_groups(ids):
    ids = ids.copy(); ids["pos"] = np.arange(len(ids))
    groups = []
    for cd, g in ids.groupby("case_day"):
        g = g.sort_values("slice")
        groups.append((cd, g["pos"].values, g["id"].values))
    return groups


def rle_encode(mask):
    pix = mask.reshape(-1)
    if pix.max() == 0:
        return ""
    p = np.concatenate([[0], pix, [0]])
    runs = np.where(p[1:] != p[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(map(str, runs))


def tune(prob_file, ids_file, gt_file):
    probs = np.load(prob_file, mmap_mode="r")
    ids = pd.read_csv(ids_file)
    assert len(ids) == probs.shape[0]
    idx = pd.read_csv(f"{WS}/cache/train_index.csv")
    id2row = dict(zip(idx["id"], np.arange(len(idx))))
    groups = scan_groups(ids)
    gtm = np.load(gt_file, mmap_mode="r")
    print("scans", len(groups), "slices", len(ids), flush=True)
    best_all = {}
    for cls in range(3):
        P = np.asarray(probs[:, cls])  # [N,320,320] fp16 in RAM
        gts = {}
        gt_surf = {}
        for cd, pos, gids in groups:
            rows = np.array([id2row[i] for i in gids])
            g = np.asarray(gtm[rows, cls]).astype(np.uint8)
            gts[cd] = g
            gt_surf[cd] = surf(g) / np.array([len(gids), S, S], np.float32) if g.any() else None
        # phase A: threshold
        resA = []
        for thr in (0.40, 0.55, 0.70):
            dices = []; hds = []
            for cd, pos, gids in groups:
                out = binarize_volume(P[pos], thr, True, 0)
                g = gts[cd]
                for z in range(len(gids)):
                    dices.append(dice_bin(out[z], g[z]))
                hds.append(hd_surf(out, gt_surf[cd], np.array([len(gids), S, S], np.float32)))
            d, h = float(np.mean(dices)), float(np.mean(hds))
            resA.append((0.4 * d + 0.6 * (1 - h), d, h, thr))
            print(f"[{CLASSES[cls]}] thr {thr}: dice {d:.4f} hd {h:.4f} combo {0.4*d+0.6*(1-h):.4f}", flush=True)
        resA.sort(reverse=True)
        t = resA[0][3]
        base = {cd: binarize_volume(P[pos], t, True, 0) for cd, pos, gids in groups}
        resB = []
        for mv, se in ((0, 0), (0, 300), (400, 0), (400, 300), (1200, 300)):
            cfg = {"thr": t, "fill": True, "min_size2d": 0,
                   "min_vol3d": mv, "keep_largest3d": False, "scan_empty": se}
            dices = []; hds = []
            for cd, pos, gids in groups:
                out = apply3d(base[cd], cfg)
                g = gts[cd]
                for z in range(len(gids)):
                    dices.append(dice_bin(out[z], g[z]))
                hds.append(hd_surf(out, gt_surf[cd], np.array([len(gids), S, S], np.float32)))
            d, h = float(np.mean(dices)), float(np.mean(hds))
            resB.append((0.4 * d + 0.6 * (1 - h), d, h, dict(cfg)))
            print(f"[{CLASSES[cls]}] mv {mv} se {se}: dice {d:.4f} hd {h:.4f} combo {0.4*d+0.6*(1-h):.4f}", flush=True)
        resB.sort(key=lambda r: -r[0])
        best_all[CLASSES[cls]] = {"combo": resB[0][0], "dice": resB[0][1], "hd": resB[0][2], **resB[0][3]}
        print(f"BEST {CLASSES[cls]}: {best_all[CLASSES[cls]]}", flush=True)
        del P, gts, gt_surf, base
    md = np.mean([best_all[c]["dice"] for c in CLASSES])
    mh = np.mean([best_all[c]["hd"] for c in CLASSES])
    print(f"OVERALL dice {md:.4f} hd {mh:.4f} combo {0.4*md+0.6*(1-mh):.4f}", flush=True)
    json.dump(best_all, open(f"{WS}/oof/tuned3d_params.json", "w"), indent=2)
    return best_all


def write_submission(prob_file, ids_file, params, out_csv,
                     sample_csv=f"{ROOT}/data/sample_submission.csv"):
    probs = np.load(prob_file, mmap_mode="r")
    ids = pd.read_csv(ids_file)
    groups = scan_groups(ids)
    lut = {}
    for cd, pos, gids in groups:
        for cls in range(3):
            pvol = np.asarray(probs[pos, cls]).astype(np.float32)
            out = apply3d(binarize_volume(pvol, params[CLASSES[cls]]["thr"],
                                          params[CLASSES[cls]].get("fill", True),
                                          params[CLASSES[cls]].get("min_size2d", 0)),
                          params[CLASSES[cls]])
            for k, i in enumerate(pos):
                r = ids.iloc[i]
                H, W = int(r["H"]), int(r["W"])
                m = cv2.resize(out[k], (W, H), interpolation=cv2.INTER_NEAREST)
                lut[(r["id"], cls)] = rle_encode(m)
    ss = pd.read_csv(sample_csv)
    preds = [lut.get((r["id"], CLASSES.index(r["class"])), "") for _, r in ss.iterrows()]
    ss["predicted"] = preds
    ss.to_csv(out_csv, index=False)
    print("wrote", out_csv, "rows", len(ss), "empty", (ss.predicted == "").sum())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="tune")
    ap.add_argument("--prob_file"); ap.add_argument("--ids_file")
    ap.add_argument("--gt_file", default=f"{WS}/cache/train_mask{S}.npy")
    ap.add_argument("--params", default=f"{WS}/oof/tuned3d_params.json")
    ap.add_argument("--out_csv", default=f"{ROOT}/final_solution/submission.csv")
    a = ap.parse_args()
    if a.mode == "tune":
        tune(a.prob_file, a.ids_file, a.gt_file)
    else:
        params = json.load(open(a.params))
        write_submission(a.prob_file, a.ids_file, params, a.out_csv)
