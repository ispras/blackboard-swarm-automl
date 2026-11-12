"""Per-class postproc tuning (thr x min_vol3d x min_area) on a val prob dump, grader-aligned metric.
Dice = mean over per-slice (id,class); HD = per (scan,class) 3D, normalized (z/Z,y/Y,x/X)/sqrt(3)."""
import os, sys, argparse
import numpy as np
from scipy import ndimage
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dataset as D

CACHE = D.CACHE; S = 320


def load(a):
    meta = D.load_meta("train"); metai = {m["scan_id"]: m for m in meta}
    prob_scan = {}
    for pref, w in zip(a.prefixes, (a.weights if a.weights else [1.0] * len(a.prefixes))):
        probs = np.load(pref + "_probs.npy"); mz = np.load(pref + "_meta.npz", allow_pickle=True)
        sids = mz["scan_id"]; ks = mz["k"]
        for i in range(len(sids)):
            sid = str(sids[i]); k = int(ks[i])
            if sid not in prob_scan:
                prob_scan[sid] = np.zeros((3, metai[sid]["n"], S, S), np.float32)
            prob_scan[sid][:, k] += probs[i].astype(np.float32) * w
    for sid in prob_scan:
        prob_scan[sid] /= sum(a.weights) if a.weights else len(a.prefixes)
    gts = {}
    for sid in prob_scan:
        gt = np.load(os.path.join(CACHE, metai[sid]["msk"]), mmap_mode="r")
        gts[sid] = (np.asarray(gt) > 0).astype(bool).transpose(3, 0, 1, 2)
    return prob_scan, gts


def pp2d(m, min_area, fill):
    if m.sum() == 0:
        return m
    if fill:
        m = ndimage.binary_fill_holes(m).astype(np.uint8)
    if min_area > 0:
        lab, n = ndimage.label(m, structure=np.ones((3, 3)))
        if n > 0:
            sz = ndimage.sum(m, lab, range(1, n + 1))
            kv = np.zeros(n + 1, np.uint8); kv[1:] = (sz >= min_area)
            m = kv[lab]
    return m.astype(np.uint8)


def cc3(vol, minvol):
    if minvol <= 0 or vol.sum() == 0:
        return vol
    lab, n = ndimage.label(vol, structure=np.ones((3, 3, 3)))
    if n <= 1:
        return vol
    sz = ndimage.sum(vol, lab, range(1, n + 1))
    kv = np.zeros(n + 1, np.uint8); kv[1:] = (sz >= minvol)
    return kv[lab].astype(np.uint8)


def surf(vol):
    er = ndimage.binary_erosion(vol, iterations=1)
    z, y, x = np.nonzero(vol & ~er)
    if len(z) == 0:
        z, y, x = np.nonzero(vol)
    return np.stack([z, y, x], 1).astype(np.float32)


def hd3(pred, gt, dims):
    pb = pred.astype(bool); gb = gt
    if not pb.any() and not gb.any():
        return 0.0
    if pb.any() != gb.any():
        return 1.0
    from scipy.spatial.distance import directed_hausdorff
    a = surf(pb) / dims; b = surf(gb) / dims
    d = max(directed_hausdorff(a, b)[0], directed_hausdorff(b, a)[0])
    return min(d / np.sqrt(3), 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefixes", nargs="+", required=True)
    ap.add_argument("--weights", nargs="*", type=float, default=None)
    ap.add_argument("--ths", nargs="+", type=float, default=[0.5, 0.6, 0.7, 0.8])
    ap.add_argument("--minvols", nargs="+", type=int, default=[0, 400, 1200])
    ap.add_argument("--minarea", type=int, default=50)
    a = ap.parse_args()
    prob_scan, gts = load(a)
    print("scans", len(prob_scan), flush=True)
    for c in range(3):
        best = None
        for th in a.ths:
            preds = {}
            for sid, pv in prob_scan.items():
                n = pv.shape[1]; vol = np.zeros((n, S, S), np.uint8)
                for z in range(n):
                    m = (pv[c, z] > th).astype(np.uint8)
                    if m.sum():
                        m = pp2d(m, a.minarea, 1)
                    vol[z] = m
                preds[sid] = vol
            for mv in a.minvols:
                ds = []; hs = []
                for sid, vol in preds.items():
                    gt = gts[sid][c]; n = vol.shape[0]
                    v = cc3(vol, mv)
                    inter = (v.astype(bool) & gt).sum(axis=(1, 2))
                    sz = v.sum(axis=(1, 2)) + gt.sum(axis=(1, 2))
                    d = np.where(sz == 0, 1.0, 2.0 * inter / np.maximum(sz, 1))
                    ds.extend(d.tolist())
                    hs.append(hd3(v, gt, np.array([n, S, S], np.float32)))
                dice = float(np.mean(ds)); hd = float(np.mean(hs)); combo = 0.4 * dice + 0.6 * (1 - hd)
                row = (combo, th, mv, dice, hd)
                if best is None or combo > best[0]:
                    best = row
                print(f"class {c} th {th} minvol {mv} -> dice {dice:.4f} hd {hd:.4f} combo {combo:.4f}", flush=True)
        print(f"BEST class {c}: {best}", flush=True)


if __name__ == "__main__":
    main()
