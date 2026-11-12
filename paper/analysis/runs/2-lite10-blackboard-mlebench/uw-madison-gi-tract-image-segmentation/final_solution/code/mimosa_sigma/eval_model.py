import os, sys, json, time, argparse
import numpy as np, pandas as pd
from scipy import ndimage
from scipy.spatial import cKDTree
sys.path.insert(0, '/download_job_code/payload/sandbox/workspaces/mimosa_sigma/code')
import simlib
import cv2

ROOT = '/download_job_code/payload/sandbox'
C = f'{ROOT}/workspaces/mimosa_sigma/cache'
CLASSES = simlib.CLASSES
NORM = np.sqrt(2 * 320 * 320)

def build_gt(mask, va, study):
    groups = {}
    for i in va:
        groups.setdefault(study[i], []).append(i)
    gt = {}
    for g, idxs in groups.items():
        idxs = np.array(idxs)
        for c in range(3):
            gt[(g, c)] = np.ascontiguousarray(mask[idxs, c] > 0)
    return gt, {g: np.array(v) for g, v in groups.items()}

def cc2d_vol(pvol, min_area):
    if min_area <= 1:
        return pvol
    out = np.zeros_like(pvol)
    for z in range(pvol.shape[0]):
        n, lab, stats, _ = cv2.connectedComponentsWithStats(pvol[z].astype(np.uint8), connectivity=8)
        for k in range(1, n):
            if stats[k, cv2.CC_STAT_AREA] >= min_area:
                out[z] |= (lab == k)
    return out

def cc3d_vol(pvol, min_vox):
    if min_vox <= 1:
        return pvol
    st = ndimage.generate_binary_structure(3, 2)
    lab, n = ndimage.label(pvol, structure=st)
    if n == 0:
        return pvol
    sizes = np.bincount(lab.ravel()); keep = sizes >= min_vox; keep[0] = False
    return keep[lab]

def dice_vol(pvol, gvol):
    inter = np.logical_and(pvol, gvol).sum(axis=(1, 2))
    ps = pvol.sum(axis=(1, 2)); gs = gvol.sum(axis=(1, 2))
    d = np.where(ps + gs == 0, 1.0, 2.0 * inter / np.maximum(ps + gs, 1))
    return float(d.mean())

def hd_vol(pvol, gvol, gtree):
    p = pvol.any(); g = gvol.any()
    if not p and not g:
        return 0.0
    if not p or not g:
        return 1.0
    if pvol.mean() > 0.6:
        return 1.0
    P = np.argwhere(pvol).astype(np.float32)
    G = np.argwhere(gvol).astype(np.float32)
    if len(P) > 12000:
        P = P[np.random.RandomState(0).choice(len(P), 12000, replace=False)]
    if len(G) > 12000:
        G = G[np.random.RandomState(0).choice(len(G), 12000, replace=False)]
    da = gtree.query(P, k=1)[0].max()
    db = cKDTree(P).query(G, k=1)[0].max()
    return float(min(1.0, max(da, db) / NORM))

def eval_cfg(Pc, va, groups, gt, gtree, c, thr, mode, mval):
    scs = []; dcs = []; hcs = []
    for g in groups:
        rows = np.searchsorted(va, groups[g])
        gvol = gt[(g, c)]
        pvol = Pc[rows] >= thr
        if mode == '2d':
            pvol = cc2d_vol(pvol, mval)
        elif mode == '3d':
            pvol = cc3d_vol(pvol, mval)
        d = dice_vol(pvol, gvol); h = hd_vol(pvol, gvol, gtree.get((g, c)))
        scs.append(0.4 * d + 0.6 * (1 - h)); dcs.append(d); hcs.append(h)
    return float(np.mean(scs)), float(np.mean(dcs)), float(np.mean(hcs))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--probs', required=True)
    ap.add_argument('--fold', type=int, default=0)
    ap.add_argument('--tag', default='model')
    ap.add_argument('--thrs', default='0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70')
    ap.add_argument('--ccs', default='0,20,50,100,200')
    ap.add_argument('--full', action='store_true')
    ap.add_argument('--out', default='')
    args = ap.parse_args()
    t0 = time.time()
    ids = json.load(open(f'{C}/train_ids.json'))
    study = json.load(open(f'{C}/train_study.json'))
    foldarr = np.load(f'{C}/train_fold.npy')
    va = np.where(foldarr == args.fold)[0]
    mask = np.load(f'{C}/train_mask.npy', mmap_mode='r')
    probs = np.load(args.probs, mmap_mode='r')
    assert len(probs) == len(va), (len(probs), len(va))
    gt, groups = build_gt(mask, va, study)
    gtree = {k: cKDTree(np.argwhere(v).astype(np.float32)) for k, v in gt.items() if v.any()}
    thrs = [float(x) for x in args.thrs.split(',')]
    ccs = [int(x) for x in args.ccs.split(',')]
    best = {}
    for c in range(3):
        P = np.asarray(probs[:, c])
        print('loaded class', c, P.shape, f'{time.time()-t0:.0f}s', flush=True)
        res = []
        for thr in thrs:
            sc, d, h = eval_cfg(P, va, groups, gt, gtree, c, thr, None, 0)
            res.append((sc, d, h, thr, 0))
            print(f'  {CLASSES[c]} thr={thr:.2f} cc=0 combo={sc:.4f} dice={d:.4f} hd={h:.4f}', flush=True)
        res.sort(reverse=True)
        bthr = res[0][3]
        for cc in ccs:
            if cc == 0:
                continue
            sc, d, h = eval_cfg(P, va, groups, gt, gtree, c, bthr, '2d', cc)
            res.append((sc, d, h, bthr, cc))
            print(f'  {CLASSES[c]} thr={bthr:.2f} cc2d={cc} combo={sc:.4f} dice={d:.4f} hd={h:.4f}', flush=True)
        res.sort(reverse=True)
        best[c] = res[0]
        del P
        print(f'CLASS {CLASSES[c]} BEST combo={res[0][0]:.4f} dice={res[0][1]:.4f} hd={res[0][2]:.4f} thr={res[0][3]} cc={res[0][4]} [{time.time()-t0:.0f}s]', flush=True)
    mc = np.mean([best[c][0] for c in range(3)]); md = np.mean([best[c][1] for c in range(3)])
    mh = np.mean([best[c][2] for c in range(3)])
    print(f'OVERALL {args.tag} combo={mc:.4f} dice={md:.4f} hd={mh:.4f} thrs={[best[c][3] for c in range(3)]} ccs={[best[c][4] for c in range(3)]}', flush=True)
    out = args.out or f'{ROOT}/workspaces/mimosa_sigma/logs/eval_{args.tag}.json'
    json.dump({'tag': args.tag, 'combo': float(mc), 'dice': float(md), 'hd': float(mh),
               'best': {CLASSES[c]: {'combo': float(best[c][0]), 'dice': float(best[c][1]), 'hd': float(best[c][2]),
                                     'thr': float(best[c][3]), 'cc': int(best[c][4])} for c in range(3)}},
              open(out, 'w'), indent=2)
    if args.full:
        # final 3D CC check at best thresholds
        for c in range(3):
            P = np.asarray(probs[:, c])
            thr = best[c][3]
            for cc3 in [100, 300, 600, 1200]:
                sc, d, h = eval_cfg(P, va, groups, gt, gtree, c, thr, '3d', cc3)
                print(f'  FULL {CLASSES[c]} thr={thr:.2f} cc3d={cc3} combo={sc:.4f} dice={d:.4f} hd={h:.4f}', flush=True)

if __name__ == '__main__':
    main()
