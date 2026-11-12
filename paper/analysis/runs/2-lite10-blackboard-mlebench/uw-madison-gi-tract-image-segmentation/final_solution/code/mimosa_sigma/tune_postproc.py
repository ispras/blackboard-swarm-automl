import os, sys, json, time, argparse, hashlib
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
import cv2
sys.path.insert(0, '/download_job_code/payload/sandbox/workspaces/mimosa_sigma/code')
import simlib
ROOT = '/download_job_code/payload/sandbox'
C = f'{ROOT}/workspaces/mimosa_sigma/cache'
CLASSES = simlib.CLASSES
NORM = np.sqrt(2 * 320 * 320)

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

def lab3d(pvol):
    st = ndimage.generate_binary_structure(3, 2)
    return ndimage.label(pvol, structure=st)

def postproc(pvol, thr, cc2d, cc3d, keep_top, fill=0):
    b = pvol >= thr
    if cc2d > 1:
        b = cc2d_vol(b, cc2d)
    if cc3d > 1 or keep_top > 0:
        lab, n = lab3d(b)
        if n == 0:
            return b
        sizes = np.bincount(lab.ravel())
        if keep_top > 0:
            order = np.argsort(sizes[1:])[::-1][:keep_top] + 1
            keep = np.zeros(n + 1, bool); keep[order] = True
        else:
            keep = sizes >= cc3d; keep[0] = False
        b = keep[lab]
    if fill:
        for z in range(b.shape[0]):
            b[z] = ndimage.binary_fill_holes(b[z])
    return b

def decide(pvol, gvol, gtree):
    inter = np.logical_and(pvol, gvol).sum(axis=(1, 2)); ps = pvol.sum(axis=(1, 2)); gs = gvol.sum(axis=(1, 2))
    dd = float(np.where(ps + gs == 0, 1.0, 2.0 * inter / np.maximum(ps + gs, 1)).mean())
    if not pvol.any() and not gvol.any():
        hh = 0.0
    elif not pvol.any() or not gvol.any() or pvol.mean() > 0.6:
        hh = 1.0
    else:
        P = np.argwhere(pvol).astype(np.float32); G = np.argwhere(gvol).astype(np.float32)
        if len(P) > 12000:
            P = P[np.random.RandomState(0).choice(len(P), 12000, replace=False)]
        if len(G) > 12000:
            G = G[np.random.RandomState(0).choice(len(G), 12000, replace=False)]
        hh = float(min(1.0, max(gtree.query(P, k=1)[0].max(), cKDTree(P).query(G, k=1)[0].max()) / NORM))
    return 0.4 * dd + 0.6 * (1 - hh), dd, hh

def eval_cfg(Pc, va, groups, gt, gtree, c, thr, cc2d, cc3d, keep_top, glist):
    scs = []; dcs = []; hcs = []
    for g in glist:
        rows = np.searchsorted(va, groups[g])
        b = postproc(Pc[rows], thr, cc2d, cc3d, keep_top)
        sc, d, h = decide(b, gt[(g, c)], gtree.get((g, c)))
        scs.append(sc); dcs.append(d); hcs.append(h)
    return float(np.mean(scs)), float(np.mean(dcs)), float(np.mean(hcs))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--probs', nargs='+', required=True)
    ap.add_argument('--weights', nargs='+', type=float, default=None)
    ap.add_argument('--fold', type=int, default=0)
    ap.add_argument('--tag', default='ens')
    ap.add_argument('--out', default='')
    args = ap.parse_args()
    t0 = time.time()
    study = json.load(open(f'{C}/train_study.json'))
    foldarr = np.load(f'{C}/train_fold.npy')
    va = np.where(foldarr == args.fold)[0]
    mask = np.load(f'{C}/train_mask.npy', mmap_mode='r')
    groups = {}
    for i in va:
        groups.setdefault(study[i], []).append(i)
    groups = {g: np.array(v) for g, v in groups.items()}
    gnames = sorted(groups)
    # deterministic tune/holdout split by stable hash of group name
    def hv(g):
        return int(hashlib.md5(g.encode()).hexdigest(), 16) % 2
    tune = [g for g in gnames if hv(g) == 0]
    hold = [g for g in gnames if hv(g) == 1]
    gt = {}
    gtree = {}
    for g in gnames:
        idxs = groups[g]
        for c in range(3):
            v = np.ascontiguousarray(mask[idxs, c] > 0)
            gt[(g, c)] = v
            if v.any():
                gtree[(g, c)] = cKDTree(np.argwhere(v).astype(np.float32))
    print(f'tune {len(tune)} hold {len(hold)} groups; loaded gt {time.time()-t0:.0f}s', flush=True)
    probs = [np.load(p, mmap_mode='r') for p in args.probs]
    w = np.array(args.weights or [1.0] * len(probs), np.float32); w /= w.sum()
    for k, p in enumerate(probs):
        assert len(p) == len(va), (p, len(p), len(va))
    thrs = [round(x, 2) for x in np.arange(0.55, 0.96, 0.05)]
    cc2ds = [0, 100, 300, 600]
    cc3ds = [0, 800, 1500, 2500, 4000]
    keepts = [0, 1]
    recipe = {}
    for c in range(3):
        Pc = np.zeros((len(va), 320, 320), np.float32)
        for k, p in enumerate(probs):
            Pc += w[k] * np.asarray(p[:, c], dtype=np.float32)
        print(f'class {CLASSES[c]} probs ready {time.time()-t0:.0f}s', flush=True)
        cands = []
        # stage 1: thr x cc2d
        for thr in thrs:
            for c2 in cc2ds:
                sc, d, h = eval_cfg(Pc, va, groups, gt, gtree, c, thr, c2, 0, 0, tune)
                cands.append((sc, thr, c2, 0, 0))
        cands.sort(reverse=True); b1 = cands[0]
        print('  s1', tuple(round(x, 4) if isinstance(x, float) else x for x in b1), flush=True)
        # stage 2: cc3d / keeptop at best
        cands2 = []
        for c3 in cc3ds:
            for kt in keepts:
                sc, d, h = eval_cfg(Pc, va, groups, gt, gtree, c, b1[1], b1[2], c3, kt, tune)
                cands2.append((sc, b1[1], b1[2], c3, kt))
        cands2.sort(reverse=True); b2 = cands2[0]
        print('  s2', tuple(round(x, 4) if isinstance(x, float) else x for x in b2), flush=True)
        # stage 3: refine thr
        cands3 = []
        for thr in [max(0.3, b2[1] - 0.05), b2[1], min(0.9, b2[1] + 0.05)]:
            sc, d, h = eval_cfg(Pc, va, groups, gt, gtree, c, thr, b2[2], b2[3], b2[4], tune)
            cands3.append((sc, thr, b2[2], b2[3], b2[4]))
        cands3.sort(reverse=True); b3 = cands3[0]
        holdsc, holdd, holdh = eval_cfg(Pc, va, groups, gt, gtree, c, b3[1], b3[2], b3[3], b3[4], hold)
        tunsc, tund, tunh = eval_cfg(Pc, va, groups, gt, gtree, c, b3[1], b3[2], b3[3], b3[4], tune)
        print(f'CLASS {CLASSES[c]} tune={tunsc:.4f} hold={holdsc:.4f} dice_h={holdd:.4f} hd_h={holdh:.4f} thr={b3[1]} cc2d={b3[2]} cc3d={b3[3]} keep={b3[4]} [{time.time()-t0:.0f}s]', flush=True)
        recipe[CLASSES[c]] = {'thr': float(b3[1]), 'cc2d': int(b3[2]), 'cc3d': int(b3[3]),
                              'keep_top': int(b3[4]), 'tune_combo': tunsc, 'hold_combo': holdsc,
                              'hold_dice': holdd, 'hold_hd': holdh}
        del Pc
    overall = float(np.mean([recipe[cc]['hold_combo'] for cc in CLASSES]))
    print(f'OVERALL HOLDOUT combo={overall:.4f}', flush=True)
    out = args.out or f'{ROOT}/workspaces/mimosa_sigma/logs/recipe_{args.tag}.json'
    json.dump({'tag': args.tag, 'probs': args.probs, 'weights': w.tolist(),
               'overall_holdout_combo': overall, 'classes': recipe}, open(out, 'w'), indent=2)
    print('saved', out)

if __name__ == '__main__':
    main()
