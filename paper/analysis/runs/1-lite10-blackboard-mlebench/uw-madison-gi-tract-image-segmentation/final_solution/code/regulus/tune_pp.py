import os, sys, json, argparse, time
import numpy as np, pandas as pd
from scipy.ndimage import label, binary_fill_holes, binary_dilation
from scipy.spatial import cKDTree

CANOPUS = '/download_job_code/payload/sandbox/workspaces/canopus_phi'
sys.path.insert(0, CANOPUS)
from metric import points_of  # noqa

ROOT = '/download_job_code/payload/sandbox'
CACHE = f'{ROOT}/data/cache_train_320'
CLASSES = ['large_bowel', 'small_bowel', 'stomach']
MAXPTS = 1000


def hd_from_pts(a_pts, b_pts):
    if len(a_pts) == 0 and len(b_pts) == 0:
        return 1.0
    if len(a_pts) == 0 or len(b_pts) == 0:
        return 0.0
    if len(a_pts) > MAXPTS:
        a_pts = a_pts[np.linspace(0, len(a_pts) - 1, MAXPTS).astype(int)]
    if len(b_pts) > MAXPTS:
        b_pts = b_pts[np.linspace(0, len(b_pts) - 1, MAXPTS).astype(int)]
    ta, tb = cKDTree(a_pts), cKDTree(b_pts)
    da, _ = tb.query(a_pts); db, _ = ta.query(b_pts)
    return float(np.clip(1.0 - max(da.max(), db.max()), 0.0, 1.0))


def filter3d(p, min_vol=0, min_slice_area=0):
    if min_slice_area > 0:
        for z in range(p.shape[0]):
            s = p[z]
            if s.any():
                lab, n = label(s)
                if n > 1:
                    sz = np.bincount(lab.ravel()); sz[0] = 0
                    keep = np.where(sz >= min_slice_area)[0]; keep = keep[keep > 0]
                    p[z] = np.isin(lab, keep)
    if min_vol > 0:
        lab, n = label(p)
        if n > 1:
            sz = np.bincount(lab.ravel()); sz[0] = 0
            keep = np.where(sz >= min_vol)[0]; keep = keep[keep > 0]
            p = np.isin(lab, keep)
    return p


def score_pred(pv, gt_flat, groups, gts_pts):
    inter = np.logical_and(pv, gt_flat).reshape(len(pv), -1).sum(1)
    s = pv.reshape(len(pv), -1).sum(1) + gt_flat.reshape(len(pv), -1).sum(1)
    d = float(np.where(s > 0, 2 * inter / np.maximum(s, 1), 1.0).mean())
    hds = []
    for ii, gp in zip(groups, gts_pts):
        hds.append(hd_from_pts(gp, points_of(pv[ii])))
    h = float(np.mean(hds))
    return d, h, 0.4 * d + 0.6 * h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--probs', required=True)
    ap.add_argument('--ids', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--n_groups_hd', type=int, default=24)
    a = ap.parse_args()
    from mycache import load_val
    vi, gt, groups_all = load_val(a.ids)
    P = np.asarray(np.load(a.probs, mmap_mode='r'))
    assert P.shape[0] == len(vi), (P.shape, len(vi))
    if a.n_groups_hd and len(groups_all) > a.n_groups_hd:
        sel = np.linspace(0, len(groups_all) - 1, a.n_groups_hd).astype(int)
        groups = [groups_all[i] for i in sel]
    else:
        groups = groups_all
    print('val slices', len(vi), 'hd case_days', len(groups), 'of', len(groups_all), flush=True)

    thrs = [0.30, 0.35, 0.40, 0.45, 0.50, 0.60]
    dils = [0, 1, 2]
    result = {}
    t0 = time.time()
    for c in range(3):
        Pc = P[:, c]
        gt_flat = (gt == c + 1)
        gts_pts = [points_of(gt_flat[ii]) for ii in groups]
        best = None
        base = {t: (Pc > int(t * 255)) for t in thrs}
        for thr in thrs:
            for dil in dils:
                pv = base[thr]
                if dil > 0:
                    pv = pv.copy()
                    for z in range(pv.shape[0]):
                        if pv[z].any():
                            pv[z] = binary_dilation(pv[z], iterations=dil)
                r = score_pred(pv, gt_flat, groups, gts_pts)
                if best is None or r[2] > best['combo']:
                    best = dict(thr=thr, dil=dil, min_vol=0, msa=0, fill=0, dice=r[0], hd=r[1], combo=r[2])
        print(f'cls{c} stage1 {best} ({time.time()-t0:.0f}s)', flush=True)
        # stage2
        best2 = dict(best)
        pv0 = base[best['thr']]
        for mv, msa in [(0, 0), (300, 0), (1000, 0), (0, 40), (300, 40)]:
            pv = pv0
            if best['dil'] > 0:
                pv = pv.copy()
                for z in range(pv.shape[0]):
                    if pv[z].any():
                        pv[z] = binary_dilation(pv[z], iterations=best['dil'])
            for ii in groups:
                if pv[ii].any():
                    pv[ii] = filter3d(pv[ii], mv, msa)
            r = score_pred(pv, gt_flat, groups, gts_pts)
            if r[2] > best2['combo']:
                best2 = dict(thr=best['thr'], dil=best['dil'], min_vol=mv, msa=msa, fill=0,
                             dice=r[0], hd=r[1], combo=r[2])
        print(f'cls{c} stage2 {best2} ({time.time()-t0:.0f}s)', flush=True)
        result[CLASSES[c]] = best2
    overall = dict(
        dice=float(np.mean([v['dice'] for v in result.values()])),
        hd=float(np.mean([v['hd'] for v in result.values()])),
        combo=float(np.mean([v['combo'] for v in result.values()])),
    )
    out = dict(per_class=result, overall=overall)
    print('FINAL', json.dumps(out), flush=True)
    json.dump(out, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
