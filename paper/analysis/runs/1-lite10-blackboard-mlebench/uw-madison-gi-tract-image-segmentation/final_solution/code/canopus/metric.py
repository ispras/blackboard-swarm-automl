import numpy as np, pandas as pd
from scipy.spatial import cKDTree

ROOT = "/download_job_code/payload/sandbox"
CACHE_TR = f"{ROOT}/data/cache_train_320"
CLASSES = ["large_bowel", "small_bowel", "stomach"]

def hd_score(a_pts, b_pts, max_pts=2000):
    if len(a_pts) == 0 and len(b_pts) == 0:
        return 1.0
    if len(a_pts) == 0 or len(b_pts) == 0:
        return 0.0
    if len(a_pts) > max_pts:
        a_pts = a_pts[np.linspace(0, len(a_pts)-1, max_pts).astype(int)]
    if len(b_pts) > max_pts:
        b_pts = b_pts[np.linspace(0, len(b_pts)-1, max_pts).astype(int)]
    ta, tb = cKDTree(a_pts), cKDTree(b_pts)
    da, _ = tb.query(a_pts); db, _ = ta.query(b_pts)
    return float(np.clip(1.0 - max(da.max(), db.max()), 0.0, 1.0))

def points_of(vol, normalize=True):
    zz, yy, xx = np.nonzero(vol)
    if len(zz) == 0:
        return np.zeros((0, 3), np.float32)
    d, h, w = vol.shape
    z = zz / max(d - 1, 1); y = yy / max(h - 1, 1); x = xx / max(w - 1, 1)
    return np.stack([z, y, x], 1).astype(np.float32)

def dice_slice_stats(pred, gt, c, empty_empty_one=True):
    """pred (N,H,W) bool, gt (N,H,W) combined masks for class c. Returns per-slice dice."""
    p = pred[:, c] if pred.ndim == 4 else pred
    g = (gt == c + 1)
    inter = np.logical_and(p, g).reshape(len(p), -1).sum(1)
    s = p.reshape(len(p), -1).sum(1) + g.reshape(len(g), -1).sum(1)
    if empty_empty_one:
        return np.where(s > 0, 2 * inter / np.maximum(s, 1), 1.0)
    return np.where(s > 0, 2 * inter / np.maximum(s, 1), 0.0)

def evaluate(probs, index, gt_masks, thr=0.5, empty_empty_one=True):
    """probs: (N,3,H,W) uint8; index rows match; gt_masks (N,H,W) combined.
    Dice = per-2D-slice per-class (Kaggle: 'mean over images'); HD = per case-day 3D per class."""
    pred = np.asarray(probs) > int(thr * 255)
    rows = []
    # dice per slice, aggregated per class
    for c in range(3):
        d = dice_slice_stats(pred, gt_masks, c, empty_empty_one)
        rows.append(dict(scope='dice', cls=CLASSES[c], val=float(d.mean())))
    dicemean = float(np.mean([r['val'] for r in rows if r['scope'] == 'dice']))
    # hd per case-day per class
    hdrows = []
    for cd, g in index[['case_day']].groupby('case_day', sort=False):
        ii = g.index.values
        for c in range(3):
            pv = pred[ii, c].astype(bool)
            gv = (gt_masks[ii] == c + 1)
            hs = hd_score(points_of(pv), points_of(gv))
            hdrows.append(dict(scope='hd', cls=CLASSES[c], val=hs, case_day=cd))
    hdmean = float(np.mean([r['val'] for r in hdrows]))
    combo = 0.4 * dicemean + 0.6 * hdmean
    return rows + hdrows, dict(dice=dicemean, hd=hdmean, combo=combo)
