"""Approximate scorer for dice-hausdorff-combo (per case_day, per class, 3D).

No official grader is available/allowed; this is our local proxy built from TASK.md.
Reports dice with empty-empty=1 (standard) and dice with empty-empty=0 (literal TASK text),
plus normalized 3D Hausdorff. combo = 0.4*dice + 0.6*(1-hd_norm).
"""
import numpy as np
from scipy.spatial import cKDTree

CLASSES = ["large_bowel", "small_bowel", "stomach"]


def dice3d(pred, gt, empty_val=1.0):
    pred = pred.astype(bool); gt = gt.astype(bool)
    s = pred.sum() + gt.sum()
    if s == 0:
        return empty_val
    return 2.0 * np.logical_and(pred, gt).sum() / s


def _surface(vol):
    # boundary voxels: voxel set minus eroded set (6-connectivity)
    from scipy.ndimage import binary_erosion
    st = np.zeros((3, 3, 3), bool)
    st[1, 1, :] = st[1, :, 1] = st[:, 1, 1] = True
    er = binary_erosion(vol, structure=st, border_value=0)
    surf = vol & ~er
    return surf


def hausdorff3d(pred, gt, spacing=(1.0, 1.0, 1.0), max_points=20000, seed=0):
    pred = pred.astype(bool); gt = gt.astype(bool)
    if not pred.any() and not gt.any():
        return 0.0
    if not pred.any() or not gt.any():
        return np.inf
    u = pred | gt
    coords = np.argwhere(u)
    lo = coords.min(0); hi = coords.max(0) + 1
    pred = pred[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    gt = gt[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    p = np.argwhere(_surface(pred)).astype(np.float32)
    g = np.argwhere(_surface(gt)).astype(np.float32)
    if len(p) == 0:
        p = np.argwhere(pred).astype(np.float32)
    if len(g) == 0:
        g = np.argwhere(gt).astype(np.float32)
    p *= np.asarray(spacing, np.float32)
    g *= np.asarray(spacing, np.float32)
    rng = np.random.RandomState(seed)
    if len(p) > max_points:
        p = p[rng.choice(len(p), max_points, replace=False)]
    if len(g) > max_points:
        g = g[rng.choice(len(g), max_points, replace=False)]
    tp = cKDTree(g); pg = cKDTree(p)
    d1, _ = tp.query(p, k=1)
    d2, _ = pg.query(g, k=1)
    return float(max(d1.max(), d2.max()))


def combo_scores(pred_vols, gt_vols, sizes):
    """pred_vols, gt_vols: dict[(case_day,cls)] -> 3D bool arrays.
    sizes: dict[case_day] -> (H,W) after resize (H=W=size). Returns dict of aggregates."""
    rows = []
    for key in gt_vols:
        cd, cls = key
        pred = pred_vols.get(key, np.zeros_like(gt_vols[key]))
        gt = gt_vols[key]
        H, W, Z = gt.shape
        d1 = dice3d(pred, gt, 1.0)
        d0 = dice3d(pred, gt, 0.0)
        if pred.any() and gt.any():
            hd = hausdorff3d(pred, gt, spacing=(1.0, 1.0, 1.0))
            hd_n = min(1.0, hd / max(H, W))
        elif (not pred.any()) and (not gt.any()):
            hd_n = 0.0
        else:
            hd_n = 1.0
        rows.append(dict(case_day=cd, cls=cls, dice1=d1, dice0=d0, hd_n=hd_n,
                         combo=0.4 * d1 + 0.6 * (1 - hd_n),
                         combo0=0.4 * d0 + 0.6 * (1 - hd_n),
                         gt_pos=bool(gt.any()), pred_pos=bool(pred.any())))
    if not rows:
        return {}
    import statistics
    out = {}
    for k in ["dice1", "dice0", "hd_n", "combo", "combo0"]:
        out[k] = float(np.mean([r[k] for r in rows]))
    out["n"] = len(rows)
    out["n_gt_pos"] = int(sum(r["gt_pos"] for r in rows))
    return out
