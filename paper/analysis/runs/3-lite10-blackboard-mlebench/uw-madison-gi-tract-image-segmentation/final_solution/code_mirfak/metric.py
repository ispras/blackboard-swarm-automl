import numpy as np
from scipy import ndimage
from scipy.spatial.distance import directed_hausdorff


def dice_bin(pred, gt):
    pred = pred.astype(bool); gt = gt.astype(bool)
    s = pred.sum() + gt.sum()
    if s == 0:
        return 1.0
    return 2.0 * (pred & gt).sum() / s


def _surf_points(vol, step=1):
    z, y, x = np.nonzero(vol)
    if len(z) == 0:
        return np.zeros((0, 3))
    # keep voxels on boundary (erode)
    er = ndimage.binary_erosion(vol, iterations=1)
    z2, y2, x2 = np.nonzero(vol & ~er)
    if len(z2) == 0:
        z2, y2, x2 = z, y, x
    pts = np.stack([z2, y2, x2], axis=1).astype(np.float32)
    if step > 1 and len(pts) > step:
        pts = pts[::step]
    return pts


def hausdorff_norm(pred_vol, gt_vol):
    """Bounded [0,1] normalized 3D hausdorff (z,y,x normalized by volume dims).
    both empty -> 0 (best). one empty -> 1 (worst)."""
    ps = pred_vol.any(); gs = gt_vol.any()
    if not ps and not gs:
        return 0.0
    if ps != gs:
        return 1.0
    Z, Y, X = gt_vol.shape
    a = _surf_points(pred_vol); b = _surf_points(gt_vol)
    a = a / np.array([Z, Y, X], np.float32)
    b = b / np.array([Z, Y, X], np.float32)
    d = max(directed_hausdorff(a, b)[0], directed_hausdorff(b, a)[0])
    return float(min(d / np.sqrt(3.0), 1.0))


def scan_combo(pred_vols, gt_vols):
    """pred_vols/gt_vols: [3, N, H, W] binary for one scan. Returns (dice, hd, combo)."""
    dices = []; hds = []
    for c in range(3):
        d = dice_bin(pred_vols[c], gt_vols[c])
        h = hausdorff_norm(pred_vols[c], gt_vols[c])
        dices.append(d); hds.append(h)
    dice = float(np.mean(dices)); hd = float(np.mean(hds))
    return dice, hd, 0.4 * dice + 0.6 * (1 - hd)


def hausdorff_norm_fast(pred, gt):
    """Fast symmetric 3D hausdorff via EDT. Bounded [0,1]. both empty->0, one empty->1."""
    p = pred.any(); g = gt.any()
    if not p and not g:
        return 0.0
    if p != g:
        return 1.0
    pb = pred.astype(bool); gb = gt.astype(bool)
    d_to_g = ndimage.distance_transform_edt(~gb)
    d_to_p = ndimage.distance_transform_edt(~pb)
    a = d_to_g[pb].max()
    b = d_to_p[gb].max()
    d = max(a, b)
    Z, Y, X = gt.shape
    return float(min(d / np.sqrt(Z * Z + Y * Y + X * X), 1.0))
