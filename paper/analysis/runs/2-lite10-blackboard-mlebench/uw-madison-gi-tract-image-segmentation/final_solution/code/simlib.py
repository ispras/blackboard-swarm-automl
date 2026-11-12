import numpy as np
from scipy.spatial import cKDTree

CLASSES = ['large_bowel', 'small_bowel', 'stomach']

def rle_decode(rle, shape):
    """shape=(H,W); Kaggle RLE is column-major (index = row + col*H)."""
    h, w = int(shape[0]), int(shape[1])
    img = np.zeros(h * w, dtype=np.uint8)
    if isinstance(rle, str) and rle.strip():
        s = np.asarray(rle.split(), dtype=np.int64)
        starts, lengths = s[0::2] - 1, s[1::2]
        for st, ln in zip(starts, lengths):
            img[st:st + ln] = 1
    return img.reshape((h, w))  # row-major (C) - verified empirically

def rle_encode(mask):
    """mask HxW binary -> column-major RLE 'start len ...' (1-indexed)."""
    flat = np.asarray(mask, dtype=np.uint8).reshape(-1)  # row-major (C)
    if flat.max() == 0:
        return ''
    d = np.diff(np.concatenate(([0], flat, [0])))
    starts = np.where(d == 1)[0] + 1
    ends = np.where(d == -1)[0] + 1
    lengths = ends - starts
    return ' '.join(f'{a} {b}' for a, b in zip(starts, lengths))

def dice2d(pred, gt):
    p = pred.astype(bool).ravel(); g = gt.astype(bool).ravel()
    denom = p.sum() + g.sum()
    if denom == 0:
        return 1.0
    return 2.0 * np.logical_and(p, g).sum() / denom

def _pts(mask3d):
    return np.argwhere(mask3d > 0).astype(np.float32)

def hd3d_norm(pred, gt, hw):
    """Directed (symmetric max) Hausdorff over 3D volume, normalized by diagonal sqrt(H^2+W^2)."""
    p = pred > 0; g = gt > 0
    if not p.any() and not g.any():
        return 0.0
    if not p.any() or not g.any():
        return 1.0
    a, b = _pts(p), _pts(g)
    if len(a) > 20000:
        a = a[np.random.RandomState(0).choice(len(a), 20000, replace=False)]
    if len(b) > 20000:
        b = b[np.random.RandomState(0).choice(len(b), 20000, replace=False)]
    da = cKDTree(b).query(a, k=1)[0].max()
    db = cKDTree(a).query(b, k=1)[0].max()
    norm = np.sqrt(hw[0] ** 2 + hw[1] ** 2)
    return min(1.0, max(da, db) / norm)

def combo(dice, hd):
    return 0.4 * dice + 0.6 * (1.0 - hd)
