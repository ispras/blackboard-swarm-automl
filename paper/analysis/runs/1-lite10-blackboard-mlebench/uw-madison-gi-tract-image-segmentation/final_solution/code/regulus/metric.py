import numpy as np
from scipy.ndimage import distance_transform_edt, label as cc_label


def rle_decode(seg, shape):
    h, w = shape
    out = np.zeros(h * w, dtype=np.uint8)
    if isinstance(seg, str) and len(seg) > 0:
        s = seg.split()
        starts = np.asarray(s[0::2], dtype=np.int64) - 1
        lengths = np.asarray(s[1::2], dtype=np.int64)
        for st, ln in zip(starts, lengths):
            out[st:st + ln] = 1
    return out.reshape((h, w), order='C')


def rle_encode(mask):
    """mask: (H,W) binary uint8. Column-major (order='F'). Returns RLE string."""
    flat = np.asarray(mask).reshape(-1, order='C').astype(np.uint8)
    if flat.sum() == 0:
        return ''
    flat = np.concatenate([[0], flat, [0]])
    d = np.diff(flat)
    starts = np.where(d == 1)[0] + 1
    ends = np.where(d == -1)[0] + 1
    lengths = ends - starts
    parts = []
    for s, l in zip(starts, lengths):
        parts.append(f'{s} {l}')
    return ' '.join(parts)


def dice_slice(pred, gt, empty_empty=1.0):
    """pred,gt: bool (H,W)."""
    sp = pred.sum()
    sg = gt.sum()
    if sp == 0 and sg == 0:
        return empty_empty
    if sp == 0 or sg == 0:
        return 0.0
    inter = np.logical_and(pred, gt).sum()
    return 2.0 * inter / (sp + sg)


def directed_hd(a_coords, b_vol, sampling):
    if len(a_coords) == 0:
        return 0.0
    dt = distance_transform_edt(~b_vol, sampling=sampling)
    return float(dt[tuple(a_coords.T)].max())


def hd3d(gt3, pred3, z_norm=True):
    """gt3,pred3: (Z,H,W) bool. Returns normalized bounded distance in [0,1]."""
    Z, H, W = gt3.shape
    a = np.argwhere(gt3)
    b = np.argwhere(pred3)
    if len(a) == 0 and len(b) == 0:
        return 0.0
    if len(a) == 0 or len(b) == 0:
        return 1.0
    # crop to union bbox (translation invariant, distances unchanged)
    lo = np.minimum(a.min(0), b.min(0))
    hi = np.maximum(a.max(0), b.max(0)) + 1
    sl = tuple(slice(int(lo[k]), int(hi[k])) for k in range(3))
    g = gt3[sl]; p = pred3[sl]
    a2 = np.argwhere(g); b2 = np.argwhere(p)
    sz = (1.0 / Z) if z_norm else 1.0
    sampling = (sz, 1.0 / H, 1.0 / W)
    d1 = directed_hd(a2, p, sampling)
    d2 = directed_hd(b2, g, sampling)
    return min(max(d1, d2), 1.0)


def postprocess_slice(mask, min_area=0, keep_largest=False, fill_holes=False):
    """mask: bool (H,W) -> bool."""
    if not mask.any():
        return mask
    if keep_largest or min_area > 0:
        lab, n = cc_label(mask)
        if n > 1:
            sizes = np.bincount(lab.ravel())
            sizes[0] = 0
            if keep_largest:
                keep = sizes.argmax()
                mask = lab == keep
            elif min_area > 0:
                keep_ids = np.where(sizes >= min_area)[0]
                keep_ids = keep_ids[keep_ids > 0]
                mask = np.isin(lab, keep_ids)
    if fill_holes:
        from scipy.ndimage import binary_fill_holes
        mask = binary_fill_holes(mask)
    return mask
