import numpy as np
from scipy.ndimage import label, binary_erosion, binary_fill_holes
from scipy.spatial.distance import directed_hausdorff

CLASSES = ["large_bowel", "small_bowel", "stomach"]


def rle_decode(rle, shape):
    if not isinstance(rle, str) or len(rle) == 0:
        return np.zeros(shape, np.uint8)
    s = rle.split()
    starts = np.asarray(s[0::2], dtype=np.int64) - 1
    lengths = np.asarray(s[1::2], dtype=np.int64)
    flat = np.zeros(shape[0] * shape[1], np.uint8)
    for st, ln in zip(starts, lengths):
        flat[st:st + ln] = 1
    return flat.reshape(shape, order="C")


def rle_encode(mask):
    pixels = mask.reshape(-1)
    if pixels.max() == 0:
        return ""
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[0::2]
    return " ".join(str(x) for x in runs)


def dice(pred, gt):
    pred = pred.astype(bool); gt = gt.astype(bool)
    s = pred.sum() + gt.sum()
    if s == 0:
        return 1.0  # convention: both empty -> perfect
    return 2.0 * (pred & gt).sum() / s


def hausdorff_norm(pred, gt, diag=None):
    pred = pred.astype(bool); gt = gt.astype(bool)
    if diag is None:
        h, w = gt.shape
        diag = float(np.sqrt(h * h + w * w))
    pe, ge = pred.any(), gt.any()
    if not pe and not ge:
        return 0.0
    if not pe or not ge:
        return 1.0
    p = np.argwhere(pred); g = np.argwhere(gt)
    d1 = directed_hausdorff(p, g)[0]
    d2 = directed_hausdorff(g, p)[0]
    return min(1.0, max(d1, d2) / diag)


def combo_score(pred, gt, diag=None):
    d = dice(pred, gt)
    h = hausdorff_norm(pred, gt, diag)
    return 0.4 * d + 0.6 * (1.0 - h), d, h


def largest_cc(mask, keep=1):
    if not mask.any():
        return mask
    lab, n = label(mask)
    if n <= keep:
        return mask
    sizes = np.bincount(lab.ravel()); sizes[0] = 0
    keep_ids = np.argsort(sizes)[-keep:]
    return np.isin(lab, keep_ids)


def remove_small(mask, min_size):
    if not mask.any():
        return mask
    lab, n = label(mask)
    sizes = np.bincount(lab.ravel())
    keep = np.where(sizes >= min_size)[0]; keep = keep[keep != 0]
    return np.isin(lab, keep)


def postprocess_bin(mask, min_size=0, largest=False, fill=True):
    m = mask.astype(bool)
    if fill:
        m = binary_fill_holes(m)
    if largest:
        m = largest_cc(m, keep=1)
    if min_size > 0:
        m = remove_small(m, min_size)
    return m.astype(np.uint8)
