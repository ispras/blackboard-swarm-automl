import numpy as np
from scipy import ndimage

def clean_volume(vol, min_vol=0, keep_top=0, min_slice_area=0):
    """vol: (D,H,W) bool. Remove 3D components < min_vol; optionally keep top-k largest.
    Also removes slice-level components (2D) smaller than min_slice_area."""
    out = vol.astype(np.uint8)
    if min_vol > 0 and out.sum() > 0:
        lab, n = ndimage.label(out, structure=np.ones((3,3,3), np.uint8))
        if n > 0:
            sizes = ndimage.sum(out, lab, range(1, n+1))
            if keep_top > 0:
                keep = set((np.argsort(sizes)[-keep_top:] + 1).tolist())
            else:
                keep = set((np.where(sizes >= min_vol)[0] + 1).tolist())
            mask = np.isin(lab, list(keep)) if keep else np.zeros_like(out, bool)
            out = mask.astype(np.uint8)
    if min_slice_area > 0 and out.sum() > 0:
        from scipy.ndimage import label
        for z in range(out.shape[0]):
            sl = out[z]
            if sl.sum() == 0:
                continue
            l2, n2 = label(sl, structure=np.ones((3,3), np.uint8))
            for k in range(1, n2+1):
                if (l2 == k).sum() < min_slice_area:
                    sl[l2 == k] = 0
            out[z] = sl
    return out.astype(bool)
