"""Shared utilities for UW-Madison GI segmentation. RLE = ROW-major order='C' on true (H,W) [verified]."""
import os, re, glob, json
import numpy as np
import pandas as pd
import cv2

DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
CLASSES = ["large_bowel", "small_bowel", "stomach"]
IMG_SIZE = 320


def rle_decode(rle, h, w):
    if not isinstance(rle, str) or rle.strip() == "":
        return np.zeros((h, w), np.uint8)
    s = rle.split()
    m = np.zeros(h * w, np.uint8)
    for i in range(0, len(s), 2):
        st = int(s[i]); ln = int(s[i + 1])
        m[st - 1:st - 1 + ln] = 1
    return m.reshape((h, w), order="C")


def rle_encode(mask):
    """mask: 2D binary. Returns space separated start length pairs (column-major)."""
    flat = np.asarray(mask, np.uint8).reshape(-1, order="C")
    if flat.sum() == 0:
        return ""
    # pad to find transitions
    padded = np.concatenate([[0], flat, [0]])
    diff = np.diff(padded)
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1
    lengths = ends - starts
    return " ".join(f"{s} {l}" for s, l in zip(starts, lengths))


def parse_id(i):
    c, d, sl, sv = i.split("_")
    return c, d, sl, sv  # case, day, slice, slice_name_suffix


def build_scan_index(split):
    """Return list of dicts describing each (case,day) scan with sorted slices."""
    root = os.path.join(DATA, split)
    scans = []
    for cdir in sorted(os.listdir(root)):
        cpath = os.path.join(root, cdir)
        if not os.path.isdir(cpath):
            continue
        for ddir in sorted(os.listdir(cpath)):
            ddpath = os.path.join(cpath, ddir)
            spath = os.path.join(ddpath, "scans")
            if not os.path.isdir(spath):
                continue
            files = glob.glob(os.path.join(spath, "*.png"))
            files = sorted(files, key=lambda p: int(re.search(r"slice_(\d+)_", p).group(1)))
            # filename = slice_<num>_<W>_<H>_<spacingW>_<spacingH>.png ; trust actual cv2 shape
            m = re.search(r"slice_(\d+)_(\d+)_(\d+)_([\d.]+)_([\d.]+)\.png$", files[0])
            h, w = cv2.imread(files[0], cv2.IMREAD_UNCHANGED).shape[:2]
            sw, sh = float(m.group(4)), float(m.group(5))
            scans.append({
                "scan_id": ddir, "case": cdir, "day": ddir,
                "files": files, "n": len(files), "H": h, "W": w,
                "sx": sw, "sy": sh,
            })
    return scans


def _norm_img(img):
    img = img.astype(np.float32)
    lo, hi = np.percentile(img, 1), np.percentile(img, 99.5)
    if hi <= lo:
        lo, hi = img.min(), img.max()
    img = (img - lo) / (hi - lo + 1e-6)
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)


def build_cache(split):
    """Precompute resized uint8 images + masks memmaps per scan, plus meta json."""
    os.makedirs(CACHE, exist_ok=True)
    scans = build_scan_index(split)
    # load labels for masks
    if split == "train":
        df = pd.read_csv(os.path.join(DATA, "train.csv"))
        lab = {}
        for i, cl, seg in zip(df["id"], df["class"], df["segmentation"]):
            lab.setdefault(i, {})[cl] = seg
    else:
        lab = {}
    meta = []
    for si, s in enumerate(scans):
        sid = s["scan_id"]
        H, W, N = s["H"], s["W"], s["n"]
        img_arr = os.path.join(CACHE, f"{split}_{sid}_img.npy")
        msk_arr = os.path.join(CACHE, f"{split}_{sid}_msk.npy")
        meta.append({"scan_id": sid, "case": s["case"], "day": s["day"],
                     "n": N, "H": H, "W": W, "sx": s["sx"], "sy": s["sy"],
                     "img": os.path.basename(img_arr), "msk": os.path.basename(msk_arr),
                     "files": s["files"]})
        if os.path.exists(img_arr) and os.path.exists(msk_arr):
            continue
        imgs = np.zeros((N, IMG_SIZE, IMG_SIZE), np.uint8)
        msks = np.zeros((N, IMG_SIZE, IMG_SIZE, 3), np.uint8)
        for k, f in enumerate(s["files"]):
            im = cv2.imread(f, cv2.IMREAD_UNCHANGED)
            imgs[k] = cv2.resize(_norm_img(im), (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
            if split == "train":
                base = os.path.basename(f)[:-4]  # drop .png -> slice_0001_...
                parts = base.split("_", 2)  # ['slice', '0001', '266_266_1.50_1.50']
                obj_id = f"{s['day']}_{parts[0]}_{parts[1]}"
                labs = lab.get(obj_id, {})
                for ci, cl in enumerate(CLASSES):
                    rle = labs.get(cl)
                    if isinstance(rle, str) and rle.strip():
                        m = rle_decode(rle, H, W)
                        msks[k, :, :, ci] = cv2.resize(m, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
        np.save(img_arr, imgs)
        np.save(msk_arr, msks)
        if si % 20 == 0:
            print(f"{split} {si}/{len(scans)} {sid}", flush=True)
    with open(os.path.join(CACHE, f"{split}_meta.json"), "w") as fh:
        json.dump(meta, fh)
    print(f"done {split}: {len(meta)} scans")
    return meta


def load_meta(split):
    with open(os.path.join(CACHE, f"{split}_meta.json")) as fh:
        return json.load(fh)


if __name__ == "__main__":
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("train", "both"):
        build_cache("train")
    if which in ("test", "both"):
        build_cache("test")
