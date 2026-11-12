import os, re, sys, glob
import numpy as np
import pandas as pd
import cv2
from concurrent.futures import ProcessPoolExecutor

ROOT = "/download_job_code/payload/sandbox"
OUT = f"{ROOT}/workspaces/capella_zeta/cache"
SIZE = 320
CLASSES = ["large_bowel", "small_bowel", "stomach"]


def norm_resize(args):
    path, size = args
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 3:
        img = img[..., 0]
    lo = np.percentile(img, 1.0)
    hi = np.percentile(img, 99.5)
    if hi <= lo:
        hi = lo + 1
    img = np.clip(img, lo, hi)
    img = ((img - lo) / (hi - lo) * 255.0).astype(np.uint8)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img


def rle_decode(rle, H, W):
    flat = np.zeros(H * W, np.uint8)
    if isinstance(rle, str) and len(rle) > 0:
        s = rle.split()
        st = np.asarray(s[0::2], dtype=np.int64) - 1
        ln = np.asarray(s[1::2], dtype=np.int64)
        for a, b in zip(st, ln):
            flat[a:a + b] = 1
    return flat.reshape((H, W), order="C")


def build(split):
    base = f"{ROOT}/data/{split}"
    if split == "train":
        lab = pd.read_csv(f"{ROOT}/data/train.csv")
    else:
        lab = pd.read_csv(f"{ROOT}/data/test.csv")
        lab["segmentation"] = None

    files = glob.glob(f"{base}/*/*/scans/*.png")
    recs = []
    for f in files:
        name = os.path.basename(f)
        m = re.match(r"slice_(\d+)_(\d+)_(\d+)_([0-9.]+)_([0-9.]+)\.png", name)
        if not m:
            continue
        sl = int(m.group(1)); W = int(m.group(2)); H = int(m.group(3))
        case = f.split("/")[-4]
        cd = f.split("/")[-3]
        rid = f"{cd}_slice_{sl:04d}"
        recs.append((rid, f, case, cd, sl, H, W))
    idx = pd.DataFrame(recs, columns=["id", "path", "case", "case_day", "slice", "H", "W"])
    idx = idx.sort_values(["case_day", "slice"]).reset_index(drop=True)

    # group labels by id -> {class: rle}
    lab_map = {}
    for rid, grp in lab.groupby("id"):
        lab_map[rid] = dict(zip(grp["class"], grp["segmentation"]))

    N = len(idx)
    imgs = np.lib.format.open_memmap(f"{OUT}/{split}_img{SIZE}.npy", mode="w+",
                                     dtype=np.uint8, shape=(N, SIZE, SIZE))
    masks = np.lib.format.open_memmap(f"{OUT}/{split}_mask{SIZE}.npy", mode="w+",
                                      dtype=np.uint8, shape=(N, 3, SIZE, SIZE)) if split == "train" else None

    with ProcessPoolExecutor(max_workers=4) as ex:
        imgs_list = list(ex.map(norm_resize, [(p, SIZE) for p in idx["path"]], chunksize=64))
    for i, im in enumerate(imgs_list):
        imgs[i] = im
    imgs.flush()

    if split == "train":
        for i, row in idx.iterrows():
            rid = row["id"]; H = int(row["H"]); W = int(row["W"])
            d = lab_map.get(rid, {})
            for c, cls in enumerate(CLASSES):
                m = rle_decode(d.get(cls), H, W)
                m = cv2.resize(m, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
                masks[i, c] = m
        masks.flush()

    # presence flags
    for cls in CLASSES:
        idx[f"has_{cls}"] = [1 if lab_map.get(r, {}).get(cls) not in (None, "") and
                             isinstance(lab_map.get(r, {}).get(cls), str) else 0 for r in idx["id"]]
    idx.to_csv(f"{OUT}/{split}_index.csv", index=False)
    print(split, "done", N, flush=True)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    build("train")
    build("test")
