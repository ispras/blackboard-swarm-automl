import os, re, glob
import numpy as np, pandas as pd, cv2, torch
from torch.utils.data import Dataset
import albumentations as A

ROOT = "/download_job_code/payload/sandbox"
WS = f"{ROOT}/workspaces/aldebaran_omicron"
CLASSES = ["large_bowel", "small_bowel", "stomach"]


def rle_decode(rle, h, w, size):
    if not isinstance(rle, str) or rle == "" or rle == "nan":
        return np.zeros((size, size), np.uint8)
    s = rle.split()
    starts = np.asarray(s[0::2], dtype=np.int64) - 1
    lens = np.asarray(s[1::2], dtype=np.int64)
    flat = np.zeros(h * w, np.uint8)
    for st, l in zip(starts, lens):
        flat[st:st + l] = 1
    arr = flat.reshape(h, w, order="C")
    if (h, w) != (size, size):
        arr = cv2.resize(arr, (size, size), interpolation=cv2.INTER_NEAREST)
    return arr


def rle_encode(mask):
    pixels = mask.flatten(order="C")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[0::2]
    return " ".join(str(x) for x in runs)


def build_hw_dict(split):
    hw = {}
    for cd in sorted(os.listdir(f"{ROOT}/data/{split}")):
        for d in sorted(os.listdir(f"{ROOT}/data/{split}/{cd}")):
            for f in os.listdir(f"{ROOT}/data/{split}/{cd}/{d}/scans"):
                m = re.match(r"slice_(\d+)_(\d+)_(\d+)_", f)
                # filename order is slice_W_H_spacing; cv2 shape is (H, W)
                hw[f"{d}_slice_{m.group(1)}"] = (int(m.group(3)), int(m.group(2)))
    return hw


def load_masks(split="train"):
    df = pd.read_csv(f"{ROOT}/data/{split}.csv")
    d = {}
    for r in df.itertuples(index=False):
        d.setdefault(r.id, {})[r._1 if hasattr(r, "_1") else r[1]] = r.segmentation
    return d


def get_mask_map():
    df = pd.read_csv(f"{ROOT}/data/train.csv")
    df = df.pivot(index="id", columns="class", values="segmentation")
    return df


def case_split(cases, val_frac=0.2, seed=42, fold=0):
    cases = sorted(cases)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(cases))
    nval = int(round(len(cases) * val_frac))
    # rotate folds of val
    val_idx = perm[(fold * nval):(fold * nval + nval)] % len(cases)
    val = set(cases[i] for i in val_idx)
    return val


MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def make_aug(size, train):
    if not train:
        return None
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.15, rotate_limit=20,
                           border_mode=cv2.BORDER_CONSTANT, p=0.6),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.3),
    ])


class GISegDataset(Dataset):
    def __init__(self, meta, img_mm, mask_map, hw, size=320, stack=3,
                 train=False, indices=None, mask_mm=None, id2row=None, id2row_cls=None):
        self.meta = meta.reset_index(drop=True)
        self.img = img_mm
        self.mask_map = mask_map
        self.hw = hw
        self.size = size
        self.stack = stack
        self.train = train
        self.aug = make_aug(size, train)
        self.indices = np.asarray(indices if indices is not None else np.arange(len(meta)))
        self.mask_mm = mask_mm
        self.id2row = id2row
        self.gid = self.meta["gid"].values
        self.gstart = {}
        self.gend = {}
        for g, grp in self.meta.groupby("gid"):
            self.gstart[g] = grp.index.min()
            self.gend[g] = grp.index.max()

    def __len__(self):
        return len(self.indices)

    def _mask(self, id_, cls):
        if self.mask_mm is not None:
            return (self.mask_mm[self.id2row[id_]] == (CLASSES.index(cls) + 1)).astype(np.uint8)
        h, w = self.hw[id_]
        mm = self.mask_map
        rle = mm.at[id_, cls] if (id_ in mm.index and cls in mm.columns) else None
        return rle_decode(rle, h, w, self.size)

    def __getitem__(self, k):
        i = int(self.indices[k])
        r = self.meta.iloc[i]
        g = self.gid[i]
        s, e = self.gstart[g], self.gend[g]
        if self.stack == 3:
            imgs = [self.img[max(s, i - 1)], self.img[i], self.img[min(e, i + 1)]]
            img = np.stack(imgs, -1)  # H,W,3
        elif self.stack == 5:
            idxs = [max(s, i - 2), max(s, i - 1), i, min(e, i + 1), min(e, i + 2)]
            img = np.stack([self.img[j] for j in idxs], -1)
        else:
            img = np.repeat(self.img[i][..., None], 3, -1)
        m = np.stack([self._mask(r.id, c) for c in CLASSES], -1)  # H,W,3
        if self.aug is not None:
            out = self.aug(image=img, mask=m)
            img, m = out["image"], out["mask"]
        img = img.astype(np.float32) / 255.0
        img = (img - MEAN) / STD
        img = np.transpose(img, (2, 0, 1))
        m = np.transpose(m.astype(np.float32), (2, 0, 1))
        return torch.from_numpy(img), torch.from_numpy(m)
