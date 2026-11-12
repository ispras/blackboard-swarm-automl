import os, json
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
IMG_SIZE = 320


def load_meta(split):
    with open(os.path.join(CACHE, f"{split}_meta.json")) as fh:
        return json.load(fh)


def make_folds(n_splits=5, seed=42):
    meta = load_meta("train")
    cases = sorted({m["case"] for m in meta})
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(cases))
    folds = {}
    for i, ci in enumerate(perm):
        folds[cases[ci]] = i % n_splits
    with open(os.path.join(CACHE, "folds.json"), "w") as fh:
        json.dump(folds, fh, indent=1)
    return folds


def get_folds():
    p = os.path.join(CACHE, "folds.json")
    if os.path.exists(p):
        with open(p) as fh:
            return json.load(fh)
    return make_folds()


def build_samples(split, cases=None, val_cases=None):
    meta = load_meta(split)
    samples = []
    for mi, m in enumerate(meta):
        if cases is not None and m["case"] not in cases:
            continue
        if val_cases is not None and m["case"] in val_cases:
            continue
        for k in range(m["n"]):
            samples.append((mi, k))
    return meta, samples


def augment(img3, msk, rng):
    # img3: HxWx3 uint8, msk: HxWx3 uint8
    if rng.random() < 0.5:
        img3 = img3[:, ::-1]; msk = msk[:, ::-1]
    if rng.random() < 0.5:
        img3 = img3[::-1]; msk = msk[::-1]
    if rng.random() < 0.5:
        k = rng.randint(1, 4)
        img3 = np.rot90(img3, k); msk = np.rot90(msk, k)
    img3 = np.ascontiguousarray(img3); msk = np.ascontiguousarray(msk)
    if rng.random() < 0.5:
        h, w = img3.shape[:2]
        ang = rng.uniform(-15, 15)
        sc = rng.uniform(0.9, 1.15)
        tx, ty = rng.uniform(-0.06, 0.06) * w, rng.uniform(-0.06, 0.06) * h
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, sc)
        M[0, 2] += tx; M[1, 2] += ty
        ni = np.empty_like(img3); nm = np.empty_like(msk)
        for c in range(3):
            ni[:, :, c] = cv2.warpAffine(img3[:, :, c], M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            nm[:, :, c] = cv2.warpAffine(msk[:, :, c], M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        img3, msk = ni, nm
    if rng.random() < 0.3:
        img3 = np.clip(img3.astype(np.int16) + rng.randint(-12, 13), 0, 255).astype(np.uint8)
    if rng.random() < 0.3:
        g = rng.uniform(0.8, 1.25)
        img3 = np.clip((img3.astype(np.float32) ** g), 0, 255).astype(np.uint8)
    return img3, msk


class GISDataset(Dataset):
    def __init__(self, split, samples, meta, train=True, seed=0, max_cache=8):
        self.split = split
        self.samples = samples
        self.meta = meta
        self.train = train
        self.seed = seed
        self.max_cache = max_cache
        self._cache = {}
        self._order = []

    def _load(self, mi):
        if mi in self._cache:
            return self._cache[mi]
        m = self.meta[mi]
        im = np.load(os.path.join(CACHE, m["img"]), mmap_mode="r")
        mk = np.load(os.path.join(CACHE, m["msk"]), mmap_mode="r") if self.split == "train" else None
        self._cache[mi] = (im, mk)
        self._order.append(mi)
        while len(self._order) > self.max_cache:
            old = self._order.pop(0)
            self._cache.pop(old, None)
        return self._cache[mi]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        mi, k = self.samples[i]
        im, mk = self._load(mi)
        n = im.shape[0]
        idx = [max(0, k - 1), k, min(n - 1, k + 1)]
        img3 = np.stack([np.asarray(im[j]) for j in idx], axis=-1)  # HxWx3
        if mk is not None:
            msk = np.asarray(mk[k])
        else:
            msk = np.zeros((IMG_SIZE, IMG_SIZE, 3), np.uint8)
        if self.train:
            rng = np.random.RandomState((self.seed * 1000003 + i) % (2**31))
            img3, msk = augment(img3, msk, rng)
        x = torch.from_numpy(np.ascontiguousarray(img3.transpose(2, 0, 1))).float().div_(255.0)
        y = torch.from_numpy(np.ascontiguousarray(msk.transpose(2, 0, 1))).float()
        return x, y
