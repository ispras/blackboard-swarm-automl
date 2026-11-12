import os
import numpy as np, pandas as pd, cv2, torch
from torch.utils.data import Dataset
import albumentations as A

ROOT = "/download_job_code/payload/sandbox"
CACHE_TR = f"{ROOT}/data/cache_train_320"
CACHE_TE = f"{ROOT}/data/cache_test_320"
CLASSES = ["large_bowel", "small_bowel", "stomach"]
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

def rle_encode(mask):
    p = mask.flatten(order="C")
    p = np.concatenate([[0], p, [0]])
    runs = np.where(p[1:] != p[:-1])[0] + 1
    runs[1::2] -= runs[0::2]
    return " ".join(str(x) for x in runs)

def make_aug(train):
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

def neighbor_arrays(index):
    """prev/next row indices within same case_day (index sorted by id -> contiguous)."""
    n = len(index)
    cd = index['case_day'].values
    prev = np.arange(n); nxt = np.arange(n)
    for i in range(1, n):
        if cd[i] == cd[i-1]:
            prev[i] = i-1
    for i in range(n-2, -1, -1):
        if cd[i] == cd[i+1]:
            nxt[i] = i+1
    return prev, nxt

class GISegDataset(Dataset):
    def __init__(self, index, images, masks, train=False, indices=None, stack=3):
        self.index = index.reset_index(drop=True)
        self.images = images
        self.masks = masks
        self.train = train
        self.aug = make_aug(train)
        self.stack = stack
        self.indices = np.asarray(indices if indices is not None else np.arange(len(index)))
        self.prev, self.nxt = neighbor_arrays(self.index)
        if stack == 5:
            self.prev2 = self.prev[self.prev]
            self.nxt2 = self.nxt[self.nxt]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, k):
        i = int(self.indices[k])
        if self.stack == 3:
            idxs = [self.prev[i], i, self.nxt[i]]
        elif self.stack == 5:
            idxs = [self.prev2[i], self.prev[i], i, self.nxt[i], self.nxt2[i]]
        else:
            idxs = [i]
        img = np.stack([np.asarray(self.images[j]) for j in idxs], -1)  # H,W,C
        m = np.asarray(self.masks[i])  # combined 0..3
        if self.aug is not None:
            out = self.aug(image=img, mask=m[..., None])
            img = out['image']; m = out['mask'][..., 0]
        img = img.astype(np.float32) / 255.0
        # replicate single channel to 3 if stack==1
        if img.shape[2] == 1:
            img = np.repeat(img, 3, 2)
        img = (img - np.resize(MEAN, img.shape[2])) / np.resize(STD, img.shape[2])
        img = np.transpose(img, (2, 0, 1))
        y = np.stack([(m == c + 1).astype(np.float32) for c in range(3)], 0)
        return torch.from_numpy(img), torch.from_numpy(y)

class GITestDataset(Dataset):
    def __init__(self, index, images, stack=3):
        self.index = index.reset_index(drop=True)
        self.images = images
        self.stack = stack
        self.prev, self.nxt = neighbor_arrays(self.index)
        if stack == 5:
            self.prev2 = self.prev[self.prev]; self.nxt2 = self.nxt[self.nxt]
    def __len__(self):
        return len(self.index)
    def __getitem__(self, i):
        if self.stack == 3:
            idxs = [self.prev[i], i, self.nxt[i]]
        elif self.stack == 5:
            idxs = [self.prev2[i], self.prev[i], i, self.nxt[i], self.nxt2[i]]
        else:
            idxs = [i]
        img = np.stack([np.asarray(self.images[j]) for j in idxs], -1).astype(np.float32) / 255.0
        if img.shape[2] == 1:
            img = np.repeat(img, 3, 2)
        img = (img - np.resize(MEAN, img.shape[2])) / np.resize(STD, img.shape[2])
        return torch.from_numpy(np.transpose(img, (2, 0, 1)))
