import os, sys, json, time, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import cv2
import segmentation_models_pytorch as smp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dataset as D
from common import CLASSES

CACHE = D.CACHE
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--split", default="test", choices=["test", "train"])
    p.add_argument("--val_cases", default="", help="comma list; if set, restrict to these cases (split=train)")
    p.add_argument("--out", required=True, help="output prefix (writes <out>_probs.npy, <out>_meta.npz)")
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--tta", type=int, default=1)
    p.add_argument("--fp16", type=int, default=1)
    return p.parse_args()


class InferDS(Dataset):
    def __init__(self, split, samples, meta):
        self.split = split; self.samples = samples; self.meta = meta
        self._c = {}

    def _load(self, mi):
        if mi not in self._c:
            m = self.meta[mi]
            self._c = {mi: np.load(os.path.join(CACHE, m["img"]), mmap_mode="r")}
        return self._c[mi]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        mi, k = self.samples[i]
        im = self._load(mi)
        n = im.shape[0]
        idx = [max(0, k - 1), k, min(n - 1, k + 1)]
        arr = np.stack([np.asarray(im[j]) for j in idx], axis=-1)
        x = torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1))).float().div_(255.0)
        return x, i


def main():
    a = parse()
    ckpt = torch.load(a.ckpt, map_location="cpu")
    cargs = ckpt["args"]
    model = smp.create_model(cargs["arch"], encoder_name=cargs["encoder"], encoder_weights=None,
                             in_channels=3, classes=3)
    model.load_state_dict(ckpt["model"])
    device = "cuda"
    model = model.to(device).eval()

    meta = D.load_meta(a.split)
    if a.split == "train" and a.val_cases:
        vc = set(a.val_cases.split(","))
        samples = [(mi, k) for mi, m in enumerate(meta) if m["case"] in vc for k in range(m["n"])]
    else:
        samples = [(mi, k) for mi, m in enumerate(meta) for k in range(m["n"])]
    ds = InferDS(a.split, samples, meta)
    dl = DataLoader(ds, batch_size=a.bs, shuffle=False, num_workers=a.workers, pin_memory=True)

    N = len(samples)
    probs = np.zeros((N, 3, D.IMG_SIZE, D.IMG_SIZE), np.float16)
    t0 = time.time()
    with torch.no_grad():
        for x, idxs in dl:
            x = x.to(device, non_blocking=True)
            with torch.cuda.amp.autocast():
                p = torch.sigmoid(model(x)).float()
                if a.tta:
                    p = p + torch.flip(torch.sigmoid(model(torch.flip(x, [3]))).float(), [3])
                    p = p + torch.flip(torch.sigmoid(model(torch.flip(x, [2]))).float(), [2])
                    p = p + torch.flip(torch.sigmoid(model(torch.flip(x, [2, 3]))).float(), [2, 3])
                    p = p / 4.0
            probs[idxs.numpy()] = p.cpu().numpy().astype(np.float16)
    np.save(a.out + "_probs.npy", probs)

    ids = []
    orig_hw = []
    for mi, k in samples:
        m = meta[mi]
        # reconstruct id from filename slice number
        f = m["files"][k]
        base = os.path.basename(f)[:-4]
        parts = base.split("_", 2)
        ids.append(f"{m['day']}_{parts[0]}_{parts[1]}")
        orig_hw.append((m["H"], m["W"]))
    np.savez(a.out + "_meta.npz", ids=np.array(ids), orig_hw=np.array(orig_hw),
             scan_id=np.array([meta[mi]["scan_id"] for mi, k in samples]),
             k=np.array([k for mi, k in samples]))
    import csv
    with open(a.out + "_ids.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "case_day", "case", "slice", "H", "W", "row"])
        for r, (mi, k) in enumerate(samples):
            m = meta[mi]
            w.writerow([ids[r], m["scan_id"], m["case"], k, m["H"], m["W"], r])
    print(f"wrote {a.out} N={N} time={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
