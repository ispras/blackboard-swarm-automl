"""Multi-scale + flip TTA inference for per-fold checkpoints produced by train.py.

For each fold checkpoint f{fold}_best.pt, average softmax over scales x {id,hflip,vflip}.
Writes f{fold}_ms_test_probs.npy (2676x5) and f{fold}_ms_val_probs.npy (fold val, base scale).
"""
import os, sys, argparse, json
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

cv2.setNumThreads(0)
ROOT = "/download_job_code/payload/sandbox"
MEAN = (0.485, 0.456, 0.406); STD = (0.229, 0.224, 0.225)


class DS(Dataset):
    def __init__(self, df, img_dir, size, labeled):
        self.df = df.reset_index(drop=True); self.dir = img_dir; self.labeled = labeled
        self.tfm = A.Compose([A.Resize(int(size * 1.14), int(size * 1.14)),
                              A.CenterCrop(size, size), A.Normalize(MEAN, STD), ToTensorV2()])

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = cv2.imread(os.path.join(self.dir, r.image_id))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.tfm(image=img)["image"]
        return (img, int(r.label)) if self.labeled else (img, r.image_id)


@torch.no_grad()
def infer(model, df, img_dir, size, bs, workers, labeled):
    dl = DataLoader(DS(df, img_dir, size, labeled), batch_size=bs, shuffle=False,
                    num_workers=workers, pin_memory=True)
    outs, ids = [], []
    model.eval()
    for x, y in dl:
        x = x.cuda(non_blocking=True)
        with autocast():
            logits = model(x)
            logits = logits + model(torch.flip(x, dims=[3])) + model(torch.flip(x, dims=[2]))
        outs.append(F.softmax(logits.float(), 1).cpu().numpy())
        ids.extend(list(y))
    return np.concatenate(outs), ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--rundir", required=True)
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--scales", default="352,384,416")
    ap.add_argument("--bs", type=int, default=24)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--split", default=f"{ROOT}/workspaces/shared/folds5_seed42.csv")
    ap.add_argument("--sample_sub", default=f"{ROOT}/data/sample_submission.csv")
    args = ap.parse_args()

    scales = [int(s) for s in args.scales.split(",")]
    df = pd.read_csv(args.split)
    sub = pd.read_csv(args.sample_sub)
    for f in [int(x) for x in args.folds.split(",")]:
        ck = f"{args.rundir}/f{f}_best.pt"
        model = timm.create_model(args.backbone, pretrained=False, num_classes=5).cuda()
        model.load_state_dict(torch.load(ck, map_location="cuda"))
        acc = []
        for s in scales:
            tp, ids = infer(model, sub, f"{ROOT}/data/test_images", s, args.bs, args.workers, False)
            acc.append(tp)
            print(f"f{f} scale {s} done", flush=True)
        ms = np.mean(acc, 0)
        np.save(f"{args.rundir}/f{f}_ms_test_probs.npy", ms)
        print(f"f{f} ms test saved", flush=True)
        del model
        torch.cuda.empty_cache()
    print("DONE")


if __name__ == "__main__":
    main()
